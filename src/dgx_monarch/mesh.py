"""Driver mesh lifecycle: cache actors, reap procs, surface failed attach/retry."""
from __future__ import annotations

import atexit
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from . import (
    TORCHMONARCH_PIN,
    __version__,
    client_lease,
    mesh_attach,
    mesh_creation,
    mesh_factory,
    mesh_helpers,
    mesh_lease,
    mesh_liveness,
    mesh_recycle,
    mesh_rpc,
    mesh_safety,
    mesh_setup,
    mesh_teardown,
)
from .config import ClusterConfig, find_config_path, load_cluster_config, local_config
from .constants import DEFAULT_NCCL_MASTER_PORT
from .log import get_logger
from .mesh_factory import FleetHandoff, WorkerSpawnError, spawn_worker_fleet
from .topology import Topology

_attach_once = mesh_helpers.attach_once
_bootstrap_proc = mesh_helpers.bootstrap_proc
config_fingerprint = mesh_helpers.config_fingerprint
_detect_comfy_dir = mesh_helpers.detect_comfy_dir
_mesh_cache_key = mesh_helpers.mesh_cache_key
_src_pythonpath = mesh_helpers.src_pythonpath
_visible_gpu_count = mesh_helpers.visible_gpu_count
log = get_logger(__name__)

_nccl_setup_key = mesh_setup.nccl_setup_key
_MESHES: dict[tuple, MeshHandle] = {}
_MESH_PENDING: dict[tuple, MeshHandle] = {}
_MESH_LOCK = threading.Lock()
_MESH_CONDITION = threading.Condition(_MESH_LOCK)
_MESH_CREATING: dict[tuple, mesh_creation.CreationAttempt] = {}
_FAULT_HOOK_INSTALLED = False
# Monarch allows one process transport; local and cluster cannot coexist.
_TRANSPORT_MODE: str | None = None
_TRANSPORT_POISON: str | None = None
_TRANSPORT_BIND: str | None = None
_TRANSPORT_CLAIM_GENERATION = 0
_TRANSPORT_CLAIMS: set[tuple[int, str, object]] = set()
_TRANSPORT_COMMITTED_GENERATION: int | None = None
_TRANSPORT_ENABLE_LOCK = threading.Lock()
class MeshAttachError(RuntimeError):
    pass


RecycleOutcome = mesh_recycle.RecycleOutcome
RecycleStatus = mesh_recycle.RecycleStatus
mark_defunct_on_supervision_failure = mesh_helpers.mark_defunct_on_supervision_failure

def _coherent_lifecycle_verdict(handle) -> str:
    return mesh_safety.coherent_lifecycle_verdict(handle, MeshAttachError)


def _evict_handle_locked(handle: MeshHandle) -> None:
    for registry in (_MESHES, _MESH_PENDING):
        for key, cached in list(registry.items()):
            if cached is handle:
                registry.pop(key, None)


def _teardown_outcome_published(handle) -> bool:
    return mesh_safety.teardown_outcome_published(handle)


def _fault_correlation_state() -> tuple[bool, bool, bool]:
    with _MESH_LOCK:
        return mesh_safety.fault_correlation_state(
            [*_MESHES.values(), *_MESH_PENDING.values()], bool(_MESH_CREATING))


def install_fault_hook() -> None:
    """Install the process-wide Monarch fault routing and teardown filter."""
    global _FAULT_HOOK_INSTALLED
    if _FAULT_HOOK_INSTALLED:
        return
    _FAULT_HOOK_INSTALLED = mesh_helpers.install_unhandled_fault_hook(
        _fault_correlation_state)


@dataclass
class MeshHandle:
    """A live worker fleet: HostMesh + ProcMesh + GPUWorker actors."""
    config: ClusterConfig
    hosts: Any
    procs: Any
    workers: Any                      # GPUWorker actor mesh
    world: int
    gpus_per_host: int
    n_hosts: int
    comfy_dir: str
    owns_hosts: bool                  # False for this_host(): no external worker loops
    setup_key: tuple | None = None
    worker_args_key: tuple | None = None
    topology: Topology | None = None
    attention: str = "TORCH_FLASH"
    setup_generation: int = 0
    setup_cleanup_state: mesh_setup.SetupCleanupState | None = None
    config_mtime: float = 0.0         # cluster.toml mtime at attach; nothing reads it
    config_fingerprint: str = "local"
    defunct: bool = False             # the fleet died or was retired; evicted from cache
    replacement_blocked: str | None = None
    _replacement_retryable: bool = field(default=False, repr=False)
    teardown_complete: bool = False
    active_worker_args: dict = field(default_factory=dict)
    sample_leases: dict[int, int] = field(default_factory=dict)
    abandoned_sample_leases: dict[int, int] = field(default_factory=dict)
    deferred_supervision_error: BaseException | None = None
    # Unforgeable capability for supervision-only dead-proc reconciliation.
    _supervision_reconcile_token: object | None = field(default=None, repr=False)
    _supervision_reconcile_thread: threading.Thread | None = field(
        default=None, repr=False)
    _creation_phase: str = field(default="building", init=False, repr=False)
    _selection_claim: object | None = field(default=None, repr=False, compare=False)
    # RLock, not Lock: apply_worker_args is reachable from inside ensure_setup
    # on the same thread while it already holds this lock. Cross-thread it
    # cannot help: eviction's reconcile stop therefore never runs under a
    # held lock (mark_defunct_on_supervision_failure holding_lock=True).
    lock: threading.RLock = field(default_factory=threading.RLock)

    def effective_worker_args(self, worker_args: dict | None = None) -> dict:
        return {**dict(self.config.worker_args), **dict(worker_args or {})}

    @staticmethod
    def _worker_args_key(worker_args: dict) -> tuple:
        return tuple(sorted(
            (mesh_factory.safe_failure_message(key),
             mesh_factory.safe_failure_evidence(value))
            for key, value in worker_args.items()))

    def apply_worker_args(self, worker_args: dict, timeout_s: float = 120.0,
                          merge_config: bool = True) -> list[dict]:
        """Broadcast a worker policy and publish its driver key only on success.

        A failed broadcast may have changed a subset of ranks, so the cached
        key is invalidated. Callers can then compensate or force the next
        render to re-apply the complete policy.
        """
        effective = (self.effective_worker_args(worker_args) if merge_config
                     else dict(worker_args))
        with self.lock:
            from .mesh_session import require_mutation_authority

            mesh_setup.require_not_dirty(self, "apply worker arguments")
            if self.setup_key is None:
                raise RuntimeError("cannot apply worker arguments before setup is READY")
            mesh_setup.require_no_sample_leases(self, "change worker arguments")
            require_mutation_authority(self, "change worker arguments")
            # Retire the old policy marker and publish DIRTY before the first
            # remote side effect. A timeout, lost dispatch return, or async
            # interruption cannot leave stale READY policy authorization.
            self.worker_args_key = None
            self.active_worker_args = {}
            self.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                int(self.setup_generation), "apply_worker_args RPC completion",
                float(timeout_s))
            send_started = rpc_complete = False
            future = None
            try:
                send_started = True
                future = self.workers.apply_worker_args.call(effective)
                value_mesh = self._await_or_evict(
                    future, timeout_s=timeout_s, holding_lock=True)
                rpc_complete = True
                mesh_helpers.publish_worker_args(
                    self, effective, self._worker_args_key(effective))
            except BaseException as exc:
                ambiguous = (
                    not rpc_complete
                    and send_started
                    and (
                        future is None
                        or isinstance(exc, TimeoutError)
                        or not isinstance(exc, Exception)
                    )
                )
                self.setup_cleanup_state = (
                    mesh_setup.cleanup_failure(
                        int(self.setup_generation),
                        "apply_worker_args RPC completion", float(timeout_s), exc)
                    if ambiguous else None
                )
                raise
            self.setup_cleanup_state = None
        return [value for _point, value in value_mesh.items()]

    @contextmanager
    def temporary_worker_args(self, original: dict, overrides: dict,
                              timeout_s: float = 120.0):
        """Temporarily change a complete policy with unconditional rollback."""
        original_effective = self.effective_worker_args(original)
        target = {**original_effective, **dict(overrides)}
        body_failed = False
        try:
            applied = self.apply_worker_args(target, timeout_s=timeout_s, merge_config=False)
            yield applied
        except BaseException:
            body_failed = True
            raise
        finally:
            # This runs even when applying the target partly failed. A
            # restore failure must never mask the body's exception: the
            # operator needs the render or CUDA error, not the cleanup error.
            # A failed restore already cleared worker_args_key, so the next
            # render re-applies the complete policy.
            try:
                self.apply_worker_args(
                    original_effective, timeout_s=timeout_s, merge_config=False)
            except BaseException as restore_exc:
                log.error("temporary worker-args restore failed: %r", restore_exc)
                if not body_failed:
                    raise

    def ensure_setup(self, topology: Topology, attention: str, sync_ulysses: bool = True,
                     worker_args: dict | None = None, timeout_s: float = 600.0,
                     fleet: bool = False) -> list[dict]:
        """Initialize NCCL/xfuser groups for the requested topology and worker policy.

        ``fleet=True`` gives each actor its own world-1 group, loopback rendezvous,
        and port, allowing independent ``submit_sample_to`` calls without cross-rank
        NCCL. Fully resident models produced identical pixels for the same prompt and
        seed on the tested pair; dynamic offload differed by about 3e-5 relative.
        See docs/VALIDATION.md, Determinism tiers.
        """
        budget = mesh_setup.SetupTransitionBudget.for_setup_timeout(timeout_s)
        # cluster.toml [worker_args] is the base layer; Init-node widgets win.
        worker_args = self.effective_worker_args(worker_args)
        # The NCCL/xfuser groups depend only on the parallel shape (topology,
        # attention, sync, fleet). worker_args (memory posture, safetensors
        # backend, LoRA residency mode) stay out of this key: tearing the groups
        # down for a memory toggle aborts the live USP communicator the next
        # render needs, and re-bringing up the same topology leaves xfuser
        # pointing at the destroyed group. Changed worker_args go through
        # apply_worker_args, which never touches NCCL.
        key = _nccl_setup_key(topology, attention, sync_ulysses)
        if fleet:
            key = (*key, "fleet")
        wa_key = self._worker_args_key(worker_args)
        with self.lock:
            from .mesh_session import require_mutation_authority

            mesh_setup.require_not_dirty(self, "begin distributed setup")
            if self.setup_key == key:
                if self.worker_args_key != wa_key:
                    mesh_setup.require_no_sample_leases(
                        self, "change worker arguments")
                    require_mutation_authority(
                        self, "change worker arguments")
                    self.apply_worker_args(worker_args, timeout_s=budget.group_cleanup_s, merge_config=False)
                return []

            mesh_setup.require_no_sample_leases(self, "change distributed setup")
            require_mutation_authority(self, "change distributed setup")

            # Consume the setup generation before any destructive work. The
            # pre-increment value still salts the rendezvous port, while the
            # published value invalidates old artifact-preflight markers even
            # when teardown or setup fails. When replacing READY groups,
            # publish DIRTY before the generation store too: an interruption
            # after advancing generation must not leave a clean old setup key
            # whose workers still advertise the prior generation.
            attempt = self.setup_generation
            base_port = self.config.nccl_master_port or DEFAULT_NCCL_MASTER_PORT
            master_port = mesh_safety.bounded_master_port(
                base_port, os.getpid(), attempt, self.world if fleet else 0)
            replacing_setup = self.setup_key is not None
            if replacing_setup:
                self.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                    attempt + 1, "topology teardown", budget.group_cleanup_s)
            self.setup_generation = attempt + 1

            if replacing_setup:
                log.info("topology change: tearing down old NCCL groups on live actors")
                # DIRTY precedes the generation store and READY retirement, so
                # every later store is fail-closed and a retry cannot reuse a
                # torn transition.
                self.setup_key = None
                self.topology = None
                try:
                    cleaned = self._await_or_evict(
                        self.workers.teardown_group.call(),
                        timeout_s=budget.group_cleanup_s, holding_lock=True)
                    mesh_setup.validate_cleanup_result(cleaned, self.world)
                except BaseException as exc:
                    state = mesh_setup.cleanup_failure(
                        self.setup_generation, "topology teardown",
                        budget.group_cleanup_s, exc)
                    self.setup_cleanup_state = state
                    if not isinstance(exc, Exception):
                        raise
                    raise mesh_setup.TopologyTransitionError(
                        state, "start replacement setup") from exc
                self.setup_cleanup_state = None

            # Initial bring-up is destructive too: actors can own partial live
            # groups before dispatch returns or the rollback handler starts.
            # Keep the generation DIRTY from before enqueue until READY is
            # atomically published or every rank confirms rollback.
            self.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                self.setup_generation, "distributed setup", budget.rank_setup_s)
            fabric_env = self.config.resolved_fabric_env()
            master_addr = self.config.resolved_master_addr()
            try:
                deadline = mesh_setup.deadline_after(budget.rank_setup_s)
                futures = mesh_setup.dispatch_setup(
                    self, topology, attention, sync_ulysses, worker_args,
                    fabric_env, master_addr, master_port, fleet)
                results = [self._await_or_evict(
                    f, mesh_setup.remaining(deadline, "distributed setup"),
                    holding_lock=True) for f in futures]
                # READY publication is part of this same rollback transaction.
                # A BaseException at the call boundary after every worker has
                # accepted setup must still tear those groups back down rather
                # than leave a clean key-less driver over live actors.
                mesh_helpers.publish_setup_ready(
                    self, key, wa_key, worker_args, topology, attention,
                    budget.rank_setup_s)
            except BaseException as setup_exc:
                # Some ranks may have completed setup before a peer failed.
                # Tear all of them back to the common uninitialized state;
                # otherwise a retry can mix generations of NCCL groups.
                self.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                    self.setup_generation, "setup rollback",
                    budget.rollback_cleanup_s)
                self.setup_key = None
                self.worker_args_key = None
                self.topology = None
                try:
                    cleaned = mesh_helpers.get_off_loop(
                        self.workers.teardown_group.call(), budget.rollback_cleanup_s)
                    mesh_setup.validate_cleanup_result(cleaned, self.world)
                except BaseException as cleanup_exc:
                    self.setup_cleanup_state = mesh_setup.cleanup_failure(
                        self.setup_generation, "setup rollback",
                        budget.rollback_cleanup_s, cleanup_exc)
                    self.defunct = True
                    stop_exc = mesh_setup.stop_failed_setup_procs(self)
                    if stop_exc is not None:
                        log.error("failed-setup proc cleanup also failed: %r", stop_exc)
                    log.error("failed-setup group rollback failed: %r", cleanup_exc)
                else:
                    self.setup_cleanup_state = (
                        None
                        if isinstance(setup_exc, Exception)
                        else mesh_setup.cleanup_failure(
                            self.setup_generation,
                            "setup rollback",
                            budget.rollback_cleanup_s,
                            setup_exc,
                        )
                    )
                raise
            mesh_setup.publish_worker_capabilities(self, results)
            return results
    def _await_or_evict(self, future, timeout_s: float, *, holding_lock: bool = False):
        """Await a monarch future; on a supervision-class failure evict the fleet
        (mark defunct and tear down) so the next Init or render respawns.
        apply_worker_args, ensure_setup, call_all, collect_one,
        mesh_rpc.verify_request_artifacts and capacity_agreement.collect await
        here, so no endpoint they dispatch (setup, load_model, compute_sigmas,
        status and clear_vram among them) keeps reusing a dead mesh;
        collect_sample applies the same rule in mesh_lease. Endpoint errors (a
        worker raised, the actor is alive) fail the classifier and re-raise, so
        a re-queue reuses the live mesh. A caller that holds self.lock must pass
        holding_lock=True: the eviction stop then moves to a detached thread
        instead of deadlocking against its own scratch thread
        (mark_defunct_on_supervision_failure). The get runs off ComfyUI's loop
        (mesh_runtime.get_off_loop).
        """
        try:
            return mesh_helpers.get_off_loop(future, timeout_s)
        except Exception as exc:
            mesh_helpers.mark_defunct_preserving_primary(self, exc, holding_lock=holding_lock)
            raise
    def call_all(self, endpoint_name: str, *args, timeout_s: float = 900.0,
                 setup_token: mesh_setup.SetupToken | None = None, **kwargs) -> list:
        return mesh_rpc.call_all(
            self, endpoint_name, *args, timeout_s=timeout_s,
            setup_token=setup_token, **kwargs)
    def _call_all_bound(self, endpoint_name: str, *args,
                        timeout_s: float = 900.0,
                        setup_token: mesh_setup.SetupToken | None = None,
                        **kwargs) -> list:
        return mesh_rpc.call_all_bound(
            self, endpoint_name, *args, timeout_s=timeout_s,
            setup_token=setup_token, **kwargs)
    def _latch_ambiguous_mutation(
        self, endpoint_name: str, timeout_s: float, exc: BaseException
    ) -> None:
        mesh_rpc.latch_ambiguous_mutation(
            self, endpoint_name, timeout_s, exc, logger=log)
    def _verify_request_artifacts(self, request: dict, worker_index: int | None = None) -> None:
        mesh_rpc.verify_request_artifacts(self, request, worker_index)

    def submit_sample_to(self, index: int, request: dict, progress_port=None,
                         setup_token: mesh_setup.SetupToken | None = None,
                         authority: mesh_setup.SetupBoundFuture | None = None):
        """Eager-send one leased world-1 fleet render."""
        if authority is None:
            authority = mesh_lease.prepare_sample(self, setup_token)
        return mesh_setup.dispatch_sample_to_actor(
            self, index, request, progress_port, setup_token, authority)
    def collect_one(self, future, timeout_s: float = 900.0) -> dict:
        """Await a submit_sample_to future through _await_or_evict."""
        return self._await_or_evict(future, timeout_s)

    def _retire_completed(self) -> None:
        """Confirmed teardown: clear any stranded token, settle, evict."""
        mesh_safety.clear_stale_token(id(self))
        self.defunct = True
        self.replacement_blocked = None  # completed outranks spurious blocked
        self._replacement_retryable = False
        with _MESH_LOCK:
            _evict_handle_locked(self)

    def _evict_recycled(self) -> None:
        with _MESH_LOCK:
            _evict_handle_locked(self)

    def recycle(self) -> bool:
        """Boolean compatibility wrapper for :meth:`recycle_detailed`."""
        return self.recycle_detailed().ok

    def recycle_detailed(self) -> RecycleOutcome:
        """Deliberately stop and evict worker procs with a typed outcome."""
        return mesh_recycle.recycle_detailed(
            self, self._retire_completed, self._evict_recycled)

    def _recycle_impl(self, lock_timeout_s: float = 10.0) -> bool:
        """Internal boolean compatibility wrapper used by lifecycle tests."""
        return self._recycle_detailed_impl(lock_timeout_s).ok

    def _recycle_detailed_impl(
        self, lock_timeout_s: float = 10.0,
    ) -> RecycleOutcome:
        return mesh_recycle.recycle_detailed_impl(
            self, self._retire_completed, self._evict_recycled,
            lock_timeout_s=lock_timeout_s)

    def submit_sample(self, request: dict, progress_port=None,
                      setup_token: mesh_setup.SetupToken | None = None,
                      authority: mesh_setup.SetupBoundFuture | None = None):
        """Eager-send one leased collective render without awaiting it."""
        if authority is None:
            authority = mesh_lease.prepare_sample(self, setup_token)
        return mesh_setup.dispatch_collective_sample(
            self, request, progress_port, setup_token, authority)

    def collect_sample(self, future, timeout_s: float = 900.0,
                       activity_fn=None) -> list:
        """Await a sample with lifecycle eviction; ``activity_fn`` arms the stall guard."""
        return mesh_lease.collect_sample(self, future, timeout_s, activity_fn)

    def cancel_sample(self, render_id: str, timeout_s: float = 10.0,
                      wait: bool = True) -> list:
        """Set the per-rank denoise cancellation event outside the GPU lock."""
        try:
            if not wait:
                self.workers.cancel_sample.broadcast(str(render_id))
                return []
            return self.call_all("cancel_sample", str(render_id), timeout_s=timeout_s)
        except Exception as exc:
            log.warning("distributed cancellation request failed for %s: %r", render_id, exc)
            return []

    def shutdown(
        self, timeout_s: float = 60.0, *, _reconcile_token: object | None = None
    ) -> None:
        """Stop client-owned mesh processes while leaving Worker services available.

        First tear down NCCL/xfuser groups, then stop the ProcMesh to release model
        memory and rendezvous ports. Never call ``hosts.shutdown()`` on attached
        loops: it can kill the coordinator while leaving its listener alive, blocking
        later attaches (docs/TROUBLESHOOTING.md #2).

        Local meshes also require an explicit process stop during eviction.
        ``owns_hosts`` controls only external Worker coordinators; this handle owns
        its ProcMesh in every mode.
        """
        if getattr(self, "teardown_complete", False):
            # A recycle (or an earlier clean detach) already stopped these
            # procs; a second stop can only fail and must not re-block a
            # handle whose teardown succeeded.
            return
        # Two phases of at most 60 s plus 30 s margin stay under mesh_safety.TOKEN_AUTHORITY_S.
        phase = min(timeout_s, 60)
        try:
            mesh_helpers.run_blocking_off_loop(
                lambda: self._shutdown_impl(timeout_s, _reconcile_token=_reconcile_token),
                2 * phase + 30, "dgxm-shutdown")
        except TimeoutError as exc:
            raise mesh_safety.PriorTeardownUnknownError(
                "worker ProcMesh stop did not complete within the caller deadline; "
                "its lifecycle thread still publishes the outcome, so starting a "
                "replacement is unsafe until that thread finishes"
            ) from exc

    def _shutdown_impl(self, timeout_s: float = 60.0, *,
                       _reconcile_token: object | None = None,
                       lock_timeout_s: float = 10.0) -> None:
        phase = min(timeout_s, 60)
        with mesh_setup.lifecycle_lock(self, "shutdown", lock_timeout_s):
            if getattr(self, "teardown_complete", False):
                return
            expected_reconcile = getattr(
                self, "_supervision_reconcile_token", None)
            force_reconcile = (
                _reconcile_token is not None
                and _reconcile_token is expected_reconcile
                and bool(getattr(self, "defunct", False))
            )
            if _reconcile_token is not None and not force_reconcile:
                raise RuntimeError(
                    "forced shutdown refused: no matching "
                    "supervision-reconciliation capability is latched on "
                    "this mesh, or the mesh is not defunct")
            mesh_setup.require_no_sample_leases(
                self, "shut down the worker fleet", allow_abandoned=True)
            if not force_reconcile:
                from .mesh_session import require_mutation_authority

                require_mutation_authority(self, "shut down the worker fleet")
            retrying_blocked = mesh_teardown.prepare_stop_attempt(self, "shutdown")
            confirmed = stop_attempted = False
            try:
                # Same pairing + retire-first rules as recycle.
                mesh_safety.begin_deliberate_teardown(id(self))
                destructive = self.setup_key is not None
                if destructive:
                    self.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                        getattr(self, "setup_generation", 0),
                        "shutdown group teardown", float(phase))
                    self.setup_key = None
                    try:
                        self.workers.teardown_group.call(release_models=False).get(timeout=phase)
                    except Exception as exc:
                        log.warning("NCCL teardown on detach: %r", exc)
                    except BaseException as exc:
                        self.setup_cleanup_state = mesh_setup.cleanup_failure(
                            getattr(self, "setup_generation", 0),
                            "shutdown group teardown", float(phase), exc)
                        raise
                try:
                    stop_attempted = True
                    mesh_teardown.enter_stop_boundary(self, retrying_blocked)
                    self.procs.stop("dgx-monarch client detach").get(timeout=phase)
                except TimeoutError as exc:
                    log.error("proc stop on detach timed out: %r", exc)
                    mesh_teardown.publish_stop_failure(
                        self, mesh_factory.safe_failure_evidence(exc), exc)
                    raise RuntimeError(
                        "worker ProcMesh stop timed out; its outcome is unknown, "
                        "so starting a replacement or issuing another stop is unsafe"
                    ) from exc
                except Exception as exc:
                    log.error("proc stop on detach failed: %r", exc)
                    mesh_teardown.publish_stop_failure(
                        self, mesh_factory.safe_failure_evidence(exc), exc)
                    raise RuntimeError(
                        "worker ProcMesh stop failed; the old processes may still hold "
                        "models or rendezvous ports, so starting a replacement is unsafe"
                    ) from exc
                confirmed = True
            finally:
                # Same handoff rules as recycle: publish before token clear.
                mesh_teardown.publish_stop_outcome(
                    self, confirmed, stop_attempted,
                    "detach interrupted during the proc stop; process state unknown")
                mesh_safety.end_deliberate_teardown(id(self), stop_confirmed=confirmed)
            self.worker_args_key = None
            self.active_worker_args = {}
            if force_reconcile:
                self._supervision_reconcile_token = None
def get_mesh(
    config_path: str = "",
    mode: str = "auto",              # auto | local | cluster
    gpus_per_host: int = 0,          # local mode: 0 = all visible GPUs
    comfy_dir: str = "", *, _selection_claim: object | None = None,
    mesh_preflight: mesh_helpers.MeshPreflight | None = None,
) -> MeshHandle:
    """Get/create a mesh; a private claim marks only same-call creation."""
    global _TRANSPORT_MODE, _TRANSPORT_POISON
    global _TRANSPORT_CLAIM_GENERATION, _TRANSPORT_COMMITTED_GENERATION
    install_fault_hook()
    mesh_creation.reap_abandoned_attempts(globals())
    resolved = find_config_path(config_path or None)
    use_cluster = mode == "cluster" or (mode == "auto" and resolved is not None)
    if use_cluster and resolved is None:
        raise MeshAttachError(
            f"cluster mode requested but no cluster.toml found (looked at {config_path or 'default paths'}). "
            "Write one with `dgxm setup` (`dgxm init` is the legacy writer), or switch the Init node to local mode."
        )
    wanted_transport = "cluster" if use_cluster else "local"
    if _TRANSPORT_POISON is not None:
        raise MeshAttachError(
            "this ComfyUI session's Monarch transport is not reusable after a "
            f"partial bring-up failure ({_TRANSPORT_POISON}); restart ComfyUI before retrying")
    if _TRANSPORT_MODE is not None and _TRANSPORT_MODE != wanted_transport:
        raise MeshAttachError(
            f"this ComfyUI session already brought up a {_TRANSPORT_MODE} mesh; monarch allows "
            f"one transport per process, so switching to {wanted_transport} mode needs a "
            "ComfyUI restart."
        )
    if use_cluster:
        assert resolved is not None  # noqa: S101  # Narrowed for mypy; guarded above.
        config = load_cluster_config(resolved)
    else:
        config = local_config(resolved)
    comfy = _detect_comfy_dir(comfy_dir or config.comfy_dir)
    if use_cluster:
        n_hosts = len(config.hosts)
        gph = config.hosts[0].gpus
        if any(h.gpus != gph for h in config.hosts):
            raise MeshAttachError("heterogeneous gpus-per-host is not supported; give every host the same gpus count")
    else:
        n_hosts = 1
        try:
            detected = (_visible_gpu_count() if gpus_per_host == 0
                        and not isinstance(gpus_per_host, bool) else 0)
            gph = mesh_helpers.validated_local_gpu_count(gpus_per_host, detected)
        except ValueError as exc:
            raise MeshAttachError(str(exc)) from exc
    proposed_world = n_hosts * gph
    mesh_helpers.run_mesh_preflight(mesh_preflight, config, proposed_world)
    config_mtime = 0.0
    config_digest = "local"
    if resolved is not None:
        try:
            config_mtime = os.path.getmtime(resolved)
        except OSError:
            pass
        # A local mesh consumes [worker_args], so it needs a digest of its own.
        config_digest = (config_fingerprint(resolved) if use_cluster
                         else mesh_helpers.local_worker_args_fingerprint(config.worker_args))
    key = _mesh_cache_key(config.source, comfy, use_cluster)
    # The provisional claim is releasable until the fail-closed publication
    # immediately before the first native transport/fleet call below.
    transport_releasable: bool | None = True
    fleet_handoff, fleet = FleetHandoff(), None
    handle: MeshHandle | None = None
    transport_claim, cleanup_claim, creation_succeeded = None, object(), False
    creation_attempt: mesh_creation.CreationAttempt | None = None
    primary_error: BaseException | None = None
    try:
        try:
            mesh_liveness.try_release(_MESHES.get(key))
            with _MESH_CONDITION:
                mesh_helpers.wait_for_mesh_creation(
                    _MESH_CONDITION, _MESH_CREATING, key, MeshAttachError)
                if _MESH_CREATING:
                    raise MeshAttachError(
                        "Monarch transport is not reusable while another mesh "
                        "creation is still in progress")
                if _TRANSPORT_MODE is not None and _TRANSPORT_MODE != wanted_transport:
                    raise MeshAttachError(
                        f"another request initialized {_TRANSPORT_MODE} transport while waiting; "
                        f"switching to {wanted_transport} requires a ComfyUI restart")
                if (_TRANSPORT_POISON is not None
                        or mesh_factory.creation_cleanup_pending()):
                    raise MeshAttachError(
                        f"Monarch transport is not reusable after a creation failure ({_TRANSPORT_POISON or mesh_factory.CREATION_CLEANUP_POISON}); "
                        "restart ComfyUI")
                old = mesh_creation.admission_occupant(globals(), key)
                verdict = _coherent_lifecycle_verdict(old) if old is not None else "live"
                if verdict == "completed":
                    mesh_safety.clear_stale_token(id(old))
                    _MESHES.pop(key, None)
                    old = None
                elif verdict == "blocked":
                    raise MeshAttachError(mesh_teardown.blocked_refusal(old, "get_mesh"))
                elif verdict == "unresolved":
                    raise MeshAttachError(
                        "the previous fleet's group or process teardown is unconfirmed "
                        "(dirty, in flight, or its outcome was lost); replacement is unsafe. "
                        "Reset the Attached mesh; if that reset cannot confirm the stop, restart "
                        "ComfyUI only after confirming the old workers are gone")
                changed = old is not None and (
                    (config_digest != getattr(old, "config_fingerprint", ""))
                    or getattr(old, "n_hosts", n_hosts) != n_hosts
                    or getattr(old, "gpus_per_host", gph) != gph
                    or getattr(old, "comfy_dir", comfy) != comfy)
                if old is not None and not changed:
                    return old
                new_transport = _TRANSPORT_MODE is None
                generation = (
                    _TRANSPORT_CLAIM_GENERATION + 1
                    if new_transport else _TRANSPORT_CLAIM_GENERATION)
                if (not new_transport and not _TRANSPORT_CLAIMS
                      and _TRANSPORT_COMMITTED_GENERATION != _TRANSPORT_CLAIM_GENERATION):
                    _TRANSPORT_COMMITTED_GENERATION = _TRANSPORT_CLAIM_GENERATION
                transport_claim = (generation, wanted_transport, object())
                creation_attempt = mesh_creation.CreationAttempt(
                    key=key,
                    owner_frame=sys._getframe(),
                    owner_thread_id=threading.get_ident(),
                    handoff=fleet_handoff,
                    cleanup_claim=cleanup_claim,
                    transport_claim=transport_claim,
                    expected_bind=(
                        getattr(config, "client_bind", None) if use_cluster else None),
                )
                _MESH_CREATING[key] = creation_attempt
                _TRANSPORT_CLAIMS.add(transport_claim)
                if new_transport:
                    _TRANSPORT_CLAIM_GENERATION = generation
                    _TRANSPORT_MODE = wanted_transport
                mesh_factory.begin_creation_cleanup(
                    cleanup_claim, fleet_handoff, reserve_transport=True)
            if old is not None:
                log.info("mesh source/config changed; recycling the previous fleet")
                try:
                    old.shutdown(timeout_s=30)  # never under _MESH_LOCK
                except Exception as exc:
                    raise MeshAttachError(
                        "configuration changed, but old workers could not be stopped safely; reset "
                        "the Attached mesh, or restart ComfyUI once they are confirmed gone") from exc
                with _MESH_LOCK:
                    if _MESHES.get(key) is old:
                        _MESHES.pop(key, None)
            src = _src_pythonpath()
            os.environ["DGXM_PYTHONPATH"] = src
            client_lease.plant_driver_marker()  # so the CLI sweep can see the owner
            existing = os.environ.get("PYTHONPATH", "")
            if src not in existing.split(os.pathsep):
                os.environ["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
            log.info("dgx-monarch %s (torchmonarch pin %s): bringing up %s mesh (%d x %d)",
                     __version__, TORCHMONARCH_PIN, wanted_transport, n_hosts, gph)
            # Publish non-releasability before entering any native call.  An
            # interruption before this STORE_FAST is still pre-native and safe;
            # every interruption after it must retain the process-global mode.
            transport_releasable = False
            try:
                fleet = spawn_worker_fleet(
                    config, use_cluster, gph, _attach_cluster, _bootstrap_proc, fleet_handoff)
            except WorkerSpawnError as exc:
                spawn_poison = mesh_factory.spawn_failure_poison()
                if _TRANSPORT_POISON is None:
                    _TRANSPORT_POISON = spawn_poison
                transport_releasable = not use_cluster and not exc.local_transport_initialized
                if transport_releasable and _TRANSPORT_POISON is spawn_poison:
                    _TRANSPORT_POISON = None
                elif not transport_releasable:
                    _TRANSPORT_POISON = f"actor spawn failed: {exc.original_text}" + (f"; rollback failed: {exc.cleanup_text}" if exc.cleanup is not None else "")
                exc.raise_cancellation()
                raise MeshAttachError(
                    f"worker actor bring-up failed ({exc.original_text}); spawned processes were "
                    f"{'rolled back' if exc.rollback_confirmed else 'not safely stopped'}") from exc
            hosts, procs, workers, owns_hosts = fleet
            handle = MeshHandle(
                config=config, hosts=hosts, procs=procs, workers=workers,
                world=proposed_world, gpus_per_host=gph, n_hosts=n_hosts,
                comfy_dir=comfy, owns_hosts=owns_hosts, config_mtime=config_mtime,
                config_fingerprint=config_digest, _selection_claim=_selection_claim)
            if creation_attempt is None:  # defensive: admission publishes it first
                raise MeshAttachError("mesh creation lost its owner-tagged attempt")
            creation_attempt.handle = handle
            handle._creation_phase = "building"
            mesh_creation.publish_pending_handle(globals(), key, handle)
            mesh_helpers.start_client_lease()  # holds the actor-side reaper off
            mesh_creation.publish_ready_handle(globals(), creation_attempt, handle)
            creation_succeeded = True
        except BaseException as primary:
            if fleet_handoff.ready_for_publication:
                primary_error = primary
                creation_succeeded = True
                raise
            fleet_handoff.failure_latched = True
            primary_error = primary
            unsafe_attach = (fleet_handoff.attach_in_progress
                             and not fleet_handoff.transport_failure_safe)
            if _TRANSPORT_POISON is None and (unsafe_attach or fleet_handoff.spawn_in_progress):
                _TRANSPORT_POISON = ("cluster attach result handoff interrupted"
                                     if unsafe_attach else
                                     "actor spawn failed; cleanup state unavailable")
            blocked = handle.replacement_blocked if handle is not None and not handle.teardown_complete else None
            fleet = fleet if fleet is not None else fleet_handoff.fleet
            owned_procs = (
                fleet[1] if fleet is not None
                else mesh_creation.recover_owned_spawned_proc(fleet_handoff, primary))
            if fleet_handoff.cleanup_started and not fleet_handoff.cleanup_confirmed:
                blocked = blocked or "worker spawn rollback outcome is unconfirmed"
            elif fleet_handoff.proc_ownership_unconfirmed:
                blocked = blocked or "spawned ProcMesh ownership is unconfirmed"
            elif owned_procs is not None and not fleet_handoff.cleanup_started and not blocked:
                if _TRANSPORT_POISON is None:
                    _TRANSPORT_POISON = mesh_factory.CREATION_CLEANUP_POISON
                blocked = mesh_factory.reconcile_creation_cleanup(
                    fleet_handoff, handle, owned_procs, primary)
            blocked = blocked or ("ProcMesh spawn callback cleanup is unconfirmed" if fleet_handoff.callback_cleanup_unconfirmed else None)
            if blocked:
                _TRANSPORT_POISON = "mesh creation failed; owned ProcMesh cleanup unconfirmed"
                blocked_text = mesh_factory.safe_failure_evidence(blocked)
                mesh_teardown.publish_creation_block(handle, blocked_text)
                primary_text = mesh_factory.safe_failure_evidence(primary)
                mesh_factory.safe_note(primary, f"owned ProcMesh cleanup after mesh creation failed: {blocked_text}")
                _TRANSPORT_POISON = (f"mesh creation failed ({primary_text}); owned "
                                     f"ProcMesh cleanup unconfirmed ({blocked_text})")
            elif (mesh_factory.creation_cleanup_registered(cleanup_claim)
                  and mesh_factory.confirm_creation_cleanup(cleanup_claim)
                  and _TRANSPORT_POISON is mesh_factory.CREATION_CLEANUP_POISON):
                _TRANSPORT_POISON = None
            if (_TRANSPORT_POISON is not None or _TRANSPORT_BIND is not None
                    or unsafe_attach or fleet_handoff.spawn_in_progress
                    or owned_procs is not None):
                transport_releasable = False
            elif use_cluster and not mesh_factory.creation_cleanup_pending():
                transport_releasable = True
            raise
    finally:
        if creation_attempt is not None:
            mesh_creation.finalize_creation(
                globals(), creation_attempt, handle, transport_releasable,
                creation_succeeded,
                primary_error if primary_error is not None else sys.exception())
    return handle


def ensure_live(
    handle: MeshHandle, *,
    mesh_preflight: mesh_helpers.MeshPreflight | None = None,
) -> MeshHandle:
    """Swap a defunct handle for a fresh mesh on the same config.

    ComfyUI caches node outputs, so a MeshSpec built before a worker death
    keeps referencing the evicted handle; routing through here lets a
    re-queue heal instead of wedging until a ComfyUI restart."""
    mesh_liveness.try_release(handle)
    verdict = _coherent_lifecycle_verdict(handle)
    if verdict in ("blocked", "unresolved"):
        raise MeshAttachError(mesh_teardown.blocked_refusal(handle, "ensure_live"))
    # Re-enter for live cluster handles because cluster config is mutable.
    if handle.config.hosts:
        return get_mesh(
            config_path=handle.config.source,
            mode="cluster",
            gpus_per_host=handle.gpus_per_host,
            comfy_dir="",  # re-read cluster.comfy_dir instead of pinning stale cached output
            mesh_preflight=mesh_preflight,
        )
    if verdict == "live":
        mesh_helpers.run_mesh_preflight(
            mesh_preflight, handle.config, int(handle.world))
        return handle
    log.info("mesh handle is retired (defunct or torn down); respawning on the same config")
    return get_mesh(
        config_path=handle.config.source,
        mode="cluster" if handle.config.hosts else "local",
        gpus_per_host=handle.gpus_per_host,
        comfy_dir=handle.comfy_dir,
        mesh_preflight=mesh_preflight,
    )


def _attach_cluster(config: ClusterConfig, handoff: FleetHandoff | None = None):
    return mesh_attach.attach_cluster(globals(), config, handoff)


def _heal_dead_loops(config: ClusterConfig, addresses: list[str]) -> None:
    """Restart the Worker service for loops the passive probe reported down."""
    mesh_attach.heal_dead_loops(config, addresses, log)


def _raise_attach(addresses, exc):
    mesh_attach.raise_attach(MeshAttachError, addresses, exc)


def _raise_released_attach(addresses, exc, *, auto_heal: bool, restarted=None) -> None:
    mesh_attach.raise_released_attach(
        MeshAttachError, addresses, exc, auto_heal=auto_heal, restarted=restarted)


@atexit.register
def _shutdown_all() -> None:
    handles = {**_MESH_PENDING, **_MESHES}
    mesh_helpers.shutdown_all(handles, log)
    _MESH_PENDING.clear()
    _MESHES.clear()
