"""Typed state and bounded waits for same-handle group reconfiguration."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

from . import mesh_setup_state as _state
from .mesh_lease import (  # noqa: F401  # Public mesh_setup compatibility API.
    SetupBoundFuture,
    prepare_sample,
    submit_sample,
    token_kwargs,
)
from .transfer_utils import safe_note

InvalidCleanupResult = _state.InvalidCleanupResult
LifecycleBusyError, SampleResultBusyError = _state.LifecycleBusyError, _state.SampleResultBusyError
SetupCleanupOutcome = _state.SetupCleanupOutcome
SetupCleanupState = _state.SetupCleanupState
SetupToken = _state.SetupToken
SetupTransitionBudget = _state.SetupTransitionBudget
SetupVerdictError = _state.SetupVerdictError
StaleSetupGenerationError = _state.StaleSetupGenerationError
TopologyTransitionError = _state.TopologyTransitionError
_positive_timeout = _state._positive_timeout
cleanup_failure = _state.cleanup_failure
cleanup_in_progress = _state.cleanup_in_progress
validate_cleanup_result = _state.validate_cleanup_result


def nccl_setup_key(topology: Any, attention: str, sync_ulysses: bool) -> tuple:
    """Identity of the process-group shape; worker policy stays out (``MeshHandle.ensure_setup`` says why)."""
    return (
        topology.ulysses, topology.ring, topology.cfg, topology.dp, topology.fsdp,
        topology.world, attention, sync_ulysses,
    )


def deadline_after(timeout_s: float) -> float:
    return time.monotonic() + timeout_s


def remaining(deadline: float, phase: str) -> float:
    value = deadline - time.monotonic()
    if value <= 0.0:
        raise TimeoutError(f"{phase} exceeded its shared deadline")
    return value


def dispatch_setup(
    handle: Any,
    topology: Any,
    attention: str,
    sync_ulysses: bool,
    worker_args: dict,
    fabric_env: dict,
    master_addr: str,
    master_port: int,
    fleet: bool,
) -> list[Any]:
    """Dispatch every rank inside the caller's setup rollback envelope."""
    from . import mesh_safety

    labels = list(handle.workers.extent.labels)
    futures = []
    for host_idx in range(handle.n_hosts):
        for gpu_idx in range(handle.gpus_per_host):
            rank = host_idx * handle.gpus_per_host + gpu_idx
            coords = {"gpus": gpu_idx}
            if "hosts" in labels:
                coords["hosts"] = host_idx
            worker = handle.workers.slice(**coords)
            env = {
                # Bind worker-side provenance barriers to the READY generation
                # that authorizes their driver dispatch. The actor validates
                # and publishes it only after setup completes, so a stale or
                # partly replaced generation cannot produce a current baseline.
                "setup_generation": int(handle.setup_generation),
                "rank": 0 if fleet else rank,
                "world": 1 if fleet else handle.world,
                "master_addr": "127.0.0.1" if fleet else master_addr,
                "master_port": master_port + (32 + rank if fleet else 0),
                "topology": {
                    "ulysses": topology.ulysses, "ring": topology.ring,
                    "cfg": topology.cfg, "dp": topology.dp, "fsdp": topology.fsdp,
                },
                "fabric_env": fabric_env,
                "comfy_dir": mesh_safety.worker_comfy_dir(
                    handle.config, host_idx, handle.comfy_dir),
                "worker_args": worker_args,
                "local_gpu_index": gpu_idx,
                "gpus_per_host": handle.gpus_per_host,
                "attention": attention,
                "sync_ulysses": sync_ulysses,
                "rdma_latent_return": handle.config.rdma_latent_return,
                "rdma_min_bytes": handle.config.rdma_min_bytes,
            }
            futures.append(worker.setup.call_one(env))
    return futures


_SETUP_INDEPENDENT_ENDPOINTS = frozenset({
    "ack_latent_handoff", "artifact_identity", "cancel_sample", "clear_vram", "status",
    "unload",
})

# These calls change setup or residency state, or queue behind earlier GPU
# mutation. MeshHandle.call_all holds a render-session capability until their
# RPC completes, not only through enqueue. ``provenance_baseline`` is the queue
# barrier the Gate's provenance proof reads (nodes/gate_provenance.py), not a
# diagnostic status read, so it also requires an exact SetupToken.
SESSION_SCOPED_ENDPOINTS = frozenset({
    "clear_vram",
    "compute_sigmas",
    "gate_fsdp_reload_cycle",
    "gate_swap_cycle",
    "load_model",
    "load_uncond_model",
    "provenance_baseline",
    "unload",
})

# A queue-barrier baseline holds the same session/token authority but is
# read-only: delayed completion cannot change setup or residency. Only these
# endpoints require a DIRTY recycle latch when completion is ambiguous.
AMBIGUOUS_MUTATION_ENDPOINTS = SESSION_SCOPED_ENDPOINTS - {"provenance_baseline"}


def require_not_dirty(handle: Any, operation: str) -> None:
    state = getattr(handle, "setup_cleanup_state", None)
    if state is not None:
        raise TopologyTransitionError(state, operation)


def require_endpoint(handle: Any, endpoint_name: str) -> None:
    if endpoint_name in {"setup", "teardown_group", "apply_worker_args"}:
        raise RuntimeError(
            f"{endpoint_name} is lifecycle-internal; use the MeshHandle lifecycle API"
        )
    if endpoint_name in _SETUP_INDEPENDENT_ENDPOINTS:
        return
    if getattr(handle, "defunct", False):
        raise RuntimeError(
            f"cannot dispatch {endpoint_name}: worker fleet is defunct")
    require_not_dirty(handle, f"dispatch {endpoint_name}")
    if getattr(handle, "setup_key", None) is None:
        raise RuntimeError(
            f"cannot dispatch {endpoint_name}: distributed setup has not completed"
        )


def require_no_sample_leases(
    handle: Any, operation: str, *, allow_abandoned: bool = False
) -> None:
    """Refuse destructive lifecycle work while an RDMA read, a sample lease or,
    unless allowed, an abandoned sample is live."""
    from .rdma_read_job import pending_job_count_for_handle

    pending_reads = pending_job_count_for_handle(handle)
    if pending_reads:
        raise SampleResultBusyError(
            f"cannot {operation}: {pending_reads} unresolved sample result "
            "lease(s) still protect RDMA descriptor ownership"
        )
    leases = getattr(handle, "sample_leases", {})
    active = sum(int(count) for count in leases.values())
    if active:
        raise SampleResultBusyError(
            f"cannot {operation}: {active} sample result lease(s) still protect "
            "worker-side latent backing; collect or abandon them before retrying"
        )
    abandoned = sum(
        int(count) for count in getattr(
            handle, "abandoned_sample_leases", {}).values())
    if abandoned and not allow_abandoned:
        raise SampleResultBusyError(
            f"cannot {operation}: {abandoned} abandoned sample(s) may still be "
            "running; reset the Attached mesh before changing setup"
        )


def require_no_abandoned_samples(handle: Any, operation: str) -> None:
    """Keep unresolved abandoned endpoint ownership out of newer work."""
    abandoned = sum(
        int(count) for count in getattr(
            handle, "abandoned_sample_leases", {}).values())
    if abandoned:
        raise LifecycleBusyError(
            f"cannot {operation}: {abandoned} abandoned sample(s) may still be "
            "running or still own a result or RDMA descriptor of their "
            "setup generation; reset the Attached mesh first"
        )


@contextmanager
def lifecycle_lock(handle: Any, operation: str, timeout_s: float = 10.0):
    """Bound lock acquisition so a timed-out caller leaves no late waiter."""
    timeout = _positive_timeout(timeout_s, f"{operation} lock timeout")
    acquired = False
    try:
        acquired = handle.lock.acquire(timeout=timeout)
        if not acquired:
            raise LifecycleBusyError(
                f"cannot {operation}: another lifecycle transition still owns the mesh"
            )
        yield
    finally:
        owned = getattr(handle.lock, "_is_owned", lambda: False)
        if acquired or owned():
            handle.lock.release()


def current_setup_token(handle: Any) -> SetupToken:
    """Capture READY identity while the caller holds ``handle.lock``."""
    require_not_dirty(handle, "capture distributed setup")
    key = getattr(handle, "setup_key", None)
    if key is None:
        raise SetupVerdictError("distributed setup has not completed")
    worker_args_key = getattr(handle, "worker_args_key", None)
    if worker_args_key is None:
        raise SetupVerdictError("distributed worker policy has not completed")
    return SetupToken(
        int(handle.setup_generation), tuple(key), tuple(worker_args_key))


def ensure_setup_token(handle: Any, *args: Any, **kwargs: Any) -> SetupToken:
    """Ensure one setup and capture its token without a transition gap."""
    with handle.lock:
        handle.ensure_setup(*args, **kwargs)
        return current_setup_token(handle)


def ensure_request_setup(handle: Any, *args: Any, **kwargs: Any) -> SetupToken | None:
    """Bind real handles while preserving lightweight node-test doubles."""
    from .mesh import MeshHandle

    if isinstance(handle, MeshHandle):
        return ensure_setup_token(handle, *args, **kwargs)
    handle.ensure_setup(*args, **kwargs)
    return None


def stop_failed_setup_procs(handle: Any) -> BaseException | None:
    """Retire an indeterminate setup rollback without masking its root error."""
    from . import mesh_helpers, mesh_teardown

    return mesh_teardown.stop_failed_setup_procs(
        handle, mesh_helpers.run_blocking_off_loop)


def dispatch_sample(
    handle: Any, send: Any, *, setup_token: SetupToken | None,
    authority: SetupBoundFuture | None = None,
) -> SetupBoundFuture:
    """Own a logical session through one sample's terminal lease."""
    from .mesh_session import RenderSession, claim_render_session

    candidate = RenderSession()
    session = None
    owns_session = False
    sample_authority = authority
    primary: BaseException | None = None
    try:
        try:
            from .mesh import MeshHandle

            if isinstance(handle, MeshHandle) and sample_authority is None:
                raise ValueError(
                    "real sample dispatch requires caller-prepared authority "
                    "before any render-session or enqueue side effect")
            session, owns_session = claim_render_session(handle, candidate)
            if sample_authority is None:
                generation = setup_token.generation if setup_token is not None else -1
                sample_authority = SetupBoundFuture.prepared(handle, generation)
            if owns_session:
                # Register the terminal callback before enqueue. The close below
                # marks this one-shot session as closing; the owner stays
                # published until the setup or read lease finalizes.
                session.track(sample_authority)
            with session.activate():
                return _dispatch_sample_bound(
                    handle, send, setup_token=setup_token,
                    sample_authority=sample_authority)
        except BaseException as exc:
            primary = exc
            if sample_authority is not None:
                for attempt in range(2):
                    try:
                        sample_authority.abandon_after_dispatch(exc)
                    except BaseException as cleanup_exc:
                        safe_note(
                            primary,
                            f"sample authority cleanup attempt {attempt + 1} also failed",
                            cleanup_exc,
                        )
                    else:
                        break
            raise
        finally:
            try:
                _close_sample_candidate(candidate, primary)
            except BaseException as cleanup_exc:
                if primary is None:
                    primary = cleanup_exc
                    raise
                safe_note(
                    primary, "sample render-session cleanup also failed", cleanup_exc)
    finally:
        try:
            _close_sample_candidate(candidate, primary)
        except BaseException as cleanup_exc:
            if primary is None:
                raise
            safe_note(
                primary, "sample render-session retry also failed", cleanup_exc)


def _close_sample_candidate(
    candidate: Any, primary: BaseException | None
) -> None:
    """Idempotently close a one-shot sample owner without masking dispatch."""
    try:
        candidate.close()
    except BaseException as cleanup_exc:
        if primary is None:
            raise
        safe_note(
            primary, "sample render-session cleanup also failed", cleanup_exc)


def _dispatch_sample_bound(
    handle: Any,
    send: Any,
    *,
    setup_token: SetupToken | None,
    sample_authority: SetupBoundFuture,
) -> SetupBoundFuture:
    """Authorize, lease, and enqueue under an activated logical session."""
    try:
        with handle.lock:
            from .mesh_session import require_mutation_authority

            require_endpoint(handle, "sample")
            require_mutation_authority(
                handle, "dispatch sample", require_owner=True)
            require_no_abandoned_samples(handle, "dispatch another sample")
            if setup_token is None:
                raise RuntimeError("setup-bound endpoint sample requires a SetupToken")
            _require_setup_token(handle, setup_token)
            if sample_authority.handle is not handle:
                raise RuntimeError("sample authority belongs to a different mesh handle")
            if sample_authority.generation != setup_token.generation:
                raise RuntimeError("sample authority belongs to a different setup generation")
            sample_authority._register_locked()
            sample_authority._mark_enqueue_started_locked()
            future = send()
            sample_authority._bind_future_locked(future)
        return sample_authority
    except BaseException as exc:
        # Once enqueue_started is published the endpoint may have accepted the
        # sample even if its call never returned. The accepted endpoint may
        # retain generation-bound state; an explicit Attached-mesh reset gates setup.
        sample_authority.abandon_after_dispatch(exc)
        raise


def dispatch_sample_to_actor(
    handle: Any, index: int, request: dict, progress_port: Any,
    setup_token: SetupToken | None,
    authority: SetupBoundFuture | None = None,
) -> SetupBoundFuture:
    """Preflight and lease one fleet sample for its selected WORLD-1 actor."""
    require_not_dirty(handle, "dispatch sample")
    payload = dict(request)
    payload["_dgxm_fleet_job"] = True
    handle._verify_request_artifacts(payload, worker_index=index)
    coords = {"gpus": index % handle.gpus_per_host}
    if "hosts" in list(handle.workers.extent.labels):
        coords["hosts"] = index // handle.gpus_per_host
    worker = handle.workers.slice(**coords)
    return dispatch_sample(
        handle,
        lambda: worker.sample.call_one(payload, progress_port=progress_port),
        setup_token=setup_token,
        authority=authority)


def dispatch_collective_sample(
    handle: Any, request: dict, progress_port: Any,
    setup_token: SetupToken | None,
    authority: SetupBoundFuture | None = None,
) -> SetupBoundFuture:
    """Preflight and lease one collective sample across the actor mesh."""
    require_not_dirty(handle, "dispatch sample")
    # The capacity quote runs before the artifact parity proof, so the quote's
    # refusal names a silent rank within the quote's deadline, instead of a
    # parity timeout that names no rank. The order is safe because the quote
    # mutates nothing, grants nothing and takes no lease, and the dirty latch
    # above still answers first.
    from . import capacity_agreement

    capacity_agreement.agree_sample(handle, request)
    handle._verify_request_artifacts(request)
    return dispatch_sample(
        handle,
        lambda: handle.workers.sample.call(request, progress_port=progress_port),
        setup_token=setup_token,
        authority=authority)


def release_sample(future: Any) -> None:
    if isinstance(future, SetupBoundFuture):
        future.release()


def abandon_sample(future: Any) -> None:
    if isinstance(future, SetupBoundFuture):
        future.abandon()


def _require_setup_token(handle: Any, expected: SetupToken | None) -> None:
    if expected is None:
        return
    actual_generation = int(getattr(handle, "setup_generation", -1))
    if (actual_generation != expected.generation
            or getattr(handle, "setup_key", None) != expected.key
            or getattr(handle, "worker_args_key", None) != expected.worker_args_key):
        raise StaleSetupGenerationError(expected, actual_generation)


def dispatch_endpoint(
    handle: Any,
    endpoint_name: str,
    send: Any,
    *,
    setup_token: SetupToken | None = None,
) -> Any:
    """Authorize and enqueue one endpoint call at a setup-state linearization point.

    Setup-independent endpoints that are not session-scoped skip ``handle.lock``,
    so they answer while a topology teardown holds it. Every setup-bound endpoint
    couples its final state check to message enqueue under that lock, so a
    transition cannot retire one generation between authorization and dispatch.
    """
    if endpoint_name == "sample":
        raise RuntimeError(
            "sample is lease-bound; use MeshHandle.submit_sample or submit_sample_to"
        )
    if endpoint_name in _SETUP_INDEPENDENT_ENDPOINTS:
        if setup_token is not None:
            raise ValueError("setup tokens apply only to setup-bound endpoints")
        if endpoint_name not in SESSION_SCOPED_ENDPOINTS:
            require_endpoint(handle, endpoint_name)
            return send()
        with handle.lock:
            from .mesh_session import require_mutation_authority

            require_endpoint(handle, endpoint_name)
            require_mutation_authority(
                handle, f"dispatch {endpoint_name}", require_owner=True)
            return send()
    with handle.lock:
        from .mesh_session import require_mutation_authority

        require_endpoint(handle, endpoint_name)
        require_mutation_authority(
            handle, f"dispatch {endpoint_name}", require_owner=True)
        if setup_token is None:
            raise RuntimeError(
                f"setup-bound endpoint {endpoint_name} requires a SetupToken"
            )
        _require_setup_token(handle, setup_token)
        return send()


def publish_worker_capabilities(handle: Any, results: list) -> None:
    """Log every rank's setup report and retain what a preflight must read.

    Optional kernels are per-box facts. Keeping the reports on the handle lets
    the driver refuse a rank-asymmetric selection before it dispatches, which
    is the only safe order: a refusal raised inside a collective on one rank
    abandons the lease and leaves its peers waiting.
    """
    from .log import get_logger

    log = get_logger(__name__)
    reports = []
    for report in results:
        log.info("worker up: %s", report)
        if isinstance(report, dict):
            reports.append(report)
    handle.worker_capabilities = reports
