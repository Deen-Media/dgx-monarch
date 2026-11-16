"""MeshHandle's setup transitions: worker-policy broadcast and process-group setup.

``MeshHandle.apply_worker_args``, ``temporary_worker_args`` and
``ensure_setup`` delegate here and keep their names, so callers and tests reach
them through the handle as before. Each function calls back through the handle
for ``_await_or_evict``, ``apply_worker_args`` and the worker-args key, so a
patched handle method still takes effect. The logger keeps the
``dgx_monarch.mesh`` name these log lines had before they moved.
"""
from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from . import mesh_helpers, mesh_safety, mesh_setup
from .constants import DEFAULT_NCCL_MASTER_PORT
from .log import get_logger
from .topology import Topology

log = get_logger("dgx_monarch.mesh")

# How long the driver waits on one rank's setup before it asks the later ranks,
# with no wait, whether one of them has already failed.
SETUP_POLL_S = 2.0


def _answer(handle: Any, future: Any, wait: float) -> Any:
    """One rank's setup answer within ``wait`` seconds, or ``TimeoutError`` while it has none.

    Monarch's ``Future.get(timeout)`` leaves an endpoint's future running on a
    timeout, so a later get still reads the answer (every future but a
    ``@returns_future`` one; ``call_one`` returns an endpoint's). A timeout is
    no supervision failure, so ``_await_or_evict`` evicts nothing for it.
    """
    return handle._await_or_evict(future, wait, holding_lock=True)


def _await_setup(handle: Any, futures: list[Any], deadline: float, failed: dict[str, Any]) -> list[dict]:
    """Every rank's setup answer, in rank order, and a later rank's failure as soon as it is seen.

    The ranks are awaited in order, a slice at a time; between slices each
    later rank is asked, with no wait, whether it has answered. A rank that
    cannot start answers at once while its peer waits at the rendezvous for
    it, so without this the driver read the peer's wait to the deadline and
    never the rank's own refusal. ``failed`` receives the rank whose error is
    raised; for one seen ahead of the rank being awaited, that rank
    (``ahead``) and what was left of the setup's wait (``spare``).
    """
    results: list[dict] = []
    for index, future in enumerate(futures):
        failed["rank"] = index
        while True:
            wait = min(SETUP_POLL_S, mesh_setup.remaining(deadline, "distributed setup"))
            try:
                results.append(_answer(handle, future, wait))
                break
            except TimeoutError:
                pass
            for later in range(index + 1, len(futures)):
                try:
                    _answer(handle, futures[later], 0.0)
                except TimeoutError:
                    continue
                except Exception:
                    failed.update(rank=later, ahead=index, spare=max(0.0, deadline - time.monotonic()))
                    raise
    return results


def apply_worker_args(handle: Any, worker_args: dict, timeout_s: float = 120.0,
                      merge_config: bool = True) -> list[dict]:
    """Broadcast a worker policy and publish its driver key only on success.

    A failed broadcast may have changed a subset of ranks, so the cached
    key is invalidated. Callers can then compensate or force the next
    render to re-apply the complete policy.
    """
    effective = (handle.effective_worker_args(worker_args) if merge_config
                 else dict(worker_args))
    with handle.lock:
        from .mesh_session import require_mutation_authority

        mesh_setup.require_not_dirty(handle, "apply worker arguments")
        if handle.setup_key is None:
            raise RuntimeError("cannot apply worker arguments before setup is READY")
        mesh_setup.require_no_sample_leases(handle, "change worker arguments")
        require_mutation_authority(handle, "change worker arguments")
        # Retire the old policy marker and publish DIRTY before the first
        # remote side effect. A timeout, lost dispatch return, or async
        # interruption cannot leave stale READY policy authorization.
        handle.worker_args_key = None
        handle.active_worker_args = {}
        handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
            int(handle.setup_generation), "apply_worker_args RPC completion",
            float(timeout_s))
        send_started = rpc_complete = False
        future = None
        try:
            send_started = True
            future = handle.workers.apply_worker_args.call(effective)
            value_mesh = handle._await_or_evict(
                future, timeout_s=timeout_s, holding_lock=True)
            rpc_complete = True
            mesh_helpers.publish_worker_args(
                handle, effective, handle._worker_args_key(effective))
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
            handle.setup_cleanup_state = (
                mesh_setup.cleanup_failure(
                    int(handle.setup_generation),
                    "apply_worker_args RPC completion", float(timeout_s), exc)
                if ambiguous else None
            )
            raise
        handle.setup_cleanup_state = None
    return [value for _point, value in value_mesh.items()]


@contextmanager
def temporary_worker_args(handle: Any, original: dict, overrides: dict,
                          timeout_s: float = 120.0) -> Iterator[list[dict]]:
    """Temporarily change a complete policy with unconditional rollback."""
    original_effective = handle.effective_worker_args(original)
    target = {**original_effective, **dict(overrides)}
    body_failed = False
    try:
        applied = handle.apply_worker_args(target, timeout_s=timeout_s, merge_config=False)
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
            handle.apply_worker_args(
                original_effective, timeout_s=timeout_s, merge_config=False)
        except BaseException as restore_exc:
            log.error("temporary worker-args restore failed: %r", restore_exc)
            if not body_failed:
                raise


def ensure_setup(handle: Any, topology: Topology, attention: str, sync_ulysses: bool = True,
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
    worker_args = handle.effective_worker_args(worker_args)
    # The NCCL/xfuser groups depend only on the parallel shape (topology,
    # attention, sync, fleet). worker_args (memory posture, safetensors
    # backend, LoRA residency mode) stay out of this key: tearing the groups
    # down for a memory toggle aborts the live USP communicator the next
    # render needs, and re-bringing up the same topology leaves xfuser
    # pointing at the destroyed group. Changed worker_args go through
    # apply_worker_args, which never touches NCCL.
    key = mesh_setup.nccl_setup_key(topology, attention, sync_ulysses)
    if fleet:
        key = (*key, "fleet")
    wa_key = handle._worker_args_key(worker_args)
    with handle.lock:
        from .mesh_session import require_mutation_authority

        mesh_setup.require_not_dirty(handle, "begin distributed setup")
        if handle.setup_key == key:
            if handle.worker_args_key != wa_key:
                mesh_setup.require_no_sample_leases(
                    handle, "change worker arguments")
                require_mutation_authority(
                    handle, "change worker arguments")
                handle.apply_worker_args(worker_args, timeout_s=budget.group_cleanup_s, merge_config=False)
            return []

        mesh_setup.require_no_sample_leases(handle, "change distributed setup")
        require_mutation_authority(handle, "change distributed setup")

        # Consume the setup generation before any destructive work. The
        # pre-increment value still salts the rendezvous port, while the
        # published value invalidates old artifact-preflight markers even
        # when teardown or setup fails. When replacing READY groups,
        # publish DIRTY before the generation store too: an interruption
        # after advancing generation must not leave a clean old setup key
        # whose workers still advertise the prior generation.
        attempt = handle.setup_generation
        base_port = handle.config.nccl_master_port or DEFAULT_NCCL_MASTER_PORT
        master_port = mesh_safety.bounded_master_port(
            base_port, os.getpid(), attempt, handle.world if fleet else 0)
        replacing_setup = handle.setup_key is not None
        if replacing_setup:
            handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                attempt + 1, "topology teardown", budget.group_cleanup_s)
        handle.setup_generation = attempt + 1

        if replacing_setup:
            log.info("topology change: tearing down old NCCL groups on live actors")
            # DIRTY precedes the generation store and READY retirement, so
            # every later store is fail-closed and a retry cannot reuse a
            # torn transition.
            handle.setup_key = None
            handle.topology = None
            try:
                cleaned = handle._await_or_evict(
                    handle.workers.teardown_group.call(),
                    timeout_s=budget.group_cleanup_s, holding_lock=True)
                mesh_setup.validate_cleanup_result(cleaned, handle.world)
            except BaseException as exc:
                state = mesh_setup.cleanup_failure(
                    handle.setup_generation, "topology teardown",
                    budget.group_cleanup_s, exc)
                handle.setup_cleanup_state = state
                if not isinstance(exc, Exception):
                    raise
                raise mesh_setup.TopologyTransitionError(
                    state, "start replacement setup") from exc
            handle.setup_cleanup_state = None

        # Initial bring-up is destructive too: actors can own partial live
        # groups before dispatch returns or the rollback handler starts.
        # Keep the generation DIRTY from before enqueue until READY is
        # atomically published or every rank confirms rollback.
        handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
            handle.setup_generation, "distributed setup", budget.rank_setup_s)
        fabric_env = handle.config.resolved_fabric_env()
        master_addr = handle.config.resolved_master_addr()
        results: list[dict] = []
        # The rank whose error is raised, None once every rank has answered.
        failed: dict[str, Any] = {"rank": None, "ahead": None, "spare": 0.0}
        try:
            deadline = mesh_setup.deadline_after(budget.rank_setup_s)
            futures = mesh_setup.dispatch_setup(
                handle, topology, attention, sync_ulysses, worker_args,
                fabric_env, master_addr, master_port, fleet)
            results = _await_setup(handle, futures, deadline, failed)
            failed["rank"] = None
            # READY publication is part of this same rollback transaction.
            # A BaseException at the call boundary after every worker has
            # accepted setup must still tear those groups back down rather
            # than leave a clean key-less driver over live actors.
            mesh_helpers.publish_setup_ready(
                handle, key, wa_key, worker_args, topology, attention,
                budget.rank_setup_s)
        except BaseException as setup_exc:
            # A rank's own setup error says which box it came from (worker_box.py):
            # the rank whose answer failed, in a Fleet too, where every actor is
            # rank 0 of its own world and can mark nothing.
            from . import worker_box

            worker_box.name(setup_exc, handle, failed["rank"])
            # A failure seen ahead of the rank being awaited leaves that rank in
            # setup, waiting at its rendezvous for a peer that will not come, and
            # the rollback's teardown queues behind it. The rollback keeps what
            # the setup's wait did not spend, so it ends no sooner than it did
            # when the driver waited the setup out before rolling back.
            rollback_s = budget.rollback_cleanup_s + float(failed["spare"])
            if failed["ahead"] is not None:
                log.warning(
                    "distributed setup: rank %d's setup failed while rank %d had not answered; the "
                    "rollback waits until the ranks still in setup end it (up to %.0f s; on a pair "
                    "a rank waits for its peer to join), then this setup raises rank %d's error",
                    failed["rank"], failed["ahead"], rollback_s, failed["rank"])
            # Some ranks may have completed setup before a peer failed.
            # Tear all of them back to the common uninitialized state;
            # otherwise a retry can mix generations of NCCL groups.
            handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                handle.setup_generation, "setup rollback", rollback_s)
            handle.setup_key = None
            handle.worker_args_key = None
            handle.topology = None
            try:
                cleaned = mesh_helpers.get_off_loop(
                    handle.workers.teardown_group.call(), rollback_s)
                mesh_setup.validate_cleanup_result(cleaned, handle.world)
            except BaseException as cleanup_exc:
                handle.setup_cleanup_state = mesh_setup.cleanup_failure(
                    handle.setup_generation, "setup rollback",
                    rollback_s, cleanup_exc)
                handle.defunct = True
                stop_exc = mesh_setup.stop_failed_setup_procs(handle)
                if stop_exc is not None:
                    log.error("failed-setup proc cleanup also failed: %r", stop_exc)
                log.error("failed-setup group rollback failed: %r", cleanup_exc)
            else:
                handle.setup_cleanup_state = (
                    None
                    if isinstance(setup_exc, Exception)
                    else mesh_setup.cleanup_failure(
                        handle.setup_generation,
                        "setup rollback",
                        rollback_s,
                        setup_exc,
                    )
                )
            raise
        mesh_setup.publish_worker_capabilities(handle, results)
        return results
