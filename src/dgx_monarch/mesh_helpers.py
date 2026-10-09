"""Small helpers and the fault-absorption surfaces for the mesh lifecycle."""
from __future__ import annotations

from collections.abc import Callable, Collection
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from . import absorb_fire as _absorb
from . import client_lease as _client_lease
from . import mesh_runtime as _mesh_runtime
from . import mesh_safety, mesh_teardown
from .error_utils import (
    failure_summary,
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_note,
)
from .log import get_logger

if TYPE_CHECKING:
    from .config import ClusterConfig

MeshPreflight = Callable[["ClusterConfig", int], None]

attach_once = _mesh_runtime.attach_once
bootstrap_proc = _mesh_runtime.bootstrap_proc
config_fingerprint = _mesh_runtime.config_fingerprint
detect_comfy_dir = _mesh_runtime.detect_comfy_dir
get_off_loop = _mesh_runtime.get_off_loop
local_worker_args_fingerprint = _mesh_runtime.local_worker_args_fingerprint
mesh_cache_key = _mesh_runtime.mesh_cache_key
mesh_creation_timeout_s = _mesh_runtime.mesh_creation_timeout_s
run_blocking_off_loop = _mesh_runtime.run_blocking_off_loop
src_pythonpath = _mesh_runtime.src_pythonpath
start_client_lease = _client_lease.start
validated_local_gpu_count = _mesh_runtime.validated_local_gpu_count
visible_gpu_count = _mesh_runtime.visible_gpu_count
_latch_reconcile_launch_failure = mesh_teardown.latch_reconcile_launch_failure
_reconcile_thread_main = mesh_teardown.reconcile_thread_main
_reconcile_thread_started = mesh_teardown.reconcile_thread_started


def run_mesh_preflight(
    callback: MeshPreflight | None, config: ClusterConfig, world: int
) -> None:
    if callback is not None:
        callback(config, world)


def shutdown_all(handles: dict, log: Any) -> None:
    """Best-effort cleanup after Monarch's own LIFO atexit hook has run."""
    for handle in list(handles.values()):
        try:
            handle.shutdown(timeout_s=15)
        except BaseException as exc:
            # Native shutdown may panic after Monarch tears down its client.
            log.debug("atexit mesh shutdown: %r", exc)
    handles.clear()


def wait_for_mesh_creation(condition, creating: Collection[tuple], key: tuple,
                           error_type: type[RuntimeError]) -> None:
    """Bound one same-key creation wait while the caller holds ``condition``.

    It reads ``mesh_creation_timeout_s`` from this module, where tests patch it.
    """
    try:
        timeout_s = mesh_creation_timeout_s()
    except ValueError as exc:
        raise error_type(str(exc)) from exc
    if condition.wait_for(lambda: key not in creating, timeout=timeout_s):
        return
    raise error_type(
        f"timed out after {timeout_s:g}s waiting for another caller to finish "
        "creating this mesh; inspect the driver log and retry after it finishes, "
        "or restart ComfyUI if that creator is wedged")

# The fault surfaces log under the mesh module's name, the logger operators
# read; tests capture it by that name.
_mesh_log = get_logger("dgx_monarch.mesh")


def publish_worker_args(
    handle: Any, worker_args: dict, worker_args_key: tuple
) -> None:
    """Publish policy metadata before its final fast-path marker."""
    try:
        handle.active_worker_args = dict(worker_args)
        handle.worker_args_key = worker_args_key
    except BaseException:
        # The broadcast already succeeded, but a torn driver publication
        # cannot authorize reuse. The next request must reapply the complete
        # policy before setup-bound work can fast-path.
        handle.worker_args_key = None
        handle.active_worker_args = {}
        raise


def publish_setup_ready(
    handle: Any,
    setup_key: tuple,
    worker_args_key: tuple,
    worker_args: dict,
    topology: Any,
    attention: str,
    budget_s: float,
) -> None:
    """Publish complete setup metadata under a pessimistic DIRTY latch."""
    from . import mesh_setup

    try:
        # Workers already own live groups. Publish fail-closed state before the
        # first metadata store so an interruption cannot expose a clean key-less
        # driver handle over those groups. Keep this first store inside the
        # helper's own compensation scope as well as ensure_setup's rollback.
        handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
            int(handle.setup_generation), "setup readiness publication", budget_s)
        handle.active_worker_args = dict(worker_args)
        handle.topology = topology
        handle.attention = attention
        handle.worker_args_key = worker_args_key
        # Store every field setup_key qualifies first; clearing DIRTY last is
        # what authorizes READY.
        handle.setup_key = setup_key
        handle.setup_cleanup_state = None
    except BaseException as exc:
        # Workers may already own live groups, so a torn publication is an
        # ambiguous generation rather than permission to dispatch setup again.
        handle.setup_cleanup_state = mesh_setup.cleanup_failure(
            int(handle.setup_generation),
            "setup readiness publication",
            budget_s,
            exc,
        )
        handle.setup_key = None
        handle.worker_args_key = None
        handle.active_worker_args = {}
        handle.topology = None
        raise


def install_unhandled_fault_hook(correlation_state) -> bool:
    """Route faults to logs/toasts, consuming them for per-actor recovery.
    Deliberate teardown noise is INFO; other faults are ERROR, and an exact
    repeat is a counted line. The supplied callable returns a coherent
    ``(live, creating, teardown-in-flight)`` snapshot. Returns whether the
    hook was installed.
    """
    from . import fault_repeats, mesh_safety
    from .log import get_logger

    hook_log = get_logger("dgx_monarch.mesh")
    repeats = fault_repeats.RepeatGate()
    try:
        import monarch.actor as monarch_actor

        def hook(failure) -> None:
            text = str(failure)
            # A fault carrying a stop reason always absorbs: the reason is the
            # attribution, so no correlation or coherence check may reject it.
            if mesh_safety.is_reason_marked_teardown(text):
                hook_log.info("cluster teardown notice (deliberate): %s",
                              text.split("\n")[0][:160])
                return
            # A fault without a reason correlates under one coherent snapshot.
            # The state version bumps on every teardown transition and the
            # bracket closes after classification, so every mutable read, the
            # grace deadline included, sits inside it; a transition in between
            # leaves the fault visible. correlation_state() counts a stop in
            # flight only from a token whose handle has published neither
            # teardown_complete nor replacement_blocked, so an interrupt in
            # end_deliberate_teardown before its store cannot hide faults
            # forever.
            version = mesh_safety.teardown_state_version()
            live, creating, in_flight = correlation_state()
            absorb = mesh_safety.is_deliberate_teardown_fault(
                text,
                teardown_in_progress=in_flight,
                creation_in_progress=creating,
                live_mesh_exists=live)
            if absorb and mesh_safety.teardown_state_version() == version:
                hook_log.info("cluster teardown notice (deliberate): %s",
                              text.split("\n")[0][:160])
                return
            if not repeats.first_report(text, hook_log):
                return
            hook_log.error("cluster fault: %s", failure)
            try:
                from server import PromptServer

                PromptServer.instance.send_sync(
                    "dgx-monarch.fault", {"message": str(failure)})
            except Exception:
                pass  # headless / non-comfy driver

        monarch_actor.unhandled_fault_hook = hook
        return True
    except Exception as exc:
        hook_log.warning("could not install fault hook: %r", exc)
        return False


def _publish_supervision_failure_state(
    handle, exc: BaseException, text: str, *, deliberate_route: bool = False,
) -> tuple[object, bool, bool, bool]:
    """Atomically publish fail-closed supervision state under the handle lock.

    The caller retries this whole helper if an asynchronous exception lands
    at its call boundary. Re-entry adopts the same token and
    completes whichever of the defunct/deferred fields was not yet stored.
    """
    lock = getattr(handle, "lock", None)
    with lock if lock is not None else nullcontext():
        mesh_teardown.retire_retryable_reconcile(handle)
        token = getattr(handle, "_supervision_reconcile_token", None)
        if token is None:
            token = object()
            handle._supervision_reconcile_token = token
        # Publish denial before any branch can return or launch cleanup.
        handle.defunct = True
        deliberate = (
            mesh_safety.is_reason_marked_teardown(text)
            or getattr(handle, "teardown_complete", False)
            or (mesh_safety.token_within_authority(id(handle))
                and not getattr(handle, "replacement_blocked", None))
        )
        if not deliberate_route and not deliberate:
            # The only writer, and only where nothing owns this teardown: a
            # fault carrying a Reset's own stop reason is a teardown someone
            # pressed, not a death, so no liveness proof may release its stop timeout.
            handle._supervision_death_report = text[:200]
        completed = bool(getattr(handle, "teardown_complete", False))
        deferred = False
        lease_count = sum(
            int(v) for v in getattr(handle, "sample_leases", {}).values())
        if not deliberate and lease_count:
            if getattr(handle, "deferred_supervision_error", None) is None:
                handle.deferred_supervision_error = exc
            # ``deliberate`` attributes a fault to an existing teardown;
            # ``deliberate_route`` records driver-initiated eviction for its resume
            # path. Missing dynamic attributes default to a non-deliberate route.
            if deliberate_route:
                handle._deferred_eviction_deliberate = True
            deferred = True
        return token, deliberate, completed, deferred


def _finish_supervision_failure_publication(
    handle,
    exc: BaseException,
    text: str,
    *,
    holding_lock: bool,
    deliberate_route: bool = False,
) -> None:
    token, deliberate, completed, deferred = _publish_supervision_failure_state(
        handle, exc, text, deliberate_route=deliberate_route)
    if deliberate:
        # Same authority rule as the hook: an expired token with no outcome
        # has no say. The publication helper reports deliberate only while
        # live authority or a terminal outcome exists.
        if completed:
            mesh_safety.note_deliberate_teardown()
        _mesh_log.info(
            "mesh teardown is deliberate (done or in progress); stray in-flight call absorbed")
        return
    if deferred:
        _mesh_log.info(
            "deferring failed-fleet reconciliation until sample/read leases retire")
        return
    _launch_reconcile_thread(handle, exc, token, wait=not holding_lock)


def mark_defunct_on_supervision_failure(handle, exc: BaseException, *,
                                        holding_lock: bool = False) -> None:
    """Evict a mesh whose procs died so the next Init/render respawns.
    Endpoint exceptions (ActorError wrapping a worker failure) leave the actor
    alive; only supervision failures (dead proc/host or lost link) poison it.
    ``holding_lock=True`` means ensure_setup or apply_worker_args already holds
    handle.lock. shutdown() may move off-loop work to a scratch thread, and
    reacquiring the caller-held lock there deadlocks: RLock ownership is
    thread-specific, so Comfy freezes through the join timeout.
    Reconciliation therefore detaches: the original failure re-raises while
    lifecycle fields make a racing attach refuse unresolved state, then a later
    retry heals after cleanup.
    """
    from monarch.actor import ActorError

    # ActorError is the endpoint-error boundary: the actor is still alive and
    # its worker exception belongs to the caller. Return before stringifying
    # it because ActorError.__str__ expands the complete remote traceback.
    # Subclasses are included so a hostile wrapper cannot bypass this privacy
    # boundary; typed callers remain responsible for validating the preserved
    # inner exception before accepting any worker-side refusal.
    if isinstance(exc, ActorError):
        return
    names = {type(e).__name__ for e in (exc, exc.__cause__, exc.__context__) if e is not None}
    text = f"{names} {failure_summary(exc)}"
    # Match the exported SupervisionError type first. The string list is a
    # secondary net only, audited against the 0.6.0 wheel. The sole live string
    # was "Supervision" (from the type name), so on their own the strings would
    # miss a renamed failure and let a dead fleet be reused; isinstance survives
    # a rename of the message text.
    try:
        from monarch._rust_bindings.monarch_hyperactor.supervision import (
            SupervisionError,
        )
        supervision_chain = any(
            isinstance(e, SupervisionError)
            for e in (exc, exc.__cause__, exc.__context__) if e is not None
        )
    except ImportError:
        supervision_chain = False
    if not supervision_chain and not _absorb.hits(
            "string-fallback", text, ("Supervision", "ProcessExited",
                                      "connection lost", "peer closed")):
        return
    _publish_defunct(handle, exc, text, holding_lock=holding_lock)


def mark_defunct_deliberate(handle, exc: BaseException, *,
                            holding_lock: bool = False) -> None:
    """Evict a fleet the driver judged unusable, skipping the supervision classifier.

    The stall guard (mesh_lease.collect_with_liveness) reaches a verdict the
    classifier cannot: no supervision event fired and the actor may still
    answer health queries, but the fleet said nothing for the whole stall
    budget. Publication uses the supervision path's own machinery, so process
    state that cannot be confirmed stays visible as after a supervision failure.
    """
    text = f"deliberate-eviction {failure_summary(exc)}"
    _publish_defunct(handle, exc, text, holding_lock=holding_lock,
                     deliberate_route=True)


def _publish_defunct(handle, exc: BaseException, text: str, *,
                     holding_lock: bool, deliberate_route: bool = False) -> None:
    if (mesh_safety.is_reason_marked_teardown(text)
            or getattr(handle, "teardown_complete", False)
            or (mesh_safety.token_within_authority(id(handle))
                and not getattr(handle, "replacement_blocked", None))):
        # The teardown owner can hold handle.lock while its proc stop blocks.
        # Never wait behind it to learn that a second stop is forbidden: the
        # live teardown token is already the authority.
        handle.defunct = True
        if getattr(handle, "teardown_complete", False):
            mesh_safety.note_deliberate_teardown()
        _mesh_log.info(
            "mesh teardown is deliberate (done or in progress); stray in-flight call absorbed")
        return
    try:
        _finish_supervision_failure_publication(
            handle, exc, text, holding_lock=holding_lock,
            deliberate_route=deliberate_route)
    except BaseException as primary:
        # Compensate the three dangerous boundaries: token-before-defunct,
        # defunct-before-deferred, and defunct-before-thread-launch. Re-entry
        # is idempotent and either records the lease-triggered resume or adopts
        # and starts the one shared reconciliation thread.
        completed = False
        winner = primary
        for _attempt in range(2):
            try:
                _finish_supervision_failure_publication(
                    handle, exc, text, holding_lock=holding_lock,
                    deliberate_route=deliberate_route)
            except BaseException as cleanup_exc:
                winner = prefer_error(
                    winner, cleanup_exc,
                    "supervision failure publication compensation also failed")
            else:
                completed = True
                break
        if not completed:
            try:
                _latch_reconcile_launch_failure(
                    handle,
                    "supervision failure state could not reach a terminal publication",
                )
            except BaseException as latch_exc:
                winner = prefer_error(
                    winner, latch_exc,
                    "supervision failure terminal latch also failed")
        if winner is not primary:
            raise_with_distinct_cause(winner, primary)
        raise


def mark_defunct_preserving_primary(
    handle, exc: Exception, *, holding_lock: bool = False
) -> None:
    """Attempt supervision publication without replacing ``exc``.

    Callers use this only from an active exception handler and re-raise the
    root failure themselves. Publication is still compensated by
    :func:`mark_defunct_on_supervision_failure`; a terminal publication error
    is attached for diagnosis instead of masking the worker/RDMA failure.
    """
    try:
        mark_defunct_on_supervision_failure(
            handle, exc, holding_lock=holding_lock)
    except BaseException as publication_exc:
        safe_note(
            exc, "supervision failure publication also failed", publication_exc)


def _launch_reconcile_thread(
    handle, exc: BaseException, token: object, *, wait: bool = False
) -> None:
    """Publish/start one shared reconciliation thread for this dead handle."""
    import threading

    lock = getattr(handle, "lock", None)
    creator = False
    thread = None
    started = False
    # Re-entry adopts a thread published before an interrupted start.
    try:
        for _attempt in range(2):
            try:
                with lock if lock is not None else nullcontext():
                    current = getattr(
                        handle, "_supervision_reconcile_thread", None)
                    if current is None:
                        if thread is None:
                            creator = True
                            thread = threading.Thread(
                                target=_reconcile_thread_main,
                                args=(handle, exc, token),
                                name="dgxm-reconcile-stop",
                                daemon=True,
                            )
                        handle._supervision_reconcile_thread = thread
                    else:
                        thread = current
                        if current is not getattr(
                                handle, "_supervision_reconcile_thread", None):
                            creator = False
                    if _reconcile_thread_started(thread):
                        started = True
                        break
                    launch_block = getattr(handle, "replacement_blocked", None)
                    if isinstance(launch_block, str) and launch_block.startswith(
                            "could not launch supervision reconcile thread"):
                        handle.replacement_blocked = None
                        handle._replacement_retryable = False
                    thread.start()
                    started = True
                    break
            except BaseException as launch_exc:
                safe_note(
                    exc,
                    "supervision reconcile thread launch failed during "
                    "construction/publication/start",
                    launch_exc)
                if thread is not None and _reconcile_thread_started(thread):
                    started = True
                    break
    finally:
        if not started:
            try:
                _latch_reconcile_launch_failure(
                    handle,
                    "could not launch supervision reconcile thread after two attempts",
                )
            except BaseException as latch_exc:
                # Retry the terminal refusal publication once if it fails or is
                # interrupted, before this handler returns.
                safe_note(
                    exc, "supervision reconcile launch latch failed", latch_exc)
                try:
                    _latch_reconcile_launch_failure(
                        handle,
                        "could not launch supervision reconcile thread after two attempts",
                    )
                except BaseException as retry_exc:
                    winner, cause = reconcile_error(
                        latch_exc, retry_exc,
                        "supervision reconcile launch latch retry failed")
                    raise_with_distinct_cause(winner, cause)
                if not isinstance(latch_exc, Exception):
                    raise
    if not started:
        return
    if creator and wait and thread is not None:
        try:
            thread.join(timeout=95.0)
        except BaseException as join_exc:
            safe_note(
                exc, "supervision reconcile thread join was interrupted", join_exc)
            if not isinstance(join_exc, Exception):
                raise
