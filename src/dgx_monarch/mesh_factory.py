"""Worker-fleet construction kept separate from cache/lifecycle locking."""
from __future__ import annotations

import threading
import time
from typing import Any

from . import proc_stops, retired_meshes
from .error_utils import raise_with_distinct_cause, reconcile_error, safe_note
from .mesh_runtime import get_off_loop, kept_root_log_level
from .mesh_stop import RECONCILE_GROUP_S, cleanup_stop_s, stop_procs

Fleet = tuple[Any, Any, Any, bool]


class FleetHandoff:
    """Caller-owned publication cell spanning both ProcMesh return boundaries."""

    def __init__(self) -> None:
        self.proc_meshes: list[Any] = []
        self.fleet: Fleet | None = None
        self.cleanup_started = False
        self.cleanup_confirmed = False
        self.callback_cleanup_unconfirmed = False
        self.failure_latched = False
        self.ready_for_publication = False
        self.spawn_host: Any | None = None
        self.spawn_host_isolated = False
        self.proc_spawn_called = False
        self.proc_ownership_unconfirmed = False
        self.transport_in_progress = False
        self.attach_in_progress = False
        self.transport_failure_safe = False
        self.spawn_in_progress = False
        self.bootstrap_refusal: BaseException | None = None


def recording_bootstrap_refusal(command: Any, handoff: FleetHandoff) -> Any:
    """Wrap the per-rank launch command so its own refusal can be recognized.

    The pinned Monarch runs the callable once per rank on the calling thread
    before spawning begins, and its error raises from ``spawn_procs`` itself
    (tests/test_spawn_contract.py). That exception therefore proves no
    process started; Monarch 0.7.0 called it inside the native spawn.
    """
    def command_for(point: Any) -> Any:
        try:
            return command(point)
        except BaseException as exc:
            handoff.bootstrap_refusal = exc
            raise
    return command_for


def safe_failure_evidence(value: Any) -> str:
    try:
        return value if type(value) is str else repr(value)
    except BaseException:
        return f"<{type(value).__name__}>"


def safe_failure_message(value: Any) -> str:
    try:
        return str(value)
    except BaseException:
        return safe_failure_evidence(value)


class _CreationCleanupPoison(str):
    pass


CREATION_CLEANUP_POISON = _CreationCleanupPoison(
    "mesh creation failed; owned ProcMesh cleanup unconfirmed")
_CONFIRMED_CLEANUP = object()
_CREATION_CLEANUP_CLAIMS: dict[object, FleetHandoff | object | None] = {}
_CREATION_CLEANUP_LOCK = threading.Lock()


def _cleanup_pending(handoff: FleetHandoff | object | None) -> bool:
    if handoff is _CONFIRMED_CLEANUP:
        return False
    if handoff is None:
        return True
    if not isinstance(handoff, FleetHandoff):
        return True
    if not handoff.failure_latched:
        return (
            (handoff.attach_in_progress and not handoff.transport_failure_safe)
            or handoff.cleanup_started
            or handoff.callback_cleanup_unconfirmed)
    return _unresolved_cleanup(handoff)


def _unresolved_cleanup(handoff: FleetHandoff) -> bool:
    return (
        (handoff.attach_in_progress and not handoff.transport_failure_safe)
        or handoff.spawn_in_progress
        or handoff.callback_cleanup_unconfirmed
        or (handoff.cleanup_started and not handoff.cleanup_confirmed)
        or (not handoff.cleanup_confirmed
            and (bool(handoff.proc_meshes) or handoff.fleet is not None)))


def creation_cleanup_settled(handoff: FleetHandoff) -> bool:
    """Whether this exact failed attempt has no unresolved native ownership."""
    return not _unresolved_cleanup(handoff)


def begin_creation_cleanup(
    claim: object | None = None, handoff: FleetHandoff | None = None, *,
    reserve_transport: bool = False,
) -> object:
    if claim is None:
        claim = object()
    with _CREATION_CLEANUP_LOCK:
        _CREATION_CLEANUP_CLAIMS[claim] = handoff
        if reserve_transport and handoff is not None:
            handoff.transport_in_progress = True
    return claim


def confirm_creation_cleanup(claim: object) -> bool:
    with _CREATION_CLEANUP_LOCK:
        if claim in _CREATION_CLEANUP_CLAIMS:
            _CREATION_CLEANUP_CLAIMS[claim] = _CONFIRMED_CLEANUP
        _CREATION_CLEANUP_CLAIMS.pop(claim, None)
        return not any(_cleanup_pending(handoff)
                       for handoff in _CREATION_CLEANUP_CLAIMS.values())


def creation_cleanup_registered(claim: object) -> bool:
    with _CREATION_CLEANUP_LOCK:
        return claim in _CREATION_CLEANUP_CLAIMS


def creation_cleanup_pending() -> bool:
    with _CREATION_CLEANUP_LOCK:
        return any(_cleanup_pending(handoff)
                   for handoff in _CREATION_CLEANUP_CLAIMS.values())


def spawn_failure_poison() -> str:
    return _CreationCleanupPoison(
        "actor spawn failed; cleanup state unavailable")


def safe_transport_failure(handoff: FleetHandoff | None) -> None:
    if handoff is not None:
        handoff.transport_failure_safe = True


def reconcile_creation_cleanup(
    handoff: FleetHandoff, handle: Any | None, procs: Any, primary: BaseException,
) -> str | None:
    """Stop once; retain an unconfirmed outcome without fallible diagnostics."""
    handoff.cleanup_started = True
    try:
        if handle is None:
            stop_procs(procs, "dgx-monarch client detach", cleanup_stop_s(), "dgxm-creation-rollback")
        elif handle.teardown_complete:
            handoff.cleanup_confirmed = True
            handle._retire_completed()
        else:
            handle.shutdown(timeout_s=RECONCILE_GROUP_S, stop_timeout_s=cleanup_stop_s())
            handoff.cleanup_confirmed = True
            handle._retire_completed()
    except BaseException as cleanup_exc:
        if handle is not None and handle.teardown_complete:
            handoff.cleanup_confirmed = True
            try:
                handle._retire_completed()
            except BaseException as retire_exc:
                safe_note(
                    primary, "confirmed ProcMesh eviction retry failed: "
                    f"{safe_failure_evidence(retire_exc)}")
        if handoff.cleanup_confirmed:
            return None
        try:
            existing = handle.replacement_blocked if handle is not None else None
        except BaseException:
            existing = None
        return (existing if isinstance(existing, str) and existing else
                "mesh creation rollback outcome is unconfirmed: "
                f"{safe_failure_evidence(cleanup_exc)}")
    else:
        handoff.cleanup_confirmed = True
        return None


_PROC_SPAWN_HANDOFF_LOCK = threading.Lock()


def _unregister_callback_once(unregister: Any, callback: Any,
                              handoff: FleetHandoff) -> None:
    """One removal attempt; absence proves an earlier attempt already landed."""
    try:
        unregister(callback)
    except ValueError:
        handoff.callback_cleanup_unconfirmed = False
    else:
        handoff.callback_cleanup_unconfirmed = False


def _isolated_spawn_host(hosts: Any) -> tuple[Any, bool]:
    """Give one production spawn an identity no foreign HostMesh call shares."""
    try:
        from monarch._src.actor.host_mesh import HostMesh
    except ImportError as exc:
        if type(hosts).__module__.startswith("monarch."):
            raise RuntimeError(
                "the pinned private HostMesh isolation surface is unavailable") from exc
        return hosts, True  # explicit non-Monarch test doubles are invocation-local
    if not isinstance(hosts, HostMesh):
        if type(hosts).__module__.startswith("monarch."):
            raise RuntimeError(
                "the attached host object cannot be isolated by exact identity")
        return hosts, True
    inner = getattr(hosts, "_inner_host_mesh", None)
    if inner is None:
        raise RuntimeError("HostMesh no longer owns a native mesh")
    return (
        HostMesh(
            inner, hosts.region, hosts.stream_logs, hosts.is_fake_in_process, None),
        True,
    )


def owned_spawned_proc(
    handoff: FleetHandoff, returned: Any | None = None,
) -> Any | None:
    """Resolve exact invocation ownership; ambiguity never grants stop authority."""
    if returned is not None:
        return returned
    if not handoff.spawn_host_isolated:
        if handoff.proc_spawn_called:
            handoff.proc_ownership_unconfirmed = True
        return None
    spawn_host = handoff.spawn_host
    candidates = list(handoff.proc_meshes)
    candidates.extend(getattr(spawn_host, "_proc_meshes", ()))
    try:
        from monarch._src.actor.proc_mesh import get_active_proc_meshes

        candidates.extend(get_active_proc_meshes())
    except (ImportError, AttributeError):
        pass
    missing_host = object()
    matched = [
        candidate for candidate in candidates
        if getattr(candidate, "_host_mesh", missing_host) is spawn_host
    ]
    unique = {id(candidate): candidate for candidate in matched if candidate is not None}
    if len(unique) == 1:
        return next(iter(unique.values()))
    if handoff.proc_spawn_called:
        handoff.proc_ownership_unconfirmed = True
    return None


_ABORTED = "actor explicitly aborted due to: "
_TRACEBACK = "Traceback (most recent call last):"


def worker_words(text: str) -> str | None:
    """The failing actor's own exception line inside a Monarch supervision text.

    Under direct spawn Monarch reports a GPUWorker constructor that raised from
    ``initialized`` as "error spawning actor mesh: statuses: Failed(supervisor
    proxy ...", with the Worker's traceback nested inside; managed spawn
    reported it from the first call. Both nest it after ``_ABORTED``. Returns the line that
    closes the last traceback there, or None when the text holds none.
    """
    lines = text.splitlines()
    start = max((index for index, line in enumerate(lines) if _ABORTED in line), default=None)
    if start is None:
        return None
    head = lines[start].split(_ABORTED, 1)[1].strip()
    if head and head != _TRACEBACK:
        return head
    base = len(lines[start]) - len(lines[start].lstrip())
    words = None
    for line in lines[start + 1:]:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not stripped:
            continue
        if indent < base:
            break
        if indent > base:
            continue
        if stripped == _TRACEBACK:
            words = None
        elif words is None and not stripped.startswith(("During handling", "The above exception")):
            words = stripped
    return words


def lead_with_worker_words(text: str) -> str:
    words = worker_words(text)
    return text if not words or text.startswith(words) else f"{words}; Monarch reported: {text}"


class WorkerSpawnError(RuntimeError):
    def __init__(self, original: BaseException, cleanup: BaseException | None,
                 local_transport_initialized: bool = False,
                 rollback_confirmed: bool = False):
        self.original_text = lead_with_worker_words(safe_failure_message(original))
        self.cleanup_text = (safe_failure_evidence(cleanup)
                             if cleanup is not None else "")
        super().__init__(self.original_text)
        self.original = original
        self.cleanup = cleanup
        self.local_transport_initialized = local_transport_initialized
        self.rollback_confirmed = rollback_confirmed

    def raise_cancellation(self) -> None:
        """Restore the exact cancellation winner after caller state is safe."""
        original_cancel = not isinstance(self.original, Exception)
        cleanup_cancel = (
            self.cleanup is not None and not isinstance(self.cleanup, Exception))
        if not original_cancel and not cleanup_cancel:
            return
        if original_cancel:
            winner, cause = self.original, self.cleanup
        elif self.cleanup is not None:
            winner, cause = self.cleanup, self.original
        else:
            return
        if winner is cause:
            raise winner from None
        raise winner from cause


def _mesh_name(workers: Any) -> str | None:
    """The new mesh's name, read once at bring-up for the fault hook (retired_meshes)."""
    try:
        return get_off_loop(workers._name, 5.0, "dgxm-mesh-name")
    except Exception:
        return None


def spawn_worker_fleet(config, use_cluster: bool, gpus_per_host: int,
                       attach_cluster, bootstrap,
                       handoff: FleetHandoff | None = None) -> Fleet:
    if handoff is None:
        handoff = FleetHandoff()
    if use_cluster:
        try:
            handoff.attach_in_progress = True
            hosts = attach_cluster(config, handoff)
            handoff.spawn_in_progress = True
            handoff.attach_in_progress = False
            owns_hosts = True
        except BaseException:
            handoff.failure_latched = True
            raise
    else:
        try:
            from monarch.actor import this_host
        except ImportError as exc:
            error = RuntimeError(
                f"torchmonarch is not importable ({exc}). Install dgx-monarch with "
                "ComfyUI's Python as docs/INSTALL.md shows; a bare `pip install -r requirements.txt` "
                "cannot find the patched xfuser wheel the guide builds.")
            raise WorkerSpawnError(error, None, rollback_confirmed=True) from exc
        handoff.spawn_in_progress = True
        with kept_root_log_level():
            hosts, owns_hosts = this_host(), False
    from monarch._src.actor.proc_mesh import (
        register_proc_mesh_spawn_callback,
        unregister_proc_mesh_spawn_callback,
    )

    from .actor import GPUWorker
    from .monarch_launch import actor_bootstrap_command, wait_initialized, wait_worker_ready
    from .runtime_diagnostics import phase

    capture = handoff.proc_meshes.append
    handoff.spawn_host, handoff.spawn_host_isolated = _isolated_spawn_host(hosts)
    procs = None
    proc_name = proc_stops.fresh_name()  # its controller's reports name it (proc_stops)
    try:
        deadline = time.monotonic() + config.monarch.startup_timeout_s
        launch_command = recording_bootstrap_refusal(
            actor_bootstrap_command(config, use_cluster), handoff)
        with _PROC_SPAWN_HANDOFF_LOCK:
            handoff.callback_cleanup_unconfirmed = True
            operation_error: BaseException | None = None
            cause: BaseException | None = None
            try:
                register_proc_mesh_spawn_callback(capture)
                handoff.proc_spawn_called = True
                with phase("spawn"):
                    procs = handoff.spawn_host.spawn_procs(
                        per_host={"gpus": gpus_per_host}, bootstrap=bootstrap,
                        bootstrap_command=launch_command, name=proc_name)
            except BaseException as exc:
                operation_error = exc
                if exc is handoff.bootstrap_refusal and not handoff.proc_meshes:
                    handoff.proc_spawn_called = False  # refused before any process started
            for _attempt in range(2):
                if not handoff.callback_cleanup_unconfirmed:
                    break
                try:
                    _unregister_callback_once(
                        unregister_proc_mesh_spawn_callback, capture, handoff)
                except BaseException as exc:
                    if operation_error is None:
                        operation_error = exc
                    else:
                        operation_error, cause = reconcile_error(
                            operation_error, exc,
                            "ProcMesh spawn callback removal also failed")
            if operation_error is not None:
                raise_with_distinct_cause(operation_error, cause)
        owned_procs = owned_spawned_proc(handoff, procs)
        if owned_procs is None:
            raise RuntimeError(
                "spawned ProcMesh ownership could not be proven for this invocation")
        procs = owned_procs
        proc_stops.remember(procs, proc_name)
        wait_initialized(procs, deadline, "dgxm-procs-ready")
        with phase("spawn"):
            workers = procs.spawn("dgxm_worker", GPUWorker)
        wait_worker_ready(workers, deadline)
        retired_meshes.remember(workers, _mesh_name(workers))
        handoff.fleet = hosts, procs, workers, owns_hosts
        handoff.spawn_in_progress = False
        return hosts, procs, workers, owns_hosts
    except BaseException as exc:
        handoff.failure_latched = True
        cleanup_error = None
        try:
            owned_procs = owned_spawned_proc(handoff, procs)
        except BaseException as ownership_exc:
            owned_procs = None
            if handoff.proc_spawn_called:
                handoff.proc_ownership_unconfirmed = True
            safe_note(
                exc, "spawned ProcMesh ownership recovery also failed",
                ownership_exc,
            )
        rollback_confirmed = not handoff.proc_spawn_called
        if owned_procs is not None:
            handoff.cleanup_started = True
            try:
                proc_stops.remember(owned_procs, proc_name)
                stop_procs(owned_procs, "dgx-monarch partial bring-up rollback", cleanup_stop_s(),
                           "dgxm-bringup-rollback")
                handoff.cleanup_confirmed = True
                rollback_confirmed = True
            except BaseException as stop_exc:
                cleanup_error = stop_exc
        elif handoff.proc_ownership_unconfirmed:
            cleanup_error = RuntimeError(
                "spawned ProcMesh ownership is unconfirmed; no foreign process was stopped")
        raise WorkerSpawnError(
            exc, cleanup_error,
            local_transport_initialized=not use_cluster,
            rollback_confirmed=rollback_confirmed) from exc
