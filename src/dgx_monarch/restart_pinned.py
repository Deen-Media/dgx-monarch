"""A pinned session's restart: retire the stale fleet first when that is safe.

docs/TROUBLESHOOTING.md #2. A KSampler Pipeline or Identity Gate render runs in
a render session bound to the mesh it opened on (mesh_session.RenderSession), so
it cannot attach again inside its run. When every Worker loop restarted and
nothing else holds the fleet, ``refuse_pinned`` retires the stale fleet with the
heal's own ``restart_reattach._retire`` and refuses; the next render attaches a
fresh one. Otherwise it leaves the fleet as it always did. Kept out of
restart_reattach for its line limit; restart_reattach re-exports it, and every
call into that module goes through its attributes, read at call time.
"""
from __future__ import annotations

import time
from typing import Any, NoReturn

from . import driver_exit
from .delivery_faults import WorkerDeliveryError

_RETIRED = {"stop_confirmed": "its stop was confirmed",
            "liveness_release": "the liveness check found its processes gone"}


def _held(handle: Any) -> str:
    """What makes a pinned session leave this fleet as before, or "" for nothing."""
    from contextlib import nullcontext

    from . import mesh_safety, mesh_setup

    if mesh_safety.token_present(id(handle)):
        return "teardown_in_progress"  # read first: that stop can hold the lock
    lock = getattr(handle, "lock", None)
    with lock if lock is not None else nullcontext():
        try:
            mesh_setup.require_no_sample_leases(
                handle, "retire the stale mesh", allow_abandoned=True)
        except mesh_setup.SampleResultBusyError:
            return "leases_held"  # an earlier render's result is still in flight
        if (getattr(handle, "replacement_blocked", None)
                and not getattr(handle, "_replacement_retryable", False)):
            return "prior_stop_unknown"
    return ""


@driver_exit.quiet_at_exit()
def refuse_pinned(handle: Any, refused: WorkerDeliveryError) -> NoReturn:
    """Refuse a pinned session's restart, retiring the stale fleet first when safe.

    ``refused`` carries the restart check's reading, or is a delivery failure read here
    with ``PRECHECK_PROBE_S``. A full restart with nothing in ``_held`` is retired by
    ``_retire``; otherwise the fleet is left as before (the check's error is evicted).
    Quiet at exit, as the heal is; once the driver is exiting it reads no id and
    ends in the exit refusal.
    """
    from . import restart_reattach as rr

    started = time.monotonic()
    moves = getattr(refused, "moves", None)
    checked = isinstance(moves, rr.Moves)
    if handle is None:
        raise refused
    if not isinstance(moves, rr.Moves):
        rr._stand_down_at_exit(refused)  # no id reading over ssh once the driver is exiting
        moves = rr.read_moves(getattr(handle, "config", None), timeout=rr.PRECHECK_PROBE_S)
    rr._stand_down_at_exit(refused)  # at exit, after a reading too: no lock wait, the exit's refusal
    held = _held(handle) if moves.verdict == rr.ALL else moves.verdict
    if held:
        from . import mesh_helpers, mesh_teardown

        rr._record("refused", moves, started, reason=f"pinned_{held}")
        if checked:
            mesh_helpers.mark_defunct_preserving_primary(handle, refused)
        if held == "teardown_in_progress" and mesh_teardown.cleanup_running(handle):
            rr._refuse(f"every Worker loop serves a new proc id (on {rr._labels(handle, moves.moved)}), "
                       "and this render cannot attach again inside its run: "
                       f"{mesh_teardown.cleanup_running_sentence()}.", refused)
        raise refused
    try:
        teardown = rr._retire(handle, moves, refused)
    except WorkerDeliveryError:
        rr._record("refused", moves, started, reason="pinned_not_retired")
        raise
    rr._stand_down_at_exit(refused)  # an exit during the liveness probe: the exit's refusal
    rr._record("retired", moves, started, teardown=teardown, reason="pinned_session")
    rr._refuse(
        f"every Worker loop serves a new proc id (on {rr._labels(handle, moves.moved)}): "
        "the loops restarted under this attached mesh (`dgxm restart`, a crash that "
        "systemd restarted, or a reboot) and its actors are gone. This render runs in a "
        "session pinned to that mesh (a pipelined render, an identity-gate leg or the "
        "Identity Gate node), so it cannot attach again inside its run; this driver "
        f"retired the stale mesh instead ({_RETIRED.get(teardown, teardown)}). Nothing "
        "was loaded. The Worker loops restarted; queue again: the next render attaches "
        "a fresh mesh by the new ids.", refused)
