"""What a refusal says while the supervision reconcile is still stopping a dead fleet.

After a supervision failure (a lost Worker), a stall eviction, or a deliberate
eviction resumed at lease retirement, the driver's reconcile stops the old
fleet on a thread of its own (mesh_helpers._launch_reconcile_thread), holding
the handle's lock across its proc stop. A render queued or a Reset pressed in
that window is refused, and a Reset or a ComfyUI restart cannot help before the
stop ends. mesh_teardown's refusals and the Reset's use these words then; the
verdicts and the locks are the stop code's own. Kept out of mesh_teardown for
its line limit; mesh_teardown re-exports them.
"""
from __future__ import annotations

import time
from typing import Any

from .mesh_stop import ACTOR_STOP_S, proc_stop_bound_s, reconcile_join_s

RECONCILE_STARTED = "_supervision_reconcile_started"


def lost_fleet_stop_s() -> float:
    """About how long the reconcile's stop of a dead fleet takes at most: the user's figure.

    The reconcile's shutdown runs the group teardown, the actor stop, then the
    proc stop. On a lost Worker the group teardown call fails at once on the
    dead actor (on loopback in the same second the render's call failed), so
    its phase is not counted: the figure is the actor stop's cap plus the proc
    stop's own limit, both read from the stop code, about 105 s at Monarch's
    defaults. reconcile_join_s, the render's own wait, adds the group phase
    and margins on top.
    """
    return ACTOR_STOP_S + proc_stop_bound_s()


def note_started(handle: Any) -> None:
    """Stamp the reconcile's start on its handle (the thread's first act); never raises."""
    try:
        setattr(handle, RECONCILE_STARTED, time.monotonic())
    except Exception:
        pass


def cleanup_running(handle: Any) -> bool:
    """Whether the reconcile is still stopping this handle's fleet; read without a lock.

    True only while its thread is alive and within reconcile_join_s of its
    start. A reconcile alive past that is stuck, and the refusals' own words,
    which say so, stand.
    """
    thread = getattr(handle, "_supervision_reconcile_thread", None)
    started = getattr(handle, RECONCILE_STARTED, None)
    try:
        return bool(thread is not None and thread.is_alive() and isinstance(started, float)
                    and time.monotonic() - started <= reconcile_join_s())
    except Exception:
        return False


def cleanup_running_sentence() -> str:
    """The words while that stop runs. They name no cause: all three causes above start it."""
    return ("the driver is still stopping the previous Worker fleet; that stop ends within about "
            f"{lost_fleet_stop_s():.0f} s; try again when it ends")


def unresolved_refusal(handle: Any) -> str:
    """get_mesh's refusal of a predecessor whose teardown has no known outcome yet."""
    if cleanup_running(handle):
        return f"the previous fleet cannot be replaced yet: {cleanup_running_sentence()}"
    return ("the previous fleet's group or process teardown is unconfirmed "
            "(dirty, in flight, or its outcome was lost); replacement is unsafe. "
            "Reset the Attached mesh; if that reset cannot confirm the stop, restart "
            "ComfyUI only after confirming the old workers are gone")
