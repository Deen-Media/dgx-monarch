"""How the pack stops a Worker fleet under the pinned Monarch, and how long it waits.

mesh_teardown re-exports these names; the teardown paths call them from there.
tests/test_stop_bounds.py checks every sum below against the budgets that hold it.
"""
from __future__ import annotations

import asyncio
import math
import re
import sys
from typing import Any

from . import proc_stops, retired_meshes
from .error_utils import failure_summary
from .log import get_logger

log = get_logger("dgx_monarch.mesh")

# The actor stop answers in milliseconds on CPU; the proc stop keeps its own budget.
ACTOR_STOP_S = 5.0
# The group teardown's caps: a shutdown's own, and the 30 s the stops that can
# follow a Worker's death give it.
GROUP_TEARDOWN_CAP_S = 60.0
RECONCILE_GROUP_S = 30.0
# Margin over the phases; it holds the 10 s lifecycle-lock wait before them.
SHUTDOWN_MARGIN_S = 30.0
# The creation cleanup's and bring-up rollback's stop waits past Monarch's
# state query of a lost host by this much (cleanup_stop_s).
CLEANUP_STATE_MARGIN_S = 5.0
# A ProcMesh stop after its host died: under direct spawn Monarch's controller
# first waits out its state query of the dead host agent (get_proc_state_max_idle),
# then the stop's own idle limit (proc_stop_max_idle). On CPU a lost host's stop
# took 52 s where a 30 s wait timed out. Each key, its default and its variable:
_STOP_IDLE_KEYS = (("get_proc_state_max_idle", 60.0, "HYPERACTOR_MESH_GET_PROC_STATE_MAX_IDLE"),
                   ("proc_stop_max_idle", 30.0, "HYPERACTOR_MESH_PROC_STOP_MAX_IDLE"))
STOP_IDLE_MAX_S = sum(default for _key, default, _env in _STOP_IDLE_KEYS)
PROC_STOP_MARGIN_S = 10.0
PROC_STOP_DEFAULT_S = STOP_IDLE_MAX_S + PROC_STOP_MARGIN_S
# Every unit humantime accepts, case kept: "M" is a month and "m" a minute.
_UNITS = {
    **dict.fromkeys(("nanos", "nsec", "ns"), 1e-9),
    **dict.fromkeys(("usec", "us", "µs"), 1e-6),
    **dict.fromkeys(("millis", "msec", "ms"), 1e-3),
    **dict.fromkeys(("seconds", "second", "secs", "sec", "s"), 1.0),
    **dict.fromkeys(("minutes", "minute", "mins", "min", "m"), 60.0),
    **dict.fromkeys(("hours", "hour", "hrs", "hr", "h"), 3600.0),
    **dict.fromkeys(("days", "day", "d"), 86400.0),
    **dict.fromkeys(("weeks", "week", "w"), 604800.0),
    **dict.fromkeys(("months", "month", "M"), 2630016.0),
    **dict.fromkeys(("years", "year", "y"), 31557600.0),
}
_PART = re.compile(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-zµ]+)")


def duration_seconds(text: object) -> float | None:
    """A duration as Monarch reads it back ("30s", "1m", "1m 30s", "2min 5sec"), or None."""
    if not isinstance(text, str) or not text.strip():
        return None
    total, end = 0.0, 0
    for part in _PART.finditer(text):
        if part.start() != end or part.group(2) not in _UNITS:
            return None
        total, end = total + float(part.group(1)) * _UNITS[part.group(2)], part.end()
    return total if end and not text[end:].strip() and math.isfinite(total) else None


def _stop_idle_limits() -> list[tuple[str, str, object, float | None]] | None:
    """Each idle limit's key, variable, raw value and seconds, or None before Monarch loads.

    A config the runtime cannot give, or a key it does not report, reads as
    "nothing" and counts as unreadable, like a form duration_seconds cannot
    parse. The canary's lifecycle_config_defaults probe records both keys.
    """
    config = sys.modules.get("monarch.config")
    if config is None:
        return None
    try:
        values = config.get_global_config()
    except Exception:
        values = None
    if not isinstance(values, dict):
        values = {}
    return [(key, env, values.get(key, "nothing"), duration_seconds(values.get(key)))
            for key, _default, env in _STOP_IDLE_KEYS]


def proc_stop_bound_s() -> float:
    """The wait for a proc stop that can follow a Worker host's death.

    Reads both idle limits from the loaded runtime's effective config, so a
    lower override shortens the bound. It never exceeds PROC_STOP_DEFAULT_S:
    the driver refuses a higher sum, and an unreadable value, at admission
    (stop_limits_refusal). Before Monarch loads, or for a value this reader
    cannot read, the defaults apply. Never imports Monarch.
    """
    limits = _stop_idle_limits()
    if limits is None:
        return PROC_STOP_DEFAULT_S
    total = PROC_STOP_MARGIN_S
    for (_key, _env, _raw, seconds), (_k, default, _e) in zip(limits, _STOP_IDLE_KEYS, strict=True):
        total += seconds if seconds is not None and seconds >= 0 else default
    return min(total, PROC_STOP_DEFAULT_S)


def cleanup_stop_s() -> float:
    """The wait for the proc stop of a failed mesh creation's cleanup or bring-up's rollback.

    min(proc_stop_bound_s(), get_proc_state_max_idle + CLEANUP_STATE_MARGIN_S),
    65 s at Monarch's defaults. On a lone lost host a stop sent more than about
    8 s after the loss confirms when Monarch's state query of the dead host
    ends, 68.0 s after the loss at the defaults and 38.0 s at a 30 s state
    limit; one sent sooner gets Monarch's report 30 s in. So the longest wait
    a confirmation needs is the state limit, and this wait keeps every one.
    An unconfirmed cleanup poisons the transport until ComfyUI restarts. With
    the lost host beside a live one the stop never confirms, and the wait
    ends on Monarch's report or at this cap, the unknown outcome either way.
    """
    limits = _stop_idle_limits()
    state = next((seconds for key, _env, _raw, seconds in limits
                  if key == "get_proc_state_max_idle"), None) if limits else None
    if state is None or state < 0:
        state = _STOP_IDLE_KEYS[0][1]
    return min(proc_stop_bound_s(), state + CLEANUP_STATE_MARGIN_S)


def stop_limits_refusal() -> str | None:
    """Why the driver's effective stop idle limits do not fit the teardown budgets, or None.

    The remedy names only the variables set in this process's environment;
    with none set, no setting can fix it, so it asks for the pinned Monarch.
    """
    limits = _stop_idle_limits()
    if limits is None:
        return None
    import os

    from .monarch_policy import AMBIENT_TROUBLESHOOTING

    reinstall = f"reinstall the pinned torchmonarch ({AMBIENT_TROUBLESHOOTING})"
    for key, env, raw, seconds in limits:
        if seconds is None or seconds < 0:
            shown = raw if raw == "nothing" else repr(raw)
            text = f"Monarch's {key} reads {shown}, which dgx-monarch cannot read as a duration; "
            if env in os.environ:
                return (text + f"unset {env} where ComfyUI starts, or set it to a duration such as "
                        f"30s, then restart ComfyUI ({AMBIENT_TROUBLESHOOTING})")
            return text + f"{env} is not set, so {reinstall}"
    total = sum(seconds for _key, _env, _raw, seconds in limits if seconds is not None)
    if total > STOP_IDLE_MAX_S:
        named = " and ".join(f"{key} ({raw})" for key, _env, raw, _seconds in limits)
        text = (f"Monarch's {named} add up to {total:g} s, above the {STOP_IDLE_MAX_S:g} s "
                "dgx-monarch's teardown budgets hold for a proc stop after a lost host; ")
        present = [env for _key, env, _raw, _seconds in limits if env in os.environ]
        if present:
            lower = ("lower them to add up to" if len(present) > 1
                     else "lower it so the two add up to")
            return (text + f"unset {' or '.join(present)} where ComfyUI starts, or {lower} "
                    f"{STOP_IDLE_MAX_S:g} s or less, then restart ComfyUI ({AMBIENT_TROUBLESHOOTING})")
        return text + f"neither variable is set, so {reinstall}"
    return None


def shutdown_phases(timeout_s: float, stop_timeout_s: float | None = None) -> tuple[float, float]:
    """MeshHandle.shutdown's (group teardown wait, proc stop wait).

    Only the proc stop takes the bound; the group teardown keeps its own cap.
    """
    stop = timeout_s if stop_timeout_s is None else stop_timeout_s
    return min(timeout_s, GROUP_TEARDOWN_CAP_S), min(stop, proc_stop_bound_s())


def shutdown_envelope_s(timeout_s: float, stop_timeout_s: float | None = None) -> float:
    """The off-loop join MeshHandle.shutdown gives its phases."""
    group, stop = shutdown_phases(timeout_s, stop_timeout_s)
    return group + min(ACTOR_STOP_S, stop) + stop + SHUTDOWN_MARGIN_S


def reconcile_join_s() -> float:
    """How long a render thread waits for the supervision reconcile's shutdown."""
    return shutdown_envelope_s(RECONCILE_GROUP_S, proc_stop_bound_s()) + 5.0


def stop_with_reason(handle: Any, reason: str, timeout_s: float) -> None:
    """Stop the Worker actors with ``reason``, then their ProcMesh, and wait.

    Under Monarch's direct spawn, its default, only ``ActorMesh.stop`` carries the
    reason to a call in flight; ``ProcMesh.stop`` alone reaches it as "stopped:
    parent draining", which no teardown marker in mesh_safety names
    (tests/test_spawn_contract.py). The actor stop is best effort: anything it
    raises but an interrupt is logged, the proc stop below still runs, and that
    stop stays the authority. Run from the pack's atexit stop during
    interpreter exit, the actor stop raises a native panic; that is caught
    here too.
    """
    workers = getattr(handle, "workers", None)
    retired_meshes.retire(workers)  # its late faults are expected from here on
    stop = getattr(workers, "stop", None)
    if callable(stop):
        try:
            stop(reason).get(timeout=min(ACTOR_STOP_S, timeout_s))
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except BaseException as exc:
            log.info("actor stop before the proc stop (%s) did not confirm: %s",
                     reason, failure_summary(exc))
    stop_procs(handle.procs, reason, timeout_s)


def stop_procs(procs: Any, reason: str, timeout_s: float,
               thread_name: str = "dgxm-proc-stop-wait") -> Any:
    """Stop a ProcMesh and wait up to ``timeout_s``; Monarch's report on this stop ends the wait.

    With one host lost and another alive, Monarch's ProcMesh controller ends
    the stop about proc_stop_max_idle in without confirming the lost host's
    procs, and fails; the stop's Future then never completes, and the wait ran
    out its whole bound (a render 40 s after a host was lost refused at 124 s,
    against 57 s under Monarch 0.7.0). The stop is registered first
    (proc_stops), so the controller's report on this ProcMesh, and no other,
    ends the wait at once. The stop's Future cannot complete after that
    report (the controller's Stop handler returns the stop's error in both its
    branches, and the Future waits on that controller), and upstream sends the
    same report when one rank is still stopping, so a report never gives a
    definite failure: the wait raises TimeoutError, the unknown outcome the
    bound gives. Only a Future that has already finished keeps its own
    outcome. A stop not confirmed stays registered for as long as it could
    report. The Future's get runs on a daemon thread, which its own timeout
    ends; during interpreter shutdown, which starts no thread, it runs here.
    """
    import threading

    stop = proc_stops.started(procs, proc_stop_bound_s() + SHUTDOWN_MARGIN_S)
    done = threading.Event()
    outcome: dict[str, Any] = {}
    confirmed = False

    def wait(future: Any) -> None:
        try:
            outcome["value"] = future.get(timeout=timeout_s)
        except BaseException as exc:
            outcome["error"] = exc
        finally:
            done.set()
            stop.ended.set()

    try:
        future = procs.stop(reason)
        try:
            threading.Thread(target=wait, args=(future,), name=thread_name, daemon=True).start()
        except RuntimeError:  # interpreter shutdown starts no thread: wait here instead
            wait(future)
        stop.ended.wait(timeout_s + ACTOR_STOP_S)
        if done.is_set():  # read first, without waiting: a finished stop keeps its outcome
            if "error" in outcome:
                raise outcome["error"]
            confirmed = True
            return outcome.get("value")
        if stop.report is None:
            raise TimeoutError(f"proc stop did not finish within {timeout_s:g} s")
        raise TimeoutError("Monarch ended the proc stop without confirming every proc, so its "
                           f"outcome is unknown: {' '.join(stop.report.split())[:400]}")
    finally:
        proc_stops.left(stop, confirmed=confirmed)
