"""Which of the pack's proc stops a Monarch controller report belongs to.

Under the pinned Monarch, a ProcMesh controller that cannot confirm every
proc stopped (a rank on a lost host beside a live one times out after
proc_stop_max_idle, or a rank is still stopping when that wait ends) fails
with "error stopping proc mesh", and the stop's Future never completes. The
report reaches the fault hook as a MeshFailure naming the controller actor,
``proc_mesh_controller_<ProcMesh id>`` sanitized to a lowercase label, and the
ProcMesh id is the name the mesh was spawned under plus a uid.

So the pack spawns every ProcMesh under a fresh name (fresh_name) and records
it (remember), and registers every proc stop it issues (started) with the
label prefix its reports must carry. A report goes to the registered stop
whose prefix it carries and to no other: it ends that stop's wait while the
wait runs (mesh_stop.stop_procs), and once the waiter has given up it is
still recognised, so kept quiet, for as long as the stop could report. A
report that matches no registered stop changes no wait and stays an error.
tests/test_worker_proc_id_integration.py proves the label on the real runtime.
Standard library only.
"""
from __future__ import annotations

import re
import secrets
import string
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

NAME_PREFIX = "dgxm-procs-"
CONTROLLER = "proc_mesh_controller_"
REPORT_MARKERS = ("actor proc_mesh_controller_", "error stopping proc mesh")
_LABEL = re.compile(r"actor (proc_mesh_controller_[a-z0-9_-]+)<")
_KEEP = 64
_LOCK = threading.Lock()
_NAMES: OrderedDict[int, str] = OrderedDict()  # id(ProcMesh) -> the name it was spawned under


@dataclass(eq=False)
class ProcStop:
    """One proc stop the pack issued, and Monarch's report on it if one came."""

    prefix: str | None
    keep_until: float
    ended: threading.Event = field(default_factory=threading.Event)
    report: str | None = None
    waiting: bool = True
    # The stop's call failed or its Future answered, so no report of the
    # controller's is this stop's: a failing controller never answers the Future.
    answered: bool = False


_STOPS: list[ProcStop] = []


def fresh_name() -> str:
    """A name for a ProcMesh the pack spawns: lowercase, so Monarch's label keeps it whole.

    Letters only, so it never reads as a host name with a hexadecimal suffix.
    """
    return NAME_PREFIX + "".join(secrets.choice(string.ascii_lowercase) for _ in range(10))


def label_prefix(name: str) -> str:
    """The start of the controller label for a ProcMesh spawned under ``name``."""
    return f"{CONTROLLER}{name}-"


def remember(procs: Any, name: str) -> None:
    """Record the name a ProcMesh was spawned under."""
    if procs is None or not name:
        return
    with _LOCK:
        _NAMES[id(procs)] = name
        _NAMES.move_to_end(id(procs))
        while len(_NAMES) > _KEEP:
            _NAMES.popitem(last=False)


def name_of(procs: Any) -> str | None:
    with _LOCK:
        return _NAMES.get(id(procs)) if procs is not None else None


def _prune(now: float) -> None:
    _STOPS[:] = [stop for stop in _STOPS if stop.waiting or stop.keep_until > now]


def started(procs: Any, keep_s: float) -> ProcStop:
    """Register a proc stop about to be issued; a report may name it for ``keep_s``."""
    name = name_of(procs)
    now = time.monotonic()
    stop = ProcStop(label_prefix(name) if name else None, now + keep_s)
    with _LOCK:
        _prune(now)
        _STOPS.append(stop)
    return stop


def left(stop: ProcStop, *, confirmed: bool, answered: bool = False) -> None:
    """The waiter is done: a confirmed stop is forgotten, any other kept until it cannot report."""
    with _LOCK:
        stop.waiting = False
        stop.answered = answered
        if confirmed and stop in _STOPS:
            _STOPS.remove(stop)
        _prune(time.monotonic())


def is_report(text: object) -> bool:
    return isinstance(text, str) and all(marker in text for marker in REPORT_MARKERS)


def claim(text: str) -> bool:
    """Hand a controller report to the registered stop it names; False when it names none.

    A retried stop of one ProcMesh is registered beside the first, under the
    same label. A retry follows only a stop that failed with an answer (a
    timeout is never retried, mesh_teardown.stop_failure_is_retryable), and a
    controller that fails never answers the stop's Future, so no report is an
    answered stop's. A report goes to the oldest stop with neither an answer
    nor a report yet: the retry's own report ends the retry, and one stop's
    report never ends another's wait. When no match is left, the newest takes
    it, so it stays recognised and quiet.
    """
    match = _LABEL.search(text)
    label = match.group(1) if match else None
    if not label:
        return False
    with _LOCK:
        _prune(time.monotonic())
        named = [stop for stop in _STOPS if stop.prefix and label.startswith(stop.prefix)]
        if not named:
            return False
        stop = next((stop for stop in named if stop.report is None and not stop.answered),
                    named[-1])
        stop.report = text
        if stop.waiting:
            stop.ended.set()
        return True


def reset() -> None:
    """Forget every record (tests)."""
    with _LOCK:
        _NAMES.clear()
        _STOPS.clear()
