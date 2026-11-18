"""Which late Monarch faults the fault hook logs at INFO instead of as an ERROR.

Under the pinned Monarch a heal or a lost host writes several "cluster fault"
ERROR lines that Monarch 0.7.0 did not: a MeshFailure reading "supervision
liveness failed" for the mesh the driver is retiring, about 30 s after the
death, and one per rank for the two meshes Monarch spawns on every proc a
spawn with a bootstrap makes, ``log_level-<uid>`` and ``setup-<uid>``. The
fault hook (mesh_helpers.install_unhandled_fault_hook) logs a fault at INFO
with no toast only when the fault's own ``mesh_name`` (the MeshFailure
property; its text's ``MeshFailure(mesh_name=...,`` when that cannot be read)
is a name recorded here
when the pack began to stop that mesh, or is one of those internal meshes and
the fault names its actor class. Any other name, an unrecorded one, or a text
it cannot read stays an ERROR with a toast, so a live Worker mesh's failure,
which comes with them, still shows.

Names are recorded at bring-up (mesh_factory) and moved to the retired set by
mesh_stop.stop_with_reason, so the hook thread never reads a mesh name itself.

A ProcMesh controller's "error stopping proc mesh" report on one of the
pack's own proc stops, a registered stop whose label it names
(proc_stops.claim), also logs at INFO: the stop path logs the outcome, and a
report that arrives after the stop's waiter gave up, as after a heal's 2 s
stop, changes nothing. One that names no registered stop stays an ERROR.
Standard library only.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import Any

from . import proc_stops

# Monarch's own meshes: name prefix, and the actor class their faults name.
INTERNAL_MESHES = (("log_level-", "monarch._src.actor.logging._LogLevelSetter"),
                   ("setup-", "monarch._src.actor.proc_mesh.SetupActor"))
RETIRED_NOTICE = "cluster teardown notice (deliberate)"
INTERNAL_NOTICE = "Monarch internal mesh notice (it follows a Worker or host fault)"
STOP_NOTICE = "cluster teardown notice (a proc stop of the pack ended without confirming every proc)"
_MESH_NAME = re.compile(r"MeshFailure\(mesh_name=([^,\s)]+),")
_KEEP = 64
_LOCK = threading.Lock()
_LIVE: OrderedDict[int, str] = OrderedDict()  # id(actor mesh) -> its name
_RETIRED: OrderedDict[str, None] = OrderedDict()


def remember(workers: Any, name: object) -> None:
    """Record a live Worker mesh's name, read off the hook thread at bring-up."""
    if workers is None or not isinstance(name, str) or not name:
        return
    with _LOCK:
        _LIVE[id(workers)] = name
        while len(_LIVE) > _KEEP:
            _LIVE.popitem(last=False)


def retire(workers: Any) -> None:
    """The pack is stopping this mesh: its faults from now on are expected."""
    with _LOCK:
        name = _LIVE.pop(id(workers), None) if workers is not None else None
        if name is None:
            return
        _RETIRED[name] = None
        _RETIRED.move_to_end(name)
        while len(_RETIRED) > _KEEP:
            _RETIRED.popitem(last=False)


def mesh_name(text: object, failure: object = None) -> str | None:
    """The fault's mesh name: its ``MeshFailure.mesh_name``, else read from its text."""
    try:
        name = getattr(failure, "mesh_name", None)
    except Exception:
        name = None
    if isinstance(name, str) and name:
        return name
    match = _MESH_NAME.match(text) if isinstance(text, str) else None
    return match.group(1) if match else None


def notice(text: object, failure: object = None) -> str | None:
    """The INFO label for a fault the hook keeps quiet, or None to report it."""
    if proc_stops.is_report(text):
        return STOP_NOTICE if proc_stops.claim(str(text)) else None
    name = mesh_name(text, failure)
    if name is None or not isinstance(text, str):
        return None
    if any(name.startswith(prefix) and actor_class in text for prefix, actor_class in INTERNAL_MESHES):
        return INTERNAL_NOTICE
    with _LOCK:
        return RETIRED_NOTICE if name in _RETIRED else None


def quiet(text: object) -> bool:
    """Whether the hook keeps this fault quiet."""
    return notice(text) is not None


def reset() -> None:
    """Forget every record (tests)."""
    with _LOCK:
        _LIVE.clear()
        _RETIRED.clear()
    proc_stops.reset()
