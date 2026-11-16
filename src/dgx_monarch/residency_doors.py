"""What the mesh layer tells the driver's residency memos when the Workers' models go.

The render-site checks credit a model as already in MemAvailable from memos
the driver keeps (``nodes/pending_load_relief``, ``nodes/render_preflight``),
and every path that drops every rank's models must make those memos true
again. Some of those paths live in the mesh layer, which may not import
``nodes/`` (``tests/test_mesh_layer_imports.py``): a fleet a configuration
change replaces, a fleet marked defunct and the new fleet in place of a
retired one, a process-group teardown and a worker-policy change that unloads
every resident. Each
announces its door here, and ``nodes/recycle_drain`` registers, at import,
what each door drains. A door with no handler drains nothing, and a handler
that fails is logged, so a mesh path never fails on an announcement.
"""
from __future__ import annotations

from collections.abc import Callable

from .log import get_logger

log = get_logger(__name__)

# Every rank's models went: the render memos and the unload mark.
UNLOADED = "unloaded"
# A configuration change replaced the fleet: an operator's drop and the full drain.
REPLACED = "replaced"

# Read-only after import: nodes/recycle_drain registers one handler per door.
_HANDLERS: dict[str, Callable[[], None]] = {}


def register(door: str, handler: Callable[[], None]) -> None:
    """Name what ``door`` drains (``nodes/recycle_drain``, at import)."""
    _HANDLERS[door] = handler


def announce(door: str) -> None:
    """Drain what ``door`` makes untrue; never raise."""
    handler = _HANDLERS.get(door)
    if handler is None:
        return
    try:
        handler()
    except Exception as exc:
        log.warning("the driver could not drain its residency memos at the %s door (%r); "
                    "the next render may credit a model the Workers no longer hold", door, exc)
