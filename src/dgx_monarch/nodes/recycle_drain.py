"""Drain the driver-side state a recycle invalidates: residency estimates,
the grant-audit set, and the telemetry caches."""
from __future__ import annotations

from typing import Any

from ..log import get_logger

log = get_logger(__name__)


def drop_residency_memos() -> None:
    """Forget every checkpoint the driver still credits to the fleet."""
    from . import loader_preflight, render_preflight, routes

    loader_preflight.reset_memos()
    # These are the only writers of the render memos; None drains each one.
    render_preflight.note_successful_render(None)
    render_preflight.note_submitted_render(None)
    render_preflight.note_driver_charged_render(None)
    routes.invalidate_telemetry_caches()


def evict_render_fleet(handle: Any, exc: Exception, *,
                       holding_lock: bool = False) -> None:
    """Evict a failed fleet and clear memory credits only if it was retired.

    A replacement fleet holds no previous weights, so loader and render residency
    memos must be cleared after proc retirement. A typed worker refusal leaves the
    fleet live; clearing credits then could double-charge resident weights and
    cause a false capacity refusal.

    Inspect the handle used for dispatch, not the graph's cached MeshSpec, which
    may still point to an older retired fleet.
    """
    from ..mesh_helpers import mark_defunct_preserving_primary

    mark_defunct_preserving_primary(handle, exc, holding_lock=holding_lock)
    if getattr(handle, "defunct", False) is not True:
        return
    # The mark compensates its own failures; a broken drain must not reshape
    # the render failure this runs inside. A second drain is harmless.
    try:
        drop_residency_memos()
    except Exception as drain_exc:
        log.warning("residency memo drain after a fleet eviction failed: %r",
                    drain_exc)
