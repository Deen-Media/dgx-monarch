"""Drain the driver-side state a recycle invalidates: residency estimates,
the grant-audit set, and the telemetry caches; and the one operator reset."""
from __future__ import annotations

from typing import Any

from ..log import get_logger

log = get_logger(__name__)


def drop_residency_memos() -> None:
    """Forget every checkpoint the driver still credits to the fleet.

    Every door that stops the procs runs this, so it also answers every open
    Old model in the way card: the model it names went with the procs.
    """
    from . import loader_preflight, old_model_clear, render_preflight, routes

    old_model_clear.drop_cards()
    loader_preflight.reset_memos()
    # These are the only writers of the render memos; None drains each one.
    render_preflight.note_successful_render(None)
    render_preflight.note_submitted_render(None)
    render_preflight.note_driver_charged_render(None)
    routes.invalidate_telemetry_caches()


def recycle_and_drain(handle: Any) -> Any:
    """Reset one attached mesh and drain what its exit invalidates.

    The one teardown every operator-chosen clear runs: the panel's Reset
    attached mesh, the "Old model in the way" card's accept and its Always
    flow (nodes/old_model_clear.py). It issues no clear_vram RPC first:
    ``recycle_detailed`` takes the lifecycle lock and refuses active render or
    result leases before any destructive step, and actor exit returns both
    loaded models and the retained allocator pool to the OS. Proc exit
    invalidates the residency credits, so they drain before a new fleet can
    inherit them. Returns the ``mesh_recycle.RecycleOutcome``.
    """
    outcome = handle.recycle_detailed()
    if outcome.ok:
        drop_residency_memos()
    return outcome


def _retired(handle: Any) -> bool:
    return bool(getattr(handle, "defunct", False) or getattr(handle, "teardown_complete", False))


def attached_fleet(handle: Any) -> Any:
    """The live mesh attached now for this handle's config, or None; spawns nothing.

    A MeshSpec the Init node cached keeps the handle it was built with, so
    after a reset its handle is retired while a fresh fleet on the same config
    serves the loads. The old-model card must ask, and bind to, that fleet.
    """
    if handle is None or not _retired(handle):
        return handle
    from ..mesh import _MESH_LOCK, _MESHES

    config = getattr(handle, "config", None)
    with _MESH_LOCK:
        attached = list(_MESHES.values())
    for each in attached:
        other = getattr(each, "config", None)
        if (not _retired(each) and getattr(other, "source", None) == getattr(config, "source", None)
                and bool(getattr(other, "hosts", ())) == bool(getattr(config, "hosts", ()))):
            return each
    return None


def recycle_attached_mesh(only: Any = None) -> tuple[dict[str, Any], int]:
    """Reset the attached mesh, as POST /dgxm/recycle does; (payload, HTTP status).

    ``only`` is the fleet a card was raised against, found among every attached
    mesh: when it is not attached any more, nothing is reset and the status
    says the fleet changed.
    """
    from ..mesh import _MESH_LOCK, _MESHES

    with _MESH_LOCK:  # snapshot under lock; recycle is idempotent anyway
        attached = list(_MESHES.values())
    if not attached:
        return ({
            "ok": False,
            "status": "no_live_mesh",
            "retryable": False,
            "detail": "no attached mesh to reset; worker services are unchanged",
        }, 200)
    handle = attached[0]
    if only is not None:
        if not any(each is only for each in attached) or _retired(only):
            return ({
                "ok": False,
                "status": "fleet_changed",
                "retryable": False,
                "detail": "the mesh the card was raised against is no longer attached; "
                          "nothing was reset",
            }, 200)
        handle = only
    outcome = recycle_and_drain(handle)
    status = 200 if outcome.ok else (409 if outcome.retryable else 503)
    return outcome.as_dict(), status


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
