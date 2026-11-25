"""Drain the driver-side state a recycle invalidates: residency estimates,
the grant-audit set, and the telemetry caches; the one operator reset; and
what each announced door, through which every rank's models go, makes untrue
(``residency_doors``)."""
from __future__ import annotations

from typing import Any

from .. import residency_doors
from ..log import get_logger

log = get_logger(__name__)

# Session lifetime, emptied by each operator's drop (a Clear VRAM node at hard,
# a configuration change, the capacity rescue's slot drop): the checkpoints the
# loader node has run for, or a load has placed, since the last one (``preset_placed``).
_LOADER_RAN: set[str] = set()


def note_loader_ran(unet_name: object) -> None:
    """The loader node ran for this checkpoint, placing it or leaving it to a
    later load, or a load placed it (``pending_load_relief.note_placed`` and
    ``note_deferred``)."""
    if isinstance(unet_name, str):
        _LOADER_RAN.add(unet_name)


def note_dropped() -> None:
    """An operator dropped every rank's models: a Clear VRAM node at hard, a
    configuration change or the capacity rescue's slot drop. The load is still
    to come (``pending_load_relief.note_unloaded``), and an explicit preset's
    credit holds again only for a checkpoint whose loader node runs, or whose
    load completes, since."""
    from . import pending_load_relief

    pending_load_relief.note_unloaded()
    _LOADER_RAN.clear()


def reset_drops() -> None:
    """Forget every loader run (tests only)."""
    _LOADER_RAN.clear()


def preset_placed(model: Any) -> bool:
    """Whether the driver footprint and MiniMax H3 checks credit this render's DiT
    for its explicit preset: since the last operator's drop its loader node has
    run for the checkpoint, placing it at the node or leaving it to a ceremony
    that loads it before any render (both always credited), or a load of it has
    completed. After a hard clear,
    ComfyUI's cached loader node does not run again, so nothing places the
    model and it is charged. A recycle, a Worker restart and the gate's own
    unloads leave the credit standing, as before 1.5.
    """
    preset = str(getattr(getattr(model, "mesh", None), "topology_preset", "auto") or "auto")
    return preset != "auto" and getattr(model, "unet_name", None) in _LOADER_RAN


def _ranks_unloaded() -> None:
    from . import pending_load_relief

    pending_load_relief.note_unloaded()


def _config_replaced() -> None:
    drop_residency_memos()
    note_dropped()


residency_doors.register(residency_doors.UNLOADED, _ranks_unloaded)
residency_doors.register(residency_doors.REPLACED, _config_replaced)


def drop_residency_memos() -> None:
    """Forget every checkpoint the driver still credits to the fleet.

    Every door that stops the procs runs this, so it also answers every open
    Old model in the way card: the model it names went with the procs.
    """
    from . import loader_preflight, old_model_clear, pending_load_relief, routes

    old_model_clear.drop_cards()
    loader_preflight.reset_memos()
    # The render memos and the placements went with the procs: the next
    # render's load is still to come (nodes/pending_load_relief.note_unloaded).
    pending_load_relief.note_unloaded()
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


def _poisoned_reset(mesh_mod: Any) -> tuple[dict[str, Any], int]:
    """A Reset with no attached mesh while a failed bring-up poisoned the session.

    The Reset runs the check the next render runs (failed_bring_up.release):
    when every host shows the bring-up's processes gone it lifts the poison,
    and otherwise it says what still refuses and the restart that clears it.
    """
    from .. import failed_bring_up

    if failed_bring_up.release(mesh_mod.__dict__):
        return ({
            "ok": True,
            "status": "poison_released",
            "retryable": False,
            "detail": "the processes the failed bring-up spawned are gone on every host; "
                      "the next render attaches a fresh fleet",
        }, 200)
    poison = mesh_mod._TRANSPORT_POISON
    return ({
        "ok": False,
        "status": "poisoned",
        "retryable": False,
        "detail": (failed_bring_up.refusal(poison) if poison is not None else
                   "no attached mesh to reset; worker services are unchanged"),
    }, 200)


def recycle_attached_mesh(only: Any = None) -> tuple[dict[str, Any], int]:
    """Reset the attached mesh, as POST /dgxm/recycle does; (payload, HTTP status).

    ``only`` is the fleet a card was raised against, found among every attached
    mesh: when it is not attached any more, nothing is reset and the status
    says the fleet changed.
    """
    from .. import mesh as mesh_mod
    from ..mesh import _MESH_LOCK, _MESHES

    with _MESH_LOCK:  # snapshot under lock; recycle is idempotent anyway
        attached = list(_MESHES.values())
    if not attached and only is None and mesh_mod._TRANSPORT_POISON is not None:
        return _poisoned_reset(mesh_mod)  # the panel's Reset; a card's clear names its own fleet
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
    if only is None and mesh_mod._TRANSPORT_POISON is not None:
        return _reset_beside_poison(mesh_mod, outcome, status)
    return outcome.as_dict(), status


def _reset_beside_poison(mesh_mod: Any, outcome: Any, status: int) -> tuple[dict[str, Any], int]:
    """The panel's Reset of an attached fleet while a failed bring-up poisoned the session.

    One Reset does both, in the safe order. The poison's proof counts every
    actor process on a host, so the attached fleet's own ranks read as alive
    until that fleet stops: it is reset first, then every host is asked afresh
    (failed_bring_up.release), and the answer says what each step did.
    """
    from .. import failed_bring_up

    if not outcome.ok:
        answer = outcome.as_dict()
        answer["detail"] = (f"{outcome.detail}. The failed bring-up's poison was not checked: the "
                            "attached fleet is not stopped, and its processes keep every host busy")
        return answer, status
    reset = "the attached mesh was reset"
    if failed_bring_up.release(mesh_mod.__dict__, fresh=True) or mesh_mod._TRANSPORT_POISON is None:
        return ({
            "ok": True,
            "status": "poison_released",
            "retryable": False,
            "detail": f"{reset}, and then the processes the failed bring-up spawned were gone on "
                      "every host; the next render attaches a fresh fleet",
        }, 200)
    return ({
        "ok": False,
        "status": "poisoned",
        "retryable": False,
        "detail": f"{reset}, but {failed_bring_up.refusal(mesh_mod._TRANSPORT_POISON, mesh_mod.__dict__)}",
    }, 200)


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
