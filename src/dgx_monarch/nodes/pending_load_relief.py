"""Free the driver's own models before a model load that is still to come and will not fit.

On a pair, p-h3-image's first use staged its text encoder and prompt upsampler
on the driver (about 25 GiB) before its 37.7 GiB file loaded, and the head's
Worker then priced the stock load at 84.1 GiB against 74.0 available: the
ranks' load agreement offered the slab card and the first-use check could not
complete (a release test, 2026-11-13). The same graph rendered when the model
loaded before those text models.

The render-site checks (``render_preflight.activation_footprint_preflight_for_request``)
run before the gate and before any dispatch, with ``driver_unload``'s rescue
behind them, but they price a pending model load as the driver footprint sees
it: under an explicit preset as placed by the loader, and under ``auto`` at the
file alone. So they found nothing to free. This step runs in the same place,
once those checks have passed, and asks what the Workers will ask. The load is
short when, first, this box's MemAvailable is below the Workers' own price
(``capacity_fit.stock_load_fit``: the file, the host copy it is placed from and
the host floor, or ``capacity_fit.slab_load_fit`` where the family's residency
resolves to slab), and then a Worker on this box would stop the load at its
own price (``workers_stop``: the agreement's own read-only quote, or before
distributed setup a status read showing the ranks here hold no model). Only
then does it run the same rescue (unload the models ComfyUI holds, then the
actors' trim where the fleet is set up) and let the render go on whatever the
rescue freed.

It refuses nothing: the ranks' load agreement and each Worker's own ladder stay
the judges of the load, with the slab card or the refusal they raise today. It
runs after the checks that refuse, so a render they refuse is refused as today,
in the same words, and never reaches it.

The load is still to come (``load_pending``):

- under ``auto``, unless a finished render or a pipelined submission placed
  this checkpoint (``render_preflight``'s memos);
- under an explicit preset, only where the loader deferred its placement
  (``note_deferred``, at the identity gate's or the dual-model deferral), since
  otherwise the loader placed it at its node;
- after every rank unloaded for an identity ceremony's clean reload
  (``note_unloaded``), whatever the memos say, until the next render's own
  check or a loader placement: the memos still name a model the Workers no
  longer hold. The no-render path, which loads with no render, passes
  ``weights_loaded=False`` itself (``gate_nothing_to_gate``);
- inside the gate's capacity check, as that check's own ``weights_loaded`` says.

A render whose model a finished render, a pipelined submission or the loader
placed prices nothing here. Where the memos do not know what the Workers hold
(a retry after a failed or interrupted render, a driver restarted beside warm
Workers, a swap for a model they hold), the render pays at most the one quote,
or before setup the one status read, and only where the local price reads
short: the Worker answers a reuse, or a swap its own credit covers, as
fitting, and a rank holding a model is never read as empty. A driver with a render, a result lease or a
Worker call live keeps its models (``driver_unload.busy_reason``), so it asks
nothing. A dual-model request, comfy-managed residency, an FSDP preset and a
driver on a box with no Worker are left to the checks that price them today.
"""
from __future__ import annotations

from typing import Any

from ..log import get_logger
from ..transfer_utils import safe_call

log = get_logger(__name__)

# Session lifetime, like the render-site memos: the checkpoints an explicit
# preset's loader left unplaced for a deferral, until it places one. A recycle
# leaves them, which errs towards a load still to come, and after a recycle
# that is true.
_DEFERRED: set[str] = set()
# Every rank unloaded since the last render's check or loader placement.
_UNLOADED = False
# A status read before setup, on the short path only (nodes/routes.py reads it at 10 s too).
_STATUS_TIMEOUT_S = 10.0


class PendingLoadShort(Exception):
    """The pending load does not fit this box now. A rescue trigger, never a refusal."""


def note_deferred(unet_name: object) -> None:
    """The loader left this checkpoint's placement to a later load (nodes/loaders.py)."""
    if isinstance(unet_name, str):
        _DEFERRED.add(unet_name)


def note_placed(unet_name: object) -> None:
    """The loader placed this checkpoint at its node."""
    global _UNLOADED
    _UNLOADED = False
    if isinstance(unet_name, str):
        _DEFERRED.discard(unet_name)


def note_unloaded() -> None:
    """Every rank just unloaded its models for a clean reload (nodes/gate_ceremony.py)."""
    global _UNLOADED
    _UNLOADED = True


def reset() -> None:
    """Forget every deferral and unload (tests only)."""
    global _UNLOADED
    _UNLOADED = False
    _DEFERRED.clear()


def load_pending(model: Any, weights_loaded: bool | None) -> bool:
    """Whether the render's model load is still to come, as described above."""
    from . import render_preflight

    if weights_loaded is not None:
        return not weights_loaded
    if _UNLOADED:
        return True
    unet_name = getattr(model, "unet_name", None)
    if unet_name in (render_preflight._LAST_RENDERED_UNET, render_preflight._LAST_SUBMITTED_UNET):
        return False
    preset = str(getattr(getattr(model, "mesh", None), "topology_preset", "auto") or "auto")
    return preset == "auto" or unet_name in _DEFERRED


def workers_stop(mesh: Any, request: dict, unet_name: str, options: Any) -> bool:
    """Whether a Worker on this box would stop this load at its own price.

    Where the fleet completed its distributed setup: one read-only quote, the
    one the ranks' agreement reads, carrying the request's own model spec. A
    reuse, a swap the Worker's credit covers, a rank that cannot price and a
    fleet that cannot be asked all say no. Before setup (``auto`` sets up at
    the first render) the quote cannot be asked (``mesh_setup.require_endpoint``):
    then a rank that holds no model at all prices exactly as the local price
    does, so the answer is whether every rank on this box holds none
    (``_holds_nothing``). A stopped fleet with none attached in its place left
    no process behind, so the next one attaches holding nothing.
    """
    from ..actor.capacity_quote import VERDICT_REFUSE, VERDICT_RESCUE
    from . import old_model_clear, recycle_drain

    handle = getattr(mesh, "handle", None)
    fleet = recycle_drain.attached_fleet(handle)
    if fleet is None:
        return bool(getattr(handle, "teardown_complete", False))
    if getattr(fleet, "setup_key", None) is None:
        return _holds_nothing(fleet)
    spec = request.get("model")
    spec = spec if isinstance(spec, dict) else {}
    asked = old_model_clear.driver_host_answers(
        mesh, unet_name, spec.get("options", options), "cond", loras=spec.get("loras"))
    if asked is None:
        return False
    return any(getattr(row, "slot", None) == "cond"
               and getattr(row, "verdict", None) in (VERDICT_REFUSE, VERDICT_RESCUE)
               for answer in asked.mine for row in getattr(answer, "quotes", ()) or ())


def _holds_nothing(fleet: Any) -> bool:
    """Whether every rank on this box reports no model in either slot.

    A status read needs no distributed setup. Any other answer, a retained
    slab, or a row that cannot be read says no.
    """
    try:
        rows = list(fleet.call_all("status", timeout_s=_STATUS_TIMEOUT_S) or ())
        hosts = tuple(getattr(getattr(fleet, "config", None), "hosts", None) or ())
        local = getattr(fleet, "owns_hosts", True) is False or not hosts
        gpus = int(getattr(fleet, "gpus_per_host", 1) or 1)
        mine = [row for row in rows if local or int(row["rank"]) // gpus == 0]
        return bool(mine) and all(
            row["models"]["cond"] is None and row["models"]["uncond"] is None
            and not row["models"].get("retained_failed_load_slabs") for row in mine)
    except Exception:
        return False


def relieve(model: Any, request: dict, *, family: str, path: str,
            weights_loaded: bool | None = None) -> None:
    """Run the driver's rescue where the pending load will not fit; never raise.

    Any fault here leaves the render to the ranks' agreement, as before.
    """
    try:
        _relieve(model, request, family=family, path=path, weights_loaded=weights_loaded)
    except Exception as exc:
        safe_call(log.debug, "pending load relief skipped (%r)", exc)


def _relieve(model: Any, request: dict, *, family: str, path: str,
             weights_loaded: bool | None) -> None:
    global _UNLOADED
    from .. import capacity_fit, driver_footprint, driver_unload, residency_mode
    from ..topology import declares_fsdp
    from . import loader_preflight

    mesh = getattr(model, "mesh", None)
    if (request.get("uncond_model") is not None
            or declares_fsdp(str(getattr(mesh, "topology_preset", "auto") or "auto"))
            or not driver_footprint.rank0_co_resident(mesh)
            or not load_pending(model, weights_loaded)):
        return
    if weights_loaded is None:
        _UNLOADED = False  # this render makes the load the unload left to come
    handle = getattr(mesh, "handle", None)
    if driver_unload.busy_reason(handle):
        return  # the rescue keeps every model while work runs: nothing to ask for
    worker_args = dict(getattr(mesh, "worker_args", None) or {})
    if handle is not None:
        from .gate_identity import resolve_effective_worker_args

        worker_args = resolve_effective_worker_args(
            model, handle, requested_worker_args=worker_args)
    slab = loader_preflight._effective_slab(worker_args, family)
    if slab == residency_mode.MODE_COMFY_MANAGED:
        return
    options = dict(getattr(model, "options", None) or {})
    unet_name = str(getattr(model, "unet_name", "") or "")

    def price() -> None:
        fit: Any = (capacity_fit.slab_load_fit(
            path, options, reserve_bytes=driver_footprint.reserve_bytes(worker_args))
            if slab is True else capacity_fit.stock_load_fit(path, options))
        room = fit.avail_bytes
        # The local price first, so a load that fits pays no quote.
        if not fit.applies or room is None or room >= fit.required_bytes:
            return
        reading = (f"the pending {'slab' if slab is True else 'stock'} load of {unet_name} "
                   f"needs {fit.required_bytes / 2**30:.1f} GiB and this box has "
                   f"{room / 2**30:.1f} GiB")
        if not workers_stop(mesh, request, unet_name, options):
            safe_call(log.info, "pending load: %s, but no Worker on this box would stop it, so "
                      "the driver frees nothing more", reading)
            return
        raise PendingLoadShort(reading)

    try:
        # No swap: the Worker's answer above already counts what it drops.
        driver_unload.check_with_unload(
            [price], handle=handle, subject=f"the pending load of {unet_name}",
            is_refusal=lambda exc: isinstance(exc, PendingLoadShort))
    except Exception as short:
        # driver_unload hands back the trigger with what it tried added, as a
        # rewrapped StockLoadCapacityError whose cause is the trigger.
        if not isinstance(short if isinstance(short, PendingLoadShort) else short.__cause__,
                          PendingLoadShort):
            raise
        # The rescue ran out of things to free, or a render kept them: the
        # Workers' agreement decides the load, with its card or refusal.
        safe_call(log.info, "pending load: %s; the Workers' load agreement decides it", short)
