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
resolves to slab), and then a Worker on this box, asked the agreement's own
read-only quote, says its price refuses the load or offers the slab card. Only
then does it run the same rescue (unload the models ComfyUI holds, then the
actors' trim) and let the render go on whatever the rescue freed.

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

A render whose model this process placed prices nothing here. One whose model
the Workers hold from before this process started (a driver restarted beside
warm Workers) or would swap for one they hold pays at most the one quote, and
only where the local price reads short: the Worker answers a reuse, or a swap
its own credit covers, as fitting. A driver with a render, a result lease or a
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
    """Whether a Worker on this box says its own price stops this load.

    One read-only quote, the one the ranks' agreement reads, carrying the
    request's own model spec. A reuse, a swap the Worker's credit covers, a
    rank that cannot price and a fleet that cannot be asked all say no.
    """
    from ..actor.capacity_quote import VERDICT_REFUSE, VERDICT_RESCUE
    from . import old_model_clear

    spec = request.get("model")
    spec = spec if isinstance(spec, dict) else {}
    asked = old_model_clear.driver_host_answers(
        mesh, unet_name, spec.get("options", options), "cond", loras=spec.get("loras"))
    if asked is None:
        return False
    return any(getattr(row, "slot", None) == "cond"
               and getattr(row, "verdict", None) in (VERDICT_REFUSE, VERDICT_RESCUE)
               for answer in asked.mine for row in getattr(answer, "quotes", ()) or ())


def relieve(model: Any, request: dict, *, family: str, path: str,
            weights_loaded: bool | None = None) -> None:
    """Run the driver's rescue where the pending load will not fit; never raise.

    Any fault here leaves the render to the Workers' agreement, as before.
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
            safe_call(log.info, "pending load: %s, and the Workers on this box fit it, so the "
                      "driver frees nothing more", reading)
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
