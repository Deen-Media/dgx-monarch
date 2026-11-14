"""Free the driver's own models before a model load that is still to come and will not fit.

On a pair under ``uly2``, p-h3-image's first use staged its text encoder and
prompt upsampler on the driver (about 25 GiB) before its 37.7 GiB file loaded,
since the first-use check defers an explicit preset's placement, and the head's
Worker then priced the stock load at 84.1 GiB against 74.0 available: the
ranks' load agreement offered the slab card and the check could not complete
(a release test, 2026-11-13). The same graph rendered when the model loaded
before those text models. Under ``auto`` the loader already charges those
models beside the load and refuses p-h3-image there, as before.

The render-site checks (``render_preflight.activation_footprint_preflight_for_request``)
run before the gate and before any dispatch, with ``driver_unload``'s rescue
behind them, but they price a pending model load as the driver footprint sees
it: under an explicit preset as placed by the loader, and under ``auto`` at the
file alone. So they found nothing to free. This step runs in the same place,
once those checks have passed, and asks what the Workers will ask. The load is
short when, first, this box's MemAvailable is below the Workers' own price with
the Workers' own inputs (``_local_fit``: ``capacity_fit.stock_load_fit``, the
file or the copy a loader cast builds, the host copy and the host floor; or
``capacity_fit.slab_load_fit`` where slab_weights is on outright, with the LoRA
stack's bake under ``lora_low_rss``), or where the local price cannot read an
input the Worker prices (an option it cannot map, or a stock load's LoRA stack
with ``lora_low_rss`` off, whose classic bake the Worker prices from its own
store); and then a Worker on
this box would stop the load at its own price (``workers_stop``: the
agreement's own read-only quote, or, where no quote can be asked, a status read
showing the ranks here hold no model, taken only where the local price is
complete and a floor of a fresh Worker's). Only then does it run the same rescue
(unload the models ComfyUI holds, then the actors' trim where the fleet is set
up) and let the render go on whatever the rescue freed.

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
- after every rank unloaded (each ``call_all("unload")`` site and a hard Clear
  VRAM) or the Workers' processes stopped (``recycle_drain.drop_residency_memos``),
  which ``note_unloaded`` marks, whatever the memos and an explicit preset's
  eager placement say, until a load succeeds (``note_placed``: the loader's
  eager placement, the no-render path's load, or a successful render). A check
  that reads the mark leaves it, since a load it lets through may still fail.
  The no-render path, which loads with no render, passes
  ``weights_loaded=False`` itself (``gate_nothing_to_gate``);
- inside the gate's capacity check, as that check's own ``weights_loaded`` says.

A render whose model a finished render, a pipelined submission or the loader
placed prices nothing here. Where the memos do not know what the Workers hold
(a retry after a failed or interrupted render, a swap for a model they hold;
a driver restart is not one, since the Workers' processes die with it), the
render pays at most the one quote,
or before setup the one status read, and only where the local price reads
short: the Worker answers a reuse, or a swap its own credit covers, as
fitting, and a rank holding a model is never read as empty. That read is
logged at DEBUG with its milliseconds, and one that fails or times out frees
nothing. A driver with a render, a result lease or a Worker call live keeps
its models (``driver_unload.busy_reason``), so it asks nothing, and so does
one with ``DGXM_DISABLE_DRIVER_UNLOAD`` set. A dual-model request,
comfy-managed residency, an FSDP preset and a driver on a box with no Worker
are left to the checks that price them today.
"""
from __future__ import annotations

import time
from typing import Any

from ..log import get_logger
from ..transfer_utils import safe_call

log = get_logger(__name__)

# Session lifetime, like the render-site memos: the checkpoints an explicit
# preset's loader left unplaced for a deferral, until it places one. A recycle
# leaves them, which errs towards a load still to come, and after a recycle
# that is true.
_DEFERRED: set[str] = set()
# Session lifetime: every rank unloaded or stopped since the last load that
# succeeded (note_placed).
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
    """A load of this checkpoint succeeded: the loader's eager placement, the
    no-render path's load, or a successful render (render_preflight)."""
    global _UNLOADED
    if isinstance(unet_name, str):
        _UNLOADED = False
        _DEFERRED.discard(unet_name)


def note_unloaded() -> None:
    """Every rank unloaded its models or stopped; the load waits for ``note_placed``."""
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


def workers_stop(mesh: Any, request: dict, unet_name: str, options: Any, *,
                 sure: bool = True) -> bool:
    """Whether a Worker on this box would stop this load at its own price.

    Where the fleet completed its distributed setup: one read-only quote, the
    one the ranks' agreement reads, carrying the request's own model spec. A
    reuse, a swap the Worker's credit covers, a rank that cannot price and a
    fleet that cannot be asked all say no. Before setup (``auto`` sets up at
    the first render) the quote cannot be asked (``mesh_setup.require_endpoint``),
    and a stopped fleet with none attached in its place has no one to ask. A
    rank that holds no model then prices no lower than the local price only
    where the caller says that price is ``sure``: every input read, and the
    smallest price any fresh Worker could choose (``no_quote_floor``). Only
    then is the answer whether every rank on this box holds none
    (``_holds_nothing``), or yes after a stop; otherwise it is no, with nothing
    read.

    It fails open: a read that raises or times out answers no, so nothing is
    freed and the render goes on. Each read is logged at DEBUG with its
    milliseconds, so a run can show what it cost.
    """
    started = time.monotonic()
    try:
        stop, read = _workers_stop(mesh, request, unet_name, options, sure)
    except Exception as exc:
        stop, read = False, f"a read that failed ({type(exc).__name__})"
    safe_call(log.debug, "pending load: %s for %s took %.0f ms and %s the load", read,
              unet_name, (time.monotonic() - started) * 1000.0,
              "would stop" if stop else "would not stop")
    return stop


def _workers_stop(mesh: Any, request: dict, unet_name: str, options: Any,
                  sure: bool = True) -> tuple[bool, str]:
    """``workers_stop``'s answer, and which read gave it."""
    from ..actor.capacity_quote import VERDICT_REFUSE, VERDICT_RESCUE
    from . import old_model_clear, recycle_drain

    handle = getattr(mesh, "handle", None)
    fleet = recycle_drain.attached_fleet(handle)
    unsure = ", with no quote to ask and a local price that is not a sure floor"
    if fleet is None:
        stopped = bool(getattr(handle, "teardown_complete", False))
        return (stopped and sure,
                "no fleet attached since the last one stopped" + ("" if sure else unsure))
    if getattr(fleet, "setup_key", None) is None:
        if not sure:
            return False, "before setup" + unsure
        return _holds_nothing(fleet), "the status read before setup"
    spec = request.get("model")
    spec = spec if isinstance(spec, dict) else {}
    asked = old_model_clear.driver_host_answers(
        mesh, unet_name, spec.get("options", options), "cond", loras=spec.get("loras"))
    if asked is None:
        return False, "the quote, with no answer from this box"
    stop = any(getattr(row, "slot", None) == "cond"
               and getattr(row, "verdict", None) in (VERDICT_REFUSE, VERDICT_RESCUE)
               for answer in asked.mine for row in getattr(answer, "quotes", ()) or ())
    return stop, "the quote"


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


def _resolve_lora(name: str) -> str:
    """This box's path for a LoRA named as its loader shows it, as the Worker
    resolves it (``actor/comfy_bridge.resolve_model_path``)."""
    import folder_paths

    found = folder_paths.get_full_path("loras", name)
    if found is None:
        raise FileNotFoundError(name)
    return str(found)


def _short(fit: Any) -> bool:
    """Whether a local price was read whole and reads short."""
    return (fit is not None and bool(fit.applies) and fit.avail_bytes is not None
            and fit.avail_bytes < fit.required_bytes)


FLOOR_SLAB = "slab"
FLOOR_STOCK = "stock"


def no_quote_floor(worker_args: Any, family: object) -> str | None:
    """The price that may stand for a fresh Worker's where no quote can be asked.

    It must be the smallest price any fresh Worker could choose for this load,
    or the step would free the driver for a load the Worker admits. So this is
    a list of what is known, not of exceptions: a fresh Worker reads its
    persisted family memo (``actor/store_family.memoized_family``), and its
    ladder (``actor/capacity_quote.price``) can choose slab by more than one
    route. Anything not in the table answers None, and the step frees nothing.

    ============  ==============  ===================================  =====
    lora_low_rss  slab_weights    family                               floor
    ============  ==============  ===================================  =====
    off           any             any                                  stock
    on            False           any                                  stock
    on            True            any, compile_dit on or off           slab
    on            auto (or unset) in ``SLAB_VOUCHED_FAMILIES``         slab
    on            auto (or unset) a known family outside that set      stock
    on            anything else   (an unknown spelling)                None
    on            auto (or unset) unknown or unread                    None
    ============  ==============  ===================================  =====

    Slab needs ``lora_low_rss``: without it the Worker turns slab off
    (``actor/comfy_bridge.py``). With slab asked for, compile_dit keeps a load
    stock only for a family compile does not leave eager, which a memo may
    name, so slab stays possible. Under auto only a vouched family's memo, or
    the retry after a first stock load that found one, reaches slab; a forced
    family never does, so a known family outside the set is stock either way.
    """
    from ..capacity_fit import SLAB_VOUCHED_FAMILIES

    args = worker_args if isinstance(worker_args, dict) else {}
    if args.get("lora_low_rss") is False:
        return FLOOR_STOCK
    mode = args.get("slab_weights", "auto")
    if mode is False:
        return FLOOR_STOCK
    if mode is True:
        return FLOOR_SLAB
    if mode != "auto" or not isinstance(family, str) or family in ("", "unknown"):
        return None
    return FLOOR_SLAB if family in SLAB_VOUCHED_FAMILIES else FLOOR_STOCK


def _local_fit(path: str, options: dict, slab: bool, lora_stack: list | None,
               reserve_bytes: int) -> Any:
    """The Worker's own price function on this box, with the Worker's inputs, or
    None where an input cannot be read.

    The options go through the Worker's own mapping
    (``actor/store_detect._to_comfy_model_options``), so a loader cast is
    priced at the copy it builds. A slab load under ``lora_low_rss`` carries
    the stack and the resolver, as the Worker passes them to ``slab_load_fit``
    for the bake's memory; a LoRA it cannot read is priced as the whole
    checkpoint there (``capacity_lora_bake.lora_bake_bytes``). The credit the
    Worker takes for a stack it already holds is left out, which only raises
    this price.
    """
    from .. import capacity_fit

    try:
        from ..actor.store_detect import _to_comfy_model_options

        comfy_options = _to_comfy_model_options(options)
    except Exception:
        return None
    if slab:
        return capacity_fit.slab_load_fit(
            path, comfy_options, reserve_bytes=reserve_bytes, lora_stack=lora_stack,
            resolve_lora_path=_resolve_lora if lora_stack else None)
    return capacity_fit.stock_load_fit(path, comfy_options)


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
    from .. import driver_footprint, driver_unload, mesh_safety, residency_mode
    from ..topology import declares_fsdp
    from . import loader_preflight

    mesh = getattr(model, "mesh", None)
    if (mesh_safety.env_enabled(driver_unload.DRIVER_UNLOAD_DISABLE_ENV)
            or request.get("uncond_model") is not None
            or declares_fsdp(str(getattr(mesh, "topology_preset", "auto") or "auto"))
            or not driver_footprint.rank0_co_resident(mesh)
            or not load_pending(model, weights_loaded)):
        return
    # The unload mark stays until a load succeeds (``note_placed``): this check
    # does not place anything, and a load it lets through may still fail.
    handle = getattr(mesh, "handle", None)
    if driver_unload.busy_reason(handle):
        return  # the rescue keeps every model while work runs: nothing to ask for
    worker_args = dict(getattr(mesh, "worker_args", None) or {})
    if handle is not None:
        from .gate_identity import resolve_effective_worker_args

        worker_args = resolve_effective_worker_args(
            model, handle, requested_worker_args=worker_args)
    if loader_preflight._effective_slab(worker_args, family) == residency_mode.MODE_COMFY_MANAGED:
        return
    # The price the quote path tests first, the larger where the Worker could
    # take either: slab only where slab_weights is on outright, compile_dit does
    # not keep it stock and lora_low_rss leaves slab on (actor/comfy_bridge.py).
    # Under auto a vouched family is priced stock on the first load of a file,
    # from a family memo this driver cannot read.
    slab = (worker_args.get("slab_weights") is True and not worker_args.get("compile_dit")
            and worker_args.get("lora_low_rss") is not False)
    options = dict(getattr(model, "options", None) or {})
    unet_name = str(getattr(model, "unet_name", "") or "")
    spec = request.get("model")
    loras = list((spec if isinstance(spec, dict) else {}).get("loras")
                 or getattr(model, "loras", None) or ())
    # The Worker's policy: lora_low_rss is on unless set off (actor/comfy_bridge.py).
    low_rss = worker_args.get("lora_low_rss") is not False
    floor = no_quote_floor(worker_args, family)
    stack = loras if low_rss and loras else None
    reserve = driver_footprint.reserve_bytes(worker_args)

    def price() -> None:
        fit = _local_fit(path, options, slab, stack, reserve)
        if fit is not None and not fit.applies:
            return  # the Worker's price makes no claim here either
        # What only the Workers can price is asked of them, never read as a fit:
        # an option this price cannot map, and a stock load's LoRA stack with
        # lora_low_rss off, whose classic bake the Worker prices from its own
        # store. Under lora_low_rss the Worker's stock price takes no LoRA term
        # (actor/in_place_fallback.route returns None), so this one is whole.
        unread = fit is None or (not slab and bool(loras) and not low_rss)
        short = fit is not None and fit.avail_bytes is not None and fit.avail_bytes < fit.required_bytes
        if not unread and not short:
            return  # the local price first, so a load that fits pays no quote
        reading = (f"the pending load of {unet_name} could not be priced here" if fit is None else
                   f"the pending {'slab' if slab else 'stock'} load of {unet_name} needs "
                   f"{fit.required_bytes / 2**30:.1f} GiB"
                   f"{' before its LoRA stack' if unread else ''} and this box has "
                   f"{(fit.avail_bytes or 0) / 2**30:.1f} GiB")
        # Where no quote can be asked, only the smallest price a fresh Worker
        # could choose may stand for it, and only read whole (``no_quote_floor``).
        sure = False
        if floor is not None and not unread:
            floor_fit = (fit if (floor == FLOOR_SLAB) == slab
                         else _local_fit(path, options, floor == FLOOR_SLAB, stack, reserve))
            sure = _short(floor_fit)
        if not workers_stop(mesh, request, unet_name, options, sure=sure):
            safe_call(log.info if short else log.debug, "pending load: %s, but no Worker on "
                      "this box would stop it, so the driver frees nothing more", reading)
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
