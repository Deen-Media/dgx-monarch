"""What the driver host gets back before a load, tried before the loader refuses.

The loader preflight (``nodes/loader_preflight.py``) prices a DiT against the
driver host's MemAvailable as it reads now. When that price does not fit, this
module asks what the box will have at the moment of the load, before any card
is raised:

1. **The swap credit.** The Workers on the driver host answer the agreement's
   own quote, and where the load replaces a stock resident in its own slot the
   store drops that model first, so its copy counts as available: only what
   the Worker's own price credited, and only where this price charges the
   DiT's load (``swap_credit.py`` holds the rule). A deferred load must also leave room
   for the graph's driver models beside the old model, because they load on
   the driver before the first render drops it.
2. **ComfyUI's own models on this host.** If the credit does not make room,
   the driver unloads what ComfyUI holds here, as the render-site checks do
   before they refuse (``driver_unload.py``): ComfyUI reloads each on demand.
   Never while a render is running in this driver, and never twice for one
   load. The loader then prices the whole load again from a fresh reading,
   with the graph's driver models charged whole, since none of them is held
   any more. The loader's price already counts held models against this
   graph's own (``loader_graph.credited_terms``), so the unload makes room
   only by what ComfyUI held beyond them: another graph's larger encoder.

Every step is read-only or ComfyUI's own unload, and none clears a Worker:
the Worker drops the model it replaces as it always did. A fault anywhere means
no relief and the loader's own refusal. The refusal that still stands says
what was counted, and the Old model in the way card is weighed against the
uncredited price, because a cleared box holds no copy to drop
(``old_model_clear.loader_offer``).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from .. import driver_unload, mesh_safety, old_model, swap_credit
from ..log import get_logger
from . import graph_load_price, loader_graph, old_model_clear

log = get_logger(__name__)

_GIB = 2 ** 30
UNASKED = old_model_clear.UNASKED


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


@dataclass(frozen=True, slots=True)
class Relief:
    """What was counted for one load, and whether it fits with it."""

    fits: bool = False
    credit: swap_credit.SwapCredit = swap_credit.NONE
    # The driver host's Worker answers, for the card: ``UNASKED`` when nobody asked.
    asked: Any = UNASKED
    # After the credit: what the worst window is still short by, and what the
    # graph's driver models are short by beside the old model, before the drop.
    short_bytes: int = 0
    beside_short_bytes: int = 0
    box: str = ""
    # ComfyUI's models were unloaded just now: the loader prices the load again.
    unloaded: driver_unload.Unloaded | None = None
    # Why the driver kept ComfyUI's models although the load does not fit.
    kept: str = ""

    def sentence(self, unet_name: str, unloaded: driver_unload.Unloaded | None = None) -> str:
        """What a refusal that still stands says was counted and freed, with a leading space."""
        parts = []
        if self.credit:
            old = old_model.display_name(self.credit.dropped) or "the old model"
            new = old_model.display_name(unet_name)
            freed = (f"The Worker on {self.box} drops {old} before this load, which frees at "
                     f"least {self.credit.gib} GiB")
            parts.append(
                f"{freed}, and {new} is still {_gib(self.short_bytes)} GiB short after that."
                if self.short_bytes else
                f"{freed}, and that would make room for {new}. But this graph's driver models "
                f"load first, beside {old}, and they are {_gib(self.beside_short_bytes)} GiB short.")
        if unloaded is not None:
            parts.append(unloaded.refusal_sentence())
        elif self.kept:
            parts.append(self.kept)
        return "".join(f" {part}" for part in parts)


NONE = Relief()


def _unload_driver_models(mesh: Any) -> tuple[driver_unload.Unloaded | None, str]:
    """Unload what ComfyUI holds on this host, or say why the driver kept it."""
    if mesh_safety.env_enabled(driver_unload.DRIVER_UNLOAD_DISABLE_ENV):
        return None, ""
    mm = driver_unload.comfy_model_management()
    held = driver_unload.held_models(mm) if mm is not None else []
    if not held:
        return None, ""
    busy = driver_unload.busy_reason(getattr(mesh, "handle", None))
    if busy:
        count = len(held)
        return None, (f"The driver did not unload the {count} model{'' if count == 1 else 's'} "
                      f"ComfyUI holds on this host before refusing, because {busy}; queue again "
                      "once that work ends.")
    outcome = driver_unload.unload(mm, held)
    if not outcome.names:
        return None, ""
    # Nothing ComfyUI held is loaded now, so no earlier charge stands for it.
    loader_graph.reset_charged_artifacts()
    return outcome, ""


def relieve(shape: Any, windows: Callable[[Any], tuple[Any, ...]], *, mesh: Any,
            unet_name: str, options: Any, slot: str,
            tried: driver_unload.Unloaded | None = None) -> Relief:
    """Count, then free, what the box gets back before this load; never raises.

    ``shape`` is the loader's own (``loader_preflight._Shape``) and ``windows``
    prices one. ``slot`` is the slot the load goes into; with none known the
    Workers are not asked and nothing is credited. ``tried`` is the unload an
    earlier pass of this load made, so it is not made again. A result with
    ``unloaded`` set tells the loader to price the load again from the start.
    """
    relief = NONE
    try:
        # A price that reads the weights as resident (the loader's memo, or a
        # first Worker on another box) charges no load, so it has no drop to count.
        pending = slot and any(window.weight_bytes for window in windows(shape))
        asked = (old_model_clear.driver_host_answers(mesh, unet_name, options, slot)
                 if pending else UNASKED)
        box = graph_load_price.driver_box(mesh)
        credit = (swap_credit.from_answers(asked.mine, slot, box=box)
                  if pending and asked is not None else swap_credit.NONE)
        relief = Relief(asked=asked, box=box)
        if credit:
            priced = windows(replace(shape, mem_available=shape.mem_available + credit.bytes))
            short = max((window.shortfall for window in priced
                         if not graph_load_price.fits_box(window)), default=0)
            # A deferred load: the driver builds this graph's models before the
            # first render, while the Worker still holds the old model.
            beside = max(shape.floor_reserve + shape.load_stack_bytes - shape.mem_available, 0)
            relief = Relief(fits=not short and not beside, credit=credit, asked=asked,
                            short_bytes=short, beside_short_bytes=beside, box=box)
        if relief.fits or tried is not None:
            return relief
        unloaded, kept = _unload_driver_models(mesh)
        return replace(relief, unloaded=unloaded, kept=kept)
    except Exception as exc:
        log.info("loader preflight could not count what the driver host gets back before "
                 "this load (%r); its own price stands", exc)
        return relief


def admitted(relief: Relief, cleared: dict[str, None], unet_name: str, mem_available: int,
             unloaded: driver_unload.Unloaded | None = None) -> None:
    """Record a load the relief let through: a journal line for an unload that
    made the room, and for a credit one more, with the dropped model's "already
    loaded" memo forgotten, since its Worker no longer holds it."""
    if unloaded is not None:
        log.info(unloaded.log_line(unet_name))
    if not relief.fits:
        return
    dropped = relief.credit.dropped
    with loader_graph._MEMO_LOCK:
        for key in [key for key in cleared if dropped and key.split("\x00", 1)[0] == dropped]:
            cleared.pop(key, None)
    log.info(
        "loader footprint preflight: %s fits on %s once the Worker there drops %s, which "
        "frees at least %s GiB; MemAvailable read %s GiB",
        unet_name, relief.box, old_model.display_name(dropped) or "the old model",
        relief.credit.gib, _gib(mem_available))


def render_credit(model: Any, request: Any) -> swap_credit.SwapCredit:
    """The swap credit for a render whose DiT load is still pending; never raises.

    The render-site checks that charge a pending load read it through
    ``driver_unload.check_with_unload``. The question carries the request's own
    model spec, LoRA stack included, so the Worker's answer is the one its
    sample-time agreement will give.
    """
    try:
        mesh = getattr(model, "mesh", None)
        spec = request.get("model") if isinstance(request, dict) else None
        if mesh is None or not isinstance(spec, dict) or not spec.get("unet_name"):
            return swap_credit.NONE
        name = str(spec["unet_name"])
        asked = old_model_clear.driver_host_answers(
            mesh, name, spec.get("options"), "cond", loras=spec.get("loras"))
        if asked is None:
            return swap_credit.NONE
        return swap_credit.from_answers(asked.mine, "cond", box=graph_load_price.driver_box(mesh))
    except Exception as exc:
        log.info("render preflight could not count what the driver host gets back before "
                 "this load (%r); its own price stands", exc)
        return swap_credit.NONE
