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
   the driver before the first render drops it. In a dual-model graph the
   credit leaves the room the other slot's load takes (``other_slot_loads``).
   ``DGXM_DISABLE_SWAP_CREDIT=1`` turns this step off; the unload's switch
   does not, since the credit is a price and not a rescue.
2. **ComfyUI's own models on this host.** If the credit does not make room,
   the driver unloads what ComfyUI holds here, as the render-site checks do
   before they refuse (``driver_unload.py``): ComfyUI reloads each on demand.
   Never while a render is running in this driver, and never twice for one
   load. The loader then prices the whole load again from a fresh reading,
   with the graph's driver models charged whole, since none of them is held
   any more. The loader's price already counts held models against this
   graph's own (``loader_graph.credited_terms``), so the unload makes room
   only by what ComfyUI held beyond them: another graph's larger encoder.
   Where ComfyUI holds no model the same step still returns the allocator's
   cached blocks, and the load is priced again if the reading rose: the
   driver's ComfyUI keeps that cache for longer since its free-memory reading
   follows MemAvailable (``driver_uma_memory.py``), and on the driver host it
   counts against the Worker's load.

Every step is read-only or ComfyUI's own unload, and none clears a Worker:
the Worker drops the model it replaces as it always did. A fault anywhere means
no relief and the loader's own refusal. The refusal that still stands says
what was counted, and the Old model in the way card is weighed against the
uncredited price, because a cleared box holds no copy to drop
(``old_model_clear.loader_offer``).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .. import driver_unload, mesh_safety, old_model, swap_credit
from ..log import get_logger
from . import graph_load_price, loader_graph, old_model_clear
from .graph_live import live_nodes

log = get_logger(__name__)

_GIB = 2 ** 30
# A released allocator cache is priced again only where it moves the reading
# a refusal prints, a tenth of a GiB.
_CACHE_WORTH_BYTES = _GIB // 10
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
    # An explicit preset: the load is eager unless a ceremony or dm-cfg2 defers
    # it, so the driver models were checked beside the old model in case it waits.
    may_wait: bool = False

    def sentence(self, unet_name: str, unloaded: driver_unload.Unloaded | None = None) -> str:
        """What a refusal that still stands says was counted and freed, with a leading space."""
        parts = []
        if self.credit:
            old = old_model.display_name(self.credit.dropped) or "the old model"
            new = old_model.display_name(unet_name)
            freed = (f"The Worker on {self.box} drops {old} before this load, which frees at "
                     f"least {self.credit.gib} GiB{self.credit.note}")
            first = ("But this load may wait for the first render (an identity ceremony or a "
                     "one-model-a-rank cfg2 split defers it), and this graph's driver models then "
                     f"load first, beside {old}" if self.may_wait else
                     f"But this graph's driver models load first, beside {old}")
            parts.append(
                f"{freed}, and {new} is still {_gib(self.short_bytes)} GiB short after that."
                if self.short_bytes else
                f"{freed}, and that would make room for {new}. {first}, and they are "
                f"{_gib(self.beside_short_bytes)} GiB short.")
        if unloaded is not None:
            parts.append(unloaded.refusal_sentence())
        elif self.kept:
            parts.append(self.kept)
        return "".join(f" {part}" for part in parts)


NONE = Relief()
# Read-only after import: the node classes that load a DiT on the Workers, and
# the slot each loads into.
_SLOT_OF_LOADER = {"DGXMonarchUNETLoader": "cond", "DGXMonarchUncondUNETLoader": "uncond"}


def other_slot_loads(prompt: Any, slot: str) -> list[dict[str, Any]] | None:
    """The DiT loads this graph makes into the other slot, as quote specs.

    Empty for a graph with none, and None when one is there and its checkpoint
    is not a plain name in the graph (a linked input), so nothing can be asked.
    Only the nodes ComfyUI will run are read, by the walk the graph price uses
    (``graph_live.live_nodes``): a loader behind a lazy switch's unselected
    branch, or one nothing consumes, never loads and takes no room.
    """
    specs: list[dict[str, Any]] = []
    if not isinstance(prompt, Mapping):
        return specs
    live = live_nodes(prompt)
    for key, node in prompt.items():
        if str(key) not in live or not isinstance(node, Mapping):
            continue
        other = _SLOT_OF_LOADER.get(str(node.get("class_type", "")))
        if other is None or other == slot:
            continue
        inputs = node.get("inputs")
        name = inputs.get("unet_name") if isinstance(inputs, Mapping) else None
        if not isinstance(inputs, Mapping) or not isinstance(name, str) or not name:
            return None
        dtype = inputs.get("weight_dtype")
        options = {"weight_dtype": dtype} if isinstance(dtype, str) and dtype != "default" else {}
        specs.append({"unet_name": name, "options": options, "loras": [], "slot": other})
    return specs


def _less_the_other_slot(credit: swap_credit.SwapCredit, mesh: Any, own: dict[str, Any],
                         others: list[dict[str, Any]] | None) -> swap_credit.SwapCredit:
    """``credit`` less what the graph's loads into the other slot add to this host.

    Each price here charges one DiT alone, so a credit for its slot must leave
    the room the other slot's load takes (``swap_credit.growth``). The Workers
    are asked once more, with this load and the others in one question as the
    sample-time agreement asks it, so a rank that holds one slot only answers
    for that slot. No answer, or a load that cannot be read, means no credit
    and the price as it was.
    """
    if not credit or others == []:
        return credit
    left = swap_credit.NONE
    asked = None if others is None else old_model_clear.driver_host_answers(
        mesh, own["unet_name"], own["options"], own["slot"], loras=own["loras"], beside=others)
    if asked is not None:
        # Only cfg2 can run one model a rank (dm-cfg2). Under any other explicit
        # preset every rank loads both, whatever topology the fleet last held.
        every = str(getattr(mesh, "topology_preset", "auto") or "auto") not in ("auto", "cfg2")
        left = swap_credit.from_answers(asked.mine, own["slot"], box=credit.box)
        for slot in sorted({spec["slot"] for spec in others or ()}):
            left = swap_credit.less(left, swap_credit.growth(asked.mine, slot, every_rank=every))
    if not left:
        log.info("the drop of %s is not counted for %s, because this graph also loads a model "
                 "into the other slot and that load %s", credit.dropped or "the old model",
                 own["unet_name"], "takes the room the drop frees" if asked is not None
                 else "could not be priced")
    return left


def _unload_driver_models(mesh: Any) -> tuple[driver_unload.Unloaded | None, str]:
    """Unload what ComfyUI holds on this host and release its allocator cache, or say why the driver kept them."""
    if mesh_safety.env_enabled(driver_unload.DRIVER_UNLOAD_DISABLE_ENV):
        return None, ""
    mm = driver_unload.comfy_model_management()
    if mm is None:
        return None, ""
    held = driver_unload.held_models(mm)
    busy = driver_unload.busy_reason(getattr(mesh, "handle", None))
    if busy:
        count = len(held)
        return None, "" if not held else (
            f"The driver did not unload the {count} model{'' if count == 1 else 's'} "
            f"ComfyUI holds on this host before refusing, because {busy}; queue again "
            "once that work ends.")
    # ComfyUI's own /free sequence. With no model held it still returns the
    # allocator's cached blocks, which the driver's ComfyUI keeps for longer
    # since its free-memory reading follows MemAvailable (driver_uma_memory.py).
    outcome = driver_unload.unload(mm, held)
    if outcome.names:
        # Nothing ComfyUI held is loaded now, so no earlier charge stands for it.
        loader_graph.reset_charged_artifacts()
        return outcome, ""
    # No model left ComfyUI: worth a second price only where the released cache shows in the reading.
    return (outcome if outcome.gain >= _CACHE_WORTH_BYTES else None), ""


def relieve(shape: Any, windows: Callable[[Any], tuple[Any, ...]], *, mesh: Any,
            unet_name: str, options: Any, slot: str,
            tried: driver_unload.Unloaded | None = None, prompt: Any = None,
            stack: int = 0) -> Relief:
    """Count, then free, what the box gets back before this load; never raises.

    ``shape`` is the loader's own (``loader_preflight._Shape``) and ``windows``
    prices one. ``slot`` is the slot the load goes into; with none known the
    Workers are not asked and nothing is credited. ``tried`` is the unload an
    earlier pass of this load made, so it is not made again. A result with
    ``unloaded`` set tells the loader to price the load again from the start.

    ``stack`` is the driver models this graph still has to build
    (``graph_load_price.load_window_models``). The Worker drops the old model
    when it loads the new one. An eager load does that at the loader node. A
    deferred load does it at the first render, after the driver has built the
    graph's text encoder and VAEs beside the old model. Under ``auto`` the
    load is deferred and ``shape`` already says so. Under an explicit preset
    it is eager unless an identity ceremony or the dm-cfg2 split defers it
    (``nodes/loaders.py``), which is decided after this price, on the live
    fleet. So there the credit must hold both ways: the load is priced as a
    deferred one, and ``stack`` must fit beside the old model on the reading
    as it is, with no credit.
    """
    relief = NONE
    try:
        # A price that reads the weights as resident (the loader's memo, or a
        # first Worker on another box) charges no load, so it has no drop to count.
        pending = (slot and not mesh_safety.env_enabled(swap_credit.DISABLE_ENV)
                   and any(window.weight_bytes for window in windows(shape)))
        asked = (old_model_clear.driver_host_answers(mesh, unet_name, options, slot)
                 if pending else UNASKED)
        box = graph_load_price.driver_box(mesh)
        credit = (swap_credit.from_answers(asked.mine, slot, box=box)
                  if pending and asked is not None else swap_credit.NONE)
        relief = Relief(asked=asked, box=box)
        credit = _less_the_other_slot(
            credit, mesh, {"unet_name": unet_name, "options": options, "loras": [], "slot": slot},
            other_slot_loads(prompt, slot))
        if credit:
            # What the driver builds before the drop if the load waits for the first render.
            waits = max(int(shape.load_stack_bytes), int(stack))
            priced = windows(replace(shape, mem_available=shape.mem_available + credit.bytes,
                                     load_stack_bytes=waits))
            short = max((window.shortfall for window in priced
                         if not graph_load_price.fits_box(window)), default=0)
            # A deferred load: the driver builds this graph's models before the
            # first render, while the Worker still holds the old model.
            beside = max(shape.floor_reserve + waits - shape.mem_available, 0)
            relief = Relief(fits=not short and not beside, credit=credit, asked=asked,
                            short_bytes=short, beside_short_bytes=beside, box=box,
                            may_wait=waits > shape.load_stack_bytes)
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
        "frees at least %s GiB%s; MemAvailable read %s GiB",
        unet_name, relief.box, old_model.display_name(dropped) or "the old model",
        relief.credit.gib, relief.credit.note, _gib(mem_available))


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
        if (mesh is None or not isinstance(spec, dict) or not spec.get("unet_name")
                or mesh_safety.env_enabled(swap_credit.DISABLE_ENV)):
            return swap_credit.NONE
        name = str(spec["unet_name"])
        asked = old_model_clear.driver_host_answers(
            mesh, name, spec.get("options"), "cond", loras=spec.get("loras"))
        if asked is None:
            return swap_credit.NONE
        credit = swap_credit.from_answers(asked.mine, "cond", box=graph_load_price.driver_box(mesh))
        # A dual-model guider's second DiT, as the request carries it.
        uncond = request.get("uncond_model")
        others: list[dict[str, Any]] | None = [] if uncond is None else None
        if isinstance(uncond, dict) and isinstance(uncond.get("unet_name"), str) and uncond["unet_name"]:
            others = [{"unet_name": uncond["unet_name"], "options": uncond.get("options"),
                       "loras": uncond.get("loras") or [], "slot": "uncond"}]
        own = {"unet_name": name, "options": spec.get("options"),
               "loras": spec.get("loras") or [], "slot": "cond"}
        return _less_the_other_slot(credit, mesh, own, others)
    except Exception as exc:
        log.info("render preflight could not count what the driver host gets back before "
                 "this load (%r); its own price stands", exc)
        return swap_credit.NONE
