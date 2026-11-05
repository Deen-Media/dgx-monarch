"""The "Old model in the way" refusal: a model loaded earlier blocks this one.

Both capacity prices that see a Worker's resident take it as still held
(docs/DESIGN.md section 5.9): the cross-rank agreement's quote and the loader
preflight on the driver host, where the head's actor shares the box. The one
exception is the slot's own stock resident a swap replaces, which the store
drops before the load and the quote credits; a clear is then weighed against
the uncredited price, because a cleared box holds no copy to drop. A clear of
the attached mesh, the same teardown as Reset attached mesh, returns all of it.
This module decides whether a clear would make room and writes the plain
sentences such a refusal carries. ``nodes/old_model_clear.py`` owns the card,
the clear and the Always flow.

A clear would make room on a short box when its shortfall is smaller than what
the clear gives back there: the file bytes of every stock resident, whose GPU
copy an in-process drop was measured to return (2026-11-04), or for a load cast
to another weight dtype the copy the cast built (a 26.6 GiB bf16-cast DiT's drop
raised the head from 58.7 to 88.5 GiB on 2026-11-05),
plus the actor's arena above a fresh actor's floor
(``actor/actor_footprint.py``). A slab or comfy-managed resident and an FSDP
shard are credited the arena only. Pure: no ComfyUI, no mesh call and no file
read.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn

from .capacity_fit import gib, hot_swap_sentence, stock_second_term
from .driver_footprint import DriverFootprintCapacityError
from .mesh_safety import StockLoadCapacityError
from .refusal import RefusalClass, refusal

if TYPE_CHECKING:
    from .actor.capacity_quote import RankPrice
    from .capacity_agreement import RankAnswer

CARD_TITLE = "Old model in the way"
CLEAR_LABEL = "Clear and retry"
ALWAYS_LABEL = "Always clear and retry"
CLEAR_ENV = "DGXM_ALLOW_CLEAR_RETRY"
# What the driver calls its own box when no cluster config names one.
LOCAL_BOX = "this box"
# The one residency whose drop was measured to return its file size in-process.
CREDITED_RESIDENCY = "stock"
_NOT_A_RESIDENT = frozenset({"in_flight"})
_MODEL_SUFFIXES = (".safetensors", ".sft", ".gguf", ".ckpt", ".pt", ".pth", ".bin")
# What a fresh Worker takes back on its first load (lazy imports, kernel
# caches) that the arena figure counts as returned: a clear must beat the
# shortfall by this much before the card is offered. The Worker ladder's copy
# is actor_footprint.RESET_MARGIN_BYTES.
CLEAR_MARGIN_BYTES = 2 << 30
FLEET_GUARD = "cross_rank_capacity"
LOADER_GUARD = "loader_footprint_preflight"
TROUBLESHOOTING = 93
# For a row from an older Worker, which names a residency but no model.
_AS_HELD = (
    "This price was taken against what each box already holds: {named}. The "
    "quote prices the slot as it stands and does not model the drop, so a load "
    "that would replace a resident is priced with that resident still held."
)


@dataclass(frozen=True, slots=True)
class ShortBox:
    """One box the load does not fit, and what a clear would give back there."""

    box: str
    old_models: tuple[str, ...]
    required_bytes: int
    available_bytes: int
    gain_bytes: int
    # Old models the price already credited, because the Worker drops them
    # before this load: what keeps the box short is what they left behind.
    dropped: tuple[str, ...] = ()

    @property
    def shortfall_bytes(self) -> int:
        return max(self.required_bytes - self.available_bytes, 0)

    @property
    def clear_makes_room(self) -> bool:
        return (bool(self.old_models)
                and self.shortfall_bytes + CLEAR_MARGIN_BYTES < self.gain_bytes)

    @property
    def cleared_bytes(self) -> int:
        """About what this box would have for the load after a clear."""
        return max(self.available_bytes + self.gain_bytes - CLEAR_MARGIN_BYTES, 0)


@dataclass(frozen=True, slots=True)
class ClearOffer:
    """Every short box, the model asked for, and how many boxes a clear covers."""

    unet_name: str
    boxes: tuple[ShortBox, ...]
    hosts: int
    local: bool
    # The attached mesh the facts came from, which a card and an Always clear
    # reset and no other; the agreement binds its own after the raise.
    fleet: Any = field(default=None, compare=False, repr=False)

    @property
    def new_model(self) -> str:
        return display_name(self.unet_name)

    @property
    def helps(self) -> bool:
        """Only when a clear would make room on every short box."""
        return bool(self.boxes) and all(box.clear_makes_room for box in self.boxes)


class OldModelRefusal(StockLoadCapacityError):
    """A capacity refusal a clear of the attached mesh would answer.

    It stays in the stock-load capacity family, so every classifier reads it
    as the shortfall it is. ``offer`` says which boxes and old models; only the
    driver raises it, so it never crosses an actor boundary.
    """

    def __init__(self, message: str, offer: ClearOffer | None = None) -> None:
        super().__init__(message)
        self.offer = offer
        # The attached mesh that refused: the offer's, or bound by the raising
        # site's caller, so a card never resets a fleet that came later.
        self.fleet: Any = getattr(offer, "fleet", None)


class LoaderOldModelError(DriverFootprintCapacityError, OldModelRefusal):
    """The loader preflight's form, which that preflight's own handler re-raises."""


def offer_of(exc: BaseException) -> ClearOffer | None:
    """The offer a refusal carries when a clear would make room, else None."""
    offer = getattr(exc, "offer", None) if isinstance(exc, OldModelRefusal) else None
    return offer if isinstance(offer, ClearOffer) and offer.helps else None


def display_name(unet_name: str) -> str:
    """A checkpoint as a person names it: no folder, no file suffix."""
    base = os.path.basename(str(unet_name or "")) or str(unet_name or "")
    root, suffix = os.path.splitext(base)
    return root if root and suffix.lower() in _MODEL_SUFFIXES else base


def box_name(answer: RankAnswer) -> str:
    """The driver's own name for a rank's box, as its other refusals use it."""
    configured = getattr(answer, "configured_host", "") or ""
    if configured == LOCAL_BOX:
        return LOCAL_BOX
    return configured or getattr(answer, "host", "") or f"rank {answer.rank}"


def kept_slots(quotes: Sequence[RankPrice], keep: frozenset[str] = frozenset()) -> frozenset[str]:
    """The slots on one rank whose resident this request or graph still needs.

    A slot the request reuses is needed, and so is one holding a model this
    graph loads: a clear would only make the fresh fleet load it again, so it
    is neither an old model nor memory a clear gives back.
    """
    kept = {quote.slot for quote in quotes if quote.would_load == "reuse"}
    for quote in quotes[:1]:
        kept.update(slot for slot, name in quote.resident_names.items()
                    if name and name in keep)
    return frozenset(kept)


# Load lifetime: the checkpoints the graph being run loads, set by the loader
# node around its eager agreement, so no card calls one of them old.
_GRAPH_LOADS: ContextVar[frozenset[str]] = ContextVar("dgxm_graph_loads", default=frozenset())


@contextmanager
def graph_loads(names: Iterable[str]) -> Iterator[None]:
    """Name the checkpoints this graph loads for the agreements inside the block."""
    token = _GRAPH_LOADS.set(frozenset(str(name) for name in names))
    try:
        yield
    finally:
        _GRAPH_LOADS.reset(token)


def graph_names() -> frozenset[str]:
    return _GRAPH_LOADS.get()


def named_in(specs: Any) -> frozenset[str]:
    """The checkpoints a quote request loads, which no card may call old."""
    return frozenset(str(spec.get("unet_name")) for spec in specs or ()
                     if isinstance(spec, dict) and spec.get("unet_name"))


def old_models(quote: RankPrice, kept: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """The models this rank's slots hold, by name, or () when its row names none."""
    names: list[str] = []
    for slot, word in quote.holds_resident.items():
        name = quote.resident_names.get(slot)
        if word and word not in _NOT_A_RESIDENT and name and slot not in kept:
            shown = display_name(name)
            if shown not in names:
                names.append(shown)
    return tuple(names)


def rank_gain(quote: RankPrice, kept: frozenset[str] = frozenset()) -> int:
    """What a clear gives back on this rank beyond an in-process unload."""
    held = sum(max(int(quote.resident_bytes.get(slot, 0) or 0), 0)
               for slot, word in quote.holds_resident.items()
               if word == CREDITED_RESIDENCY and slot not in kept)
    return held + max(int(quote.arena_bytes or 0), 0)


def _numbers(quote: RankPrice, kept: frozenset[str] = frozenset()) -> tuple[int, int] | None:
    """The row's price and availability, uncredited where the gain counts the credit.

    A swap credit is the slot's own stock resident, and ``rank_gain`` counts that
    resident too unless the slot is kept, so a clear is weighed against the price
    a cleared box pays: the row's figure plus its credit.
    """
    block = (quote.slab_measured if quote.use_slab else quote.measured) or {}
    try:
        required, available = int(block["required_bytes"]), int(block["mem_available_bytes"])
    except (KeyError, TypeError, ValueError):
        return None
    if _counts_credit(quote, kept):
        required += max(int(block.get("credit_bytes") or 0), 0)
    return required, available


def _counts_credit(quote: RankPrice, kept: frozenset[str]) -> bool:
    """Whether ``rank_gain`` counts the stock resident this row's price credited."""
    return quote.slot not in kept and quote.holds_resident.get(quote.slot) == CREDITED_RESIDENCY


def _dropped(quote: RankPrice, kept: frozenset[str]) -> tuple[str, ...]:
    """The old model this row's swap credit already took out of its price."""
    block = quote.measured or {}
    name = block.get("dropped") if block.get("credit_bytes") and not quote.use_slab else None
    return (display_name(str(name)),) if name and _counts_credit(quote, kept) else ()


def credited_sentence(where_: str, unet_name: str, quote: RankPrice, needed: float,
                      weights: float, avail: float, short: float) -> str:
    """The fleet card's sentence for a credited price that still falls short.

    A LoRA change the store makes in place on the held base keeps its own words;
    a swap credit (``dropped`` in the block) names the copy the Worker drops
    before it loads, a change of dtype on the same file included. The held size
    is the credit itself, never the new load's copy.
    """
    block = quote.measured or {}
    held = gib(int(block.get("credit_bytes") or 0))
    if not block.get("dropped"):
        return hot_swap_sentence(where_, unet_name, needed, held, avail, short)
    return (f"{where_} priced a stock load of {unet_name} after the Worker drops "
            f"{display_name(str(block['dropped']))}, the {held} GiB copy "
            f"its slot holds. Beyond that copy the load needs {needed} GiB, a {weights} GiB "
            f"file {stock_second_term(block)}, and {avail} GiB of unified memory is "
            f"available, {short} GiB short.")


def fleet_offer(short: Sequence[tuple[RankAnswer, RankPrice]],
                answers: Sequence[RankAnswer], unet_name: str,
                keep: frozenset[str] = frozenset()) -> ClearOffer | None:
    """The offer for a fleet's short rows, or None when it cannot be stated.

    None when no short box names an old model, or when a short row carries no
    figures: a price that cannot be read can never show that a clear makes
    room. One rank's rows share one snapshot, so a rank counts its gain once
    and keeps its worst row; ranks that share a box add their gains. ``keep``
    names the models this request and graph load (``kept_slots``).
    """
    ranks: dict[int, tuple[str, tuple[str, ...], int, int, int, tuple[str, ...]]] = {}
    for answer, quote in short:
        kept = kept_slots(answer.quotes or (quote,), keep)
        numbers = _numbers(quote, kept)
        if numbers is None:
            return None
        seen = ranks.get(answer.rank)
        if seen is None or numbers[0] - numbers[1] > seen[2] - seen[3]:
            ranks[answer.rank] = (box_name(answer), old_models(quote, kept), *numbers,
                                  rank_gain(quote, kept), _dropped(quote, kept))
    boxes: dict[str, ShortBox] = {}
    for _rank, (name, olds, required, available, gain, dropped) in sorted(ranks.items()):
        seen_box = boxes.get(name)
        if seen_box is not None:
            gain += seen_box.gain_bytes
            olds = tuple(dict.fromkeys((*seen_box.old_models, *olds)))
            dropped = tuple(dict.fromkeys((*seen_box.dropped, *dropped)))
            if seen_box.shortfall_bytes > max(required - available, 0):
                required, available = seen_box.required_bytes, seen_box.available_bytes
        boxes[name] = ShortBox(name, olds, required, available, gain, dropped)
    if not any(box.old_models for box in boxes.values()):
        return None
    local = bool(answers) and all(
        getattr(answer, "configured_host", "") == LOCAL_BOX for answer in answers)
    hosts = 1 if local else len({box_name(answer) for answer in answers}) or 1
    return ClearOffer(unet_name, tuple(boxes.values()), hosts, local)


def box_offer(unet_name: str, box: ShortBox, *, hosts: int, local: bool,
              fleet: Any = None) -> ClearOffer | None:
    """The one-box offer the loader preflight makes for the driver host."""
    if not box.old_models:
        return None
    return ClearOffer(unet_name, (box,), max(int(hosts), 1), bool(local), fleet)


def _join(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def held_sentence(offer: ClearOffer) -> str:
    """'The old model (<name>) is still loaded on <box> and <new> does not fit beside it.'"""
    groups: dict[tuple[str, ...], list[str]] = {}
    for box in offer.boxes:
        named = tuple(name for name in box.old_models if name not in box.dropped)
        if named:
            groups.setdefault(named, []).append(box.box)
    if not groups:
        # Every old model is one the price already dropped: what is left in the
        # way is the memory those loads left behind, which a clear returns.
        left = list(dict.fromkeys(name for box in offer.boxes for name in box.old_models))
        where_ = [box.box for box in offer.boxes if box.old_models]
        return (f"{_join(where_)} still {'holds' if len(where_) == 1 else 'hold'} memory that "
                f"loading {_join(left)} earlier left behind, and {offer.new_model} does not "
                "fit beside it.")
    parts = [f"({_join(names)}) is still loaded on {_join(where)}"
             if not index else f"the old model ({_join(names)}) on {_join(where)}"
             for index, (names, where) in enumerate(groups.items())]
    if len(parts) == 1:
        # A comma when the box list already has an "and" of its own.
        joint = "," if " and " in parts[0].rsplit(" on ", 1)[-1] else ""
        return (f"The old model {parts[0]}{joint} and {offer.new_model} does not fit "
                "beside it.")
    return (f"The old model {', '.join(parts)}, and {offer.new_model} does not fit "
            "beside them.")


def scope_words(offer: ClearOffer, *, cleared: bool) -> str:
    """'this box', 'the Sparks', or after a clear 'both Sparks'."""
    if offer.local:
        return LOCAL_BOX
    if offer.hosts == 1:
        return "the Spark"
    if cleared:
        return "both Sparks" if offer.hosts == 2 else f"all {offer.hosts} Sparks"
    return "the Sparks"


def card_body(offer: ClearOffer) -> str:
    """The card's sentence: what is in the way, and the question."""
    scope = scope_words(offer, cleared=False)
    return f"{held_sentence(offer)} Clear {scope} and run again?"


def remedy(offer: ClearOffer, *, card: bool = True) -> str:
    """The one action that clears it, in the panel and headless.

    ``card`` is whether a card was raised: a driver that cannot resolve the
    checkpoint in its own model folders raises none, so the text names only
    the panel's Reset attached mesh.
    """
    click = (f'click "{CLEAR_LABEL}" on the "{CARD_TITLE}" card, or click Reset attached '
             "mesh," if card else "click Reset attached mesh,")
    return (f"Open the DGX Monarch panel and {click} then queue the render again. Headless: "
            f"set {CLEAR_ENV}=1 on the driver and it clears and retries on its own.")


def no_room(offer: ClearOffer) -> str:
    """Why the ordinary refusal stands although an old model is still loaded."""
    for box in offer.boxes:
        if not box.clear_makes_room:
            whole = (" That is the whole price, because a cleared box holds no copy for "
                     "the Worker to drop." if box.dropped else "")
            return (f"A clear would not make room: {offer.new_model} needs "
                    f"{gib(box.required_bytes)} GiB on {box.box}, and a cleared box "
                    f"would have about {gib(box.cleared_bytes)} GiB.{whole} Choose a smaller or "
                    "quantized artifact, free memory on that box, or use slab residency "
                    "where a card offers it.")
    return ""


def log_line(offer: ClearOffer) -> str:
    """'cleared both Sparks to make room for <new>; <old> was loaded earlier'.

    "Earlier", not "an earlier render": on the sample path the old model can be
    one an earlier sampler of the same graph loaded.
    """
    olds = list(dict.fromkeys(name for box in offer.boxes for name in box.old_models))
    were = "was loaded earlier" if len(olds) == 1 else "were loaded earlier"
    return (f"cleared {scope_words(offer, cleared=True)} to make room for {offer.new_model}; "
            f"{_join(olds) if olds else 'the old model'} {were}")


def card_numbers(offer: ClearOffer) -> str:
    """The card's figure line, from the leanest short box."""
    box = max(offer.boxes, key=lambda item: item.shortfall_bytes)
    return (f"needs {gib(box.required_bytes)} GiB, {gib(box.available_bytes)} GiB "
            f"available on {box.box}")


def where(answer: RankAnswer) -> str:
    """A rank by the box name the held sentence uses: 'rank 0 on head'."""
    return f"rank {answer.rank} on {box_name(answer)}"


def held_clause(pairs: Sequence[tuple[RankAnswer, RankPrice]],
                offer: ClearOffer | None) -> str:
    """What an ordinary refusal says about the residents it was priced over."""
    if offer is not None:
        return " ".join(part for part in (held_sentence(offer), no_room(offer)) if part)
    named = []
    for answer, quote in pairs:
        block = (quote.slab_measured if quote.use_slab else quote.measured) or {}
        if quote.would_load not in ("swap", "reuse") or not block or block.get("credit_bytes"):
            continue    # a row with no price, or one taken after the drop it names
        word = quote.holds_resident.get(quote.slot)
        if word:
            named.append(f"{answer.where} holds a {word} resident in the "
                         f"{quote.slot} slot and this load reads as a {quote.would_load}")
    return _AS_HELD.format(named="; ".join(named)) if named else ""


def raise_fleet_refusal(offer: ClearOffer, short_ranks: int, world: int,
                        priced: str, *, card: bool = True) -> NoReturn:
    """The cross-rank agreement's refusal when a clear would make room."""
    raise OldModelRefusal(refusal(
        RefusalClass.CAPACITY,
        f"this render needs every rank to hold the model and {short_ranks} of {world} "
        "cannot hold it beside the model loaded there earlier, so nothing was "
        f"loaded on any rank. {priced} {held_sentence(offer)} {remedy(offer, card=card)} "
        "This refusal is "
        "capacity-protective (class C), and nothing was quarantined.",
        guard=FLEET_GUARD, waivable=False, troubleshooting=TROUBLESHOOTING), offer)


def raise_loader_refusal(offer: ClearOffer, message: str, *, card: bool = True) -> NoReturn:
    """The loader preflight's refusal when a clear would make room."""
    raise LoaderOldModelError(refusal(
        RefusalClass.CAPACITY,
        f"{message} {held_sentence(offer)} {remedy(offer, card=card)}",
        guard=LOADER_GUARD, waivable=False), offer)


def as_mapping(offer: ClearOffer) -> dict[str, Any]:
    """The offer as plain data, for logs and tests."""
    return {"unet_name": offer.unet_name, "hosts": offer.hosts, "local": offer.local,
            "helps": offer.helps,
            "boxes": [{"box": box.box, "old_models": list(box.old_models),
                       "shortfall_bytes": box.shortfall_bytes,
                       "gain_bytes": box.gain_bytes} for box in offer.boxes]}
