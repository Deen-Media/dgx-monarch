"""The "Old model in the way" refusal: a model an earlier render loaded blocks this one.

Both capacity prices that see a Worker's resident take it as still held
(docs/DESIGN.md section 5.9): the cross-rank agreement's quote and the loader
preflight on the driver host, where the head's actor shares the box. A clear of
the attached mesh, the same teardown as Reset attached mesh, returns all of it.
This module decides whether a clear would make room and writes the plain
sentences such a refusal carries. ``nodes/old_model_clear.py`` owns the card,
the clear and the Always flow.

A clear would make room on a short box when its shortfall is smaller than what
the clear gives back there: the file bytes of every stock resident, whose GPU
copy an in-process drop was measured to return (2026-11-04), plus the actor's
arena above a fresh actor's floor (``actor/actor_footprint.py``). A slab or
comfy-managed resident is credited the arena only. Pure: no ComfyUI, no mesh
call and no file read.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from .capacity_fit import gib
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

    @property
    def shortfall_bytes(self) -> int:
        return max(self.required_bytes - self.available_bytes, 0)

    @property
    def clear_makes_room(self) -> bool:
        return bool(self.old_models) and self.shortfall_bytes < self.gain_bytes


@dataclass(frozen=True, slots=True)
class ClearOffer:
    """Every short box, the model asked for, and how many boxes a clear covers."""

    unet_name: str
    boxes: tuple[ShortBox, ...]
    hosts: int
    local: bool

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


def old_models(quote: RankPrice) -> tuple[str, ...]:
    """The models this rank's slots hold, by name, or () when its row names none."""
    names: list[str] = []
    for slot, word in quote.holds_resident.items():
        name = quote.resident_names.get(slot)
        if word and word not in _NOT_A_RESIDENT and name:
            shown = display_name(name)
            if shown not in names:
                names.append(shown)
    return tuple(names)


def rank_gain(quote: RankPrice) -> int:
    """What a clear gives back on this rank beyond an in-process unload."""
    held = sum(max(int(quote.resident_bytes.get(slot, 0) or 0), 0)
               for slot, word in quote.holds_resident.items()
               if word == CREDITED_RESIDENCY)
    return held + max(int(quote.arena_bytes or 0), 0)


def _numbers(quote: RankPrice) -> tuple[int, int] | None:
    block = quote.slab_measured if quote.use_slab else quote.measured
    try:
        return (int((block or {})["required_bytes"]),
                int((block or {})["mem_available_bytes"]))
    except (KeyError, TypeError, ValueError):
        return None


def fleet_offer(short: Sequence[tuple[RankAnswer, RankPrice]],
                answers: Sequence[RankAnswer], unet_name: str) -> ClearOffer | None:
    """The offer for a fleet's short rows, or None when it cannot be stated.

    None when no short box names an old model, or when a short row carries no
    figures: a price that cannot be read can never show that a clear makes
    room. One rank's rows share one snapshot, so a rank counts its gain once
    and keeps its worst row; ranks that share a box add their gains.
    """
    ranks: dict[int, tuple[str, tuple[str, ...], int, int, int]] = {}
    for answer, quote in short:
        numbers = _numbers(quote)
        if numbers is None:
            return None
        seen = ranks.get(answer.rank)
        if seen is None or numbers[0] - numbers[1] > seen[2] - seen[3]:
            ranks[answer.rank] = (box_name(answer), old_models(quote), *numbers,
                                  rank_gain(quote))
    boxes: dict[str, ShortBox] = {}
    for _rank, (name, olds, required, available, gain) in sorted(ranks.items()):
        seen_box = boxes.get(name)
        if seen_box is not None:
            gain += seen_box.gain_bytes
            olds = tuple(dict.fromkeys((*seen_box.old_models, *olds)))
            if seen_box.shortfall_bytes > max(required - available, 0):
                required, available = seen_box.required_bytes, seen_box.available_bytes
        boxes[name] = ShortBox(name, olds, required, available, gain)
    if not any(box.old_models for box in boxes.values()):
        return None
    local = bool(answers) and all(
        getattr(answer, "configured_host", "") == LOCAL_BOX for answer in answers)
    hosts = 1 if local else len({box_name(answer) for answer in answers}) or 1
    return ClearOffer(unet_name, tuple(boxes.values()), hosts, local)


def box_offer(unet_name: str, box: ShortBox, *, hosts: int, local: bool) -> ClearOffer | None:
    """The one-box offer the loader preflight makes for the driver host."""
    if not box.old_models:
        return None
    return ClearOffer(unet_name, (box,), max(int(hosts), 1), bool(local))


def _join(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def held_sentence(offer: ClearOffer) -> str:
    """'The old model (<name>) is still loaded on <box> and <new> does not fit beside it.'"""
    groups: dict[tuple[str, ...], list[str]] = {}
    for box in offer.boxes:
        if box.old_models:
            groups.setdefault(box.old_models, []).append(box.box)
    parts = [f"({_join(names)}) is still loaded on {_join(where)}"
             if not index else f"the old model ({_join(names)}) on {_join(where)}"
             for index, (names, where) in enumerate(groups.items())]
    if len(parts) == 1:
        return f"The old model {parts[0]} and {offer.new_model} does not fit beside it."
    return (f"The old model {', '.join(parts)}, and {offer.new_model} does not fit "
            "beside them.")


def _scope(offer: ClearOffer, *, cleared: bool) -> str:
    if offer.local:
        return LOCAL_BOX
    if offer.hosts == 1:
        return "the Spark"
    if cleared:
        return "both Sparks" if offer.hosts == 2 else f"all {offer.hosts} Sparks"
    return "the Sparks"


def card_body(offer: ClearOffer) -> str:
    """The card's sentence: what is in the way, and the question."""
    scope = _scope(offer, cleared=False)
    return f"{held_sentence(offer)} Clear {scope} and run again?"


def remedy(offer: ClearOffer) -> str:
    """The one action that clears it, in the panel and headless."""
    return (f'Open the DGX Monarch panel and click "{CLEAR_LABEL}" on the "{CARD_TITLE}" '
            "card, or click Reset attached mesh, then queue the render again. Headless: "
            f"set {CLEAR_ENV}=1 on the driver and it clears and retries on its own.")


def no_room(offer: ClearOffer) -> str:
    """Why the ordinary refusal stands although an old model is still loaded."""
    for box in offer.boxes:
        if not box.clear_makes_room:
            return (f"A clear would not make room: {offer.new_model} needs "
                    f"{gib(box.required_bytes)} GiB on {box.box}, and a cleared box "
                    f"would have about {gib(box.available_bytes + box.gain_bytes)} GiB.")
    return ""


def log_line(offer: ClearOffer) -> str:
    """'cleared both Sparks to make room for <new>; <old> was loaded from an earlier render'."""
    olds = list(dict.fromkeys(name for box in offer.boxes for name in box.old_models))
    were = ("was loaded from an earlier render" if len(olds) == 1
            else "were loaded from earlier renders")
    return (f"cleared {_scope(offer, cleared=True)} to make room for {offer.new_model}; "
            f"{_join(olds) if olds else 'the old model'} {were}")


def card_numbers(offer: ClearOffer) -> str:
    """The card's figure line, from the leanest short box."""
    box = max(offer.boxes, key=lambda item: item.shortfall_bytes)
    return (f"needs {gib(box.required_bytes)} GiB, {gib(box.available_bytes)} GiB "
            f"available on {box.box}")


def held_clause(pairs: Sequence[tuple[RankAnswer, RankPrice]],
                offer: ClearOffer | None) -> str:
    """What an ordinary refusal says about the residents it was priced over."""
    if offer is not None:
        return " ".join(part for part in (held_sentence(offer), no_room(offer)) if part)
    named = []
    for answer, quote in pairs:
        if quote.would_load not in ("swap", "reuse"):
            continue
        word = quote.holds_resident.get(quote.slot)
        if word:
            named.append(f"{answer.where} holds a {word} resident in the "
                         f"{quote.slot} slot and this load reads as a {quote.would_load}")
    return _AS_HELD.format(named="; ".join(named)) if named else ""


def raise_fleet_refusal(offer: ClearOffer, short_ranks: int, world: int,
                        priced: str) -> NoReturn:
    """The cross-rank agreement's refusal when a clear would make room."""
    raise OldModelRefusal(refusal(
        RefusalClass.CAPACITY,
        f"this render needs every rank to hold the model and {short_ranks} of {world} "
        "cannot beside the model an earlier render left there, so nothing was loaded on "
        f"any rank. {priced} {held_sentence(offer)} {remedy(offer)} This refusal is "
        "capacity-protective (class C), and nothing was quarantined.",
        guard=FLEET_GUARD, waivable=False, troubleshooting=TROUBLESHOOTING), offer)


def raise_loader_refusal(offer: ClearOffer, message: str) -> NoReturn:
    """The loader preflight's refusal when a clear would make room."""
    raise LoaderOldModelError(refusal(
        RefusalClass.CAPACITY, f"{message} {held_sentence(offer)} {remedy(offer)}",
        guard=LOADER_GUARD, waivable=False), offer)


def as_mapping(offer: ClearOffer) -> dict[str, Any]:
    """The offer as plain data, for logs and tests."""
    return {"unet_name": offer.unet_name, "hosts": offer.hosts, "local": offer.local,
            "helps": offer.helps,
            "boxes": [{"box": box.box, "old_models": list(box.old_models),
                       "shortfall_bytes": box.shortfall_bytes,
                       "gain_bytes": box.gain_bytes} for box in offer.boxes]}
