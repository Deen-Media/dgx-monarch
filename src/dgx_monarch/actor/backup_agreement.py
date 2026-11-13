"""On a pair, the ranks free ComfyUI's weight backup together or not at all, and refuse together.

With ``lora_low_rss`` off a rank that ComfyUI loaded whole keeps the host copy
of every patched weight through the render. On one GPU the Worker frees it
before the first step where the sampler will need the room
(``lora_partial_load.free_backup_for_sampling``). A rank must not decide that
alone: its peer would keep its backup, the two slots would differ, the next
stack change would reload on one rank and swap in place on the other, and a
rank that then found its model wrong would raise by itself and leave its peer
waiting in a collective.

So the ranks agree, on the exchange the pair's guard already makes after
every load (``partial_load_guard.check``). Beside its "my load is whole" flag
each rank sends four more integers, and every decision is made from what the
MIN over the ranks brings back, never from a rank's own value:

1. 0 if this rank is short: it holds a backup and its room above the host
   floor is under the ask (1.5 times ComfyUI's full estimate, as on one GPU);
2. 0 if this rank cannot free (a patched model that is not a stock slot of
   this Worker's store, or not on unified memory);
3. 0 if ``DGXM_DISABLE_BACKUP_FREE`` is set on this rank;
4. this rank's margin, its room less what the sampler still needs, in MiB,
   with its rank number in the low bits so the answer names the tightest rank.

- Any rank's switch set: the rule is off for the fleet, the refusal too. That
  is dev's behaviour, for diagnosis and for reading one build both ways.
- No rank short: nothing happens and nothing is logged.
- A rank short and a rank that cannot free: every rank keeps its backup, each
  says what was left standing, and the margin is checked as it is.
- A rank short and none unable: EVERY rank frees its backup, the roomy one
  too, so the slots stay alike and the next stack change reloads on all of
  them. Then each rank reads its model and its room again and the ranks make
  one more exchange: a model that is not what a full load leaves, or a
  tightest margin under zero, refuses on every rank before the first forward.

The need behind the margin is ComfyUI's single-batch estimate for the latent
plus 1 GiB. On record (2026-11-12, a pair of DGX Sparks, the first under its
clock cap, max 2100 MHz) a rank added 6.4 GiB from this point to the end of
sampling where the estimate was 6.2. It is checked against the room as READ
after the free, because the free does not always return the whole copy
(``host_trim``).

This module makes no collective itself: ``partial_load_guard`` owns them and
hands ``settle`` its exchange. A render that does not free makes exactly the
one collective it made before, with more elements in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .. import mesh_safety
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from . import lora_partial_load

log = get_logger(__name__)

GUARD = "pair_backup_room"
TROUBLESHOOTING = 145
_MIB = 2 ** 20
_GIB = 2 ** 30
NEED_OVER_ESTIMATE_BYTES = _GIB
RANK_SLOTS = 64          # the margin's low bits hold the rank: MIN then names the tightest one
NO_CLAIM_MIB = 1 << 20   # a rank that cannot work out its need claims room to spare, not a shortfall
ELEMENTS = 4             # what ``Vote.elements`` adds to the guard's exchange, the same on every rank


class PairBackupRoomError(RuntimeError):
    """Typed fleet refusal: after the ranks' decision a rank has too little room to sample above the host floor."""

    fleet_wide = True


@dataclass
class Vote:
    """What one rank found after its load. ``elements`` is what it sends."""

    rank: int
    short: bool = False
    can: bool = True
    switch: bool = False
    room: int = 0
    need: int | None = None
    ask: int | None = None
    why_not: str = ""
    freeable: list[tuple[Any, Any]] = field(default_factory=list)

    @property
    def elements(self) -> list[int]:
        return [0 if self.short else 1, 1 if self.can else 0, 0 if self.switch else 1,
                _margin(self.room, self.need, self.rank)]


def _margin(room: int, need: int | None, rank: int) -> int:
    mib = NO_CLAIM_MIB if need is None else max(-NO_CLAIM_MIB, min(NO_CLAIM_MIB, (room - need) // _MIB))
    return mib * RANK_SLOTS + (int(rank) % RANK_SLOTS)


def _read_margin(value: int) -> tuple[float, int]:
    """(GiB of margin, rank) from an exchanged element."""
    return (value // RANK_SLOTS) * _MIB / _GIB, value % RANK_SLOTS


def _single_estimate(patcher: Any, noise_shape: Any, conds: Any) -> int | None:
    try:
        import comfy.sampler_helpers

        _full, minimum = comfy.sampler_helpers.estimate_memory(patcher, noise_shape, conds)
        return int(minimum)
    except Exception:
        return None


def vote(patchers: tuple, sampling: tuple[Any, Any], rank: int) -> Vote:
    """This rank's facts after its load. Never raises: a rank that cannot tell says so in its vote."""
    found = Vote(rank=int(rank or 0))
    try:
        found.switch = mesh_safety.env_enabled(lora_partial_load.BACKUP_FREE_DISABLE_ENV)
        seen: set[int] = set()
        held = False
        for patcher in patchers:
            model = getattr(patcher, "model", None)
            if model is None or id(model) in seen or not getattr(patcher, "backup", None):
                continue
            seen.add(id(model))
            if getattr(model, "current_weight_patches_uuid", None) is None:
                continue
            held = True
            slot = getattr(model, lora_partial_load.STORED, None)
            stored = slot() if callable(slot) else None
            if stored is None:
                found.can, found.why_not = False, "a patched model here is not one this Worker's store holds"
            elif patcher.is_dynamic() or not lora_partial_load.shares_host_memory(patcher):
                found.can, found.why_not = False, "a patched model here is not a stock load on unified memory"
            else:
                found.freeable.append((patcher, stored))
        found.room = lora_partial_load._host_room()
        first = patchers[0] if patchers else None
        estimate = None if first is None else _single_estimate(first, *sampling)
        found.need = None if estimate is None else estimate + NEED_OVER_ESTIMATE_BYTES
        found.ask = None if first is None else lora_partial_load._sampling_ask(first, *sampling)
        found.short = bool(held and found.ask is not None and found.room < found.ask)
    except Exception as exc:
        found.short, found.can, found.need = False, False, None
        found.why_not = f"this rank could not weigh its load ({type(exc).__name__}: {exc})"
        found.freeable = []
    return found


def _refuse_room(margin_gib: float, rank: int, mine: Vote, kept: bool) -> None:
    after = "every rank kept its weight backup" if kept else "every rank freed its weight backup"
    raise PairBackupRoomError(refusal(
        RefusalClass.CAPACITY,
        "a LoRA stack is on a model the ranks loaded whole with lora_low_rss off, and after the ranks' "
        f"decision ({after}) rank {rank} is {abs(margin_gib):.1f} GiB short of what sampling this latent "
        "still needs above the 5 GiB host floor. This rank read "
        f"{max(0, mine.room) / _GIB:.1f} GiB of room against {(mine.need or 0) / _GIB:.1f} GiB needed "
        "(ComfyUI's estimate for the latent plus 1 GiB); each Worker's journal gives its own figures. "
        "Nothing was sampled and no rank went on. What would fit: a shorter clip or a smaller latent; "
        "lora_low_rss on, which keeps no weight backup; more free memory on that box; or the same "
        "render on a single box.",
        guard=GUARD, waivable=False, troubleshooting=TROUBLESHOOTING))


def settle(mine: Vote, answer: list[int], exchange: Any) -> None:
    """Act on the fleet's answer to the votes; ``exchange`` is the guard's MIN exchange, used once more on a free."""
    not_short, can, switch_off, margin = (int(value) for value in answer)
    if switch_off == 0:
        if mine.short or mine.switch:
            log.info("pair backup: the step that frees ComfyUI's weight backup is off for this fleet (%s=1 on "
                     "%s), so every rank keeps its backup through the render and no room is checked",
                     lora_partial_load.BACKUP_FREE_DISABLE_ENV, "this rank" if mine.switch else "a peer")
        return
    if not_short == 1:
        return
    if can == 0:
        standing = sum(len(getattr(patcher, "backup", None) or ()) for patcher, _stored in mine.freeable)
        log.info("pair backup: a rank is short of room for the sampler and a rank cannot free its backup (%s), "
                 "so every rank keeps its own: this rank left the backup of %d weight(s) standing with "
                 "%.1f GiB of room above the floor", mine.why_not or "a peer's cause; its journal names it",
                 standing, max(0, mine.room) / _GIB)
        margin_gib, rank = _read_margin(margin)
        if margin_gib < 0:
            _refuse_room(margin_gib, rank, mine, kept=True)
        return
    wrong: str | None = None
    returned = "it held no backup"
    try:
        for patcher, stored in mine.freeable:
            returned = lora_partial_load._free_backup(patcher, stored)
        for patcher, _stored in mine.freeable:
            wrong = wrong or lora_partial_load._what_is_wrong_after_the_route(patcher, False)
        mine.room = lora_partial_load._host_room()
    except Exception as exc:
        wrong = f"freeing its backup failed ({type(exc).__name__}: {exc})"
    log.info("pair backup: %s, so every rank freed ComfyUI's weight backup before the first step (no weight "
             "moved; a stack change on this model now reloads it on every rank). This rank: %s; %.1f GiB "
             "of room above the floor against %.1f GiB the sampler still needs",
             "this rank was short of room for the sampler" if mine.short else "a peer was short of room",
             returned, max(0, mine.room) / _GIB, (mine.need or 0) / _GIB)
    ok, after = (int(value) for value in exchange([0 if wrong else 1, _margin(mine.room, mine.need, mine.rank)]))
    if wrong is not None:
        raise lora_partial_load.OffloadedInputQuantError(refusal(
            RefusalClass.CAPACITY,
            "a LoRA stack is on a model ComfyUI loaded whole, and after this rank freed its weight backup "
            f"for the sampler the model is not as a full load leaves it: {wrong}. Sampling it could run "
            "without its LoRA, so nothing was sampled on any rank. Reset the attached mesh so the model "
            "loads again, and turn lora_low_rss on (the default on a DGX Spark), which bakes at load time.",
            guard=lora_partial_load.GUARD, waivable=False, troubleshooting=lora_partial_load.TROUBLESHOOTING))
    if ok == 0:
        from . import rank_readiness

        raise rank_readiness.PeerLoadFailureError(refusal(RefusalClass.PHYSICS, rank_readiness.PEER_LOAD_TEXT))
    margin_gib, rank = _read_margin(after)
    if margin_gib < 0:
        _refuse_room(margin_gib, rank, mine, kept=False)


__all__ = ["ELEMENTS", "GUARD", "PairBackupRoomError", "Vote", "settle", "vote"]
