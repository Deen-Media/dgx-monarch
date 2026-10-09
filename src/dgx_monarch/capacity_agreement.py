"""Collect per-rank capacity quotes before any rank allocates.

The driver sends read-only, GPU-free endpoint calls under one shared deadline,
then applies a pure decision function. Per-rank futures identify a missing
reply without waiting for an NCCL collective watchdog.

Both refusal types use class C and the same guard. A measured shortfall stays
in the ``StockLoadCapacityError`` family for cross-residency classification.
A missing reply is a liveness fault: its type and message must not contain
that family name, which the remote-error classifier matches as text.

Agreement does not validate price accuracy or replace post-load checks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, NoReturn

from .actor import store_slab_admit
from .actor.capacity_quote import (
    RUNG_COMFY_MANAGED,
    VERDICT_REFUSE,
    VERDICT_RESCUE,
    WOULD_LOAD_REUSE,
    WOULD_LOAD_SWAP,
    RankPrice,
)
from .capacity_fit import PROBE_PREFLIGHT_WALL, SlabFit, gib
from .consent_descriptor import ConsentDescriptor, SlabResidencyRescueOffer
from .gate_audit_evidence import measured_tag
from .log import get_logger
from .mesh_safety import StockLoadCapacityError
from .refusal import PanelAction, RefusalClass, refusal
from .transfer_utils import failure_summary, safe_call, safe_note

log = get_logger(__name__)

GUARD = "cross_rank_capacity"
# The rescue arm raises under the worker stock guard, whose consent is wired.
# GUARD has no card: no consent makes a refusing rank fit, and one refusing rank refuses the whole fleet.
RESCUE_GUARD = "stock_load_preflight"
TROUBLESHOOTING = 93

# One fleet-wide deadline. The off-loop join adds
# ``mesh_runtime.FUTURE_GET_JOIN_MARGIN_S``; a timed-out await can leave
# its scratch thread blocked until the future's own deadline.
CAPACITY_QUOTE_DEADLINE_S = 30.0

_MAX_REASON_CHARS = 320
_NEEDS_EVERY_RANK = (
    "this render needs every rank to price the load before any rank allocates"
)
_NOTHING_LOADED = (
    "Nothing was loaded on any rank and the fleet was not torn down. If a rank "
    "did not answer, click Reset attached mesh in the DGX Monarch panel. Then "
    "queue again. This refusal is capacity-protective (class C)."
)
# Price existing residency without assuming an unload frees it: a stock
# load retains its arena, while dropping a slab barely changes MemAvailable.
_AS_HELD = (
    "This price was taken against what each box already holds: {named}. The "
    "quote prices the slot as it stands and does not model the drop, so a load "
    "that would replace a resident is priced with that resident still held."
)


class FleetCapacityError(StockLoadCapacityError):
    """At least one rank priced this load and cannot hold it."""


class FleetQuoteUnavailableError(RuntimeError):
    """The fleet could not be asked, so nothing about the load is known.

    It stays outside the stock-load capacity family: a liveness fault must never
    be recorded as a capacity verdict.
    """


# What the driver calls its own box when no cluster config names one: a local
# fleet runs every rank in the driver's process tree, so this is not a guess.
LOCAL_HOST_NAME = "this box"


@dataclass(frozen=True, slots=True)
class RankAnswer:
    """One rank's reply, or the fact that it did not give one."""

    rank: int
    host: str = ""
    quotes: tuple[RankPrice, ...] = ()
    silent: str = ""
    elapsed_s: float = 0.0
    # Configured host label for diagnostics when the rank did not reply.
    configured_host: str = ""

    def verdicts(self, verdict: str) -> list[RankPrice]:
        return [quote for quote in self.quotes if quote.verdict == verdict]

    @property
    def where(self) -> str:
        """Name the box, and say who named it.

        The rank's own report wins. The config name is the fallback and is
        marked as one, because a silent rank reported nothing and the card must
        not present the driver's name as the rank's own report.
        """
        if self.host:
            return f"rank {self.rank} (host {self.host})"
        if self.configured_host:
            return f"rank {self.rank} (host {self.configured_host} as configured)"
        return f"rank {self.rank} (host unnamed)"


def _configured_hosts(handle: Any, world: int, gpus: int) -> list[str]:
    """The driver's name for each rank's box, in rank order.

    Ranks fill hosts in blocks of ``gpus``, the order ``collect`` slices the
    worker mesh in, so rank ``i`` sits on host ``i // gpus``. No host list is
    a local fleet, where every rank is the driver's own box.
    """
    hosts = tuple(getattr(getattr(handle, "config", None), "hosts", None) or ())
    if not hosts:
        return [LOCAL_HOST_NAME] * world
    return [str(getattr(hosts[index // gpus], "name", "") or "")
            if index // gpus < len(hosts) else "" for index in range(world)]


def _block(quote: RankPrice) -> dict[str, Any]:
    """Return the measurements used for this row's verdict.

    Slab results carry ``slab_measured`` rather than ``measured``. Reading only
    the stock field would omit slab capacity values and sort refusals incorrectly.
    """
    block = quote.slab_measured if quote.use_slab else quote.measured
    return block if isinstance(block, dict) else {}


def _stated_headroom(quote: RankPrice) -> float | None:
    """This row's own headroom, or None when the row states none."""
    value = _block(quote).get("headroom_gib")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _headroom(quote: RankPrice) -> float:
    """The sort key. Only refuse and rescue rows reach it, and both priced."""
    value = _stated_headroom(quote)
    return value if value is not None else 0.0


def _headroom_text(quote: RankPrice) -> str:
    """Format reported headroom without inventing a measurement.

    Distinguish an unmeasured reuse from a probe that could not read
    MemAvailable; neither is zero headroom.
    """
    value = _stated_headroom(quote)
    if value is not None:
        return f"{value}GiB"
    if not quote.priced and quote.would_load == WOULD_LOAD_REUSE:
        return "resident"
    return "headroom unstated"


def _slab_fit(quote: RankPrice) -> SlabFit | None:
    """Rebuild the refusing rank's slab verdict from its reported measurements.

    Use ``slab_measured`` with the slab formatter so the driver does not recompute
    values already measured by the worker.
    """
    block = quote.slab_measured
    if not quote.use_slab or not isinstance(block, dict):
        return None
    try:
        avail = block["mem_available_bytes"]
        return SlabFit(
            fits=bool(block["fits"]), applies=bool(block["applies"]),
            size_bytes=int(block["checkpoint_bytes"]),
            avail_bytes=None if avail is None else int(avail),
            floor_bytes=int(block.get("floor_bytes") or 0), lora_bytes=int(block.get("lora_bytes") or 0))
    except (KeyError, TypeError, ValueError):
        return None


def _pairs(answers: list[RankAnswer], verdict: str) -> list[tuple[RankAnswer, RankPrice]]:
    """Every rank carrying ``verdict``, worst headroom first."""
    return sorted(((answer, quote) for answer in answers
                   for quote in answer.verdicts(verdict)),
                  key=lambda pair: _headroom(pair[1]))


def _priced_sentence(answer: RankAnswer, quote: RankPrice, unet_name: str) -> str:
    """The refusing rank's own numbers, formatted rather than re-derived."""
    measured = _block(quote)
    try:
        needed, weights, avail, headroom = (
            float(measured[key]) for key in
            ("required_gib", "weights_gib", "mem_available_gib", "headroom_gib"))
    except (KeyError, TypeError, ValueError):
        text = " ".join(str(quote.reason).split())
        if len(text) > _MAX_REASON_CHARS:
            text = text[:_MAX_REASON_CHARS] + "..."
        return f"{answer.where} refused: {text}."
    if measured.get("probe") == PROBE_PREFLIGHT_WALL:
        # Managed and FSDP estimates come from their own capacity checks; neither
        # charges the stock arena, so the sentence names no second term.
        kind = "comfy-managed" if quote.rung == RUNG_COMFY_MANAGED else "sharded"
        return (f"{answer.where} priced a {kind} load of {unet_name}: the load "
                f"needs {needed} GiB against a {weights} GiB file, and {avail} "
                f"GiB of unified memory is available, {abs(headroom)} GiB short.")
    if quote.use_slab:
        floor = gib(int(measured.get("floor_bytes") or 0))
        lora = gib(int(measured.get("lora_bytes") or 0))
        # The slab holds the checkpoint's own bytes, so naming the stock arena
        # here would quote a price this rung never charged.
        kind = "slab"
        second_term = f"plus a {floor} GiB floor for the rest of the box"
        if lora:
            second_term += f" and {lora} GiB for the LoRA stack's bake"
    else:
        kind, second_term = "stock", "plus the host copy it is placed from"
    return (
        f"{answer.where} priced a {kind} load of {unet_name}: the load needs "
        f"{needed} GiB, a {weights} GiB file {second_term}, and {avail} GiB of "
        f"unified memory is available, {abs(headroom)} GiB short."
    )


def _held_clause(refusing: list[tuple[RankAnswer, RankPrice]]) -> str:
    """Name the residents the prices were taken over, or say nothing."""
    named = []
    for answer, quote in refusing:
        if quote.would_load not in (WOULD_LOAD_SWAP, WOULD_LOAD_REUSE):
            continue
        word = quote.holds_resident.get(quote.slot)
        if not word:
            continue
        named.append(f"{answer.where} holds a {word} resident in the "
                     f"{quote.slot} slot and this load reads as a "
                     f"{quote.would_load}")
    return _AS_HELD.format(named="; ".join(named)) if named else ""


def _log_fleet(answers: list[RankAnswer], outcome: str) -> None:
    # A price logged only on a refusal leaves a fitting fleet
    # indistinguishable from a host where the price made no claim at all.
    rows = [f"rank {answer.rank}={quote.slot}:{quote.verdict}/{quote.rung}"
            f"@{_headroom_text(quote)}"
            for answer in answers for quote in answer.quotes]
    rows += [f"rank {answer.rank}={answer.silent or 'no rows'}"
             for answer in answers if answer.silent or not answer.quotes]
    safe_call(log.info, "cross-rank capacity %s: %s", outcome, ", ".join(rows))


def _refuse_unavailable(text: str) -> NoReturn:
    """The fleet could not be asked, so the load's price is unknown."""
    raise FleetQuoteUnavailableError(refusal(
        RefusalClass.CAPACITY, text, guard=GUARD, waivable=False,
        troubleshooting=TROUBLESHOOTING))


def decide(answers: list[RankAnswer], world: int, *,
           unet_name: str = "this checkpoint",
           deadline_s: float = CAPACITY_QUOTE_DEADLINE_S) -> None:
    """Evaluate fleet replies and raise on a definite refusal or invalid evidence.

    Check missing replies, incomplete or duplicated rank inventories, and
    inconsistent checkpoint sizes before capacity verdicts. ``unpriced`` and
    ``no_claim`` are logged but do not refuse: missing pricing support does not
    establish that a load cannot fit.
    """
    if world < 2:
        return
    silent = [answer for answer in answers if answer.silent]
    ranks = [answer.rank for answer in answers]
    sizes: dict[str, set[int]] = {}
    for answer in answers:
        for quote in answer.quotes:
            # Only rows that stated a size are compared. Zero is the absence of
            # a content-derived value, not a smaller artifact, and a rank that
            # made no claim or took a rung this release does not price would
            # otherwise read as a fleet that sees different bytes.
            if quote.checkpoint_bytes > 0:
                sizes.setdefault(quote.slot, set()).add(int(quote.checkpoint_bytes))
    disagree = {slot: sorted(seen) for slot, seen in sizes.items() if len(seen) > 1}
    if silent or len(answers) != world or len(set(ranks)) != len(ranks) or disagree:
        _log_fleet(answers, "unavailable")
        _refuse_unavailable(_unavailable_text(
            answers, silent, world, ranks, disagree, deadline_s))
    refusing = _pairs(answers, VERDICT_REFUSE)
    if refusing:
        _log_fleet(answers, "refused")
        _refuse_shortfall(refusing, answers, world, unet_name)
    rescuing = _pairs(answers, VERDICT_RESCUE)
    if rescuing:
        _log_fleet(answers, "rescue offered")
        _offer_rescue(*rescuing[0])
    _log_fleet(answers, "admitted")


def _unavailable_text(answers: list[RankAnswer], silent: list[RankAnswer],
                      world: int, ranks: list[int],
                      disagree: dict[str, list[int]], deadline_s: float) -> str:
    if silent:
        named = "; ".join(f"{answer.where} {answer.silent}" for answer in silent)
        heard = "; ".join(f"rank {answer.rank} answered in {answer.elapsed_s:.2f} s"
                          for answer in answers if not answer.silent)
        return (
            f"{_NEEDS_EVERY_RANK}, and {named} (deadline {deadline_s:.0f} s). "
            f"{heard or 'No other rank answered either'}. A rank that cannot "
            "answer a memory question in that time is starved or gone, and "
            "placing a checkpoint beside it is how a box stops answering at "
            f"all. {_NOTHING_LOADED}")
    if disagree:
        return (
            "this render needs every rank to price the same bytes, and the "
            f"ranks report different checkpoint file sizes ({disagree!r}, in "
            "bytes per slot), so the worker hosts hold different files under "
            f"the same checkpoint name. {_NOTHING_LOADED} Stage the same "
            "artifact on every host first.")
    return (
        f"{_NEEDS_EVERY_RANK}, and the fleet of {world} returned "
        f"{len(answers)} answers from ranks {sorted(ranks)}. A rank inventory "
        "that is short or duplicated cannot say what the leanest box holds. "
        f"{_NOTHING_LOADED}")


def _refuse_shortfall(refusing: list[tuple[RankAnswer, RankPrice]],
                      answers: list[RankAnswer], world: int,
                      unet_name: str) -> NoReturn:
    """At least one rank priced the load and cannot hold it."""
    short = {answer.rank for answer, _quote in refusing}
    blocker = next((quote.blocker[1] for _answer, quote in refusing
                    if quote.blocker), "")
    if not blocker:
        # A slab rung refuses with no blocker pair, because slab is the last
        # option and its capacity check offers no consent card. Its remedy is built
        # from the same fit, so the fleet card is never left with none.
        blocker = next((store_slab_admit.rescue_blocker(fit)[1]
                        for fit in (_slab_fit(quote) for _answer, quote in refusing)
                        if fit is not None and fit.applies and not fit.fits), "")
    held = _held_clause(refusing)
    fitting = [f"{answer.where} took the {quote.rung} rung" if quote.rung
               else f"{answer.where} made no claim"
               for answer in answers for quote in answer.quotes
               if quote.verdict not in (VERDICT_REFUSE, VERDICT_RESCUE)]
    raise FleetCapacityError(refusal(
        RefusalClass.CAPACITY,
        f"this render needs every rank to hold the model and {len(short)} of "
        f"{world} cannot, so nothing was loaded on any rank. "
        + " ".join(_priced_sentence(answer, quote, unet_name)
                   for answer, quote in refusing)
        + (f" {'; '.join(fitting)}." if fitting else "")
        + " The fleet takes the leanest answer, because the load runs everywhere."
        + (f" {held}" if held else "")
        + (f" What would fit: {blocker}." if blocker else "")
        + " This refusal is capacity-protective (class C), and nothing was "
          "quarantined.",
        guard=GUARD, waivable=False, troubleshooting=TROUBLESHOOTING))


def _offer_rescue(answer: RankAnswer, quote: RankPrice) -> NoReturn:
    """Raise the existing offer earlier, with the leanest rank's numbers.

    Ranks agree on the rank-invariant half of a descriptor and never on the
    measured half: the host name and every availability figure are per box, so
    an equality check across ranks would refuse every world-2 rescue, the
    remedy class C capacity refusals rely on. The offer carries one rank's
    descriptor whole.
    """
    from . import consent_descriptor
    from .consent_kinds import KIND_SPECS

    descriptor = _descriptor(quote)
    if descriptor is None:
        _refuse_unavailable(
            f"{answer.where} offered a capacity rescue this driver could not "
            f"read, so the offer cannot be registered. {_NOTHING_LOADED}")
    spec = KIND_SPECS[consent_descriptor.KIND_RESCUE_SLAB]
    tail = []
    for build, note in ((lambda: measured_tag(descriptor.measured), "measured numbers"),
                        (lambda: consent_descriptor.encode(descriptor), "descriptor")):
        try:
            tail.append(build())
        except (TypeError, ValueError) as exc:
            safe_call(log.warning, "cross-rank rescue offer dropped its %s (%s)",
                      note, failure_summary(exc))
    raise SlabResidencyRescueOffer(refusal(
        RefusalClass.CAPACITY,
        f"capacity rescue available for {descriptor.unet_name}: "
        f"{descriptor.human_reason} Every rank must hold this checkpoint and "
        f"{answer.where} is the leanest, so its price governs the fleet. "
        "Nothing was loaded and nothing was quarantined.",
        guard=RESCUE_GUARD, waivable=True, troubleshooting=TROUBLESHOOTING,
        panel_action=PanelAction(label=spec.primary_label, env=spec.env_var),
    ) + (("\n" + "\n".join(tail)) if tail else ""))


def _descriptor(quote: RankPrice) -> ConsentDescriptor | None:
    """Rebuild the worker's descriptor from the row that carried it."""
    public = quote.descriptor
    if isinstance(public, ConsentDescriptor):
        return public
    if not isinstance(public, dict):
        return None
    try:
        return ConsentDescriptor(**public)
    except (TypeError, ValueError):
        return None


def _answer_from(rank: int, reply: Any, elapsed: float) -> RankAnswer:
    if not isinstance(reply, dict):
        return RankAnswer(rank, silent="returned a reply this driver cannot read",
                          elapsed_s=elapsed)
    host = str(reply.get("host") or "")
    reported = reply.get("rank")
    if isinstance(reported, int) and not isinstance(reported, bool) and reported != rank:
        # The driver knows which rank it sent to and keeps that name: a reply
        # that renames itself makes one rank's answer stand for another's.
        return RankAnswer(rank, host, elapsed_s=elapsed,
                          silent=f"answered as rank {reported}")
    rows = reply.get("quotes")
    if not isinstance(rows, list) or not rows:
        # ``decide`` counts answers, not rows: an empty inventory priced nothing.
        return RankAnswer(rank, host, silent="returned no quote rows", elapsed_s=elapsed)
    try:
        quotes = tuple(RankPrice.from_row(row) for row in rows)
    except (ValueError, TypeError):
        return RankAnswer(rank, host, elapsed_s=elapsed,
                          silent="returned a quote row this driver cannot read")
    return RankAnswer(rank, host, quotes, elapsed_s=elapsed)


def collect(handle: Any, request: dict, world: int, *,
            deadline_s: float = CAPACITY_QUOTE_DEADLINE_S) -> list[RankAnswer]:
    """Send every rank a quote request and collect replies under one deadline.

    Per-rank futures identify the rank that timed out; an all-rank ValueMesh
    would report only an aggregate failure. Send all requests before waiting so
    the total wait is not multiplied by world size.
    """
    from . import mesh_setup

    mesh_setup.require_endpoint(handle, "capacity_quote")
    gpus = int(getattr(handle, "gpus_per_host", 1) or 1)
    labels = list(handle.workers.extent.labels)
    futures = []
    for index in range(world):
        coords: dict[str, int] = {"gpus": index % gpus}
        if "hosts" in labels:
            coords["hosts"] = index // gpus
        futures.append(
            (index, handle.workers.slice(**coords).capacity_quote.call_one(request)))
    started = time.monotonic()
    answers = []
    for index, future in futures:
        asked = time.monotonic()
        left = deadline_s - (asked - started)
        if left <= 0.0:
            answers.append(RankAnswer(index, silent="did not answer in time"))
            continue
        try:
            reply = handle._await_or_evict(future, left)
        except TimeoutError:
            answers.append(RankAnswer(index, silent="did not answer in time"))
        except BaseException as exc:
            # The remote text is not repeated: an actor error stringifies the
            # whole remote traceback, and no card names another rank's internals.
            answers.append(RankAnswer(
                index, silent=f"failed to answer ({type(exc).__name__})"))
        else:
            # Measure this rank's wait, not cumulative collection time: all futures
            # start together but are read in order.
            answers.append(_answer_from(index, reply, time.monotonic() - asked))
    # A silent rank never names its own box, so the driver's own list goes on
    # every row here rather than at four raise sites.
    named = _configured_hosts(handle, world, gpus)
    return [replace(answer, configured_host=named[answer.rank])
            if answer.rank < len(named) else answer for answer in answers]


def quote_request(specs: list[dict], rescue_consent: dict | None = None) -> dict:
    """The narrow request the endpoint prices, with no latent riding along."""
    return {
        "specs": [{"unet_name": spec.get("unet_name"), "options": spec.get("options"),
                   "loras": spec.get("loras"), "slot": spec.get("slot", "cond")}
                  for spec in specs],
        "rescue_consent": rescue_consent,
    }


def agree_sample(handle: Any, request: dict) -> None:
    """Agree the render's checkpoints before parity and dispatch."""
    from .actor.worker_authorization import sample_rescue_consent

    model = request.get("model")
    if not isinstance(model, dict):
        return
    specs = [dict(model, slot="cond")]
    uncond = request.get("uncond_model")
    if isinstance(uncond, dict):
        specs.append(dict(uncond, slot="uncond"))
    agree_or_refuse(
        handle, quote_request(specs, sample_rescue_consent(request)),
        unet_name=str(model.get("unet_name") or "this checkpoint"))


def agree_load(handle: Any, unet_name: str, options: dict | None,
               lora_stack: list | None, slot: str) -> None:
    """Agree one eager loader-node load before any rank allocates."""
    agree_or_refuse(
        handle,
        quote_request([{"unet_name": unet_name, "options": options,
                        "loras": lora_stack, "slot": slot}]),
        unet_name=unet_name)


def agree_or_refuse(handle: Any, request: dict, *,
                    unet_name: str = "this checkpoint",
                    deadline_s: float = CAPACITY_QUOTE_DEADLINE_S) -> None:
    """Agree the load across the fleet, or raise before any rank allocates.

    World 1 is a no op, the way the post-load weight-residency guard already is.
    A refusal or a timeout evicts nothing and latches nothing: the quote mutates
    no state, so an unanswered one leaves exactly what an artifact parity
    failure at the same line leaves.
    """
    world = int(getattr(handle, "world", 0) or 0)
    if world < 2 or getattr(handle, "workers", None) is None:
        return
    answers = collect(handle, request, world, deadline_s=deadline_s)
    try:
        decide(answers, world, unet_name=unet_name, deadline_s=deadline_s)
    except BaseException as exc:
        # The one driver frame that sees this raise, so the rescue arm becomes a
        # panel card here. Best effort, and it never replaces the refusal.
        from .consent_observe import observe_refusal

        try:
            observe_refusal(exc)
        except BaseException as observe_exc:
            safe_note(exc, "cross-rank capacity offer observation failed", observe_exc)
        raise
