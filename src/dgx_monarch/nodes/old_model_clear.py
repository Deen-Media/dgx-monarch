"""The "Old model in the way" card, its Clear and retry, and Always clear and retry.

``old_model.py`` decides whether a clear would make room and words the
refusal. This module raises the card, runs its accept, keeps the Always toggle
and runs the Always flow. Every clear here is the panel's Reset attached mesh
(``recycle_drain.recycle_and_drain``); there is no second teardown, and every
successful reset answers the open cards (``recycle_drain.drop_residency_memos``).

The card is an action kind (``consent_kinds.KindSpec.action``): its accept
grants nothing, writes no memo and no gate-ledger row, and bypasses no guard.
A card is bound to the attached mesh that refused, so accepting it never resets
a fleet that came later. The load is priced again on the fresh fleet and still
refuses if it does not fit. Always clear and retry is the memo flag
``auto_clear`` or, headless, the driver's ``DGXM_ALLOW_CLEAR_RETRY``. With it
on, a refusal a clear would answer makes the driver clear once, under the reset
route's single-flight gate, wait for the driver host's memory to settle, and
retry once; the retry sits outside the catch, so a second refusal is plain.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, NoReturn

from .. import consent_pending, consent_store, mesh_safety, old_model
from ..consent_descriptor import ENV_CLEAR_RETRY, KIND_CLEAR_RETRY
from ..log import get_logger
from ..refusal import parse_leading_refusal_tag
from . import recycle_drain

log = get_logger(__name__)

FLAG = "auto_clear"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
# After the actors exit, the driver host's MemAvailable climbs for a few
# seconds (2026-11-04: 12.5 to 97.5 GiB within five). Settled is two readings
# half a second apart within 256 MiB, or 15 s, whichever comes first.
SETTLE_TIMEOUT_S = 15.0
SETTLE_POLL_S = 0.5
SETTLE_STEP_BYTES = 256 << 20
# The loader preflight's question to the driver host's Workers is advisory and
# asked only after its own estimate refused, so it waits less than a load does.
LOADER_QUOTE_DEADLINE_S = 10.0
NOTHING_TO_CLEAR = "Nothing was loaded any more; queue the render again."
GONE = "The old model is no longer loaded; queue the render again."



def cleared(offer: old_model.ClearOffer | None) -> str:
    """The accept's answer: 'Cleared both Sparks; queue the render again.'"""
    scope = old_model.scope_words(offer, cleared=True) if offer else "the Sparks"
    return f"Cleared {scope}; queue the render again."


def always_env() -> bool:
    """The headless Always clear and retry, read on the driver."""
    return str(os.environ.get(ENV_CLEAR_RETRY) or "").strip().lower() in _TRUTHY


def always_on() -> bool:
    """The toggle or the env var. A store fault reads as off, never as on."""
    if always_env():
        return True
    try:
        return consent_store.standing(FLAG)
    except Exception as exc:
        log.warning("Always clear and retry could not be read, so it is off (%r)", exc)
        return False


def _local_path(unet_name: str) -> str | None:
    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", unet_name)
    return str(path) if path else None


def card_possible(unet_name: str) -> bool:
    """Whether this driver can raise a card for the checkpoint: it resolves it."""
    try:
        return _local_path(unet_name) is not None
    except Exception:
        return False


def register_card(offer: old_model.ClearOffer, guard: str,
                  fleet: Any = None) -> dict[str, Any] | None:
    """Publish the card for a refusal; best effort, never in place of the refusal."""
    try:
        path = _local_path(offer.unet_name)
        if not path:
            log.warning("no Old model in the way card for %s: this driver cannot "
                        "resolve it in its own model folders", offer.unet_name)
            return None
        fleet = offer.fleet if fleet is None else fleet
        card = consent_pending.register_pending({
            "kind": KIND_CLEAR_RETRY, "path": path,
            "memo_context": {"unet_name": offer.unet_name},
            "artifact": offer.new_model, "unet_name": offer.unet_name,
            "target_guard": guard, "numbers": old_model.card_numbers(offer),
            "detail": old_model.card_body(offer),
        })
        entry = (consent_pending.peek_pending(str(card["key"]), str(card["id"]))
                 if card is not None else None)
        if entry is not None:
            # On the pending entry, so the binding goes wherever the card goes.
            entry.binding = (fleet, offer)
        return card
    except Exception as exc:
        log.warning("Old model in the way card not raised for %s (%r)", offer.unet_name, exc)
        return None


def observe(exc: BaseException) -> dict[str, Any] | None:
    """Raise the card for an agreement refusal a clear would answer, or nothing."""
    offer = old_model.offer_of(exc)
    if offer is None:
        return None
    tag = parse_leading_refusal_tag(str(exc))
    return register_card(offer, (tag.guard if tag and tag.guard else old_model.FLEET_GUARD),
                         getattr(exc, "fleet", None))


def drop_cards() -> int:
    """Take every open card of this kind: a reset just answered them.

    Each card's fleet binding lives on its pending entry and goes with it, as
    it does on a dismissal, an expiry or the card limit.
    """
    dropped = 0
    for card in consent_pending.pending_cards():
        if card.get("kind") == KIND_CLEAR_RETRY and consent_pending.take_pending(
                str(card["key"]), str(card["id"])) is not None:
            dropped += 1
    return dropped


def _binding(key: str, pending_id: str) -> tuple[Any, old_model.ClearOffer | None]:
    """The fleet and offer a pending card is bound to, read with the card itself."""
    entry = consent_pending.peek_pending(key, pending_id)
    binding = getattr(entry, "binding", None) if entry is not None else None
    return binding if isinstance(binding, tuple) and len(binding) == 2 else (None, None)


def bound_fleet(key: str) -> Any:
    """The fleet the open card with this key would reset, or None."""
    for card in consent_pending.pending_cards():
        if str(card["key"]) == key:
            return _binding(key, str(card["id"]))[0]
    return None


def is_clear_card(key: object, pending_id: object) -> bool:
    if not isinstance(key, str) or not isinstance(pending_id, str):
        return False
    entry = consent_pending.peek_pending(key, pending_id)
    return entry is not None and entry.descriptor.kind == KIND_CLEAR_RETRY


def set_always(value: object, source: str = "panel", *,
               answer_cards: bool = True) -> tuple[int, dict[str, Any]]:
    """Write the Always toggle. It authorizes no bypass, so it writes no ledger row."""
    if not isinstance(value, bool):
        return 400, {"ok": False, "detail": "auto_clear needs a boolean value"}
    try:
        consent_store.set_standing(FLAG, value)
    except consent_store.ConsentStoreError as exc:
        return 503, {"ok": False, "detail": f"the toggle could not be saved: {exc}"}
    # On, the next refusal clears by itself, so the open cards are answered.
    answered = drop_cards() if value and answer_cards else 0
    log.info("Always clear and retry set to %s from the %s (%d card(s) cleared)",
             value, source, answered)
    return 200, {"ok": True, "auto_clear": value, "cleared": answered}


def accept(key: str, pending_id: str, body: dict[str, Any],
           source: str = "panel") -> tuple[int, dict[str, Any]]:
    """Clear and retry: the recycle route's own teardown, under its single-flight gate.

    Only the fleet that refused is reset. When that fleet is no longer
    attached, or the card carries no binding to one, the old model is not
    known to be there, so the card is answered and nothing is reset.
    """
    from . import routes

    if not is_clear_card(key, pending_id):
        return 409, {"ok": False,
                     "detail": "no pending card for that request; queue the render again"}
    fleet, offer = _binding(key, pending_id)
    always = body.get("always") is True
    if always:
        status, payload = set_always(True, source, answer_cards=False)
        if status != 200:
            return status, payload
    if fleet is None:
        consent_pending.take_pending(key, pending_id)
        return 200, {"ok": True, "cleared": False, "always": always, "detail": GONE}
    if not routes._begin_recycle_request():
        return 429, {"ok": False, "detail": ("another reset of the Sparks is running or "
                                             "just finished; wait, then click again")}
    try:
        # Read again under the gate: a toggle or a reset may have answered the
        # card meanwhile, and an answered card resets nothing.
        if _binding(key, pending_id)[0] is not fleet:
            result, status = {"ok": False, "status": "fleet_changed"}, 200
        else:
            result, status = recycle_drain.recycle_attached_mesh(only=fleet)
    finally:
        routes.invalidate_telemetry_caches()
        routes._finish_recycle_request()
    outcome = result.get("status")
    if not result.get("ok") and outcome not in ("no_live_mesh", "fleet_changed"):
        return status, {"ok": False, "recycle": result,
                        "detail": f"the Sparks were not cleared: {result.get('detail')}"}
    consent_pending.take_pending(key, pending_id)
    detail = (cleared(offer) if result.get("ok") else
              GONE if outcome == "fleet_changed" else NOTHING_TO_CLEAR)
    log.info("Clear and retry from the %s: %s", source, result.get("detail"))
    return 200, {"ok": True, "cleared": bool(result.get("ok")), "always": always,
                 "recycle": result, "detail": detail}


def wait_settled(read: Callable[[], int | None] = mesh_safety.mem_available_bytes,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> int | None:
    """Wait until the driver host's MemAvailable stops climbing; the last reading."""
    deadline = clock() + SETTLE_TIMEOUT_S
    last = read()
    while last is not None and clock() < deadline:
        sleep(SETTLE_POLL_S)
        now = read()
        if now is None or abs(now - last) < SETTLE_STEP_BYTES:
            return now
        last = now
    return last


def clear_for_retry(handle: Any, exc: BaseException, *, inherited: bool = False) -> bool:
    """Always clear and retry: clear once for this load; True when the caller may retry.

    Never under a session pinned to its mesh (a Pipeline or Gate render owns
    it), never for a refusal a clear would not answer, and never when the
    reset route's single-flight gate is taken or the reset is refused: then
    the caller re-raises the original refusal, and the card it names stays
    open. A reset that runs answers every card through
    ``recycle_drain.drop_residency_memos`` before the gate opens again, and a
    click on a card after that meets its fleet binding and resets nothing.
    """
    from . import routes

    offer = old_model.offer_of(exc)
    # The fleet the refusal came from: a cached MeshSpec may still hold a
    # retired handle while a fresh fleet served this load.
    handle = getattr(exc, "fleet", None) or handle
    if offer is None or inherited or handle is None or not always_on():
        return False
    if not routes._begin_recycle_request():
        log.warning("Always clear and retry found another reset running; the refusal stands")
        return False
    try:
        outcome = recycle_drain.recycle_and_drain(handle)
    except Exception as clear_exc:
        log.warning("Always clear and retry could not clear the Sparks (%r); "
                    "the refusal stands", clear_exc)
        return False
    finally:
        routes.invalidate_telemetry_caches()
        routes._finish_recycle_request()
    if not outcome.ok:
        log.warning("Always clear and retry could not clear the Sparks (%s: %s); "
                    "the refusal stands", getattr(outcome.status, "value", outcome.status),
                    outcome.detail)
        return False
    wait_settled()
    log.info("%s", old_model.log_line(offer))
    return True


def _worst(windows: Iterable[Any]) -> Any:
    """The failing window short by the most, against its unclamped budget.

    The loader's own test (``graph_load_price.fits_box``), so the card and the
    refusal agree: a window with no DiT left to place is short when its driver
    models are, and a clear that makes room there still raises the card.
    """
    from .graph_load_price import fits_box

    short = [window for window in windows if not fits_box(window)]
    return max(short, key=lambda window: window.shortfall) if short else None


@dataclass(frozen=True, slots=True)
class HostAnswers:
    """What the Workers on the driver host answered, and the fleet they are on."""

    mine: tuple[Any, ...]
    handle: Any
    hosts: int
    local: bool


def driver_host_answers(mesh: Any, unet_name: str, options: Any,
                        slot: str = "", loras: Any = None) -> HostAnswers | None:
    """Ask the Workers on the driver host what they hold and what this load would do.

    Read-only and best effort: only a live, set-up fleet whose rank 0 shares
    the driver host is asked, through the agreement's own quote, and any fault
    means no answer. ``slot`` is the slot the load goes into, "cond" where the
    caller does not know it, and ``loras`` its stack where the caller has one (the
    loader node has none: the LoRA loaders run after it).
    """
    from .. import capacity_agreement, driver_footprint

    # The fleet attached now: after a clear, the MeshSpec the Init node cached
    # still holds the retired handle while a fresh fleet serves the loads.
    handle = recycle_drain.attached_fleet(getattr(mesh, "handle", None))
    if (handle is None or getattr(handle, "defunct", False)
            or getattr(handle, "setup_key", None) is None
            or getattr(handle, "workers", None) is None
            or not driver_footprint.rank0_co_resident(mesh)):
        return None
    try:
        world = int(getattr(handle, "world", 0) or 0)
        answers = capacity_agreement.collect(
            handle, capacity_agreement.quote_request(
                [{"unet_name": unet_name, "options": options, "loras": loras or [],
                  "slot": slot or "cond"}]),
            world, deadline_s=LOADER_QUOTE_DEADLINE_S)
    except Exception as exc:
        log.info("loader preflight could not ask the driver host's Workers what they "
                 "hold (%r); its own refusal stands", exc)
        return None
    hosts = tuple(getattr(getattr(handle, "config", None), "hosts", None) or ())
    local = getattr(handle, "owns_hosts", True) is False or not hosts
    gpus = int(getattr(handle, "gpus_per_host", 1) or 1)
    mine = tuple(answer for answer in answers if answer.quotes and not answer.silent
                 and (local or answer.rank // gpus == 0))
    return HostAnswers(mine, handle, 1 if local else len(hosts), local) if mine else None


UNASKED: Any = object()


def loader_offer(mesh: Any, unet_name: str, options: Any, *, windows: Iterable[Any],
                 keep: Iterable[str] = (), window: str = "", asked: Any = UNASKED,
                 credit: Any = None) -> old_model.ClearOffer | None:
    """What a clear of the attached mesh would give back, after the loader refused.

    The Workers on the driver host say what they hold (``driver_host_answers``,
    or ``asked`` where the loader already asked them). The box is short by the
    worst failing window's unclamped shortfall
    (``DriverFootprintEstimate.shortfall``): a clear must make room in both
    windows. Slots this load reuses, and models this graph loads (``keep``),
    are not old models. ``window`` names the worst window, the one the
    loader's own refusal leads with. ``windows`` are the uncredited ones: a
    cleared box holds no copy for a Worker to drop, so where the loader's price
    already counted a swap ``credit`` and still refused, the offer names that
    model as dropped and what a clear returns beyond it.
    """
    worst = _worst(windows)
    if worst is None:
        return None
    if asked is UNASKED:
        asked = driver_host_answers(mesh, unet_name, options)
    if asked is None:
        return None
    kept_names = frozenset(str(name) for name in keep) | {unet_name}
    names: list[str] = []
    gain = 0
    for answer in asked.mine:
        kept = old_model.kept_slots(answer.quotes, kept_names)
        names.extend(old_model.old_models(answer.quotes[0], kept))
        gain += old_model.rank_gain(answer.quotes[0], kept)
    # The agreement named the box (box_names.py): rank 0 shares the driver
    # host here, so it is the head, or this box on a local fleet.
    box = old_model.box_name(asked.mine[0])
    required = int(worst.projected) + int(worst.reserve)
    dropped = (old_model.display_name(credit.dropped),) if credit else ()
    return old_model.box_offer(
        unet_name, old_model.ShortBox(
            box, tuple(dict.fromkeys(names)), required, int(worst.mem_available), gain,
            dropped=tuple(name for name in dropped if name in names),
            left_bytes=max(gain - int(credit.bytes), 0) if credit else 0, window=window),
        hosts=asked.hosts, local=asked.local, fleet=asked.handle)


def refuse_at_loader(offer: old_model.ClearOffer, message: str) -> NoReturn:
    """Raise the card and the loader's refusal when a clear would make room.

    Both are bound to the fleet the facts came from (``offer.fleet``).
    """
    card = register_card(offer, old_model.LOADER_GUARD)
    old_model.raise_loader_refusal(offer, message, card=card is not None)


def held_text(offer: old_model.ClearOffer) -> str:
    """What the loader's ordinary refusal adds when a clear would not make room."""
    return old_model.held_clause((), offer)
