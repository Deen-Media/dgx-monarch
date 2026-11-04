"""The "Old model in the way" card, its Clear and retry, and Always clear and retry.

``old_model.py`` decides whether a clear would make room and words the
refusal. This module raises the card, runs its accept, keeps the Always toggle
and runs the Always flow. Every clear here is the panel's Reset attached mesh
(``recycle_drain.recycle_and_drain``); there is no second teardown.

The card is an action kind (``consent_kinds.KindSpec.action``): its accept
grants nothing, writes no memo and no gate-ledger row, and bypasses no guard.
The load is priced again on the fresh fleet and still refuses if it does not
fit. Always clear and retry is the memo flag ``auto_clear`` or, headless, the
driver's ``DGXM_ALLOW_CLEAR_RETRY``. With it on, a refusal a clear would answer
makes the driver clear once, wait for the driver host's memory to settle, and
retry once; the retry sits outside the catch, so a second refusal is plain.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
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
CLEARED = "Cleared the attached mesh; then queue again."
NOTHING_TO_CLEAR = "The attached mesh was already gone; then queue again."


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


def register_card(offer: old_model.ClearOffer, guard: str) -> dict[str, Any] | None:
    """Publish the card for a refusal; best effort, never in place of the refusal."""
    try:
        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", offer.unet_name)
        if not path:
            log.warning("no Old model in the way card for %s: this driver cannot "
                        "resolve it in its own model folders", offer.unet_name)
            return None
        return consent_pending.register_pending({
            "kind": KIND_CLEAR_RETRY, "path": str(path),
            "memo_context": {"unet_name": offer.unet_name},
            "artifact": offer.new_model, "unet_name": offer.unet_name,
            "target_guard": guard, "numbers": old_model.card_numbers(offer),
            "detail": old_model.card_body(offer),
        })
    except Exception as exc:
        log.warning("Old model in the way card not raised for %s (%r)", offer.unet_name, exc)
        return None


def observe(exc: BaseException) -> dict[str, Any] | None:
    """Raise the card for an agreement refusal a clear would answer, or nothing."""
    offer = old_model.offer_of(exc)
    if offer is None:
        return None
    tag = parse_leading_refusal_tag(str(exc))
    return register_card(offer, (tag.guard if tag and tag.guard else old_model.FLEET_GUARD))


def drop_cards() -> int:
    """Take every open card of this kind: a clear just answered them."""
    dropped = 0
    for card in consent_pending.pending_cards():
        if card.get("kind") == KIND_CLEAR_RETRY and consent_pending.take_pending(
                str(card["key"]), str(card["id"])) is not None:
            dropped += 1
    return dropped


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
    cleared = drop_cards() if value and answer_cards else 0
    log.info("Always clear and retry set to %s from the %s (%d card(s) cleared)",
             value, source, cleared)
    return 200, {"ok": True, "auto_clear": value, "cleared": cleared}


def accept(key: str, pending_id: str, body: dict[str, Any],
           source: str = "panel") -> tuple[int, dict[str, Any]]:
    """Clear and retry: the recycle route's own teardown, under its single-flight gate."""
    from . import routes

    if not is_clear_card(key, pending_id):
        return 409, {"ok": False,
                     "detail": "no pending card for that request; queue the render again"}
    always = body.get("always") is True
    if always:
        status, payload = set_always(True, source, answer_cards=False)
        if status != 200:
            return status, payload
    if not routes._begin_recycle_request():
        return 429, {"ok": False, "detail": ("another attached-mesh reset is active or "
                                             "just settled; wait, then click again")}
    try:
        result, status = recycle_drain.recycle_attached_mesh()
    finally:
        routes.invalidate_telemetry_caches()
        routes._finish_recycle_request()
    gone = result.get("status") == "no_live_mesh"
    if not result.get("ok") and not gone:
        return status, {"ok": False, "recycle": result,
                        "detail": f"the attached mesh was not cleared: {result.get('detail')}"}
    consent_pending.take_pending(key, pending_id)
    log.info("Clear and retry from the %s: %s", source, result.get("detail"))
    return 200, {"ok": True, "cleared": not gone, "always": always, "recycle": result,
                 "detail": NOTHING_TO_CLEAR if gone else CLEARED}


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
    reset is refused: then the caller re-raises the original refusal.
    """
    offer = old_model.offer_of(exc)
    if offer is None or inherited or handle is None or not always_on():
        return False
    try:
        outcome = recycle_drain.recycle_and_drain(handle)
    except Exception as clear_exc:
        log.warning("Always clear and retry could not clear the attached mesh (%r); "
                    "the refusal stands", clear_exc)
        return False
    if not outcome.ok:
        log.warning("Always clear and retry could not clear the attached mesh (%s: %s); "
                    "the refusal stands", getattr(outcome.status, "value", outcome.status),
                    outcome.detail)
        return False
    wait_settled()
    drop_cards()
    log.info("%s", old_model.log_line(offer))
    return True


def loader_offer(mesh: Any, unet_name: str, options: Any, *, required: int,
                 available: int) -> old_model.ClearOffer | None:
    """Ask the Workers on the driver host what they hold, after the loader refused.

    Read-only and best effort: only a live, set-up fleet whose rank 0 shares
    the driver host is asked, through the agreement's own quote, and any fault
    means no card and the loader's own refusal.
    """
    from .. import capacity_agreement, driver_footprint

    handle = getattr(mesh, "handle", None)
    if (handle is None or getattr(handle, "defunct", False)
            or getattr(handle, "setup_key", None) is None
            or getattr(handle, "workers", None) is None
            or not driver_footprint.rank0_co_resident(mesh)):
        return None
    try:
        world = int(getattr(handle, "world", 0) or 0)
        answers = capacity_agreement.collect(
            handle, capacity_agreement.quote_request(
                [{"unet_name": unet_name, "options": options, "loras": [], "slot": "cond"}]),
            world, deadline_s=LOADER_QUOTE_DEADLINE_S)
    except Exception as exc:
        log.info("loader preflight could not ask the driver host's Workers what they "
                 "hold (%r); its own refusal stands", exc)
        return None
    hosts = tuple(getattr(getattr(handle, "config", None), "hosts", None) or ())
    local = getattr(handle, "owns_hosts", True) is False or not hosts
    gpus = int(getattr(handle, "gpus_per_host", 1) or 1)
    mine = [answer for answer in answers if answer.quotes and not answer.silent
            and (local or answer.rank // gpus == 0)
            and answer.quotes[0].would_load != "reuse"]
    if not mine:
        return None
    names = tuple(dict.fromkeys(
        name for answer in mine for name in old_model.old_models(answer.quotes[0])))
    gain = sum(old_model.rank_gain(answer.quotes[0]) for answer in mine)
    box = old_model.LOCAL_BOX if local else old_model.box_name(mine[0])
    return old_model.box_offer(
        unet_name, old_model.ShortBox(box, names, int(required), int(available), gain),
        hosts=1 if local else len(hosts), local=local)


def refuse_at_loader(offer: old_model.ClearOffer, message: str) -> NoReturn:
    """Raise the card and the loader's refusal when a clear would make room."""
    register_card(offer, old_model.LOADER_GUARD)
    old_model.raise_loader_refusal(offer, message)


def held_text(offer: old_model.ClearOffer) -> str:
    """What the loader's ordinary refusal adds when a clear would not make room."""
    return old_model.held_clause((), offer)
