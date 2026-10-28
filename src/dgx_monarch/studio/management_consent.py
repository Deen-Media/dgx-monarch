"""Review a consent accept against the driver's live pending consent requests.

The driver issues each pending request with a key and an id, and lists them at
GET /dgxm/consents. The review reads that list, so the owner can accept a
request from Studio without copying identities out of the ComfyUI panel.
Studio offers accept only; dismiss, revoke and the automatic rescue switch stay
in the panel.
"""
from __future__ import annotations

import re
from typing import Any

from .contracts import StudioError

PENDING_LIMIT = 64
IDENTITY_BYTES = 512
# The consent id lets a reviewer match a listed request to the job whose refusal asked for it.
_SHOWN = ("kind", "class", "title", "artifact", "risk", "occurrences", "seen_at", "consent_id")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def _identity(value: Any) -> str | None:
    if not isinstance(value, str) or not value or len(value) > IDENTITY_BYTES or _CONTROL.search(value):
        return None
    return value


def _shown(value: Any) -> Any:
    if isinstance(value, str):
        return _CONTROL.sub(" ", value)[:240]
    if value is None or type(value) in {bool, int, float}:
        return value
    return None


def pending_cards(state: Any) -> list[dict[str, Any]]:
    """The pending requests the driver listed, with exact identities and bounded text."""
    pending = state.get("pending") if isinstance(state, dict) else None
    if not isinstance(pending, list):
        raise StudioError("consent_state", "The driver did not return its pending consent requests; reconcile the driver first.", 502)
    cards = []
    for card in pending[:PENDING_LIMIT]:
        if not isinstance(card, dict):
            continue
        key, pending_id = _identity(card.get("key")), _identity(card.get("id"))
        if key and pending_id:
            cards.append({"key": key, "id": pending_id, **{name: _shown(card.get(name)) for name in _SHOWN if name in card}})
    return cards


def consent_plan(parameters: dict[str, Any], state: Any) -> tuple[dict[str, str], dict[str, Any]]:
    """Bind the review to one exact pending request.

    With both fields empty and exactly one request pending, the review selects
    it. Otherwise the plan lists every pending request so the owner can enter
    the key and id of the one to accept, and the review stays blocked.
    """
    key, pending_id = parameters.get("key", ""), parameters.get("id", "")
    if not isinstance(key, str) or not isinstance(pending_id, str) or len(key) > IDENTITY_BYTES or len(pending_id) > IDENTITY_BYTES:
        raise StudioError("operation_fields", "An exact pending consent key and id are required.")
    if _CONTROL.search(key + pending_id) or bool(key) != bool(pending_id):
        raise StudioError("operation_fields", "Enter both the pending consent key and id, or leave both empty to list pending requests.")
    cards = pending_cards(state)
    scope = "Accept exactly this pending consent request; the driver checks it again when the accept arrives."
    if key:
        chosen = next((card for card in cards if card["key"] == key and card["id"] == pending_id), None)
        if chosen is None:
            return {"key": key, "id": pending_id}, {
                "ready": False, "pending": cards,
                "refusal": "That key and id are not a pending consent request; queue the render again or pick one listed here."}
    elif len(cards) == 1:
        chosen = cards[0]
    else:
        return {"key": "", "id": ""}, {
            "ready": False, "pending": cards,
            "refusal": ("No consent request is pending; queue the render that asked for consent again." if not cards else
                        "Several consent requests are pending; enter the key and id of the one to accept, then review again.")}
    return {"key": chosen["key"], "id": chosen["id"]}, {"ready": True, "card": chosen, "pending_count": len(cards), "scope": scope}
