"""Consent kinds, volatile pending invitations, and panel cards.

Comfy cannot pause an active render, so a consent point fails with a typed
refusal while the driver publishes a card. Python defines the complete card;
the browser only renders it. Pending invitations remain process-local and are
never persisted. Durable grants live in the memo; the ledger records audit
evidence.

Resolution checks an exact memo, then an eligible standing auto-rescue, then
the kind-specific environment fallback. Environment grants never create memos;
the bypass site still writes the required audit row.
"""
from __future__ import annotations

import os
import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from . import accuracy_waiver, consent_store
from .consent_kinds import KIND_SPECS, KindSpec, consent_context
from .gate_audit_vocab import check_guard, new_consent_id
from .log import get_logger

# Re-export these names so refusal sites use one import. ``refusal.refusal`` is
# the sole source of the user-facing action sentence.
__all__ = [
    "KIND_SPECS",
    "ConsentDescriptor",
    "ConsentGrant",
    "KindSpec",
    "PendingConsent",
    "consent_context",
    "describe",
    "grant_from_wire",
    "grant_wire",
    "pending_cards",
    "register_pending",
    "resolve",
]

log = get_logger(__name__)

PENDING_LIMIT = 32
PENDING_TTL_S = 30 * 60.0
AUTO_RESCUE_ENV = "DGXM_AUTO_RESCUE"
# Hex width required by the permanent audit-row schema.
CONSENT_ID_HEX = 32
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class ConsentDescriptor:
    """Consent data shared by the refusal, panel endpoint, and audit row.

    ``capability_context`` records the full trust context. ``memo_context`` is
    the kind's narrow context used by the memo key (``consent_store``).
    """

    kind: str
    path: str
    memo_context: dict[str, Any]
    combo_key: str
    artifacts: str
    artifacts_legacy: str | None = None
    artifacts_legacy_complete: bool = False
    artifact: str = ""
    target_guard: str = ""
    capability_context: dict[str, Any] | None = None
    file_identity: str | None = None
    unet_name: str | None = None
    loras: int = 0
    numbers: str | None = None
    consent_id: str = ""


@dataclass(frozen=True, slots=True)
class ConsentGrant:
    """A live authorization a bypass site may use.

    ``key`` identifies its memo and is empty for standing or environment grants.
    Callers derive audit keys from the combination being waived.
    """

    id: str
    kind: str
    consent_source: str
    key: str
    reason: str
    capability_context: dict[str, Any] | None = None


@dataclass(slots=True)
class PendingConsent:
    """One invitation, keyed by memo key to deduplicate repeated queues.

    ``consent_id`` is minted with the refusal so the card, memo, and audit rows
    share one identifier.
    """

    id: str
    key: str
    descriptor: ConsentDescriptor
    consent_id: str = field(default_factory=new_consent_id)
    occurrences: int = 1
    first_seen: float = field(default_factory=time.time)
    seen_at: float = field(default_factory=time.time)


_PENDING: OrderedDict[str, PendingConsent] = OrderedDict()
_PENDING_LOCK = threading.Lock()
_DESCRIPTOR_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ConsentDescriptor))


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in _TRUTHY


def _usable_consent_id(value: str) -> bool:
    """Whether an id from elsewhere is the shape the permanent row validates."""
    if len(value) != CONSENT_ID_HEX:
        return False
    return all(character in "0123456789abcdef" for character in value)


def describe(source: object) -> ConsentDescriptor:
    """Normalize a descriptor object or mapping to the local wire shape."""
    if isinstance(source, ConsentDescriptor):
        descriptor = source
    else:
        raw: Mapping[str, Any] = source if isinstance(source, Mapping) else {
            name: getattr(source, name) for name in _DESCRIPTOR_FIELDS if hasattr(source, name)
        }
        get = raw.get
        descriptor = ConsentDescriptor(
            kind=str(get("kind", "")),
            path=str(get("path", "")),
            memo_context=dict(get("memo_context", {}) or {}),
            combo_key=str(get("combo_key", "") or ""),
            artifacts=str(get("artifacts", "") or ""),
            artifacts_legacy=_optional_str(get("artifacts_legacy", None)),
            artifacts_legacy_complete=get("artifacts_legacy_complete", False) is True,
            artifact=str(get("artifact", "") or ""),
            target_guard=str(get("target_guard", "") or ""),
            capability_context=get("capability_context", None),
            file_identity=get("file_identity", None),
            unet_name=get("unet_name", None),
            loras=int(get("loras", 0) or 0),
            numbers=get("numbers", None),
            consent_id=str(get("consent_id", "") or ""),
        )
    spec = KIND_SPECS.get(descriptor.kind)
    if spec is None:
        raise ValueError(f"unknown consent kind {descriptor.kind!r}")
    # Cards and permanent rows validate against the same guard vocabulary.
    guard = check_guard(spec.kind, descriptor.target_guard or spec.default_guard) or ""
    if not descriptor.path or not descriptor.combo_key or not descriptor.artifacts:
        # A card without joinable audit identity cannot authorize a memo.
        raise ValueError("a consent descriptor needs path, combo_key and artifacts")
    artifact = descriptor.artifact or os.path.basename(descriptor.path)
    return ConsentDescriptor(
        kind=descriptor.kind,
        path=descriptor.path,
        memo_context=consent_context(descriptor.kind, **descriptor.memo_context),
        combo_key=descriptor.combo_key,
        artifacts=descriptor.artifacts,
        artifacts_legacy=descriptor.artifacts_legacy,
        artifacts_legacy_complete=descriptor.artifacts_legacy_complete,
        artifact=artifact,
        target_guard=guard,
        capability_context=(dict(descriptor.capability_context)
                            if isinstance(descriptor.capability_context, dict) else None),
        file_identity=descriptor.file_identity,
        unet_name=descriptor.unet_name or artifact,
        loras=descriptor.loras,
        numbers=descriptor.numbers,
        consent_id=descriptor.consent_id,
    )


def numbers_from_measured(measured: Mapping[str, Any] | None) -> str | None:
    """Format the card's capacity numbers from structured refusal evidence.

    ``headroom_bytes`` is what is available minus the refusing site's price,
    so ``needs`` comes back as that price, not the file size. For a stock load
    the price is ``capacity_fit.stock_required_bytes``: the file, the host copy
    it is placed from, and the host floor. A block that carries no headroom
    falls back to the file size.
    """
    if not isinstance(measured, Mapping):
        return None
    needs = measured.get("checkpoint_bytes")
    available = measured.get("mem_available_bytes")
    if not isinstance(needs, int) or not isinstance(available, int):
        return None
    headroom = measured.get("headroom_bytes")
    if isinstance(headroom, int) and not isinstance(headroom, bool):
        needs = available - headroom
    gib = float(1 << 30)
    return f"needs {needs / gib:.1f} GiB, {available / gib:.1f} GiB available"


def descriptor_from_refusal(
    refusal: object,
    *,
    path: str,
    combo_key: str,
    artifacts: str,
    artifacts_legacy: str | None = None,
    artifacts_legacy_complete: bool = False,
    memo_context: dict[str, Any],
    capability_context: dict[str, Any] | None = None,
    loras: int = 0,
) -> ConsentDescriptor:
    """Combine worker refusal evidence with driver-held artifact identity.

    The worker sends no host path. The driver supplies the local path,
    combination, artifact-set digest, and trust context. The refusal's consent
    identifier remains stable across the card, memo, and audit rows.
    """
    get = (refusal.get if isinstance(refusal, Mapping)
           else lambda name, default=None: getattr(refusal, name, default))
    kind = str(get("kind", "") or "")
    measured = get("measured", None)
    guard = ""
    probe = measured.get("probe") if isinstance(measured, Mapping) else None
    if isinstance(probe, str) and kind in KIND_SPECS:
        try:
            guard = check_guard(kind, probe) or ""
        except ValueError:
            guard = ""  # An invalid probe cannot override the kind's default guard.
    return describe(ConsentDescriptor(
        kind=kind,
        path=path,
        memo_context=memo_context,
        combo_key=combo_key,
        artifacts=artifacts,
        artifacts_legacy=artifacts_legacy,
        artifacts_legacy_complete=artifacts_legacy_complete,
        artifact=str(get("unet_name", "") or "") or os.path.basename(path),
        target_guard=guard,
        capability_context=capability_context,
        file_identity=str(get("file_identity", "") or "") or None,
        unet_name=str(get("unet_name", "") or "") or None,
        loras=loras,
        numbers=numbers_from_measured(measured if isinstance(measured, Mapping) else None),
        consent_id=str(get("consent_id", "") or ""),
    ))


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _card(entry: PendingConsent) -> dict[str, Any]:
    spec = KIND_SPECS[entry.descriptor.kind]
    return {
        "id": entry.id,
        "key": entry.key,
        "consent_id": entry.consent_id,
        "kind": spec.kind,
        "class": spec.refusal_class,
        "style": spec.style,
        "title": spec.title.format(artifact=entry.descriptor.artifact),
        "artifact": entry.descriptor.artifact,
        "risk": spec.risk,
        "evidence": spec.evidence,
        "numbers": entry.descriptor.numbers,
        "measured": spec.measured,
        "primary": {"label": spec.primary_label, "action": "accept"},
        "dismissible": True,
        "auto_eligible": spec.auto_eligible,
        "occurrences": entry.occurrences,
        "seen_at": entry.seen_at,
    }


def _expire(now: float) -> None:
    for key, entry in list(_PENDING.items()):
        if now - entry.seen_at > PENDING_TTL_S:
            _PENDING.pop(key, None)


def register_pending(descriptor: object) -> dict[str, object] | None:
    """Publish a refusal card without allowing bookkeeping to mask the refusal."""
    try:
        desc = describe(descriptor)
        spec = KIND_SPECS[desc.kind]
        if not spec.wired:
            log.debug("consent kind %s is not wired in this release; no card raised", desc.kind)
            return None
        key = consent_store.memo_key(desc.kind, desc.path, desc.memo_context)
        now = time.time()
        with _PENDING_LOCK:
            _expire(now)
            existing = _PENDING.get(key)
            if existing is not None:
                # Keep one card and identifier per question while refreshing its
                # live capacity numbers. Rotating it could reject the displayed card.
                existing.occurrences += 1
                existing.seen_at = now
                existing.descriptor = desc
                return _card(existing)
            entry = PendingConsent(id="p-" + secrets.token_hex(6), key=key, descriptor=desc)
            if _usable_consent_id(desc.consent_id):
                # Preserve the refusal's valid identifier across all records.
                # Invalid identifiers are replaced before click-time validation.
                entry.consent_id = desc.consent_id
            _PENDING[key] = entry
            while len(_PENDING) > PENDING_LIMIT:
                _PENDING.popitem(last=False)
            return _card(entry)
    except Exception as exc:  # registration never masks the refusal it accompanies
        log.warning("pending consent not registered (%r)", exc)
        return None


def pending_cards() -> list[dict[str, Any]]:
    """Every live card, oldest first."""
    now = time.time()
    with _PENDING_LOCK:
        _expire(now)
        return [_card(entry) for entry in _PENDING.values()]


def pending_count() -> int:
    with _PENDING_LOCK:
        _expire(time.time())
        return len(_PENDING)


def peek_pending(key: str, pending_id: str) -> PendingConsent | None:
    """Return the matching invitation without removing it.

    Validation and persistence occur before removal so a failed grant remains
    retryable.
    """
    with _PENDING_LOCK:
        _expire(time.time())
        entry = _PENDING.get(key)
        return entry if entry is not None and entry.id == pending_id else None


def take_pending(key: str, pending_id: str) -> PendingConsent | None:
    """Remove and return the invitation matching both key and identifier."""
    with _PENDING_LOCK:
        _expire(time.time())
        entry = _PENDING.get(key)
        if entry is None or entry.id != pending_id:
            return None
        _PENDING.pop(key, None)
        return entry


def clear_auto_eligible() -> int:
    """Drop the cards a standing auto-rescue toggle just answered."""
    with _PENDING_LOCK:
        cleared = [key for key, entry in _PENDING.items()
                   if KIND_SPECS[entry.descriptor.kind].auto_eligible]
        for key in cleared:
            _PENDING.pop(key, None)
        return len(cleared)


def clear_all() -> None:
    """Drop every live invitation (driver teardown, tests)."""
    with _PENDING_LOCK:
        _PENDING.clear()


def auto_rescue_env() -> bool:
    """Headless standing authorization, the environment mirror of the toggle."""
    return _truthy(os.environ.get(AUTO_RESCUE_ENV))


def _env_grants(spec: KindSpec) -> bool:
    raw = os.environ.get(spec.env_var)
    if raw is None:
        return False
    if spec.refusal_class != "K":
        return _truthy(raw)
    # Known-wrong math requires the kind or guard name, never a bare truthy flag.
    # The warning runs whether or not this kind matched: one variable carries
    # every wired kind, so a value can grant one and still name a token nothing
    # in this build can read.
    accuracy_waiver.warn_unreadable_env(spec, raw)
    return accuracy_waiver.env_grants_kind(spec, raw)


def resolve(*, kind: str, path: str, context: dict[str, Any]) -> ConsentGrant | None:
    """Resolve a memo, eligible standing grant, or environment fallback.

    This function neither raises nor writes. The bypass site records a WAIVER
    only when used. A grant cannot override quarantine or turn INCONCLUSIVE into
    PASS.
    """
    spec = KIND_SPECS.get(kind)
    if spec is None:
        return None
    try:
        record = consent_store.lookup(kind, path, context)
        if record is not None:
            return ConsentGrant(
                id=record.id, kind=kind, consent_source=record.consent_source,
                key=consent_store.memo_key(kind, path, context), reason=record.reason,
                capability_context=(dict(record.capability_context)
                                    if record.capability_context is not None else None))
        if spec.auto_eligible and (consent_store.auto_rescue() or auto_rescue_env()):
            return _minted(spec, "auto_rescue")
        if _env_grants(spec):
            return _minted(spec, "env")
    except Exception as exc:  # a store fault degrades to "ask", never to a grant
        log.warning("consent lookup failed, so nothing is granted (%r)", exc)
    return None


def _minted(spec: KindSpec, source: str) -> ConsentGrant:
    """A grant with no memo behind it: standing toggle or environment."""
    return ConsentGrant(id=new_consent_id(), kind=spec.kind, consent_source=source,
                        key="", reason=spec.risk)


def grant_wire(grant: ConsentGrant) -> dict[str, str]:
    """The private request key the driver stamps for the worker side."""
    return {"id": grant.id, "kind": grant.kind, "consent_source": grant.consent_source}


def grant_from_wire(payload: object) -> ConsentGrant | None:
    """Read a grant across driver-worker version skew.

    Unknown or malformed fields fail closed to no grant. Extra fields are
    ignored for forward compatibility.
    """
    if not isinstance(payload, Mapping):
        return None
    kind = payload.get("kind")
    consent_id = payload.get("id")
    source = payload.get("consent_source", "unknown")
    if not isinstance(kind, str) or kind not in KIND_SPECS:
        return None
    if not isinstance(consent_id, str) or not consent_id:
        return None
    return ConsentGrant(
        id=consent_id, kind=kind,
        consent_source=source if isinstance(source, str) else "unknown",
        key="", reason=KIND_SPECS[kind].risk)
