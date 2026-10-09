"""Machine-readable consent descriptors carried in refusal text.

Workers raise consent offers (slab rescue and class-K waivers) and drivers
draw the cards. Monarch preserves only exception text across that boundary,
so the descriptor is bounded, self-delimiting JSON with duplicate-key,
non-finite-value, version, and field validation. Multiple or malformed
descriptors return ``None`` and leave the refusal active without a card.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import Any

from .mesh_safety import StockLoadCapacityError

CONSENT_DESCRIPTOR_VERSION = 1
DESCRIPTOR_BEGIN = "<<<DGXM-CONSENT-V1 "
DESCRIPTOR_END = " DGXM-CONSENT-END>>>"
MAX_DESCRIPTOR_BYTES = 4096

# One slab-rescue grant covers the first-load-stock and unvouched-family
# conditions for that load, producing at most one card. Explicit slab
# residency needs no rescue consent.
KIND_RESCUE_SLAB = "rescue-slab"

# A class-K kind is this prefix plus a short name (letters, digits, "-", "_", ".").
# Any such name parses, so an older driver still parses a newer worker's
# descriptor, then raises no card for a kind it lacks. A family-scoped guard
# travels in measured["probe"], never in the kind.
KIND_WAIVE_KNOWN_WRONG_PREFIX = "waive-known-wrong:"
_MAX_GUARD_ID = 48

# Driver-side headless fallback for slab rescue. Class C and U kinds each have
# their own variable; every class-K kind shares DGXM_WAIVE_KNOWN_WRONG.
ENV_SLAB_RESCUE = "DGXM_ALLOW_SLAB_RESCUE"

# The card's evidence sentence is fixed by the certificate contract.
EVIDENCE_BYTE_VERIFIED = (
    "weights will be byte-verified against the checkpoint during load"
)

_REQUIRED_STR_FIELDS = (
    "kind", "consent_id", "unet_name", "file_identity", "context_fingerprint",
    "human_reason", "evidence", "panel_action", "env_fallback",
)


class ConsentRequiredError(StockLoadCapacityError):
    """A class-C or class-U refusal that explicit consent can clear.

    Subclassing preserves local capacity handling. Subclass identity does not
    survive Monarch wrapping, so a worker-raised refusal also names the parent
    class in its text, which ``mesh_safety.is_stock_load_capacity_error`` matches.
    """


class SlabResidencyRescueOffer(ConsentRequiredError):
    """Stock residency cannot fit this checkpoint; slab residency can."""


def is_slab_rescue_offer(exc: BaseException) -> bool:
    """Recognize local and Monarch-wrapped slab rescue offers."""
    return (isinstance(exc, SlabResidencyRescueOffer)
            or "SlabResidencyRescueOffer" in str(exc))


@dataclass(frozen=True, slots=True)
class ConsentDescriptor:
    """Everything the driver needs to draw one consent card, and nothing more."""

    version: int
    kind: str
    consent_id: str
    unet_name: str            # comfy-relative name, never a host path
    file_identity: str        # store_family.file_identity() spelling
    context_fingerprint: str
    family_hint: str | None
    measured: dict[str, Any]
    human_reason: str
    evidence: str
    panel_action: str
    env_fallback: str
    # Narrow memo context for the driver store. It is optional so the descriptor
    # parses across version skew; a driver that gets none raises no card
    # (consent_observe).
    memo_context: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "kind": self.kind,
            "consent_id": self.consent_id,
            "unet_name": self.unet_name,
            "file_identity": self.file_identity,
            "context_fingerprint": self.context_fingerprint,
            "family_hint": self.family_hint,
            "measured": dict(self.measured),
            "memo_context": dict(self.memo_context),
            "human_reason": self.human_reason,
            "evidence": self.evidence,
            "panel_action": self.panel_action,
            "env_fallback": self.env_fallback,
        }

    def with_enrichment(self, **fields: Any) -> ConsentDescriptor:
        """Driver-side enrichment stays typed: only existing fields can be replaced."""
        return replace(self, **fields)


def valid_kind(kind: Any) -> bool:
    """Whether ``kind`` is a member of the frozen consent grammar."""
    if not isinstance(kind, str) or not kind:
        return False
    if kind == KIND_RESCUE_SLAB:
        return True
    if not kind.startswith(KIND_WAIVE_KNOWN_WRONG_PREFIX):
        return False
    guard = kind[len(KIND_WAIVE_KNOWN_WRONG_PREFIX):]
    return (0 < len(guard) <= _MAX_GUARD_ID
            and all(c.isalnum() or c in "-_." for c in guard))


# Width the audit schema and the pending registry require.
CONSENT_ID_HEX = 32


def consent_id(kind: str, file_identity: str, context_fingerprint: str) -> str:
    """Return a deterministic identifier that deduplicates repeated refusals."""
    raw = f"{kind}\0{file_identity}\0{context_fingerprint}".encode()
    return hashlib.sha256(raw).hexdigest()[:CONSENT_ID_HEX]


def memo_context_fingerprint(*, unet_name: str, weight_dtype: str | None) -> str:
    """Return the worker's narrow residency-context fingerprint.

    It excludes topology, latent size, and worker args, as the residency memo
    key does, so an authorized policy change keeps the same consent id. The
    ledger still records the full capability context. The durable key is the
    driver's ``consent_store.memo_key``; this fingerprint feeds ``consent_id``,
    so a repeated refusal mints the same id.
    """
    payload = json.dumps(
        {"unet_name": unet_name, "weight_dtype": weight_dtype or ""},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def encode(descriptor: ConsentDescriptor) -> str:
    """Canonical sentinel-wrapped JSON, or raise if it cannot be represented."""
    payload = json.dumps(
        descriptor.public(), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    if json.loads(payload) != descriptor.public():
        # A tuple (a namedtuple too) or a non-string key comes back from JSON
        # changed. Refuse to emit what will not come back.
        raise ValueError("consent descriptor is not JSON-native: it changed shape through JSON")
    raw = payload.encode()
    if len(raw) > MAX_DESCRIPTOR_BYTES:
        raise ValueError(
            f"consent descriptor is {len(raw)} bytes, over the {MAX_DESCRIPTOR_BYTES} bound"
        )
    return f"{DESCRIPTOR_BEGIN}{payload}{DESCRIPTOR_END}"


def parse(text: str) -> ConsentDescriptor | None:
    """Parse wrapped refusal text; return ``None`` for every malformed shape."""
    try:
        return _parse(text)
    except (ValueError, TypeError, AttributeError, UnicodeError):
        return None


def _parse(text: str) -> ConsentDescriptor | None:
    if not isinstance(text, str) or text.count(DESCRIPTOR_BEGIN) != 1:
        return None
    tail = text.split(DESCRIPTOR_BEGIN, 1)[1]
    if tail.count(DESCRIPTOR_END) != 1:
        return None
    payload = tail.split(DESCRIPTOR_END, 1)[0]
    raw = payload.encode()          # surrogates raise UnicodeError: refused
    if not raw or len(raw) > MAX_DESCRIPTOR_BYTES:
        return None
    body = json.loads(payload, object_pairs_hook=_no_duplicate_keys,
                      parse_constant=_no_constants)
    if not isinstance(body, dict) or body.get("version") != CONSENT_DESCRIPTOR_VERSION:
        return None
    if not valid_kind(body.get("kind")):
        return None
    for field_name in _REQUIRED_STR_FIELDS:
        if not isinstance(body.get(field_name), str) or not body[field_name]:
            return None
    hint = body.get("family_hint")
    if hint is not None and not isinstance(hint, str):
        return None
    measured = body.get("measured")
    if not isinstance(measured, dict) or not _measured_is_flat_and_finite(measured):
        return None
    narrow = body.get("memo_context", {})
    if not isinstance(narrow, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in narrow.items()):
        # Drop malformed narrow context; consent_observe then raises no card.
        narrow = {}
    return ConsentDescriptor(
        version=CONSENT_DESCRIPTOR_VERSION,
        kind=body["kind"],
        consent_id=body["consent_id"],
        unet_name=body["unet_name"],
        file_identity=body["file_identity"],
        context_fingerprint=body["context_fingerprint"],
        family_hint=hint,
        measured=dict(measured),
        memo_context=dict(narrow),
        human_reason=body["human_reason"],
        evidence=body["evidence"],
        panel_action=body["panel_action"],
        env_fallback=body["env_fallback"],
    )


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r} in a consent descriptor")
        out[key] = value
    return out


def _no_constants(token: str) -> Any:
    raise ValueError(f"non-finite constant {token!r} in a consent descriptor")


def _measured_is_flat_and_finite(measured: dict[str, Any]) -> bool:
    for key, value in measured.items():
        if not isinstance(key, str):
            return False
        if isinstance(value, bool) or value is None or isinstance(value, str):
            continue
        if isinstance(value, int):
            continue
        if isinstance(value, float) and math.isfinite(value):
            continue
        return False
    return True
