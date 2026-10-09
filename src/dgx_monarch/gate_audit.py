"""Permanent WAIVER and CAPACITY_CERTIFIED audit rows.

Audit rows are evidence, never authority. Namespaced keys, contexts carrying a
top-level ``record`` discriminator, and non-trust verdicts prevent them from
matching or granting trust. They cannot convert INCONCLUSIVE, override
quarantine, or authorize consent; the consent memo is the authority.

All rows use ``GateLedger.record`` so malformed audit data cannot damage scans.
This leaf imports without nodes, mesh, torch, or ComfyUI.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .gate_audit_evidence import certificate_digest as certificate_digest
from .gate_audit_evidence import fold_rank_certificates as fold_rank_certificates
from .gate_audit_evidence import measured_tag as measured_tag
from .gate_audit_evidence import parse_measured as parse_measured
from .gate_audit_evidence import validate_certificate, validate_measured
from .gate_audit_vocab import (
    ACTIONS,
    CAPACITY_VERDICT,
    CERTIFICATE_ORIGINS,
    CONSENT_SOURCES,
    KIND_CLASS,
    REASON_MAX_CHARS,
    REVOKED_BY,
    WAIVER_KINDS,
    WAIVER_VERDICT,
    capacity_key,
    check_count,
    check_guard,
    check_hex,
    check_member,
    check_text,
    waiver_key,
)
from .gate_audit_vocab import WaiverNotAuditedError as WaiverNotAuditedError
from .gate_audit_vocab import audit_rows as audit_rows
from .gate_audit_vocab import is_audit_key as is_audit_key
from .gate_audit_vocab import is_audit_row as is_audit_row
from .gate_audit_vocab import new_consent_id as new_consent_id
from .gate_audit_vocab import trust_rows as trust_rows
from .gate_ledger import ArtifactSetSignature, GateLedger

# TRUST citations anchor the private name. Sharing the ledger encoder prevents
# fingerprint and stored-context drift.
from .gate_ledger import _canonical_capability_context as canonical_capability_context
from .log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One built, validated audit row, ready for ``GateLedger.record``."""

    key: str
    verdict: str
    detail: dict[str, Any]
    context: dict[str, Any]


def context_fingerprint(context: Mapping[str, Any]) -> str:
    """sha256 over the ledger's own canonical encoding of one context."""
    return hashlib.sha256(canonical_capability_context(dict(context)).encode()).hexdigest()


def waiver_audit_context(kind: str, guard: str | None, memo_fingerprint: str) -> dict[str, Any]:
    return {
        "record": "waiver",
        "waiver_kind": kind,
        "target_guard": guard,
        "memo_fingerprint": memo_fingerprint,
    }


def capacity_audit_context(capability_fingerprint: str) -> dict[str, Any]:
    return {
        "record": "capacity_certificate",
        "residency": "slab",
        "capability_fingerprint": capability_fingerprint,
    }


def _check_artifacts(artifacts: str | ArtifactSetSignature) -> str | ArtifactSetSignature:
    """Require the ledger's 64-hex artifact-set digest, not a file signature."""
    if isinstance(artifacts, ArtifactSetSignature):
        return artifacts
    check_hex(artifacts, "artifacts", 64)
    return artifacts


def build_waiver_row(
    *,
    combo_key: str,
    waiver_kind: str,
    target_guard: str | None,
    action: str,
    consent_id: str,
    consent_source: str,
    memo_context: Mapping[str, Any],
    file_identity: str,
    unet_name: str,
    loras: int,
    reason: str,
    capability_context: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    stamp: str | None = None,
    revoked_by: str | None = None,
) -> AuditRow:
    """Build a WAIVER row with derived class and narrow memo context.

    The narrow context binds the property consent covers without becoming
    self-referential when residency policy changes. Any full trust context is
    recorded only as audit data. Keyword-only fields cannot shadow ledger base
    fields.
    """
    kind = check_member(waiver_kind, "waiver_kind", WAIVER_KINDS)
    guard = check_guard(kind, target_guard)
    waiver_class = KIND_CLASS[kind]
    memo_fingerprint = context_fingerprint(memo_context)
    detail: dict[str, Any] = {
        "waiver_kind": kind,
        "waiver_class": waiver_class,
        "target_guard": guard,
        "action": check_member(action, "action", ACTIONS),
        "consent_id": check_hex(consent_id, "consent_id", 32),
        "consent_source": check_member(consent_source, "consent_source", CONSENT_SOURCES),
        "memo_fingerprint": memo_fingerprint,
        "memo_context": canonical_capability_context(dict(memo_context)),
        "file_identity": check_text(file_identity, "file_identity"),
        "unet_name": check_text(unet_name, "unet_name"),
        "loras": check_count(loras, "loras"),
        # Preserve the exact operator-visible consent sentence.
        "reason": check_text(reason, "reason", limit=REASON_MAX_CHARS),
        # Audit needs a sortable value in addition to the ledger display time.
        "waived_at": time.time(),
    }
    if capability_context is not None:
        detail["target_context"] = canonical_capability_context(dict(capability_context))
    if detail["action"] == "use":
        detail["run_id"] = check_text(run_id, "run_id")
    if waiver_class == "K":
        # Frozen with class-K vocabulary; additions require a protocol version change.
        detail["stamp"] = check_text(stamp, "stamp", limit=REASON_MAX_CHARS)
    if detail["action"] == "revoke":
        detail["revoked_by"] = check_member(revoked_by, "revoked_by", REVOKED_BY)
    return AuditRow(
        key=waiver_key(kind, combo_key),
        verdict=WAIVER_VERDICT,
        detail=detail,
        context=waiver_audit_context(kind, guard, memo_fingerprint),
    )


def build_capacity_row(
    *,
    combo_key: str,
    capability_context: Mapping[str, Any],
    certificate: Mapping[str, Any],
    measured: Mapping[str, Any],
    certificate_origin: str,
    unet_name: str,
    file_identity: str,
    consent_id: str | None = None,
    origin: str | None = None,
    run_id: str | None = None,
) -> AuditRow:
    """Record verified resident bytes, never render correctness or PASS."""
    origin_kind = check_member(certificate_origin, "certificate_origin", CERTIFICATE_ORIGINS)
    detail: dict[str, Any] = {
        # The frozen vocabulary covers only slab certificate evidence.
        "residency": "slab",
        # Explicit markers make certified loads searchable without enum parsing.
        "slab_resident": True,
        "stock_available": False,
        "certificate_origin": origin_kind,
        "unet_name": check_text(unet_name, "unet_name"),
        "file_identity": check_text(file_identity, "file_identity"),
        "capability_fingerprint": context_fingerprint(capability_context),
        "target_context": canonical_capability_context(dict(capability_context)),
        "measured": validate_measured(measured),
        "certificate": validate_certificate(certificate),
    }
    if origin_kind == "rescue_load":
        detail["consent_id"] = check_hex(consent_id, "consent_id", 32)
    elif consent_id is not None:
        raise ValueError(
            "a gate_cross_leg certificate ran under no consent; consent_id must be None")
    if origin is not None:
        detail["origin"] = check_text(origin, "origin")
    if run_id is not None:
        detail["run_id"] = check_text(run_id, "run_id")
    return AuditRow(
        key=capacity_key(combo_key),
        verdict=CAPACITY_VERDICT,
        detail=detail,
        context=capacity_audit_context(detail["capability_fingerprint"]),
    )


def _record(ledger: GateLedger, row: AuditRow, artifacts: str | ArtifactSetSignature,
            commit: str) -> bool:
    """Append a row, normalizing record rejection to ``False``."""
    try:
        return bool(ledger.record(
            row.key, _check_artifacts(artifacts), commit, row.verdict, row.detail, row.context))
    except (TypeError, ValueError, RecursionError) as exc:
        log.error("v9 %s row could not be written for %s: %r", row.verdict, row.key, exc)
        return False


def record_waiver(ledger: GateLedger, artifacts: str | ArtifactSetSignature, commit: str,
                  **fields: Any) -> bool:
    """Append grant, revoke, or actual class-K use; raise on malformed input.

    Class C/U residency facts belong in per-load CAPACITY_CERTIFIED rows, which
    avoids redundant per-render rows in a linearly scanned ledger.
    """
    _check_artifacts(artifacts)
    return _record(ledger, build_waiver_row(**fields), artifacts, commit)


def record_waiver_grant(ledger: GateLedger, artifacts: str | ArtifactSetSignature, commit: str,
                        **fields: Any) -> dict[str, Any]:
    """Persist the audit row before granting consent or refuse the grant.

    Grant order is row, memo, then success. Revocation removes consent first
    and audits best effort, since audit failure must never preserve consent.
    """
    _check_artifacts(artifacts)
    row = build_waiver_row(action="grant", **fields)
    if not _record(ledger, row, artifacts, commit):
        raise WaiverNotAuditedError(
            "this consent could not be written to the identity-gate ledger. If it came from the panel, it was "
            "not granted; a standing or environment bypass still applies without its audit row. "
            "Check that the ComfyUI output directory is writable; the next click or load tries again.")
    return row.detail


def record_ceremony_certificate(ledger: GateLedger, key: str,
                                artifacts: str | ArtifactSetSignature, commit: str,
                                cross_mode: Mapping[str, Any] | None,
                                capability_context: Mapping[str, Any],
                                detail: Mapping[str, Any]) -> bool:
    """Best-effort certificate for an INCONCLUSIVE capacity-skipped stock leg.

    Missing or malformed evidence writes no row and never raises.
    """
    if not isinstance(cross_mode, Mapping) or cross_mode.get("verdict") != "CAPACITY":
        return False
    certificate = cross_mode.get("certificate")
    measured = cross_mode.get("measured")
    if not isinstance(certificate, Mapping) or not isinstance(measured, Mapping):
        return False
    try:
        row = build_capacity_row(
            combo_key=key,
            capability_context=capability_context,
            certificate=certificate,
            measured=measured,
            certificate_origin="gate_cross_leg",
            unet_name=str(detail.get("model") or ""),
            file_identity=ceremony_file_identity(detail.get("model"), certificate),
            origin=detail.get("origin") or None,
            run_id=detail.get("run_id") or None,
        )
    except (TypeError, ValueError) as exc:
        log.warning("capacity certificate row not recorded for %s: %r", detail.get("model"), exc)
        return False
    return _record(ledger, row, artifacts, commit)


def record_rescue_certificate(ledger: GateLedger, key: str,
                              artifacts: str | ArtifactSetSignature, commit: str, *,
                              capability_context: Mapping[str, Any],
                              certificate: Mapping[str, Any],
                              measured: Mapping[str, Any],
                              consent_id: str,
                              unet_name: str,
                              file_identity: str,
                              run_id: str | None = None) -> bool:
    """Record one driver-side certificate for a consented pre-gate slab load.

    Persistence is loud but best effort because load and grant audit already
    completed; failure must not kill the render.
    """
    try:
        row = build_capacity_row(
            combo_key=key,
            capability_context=capability_context,
            certificate=certificate,
            measured=measured,
            certificate_origin="rescue_load",
            unet_name=unet_name,
            file_identity=file_identity,
            consent_id=consent_id,
            run_id=run_id,
        )
    except (TypeError, ValueError) as exc:
        log.error("capacity certificate row not recorded for %s: %r", unet_name, exc)
        return False
    written = _record(ledger, row, artifacts, commit)
    if not written:
        log.error("slab load of %s was certified but its v9 audit row did not persist",
                  unet_name)
    return written


def ceremony_file_identity(unet_name: object, certificate: Mapping[str, Any]) -> str:
    """Resolve driver checkpoint identity, then fall back to rank-0 evidence.

    Guarded local imports preserve no-Comfy importability.
    """
    if isinstance(unet_name, str) and unet_name:
        try:
            import folder_paths

            from .actor.store_family import file_identity

            path = folder_paths.get_full_path("diffusion_models", unet_name)
            if path:
                return file_identity(path)
        except (ImportError, OSError, AttributeError, TypeError):
            pass
    return str(certificate.get("file_identity") or "")
