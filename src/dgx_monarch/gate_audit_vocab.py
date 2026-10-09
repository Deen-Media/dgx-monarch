"""Audit-row vocabulary, namespace, and validators.

The vocabulary is protocol-bound. Changing an enum, guard, required field, or
key spelling after rows exist requires a protocol version change. This leaf uses
only the standard library.
"""
from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

WAIVER_VERDICT = "WAIVER"
CAPACITY_VERDICT = "CAPACITY_CERTIFIED"
AUDIT_VERDICTS = frozenset({WAIVER_VERDICT, CAPACITY_VERDICT})

# Audit keys use a namespace that cannot collide with 24-character hexadecimal
# combination keys. One predicate therefore covers every audit row type.
AUDIT_KEY_PREFIX = "audit:"

REASON_MAX_CHARS = 400

# Map each kind to its refusal class. Builders derive the class from this table
# so callers cannot label a known-wrong-math bypass as a capacity rescue.
KIND_CLASS: dict[str, str] = {
    "rescue-slab": "C",
    "auto-rescue-standing": "C",
    "waive-preflight:activation_footprint": "C",
    "waive-preflight:driver_footprint": "C",
    "waive-first-load-stock": "U",
    "waive-unvouched-slab": "U",
    "waive-known-wrong:ring-pad": "K",
    "waive-known-wrong:pixeldit-sp": "K",
    "waive-known-wrong:sol-attn": "K",
    "waive-known-wrong:shard-quant": "K",
}
WAIVER_KINDS = frozenset(KIND_CLASS)
# Class-P physics refusals are not waivable and cannot be expressed here.

# A builder refuses any guard outside the kind's allowlist.
KIND_GUARDS: dict[str, frozenset[str]] = {
    "rescue-slab": frozenset({
        "stock_load_preflight",
        "activation_footprint_preflight",
        "driver_footprint_preflight",
        "loader_footprint_preflight",
        "partial_load_divergence",
        # None of the three below is waivable, so no waiver row cites them.
        # They are listed because check_guard must accept every refusal.GUARDS
        # entry under the kind it declares (tests/test_refusal_classes.py).
        "slab_load_preflight",
        "cross_rank_capacity",
        "slab_lora_bake_preflight",
    }),
    "auto-rescue-standing": frozenset(),
    "waive-preflight:activation_footprint": frozenset({"activation_footprint_preflight"}),
    "waive-preflight:driver_footprint": frozenset({
        "driver_footprint_preflight", "loader_footprint_preflight"}),
    "waive-first-load-stock": frozenset({"first_load_stock_memo"}),
    "waive-unvouched-slab": frozenset({"slab_vouched_families"}),
    "waive-known-wrong:ring-pad": frozenset(),
    "waive-known-wrong:pixeldit-sp": frozenset(),
    "waive-known-wrong:sol-attn": frozenset(),
    "waive-known-wrong:shard-quant": frozenset(),
}
# Known-wrong guards include a family or adapter scope so one waiver cannot
# silence the same guard in another family.
KIND_GUARD_PREFIXES: dict[str, str] = {
    "waive-known-wrong:ring-pad": "ring_pad",
    "waive-known-wrong:pixeldit-sp": "sp_unvalidated",
    "waive-known-wrong:sol-attn": "sol_attn",
    "waive-known-wrong:shard-quant": "shard_quant_scale",
}
_SCOPE_RE = re.compile(r"^[a-z0-9_]+$")

ACTIONS = frozenset({"grant", "use", "revoke"})
CONSENT_SOURCES = frozenset({"panel", "env", "auto_rescue", "cli"})
REVOKED_BY = frozenset({"user", "gate_fail", "file_change"})

CERTIFICATE_ORIGINS = frozenset({"rescue_load", "gate_cross_leg"})
RESIDENCY_KINDS = frozenset({"slab"})
MEASURED_PROBES = frozenset({
    "stock_load_preflight",
    "activation_footprint_preflight",
    "driver_footprint_preflight",
    "loader_footprint_preflight",
    "worker_stock_load",
    "worker_slab_load",
})
CERTIFICATE_ALGORITHMS = frozenset({
    "memcmp-tensor-v1", "memcmp-tensor-sampled-v1", "memcmp-tensor-full-v1",
    "sha256-tensor-v1", "sha256-tensor-sampled-v1", "sha256-tensor-full-v1",
})
# Artifact-signature sentinels do not prove that two ranks read the same bytes.
SIGNATURE_SENTINELS = frozenset({"unreadable", "unstable"})

# Required per-rank certificate keys. Readers ignore extra keys for forward
# compatibility and treat missing keys as incomplete evidence.
CERTIFICATE_WIRE_KEYS = (
    "algorithm", "digest", "complete", "tensors_verified", "tensors_total",
    "bytes_verified", "checkpoint_bytes", "artifact_signature", "file_identity",
)

# Opens the tail ``gate_audit_evidence.measured_tag`` appends to a refusal.
MEASURED_TAG_PREFIX = "[dgxm:measured "

_HEX_RE = re.compile(r"^[0-9a-f]+$")


class GateAuditError(RuntimeError):
    """An audit row could not be built or recorded."""


class WaiverNotAuditedError(GateAuditError):
    """A grant's audit row was not written; the panel refuses that grant.

    A standing or environment bypass still applies, and
    ``audit_non_memo_grant`` logs the failure.
    """


def new_consent_id() -> str:
    """Mint the identifier shared by a pending consent, memo, and audit rows.

    It is created with the refusal so all later records join without inference.
    """
    return uuid.uuid4().hex


def waiver_key(kind: str, combo_key: str) -> str:
    return f"{AUDIT_KEY_PREFIX}waiver:{kind}:{combo_key}"


def capacity_key(combo_key: str) -> str:
    return f"{AUDIT_KEY_PREFIX}capacity:{combo_key}"


def is_audit_key(key: object) -> bool:
    """Whether one ledger row key belongs to the audit namespace."""
    return isinstance(key, str) and key.startswith(AUDIT_KEY_PREFIX)


def is_audit_row(entry: Mapping[str, Any]) -> bool:
    """Recognize an audit row by either namespace or verdict."""
    return is_audit_key(entry.get("key")) or entry.get("verdict") in AUDIT_VERDICTS


def trust_rows(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return trust rows safe to group and count as gate verdicts.

    Audit rows must not appear as gated combinations in status consumers.
    """
    return [dict(entry) for entry in entries if not is_audit_row(entry)]


def audit_rows(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The complement of :func:`trust_rows`, for the panel's audit views."""
    return [dict(entry) for entry in entries if is_audit_row(entry)]


def check_text(value: object, field: str, *, limit: int | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if limit is not None and len(value) > limit:
        # Refuse over-length authorization text instead of truncating its meaning.
        raise ValueError(f"{field} must be at most {limit} characters, got {len(value)}")
    return value


def check_hex(value: object, field: str, length: int) -> str:
    text = check_text(value, field)
    if len(text) != length or not _HEX_RE.match(text):
        raise ValueError(f"{field} must be {length} lowercase hex characters")
    return text


def check_member(value: object, field: str, allowed: frozenset[str]) -> str:
    text = check_text(value, field)
    if text not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}, got {text!r}")
    return text


def check_count(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def check_guard(kind: str, guard: object) -> str | None:
    """The guard this kind is allowed to name, scope suffix included."""
    prefix = KIND_GUARD_PREFIXES.get(kind)
    if prefix is not None:
        text = check_text(guard, "target_guard")
        base, _, scope = text.partition(":")
        if base != prefix or (scope and not _SCOPE_RE.match(scope)):
            raise ValueError(
                f"target_guard for {kind} must be {prefix!r} or {prefix}:<scope>, got {text!r}")
        return text
    allowed = KIND_GUARDS[kind]
    if not allowed:
        if guard is not None:
            raise ValueError(f"{kind} names no guard; target_guard must be None")
        return None
    return check_member(guard, "target_guard", allowed)
