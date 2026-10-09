"""Evidence a CAPACITY_CERTIFIED row carries: measured numbers and the folded
byte-verify certificate.

Neither half is authority. The numbers say why stock residency was impossible
on this box; the certificate says the slab holds the checkpoint's own bytes.
Whether the render math is right stays the identity gate's question alone.

Leaf module: stdlib, the frozen vocabulary and the logger.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .gate_audit_vocab import (
    CERTIFICATE_ALGORITHMS,
    CERTIFICATE_WIRE_KEYS,
    MEASURED_PROBES,
    MEASURED_TAG_PREFIX,
    SIGNATURE_SENTINELS,
    check_count,
    check_hex,
    check_member,
    check_text,
)
from .log import get_logger

log = get_logger(__name__)

_MEASURED_RE = re.compile(re.escape(MEASURED_TAG_PREFIX) + r"(\{.*?\})\]", re.DOTALL)


def validate_measured(measured: Mapping[str, Any]) -> dict[str, Any]:
    """The numbers that justify the word CAPACITY, or a refusal to claim it.

    ``checkpoint_bytes`` and ``mem_available_bytes`` are mandatory: a capacity
    claim with no numbers is not evidence. The two projection terms are
    nullable because the pure worker-side refusal path has no driver estimate
    and the driver estimator does not cover every family.
    """
    if not isinstance(measured, Mapping):
        raise TypeError("measured must be a mapping")
    row: dict[str, Any] = {
        "probe": check_member(measured.get("probe"), "measured.probe", MEASURED_PROBES),
        "checkpoint_bytes": check_count(
            measured.get("checkpoint_bytes"), "measured.checkpoint_bytes"),
        "mem_available_bytes": check_count(
            measured.get("mem_available_bytes"), "measured.mem_available_bytes"),
        "driver_projection_bytes": None,
        "worker_projection_bytes": None,
        "headroom_bytes": 0,
    }
    for field in ("driver_projection_bytes", "worker_projection_bytes"):
        value = measured.get(field)
        row[field] = None if value is None else check_count(value, f"measured.{field}")
    headroom = measured.get("headroom_bytes")
    if isinstance(headroom, bool) or not isinstance(headroom, int):
        # Signed on purpose: negative is the normal case on this row kind.
        raise ValueError("measured.headroom_bytes must be a signed integer")
    row["headroom_bytes"] = headroom
    return row


def measured_tag(measured: Mapping[str, Any]) -> str:
    """Machine tail a capacity refusal appends so its numbers survive the wire.

    A worker-raised refusal reaches the driver as Monarch-wrapped text, which
    drops every attribute on the exception object. Refusal sites that want
    their capacity row to carry evidence append this to the message.
    """
    return MEASURED_TAG_PREFIX + json.dumps(
        validate_measured(measured), sort_keys=True, separators=(",", ":")) + "]"


def parse_measured(source: object) -> dict[str, Any] | None:
    """Read capacity measurements from an attribute, then a message tag.

    Return None without raising if evidence is missing or malformed. A refusal
    without measurements must not produce a capacity-certified audit row.
    """
    attribute = getattr(source, "measured", None)
    if isinstance(attribute, Mapping):
        try:
            return validate_measured(attribute)
        except (TypeError, ValueError):
            log.warning("capacity refusal carried a malformed measured attribute")
    text = source if isinstance(source, str) else str(source)
    match = _MEASURED_RE.search(text)
    if match is None:
        return None
    try:
        return validate_measured(json.loads(match.group(1)))
    except (TypeError, ValueError):
        log.warning("capacity refusal carried a malformed measured tag")
        return None


def certificate_digest(rank_digests: Sequence[str]) -> str:
    """Bind the per-rank digests in rank order.

    Sorting would hide a rank/digest swap, the divergence this digest catches,
    so it keeps the order ``call_all`` returned.
    """
    return hashlib.sha256(
        json.dumps(list(rank_digests), separators=(",", ":")).encode()).hexdigest()


def _rank_certificate(row: object) -> dict[str, Any] | None:
    if not isinstance(row, Mapping):
        return None
    certificate = row.get("certificate")
    if not isinstance(certificate, Mapping):
        return None
    if any(key not in certificate for key in CERTIFICATE_WIRE_KEYS):
        return None
    return dict(certificate)


def fold_rank_certificates(
    rows: Sequence[Any] | None, expected_ranks: int | None = None,
) -> dict[str, Any] | None:
    """Summarize every rank's byte-verify certificate, or refuse the evidence.

    Returns None (never raises) when any rank is missing a certificate, when
    the rank count disagrees with the world, when a rank reports an incomplete
    verification, or when the ranks do not agree on which artifact they read.
    A worker too old to produce a certificate has none, so a missing key is a
    silent incomplete rather than an error.

    The cross-rank binding is content derived: ``artifact_signature`` plus
    ``checkpoint_bytes``. ``file_identity`` is ``dev:ino:size:mtime:ctime`` of a
    local copy, so on a two-box cluster it differs between boxes and cannot
    bind; it is kept as descriptive rank-0 provenance only. Counts are summed
    rather than required equal because an FSDP rank certifies its own shard.
    """
    if not rows:
        return None
    certificates = [_rank_certificate(row) for row in rows]
    if any(certificate is None for certificate in certificates):
        return None
    complete: list[dict[str, Any]] = [c for c in certificates if c is not None]
    if expected_ranks is not None and (
            not isinstance(expected_ranks, int) or len(complete) != expected_ranks):
        return None
    signature = complete[0].get("artifact_signature")
    if not isinstance(signature, str) or signature in SIGNATURE_SENTINELS:
        return None
    try:
        algorithm = check_member(
            complete[0].get("algorithm"), "algorithm", CERTIFICATE_ALGORITHMS)
        checkpoint_bytes = check_count(complete[0].get("checkpoint_bytes"), "checkpoint_bytes")
        digests = [check_hex(c.get("digest"), "digest", 64) for c in complete]
        totals = {
            field: sum(check_count(c.get(field), field) for c in complete)
            for field in ("tensors_verified", "tensors_total", "bytes_verified")
        }
    except (TypeError, ValueError):
        return None
    for certificate in complete:
        if (certificate.get("complete") is not True
                or certificate.get("algorithm") != algorithm
                or certificate.get("artifact_signature") != signature
                or certificate.get("checkpoint_bytes") != checkpoint_bytes):
            return None
    if totals["tensors_verified"] != totals["tensors_total"]:
        return None
    return {
        "algorithm": algorithm,
        "digest": certificate_digest(digests),
        "complete": True,
        "ranks_certified": len(complete),
        # The world this evidence was measured against, not the fold count:
        # emitting the count for both would make `validate_certificate`'s
        # ranks_expected == ranks_certified check compare a value with itself,
        # and a reader would take the world size to be whatever came back.
        "ranks_expected": (expected_ranks if isinstance(expected_ranks, int)
                           and not isinstance(expected_ranks, bool) else len(complete)),
        "tensors_verified": totals["tensors_verified"],
        "tensors_total": totals["tensors_total"],
        "bytes_verified": totals["bytes_verified"],
        "checkpoint_bytes": checkpoint_bytes,
        "artifact_signature": signature,
        "file_identity": str(complete[0].get("file_identity", "")),
    }


def validate_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """The folded summary a CAPACITY_CERTIFIED row may carry.

    The row carries this summary, never per-tensor data: every lookup scans
    every ledger line, so a per-tensor payload there would be a performance
    and integrity liability. Each rank's own certificate stays in its slab
    telemetry (``actor/slab.py`` ``slab_cert``).
    """
    if not isinstance(certificate, Mapping):
        raise TypeError("certificate must be a mapping")
    row: dict[str, Any] = {
        "algorithm": check_member(
            certificate.get("algorithm"), "certificate.algorithm", CERTIFICATE_ALGORITHMS),
        "digest": check_hex(certificate.get("digest"), "certificate.digest", 64),
        "complete": certificate.get("complete"),
        "ranks_certified": check_count(
            certificate.get("ranks_certified"), "certificate.ranks_certified", minimum=1),
        "ranks_expected": check_count(
            certificate.get("ranks_expected"), "certificate.ranks_expected", minimum=1),
        "tensors_verified": check_count(
            certificate.get("tensors_verified"), "certificate.tensors_verified"),
        "tensors_total": check_count(
            certificate.get("tensors_total"), "certificate.tensors_total"),
        "bytes_verified": check_count(
            certificate.get("bytes_verified"), "certificate.bytes_verified"),
        "checkpoint_bytes": check_count(
            certificate.get("checkpoint_bytes"), "certificate.checkpoint_bytes"),
        "artifact_signature": check_text(
            certificate.get("artifact_signature"), "certificate.artifact_signature"),
        "file_identity": str(certificate.get("file_identity", "")),
    }
    if row["complete"] is not True:
        raise ValueError("certificate.complete must be true; an incomplete load must refuse")
    if row["ranks_expected"] != row["ranks_certified"]:
        raise ValueError("certificate.ranks_expected must equal certificate.ranks_certified")
    if row["tensors_verified"] != row["tensors_total"]:
        raise ValueError("certificate.tensors_verified must equal certificate.tensors_total")
    if row["artifact_signature"] in SIGNATURE_SENTINELS:
        raise ValueError("certificate.artifact_signature is a read-failure sentinel")
    return row
