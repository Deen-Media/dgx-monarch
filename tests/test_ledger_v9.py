"""The WAIVER and CAPACITY_CERTIFIED audit rows, added at protocol v9.

Both row kinds are permanent evidence and never authority. These tests pin
three separate properties that keep them out of trust decisions (the audit key
namespace, a `record` discriminator in the context, a verdict the lookup does
not grant), the row shapes a single-use protocol number freezes, and the write
order: a grant whose row cannot be written is refused.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

import dgx_monarch.gate_ledger as ledger_mod
import dgx_monarch.nodes.gate as gate_mod
from dgx_monarch import gate_audit
from dgx_monarch.gate_audit_vocab import (
    ACTIONS,
    AUDIT_KEY_PREFIX,
    CERTIFICATE_ALGORITHMS,
    CONSENT_SOURCES,
    KIND_CLASS,
    KIND_GUARDS,
    MEASURED_PROBES,
    REASON_MAX_CHARS,
    RESIDENCY_KINDS,
    REVOKED_BY,
    WAIVER_KINDS,
)
from dgx_monarch.gate_ledger import GateLedger, artifact_set_signature

COMMIT = "9a9fdb10ab12"
COMBO = "3a91c0d4e77b21f508ac6b39"
CONSENT = "9f2c4ab17d0e4c1fa3b58e6d21c7f480"
FILE_IDENTITY = "66306:12583041:66266529792:1754300000000000000:1754300000000000000"
SIGNATURE = "9d1c0b47ee2a6f3184c5b70a"
MEMO_CONTEXT = {"combo_key": COMBO, "weight_dtype": "default"}
TRUST_CONTEXT = {
    "worker_args": {"slab_weights": False, "lora_low_rss": True},
    "world": 2, "hosts": 2, "gpus_per_host": 1, "resolved_topology": "uly2",
}


def _artifacts():
    return artifact_set_signature(["b" * 24, "c" * 24])


def _waiver_fields(**overrides):
    fields = {
        "combo_key": COMBO,
        "waiver_kind": "rescue-slab",
        "target_guard": "stock_load_preflight",
        "consent_id": CONSENT,
        "consent_source": "panel",
        "memo_context": MEMO_CONTEXT,
        "capability_context": TRUST_CONTEXT,
        "file_identity": FILE_IDENTITY,
        "unet_name": "big_model.safetensors",
        "loras": 0,
        "reason": "Stock residency cannot fit this 61.7 GiB checkpoint in 41.2 GiB of "
                  "available unified memory.",
    }
    fields.update(overrides)
    return fields


def _certificate(**overrides):
    certificate = {
        "algorithm": "memcmp-tensor-sampled-v1",
        "digest": "e4" + "a" * 62,
        "complete": True,
        "ranks_certified": 1,
        "ranks_expected": 1,
        "tensors_verified": 1042,
        "tensors_total": 1042,
        "bytes_verified": 66266529792,
        "checkpoint_bytes": 66266529792,
        "artifact_signature": SIGNATURE,
        "file_identity": FILE_IDENTITY,
    }
    certificate.update(overrides)
    return certificate


def _measured(**overrides):
    measured = {
        "probe": "driver_footprint_preflight",
        "checkpoint_bytes": 66266529792,
        "mem_available_bytes": 41234567168,
        "driver_projection_bytes": 32212254720,
        "worker_projection_bytes": None,
        "headroom_bytes": -57244221440,
    }
    measured.update(overrides)
    return measured


def _rank(**overrides):
    return {"rank": 0, "certificate": _certificate(**overrides)}


def test_waiver_row_round_trips_every_field(tmp_path):
    led = GateLedger(str(tmp_path))
    assert gate_audit.record_waiver(
        led, _artifacts(), COMMIT, action="grant", **_waiver_fields())
    row = led.entries()[0]
    assert GateLedger._valid_entry_schema(row)
    assert row["key"] == f"{AUDIT_KEY_PREFIX}waiver:rescue-slab:{COMBO}"
    assert row["verdict"] == "WAIVER"
    assert row["artifacts"] == _artifacts().current
    assert row["comfy"] == COMMIT
    assert row["gate_protocol"] == ledger_mod.GATE_PROTOCOL_VERSION == 13
    assert row["dgx_monarch"] == ledger_mod.__version__
    assert row["waiver_kind"] == "rescue-slab"
    assert row["waiver_class"] == "C"
    assert row["target_guard"] == "stock_load_preflight"
    assert row["action"] == "grant"
    assert row["consent_id"] == CONSENT
    assert row["consent_source"] == "panel"
    assert row["file_identity"] == FILE_IDENTITY
    assert row["unet_name"] == "big_model.safetensors"
    assert row["loras"] == 0
    assert row["reason"].startswith("Stock residency cannot fit")
    assert isinstance(row["waived_at"], float)
    # The memo fingerprint is derived from the narrow memo context, and the
    # full trust context is recorded verbatim beside it for audit.
    assert row["memo_fingerprint"] == gate_audit.context_fingerprint(MEMO_CONTEXT)
    assert json.loads(row["memo_context"]) == MEMO_CONTEXT
    assert json.loads(row["target_context"]) == TRUST_CONTEXT
    # The row's own capability context is the audit context, marked by `record`.
    assert json.loads(row["capability_context"]) == {
        "record": "waiver", "waiver_kind": "rescue-slab",
        "target_guard": "stock_load_preflight",
        "memo_fingerprint": row["memo_fingerprint"]}


def test_use_and_revoke_rows_carry_their_extra_fields(tmp_path):
    led = GateLedger(str(tmp_path))
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="use", run_id="run-7",
                             **_waiver_fields(waiver_kind="waive-known-wrong:ring-pad",
                                              target_guard="ring_pad:minimax_h3",
                                              stamp="rendered-under-waiver"))
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="revoke",
                             revoked_by="gate_fail", **_waiver_fields())
    use, revoke = led.entries()
    assert (use["action"], use["run_id"], use["stamp"], use["waiver_class"]) == (
        "use", "run-7", "rendered-under-waiver", "K")
    assert use["target_guard"] == "ring_pad:minimax_h3"
    assert (revoke["action"], revoke["revoked_by"]) == ("revoke", "gate_fail")
    # Revocation appends; it never deletes the grant it revokes.
    assert len(led.entries()) == 2


@pytest.mark.parametrize("overrides", [
    {"waiver_kind": "rescue-everything"},
    {"target_guard": "some_other_guard"},
    {"target_guard": None},
    {"consent_id": "not-hex"},
    {"consent_id": CONSENT[:8]},
    {"consent_source": "telepathy"},
    {"reason": "x" * (REASON_MAX_CHARS + 1)},
    {"reason": 17},
    {"reason": ""},
    {"unet_name": ""},
    {"file_identity": None},
    {"loras": -1},
    {"loras": "two"},
    {"waiver_kind": "auto-rescue-standing", "target_guard": "stock_load_preflight"},
    {"waiver_kind": "waive-known-wrong:ring-pad", "target_guard": "ring_pad:Not A Family"},
    {"waiver_kind": "waive-known-wrong:ring-pad", "target_guard": "sp_unvalidated"},
])
def test_waiver_builder_refuses_malformed_input(overrides):
    with pytest.raises((ValueError, TypeError)):
        gate_audit.build_waiver_row(action="grant", **_waiver_fields(**overrides))


def test_waiver_builder_refuses_a_missing_conditional_field():
    with pytest.raises(ValueError):  # action="use" needs the run id
        gate_audit.build_waiver_row(action="use", **_waiver_fields())
    with pytest.raises(ValueError):  # class K needs the output stamp
        gate_audit.build_waiver_row(
            action="grant", **_waiver_fields(waiver_kind="waive-known-wrong:pixeldit-sp",
                                             target_guard="sp_unvalidated:pixeldit_comfy"))
    with pytest.raises(ValueError):  # revoke needs a known revoker
        gate_audit.build_waiver_row(action="revoke", revoked_by="vibes", **_waiver_fields())


def test_waiver_class_is_derived_not_accepted():
    parameters = inspect.signature(gate_audit.build_waiver_row).parameters
    assert "waiver_class" not in parameters
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY
               for parameter in parameters.values())
    assert set(KIND_CLASS) == set(WAIVER_KINDS) == set(KIND_GUARDS)
    assert set(KIND_CLASS.values()) == {"C", "U", "K"}  # never P: physics has no bypass
    for kind, expected in KIND_CLASS.items():
        guard = ("ring_pad" if kind == "waive-known-wrong:ring-pad" else
                 "sp_unvalidated" if kind == "waive-known-wrong:pixeldit-sp" else
                 "sol_attn" if kind == "waive-known-wrong:sol-attn" else
                 "shard_quant_scale" if kind == "waive-known-wrong:shard-quant" else
                 next(iter(KIND_GUARDS[kind]), None))
        row = gate_audit.build_waiver_row(
            action="grant", **_waiver_fields(
                waiver_kind=kind, target_guard=guard,
                **({"stamp": "rendered-under-waiver"} if expected == "K" else {})))
        assert row.detail["waiver_class"] == expected


def test_the_frozen_vocabulary_is_complete():
    """Every enum member ships with the protocol bump that introduces it.

    A member added later burns the current number and must ship the next one,
    so the vocabulary is pinned here. It has happened twice:
    ``waive-known-wrong:sol-attn`` burned v10 and shipped in v11 (2026-08-15),
    and ``waive-known-wrong:shard-quant`` burned v12 and shipped in v13
    (2026-09-05).
    """
    assert set(WAIVER_KINDS) == {
        "rescue-slab", "auto-rescue-standing",
        "waive-preflight:activation_footprint", "waive-preflight:driver_footprint",
        "waive-first-load-stock", "waive-unvouched-slab",
        "waive-known-wrong:ring-pad", "waive-known-wrong:pixeldit-sp",
        "waive-known-wrong:sol-attn", "waive-known-wrong:shard-quant"}
    assert set(ACTIONS) == {"grant", "use", "revoke"}
    assert set(CONSENT_SOURCES) == {"panel", "env", "auto_rescue", "cli"}
    assert set(REVOKED_BY) == {"user", "gate_fail", "file_change"}
    assert set(RESIDENCY_KINDS) == {"slab"}
    assert "driver_footprint_preflight" in MEASURED_PROBES
    assert "driver_footprint_preflight" in KIND_GUARDS["rescue-slab"]
    assert len(CERTIFICATE_ALGORITHMS) == 6


def test_the_artifacts_column_must_be_the_artifact_set_digest(tmp_path):
    """A per-file signature (24 hex) here would match no gate row's artifact set."""
    led = GateLedger(str(tmp_path))
    with pytest.raises(ValueError):
        gate_audit.record_waiver(led, SIGNATURE, COMMIT, action="grant", **_waiver_fields())
    assert led.entries() == []
    assert gate_audit.record_waiver(
        led, _artifacts().current, COMMIT, action="grant", **_waiver_fields())


def test_audit_context_can_never_equal_a_trust_context():
    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}, hosts=(), source="/configs/cluster.toml"),
        config_fingerprint="config-a", world=2, n_hosts=2, gpus_per_host=1,
    )
    model = SimpleNamespace(mesh=SimpleNamespace(
        handle=handle, worker_args={"slab_weights": True, "lora_low_rss": True},
        topology_preset="uly2", attention="TORCH_FLASH", sync_ulysses=True))
    trust_contexts = [
        gate_mod.gate_capability_context(model, handle),
        gate_mod.fleet_residency_capability_context(model, handle),
        # render_quarantine.py builds this fallback shape with no handle.
        {"worker_args": {"slab_weights": True}},
    ]
    audit_contexts = [
        gate_audit.waiver_audit_context("rescue-slab", "stock_load_preflight", "f" * 64),
        gate_audit.capacity_audit_context("f" * 64),
    ]
    encode = ledger_mod._canonical_capability_context
    for trust in trust_contexts:
        assert "record" not in trust
        for audit in audit_contexts:
            assert encode(trust) != encode(audit)


def test_audit_rows_are_trust_neutral(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record(COMBO, _artifacts(), COMMIT, "PASS", {"model": "m"}, TRUST_CONTEXT)
    for _ in range(3):
        gate_audit.record_waiver(led, _artifacts(), COMMIT, action="grant", **_waiver_fields())
        gate_audit.record_rescue_certificate(
            led, COMBO, _artifacts(), COMMIT, capability_context=TRUST_CONTEXT,
            certificate=_certificate(), measured=_measured(), consent_id=CONSENT,
            unet_name="big_model.safetensors", file_identity=FILE_IDENTITY)
    result = led.lookup_with_integrity(COMBO, _artifacts(), COMMIT, TRUST_CONTEXT)
    assert result.state == "pass"
    assert result.entry["verdict"] == "PASS"
    assert result.session_pass_safe is True


def test_audit_rows_are_not_damage(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record(COMBO, _artifacts(), COMMIT, "FAIL",
               {"quarantine_levers": ["slab_weights"]}, TRUST_CONTEXT)
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="grant", **_waiver_fields())
    result = led.lookup_with_integrity(COMBO, _artifacts(), COMMIT, TRUST_CONTEXT)
    assert (result.state, result.session_pass_safe) == ("fail", True)
    assert result.entry["quarantine_levers"] == ["slab_weights"]
    recovered = led.matching_entry(COMBO, _artifacts(), TRUST_CONTEXT, verdict="FAIL")
    assert recovered["quarantine_levers"] == ["slab_weights"]


def test_audit_context_lookup_never_grants(tmp_path):
    led = GateLedger(str(tmp_path))
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="grant", **_waiver_fields())
    row = led.entries()[0]
    audit_context = json.loads(row["capability_context"])
    # Even asked with the audit context itself, under the audit key, the
    # unrecognized verdict falls through to "unknown", never a grant.
    assert led.lookup(row["key"], _artifacts(), COMMIT, audit_context) == "unknown"


def test_waiver_row_never_becomes_latest_for_a_gated_combo(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record(COMBO, _artifacts(), COMMIT, "PASS", {"model": "m"}, TRUST_CONTEXT)
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="grant", **_waiver_fields())
    gate_audit.record_rescue_certificate(
        led, COMBO, _artifacts(), COMMIT, capability_context=TRUST_CONTEXT,
        certificate=_certificate(), measured=_measured(), consent_id=CONSENT,
        unet_name="big_model.safetensors", file_identity=FILE_IDENTITY)
    assert led.lookup(COMBO, _artifacts(), COMMIT, TRUST_CONTEXT) == "pass"
    assert led.matching_entry(COMBO, _artifacts())["verdict"] == "PASS"
    # The three summary consumers group by key and count verdicts. Audit rows
    # are not verdicts; counting one would flip the panel's PASS chip to WAIVER.
    latest = {row["key"]: row for row in gate_audit.trust_rows(led.entries())}
    counts = {}
    for row in latest.values():
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    assert counts == {"PASS": 1}
    assert len(gate_audit.audit_rows(led.entries())) == 2


def test_gate_list_cli_does_not_print_audit_rows(tmp_path, capsys):
    from dgx_monarch.cli import main as cli

    led = GateLedger(str(tmp_path))
    led.record(COMBO, _artifacts(), COMMIT, "PASS",
               {"model": "big_model.safetensors", "loras": 0}, TRUST_CONTEXT)
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="grant", **_waiver_fields())
    assert cli._cmd_gate_impl(
        SimpleNamespace(list=True, report_dir=str(tmp_path))) == 0
    printed = capsys.readouterr().out
    assert "PASS" in printed
    assert "WAIVER" not in printed


def test_certificate_fold_summarizes_every_rank():
    folded = gate_audit.fold_rank_certificates([_rank(), _rank()], 2)
    assert folded["ranks_certified"] == folded["ranks_expected"] == 2
    assert folded["tensors_verified"] == folded["tensors_total"] == 2084
    assert folded["bytes_verified"] == 2 * 66266529792
    assert folded["checkpoint_bytes"] == 66266529792
    assert folded["artifact_signature"] == SIGNATURE
    assert folded["complete"] is True
    assert folded["digest"] == gate_audit.certificate_digest([_certificate()["digest"]] * 2)


@pytest.mark.parametrize("rows,expected_ranks", [
    ([{"rank": 0}], None),                                        # v8 worker: no key
    ([_rank(), {"rank": 1}], 2),                                  # one rank missing
    ([_rank()], 2),                                               # short of the world
    ([_rank(), _rank(artifact_signature="0" * 24)], 2),           # different artifact
    ([_rank(), _rank(checkpoint_bytes=1024)], 2),                 # different size
    ([_rank(), _rank(algorithm="sha256-tensor-v1")], 2),          # different algorithm
    ([_rank(complete=False)], 1),                                 # incomplete verification
    ([_rank(digest="short")], 1),                                 # unusable digest
    ([_rank(tensors_verified=3)], 1),                             # a tensor was skipped
    ([_rank(algorithm="rot13")], 1),                              # unknown algorithm
    ([], 1),
    (None, 1),
])
def test_certificate_fold_refuses_incomplete_evidence(rows, expected_ranks):
    assert gate_audit.fold_rank_certificates(rows, expected_ranks) is None


@pytest.mark.parametrize("sentinel", ["unreadable", "unstable"])
def test_certificate_fold_rejects_signature_sentinels(sentinel):
    rows = [_rank(artifact_signature=sentinel), _rank(artifact_signature=sentinel)]
    assert gate_audit.fold_rank_certificates(rows, 2) is None


def test_certificate_fold_succeeds_across_hosts_with_differing_file_identity():
    """Each host stats its own copy, so dev:ino:ctime differ by construction.

    Binding on file_identity would make a two-host certificate impossible.
    """
    rows = [_rank(), _rank(file_identity="66307:99:66266529792:17:17")]
    folded = gate_audit.fold_rank_certificates(rows, 2)
    assert folded is not None
    assert folded["file_identity"] == FILE_IDENTITY  # rank 0, descriptive only


def test_certificate_digest_is_order_bound_and_content_bound():
    reordered = dict(reversed(list(_certificate().items())))
    assert gate_audit.fold_rank_certificates([{"certificate": reordered}], 1)["digest"] == (
        gate_audit.fold_rank_certificates([_rank()], 1)["digest"])
    left, right = "a" * 64, "b" * 64
    assert gate_audit.certificate_digest([left, right]) != gate_audit.certificate_digest(
        [right, left])
    assert gate_audit.certificate_digest([left]) != gate_audit.certificate_digest([left, left])


def test_capacity_row_records_the_numbers_and_the_certificate(tmp_path):
    led = GateLedger(str(tmp_path))
    assert gate_audit.record_rescue_certificate(
        led, COMBO, _artifacts(), COMMIT, capability_context=TRUST_CONTEXT,
        certificate=_certificate(), measured=_measured(), consent_id=CONSENT,
        unet_name="big_model.safetensors", file_identity=FILE_IDENTITY, run_id="run-9")
    row = led.entries()[0]
    assert GateLedger._valid_entry_schema(row)
    assert row["key"] == f"{AUDIT_KEY_PREFIX}capacity:{COMBO}"
    assert row["verdict"] == "CAPACITY_CERTIFIED"
    assert (row["residency"], row["slab_resident"], row["stock_available"]) == (
        "slab", True, False)
    assert row["certificate_origin"] == "rescue_load"
    assert row["consent_id"] == CONSENT
    assert row["run_id"] == "run-9"
    assert row["measured"]["headroom_bytes"] < 0
    assert row["measured"]["worker_projection_bytes"] is None
    assert row["certificate"]["bytes_verified"] == 66266529792
    assert row["capability_fingerprint"] == gate_audit.context_fingerprint(TRUST_CONTEXT)
    assert json.loads(row["capability_context"])["record"] == "capacity_certificate"


@pytest.mark.parametrize("kwargs", [
    {"consent_id": None},                                            # rescue with no consent
    {"certificate": _certificate(complete=False)},
    {"certificate": _certificate(tensors_verified=1)},
    {"certificate": _certificate(ranks_expected=2)},
    {"certificate": _certificate(artifact_signature="unstable")},
    {"measured": _measured(checkpoint_bytes=None)},
    {"measured": _measured(mem_available_bytes=None)},
    {"measured": _measured(probe="vibes")},
    {"measured": _measured(headroom_bytes=1.5)},
    {"unet_name": ""},
    {"file_identity": ""},
])
def test_capacity_row_builder_refuses_dishonest_shapes(kwargs):
    fields = {
        "combo_key": COMBO, "capability_context": TRUST_CONTEXT,
        "certificate": _certificate(), "measured": _measured(),
        "certificate_origin": "rescue_load", "unet_name": "big_model.safetensors",
        "file_identity": FILE_IDENTITY, "consent_id": CONSENT,
    }
    fields.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        gate_audit.build_capacity_row(**fields)


def test_a_ceremony_certificate_never_invents_a_consent():
    with pytest.raises(ValueError):
        gate_audit.build_capacity_row(
            combo_key=COMBO, capability_context=TRUST_CONTEXT, certificate=_certificate(),
            measured=_measured(), certificate_origin="gate_cross_leg",
            unet_name="m.safetensors", file_identity=FILE_IDENTITY, consent_id=CONSENT)
    row = gate_audit.build_capacity_row(
        combo_key=COMBO, capability_context=TRUST_CONTEXT, certificate=_certificate(),
        measured=_measured(), certificate_origin="gate_cross_leg",
        unet_name="m.safetensors", file_identity=FILE_IDENTITY, origin="auto_first_use")
    assert "consent_id" not in row.detail
    assert row.detail["origin"] == "auto_first_use"


def test_the_builders_cannot_claim_stock_was_available():
    parameters = inspect.signature(gate_audit.build_capacity_row).parameters
    assert "slab_resident" not in parameters and "stock_available" not in parameters
    assert "residency" not in parameters


@pytest.mark.parametrize("failure", ["returns_false", "raises_type_error"])
def test_grant_refuses_when_the_row_cannot_be_written(tmp_path, failure):
    led = GateLedger(str(tmp_path))
    if failure == "returns_false":
        led.record = lambda *args, **kwargs: False
    else:
        def _raise(*args, **kwargs):
            raise TypeError("capability context must contain only JSON-native values")
        led.record = _raise
    with pytest.raises(gate_audit.WaiverNotAuditedError, match="not granted"):
        gate_audit.record_waiver_grant(led, _artifacts(), COMMIT, **_waiver_fields())
    assert led.entries() == []


def test_a_written_grant_returns_the_detail_the_memo_is_built_from(tmp_path):
    led = GateLedger(str(tmp_path))
    detail = gate_audit.record_waiver_grant(led, _artifacts(), COMMIT, **_waiver_fields())
    assert detail["consent_id"] == CONSENT
    assert detail["action"] == "grant"
    assert detail["memo_fingerprint"] == gate_audit.context_fingerprint(MEMO_CONTEXT)
    assert led.entries()[0]["consent_id"] == CONSENT


def test_measured_numbers_survive_monarch_text_wrapping():
    from dgx_monarch.mesh_safety import StockLoadCapacityError

    refusal = StockLoadCapacityError(
        "stock residency cannot load big_model.safetensors "
        + gate_audit.measured_tag(_measured(probe="stock_load_preflight")))
    wrapped = RuntimeError(f"ActorError: StockLoadCapacityError: {refusal}")
    assert gate_audit.parse_measured(wrapped) == _measured(probe="stock_load_preflight")
    assert gate_audit.parse_measured(RuntimeError("no numbers here")) is None
    assert gate_audit.parse_measured(
        RuntimeError("[dgxm:measured {\"probe\": \"vibes\"}]")) is None


def test_measured_attribute_beats_the_tag_and_a_malformed_one_falls_back():
    tagged = SimpleNamespace(measured={"probe": "not a probe"})
    tagged.__str__ = lambda self=None: ""  # type: ignore[assignment]
    assert gate_audit.parse_measured(tagged) is None
    carrier = RuntimeError("wrapped " + gate_audit.measured_tag(_measured()))
    carrier.measured = _measured(probe="worker_stock_load")  # type: ignore[attr-defined]
    assert gate_audit.parse_measured(carrier)["probe"] == "worker_stock_load"


def test_a_certified_load_that_later_fails_still_quarantines(tmp_path):
    """Certificates and consents never override a quarantine.

    The waiver records what was allowed, the certificate records that the
    bytes were verified, and the FAIL records what was later proven unsafe.
    The trust lookup selects the FAIL whether audit rows land before or after it.
    """
    led = GateLedger(str(tmp_path))
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="grant", **_waiver_fields())
    gate_audit.record_rescue_certificate(
        led, COMBO, _artifacts(), COMMIT, capability_context=TRUST_CONTEXT,
        certificate=_certificate(), measured=_measured(), consent_id=CONSENT,
        unet_name="big_model.safetensors", file_identity=FILE_IDENTITY)
    led.record(COMBO, _artifacts(), COMMIT, "FAIL",
               {"quarantine_levers": ["slab_weights"]}, TRUST_CONTEXT)
    gate_audit.record_waiver(led, _artifacts(), COMMIT, action="revoke",
                             revoked_by="gate_fail", **_waiver_fields())
    result = led.lookup_with_integrity(COMBO, _artifacts(), COMMIT, TRUST_CONTEXT)
    assert result.state == "fail"
    assert result.entry["quarantine_levers"] == ["slab_weights"]
    # The grant row is still on disk: revocation appends, it never deletes.
    grants = [row for row in gate_audit.audit_rows(led.entries())
              if row.get("action") == "grant"]
    assert len(grants) == 1
