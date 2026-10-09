"""Gate ledger semantics: a verdict binds to the identity that gate_ledger's module docstring
lists; a PASS reads stale when any part of it moves, and a FAIL quarantine stays sticky."""
import copy
import hashlib
import json
import os
import pickle
import sys
import types

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
from dgx_monarch.gate_ledger import (
    GateLedger,
    artifact_set_signature,
    artifact_signature,
    combo_key,
)


def test_combo_key_ignores_lora_order_and_detects_name_change():
    a = combo_key("m.safetensors", {"weight_dtype": "default"}, ["x", "y"])
    b = combo_key("m.safetensors", {"weight_dtype": "default"}, ["y", "x"])
    c = combo_key("m.safetensors", {"weight_dtype": "default"}, ["y", "z"])
    assert a == b and a != c


def test_combo_key_is_canonical_for_nested_options_and_binds_every_axis():
    left = combo_key(
        "model.safetensors",
        {"compile": {"mode": "default", "dynamic": False}, "dtype": "bf16"},
        ["style-b.safetensors", "style-a.safetensors"],
    )
    reordered = combo_key(
        "model.safetensors",
        {"dtype": "bf16", "compile": {"dynamic": False, "mode": "default"}},
        ["style-a.safetensors", "style-b.safetensors"],
    )
    assert left == reordered
    assert len(left) == 24
    assert left != combo_key("other.safetensors", {"dtype": "bf16"}, [])
    assert left != combo_key("model.safetensors", {"dtype": "fp16"}, [])
    assert left != combo_key("model.safetensors", {"dtype": "bf16"}, ["other.safetensors"])


def test_artifact_signature_tracks_bytes(tmp_path):
    p = tmp_path / "w.safetensors"
    p.write_bytes(torch.randn(1024).numpy().tobytes())
    s1 = artifact_signature(str(p))
    s2 = artifact_signature(str(p))
    data = p.read_bytes()
    expected = hashlib.sha256(str(len(data)).encode() + data * 3).hexdigest()[:24]
    assert s1 == s2 == expected
    data = bytearray(p.read_bytes())
    data[100] ^= 0xFF
    p.write_bytes(bytes(data))
    assert artifact_signature(str(p)) != s1     # same size, different bytes


def test_artifact_signature_bounds_io_to_sample_windows(tmp_path, monkeypatch):
    p = tmp_path / "large.safetensors"
    data = bytearray(5 * 1024 * 1024)
    p.write_bytes(data)
    advised = []
    reads = []
    real_pread = os.pread

    def tracked_pread(fd, length, offset):
        reads.append((offset, length))
        return real_pread(fd, length, offset)

    monkeypatch.setattr(ledger_mod.os, "pread", tracked_pread)
    monkeypatch.setattr(
        ledger_mod.os,
        "posix_fadvise",
        lambda _fd, offset, length, advice: advised.append((offset, length, advice)),
    )
    before = artifact_signature(str(p))
    assert {offset for offset, _, _ in advised} == {
        0,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
    }
    assert sum(length for _, length, _ in advised) == 3 * 1024 * 1024
    assert all(advice == os.POSIX_FADV_DONTNEED for _, _, advice in advised)
    assert sum(length for _, length in reads) == 3 * 1024 * 1024

    # The bounded fingerprint reads only its sample windows; other bytes do not count.
    data[1_250_000] = 1
    p.write_bytes(data)
    assert artifact_signature(str(p)) == before


def test_artifact_signature_fails_closed_on_a_short_sample_read(tmp_path, monkeypatch):
    p = tmp_path / "truncated-read.safetensors"
    p.write_bytes(b"x" * 4096)
    monkeypatch.setattr(ledger_mod.os, "pread", lambda *_args: b"")
    assert artifact_signature(str(p)) == "unreadable"


def test_artifact_signature_missing_path_is_unreadable(tmp_path):
    assert artifact_signature(str(tmp_path / "missing.safetensors")) == "unreadable"


def test_cached_signature_rechecks_path_after_atomic_replacement(tmp_path, monkeypatch):
    path = tmp_path / "model.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    old_data = b"A" * 4096
    new_data = b"B" * 4096
    path.write_bytes(old_data)
    replacement.write_bytes(new_data)
    old_signature = artifact_signature(str(path))
    expected = hashlib.sha256(str(len(new_data)).encode() + new_data * 3).hexdigest()[:24]
    real_fstat = os.fstat
    replace_on_first_stat = True

    def replacing_fstat(fd):
        nonlocal replace_on_first_stat
        stat = real_fstat(fd)
        if replace_on_first_stat:
            replace_on_first_stat = False
            replacement.replace(path)
        return stat

    monkeypatch.setattr(ledger_mod.os, "fstat", replacing_fstat)
    assert artifact_signature(str(path)) == expected
    assert expected != old_signature


def test_artifact_set_signature_binds_every_entry_without_truncation():
    prefix = ["a" * 64, "b" * 64]
    left = artifact_set_signature([*prefix, "c" * 64])
    right = artifact_set_signature([*prefix, "d" * 64])
    assert left.current != right.current


def test_artifact_set_signature_is_an_explicit_copyable_value():
    artifacts = artifact_set_signature(["a" * 24, "b" * 24])
    assert not isinstance(artifacts, str)
    assert copy.deepcopy(artifacts) == artifacts
    restored = pickle.loads(pickle.dumps(artifacts))  # noqa: S301 - trusted round-trip
    assert restored == artifacts


def test_artifact_set_signature_pins_current_digest_and_legacy_boundary():
    signatures = ["a" * 24, "b" * 24]
    artifacts = artifact_set_signature(signatures)
    payload = json.dumps(signatures, separators=(",", ":"), ensure_ascii=True)
    assert artifacts.current == hashlib.sha256(payload.encode()).hexdigest()
    assert artifact_set_signature(["x" * 96]).legacy_complete
    assert not artifact_set_signature(["x" * 97]).legacy_complete


def test_comfy_commit_reads_loose_packed_and_detached_git_state(tmp_path, monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.base_path = str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    git = tmp_path / ".git"
    refs = git / "refs" / "heads"
    refs.mkdir(parents=True)
    head = git / "HEAD"

    head.write_text("ref: refs/heads/main\n")
    (refs / "main").write_text("1" * 40 + "\n")
    assert ledger_mod.comfy_commit() == "1" * 12

    (refs / "main").unlink()
    (git / "packed-refs").write_text("2" * 40 + " refs/heads/main\n")
    assert ledger_mod.comfy_commit() == "2" * 12

    head.write_text("3" * 40 + "\n")
    assert ledger_mod.comfy_commit() == "3" * 12

    head.unlink()
    assert ledger_mod.comfy_commit() == "unknown"


def test_lookup_accepts_complete_pre_upgrade_artifact_aggregate(tmp_path):
    signatures = ["a" * 24, "b" * 24, "c" * 24]
    artifacts = artifact_set_signature(signatures)
    led = GateLedger(str(tmp_path))

    assert len(artifacts.current) == 64
    assert artifacts.legacy_complete
    led.record("k1", artifacts.legacy, "c1", "PASS")
    assert led.lookup("k1", artifacts, "c1") == "pass"


def test_incomplete_legacy_aggregate_preserves_fail_but_not_pass(tmp_path):
    signatures = ["a" * 24, "b" * 24, "c" * 24, "d" * 24, "e" * 24]
    artifacts = artifact_set_signature(signatures)
    changed = artifact_set_signature([*signatures[:-1], "f" * 24])
    led = GateLedger(str(tmp_path))

    assert not artifacts.legacy_complete
    assert artifacts.current != changed.current
    assert artifacts.legacy == changed.legacy
    led.record("k1", artifacts.legacy, "c1", "PASS")
    assert led.lookup("k1", artifacts, "c1") == "stale"
    assert led.lookup("k1", changed, "c1") == "stale"
    led.record("k1", artifacts.legacy, "c1", "FAIL")
    assert led.lookup("k1", artifacts, "c1") == "fail"
    assert led.lookup("k1", changed, "c1") == "fail"


def test_record_normalizes_explicit_artifact_set_to_current_digest(tmp_path):
    artifacts = artifact_set_signature(["a" * 24])
    led = GateLedger(str(tmp_path))
    led.record("k1", artifacts, "c1", "PASS")
    assert led.entries()[0]["artifacts"] == artifacts.current


def test_lookup_semantics(tmp_path):
    led = GateLedger(str(tmp_path))
    assert led.lookup("k1", "sigA", "c1") == "unknown"
    led.record("k1", "sigA", "c1", "PASS")
    assert led.lookup("k1", "sigA", "c1") == "pass"
    assert led.lookup("k1", "sigB", "c1") == "stale"     # file changed
    assert led.lookup("k1", "sigA", "c2") == "stale"     # comfy changed
    assert led.lookup("k1", "sigA", "unknown") == "stale"  # cannot inherit PASS
    led.record("k1", "sigA", "c1", "FAIL")
    assert led.lookup("k1", "sigA", "c1") == "fail"      # last writer wins
    led.record("k1", "sigA", "c1", "INCONCLUSIVE")
    assert led.lookup("k1", "sigA", "c1") == "inconclusive"
    led.record("k1", "sigA", "c1", "PASS")
    assert led.lookup("k1", "sigA", "c1") == "pass"


def test_unknown_recorded_verdict_never_becomes_a_trust_grant(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("k", "sig", "commit", "MAYBE")
    state, entry = led.lookup_with_entry("k", "sig", "commit")
    assert state == "unknown"
    assert entry is not None and entry["verdict"] == "MAYBE"


def test_context_value_has_exact_canonical_encoding(tmp_path):
    led = GateLedger(str(tmp_path))
    context = {"world": 2, "worker_args": {"slab_weights": True, "compile_dit": False}}
    expected = '{"worker_args":{"compile_dit":false,"slab_weights":true},"world":2}'
    assert led._context_value(context) == expected
    assert led._context_value(expected) == expected


def test_pass_is_bound_to_protocol_version_and_capability_context(tmp_path):
    led = GateLedger(str(tmp_path))
    slab = {"worker_args": {"slab_weights": True, "lora_low_rss": True,
                            "compile_dit": False}, "world": 2}
    stock = {"worker_args": {"slab_weights": False, "lora_low_rss": True,
                             "compile_dit": False}, "world": 2}
    led.record("k", "artifact", "commit", "PASS", context=slab)
    assert led.lookup("k", "artifact", "commit", slab) == "pass"
    assert led.lookup("k", "artifact", "commit", stock) == "stale"
    changed_config = {**slab, "config_fingerprint": "different-cluster-toml"}
    assert led.lookup("k", "artifact", "commit", changed_config) == "stale"
    row = led.entries()[0]
    # The comment above GATE_PROTOCOL_VERSION in gate_ledger.py records why each
    # version from v4 on was added or burned. Older PASSes re-prove.
    assert ledger_mod.GATE_PROTOCOL_VERSION == 13
    assert row["gate_protocol"] == ledger_mod.GATE_PROTOCOL_VERSION
    assert row["dgx_monarch"]

    # A historical row without the capability identity never grants runtime
    # trust; context-free reporting still reads it.
    led.record("legacy", "artifact", "commit", "PASS")
    assert led.lookup("legacy", "artifact", "commit") == "pass"
    assert led.lookup("legacy", "artifact", "commit", slab) == "stale"


def test_v5_pass_without_frozen_transaction_inputs_reads_stale(tmp_path):
    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True}, "world": 2}
    row = {
        "key": "k",
        "artifacts": "artifact",
        "comfy": "commit",
        "verdict": "PASS",
        "gate_protocol": 5,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": led._context_value(context),
    }
    led.path = str(tmp_path / "dgxm_gate_ledger.jsonl")
    with open(led.path, "w") as f:
        f.write(json.dumps(row) + "\n")

    assert led.lookup("k", "artifact", "commit", context) == "stale"


def test_divergent_v6_lineages_read_stale_under_current_protocol(tmp_path):
    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True}, "world": 2}
    row = {
        "key": "k",
        "artifacts": "artifact",
        "comfy": "commit",
        "verdict": "PASS",
        "gate_protocol": 6,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": led._context_value(context),
    }
    with open(led.path, "w") as f:
        f.write(json.dumps(row) + "\n")

    assert led.lookup("k", "artifact", "commit", context) == "stale"
    led.record("fresh", "artifact", "commit", "PASS", context=context)
    assert led.lookup("fresh", "artifact", "commit", context) == "pass"


def test_burned_v4_rows_read_stale_not_current_trust(tmp_path):
    """Code committed on 2026-07-10 wrote v4 rows whose semantics were later rejected:
    slab_unreferenced PASS grants and on/auto siblings stamped in both directions. Both
    v4 rows must read stale under the current protocol, so auto-gating re-proves them
    instead of inheriting them."""
    import json

    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True, "lora_low_rss": True},
               "world": 2}
    context_value = led._context_value(context)
    unsafe_rows = [
        {"key": "k", "artifacts": "artifact", "comfy": "commit",
         "verdict": "PASS", "slab_unreferenced": True,
         "cross_mode": "UNREFERENCED", "gate_protocol": 4,
         "dgx_monarch": ledger_mod.__version__,
         "capability_context": context_value},
        {"key": "k2", "artifacts": "artifact", "comfy": "commit",
         "verdict": "PASS", "stamped": "slab-mode-equivalence",
         "gate_protocol": 4, "dgx_monarch": ledger_mod.__version__,
         "capability_context": context_value},
    ]
    led.path = str(tmp_path / "dgxm_gate_ledger.jsonl")
    with open(led.path, "w") as f:
        for row in unsafe_rows:
            f.write(json.dumps(row) + "\n")
    assert led.lookup("k", "artifact", "commit", context) == "stale"
    assert led.lookup("k2", "artifact", "commit", context) == "stale"
    # Row shape does not decide trust: the ledger never reads `stamped` or
    # `slab_unreferenced`, and nodes/gate_verdict.py still writes stamped sibling
    # rows at the current protocol. A fresh record still reads pass.
    led.record("k3", "artifact", "commit", "PASS", context=context)
    assert led.lookup("k3", "artifact", "commit", context) == "pass"


def test_recorded_unknown_commit_never_grants_pass(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("k1", "sigA", "unknown", "PASS")
    assert led.lookup("k1", "sigA", "c1") == "stale"
    assert led.lookup("k1", "sigA", "unknown") == "stale"


def test_contextual_pass_is_invalidated_by_package_version(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True}, "world": 2}
    led.record("k1", "sigA", "c1", "PASS", context=context)
    assert led.lookup("k1", "sigA", "c1", context) == "pass"

    monkeypatch.setattr(ledger_mod, "__version__", f"{ledger_mod.__version__}+next")
    assert led.lookup("k1", "sigA", "c1", context) == "stale"


def test_fail_quarantine_is_sticky_across_commit_uncertainty(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("k1", "sigA", "unknown", "FAIL")
    assert led.lookup("k1", "sigA", "c1") == "fail"
    assert led.lookup("k1", "sigA", "unknown") == "fail"
    assert led.lookup("k1", "sigB", "c1") == "stale"


def test_lookup_with_entry_pairs_state_with_its_exact_row(tmp_path):
    """Lever recovery reads verdict and row from one scan: a newer row moves
    state and entry together, never a FAIL verdict with another row's levers."""
    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True, "lora_low_rss": True},
               "world": 2}
    led.record("k", "sig", "commit", "FAIL",
               detail={"quarantine_levers": ["slab_weights"]}, context=context)
    state, entry = led.lookup_with_entry("k", "sig", "commit", context)
    assert state == "fail"
    assert entry["verdict"] == "FAIL"
    assert entry["quarantine_levers"] == ["slab_weights"]

    led.record("k", "sig", "commit", "INCONCLUSIVE", context=context)
    state, entry = led.lookup_with_entry("k", "sig", "commit", context)
    assert state == "inconclusive"
    assert entry["verdict"] == "INCONCLUSIVE"


def test_lookup_delegates_to_single_read_implementation(tmp_path):
    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True}, "world": 2}
    led.record("k", "sig", "commit", "PASS", context=context)
    assert led.lookup("k", "sig", "commit", context) == "pass"
    assert led.lookup_with_entry("k", "sig", "commit", context)[0] == "pass"
    assert led.lookup_with_entry("missing", "sig", "commit", context) == ("unknown", None)
    clean_missing = led.lookup_with_integrity("missing", "sig", "commit", context)
    assert clean_missing.session_pass_safe is True


def test_matching_entry_verdict_filter_skips_newer_other_verdicts(tmp_path):
    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True}, "world": 2}
    led.record("k", "sig", "commit", "FAIL",
               detail={"quarantine_levers": ["slab_weights"]}, context=context)
    led.record("k", "sig", "commit", "INCONCLUSIVE", context=context)
    assert led.matching_entry("k", "sig", context)["verdict"] == "INCONCLUSIVE"
    fail_row = led.matching_entry("k", "sig", context, verdict="FAIL")
    assert fail_row["verdict"] == "FAIL"
    assert fail_row["quarantine_levers"] == ["slab_weights"]
    assert led.matching_entry("k", "sig", context, verdict="PASS") is None


def test_matching_entry_skips_every_foreign_identity_before_valid_older_row(tmp_path):
    """Diagnostic/quarantine recovery is as identity-bound as trust lookup."""
    led = GateLedger(str(tmp_path))
    context = {"worker_args": {"slab_weights": True}, "world": 2}
    other_context = {"worker_args": {"slab_weights": False}, "world": 2}
    led.record("k", "sig", "commit", "FAIL", detail={"marker": "valid"}, context=context)
    led.record("other-key", "sig", "commit", "FAIL",
               detail={"marker": "wrong-key"}, context=context)
    led.record("k", "other-sig", "commit", "FAIL",
               detail={"marker": "wrong-artifact"}, context=context)
    led.record("k", "sig", "commit", "FAIL",
               detail={"marker": "wrong-context"}, context=other_context)

    rows = led.entries()
    context_value = led._context_value(context)
    for marker, protocol, version in (
        ("old-protocol", ledger_mod.GATE_PROTOCOL_VERSION - 1, ledger_mod.__version__),
        ("other-version", ledger_mod.GATE_PROTOCOL_VERSION, "0.0.0-test"),
    ):
        rows.append({
            "key": "k",
            "artifacts": "sig",
            "comfy": "commit",
            "verdict": "FAIL",
            "marker": marker,
            "gate_protocol": protocol,
            "dgx_monarch": version,
            "capability_context": context_value,
        })
    with open(led.path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    match = led.matching_entry("k", "sig", context, verdict="FAIL")
    assert match is not None
    assert match["marker"] == "valid"


def test_record_concurrent_writers_preserve_every_row(tmp_path):
    """record() takes an advisory flock, so N racing writers leave N intact rows."""
    import threading

    led = GateLedger(str(tmp_path))
    n = 16
    barrier = threading.Barrier(n)

    def write(i):
        barrier.wait()
        led.record(f"k{i}", "sig", "commit", "PASS")

    threads = [threading.Thread(target=write, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    rows = led.entries()
    assert len(rows) == n
    assert {row["key"] for row in rows} == {f"k{i}" for i in range(n)}


def test_record_write_failure_logs_and_leaves_no_durable_row(
    tmp_path, monkeypatch, caplog,
):
    led = GateLedger(str(tmp_path))

    def fail_makedirs(*_args, **_kwargs):
        raise OSError("read-only ledger directory")

    monkeypatch.setattr(ledger_mod.os, "makedirs", fail_makedirs)
    led.record("k", "sig", "commit", "PASS")

    assert "gate ledger not writable" in caplog.text
    assert not os.path.exists(led.path)
    assert led.lookup("k", "sig", "commit") == "unknown"


def test_retest_guard_atomically_denies_every_context_until_exact_finals(
    tmp_path,
):
    contexts = [
        {"kind": "normal", "worker_args": {"slab_weights": "auto"}},
        {"kind": "fleet", "worker_args": {"slab_weights": "auto"}},
        {"kind": "normal", "worker_args": {"slab_weights": True}},
        {"kind": "fleet", "worker_args": {"slab_weights": True}},
    ]
    led = GateLedger(str(tmp_path))
    for context in contexts:
        led.record("k", "sig", "commit", "PASS", context=context)
        assert led.lookup("k", "sig", "commit", context) == "pass"

    led.begin_retest_required(
        "k", "sig", "commit", contexts, {"phase": "ceremony-preflight"})

    # A fresh instance models restart/crash before any final verdict append.
    restarted = GateLedger(str(tmp_path))
    assert [
        restarted.lookup("k", "sig", "commit", context)
        for context in contexts
    ] == ["inconclusive"] * 4
    guard = restarted.entries()[-1]
    assert guard["verdict"] == "RETESTING"
    assert len(guard["blocked_contexts"]) == 4

    # Final rows supersede the one guard independently. A partial final-write
    # sequence can grant only the context whose PASS was appended.
    restarted.record("k", "sig", "commit", "PASS", context=contexts[0])
    healed = GateLedger(str(tmp_path))
    assert healed.lookup("k", "sig", "commit", contexts[0]) == "pass"
    assert [
        healed.lookup("k", "sig", "commit", context)
        for context in contexts[1:]
    ] == ["inconclusive"] * 3


def test_a_terminal_inconclusive_supersedes_one_blocked_context_only(tmp_path):
    """The abort arm closes its transaction one context at a time (2026-09-02).

    Closing the context the ceremony proved must not close the siblings the
    same row still guards; those stay open for `dgxm gate --repair`.
    """
    contexts = [
        {"kind": "normal", "worker_args": {"slab_weights": "auto"}},
        {"kind": "fleet", "worker_args": {"slab_weights": "auto"}},
        {"kind": "normal", "worker_args": {"slab_weights": True}},
    ]
    led = GateLedger(str(tmp_path))
    led.begin_retest_required("k", "sig", "commit", contexts)

    led.record("k", "sig", "commit", "INCONCLUSIVE",
               {"inconclusive_reasons": ["the FSDP clean-reload proof met a "
                                         "typed class P refusal"]},
               contexts[0])

    closed = GateLedger(str(tmp_path))
    rows = closed.entries()
    assert [row["verdict"] for row in rows] == ["RETESTING", "INCONCLUSIVE"]
    # State-neutral: a closed context reads exactly what the open one read.
    assert [
        closed.lookup("k", "sig", "commit", context) for context in contexts
    ] == ["inconclusive"] * 3
    assert closed.matching_entry(
        "k", "sig", contexts[0], "INCONCLUSIVE") is not None
    assert closed.matching_entry(
        "k", "sig", contexts[1], "INCONCLUSIVE") is None


def test_a_repaired_row_keeps_its_own_time_beside_the_row_it_closed(tmp_path):
    """`_entry` owns "time"; a repair detail may only name the original."""
    context = {"kind": "normal", "worker_args": {"slab_weights": "auto"}}
    led = GateLedger(str(tmp_path))
    led.begin_retest_required("k", "sig", "commit", [context])
    opened = led.entries()[-1]

    led.record("k", "sig", "commit", "INCONCLUSIVE", {
        "origin": "gate_repair",
        "inconclusive_reasons": ["repaired: ceremony ended without a terminal row"],
        "repaired_from_time": opened["time"],
        "repaired_from_line": 1,
    }, context)

    repaired = GateLedger(str(tmp_path)).entries()[-1]
    assert repaired["verdict"] == "INCONCLUSIVE"
    assert repaired["repaired_from_time"] == opened["time"]
    assert repaired["repaired_from_line"] == 1
    assert isinstance(repaired["time"], str) and repaired["time"]
    assert repaired["capability_context"] == led._context_value(context)


@pytest.mark.parametrize("missing_field", ["gate_protocol", "dgx_monarch"])
def test_retest_guard_missing_runtime_identity_cannot_resurrect_older_pass(
    tmp_path, missing_field,
):
    context = {"kind": "normal", "worker_args": {"slab_weights": "auto"}}
    led = GateLedger(str(tmp_path))
    led.record("k", "sig", "commit", "PASS", context=context)
    guard = {
        "key": "k",
        "artifacts": "sig",
        "comfy": "commit",
        "verdict": "RETESTING",
        "gate_protocol": ledger_mod.GATE_PROTOCOL_VERSION,
        "dgx_monarch": ledger_mod.__version__,
        "blocked_contexts": [led._context_value(context)],
    }
    del guard[missing_field]
    with open(led.path, "a") as stream:
        stream.write(json.dumps(guard) + "\n")

    result = led.lookup_with_integrity("k", "sig", "commit", context)
    assert result.state == "error"
    assert result.entry is not None and result.entry["verdict"] == "PASS"
    assert result.session_pass_safe is False
    assert [entry["verdict"] for entry in led.entries()] == ["PASS"]


def test_retest_guard_write_failure_is_not_best_effort(
    tmp_path, monkeypatch,
):
    led = GateLedger(str(tmp_path))

    def fail_makedirs(*_args, **_kwargs):
        raise OSError("read-only ledger directory")

    monkeypatch.setattr(ledger_mod.os, "makedirs", fail_makedirs)
    with pytest.raises(
        ledger_mod.GateLedgerWriteError, match="could not durably begin"
    ):
        led.begin_retest_required(
            "k", "sig", "commit", [{"kind": "normal"}])


def test_retest_guard_requires_lock_support_but_record_remains_best_effort(
    tmp_path, monkeypatch,
):
    led = GateLedger(str(tmp_path))
    monkeypatch.setitem(sys.modules, "fcntl", None)

    with pytest.raises(
        ledger_mod.GateLedgerWriteError, match="could not durably begin"
    ):
        led.begin_retest_required(
            "k", "sig", "commit", [{"kind": "normal"}])
    assert led.entries() == []

    assert led.record("diagnostic", "sig", "commit", "INCONCLUSIVE") is True
    assert [entry["key"] for entry in led.entries()] == ["diagnostic"]


def test_existing_ledger_read_error_is_not_clean_unknown(
    tmp_path, monkeypatch,
):
    led = GateLedger(str(tmp_path))
    assert led.lookup("k", "sig", "commit") == "unknown"
    open(led.path, "wb").close()

    def unreadable(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(ledger_mod, "open", unreadable, raising=False)
    with pytest.raises(ledger_mod.GateLedgerReadError, match="could not read"):
        led.lookup("k", "sig", "commit")


def test_damaged_newer_row_cannot_resurrect_an_older_pass(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("k", "sig", "commit", "PASS")
    with open(led.path, "ab") as stream:
        stream.write(
            b'{"key":"k","artifacts":"sig","comfy":"commit",'
            b'"verdict":"FAIL","damaged":"\xff"}\n')

    assert led.lookup("k", "sig", "commit") == "error"
    assert led.entries()[0]["verdict"] == "PASS"


def test_unscoped_damage_does_not_pollute_an_absent_identity_state(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("other", "other-sig", "commit", "FAIL")
    with open(led.path, "ab") as stream:
        stream.write(b'{"torn":')

    # There is no positive target row to resurrect, so the canonical state is
    # still unknown and first-use orchestration may re-prove it. The same scan
    # marks an older process PASS unsafe, because the torn row's key is unknown.
    assert led.lookup_with_entry("new", "sig", "commit") == ("unknown", None)
    result = led.lookup_with_integrity("new", "sig", "commit")
    assert result.state == "unknown"
    assert result.entry is None
    assert result.session_pass_safe is False


def test_unscoped_damage_does_not_pollute_an_unmatched_identity_state(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("target", "old-sig", "commit", "PASS")
    with open(led.path, "ab") as stream:
        stream.write(b'not-json\n')

    result = led.lookup_with_integrity("target", "new-sig", "commit")
    assert result.state == "stale"
    assert result.entry is None
    assert result.session_pass_safe is False


def test_later_exact_pass_heals_older_damage(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("k", "sig", "commit", "PASS")
    with open(led.path, "ab") as stream:
        stream.write(b'not-json\n')

    assert led.lookup("k", "sig", "commit") == "error"
    led.record("k", "sig", "commit", "PASS")
    assert GateLedger(str(tmp_path)).lookup("k", "sig", "commit") == "pass"


@pytest.mark.parametrize(
    "damaged",
    [
        b'[]\n',
        b'{"key":"k"}\n',
        b'{"key":"k","artifacts":"sig","comfy":7,"verdict":"FAIL"}\n',
    ],
)
def test_non_dict_and_invalid_schema_rows_taint_older_positive_authority(
    tmp_path, damaged,
):
    led = GateLedger(str(tmp_path))
    led.record("k", "sig", "commit", "PASS")
    with open(led.path, "ab") as stream:
        stream.write(damaged)

    assert led.lookup("k", "sig", "commit") == "error"


def test_capability_token_and_ledger_share_strict_canonical_encoding():
    context = {
        "world": 2,
        "worker_args": {
            "slab_weights": True,
            "private_metadata": ["stable", {"rank": 1}],
        },
    }
    reordered = {
        "worker_args": {
            "private_metadata": ["stable", {"rank": 1}],
            "slab_weights": True,
        },
        "world": 2,
    }

    expected = GateLedger._context_value(context)
    token = ledger_mod.gate_verdict_token("key", "artifact", "commit", context)
    reordered_token = ledger_mod.gate_verdict_token(
        "key", "artifact", "commit", reordered)

    assert expected == GateLedger._context_value(reordered)
    assert token == reordered_token
    assert token[3] == expected


@pytest.mark.parametrize("opaque", [object(), {"set-values"}])
def test_capability_context_rejects_opaque_metadata_before_grant_or_record(
    tmp_path, opaque,
):
    context = {"worker_args": {"_private": opaque}}
    copied_context = copy.deepcopy(context)

    for candidate in (context, copied_context):
        with pytest.raises(TypeError):
            ledger_mod.gate_verdict_token(
                "key", "artifact", "commit", candidate)
        with pytest.raises(TypeError):
            GateLedger._context_value(candidate)

    led = GateLedger(str(tmp_path))
    with pytest.raises(TypeError):
        led.record("key", "artifact", "commit", "PASS", context=context)
    assert not os.path.exists(led.path)


@pytest.mark.parametrize(
    "context",
    [
        {"worker_args": {"value": float("nan")}},
        {"worker_args": {"value": float("inf")}},
        {"worker_args": {"value": float("-inf")}},
    ],
)
def test_capability_context_rejects_non_finite_numbers(context):
    with pytest.raises(ValueError, match="Out of range float values"):
        ledger_mod.gate_verdict_token("key", "artifact", "commit", context)
    with pytest.raises(ValueError, match="Out of range float values"):
        GateLedger._context_value(context)


@pytest.mark.parametrize(
    "context",
    [
        {"worker_args": {1: "integer key"}},
        {"worker_args": {"tuple": ("not", "a", "JSON", "array")}},
    ],
)
def test_capability_context_rejects_non_json_container_shapes(context):
    with pytest.raises(TypeError, match="only JSON-native values"):
        ledger_mod.gate_verdict_token("key", "artifact", "commit", context)
    with pytest.raises(TypeError, match="only JSON-native values"):
        GateLedger._context_value(context)


def test_v7_pass_without_fsdp_clean_reload_reads_stale_under_v8(tmp_path):
    led = GateLedger(str(tmp_path))
    context = {
        "worker_args": {"slab_weights": False, "lora_low_rss": False},
        "world": 2,
        "resolved_topology": {
            "ulysses": 2,
            "ring": 1,
            "cfg": 1,
            "dp": 1,
            "fsdp": True,
        },
        "proof_scope": "fsdp_clean_reload_v1",
    }
    row = {
        "key": "k",
        "artifacts": "artifact",
        "comfy": "commit",
        "verdict": "PASS",
        "gate_protocol": 7,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": led._context_value(context),
    }
    with open(led.path, "w") as f:
        f.write(json.dumps(row) + "\n")

    assert led.lookup("k", "artifact", "commit", context) == "stale"
    led.record("fresh", "artifact", "commit", "PASS", context=context)
    assert led.lookup("fresh", "artifact", "commit", context) == "pass"


def test_v8_pass_without_the_byte_verify_rule_reads_stale_under_v9(tmp_path):
    """v9 changed what a trusted PASS means: a pre-gate slab load is permitted
    only under a byte-verify certificate, and every bypass writes a
    permanent waiver row. A v8 PASS predates both, so it re-proves once per
    (combination, artifact digest, comfy commit, capability context) on first
    use. No migration and no backfill: the JSONL keeps every older row."""
    led = GateLedger(str(tmp_path))
    context = {
        "worker_args": {"slab_weights": True, "lora_low_rss": True},
        "world": 1,
    }
    row = {
        "key": "k",
        "artifacts": "artifact",
        "comfy": "commit",
        "verdict": "PASS",
        "gate_protocol": 8,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": led._context_value(context),
    }
    with open(led.path, "w") as f:
        f.write(json.dumps(row) + "\n")

    assert led.lookup("k", "artifact", "commit", context) == "stale"
    # Context-free reporting still reads the historical row.
    assert led.lookup("k", "artifact", "commit") == "pass"
    led.record("fresh", "artifact", "commit", "PASS", context=context)
    assert led.lookup("fresh", "artifact", "commit", context) == "pass"


def test_v11_pass_reads_stale_under_v12(tmp_path):
    """v11 slab PASS rows lacked exact current rank-cohort evidence.

    A response count alone cannot prove that each rank participated in the
    current setup generation, so every v11 PASS re-proves once.
    """
    led = GateLedger(str(tmp_path))
    context = {
        "worker_args": {"slab_weights": True, "lora_low_rss": False},
        "world": 2,
    }
    row = {
        "key": "k",
        "artifacts": "artifact",
        "comfy": "commit",
        "verdict": "PASS",
        "gate_protocol": 11,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": led._context_value(context),
    }
    with open(led.path, "w") as f:
        f.write(json.dumps(row) + "\n")

    assert led.lookup("k", "artifact", "commit", context) == "stale"
    led.record("fresh", "artifact", "commit", "PASS", context=context)
    assert led.lookup("fresh", "artifact", "commit", context) == "pass"


def test_v12_pass_reads_stale_under_v13(tmp_path):
    """v12 PASS rows predate the fourth class-K waiver kind (shard-quant scale).

    A semantic change to the frozen waiver vocabulary advances the protocol, so
    every v12 PASS re-proves once.
    """
    led = GateLedger(str(tmp_path))
    context = {
        "worker_args": {"slab_weights": True, "lora_low_rss": False},
        "world": 2,
    }
    row = {
        "key": "k",
        "artifacts": "artifact",
        "comfy": "commit",
        "verdict": "PASS",
        "gate_protocol": 12,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": led._context_value(context),
    }
    with open(led.path, "w") as f:
        f.write(json.dumps(row) + "\n")

    assert led.lookup("k", "artifact", "commit", context) == "stale"
    led.record("fresh", "artifact", "commit", "PASS", context=context)
    assert led.lookup("fresh", "artifact", "commit", context) == "pass"


def test_v9_cross_mode_pass_without_repeat_identity_reads_stale_under_v10(tmp_path):
    """v9 could record PASS when slab matched stock but the repeated slab
    render diverged. v10 requires both comparisons, so every v9 PASS re-proves.
    """
    led = GateLedger(str(tmp_path))
    context = {
        "worker_args": {"slab_weights": True, "lora_low_rss": False},
        "world": 1,
    }
    row = {
        "key": "k",
        "artifacts": "artifact",
        "comfy": "commit",
        "verdict": "PASS",
        "gate_protocol": 9,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": led._context_value(context),
    }
    with open(led.path, "w") as f:
        f.write(json.dumps(row) + "\n")

    assert led.lookup("k", "artifact", "commit", context) == "stale"
    assert led.lookup("k", "artifact", "commit") == "pass"
    led.record("fresh", "artifact", "commit", "PASS", context=context)
    assert led.lookup("fresh", "artifact", "commit", context) == "pass"


def test_contextual_authority_and_token_bind_exact_dgx_source(tmp_path, monkeypatch):
    source = "a" * 64
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        lambda: source,
    )
    led = GateLedger(str(tmp_path))
    pass_context = {"kind": "normal", "world": 2}
    inconclusive_context = {"kind": "fleet", "world": 1}
    retest_context = {"kind": "normal", "world": 1}

    led.record("k", "artifact", "commit", "PASS", context=pass_context)
    led.record(
        "k", "artifact", "commit", "INCONCLUSIVE",
        context=inconclusive_context,
    )
    led.begin_retest_required("k", "artifact", "commit", [retest_context])

    rows = led.entries()
    assert [row["dgx_source"] for row in rows] == [source] * 3
    assert led.lookup("k", "artifact", "commit", pass_context) == "pass"
    assert led.lookup("k", "artifact", "commit", inconclusive_context) == "inconclusive"
    assert led.lookup("k", "artifact", "commit", retest_context) == "inconclusive"
    token = ledger_mod.gate_verdict_token(
        "k", "artifact", "commit", pass_context)
    assert token[-3:] == (
        source, str(ledger_mod.GATE_PROTOCOL_VERSION), str(ledger_mod.__version__))


def test_contextual_legacy_authority_without_dgx_source_reads_stale(tmp_path):
    context = {"kind": "normal", "world": 2}
    context_value = GateLedger._context_value(context)
    for verdict, unscoped_state in (
        ("PASS", "pass"),
        ("INCONCLUSIVE", "inconclusive"),
        ("RETESTING", "inconclusive"),
    ):
        led = GateLedger(str(tmp_path / verdict.lower()))
        row = {
            "key": "k",
            "artifacts": "artifact",
            "comfy": "commit",
            "verdict": verdict,
            "gate_protocol": ledger_mod.GATE_PROTOCOL_VERSION,
            "dgx_monarch": ledger_mod.__version__,
        }
        if verdict == "RETESTING":
            row["blocked_contexts"] = [context_value]
        else:
            row["capability_context"] = context_value
        led._append(row, strict_lock=False)

        result = led.lookup_with_integrity("k", "artifact", "commit", context)
        assert result.state == "stale"
        assert result.entry == row
        assert result.session_pass_safe is False
        assert led.lookup("k", "artifact", "commit") == unscoped_state


def test_same_version_source_change_invalidates_after_process_cache_reset(
    tmp_path, monkeypatch,
):
    source = ["a" * 64]
    cached_source = ledger_mod.runtime_provenance.cached_dgx_source_manifest_sha256
    cached_source.cache_clear()
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "dgx_source_manifest_sha256",
        lambda: source[0],
    )
    try:
        led = GateLedger(str(tmp_path))
        context = {"kind": "normal", "world": 2}
        led.record("k", "artifact", "commit", "PASS", context=context)
        before = ledger_mod.gate_verdict_token(
            "k", "artifact", "commit", context)
        row = led.entries()[0]

        source[0] = "b" * 64
        assert led.lookup("k", "artifact", "commit", context) == "pass"
        assert ledger_mod.gate_verdict_token(
            "k", "artifact", "commit", context) == before

        # A process restart gets a fresh cache; clear() models that boundary.
        cached_source.cache_clear()
        result = led.lookup_with_integrity("k", "artifact", "commit", context)
        assert (result.state, result.entry, result.session_pass_safe) == (
            "stale", row, False)
        assert led.matching_entry(
            "k", "artifact", context, verdict="PASS") is None
        after = ledger_mod.gate_verdict_token(
            "k", "artifact", "commit", context)
        assert after != before and after[-3] == source[0]
        assert led.lookup("k", "artifact", "commit") == "pass"

        led.record("failed", "artifact", "commit", "FAIL", context=context)
        assert "dgx_source" not in led.entries()[-1]
        source[0] = "c" * 64
        cached_source.cache_clear()
        assert led.lookup("failed", "artifact", "other-commit", context) == "fail"
    finally:
        cached_source.cache_clear()


@pytest.mark.parametrize("newer_verdict", ["PASS", "INCONCLUSIVE", "RETESTING"])
def test_foreign_source_row_cannot_displace_sticky_fail_or_revive_older_pass(
    tmp_path, monkeypatch, newer_verdict,
):
    source = ["a" * 64]
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        lambda: source[0],
    )
    led = GateLedger(str(tmp_path))
    context = {"kind": "normal", "world": 2}
    led.record("failed", "artifact", "commit", "FAIL", context=context)
    fail_row = led.entries()[-1]
    led.record("positive", "artifact", "commit", "PASS", context=context)
    led.record("healed", "artifact", "commit", "FAIL", context=context)
    led.record("healed", "artifact", "commit", "PASS", context=context)

    source[0] = "b" * 64
    if newer_verdict == "RETESTING":
        led.begin_retest_required(
            "failed", "artifact", "commit", [context])
        led.begin_retest_required(
            "positive", "artifact", "commit", [context])
        led.begin_retest_required(
            "healed", "artifact", "commit", [context])
    else:
        led.record(
            "failed", "artifact", "commit", newer_verdict, context=context)
        led.record(
            "positive", "artifact", "commit", newer_verdict, context=context)
        led.record(
            "healed", "artifact", "commit", newer_verdict, context=context)

    source[0] = "a" * 64
    result = led.lookup_with_integrity(
        "failed", "artifact", "commit", context)
    assert (result.state, result.entry) == ("fail", fail_row)
    positive = led.lookup_with_integrity(
        "positive", "artifact", "commit", context)
    assert positive.state == "stale"
    assert positive.session_pass_safe is False
    assert led.lookup("healed", "artifact", "commit", context) == "stale"


def test_retest_source_manifest_failure_is_typed_and_writes_nothing(
    tmp_path, monkeypatch,
):
    def fail_source_manifest():
        raise RuntimeError("source changed while hashing")

    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        fail_source_manifest,
    )
    led = GateLedger(str(tmp_path))

    with pytest.raises(ledger_mod.GateLedgerWriteError, match="durably begin"):
        led.begin_retest_required(
            "k", "artifact", "commit", [{"kind": "normal"}])

    assert not os.path.exists(led.path)


@pytest.mark.parametrize("audit_verdict", ["WAIVER", "CAPACITY_CERTIFIED"])
def test_source_failure_preserves_unscoped_diagnostics_and_audit_rows(
    tmp_path, monkeypatch, audit_verdict,
):
    led = GateLedger(str(tmp_path))
    context = {"kind": "normal"}
    led.record("trusted", "artifact", "commit", "PASS", context=context)
    pass_row = led.entries()[-1]
    monkeypatch.setattr(
        ledger_mod.runtime_provenance,
        "cached_dgx_source_manifest_sha256",
        lambda: (_ for _ in ()).throw(RuntimeError("manifest unavailable")),
    )

    assert led.lookup("trusted", "artifact", "commit") == "pass"
    assert led.matching_entry("trusted", "artifact") == pass_row
    assert led.record(
        "audit", "artifact", "commit", audit_verdict,
        context={"record": audit_verdict.lower()},
    )
    audit_row = led.entries()[-1]
    assert "dgx_source" not in audit_row
    assert led.lookup(
        "audit", "artifact", "commit",
        {"record": audit_verdict.lower()},
    ) == "unknown"


def test_public_entries_returns_the_intact_diagnostic_snapshot(tmp_path):
    led = GateLedger(str(tmp_path))
    led.record("k", "sig", "commit", "FAIL")

    rows = led.entries()
    positioned, damage = led.entries_with_integrity()

    assert rows == led.entries()
    assert positioned == [(1, rows[0])] and damage == 0
    rows[0]["verdict"] = "PASS"
    assert led.lookup("k", "sig", "commit") == "fail"


def test_the_first_append_to_a_fresh_directory_syncs_the_directory(tmp_path, monkeypatch):
    """The file's bytes are durable; without this the name that reaches them is not.

    ``consent_routes`` writes the permanent audit row here before the consent
    memo, and the memo syncs its own parent, so a power cut right after a first
    grant on a fresh output directory could take the whole ledger while the
    memo survived (CHANGELOG, 2026-09-09). Later appends need no directory
    sync: the name is already on disk.
    """
    directory = tmp_path / "fresh"
    gate = GateLedger(str(directory))
    synced: list[bool] = []
    real_fsync = os.fsync

    def record_kind(fd):
        synced.append(os.fstat(fd).st_mode & 0o170000 == 0o040000)
        return real_fsync(fd)

    monkeypatch.setattr(ledger_mod.os, "fsync", record_kind)
    assert gate.record("combo", "sig", "commit", "PASS")
    assert synced == [False, True]

    synced.clear()
    assert gate.record("combo", "sig", "commit", "FAIL")
    assert synced == [False]


def test_a_directory_that_will_not_sync_leaves_the_row_written(tmp_path, monkeypatch):
    """Durability of the name is unknown; the row itself is on disk either way."""
    gate = GateLedger(str(tmp_path / "fresh"))

    def refuse(path, flags):
        raise OSError("no directory handle here")

    monkeypatch.setattr(ledger_mod.os, "open", refuse)
    assert gate.record("combo", "sig", "commit", "PASS")
    assert gate.lookup("combo", "sig", "commit") == "pass"
