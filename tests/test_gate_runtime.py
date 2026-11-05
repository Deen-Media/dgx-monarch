"""A Gate PASS names the torch and CUDA build it ran under, and a torch change proves again."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
from dgx_monarch import first_render, gate_runtime, mesh_residency
from dgx_monarch.gate_ledger import GateLedger, gate_verdict_token
from gate_orchestration_helpers import _clear_process_gate_verdicts  # noqa: F401  # autouse fixture import.

SOURCE = "a" * 64
PROVEN = {"torch": "0.0.1+proven", "cuda": "1.1"}
MOVED = {"torch": "0.0.2+moved", "cuda": "1.1"}


@pytest.fixture(autouse=True)
def fixed_source(monkeypatch):
    monkeypatch.setattr(ledger_mod.runtime_provenance, "cached_dgx_source_manifest_sha256",
                        lambda: SOURCE)


def _handle():
    return SimpleNamespace(
        config=SimpleNamespace(worker_args={"slab_weights": True},
                               hosts=(object(),), source="/configs/cluster.toml"),
        config_fingerprint="config-a", world=2, n_hosts=2, gpus_per_host=1)


def _model(handle):
    return SimpleNamespace(mesh=SimpleNamespace(
        handle=handle, worker_args={"lora_low_rss": True},
        topology_preset="uly2", attention="TORCH_FLASH", sync_ulysses=True))


def _context(monkeypatch, runtime, **worker_args):
    monkeypatch.setattr(gate_runtime, "runtime_build_identity", lambda: dict(runtime))
    handle = _handle()
    model = _model(handle)
    model.mesh.worker_args.update(worker_args)
    return gate_mod.gate_capability_context(model, handle)


def _token(context):
    return gate_verdict_token("k", "artifact", "commit", context)


def test_runtime_identity_reads_this_interpreters_torch():
    assert gate_runtime.runtime_build_identity() == {
        "torch": str(torch.__version__), "cuda": str(torch.version.cuda or "none")}


def test_every_capability_context_carries_the_same_runtime(monkeypatch):
    monkeypatch.setattr(gate_runtime, "runtime_build_identity", lambda: dict(PROVEN))
    handle = _handle()
    model = _model(handle)
    assert mesh_residency.physical_capability_context(handle)["runtime"] == PROVEN
    assert gate_mod.gate_capability_context(model, handle)["runtime"] == PROVEN
    assert gate_mod.fleet_residency_capability_context(model, handle)["runtime"] == PROVEN


def test_a_torch_change_changes_the_token_and_the_pass_reads_stale(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    proven = _context(monkeypatch, PROVEN)
    led.record("k", "artifact", "commit", "PASS", context=proven)
    assert led.lookup("k", "artifact", "commit", proven) == "pass"

    moved = _context(monkeypatch, MOVED)
    assert {name: value for name, value in moved.items() if name != "runtime"} == {
        name: value for name, value in proven.items() if name != "runtime"}
    assert _token(moved) != _token(proven)
    result = led.lookup_with_integrity("k", "artifact", "commit", moved)
    assert (result.state, result.entry) == ("stale", None)


def test_a_rollback_reuses_the_pass_proven_under_the_earlier_torch(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    proven, moved = _context(monkeypatch, PROVEN), _context(monkeypatch, MOVED)
    led.record("k", "artifact", "commit", "PASS", context=proven)
    led.record("k", "artifact", "commit", "PASS", context=moved)
    assert led.lookup("k", "artifact", "commit", moved) == "pass"
    assert led.lookup("k", "artifact", "commit", proven) == "pass"

def test_a_row_without_the_runtime_field_is_read_and_never_vouches(tmp_path, monkeypatch):
    """A 1.4.0 row: the same protocol, version and source, and a context with no runtime."""
    current = _context(monkeypatch, PROVEN)
    old = {name: value for name, value in current.items() if name != "runtime"}
    (tmp_path / ledger_mod.LEDGER_NAME).write_text(json.dumps({
        "key": "k", "artifacts": "artifact", "comfy": "commit", "verdict": "PASS",
        "time": "2026-10-28 12:00:00", "gate_protocol": ledger_mod.GATE_PROTOCOL_VERSION,
        "dgx_monarch": ledger_mod.__version__,
        "capability_context": ledger_mod._canonical_capability_context(old),
        "dgx_source": SOURCE}) + "\n")
    led = GateLedger(str(tmp_path))
    assert len(led.entries()) == 1
    assert led.lookup("k", "artifact", "commit", old) == "pass"
    result = led.lookup_with_integrity("k", "artifact", "commit", current)
    assert (result.state, result.entry) == ("stale", None)
    assert gate_runtime.torch_change(str(tmp_path), _token(current)) == (
        "proven under no recorded torch build, not under this driver's torch 0.0.1+proven "
        "with CUDA 1.1")


def test_the_first_use_notice_names_a_torch_only_change(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    led.record("k", "artifact", "commit", "PASS", context=_context(monkeypatch, PROVEN))
    token = _token(_context(monkeypatch, MOVED))
    note = ("proven under torch 0.0.1+proven with CUDA 1.1, not under this driver's "
            "torch 0.0.2+moved with CUDA 1.1")
    assert gate_runtime.torch_change(str(tmp_path), token) == note
    assert gate_runtime.torch_change(lambda: str(tmp_path), token) == note


def test_the_notice_stays_general_when_more_than_torch_moved(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    led.record("k", "artifact", "commit", "PASS", context=_context(monkeypatch, PROVEN))
    led.record("k", "artifact", "commit", "FAIL", context=_context(monkeypatch, PROVEN, compile_dit=True))
    moved = _context(monkeypatch, MOVED, compile_dit=True)
    assert gate_runtime.torch_change(str(tmp_path), _token(moved)) is None
    same = _context(monkeypatch, MOVED)
    assert gate_runtime.torch_change(str(tmp_path), ("k", "artifact", "other", *_token(same)[3:])) is None
    monkeypatch.setattr(ledger_mod.runtime_provenance, "cached_dgx_source_manifest_sha256",
                        lambda: "b" * 64)
    assert gate_runtime.torch_change(str(tmp_path), _token(same)) is None



def test_a_pass_the_earlier_build_later_failed_is_never_named(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    proven = _context(monkeypatch, PROVEN)
    led.record("k", "artifact", "commit", "PASS", context=proven)
    led.record("k", "artifact", "commit", "FAIL", context=proven)
    assert led.lookup("k", "artifact", "commit", proven) == "fail"
    assert gate_runtime.torch_change(str(tmp_path), _token(_context(monkeypatch, MOVED))) is None


def test_a_pass_the_earlier_build_is_retesting_is_never_named(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    proven = _context(monkeypatch, PROVEN)
    led.record("k", "artifact", "commit", "PASS", context=proven)
    led.begin_retest_required("k", "artifact", "commit", [proven])
    assert led.lookup("k", "artifact", "commit", proven) == "inconclusive"
    assert gate_runtime.torch_change(str(tmp_path), _token(_context(monkeypatch, MOVED))) is None


def test_a_revoked_build_falls_back_to_a_build_whose_pass_stands(tmp_path, monkeypatch):
    led = GateLedger(str(tmp_path))
    third = {"torch": "0.0.3+third", "cuda": "1.1"}
    led.record("k", "artifact", "commit", "PASS", context=_context(monkeypatch, third))
    led.record("k", "artifact", "commit", "PASS", context=_context(monkeypatch, PROVEN))
    led.record("k", "artifact", "commit", "FAIL", context=_context(monkeypatch, PROVEN))
    assert gate_runtime.torch_change(str(tmp_path), _token(_context(monkeypatch, MOVED))) == (
        "proven under torch 0.0.3+third with CUDA 1.1, not under this driver's torch "
        "0.0.2+moved with CUDA 1.1")

def test_the_notice_never_raises(tmp_path, monkeypatch):
    def unreadable():
        raise OSError("no output directory")

    token = _token(_context(monkeypatch, MOVED))
    assert gate_runtime.torch_change(unreadable, token) is None
    assert gate_runtime.torch_change(str(tmp_path), ("short",)) is None
    GateLedger(str(tmp_path)).record("k", "artifact", "unknown", "PASS",
                                     context=_context(monkeypatch, PROVEN))
    unknown = _token(_context(monkeypatch, MOVED))
    assert gate_runtime.torch_change(str(tmp_path), (*unknown[:2], "unknown", *unknown[3:])) is None
    (tmp_path / ledger_mod.LEDGER_NAME).write_bytes(
        b'{"key":"k","artifacts":"artifact","comfy":"commit","verdict":"PASS",'
        b'"capability_context":"{not json"}\n\xff torn\n')
    assert gate_runtime.torch_change(str(tmp_path), token) is None


@pytest.mark.parametrize(("state", "consulted"), [("stale", True), ("unknown", False)])
def test_first_use_passes_the_torch_change_to_the_panel_notice(monkeypatch, state, consulted):
    token = ("stale-token",)
    asked, started = [], []
    monkeypatch.setattr(common, "_auto_gate_context", lambda *_args: (state, token))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: "/ledger")
    monkeypatch.setattr(gate_mod, "run_identity_ceremony",
                        lambda *_args, **_kwargs: {"verdict": "PASS", "_gate_token": token})
    monkeypatch.setattr(gate_runtime, "torch_change",
                        lambda directory, seen: asked.append((directory(), seen)) or "torch moved")
    monkeypatch.setattr(first_render, "gate_started", lambda *args: started.append(args))
    common._AUTO_GATE_RUNNING.clear()
    assert common._maybe_auto_gate(
        SimpleNamespace(), {"kind": "ksampler", "steps": 2}, {}, 1.0, 2) == "PASS"
    assert asked == ([("/ledger", token)] if consulted else [])
    assert started == [(state, "torch moved" if consulted else None)]
