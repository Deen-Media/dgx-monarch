"""Check that auto_gate=off skips proof without bypassing measured quarantines.

For single-model dispatches, ``authorize_normal_render`` stamps
``operator_off`` and preserves the graph's residency policy. The worker
accepts that stamp without a grant; the operator accepts the accuracy risk.
Dual-model requests still force stock before this branch
(docs/TROUBLESHOOTING.md #67, item 4).

An existing FAIL still quarantines both residency levers. The tooltip and
runbook must distinguish skipping a missing proof from overriding a failure.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_residency

REPO = Path(__file__).resolve().parents[1]

_RISKY_POLICY = {"slab_weights": "auto", "lora_low_rss": True}


def _stub_folder_paths(monkeypatch, tmp_path):
    """Resolve every artifact name to one readable file, as the driver would."""
    artifact = tmp_path / "artifact.safetensors"
    artifact.write_bytes(b"x" * 4096)
    module = types.ModuleType("folder_paths")
    module.get_full_path = lambda _kind, _name: str(artifact)
    monkeypatch.setitem(sys.modules, "folder_paths", module)


def _lora_model(monkeypatch, tmp_path, *, auto_gate, state, entry=None):
    """One LoRA-bearing KSampler graph with an injected ledger verdict."""
    import dgx_monarch.gate_ledger as ledger_mod
    import dgx_monarch.nodes.gate as gate_mod
    from dgx_monarch.actor import model_store as model_store_mod
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec

    _stub_folder_paths(monkeypatch, tmp_path)

    def _identity(unet_name, loras):
        signatures = ["model-sig", *[f"lora-sig-{index}"
                                     for index, _ in enumerate(loras or [])]]
        artifacts = [{"kind": "diffusion_models", "name": unet_name,
                      "signature": "model-sig"}]
        artifacts += [{"kind": "loras", "name": item["name"],
                       "signature": f"lora-sig-{index}"}
                      for index, item in enumerate(loras or [])]
        return {
            "digest": ledger_mod.artifact_set_signature(signatures).current,
            "comfy": "known",
            "artifacts": artifacts,
        }

    class _Ledger:
        def __init__(self, _directory):
            pass

        def lookup_with_integrity(self, _key, _artifacts, _commit, _context):
            return ledger_mod.GateLedgerLookup(state, entry, True)

        def lookup_with_entry(self, _key, _artifacts, _commit, _context=None):
            return state, entry

    monkeypatch.setattr(ledger_mod, "GateLedger", _Ledger)
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(model_store_mod, "request_artifact_identity", _identity)

    handle = SimpleNamespace(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""),
        config_fingerprint="local", world=1, n_hosts=1, gpus_per_host=1)
    handle.effective_worker_args = lambda requested: dict(requested)
    model = ModelSpec(
        mesh=MeshSpec(handle=handle, topology_preset="single",
                      attention="TORCH_FLASH", sync_ulysses=False,
                      worker_args=dict(_RISKY_POLICY), auto_gate=auto_gate),
        unet_name="model.safetensors",
        loras=({"name": "style.safetensors", "strength": 1.0},))
    return model, handle


def _authorize(model, handle):
    from dgx_monarch.nodes import gate_identity
    from dgx_monarch.topology import Topology

    return gate_identity.authorize_normal_render(
        model, handle, gate_active=False, session_verdict=lambda _token: None,
        resolved_topology=Topology(world=1), resolved_attention="TORCH_FLASH")


def test_an_unproved_lora_render_keeps_every_lever_with_gating_off(
    monkeypatch, tmp_path,
):
    """No refusal, and no forcing to the baked path."""
    model, handle = _lora_model(
        monkeypatch, tmp_path, auto_gate="off", state="unknown")
    authorization = _authorize(model, handle)

    assert authorization.residency_mode == "operator_off"
    assert authorization.residency_grant is None
    # The graph's own policy, not the forced-stock pair.
    assert authorization.worker_args == _RISKY_POLICY
    assert authorization.model_request["loras"] == [
        {"name": "style.safetensors", "strength": 1.0}]


def test_the_same_render_is_forced_stock_when_gating_is_on(monkeypatch, tmp_path):
    """The control. One field differs, and it is the field under test: without
    this the passthrough above could pass on a graph that was never risky."""
    model, handle = _lora_model(
        monkeypatch, tmp_path, auto_gate="first_use", state="unknown")
    authorization = _authorize(model, handle)

    assert authorization.residency_mode == "stock"
    assert authorization.worker_args["lora_low_rss"] is False
    assert authorization.worker_args["slab_weights"] is False


def test_the_worker_accepts_the_operator_off_stamp_without_a_grant():
    """A risky policy plus a LoRA stack plus no grant is the shape the worker
    refuses under the other stamps, and must not refuse under this one."""
    request = {"model": {"loras": [{"name": "style.safetensors"}]},
               "_dgxm_normal_residency_mode": "operator_off"}
    assert mesh_residency.assert_normal_render_residency_mode(
        request, dict(_RISKY_POLICY)) == "operator_off"

    request["_dgxm_normal_residency_mode"] = "stock"
    with pytest.raises(RuntimeError):
        mesh_residency.assert_normal_render_residency_mode(
            request, dict(_RISKY_POLICY))


def test_a_persisted_gate_fail_still_quarantines_with_gating_off(
    monkeypatch, tmp_path,
):
    """Off skips a missing proof, never a measured one."""
    from dgx_monarch.nodes.render_quarantine import _enforce_persisted_quarantine
    from dgx_monarch.topology import Topology

    model, handle = _lora_model(
        monkeypatch, tmp_path, auto_gate="off", state="fail",
        entry={"verdict": "FAIL",
               "quarantine_levers": {"lora_low_rss": False, "slab_weights": False}})
    _enforce_persisted_quarantine(
        model, handle, resolved_topology=Topology(world=1),
        resolved_attention="TORCH_FLASH")

    assert model.mesh.worker_args["lora_low_rss"] is False
    assert model.mesh.worker_args["slab_weights"] is False


def test_the_tooltip_states_what_off_costs_and_what_it_does_not():
    from dgx_monarch.nodes.init import DGXMonarchInit

    values, meta = DGXMonarchInit.INPUT_TYPES()["optional"]["auto_gate"]
    assert values == ["first_use", "off"]
    assert meta["default"] == "first_use"
    tooltip = meta["tooltip"]
    assert "docs/TROUBLESHOOTING.md #67" in tooltip
    for promise in ("lora_low_rss", "slab_weights", "Fleet", "FAILED"):
        assert promise in tooltip


def test_troubleshooting_67_answers_the_question():
    text = (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    head = "\n## 67. "
    assert head in text
    entry = text.split(head, 1)[1].split("\n## ", 1)[0]
    # The three things an operator has to be able to find in it.
    assert "operator_off" in entry
    assert "Fleet" in entry
    assert "comfy_managed" in entry
    # The capacity entry, docs/TROUBLESHOOTING.md #5, must link here.
    assert "#67" in text.split("\n## 6. ", 1)[0]
