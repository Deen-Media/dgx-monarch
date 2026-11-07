"""A dual-model render runs the first-use identity gate over both of its model slots.

Before 1.5 a dual-model render (Ideogram 4) never ran the gate and always
dispatched on stock residency, because a ceremony proved one model. Now the
proof is the dual render itself, and a PASS binds both checkpoints and both
LoRA stacks under the dual row identity (gate_dual_identity.py), which only a
dual render reads. Three placements keep a dual render off the proof: a
cfg-parallel topology renders stock, an FSDP topology refuses under first_use,
and comfy-managed residency keeps its refusal (nodes/gate_dual.py).
"""
from __future__ import annotations

import copy

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
from dgx_monarch.actor import gate_cycle
from dgx_monarch.gate_dual_identity import (
    dual_artifact_signature,
    dual_combo_key,
    dual_identity_summary,
)
from dgx_monarch.mesh_safety import assert_normal_render_residency_grant
from dgx_monarch.nodes import auto_gate, common, gate_identity, refusal_lead
from dgx_monarch.nodes import gate as gate_mod
from dgx_monarch.nodes.gate_fsdp import FsdpGateProofError
from dgx_monarch.residency_mode import ComfyManagedResidencyError
from dgx_monarch.topology import Topology
from first_use_gate_helpers import (  # noqa: F401  # autouse fixture import.
    GENERATED,
    REPORTS,
    TEMPLATES,
    WORLD,
    _forget_classifications,
    cycle_rows,
    doc,
    fresh_process,
    graph_model,
    mesh_rig,
    no_material_rows,
    node,
    proof_renders,
    queue,
    rows,
    sigmas,
    swapped_rows,
    uncond_model,
)
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
)

IDEOGRAM = TEMPLATES / "dgx-monarch-ideogram4-t2i.json"
IDEOGRAM_LORA = GENERATED / "dgx-monarch-test-ideogram4-lora.json"


def _ledger_keys(tmp_path, verdict: str) -> set[str]:
    return {row["key"] for row in rows(tmp_path, ledger_mod.LEDGER_NAME)
            if row.get("verdict") == verdict}


def _authorize(model, handle, uncond, topology, *, gate_active=False):
    return gate_identity.authorize_normal_render(
        model, handle, uncond_model_request=uncond, gate_active=gate_active,
        session_verdict=common._process_gate_verdict, resolved_topology=topology,
        resolved_attention=model.mesh.attention)


def test_the_ideogram_graph_gates_both_slots_on_first_use_then_renders_and_skips_after(
    monkeypatch, tmp_path,
):
    graph = doc(IDEOGRAM)
    latent = torch.zeros(1, 4, 8, 8)
    handle, _topology, dispatched = mesh_rig(
        monkeypatch, tmp_path, no_material_rows(), [latent.clone(), latent.clone()])
    model = graph_model(graph)
    uncond = uncond_model(graph, model)

    queue(graph, model, uncond)

    # The proof is the dual render itself, both slots bound, cut to two steps.
    proofs = dispatched[0]["proof_renders"]
    assert len(proofs) == 2
    for proof in proofs:
        assert proof["uncond_model"] == uncond.request_dict()
        assert proof["guider"]["kind"] == "dual_model"
        assert torch.equal(proof["sigmas"], sigmas(graph)[:3])
        binding = proof["_dgxm_artifact_binding"]
        assert binding["uncond_model_request"] == uncond.request_dict()
        assert len(binding["artifact_sets"]) == 2
    # One cycle, on the conditional slot: neither slot carries a LoRA stack.
    assert [kwargs.get("slot") for _args, kwargs in handle.cycle_calls] == [None]
    (report,) = rows(tmp_path, REPORTS)
    assert (report["origin"], report["verdict"]) == ("auto_first_use", "INCONCLUSIVE")
    # The verdict sits under the dual key, which no single-model lookup reads.
    dual_key = dual_combo_key(model.request_dict(), uncond.request_dict())
    assert _ledger_keys(tmp_path, "INCONCLUSIVE") == {dual_key}
    assert torch.equal(dispatched[0]["request"]["sigmas"], sigmas(graph))

    queue(graph, model, uncond)
    fresh_process()
    queue(graph, model, uncond)
    assert len(proof_renders(handle)) == 2 and len(dispatched) == 3


def test_a_conditional_lora_earns_a_dual_pass_and_a_grant_the_worker_accepts(
    monkeypatch, tmp_path,
):
    graph = doc(IDEOGRAM_LORA)
    (lora_name, strength) = node(graph, "DGXMonarchLoraLoader")["widgets_values"]
    latent = torch.zeros(1, 4, 8, 8)
    handle, topology, _dispatched = mesh_rig(
        monkeypatch, tmp_path, swapped_rows("ideogram4"), [latent.clone(), latent.clone()])
    model = graph_model(graph, loras=[(lora_name, strength)])
    uncond = uncond_model(graph, model)

    queue(graph, model, uncond)

    (report,) = rows(tmp_path, REPORTS)
    assert (report["verdict"], report["proof_kind"]) == ("PASS", "lora_low_rss_swap")
    granted = _authorize(model, handle, uncond.request_dict(), topology)
    assert granted.residency_mode == "required"
    grant = granted.residency_grant
    assert grant["combo_key"] == dual_combo_key(model.request_dict(), uncond.request_dict())
    assert grant["uncond_model_request"] == uncond.request_dict()
    assert len(grant["artifact_sets"]) == 2
    assert grant["artifact_digest"] == dual_artifact_signature(grant["artifact_sets"]).current
    # The Worker recomputes the same identity from the request's two specs.
    request = {"model": model.request_dict(), "uncond_model": uncond.request_dict(),
               "sage_kernel": model.mesh.attention, "sync_ulysses": model.mesh.sync_ulysses,
               "_dgxm_normal_residency_mode": "required", "_dgxm_normal_residency_grant": grant}
    effective = grant["capability_context"]["worker_args"]
    assert assert_normal_render_residency_grant(
        request, grant["artifact_sets"], effective_worker_args=effective)
    # A single-model render of the conditional model gets nothing from it.
    single = gate_identity.authorize_normal_render(
        model, handle, gate_active=False, session_verdict=common._process_gate_verdict,
        resolved_topology=topology, resolved_attention=model.mesh.attention)
    assert single.residency_mode == "stock"


@pytest.mark.parametrize("mutation", [
    lambda request: request.pop("uncond_model"),
    lambda request: request["uncond_model"].update(unet_name="other.safetensors"),
    lambda request: request["_dgxm_normal_residency_grant"]["artifact_sets"].pop(),
    lambda request: request["_dgxm_normal_residency_grant"].update(artifact_digest="forged"),
])
def test_the_worker_refuses_a_dual_grant_that_does_not_bind_both_slots(mutation):
    model = {"unet_name": "cond.safetensors", "options": {}, "loras": []}
    uncond = {"unet_name": "uncond.safetensors", "options": {}, "loras": []}
    sets = [{"comfy": "c", "digest": "d0", "artifacts": [{"signature": "s0"}]},
            {"comfy": "c", "digest": "d1", "artifacts": [{"signature": "s1"}]}]
    context = {"worker_args": {"slab_weights": True}}
    summary = dual_identity_summary(sets)
    key = dual_combo_key(model, uncond)
    grant = {"capability": "normal_render_residency",
             "gate_protocol": ledger_mod.GATE_PROTOCOL_VERSION,
             "dgx_monarch": __import__("dgx_monarch").__version__, "comfy": "c",
             "artifact_digest": summary["digest"],
             "gate_token": list(ledger_mod.gate_verdict_token(key, summary["digest"], "c", context)),
             "combo_key": key, "model_request": model, "uncond_model_request": uncond,
             "artifact_sets": sets, "capability_context": context}
    request = {"model": model, "uncond_model": uncond, "_dgxm_normal_residency_mode": "required",
               "_dgxm_normal_residency_grant": grant}
    assert assert_normal_render_residency_grant(
        copy.deepcopy(request), sets, effective_worker_args={"slab_weights": True})
    tampered = copy.deepcopy(request)
    mutation(tampered)
    with pytest.raises(RuntimeError):
        assert_normal_render_residency_grant(
            tampered, tampered["_dgxm_normal_residency_grant"]["artifact_sets"]
            if tampered.get("uncond_model") else sets[:1],
            effective_worker_args={"slab_weights": True})


@pytest.mark.parametrize(("swapped", "verdict"), [(True, "PASS"), (False, "INCONCLUSIVE")])
def test_an_unconditional_lora_runs_its_own_cycle_on_its_own_slot(
    monkeypatch, tmp_path, swapped, verdict,
):
    graph = doc(IDEOGRAM)
    latent = torch.zeros(1, 4, 8, 8)
    uncond_rows = swapped_rows("ideogram4") if swapped else cycle_rows(
        conclusive=False, family="ideogram4", slab_active=False, certificate=None,
        transitions={"load": "load", "nudge": "load", "back": "load"},
        reason="swaps fell back to reload")
    handle, _topology, _dispatched = mesh_rig(
        monkeypatch, tmp_path, no_material_rows(), [latent.clone(), latent.clone()],
        uncond_rows=uncond_rows)
    model = graph_model(graph)
    uncond = uncond_model(graph, model, loras=[("negative-style.safetensors", 0.5)])

    queue(graph, model, uncond)

    (cond_call, uncond_call) = handle.cycle_calls
    assert cond_call[1].get("slot") is None and cond_call[0][2] == []
    assert uncond_call[1]["slot"] == "uncond"
    assert uncond_call[0][0] == uncond.unet_name
    assert uncond_call[0][2] == [{"name": "negative-style.safetensors", "strength": 0.5}]
    (report,) = rows(tmp_path, REPORTS)
    assert report["verdict"] == verdict
    assert report["loras"] == 1


@pytest.mark.parametrize(("topology", "widgets"), [
    (Topology(cfg=2, world=2), {}),
    (Topology(ulysses=2, world=2), {"comfy_managed": "on"}),
])
def test_a_dual_render_the_gate_cannot_prove_runs_no_ceremony(
    monkeypatch, tmp_path, topology, widgets,
):
    graph = doc(IDEOGRAM)
    handle, _resolved, dispatched = mesh_rig(
        monkeypatch, tmp_path, no_material_rows(), [], topology=topology)
    monkeypatch.setattr(gate_mod, "run_identity_ceremony",
                        lambda *_args, **_kwargs: pytest.fail("no dual proof here"))
    model = graph_model(graph, widgets=widgets)
    uncond = uncond_model(graph, model)

    queue(graph, model, uncond)

    assert len(dispatched) == 1 and rows(tmp_path, REPORTS) == []
    granted = _authorize(model, handle, uncond.request_dict(), topology)
    assert granted.residency_mode == "stock" and granted.residency_grant is None


def test_cfg_parallel_dual_stock_names_its_cause():
    assert refusal_lead.forced_stock_state() is None

    @refusal_lead.lead_with_refusal
    def render():
        from dgx_monarch.nodes.gate_dual import stock_by_rule

        request = {"uncond_model": {"unet_name": "u"}}
        refusal_lead.note_if_slab_dropped(
            {"slab_weights": True}, {"slab_weights": False},
            dual=stock_by_rule(request, Topology(cfg=2, world=2)))
        assert refusal_lead.forced_stock_state() == refusal_lead.DUAL_MODEL
        assert not stock_by_rule(request, Topology(ulysses=2, world=2))

    render()


def test_comfy_managed_dual_keeps_its_refusal_with_the_true_reason():
    from dgx_monarch.mesh_safety import assert_normal_render_residency_mode

    request = {"model": {"unet_name": "c", "loras": []}, "uncond_model": {"unet_name": "u"},
               "_dgxm_normal_residency_mode": "stock"}
    with pytest.raises(ComfyManagedResidencyError, match="no comfy-managed proof covers two"):
        assert_normal_render_residency_mode(request, {"comfy_managed": True})


def test_an_fsdp_dual_render_refuses_with_the_true_reason(monkeypatch, tmp_path):
    graph = doc(IDEOGRAM)
    fsdp = Topology(ulysses=2, fsdp=True, world=2)
    handle, _resolved, _dispatched = mesh_rig(
        monkeypatch, tmp_path, no_material_rows(), [], topology=fsdp)
    model = graph_model(graph)
    uncond = uncond_model(graph, model).request_dict()

    assert common._auto_gate_context(model, "custom", {"samples": torch.zeros(1)}, 7.0,
                                     uncond) is None
    with pytest.raises(FsdpGateProofError, match="reloads one model slot") as raised:
        _authorize(model, handle, uncond, fsdp)
    assert "dual-model" in str(raised.value)


def test_a_dual_ceremony_leg_keeps_the_levers_it_proves(monkeypatch, tmp_path):
    graph = doc(IDEOGRAM)
    handle, topology, _dispatched = mesh_rig(monkeypatch, tmp_path, no_material_rows(), [])
    model = graph_model(graph, widgets={"slab_weights": "on"})
    uncond = uncond_model(graph, model).request_dict()

    leg = _authorize(model, handle, uncond, topology, gate_active=True)
    assert leg.residency_mode == "gate_internal"
    assert leg.worker_args["slab_weights"] is True
    # The dispatch path no longer strips them before authorization decides.
    assert gate_identity.model_for_request(model, {"uncond_model": uncond}) is model


def test_auto_gate_off_keeps_a_dual_render_on_stock(monkeypatch, tmp_path):
    graph = doc(IDEOGRAM)
    handle, topology, _dispatched = mesh_rig(monkeypatch, tmp_path, no_material_rows(), [])
    model = graph_model(graph, widgets={"slab_weights": "on", "auto_gate": "off"})
    uncond = uncond_model(graph, model).request_dict()

    assert _authorize(model, handle, uncond, topology).residency_mode == "stock"
    clone = gate_identity.model_for_request(model, {"uncond_model": uncond})
    assert clone is not model and clone.mesh.worker_args["slab_weights"] is False


def test_a_single_model_proof_request_never_reaches_the_dual_identity(monkeypatch):
    seen = []
    monkeypatch.setattr(common, "_auto_gate_context", lambda *args: seen.append(args))
    auto_gate.maybe_auto_gate(
        type("Model", (), {"unet_name": "m"})(), {"kind": "ksampler", "steps": 4},
        {"samples": torch.zeros(1)}, 1.0, 4)
    assert [len(args) for args in seen] == [4]


class _Store:
    def __init__(self):
        self.current = self.uncond = None


def test_the_worker_cycle_swaps_and_reports_the_slot_it_was_given(monkeypatch):
    worker = type("Worker", (), {})()
    worker._setup_key, worker.rank, worker.world, worker._setup_generation = ("s",), 1, 2, 4
    worker._inject_for_topology = None
    worker.store = _Store()
    worker.store.lora_low_rss = True
    resident = type("Resident", (), {"family": "ideogram4", "slab": None,
                                     "artifact_identity": None})()
    slots = []

    def ensure(_worker, _unet, _options, stack, *, slot, on_base_loaded):
        slots.append(slot)
        setattr(worker.store, "uncond" if slot == "uncond" else "current", resident)
        return None, "load" if len(slots) == 1 else "hot-swap"

    monkeypatch.setattr(gate_cycle.store_fsdp, "ensure", ensure)
    report = gate_cycle.run(worker, "uncond.safetensors", {},
                            [{"name": "l", "strength": 1.0}], None, slot="uncond")
    assert slots == ["uncond"] * 3
    assert report["conclusive"] and report["family"] == "ideogram4"
    assert worker.store.current is None
    with pytest.raises(ValueError, match="cond or uncond"):
        gate_cycle.run(worker, "u", {}, [], None, slot="both")


def test_the_dual_row_identity_is_stable_and_apart_from_the_single_one():
    cond = {"unet_name": "c.safetensors", "options": {"a": 1},
            "loras": [{"name": "x", "strength": 1.0}]}
    uncond = {"unet_name": "u.safetensors", "options": {}, "loras": []}
    key = dual_combo_key(cond, uncond)
    assert key == dual_combo_key({**cond, "loras": [{"name": "x", "strength": 0.2}]}, uncond)
    assert key != dual_combo_key(uncond, cond)
    assert key != ledger_mod.combo_key("c.safetensors", {"a": 1}, ["x"])
    with pytest.raises(RuntimeError, match="different ComfyUI commits"):
        dual_identity_summary([{"comfy": "a", "artifacts": []}, {"comfy": "b", "artifacts": []}])
    assert WORLD == 2


def test_a_runtime_mismatch_refuses_a_dual_grant_instead_of_running_stock(monkeypatch, tmp_path):
    """Two torch builds hold no Gate authority, for two models as for one (worker_runtime.py)."""
    from dgx_monarch.worker_runtime import WorkerRuntimeMismatch

    graph = doc(IDEOGRAM)
    handle, topology, _dispatched = mesh_rig(monkeypatch, tmp_path, no_material_rows(), [])
    model = graph_model(graph)
    uncond = uncond_model(graph, model).request_dict()

    def mismatch(*_args, **_kwargs):
        raise WorkerRuntimeMismatch("the head runs one torch and the sibling another")

    monkeypatch.setattr(gate_identity, "gate_capability_context", mismatch)
    with pytest.raises(WorkerRuntimeMismatch):
        _authorize(model, handle, uncond, topology)
