"""Sampler Custom renders run the first-use identity gate, as KSampler renders do.

Before 1.5 the automatic gate covered only the two KSampler kinds, so the six
LTX graphs and p-h3-image rendered on stock residency until a user ran the
Identity Gate node by hand, and refused on a ``*+fsdp`` preset. Now a custom
request reaches the gate, and its proof renders replay the request's own call:
the same guider, SAMPLER and NOISE objects and seed, with the sigma schedule
cut to its first two steps (nodes/gate_proof_request.py).

The end-to-end tests drive each shipped graph's own nodes through the real
render path and ceremony (tests/first_use_gate_helpers.py).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
from dgx_monarch.nodes import auto_gate, common, gate_inconclusive, gate_proof_request
from dgx_monarch.nodes import gate as gate_mod
from dgx_monarch.nodes.gate_fsdp import token_requires_fsdp_proof
from dgx_monarch.nodes.gate_identity import authorize_normal_render
from dgx_monarch.topology import Topology
from first_use_gate_helpers import (  # noqa: F401  # autouse fixture import.
    GENERATED,
    REPORTS,
    TEMPLATES,
    WORLD,
    Sampler,
    _forget_classifications,
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
)
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
)

# The single-model Sampler Custom graphs that ship.
CUSTOM_GRAPHS = ("ltx25-t2v", "ltx25-i2v", "ltx25-i2v-guide", "ltx25-flf2v",
                 "ltx-t2v", "ltx-i2v-av", "p-h3-image")


@pytest.mark.parametrize("name", CUSTOM_GRAPHS)
def test_each_custom_graph_gates_on_first_use_then_renders_and_skips_after(
    name, monkeypatch, tmp_path,
):
    graph = doc(TEMPLATES / f"dgx-monarch-{name}.json")
    latent = torch.zeros(1, 4, 8, 8)
    handle, _topology, dispatched = mesh_rig(
        monkeypatch, tmp_path, no_material_rows(), [latent.clone(), latent.clone()])
    model = graph_model(graph)
    schedule = sigmas(graph)

    queue(graph, model)

    # The ceremony ran first, on the request's own call cut to two steps.
    proofs = dispatched[0]["proof_renders"]
    assert len(proofs) == 2
    for proof in proofs:
        assert proof["kind"] == "custom"
        assert torch.equal(proof["sigmas"], schedule[:3])
        assert proof["sampler_object"].name == node(graph, "KSamplerSelect")["widgets_values"][0]
        assert proof["noise_object"].seed == 42
        assert proof["guider"]["kind"] in ("cfg", "basic")
    # Its verdict is recorded where the KSampler graphs record theirs.
    (report,) = rows(tmp_path, REPORTS)
    assert report["origin"] == "auto_first_use"
    assert report["verdict"] == "INCONCLUSIVE"
    assert report[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    assert (report["sampler"], report["steps"], report["seed"]) == ("custom", 2, 42)
    verdicts = [row for row in rows(tmp_path, ledger_mod.LEDGER_NAME)
                if row.get("verdict") == "INCONCLUSIVE"]
    assert verdicts and verdicts[-1]["origin"] == "auto_first_use"
    # The user's render followed, whole.
    (render,) = dispatched
    assert render["request"]["kind"] == "custom"
    assert torch.equal(render["request"]["sigmas"], schedule)

    # The next queue skips the gate, in this process and in a fresh one.
    queue(graph, model)
    fresh_process()
    queue(graph, model)
    assert len(proof_renders(handle)) == 2
    assert len(dispatched) == 3
    assert len(rows(tmp_path, REPORTS)) == 1


def test_a_graph_the_driver_reads_as_nothing_to_compare_settles_with_no_proof_render(
    monkeypatch, tmp_path,
):
    """The same queues as above with the driver's read answering (an unvouched
    family, no consent): the Workers' empty-stack cycle settles the first use
    and the later queues skip, through the real auto path and ledger
    (nodes/gate_nothing_to_gate.py)."""
    from dgx_monarch import first_render
    from dgx_monarch.nodes import gate_nothing_to_gate

    monkeypatch.setattr(gate_nothing_to_gate, "_family", lambda _model: ("ltx", False))
    monkeypatch.setattr(gate_nothing_to_gate, "_rescue_consent_live", lambda _model: False)
    notices: list[tuple[str, str]] = []
    monkeypatch.setattr(first_render, "notice",
                        lambda phase, message, note, **_kw: notices.append((phase, message)))
    graph = doc(TEMPLATES / "dgx-monarch-ltx25-t2v.json")
    latent = torch.zeros(1, 4, 8, 8)
    handle, _topology, dispatched = mesh_rig(
        monkeypatch, tmp_path, no_material_rows(), [latent.clone(), latent.clone()])
    model = graph_model(graph)
    # Each queue's gate decision through common._maybe_auto_gate: what it
    # returned and the Worker calls it made.
    decisions: list[tuple[str | None, list[str]]] = []
    decide = common._maybe_auto_gate

    def recorded(*args, **kwargs):
        before = len(handle.calls)
        verdict = decide(*args, **kwargs)
        decisions.append((verdict, [method for method, _args in handle.calls[before:]]))
        return verdict

    monkeypatch.setattr(common, "_maybe_auto_gate", recorded)

    queue(graph, model)

    methods = [method for method, _args in handle.calls]
    assert proof_renders(handle) == [] and dispatched[0]["proof_renders"] == []
    assert methods.count("load_model") == 1 and methods.count("gate_swap_cycle") == 1
    ((verdict, calls),) = decisions
    assert verdict == gate_inconclusive.NO_MATERIAL_VERDICT
    assert [call for call in calls if call in (
        "provenance_baseline", "unload", "load_model", "gate_swap_cycle", "render")] == [
        "provenance_baseline", "unload", "load_model", "gate_swap_cycle", "provenance_baseline"]
    (report,) = rows(tmp_path, REPORTS)
    assert report[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    assert report[gate_inconclusive.BASIS_KEY] == gate_inconclusive.BASIS_NO_RENDER
    assert report["proof_renders"] == 0
    (row,) = [row for row in rows(tmp_path, ledger_mod.LEDGER_NAME)
              if row.get("verdict") == "INCONCLUSIVE"]
    assert row[gate_inconclusive.RESULT_KEY] == gate_inconclusive.KIND_NO_MATERIAL
    assert row[gate_inconclusive.BASIS_KEY] == gate_inconclusive.BASIS_NO_RENDER
    (closing,) = [message for phase, message in notices if phase == "gate_done"]
    assert "ran no proof render" in closing
    (render,) = dispatched
    assert render["request"]["kind"] == "custom"

    # The next queue skips the gate, in this process and in a fresh one.
    queue(graph, model)
    fresh_process()
    queue(graph, model)
    methods = [method for method, _args in handle.calls]
    assert methods.count("load_model") == 1 and methods.count("gate_swap_cycle") == 1
    assert proof_renders(handle) == [] and len(dispatched) == 3
    # The later decisions read the session verdict, then the real ledger row
    # in a fresh process, and make no Worker call at all.
    assert decisions[1:] == [(gate_inconclusive.NO_MATERIAL_VERDICT, [])] * 2
    assert len(rows(tmp_path, REPORTS)) == 1
    assert len([row for row in rows(tmp_path, ledger_mod.LEDGER_NAME)
                if row.get("verdict") == "INCONCLUSIVE"]) == 1


def test_a_lora_on_the_p_h3_image_route_earns_a_pass_and_a_residency_grant(
    monkeypatch, tmp_path,
):
    """Runbook cell 11: the lazy-swap cycle gives the LoRA a real comparison."""
    graph = doc(GENERATED / "dgx-monarch-test-p-h3-image-lora.json")
    (lora_name, strength) = node(graph, "DGXMonarchLoraLoader")["widgets_values"]
    latent = torch.zeros(1, 4, 8, 8)
    handle, topology, dispatched = mesh_rig(
        monkeypatch, tmp_path, swapped_rows("minimax_h3"), [latent.clone(), latent.clone()])
    model = graph_model(graph, loras=[(lora_name, strength)])

    queue(graph, model)

    (report,) = rows(tmp_path, REPORTS)
    assert (report["verdict"], report["proof_kind"]) == ("PASS", "lora_low_rss_swap")
    assert len(dispatched) == 1 and len(dispatched[0]["proof_renders"]) == 2
    granted = authorize_normal_render(
        model, handle, gate_active=False, session_verdict=common._process_gate_verdict,
        resolved_topology=topology, resolved_attention=model.mesh.attention)
    assert granted.residency_mode == "required"
    assert granted.residency_grant["combo_key"] == ledger_mod.combo_key(
        model.unet_name, model.options, [lora_name])


def test_a_custom_render_on_an_fsdp_preset_requires_the_clean_reload_proof(
    monkeypatch, tmp_path,
):
    """It used to refuse at authorization with no proof run (TROUBLESHOOTING, the FSDP entry)."""
    graph = doc(TEMPLATES / "dgx-monarch-p-h3-image.json")
    mesh_rig(monkeypatch, tmp_path, no_material_rows(), [],
             topology=Topology(ulysses=WORLD, fsdp=True, world=WORLD))
    model = graph_model(graph)
    latent = {"samples": torch.zeros(1, 4, 8, 8)}

    context = common._auto_gate_context(model, "custom", latent, 1.0)

    assert context is not None and token_requires_fsdp_proof(context[1])
    assert common.auto_gate_required(model, "custom", latent, 1.0) is True


def test_a_custom_render_under_comfy_managed_residency_reaches_the_gate(monkeypatch, tmp_path):
    """Refusal 6 tells the user to render once with auto_gate=first_use; that now reaches it."""
    graph = doc(TEMPLATES / "dgx-monarch-p-h3-image.json")
    mesh_rig(monkeypatch, tmp_path, no_material_rows(), [])
    model = graph_model(graph, widgets={"comfy_managed": "on"})
    latent = {"samples": torch.zeros(1, 4, 8, 8)}
    assert common._auto_gate_context(model, "custom", latent, 1.0) is not None
    assert common.auto_gate_required(model, "custom", latent, 1.0) is True


@pytest.mark.parametrize("kind", ["ksampler", "ksampler_advanced"])
def test_the_ksampler_proof_request_is_unchanged(kind):
    request = {"kind": kind, "steps": 20, "noise_seed": 7, "cfg": 3.5,
               "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
               "advanced": {"add_noise": False, "start_at_step": 4,
                            "end_at_step": 12, "return_with_leftover_noise": True}}
    light, steps = gate_proof_request.proof_request(request)
    assert steps == 2
    assert light == {
        **request,
        "kind": "ksampler_advanced",
        "steps": 2,
        "advanced": {"add_noise": True, "start_at_step": 0, "end_at_step": None,
                     "return_with_leftover_noise": False},
    }
    assert request["kind"] == kind and request["steps"] == 20


@pytest.mark.parametrize(("schedule", "kept", "steps"), [
    ([1.0, 0.9, 0.5, 0.2, 0.0], [1.0, 0.9, 0.5], 2),
    ([1.0, 0.5, 0.0], [1.0, 0.5, 0.0], 2),
    ([1.0, 0.0], [1.0, 0.0], 1),
])
def test_a_custom_proof_keeps_its_call_and_the_first_two_steps(schedule, kept, steps):
    sampler, noise = Sampler("euler"), SimpleNamespace(seed=9)
    request = {"kind": "custom", "guider": {"kind": "basic"}, "uncond_model": None,
               "sampler_object": sampler, "noise_object": noise, "noise_seed": 9,
               "sigmas": torch.tensor(schedule)}
    light, proof_steps = gate_proof_request.proof_request(request)
    assert proof_steps == steps
    assert light["kind"] == "custom"
    assert light["sigmas"].tolist() == pytest.approx(kept)
    assert light["sampler_object"] is sampler and light["noise_object"] is noise
    assert light["guider"] is request["guider"]
    assert "steps" not in light and "advanced" not in light
    assert torch.equal(request["sigmas"], torch.tensor(schedule))


def test_a_one_sigma_schedule_runs_no_ceremony_and_grants_nothing(monkeypatch):
    monkeypatch.setattr(common, "_auto_gate_context", lambda *_args: ("unknown", ("t",)))
    monkeypatch.setattr(gate_mod, "run_identity_ceremony",
                        lambda *_args, **_kwargs: pytest.fail("no proof to render"))
    request = {"kind": "custom", "guider": {"kind": "basic"}, "uncond_model": None,
               "sampler_object": Sampler("euler"), "noise_seed": 1,
               "sigmas": torch.tensor([1.0])}
    verdict = auto_gate.maybe_auto_gate(
        SimpleNamespace(unet_name="m"), request, {"samples": torch.zeros(1)}, 1.0, 1)
    assert verdict is None
    assert common._process_gate_verdict(("t",)) is None
    assert common._AUTO_GATE_RUNNING == set()


def test_a_custom_report_row_names_the_call_it_proved():
    request = {"kind": "custom", "guider": {"kind": "cfg", "cfg": 4.0},
               "noise_seed": 11, "sigmas": torch.tensor([1.0, 0.5, 0.0])}
    assert gate_proof_request.report_fields(request) == {
        "seed": 11, "steps": 2, "cfg": 4.0, "sampler": "custom", "scheduler": None}
    assert gate_proof_request.report_fields({**request, "guider": {"kind": "basic"}})["cfg"] == 1.0
    assert gate_proof_request.report_fields({**request, "sigmas": torch.tensor(1.0)})["steps"] is None
