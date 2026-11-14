"""The UNET loader leaves a placement an identity gate would drop to the gate (#157).

On 2026-11-06 the Kandinsky 5 Pro template on an explicit uly2 topology placed
its 40.4 GiB file on both boxes at the loader, then the stale combination's
ceremony unloaded every rank and loaded the file again for proof render 1. The
loader now defers that placement when a ceremony will run before the first
render, so the file loads once, inside proof render 1, and the ceremony, its
verdict and the render are what they were (nodes/loader_gate_defer.py).
"""
from __future__ import annotations

import contextlib
import copy
import json
from pathlib import Path

import pytest
import torch
from benchmark.sweep import convert

import dgx_monarch.gate_ledger as ledger_mod
from dgx_monarch import capacity_agreement, first_render, mesh_setup
from dgx_monarch.nodes import (
    common,
    consent_projection,
    gate_inconclusive,
    loader_gate_defer,
    loader_preflight,
    loaders,
    render_preflight,
    render_session,
    samplers,
)
from dgx_monarch.nodes import gate as gate_mod
from first_use_gate_helpers import (  # noqa: F401  # autouse fixture import.
    REPORTS,
    TEMPLATES,
    _forget_classifications,
    doc,
    fresh_process,
    graph_model,
    mesh_rig,
    no_material_rows,
    proof_renders,
    rows,
)
from gate_orchestration_helpers import (  # noqa: F401  # autouse fixture import.
    _clear_process_gate_verdicts,
)

ROOT = Path(__file__).resolve().parents[1]
PRO = TEMPLATES / "dgx-monarch-kandinsky5-video-pro.json"
IDEOGRAM = TEMPLATES / "dgx-monarch-ideogram4-t2i.json"
UNET = "kandinsky5pro_t2v_sft_5s.safetensors"


def _api(path: Path) -> dict:
    return convert.convert(json.loads(path.read_text(encoding="utf-8")),
                           convert.repo_widget_names(ROOT))


def _loader_rig(monkeypatch, handle) -> dict:
    """Every answer around the eager load stubbed; records placements and deferrals."""
    # A residency model of the Workers: a placement or a render onto an empty
    # store loads the file; an unload empties the store.
    seen: dict = {"deferred": [], "placed": [], "loads": 0, "resident": False}
    monkeypatch.setattr(loader_preflight, "preflight_upstream_gated_artifact", lambda *_a: None)
    monkeypatch.setattr(loader_preflight, "preflight_comfy_managed_topology", lambda *_a: None)
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint", lambda *_a, **_k: None)
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, **_k: None)
    monkeypatch.setattr(consent_projection, "project_for_loader", lambda *_a, **_k: None)
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel", lambda *_a: None)
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *_a, **_k: None)
    monkeypatch.setattr(render_session, "mutation_render_session",
                        lambda _handle: contextlib.nullcontext())
    monkeypatch.setattr(capacity_agreement, "agree_load", lambda *_a, **_k: None)
    monkeypatch.setattr(loaders, "ensure_live", lambda *_a, **_k: handle)
    monkeypatch.setattr(first_render, "load_deferred",
                        lambda endpoint, reason="": seen["deferred"].append((endpoint, reason)))
    handle.defunct = False
    inner = handle.call_all

    def call_all(method, *args, **kwargs):
        if method in ("load_model", "load_uncond_model"):
            seen["placed"].append((method, args[0]))
            seen["loads"] += 1
            seen["resident"] = True
            return [{"host": f"rank{rank}", "transition": "load"} for rank in range(2)]
        if method == "unload":
            seen["resident"] = False
        return inner(method, *args, **kwargs)

    handle.call_all = call_all
    return seen


def _persist(tmp_path, model, verdict: str, commit: str, uncond=None) -> None:
    """One ledger row for this combination's exact context, under ``commit``."""
    state_token = common._auto_gate_context(
        model, "ksampler", None, None, *(() if uncond is None else (uncond,)))
    assert state_token is not None
    key, artifacts = (gate_mod._combo_of(model) if uncond is None
                      else gate_mod._dual_combo_of(model, uncond))
    ledger_mod.GateLedger(str(tmp_path)).record(
        key, artifacts, commit, verdict,
        {gate_inconclusive.RESULT_KEY: gate_inconclusive.KIND_NO_MATERIAL}
        if verdict == "INCONCLUSIVE" else {"quarantine_levers": []},
        state_token[1][3])


def _render(model) -> None:
    cond = [[torch.ones(1, 4, 8), {}]]
    samplers.DGXMonarchKSampler().sample(
        model, 0, 4, 5.0, "euler_ancestral", "beta", cond, cond,
        {"samples": torch.zeros(1, 16, 2, 8, 8)})


def _pro_queue(monkeypatch, tmp_path, *, defer: bool = True):
    """The Pro template on uly2 with a stale PASS row: loader, then KSampler."""
    graph = doc(PRO)
    latent = torch.zeros(1, 16, 2, 8, 8)
    handle, _topology, dispatched = mesh_rig(
        monkeypatch, tmp_path, no_material_rows(), [latent.clone(), latent.clone()])
    seen = _loader_rig(monkeypatch, handle)
    proof_render = gate_mod.run_render

    def render(*args, **kwargs):
        if not seen["resident"]:  # a proof render onto an empty store loads the file
            seen["loads"] += 1
            seen["resident"] = True
        return proof_render(*args, **kwargs)

    monkeypatch.setattr(gate_mod, "run_render", render)
    if not defer:
        monkeypatch.setattr(loader_gate_defer, "ceremony_drops_load", lambda *_a: False)
    probe = graph_model(graph, widgets={"topology": "uly2"})
    _persist(tmp_path, probe, "PASS", "an-older-comfy-commit")
    assert common._auto_gate_context(probe, "ksampler")[0] == "stale"
    (model,) = loaders.DGXMonarchUNETLoader().load(
        probe.mesh, UNET, prompt=_api(PRO), unique_id="3")
    _render(model)
    return handle, seen, dispatched


def _stable(report: dict) -> dict:
    return {key: value for key, value in report.items() if key not in ("time", "wall_s")}


def test_a_stale_combination_loads_once_in_the_ceremony_with_the_same_verdict(
    monkeypatch, tmp_path,
):
    handle, seen, dispatched = _pro_queue(monkeypatch, tmp_path / "deferred")
    assert seen["placed"] == []
    assert seen["deferred"] == [("load_model", loader_gate_defer.REASON)]
    assert seen["loads"] == 1  # proof render 1's, and no other
    assert len(proof_renders(handle)) == 2 and handle.unload_calls >= 1
    deferred_report = rows(tmp_path / "deferred", REPORTS)

    monkeypatch.undo()
    fresh_process()  # the second run is a fresh driver with its own ledger
    eager_handle, eager_seen, eager_dispatched = _pro_queue(
        monkeypatch, tmp_path / "eager", defer=False)
    assert eager_seen["placed"] == [("load_model", UNET)]
    assert eager_seen["loads"] == 2  # the loader's, dropped, then proof render 1's
    eager_report = rows(tmp_path / "eager", REPORTS)
    # One placement fewer, and the ceremony, its verdict and the render as before.
    assert [_stable(row) for row in deferred_report] == [_stable(row) for row in eager_report]
    assert deferred_report[0]["verdict"] == "INCONCLUSIVE"
    assert len(proof_renders(eager_handle)) == len(proof_renders(handle))
    assert [entry["request"]["kind"] for entry in dispatched] == [
        entry["request"]["kind"] for entry in eager_dispatched]


@pytest.fixture
def pro_loader(monkeypatch, tmp_path):
    handle, _topology, _dispatched = mesh_rig(monkeypatch, tmp_path, no_material_rows(), [])
    seen = _loader_rig(monkeypatch, handle)
    model = graph_model(doc(PRO), widgets={"topology": "uly2"})
    return model, seen


def _load(model, prompt, unique_id="3"):
    loaders.DGXMonarchUNETLoader().load(model.mesh, model.unet_name, prompt=prompt,
                                        unique_id=unique_id)


def test_a_vouched_combination_still_loads_at_the_loader(pro_loader, tmp_path):
    model, seen = pro_loader
    _persist(tmp_path, model, "PASS", "known")
    _load(model, _api(PRO))
    assert seen["placed"] == [("load_model", UNET)] and seen["deferred"] == []


def _with_lora(prompt: dict, *, linked: bool = False) -> dict:
    graph = copy.deepcopy(prompt)
    graph["40"] = {"class_type": "DGXMonarchLoraLoader", "inputs": {
        "model": ["3", 0], "lora_name": ["41", 0] if linked else "style.safetensors",
        "strength_model": 0.8}}
    graph["4"]["inputs"]["model"] = ["40", 0]
    return graph


def test_a_lora_chain_defers_when_the_lora_combination_needs_its_gate(pro_loader, tmp_path):
    model, seen = pro_loader
    _persist(tmp_path, model, "PASS", "known")  # the bare model is vouched
    _load(model, _with_lora(_api(PRO)))
    assert seen["placed"] == [] and seen["deferred"] == [("load_model", loader_gate_defer.REASON)]


def test_a_vouched_lora_combination_keeps_the_eager_load(pro_loader, tmp_path):
    model, seen = pro_loader
    _persist(tmp_path, model.with_lora("style.safetensors", 0.8), "PASS", "known")
    _load(model, _with_lora(_api(PRO)))
    assert seen["placed"] == [("load_model", UNET)]


@pytest.mark.parametrize("change", ["fleet", "unknown_consumer", "linked_lora", "no_prompt"])
def test_any_doubt_keeps_the_eager_load(pro_loader, change):
    model, seen = pro_loader
    prompt = _api(PRO)
    if change == "fleet":
        prompt["50"] = {"class_type": "DGXMonarchFleetKSampler", "inputs": {"model": ["3", 0]}}
    elif change == "unknown_consumer":
        prompt["50"] = {"class_type": "SomeoneElsesModelNode", "inputs": {"model": ["4", 0]}}
    elif change == "linked_lora":
        prompt = _with_lora(prompt, linked=True)
    _load(model, None if change == "no_prompt" else prompt)
    assert seen["placed"] == [("load_model", UNET)] and seen["deferred"] == []


def test_an_identity_gate_node_always_defers_its_loader(pro_loader, tmp_path):
    model, seen = pro_loader
    _persist(tmp_path, model, "PASS", "known")
    prompt = _api(PRO)
    prompt["9"]["class_type"] = "DGXMonarchIdentityGate"
    _load(model, prompt)
    assert seen["placed"] == []


def test_a_projected_consent_keeps_its_eager_load(pro_loader, monkeypatch):
    model, seen = pro_loader
    monkeypatch.setattr(consent_projection, "project_for_loader", lambda *_a, **_k: object())
    _load(model, _api(PRO))
    assert seen["placed"] == [("load_model", UNET)]


def test_both_ideogram_loaders_defer_on_first_use_and_load_once_vouched(
    monkeypatch, tmp_path,
):
    from first_use_gate_helpers import uncond_model

    graph = doc(IDEOGRAM)
    handle, _topology, _dispatched = mesh_rig(monkeypatch, tmp_path, no_material_rows(), [])
    seen = _loader_rig(monkeypatch, handle)
    model = graph_model(graph)
    uncond = uncond_model(graph, model)
    prompt = _api(IDEOGRAM)
    loaders.DGXMonarchUNETLoader().load(model.mesh, model.unet_name, prompt=prompt, unique_id="3")
    loaders.DGXMonarchUncondUNETLoader().load(
        model.mesh, uncond.unet_name, prompt=prompt, unique_id="4")
    assert seen["placed"] == []
    assert [endpoint for endpoint, _reason in seen["deferred"]] == [
        "load_model", "load_uncond_model"]

    _persist(tmp_path, model, "PASS", "known", uncond=uncond.request_dict())
    loaders.DGXMonarchUNETLoader().load(model.mesh, model.unet_name, prompt=prompt, unique_id="3")
    assert seen["placed"] == [("load_model", model.unet_name)]


def test_a_lookup_error_keeps_the_eager_load(pro_loader, monkeypatch):
    model, seen = pro_loader

    def broken(*_args, **_kwargs):
        raise RuntimeError("the ledger cannot be read")

    monkeypatch.setattr(common, "_auto_gate_context", broken)
    _load(model, _api(PRO))
    assert seen["placed"] == [("load_model", UNET)] and seen["deferred"] == []


@pytest.mark.parametrize("other", ["missing", "unknown"])
def test_a_dual_guider_whose_other_chain_reaches_no_loader_keeps_the_eager_load(
    monkeypatch, tmp_path, other,
):
    graph = doc(IDEOGRAM)
    handle, _topology, _dispatched = mesh_rig(monkeypatch, tmp_path, no_material_rows(), [])
    seen = _loader_rig(monkeypatch, handle)
    model = graph_model(graph)
    prompt = _api(IDEOGRAM)
    guider = next(key for key, entry in prompt.items()
                  if entry["class_type"] == "DGXMonarchDualModelGuider")
    if other == "missing":
        prompt[guider]["inputs"]["model_negative"] = ["999", 0]
    else:
        prompt["60"] = {"class_type": "SomeoneElsesModelNode",
                        "inputs": {"model": prompt[guider]["inputs"]["model_negative"]}}
        prompt[guider]["inputs"]["model_negative"] = ["60", 0]
    loaders.DGXMonarchUNETLoader().load(model.mesh, model.unet_name, prompt=prompt, unique_id="3")
    assert seen["placed"] == [("load_model", model.unet_name)] and seen["deferred"] == []


def test_a_basic_scheduler_keeps_the_eager_load(pro_loader):
    """compute_sigmas loads the model itself before the ceremony; deferring saves nothing."""
    model, seen = pro_loader
    prompt = _api(PRO)
    prompt["70"] = {"class_type": "DGXMonarchBasicScheduler",
                    "inputs": {"model": ["4", 0], "scheduler": "beta", "steps": 8, "denoise": 1.0}}
    _load(model, prompt)
    assert seen["placed"] == [("load_model", UNET)] and seen["deferred"] == []
