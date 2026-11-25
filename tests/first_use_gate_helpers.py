"""The end-to-end first-use rig the Sampler Custom and dual-model gate suites share.

A shipped graph's own Init, guider and Sampler Custom nodes drive the real
``run_render``, ``maybe_auto_gate`` and ceremony. The stand-ins are the Worker
mesh (the cross-mode rig's handle), the model files (signatures come from the
rig's ``_identity``), the topology resolver and the final dispatch, which
records what it would send. The ledger is a real file in ``tmp_path``.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.gate_ledger as ledger_mod
from dgx_monarch import first_render
from dgx_monarch.gate_dual_identity import dual_combo_key
from dgx_monarch.nodes import (
    common,
    gate_inconclusive,
    gate_process_state,
    guiders,
    render_preflight,
    samplers,
)
from dgx_monarch.nodes import gate as gate_mod
from dgx_monarch.nodes import init as init_module
from dgx_monarch.nodes.common import ModelSpec
from dgx_monarch.topology import Topology
from gate_orchestration_helpers import _cross_mode_rig

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "example_workflows"
GENERATED = ROOT / "tests" / "fixtures" / "workflows" / "generated"
WORLD = 2
REPORTS = "dgxm_gate_reports.jsonl"


@pytest.fixture(autouse=True)
def _forget_classifications():
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
    common._AUTO_GATE_RUNNING.clear()
    previous = getattr(common._AUTO_GATE_ACTIVE, "on", False)
    common._AUTO_GATE_ACTIVE.on = False
    yield
    common._AUTO_GATE_ACTIVE.on = previous
    common._AUTO_GATE_RUNNING.clear()
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()


class Sampler:
    """A SAMPLER object as the graph's KSamplerSelect hands it over."""

    def __init__(self, name: str):
        self.name = name


def doc(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def node(graph: dict, kind: str) -> dict:
    return next(entry for entry in graph["nodes"] if entry["type"] == kind)


def types(graph: dict) -> set[str]:
    return {entry["type"] for entry in graph["nodes"]}


def sigmas(graph: dict) -> torch.Tensor:
    if "ManualSigmas" not in types(graph):  # Ideogram4Scheduler computes its own
        return torch.linspace(1.0, 0.0, 21)
    (text,) = node(graph, "ManualSigmas")["widgets_values"]
    return torch.tensor([float(value) for value in text.split(",")])


def _signatures(loras) -> list[str]:
    return ["model-sig", *(f"lora-sig-{index}" for index in range(len(list(loras or ()))))]


def cycle_rows(**fields) -> list[dict]:
    """One swap-cycle report per rank; the rig adds rank, world and generation."""
    return [dict(fields) for _rank in range(WORLD)]


def no_material_rows() -> list[dict]:
    return cycle_rows(conclusive=False, no_material=True, family="unvouched",
                      reason="no lora stack: lazy swap is not applicable",
                      slab_active=False, certificate=None)


def swapped_rows(family: str = "unvouched") -> list[dict]:
    return cycle_rows(conclusive=True, family=family, slab_active=False, certificate=None,
                      transitions={"load": "load", "nudge": "hot-swap", "back": "hot-swap"})


def mesh_rig(monkeypatch, tmp_path, rows, renders, *, uncond_rows=None, topology=None):
    """The rig's Worker handle at world 2, a real ledger, and a recording dispatch."""
    handle, _ledger, _model = _cross_mode_rig(
        monkeypatch, tmp_path, renders, {}, cycle_response=rows)
    handle.world = WORLD
    handle.ensure_setup = lambda *_args, **_kwargs: None
    handle.cycle_calls = []
    inner = handle.call_all

    def call_all(method, *args, **kwargs):
        if method == "gate_swap_cycle":
            handle.cycle_calls.append((args, dict(kwargs)))
            if kwargs.get("slot") == "uncond" and uncond_rows is not None:
                handle.calls.append((method, args))
                return [{"rank": rank, "world": WORLD,
                         "setup_generation": handle.setup_generation, **dict(row)}
                        for rank, row in enumerate(uncond_rows)]
        return inner(method, *args, **kwargs)

    handle.call_all = call_all

    def combo_of(model):
        names = [entry["name"] for entry in model.loras]
        return (ledger_mod.combo_key(model.unet_name, model.options, names),
                ledger_mod.artifact_set_signature(_signatures(model.loras)))

    def dual_combo_of(model, uncond):
        spec = {"unet_name": model.unet_name, "options": model.options, "loras": model.loras}
        return (dual_combo_key(spec, uncond), ledger_mod.artifact_set_signature(
            [*_signatures(model.loras), *_signatures(uncond.get("loras"))]))

    resolved = topology or Topology(ulysses=WORLD, world=WORLD)
    monkeypatch.setattr(gate_mod, "GateLedger", ledger_mod.GateLedger)
    monkeypatch.setattr(gate_mod, "_combo_of", combo_of)
    monkeypatch.setattr(gate_mod, "_dual_combo_of", dual_combo_of)
    monkeypatch.setattr(ledger_mod, "comfy_commit", lambda: "known")
    monkeypatch.setattr(common, "ensure_live", lambda value: value)
    monkeypatch.setattr(common, "_bind_packed_render_model",
                        lambda model, *_args, **_kwargs: (model, None))
    monkeypatch.setattr(common, "_resolve_topology_for_latent",
                        lambda *_args: (resolved, False, "test topology"))
    monkeypatch.setattr(render_preflight, "activation_footprint_preflight_for_request",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(samplers.requeue_price, "price_cached_loader",
                        lambda *_args, **_kwargs: None)
    monkeypatch.setattr(init_module, "get_mesh", lambda **_kwargs: handle)
    monkeypatch.setattr(first_render, "nccl_deferred", lambda: None)
    dispatched: list[dict] = []

    class Pending:
        def __init__(self, latent):
            self.latent = latent

        def result(self):
            samples = self.latent["samples"]
            return {"samples": samples, "_dgxm_denoised": samples.clone()}

    def submit(model, request, latent, *_args, **_kwargs):
        dispatched.append({"request": request, "worker_args": dict(model.mesh.worker_args),
                           "proof_renders": proof_renders(handle)})
        return Pending(latent)

    monkeypatch.setattr(common, "submit_render", submit)
    return handle, resolved, dispatched


def proof_renders(handle) -> list[dict]:
    return [args[1] for method, args in handle.calls if method == "render"]


def graph_model(graph: dict, loras=(), widgets: dict | None = None) -> ModelSpec:
    init = {**node(graph, "DGXMonarchInit")["dgxm_widgets"], **(widgets or {})}
    (mesh,) = init_module.DGXMonarchInit().init(**init)
    model = ModelSpec(mesh=mesh, unet_name=node(graph, "DGXMonarchUNETLoader")["widgets_values"][0])
    for name, strength in loras:
        model = model.with_lora(name, strength)
    return model


def uncond_model(graph: dict, model: ModelSpec, loras=()) -> ModelSpec:
    uncond = ModelSpec(mesh=model.mesh, slot="uncond",
                       unet_name=node(graph, "DGXMonarchUncondUNETLoader")["widgets_values"][0])
    for name, strength in loras:
        uncond = uncond.with_lora(name, strength)
    return uncond


def queue(graph: dict, model: ModelSpec, uncond: ModelSpec | None = None) -> dict:
    """One queue press: the graph's guider feeds the DGX Monarch Sampler Custom node."""
    cond = [[torch.ones(1, 4, 8), {}]]
    kinds = types(graph)
    if "DGXMonarchDualModelGuider" in kinds:
        (cfg,) = node(graph, "DGXMonarchDualModelGuider")["widgets_values"]
        (guider,) = guiders.DGXMonarchDualModelGuider().get_guider(model, cond, cfg, uncond)
    elif "DGXMonarchCFGGuider" in kinds:
        (cfg,) = node(graph, "DGXMonarchCFGGuider")["widgets_values"]
        (guider,) = guiders.DGXMonarchCFGGuider().get_guider(model, cond, cond, cfg)
    else:
        (guider,) = guiders.DGXMonarchBasicGuider().get_guider(model, cond)
    sampler_name = node(graph, "KSamplerSelect")["widgets_values"][0]
    seed = node(graph, "RandomNoise")["widgets_values"][0]
    output, _denoised = samplers.DGXMonarchSamplerCustom().sample(
        SimpleNamespace(seed=seed), guider, Sampler(sampler_name), sigmas(graph),
        {"samples": torch.zeros(1, 4, 8, 8)})
    return output


def rows(tmp_path: Path, name: str) -> list[dict]:
    path = tmp_path / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fresh_process() -> None:
    """What a driver restart forgets: every process-local verdict and memo."""
    common._AUTO_GATE_SESSION.clear()
    gate_process_state._PROCESS_GATE_DENIALS = {}
    gate_inconclusive._NO_MATERIAL_TOKENS.clear()
