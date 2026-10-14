"""A cfg++ sampler keeps the uncond pass at CFG 1, so auto may pick cfg2 there.

The KSampler, Pipeline, Advanced and Gate nodes tag the latent they hand on
when their sampler keeps that pass. Every site that resolves the render's
topology reads the tag off that latent, so the Gate token and the render
agree, and the tag never reaches the wire or the output.

No comfy, no GPU, no mesh: the render and resolution seams are stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.nodes import common, gate_identity, render_preflight, render_result, samplers
from dgx_monarch.topology import Topology

KEY = render_preflight.CFG_KEEPS_UNCOND_METADATA_KEY


def _capture_run_render(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    def run_render(model, request, latent, cfg_value, steps_hint):
        seen.append(latent)
        return {"samples": torch.zeros(1, 4, 8, 8)}

    monkeypatch.setattr(samplers, "run_render", run_render)
    monkeypatch.setattr(samplers, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(samplers, "krea2_ref_preflight_summary", lambda *_args: None)
    return seen


@pytest.mark.parametrize(("sampler", "tagged"), [
    ("euler_cfg_pp", True), ("res_multistep_cfg_pp", True), ("euler", False)])
def test_the_ksampler_tags_only_a_cfg_pp_latent(monkeypatch, sampler, tagged):
    seen = _capture_run_render(monkeypatch)
    source = {"samples": torch.zeros(1, 4, 8, 8)}
    samplers.DGXMonarchKSampler().sample(
        "model", 0, 8, 1.0, sampler, "simple", [], [], source)
    (latent,) = seen
    assert latent.get(KEY, False) is tagged
    assert KEY not in source  # the node's input is never written


@pytest.mark.parametrize(("sampler", "tagged"), [("euler_cfg_pp", True), ("euler", False)])
def test_the_advanced_sampler_tags_only_a_cfg_pp_latent(monkeypatch, sampler, tagged):
    seen = _capture_run_render(monkeypatch)
    samplers.DGXMonarchKSamplerAdvanced().sample(
        "model", "enable", 0, 8, 1.0, sampler, "simple", [], [],
        {"samples": torch.zeros(1, 4, 8, 8)}, 0, 10000, "disable")
    (latent,) = seen
    assert latent.get(KEY, False) is tagged


def test_a_plain_sampler_drops_an_inherited_tag(monkeypatch):
    seen = _capture_run_render(monkeypatch)
    samplers.DGXMonarchKSampler().sample(
        "model", 0, 8, 1.0, "euler", "simple", [], [],
        {"samples": torch.zeros(1, 4, 8, 8), KEY: True})
    (latent,) = seen
    assert KEY not in latent


def test_the_custom_sampler_never_carries_a_tag(monkeypatch):
    # A SAMPLER object has no name to read, so the fail-closed answer holds.
    seen = _capture_run_render(monkeypatch)
    monkeypatch.setattr(samplers, "_canonical_noise_for_wire", lambda _noise: None)
    guider = {"model": "model", "spec": {}, "cfg": 1.0}
    samplers.DGXMonarchSamplerCustom().sample(
        SimpleNamespace(seed=0), guider, object(), torch.tensor([1.0, 0.0]),
        {"samples": torch.zeros(1, 4, 8, 8), KEY: True})
    (latent,) = seen
    assert KEY not in latent


def test_the_pipeline_hands_its_gate_check_and_every_render_the_tagged_latent(monkeypatch):
    """The first-seed check computes a Gate token for the topology the renders
    resolve, so both auto_gate_required calls, the first render and every
    pushed render get the same tagged latent."""
    seen = _capture_run_render(monkeypatch)
    checked: list[dict] = []
    pushed: list[dict] = []

    def required(_model, _kind, latent, _cfg):
        checked.append(latent)
        return len(checked) == 1

    class Pipeline:
        def __init__(self, depth, on_step=None):
            self.depth = depth

        def push(self, model, request, latent, cfg_value, steps_hint):
            pushed.append(latent)

        def drain(self):
            return [{"samples": torch.zeros(1, 4, 8, 8)} for _latent in pushed]

    monkeypatch.setattr(samplers, "auto_gate_required", required)
    monkeypatch.setattr(samplers, "RenderPipeline", Pipeline)
    model = SimpleNamespace(mesh=SimpleNamespace(pipeline_depth=2))
    samplers.DGXMonarchKSamplerPipeline().sample(
        model, "1, 2, 3", 8, 1.0, "euler_cfg_pp", "simple", [], [],
        {"samples": torch.zeros(1, 4, 8, 8)})
    assert len(checked) == 2 and len(seen) == 1 and len(pushed) == 2
    assert all(latent.get(KEY) is True for latent in (*checked, *seen, *pushed))


def test_the_gate_node_proves_the_tagged_render(monkeypatch):
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes.gate_node import DGXMonarchIdentityGate

    proved: list[dict] = []

    def ceremony(model, request, latent, *_args, **_kwargs):
        proved.append(latent)
        return {"latent": latent, "verdict": "PASS"}

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    monkeypatch.setattr(gate_mod, "conditioning_for_wire", lambda value: value)
    model = SimpleNamespace(mesh=SimpleNamespace(attention="TORCH_FLASH"))
    DGXMonarchIdentityGate().gate(model, [], [], {"samples": torch.zeros(1, 4, 8, 8)},
                          0, 8, 1.0, "euler_cfg_pp", "simple")
    (latent,) = proved
    assert latent.get(KEY) is True


def test_the_tag_never_reaches_the_wire_or_the_output(monkeypatch):
    latent = {"samples": torch.zeros(1, 4, 8, 8), KEY: True, "noise_mask": "kept"}
    assert KEY not in render_preflight.latent_without_topology_metadata(latent)
    monkeypatch.setattr(render_result, "verify_cross_rank_signatures", lambda *_args: None)
    monkeypatch.setattr(render_result, "read_latent_result", lambda value: value)
    out = render_result._finish_render(
        [{"rank": 0, "dp_rank": 0, "latent": torch.ones(1, 4, 8, 8)}],
        SimpleNamespace(dp=1), latent)
    assert KEY not in out and out["noise_mask"] == "kept"
    # The ceremony's frozen copy keeps it, so its proof resolves the same way.
    assert gate_identity.freeze_transaction(latent)[KEY] is True


def test_the_resolver_forwards_the_tag_only_when_set(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(common, "resolve_topology", lambda *_args, **kwargs: calls.append(kwargs))
    samples = object()
    for latent in ({"samples": samples, KEY: True}, {"samples": samples},
                   {"samples": samples, KEY: False}):
        common._resolve_topology_for_latent("model", latent, 1.0, 2)
    assert calls == [{"keeps_uncond": True}, {}, {}]


def _spy_resolver(monkeypatch) -> list[bool]:
    """Resolve cfg2 for a tagged latent and uly2 otherwise, and record the flag."""
    seen: list[bool] = []

    def resolve(_model, _samples, _cfg, world=None, **kwargs):
        kept = bool(kwargs.get("keeps_uncond"))
        seen.append(kept)
        topology = Topology(world=2, cfg=2) if kept else Topology(world=2, ulysses=2)
        return topology, False, "spy"

    monkeypatch.setattr(common, "resolve_topology", resolve)
    return seen


@pytest.mark.parametrize("tagged", [True, False])
def test_the_submit_path_gate_context_and_setup_token_resolve_alike(
        monkeypatch, tmp_path, tagged):
    """The submit path and auto_gate_context run on every first-use render;
    the setup token is the ceremony's reassertion. All three read the one
    latent, so all three resolve cfg2 when it is tagged and uly2 when not."""
    from dgx_monarch import gate_ledger, mesh_setup
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes.pending import PendingRenderHandoff
    from dgx_monarch.nodes.setup_binding import render_setup_token
    from render_sessions_helpers import _real_handle, _stub_direct_submit

    latent = {"samples": torch.zeros(1, 4, 8, 8), **({KEY: True} if tagged else {})}
    expected = "cfg2" if tagged else "uly2"

    # 1. The submit path, whose request latent leaves without the tag.
    handle = _real_handle()
    wired: list[dict] = []
    spec = _stub_direct_submit(
        monkeypatch, handle, lambda _handle, request, *_args, **_kwargs: wired.append(request))
    seen = _spy_resolver(monkeypatch)
    setups: list[Topology] = []
    monkeypatch.setattr(mesh_setup, "ensure_request_setup",
                        lambda _handle, topology, *_args, **_kwargs: setups.append(topology))
    monkeypatch.setattr(common, "validate_render_topology", lambda *_args: None, raising=False)
    pending = common.submit_render(
        spec, {}, latent, cfg_value=1.0, steps_hint=2, handoff=PendingRenderHandoff())
    pending.abandon()
    assert KEY not in wired[0]["latent"]

    # 2. auto_gate_context, which feeds the Gate capability context.
    gate_handle = SimpleNamespace(
        world=2, config=SimpleNamespace(worker_args={}, hosts=(), source=""))
    gate_handle.effective_worker_args = lambda requested: dict(requested)
    gate_model = SimpleNamespace(
        unet_name="model.safetensors", options={}, loras=(),
        mesh=SimpleNamespace(handle=gate_handle, auto_gate="first_use", worker_args={},
                             topology_preset="auto", attention="TORCH_FLASH",
                             sync_ulysses=False))
    contexts: list[Topology] = []
    monkeypatch.setattr(
        gate_mod, "gate_capability_context",
        lambda *_args, resolved_topology=None, **_kwargs: contexts.append(resolved_topology) or {})
    artifacts = gate_ledger.ArtifactSetSignature("artifact", "artifact", True)
    monkeypatch.setattr(gate_mod, "_combo_of", lambda _model: ("target", artifacts))
    monkeypatch.setattr(gate_mod, "_ledger_dir", lambda: str(tmp_path))
    monkeypatch.setattr(gate_ledger, "comfy_commit", lambda: "commit")
    common._auto_gate_context(gate_model, "ksampler", latent, 1.0)

    # 3. The setup token a ceremony reasserts before its follow-up calls.
    setup_handle = _real_handle()
    setup_handle.world = 2
    render_setup_token(spec, latent, 1.0, setup_handle)

    assert seen == [tagged] * len(seen) and len(seen) >= 3
    assert [topology.describe() for topology in (*setups, *contexts)] == [expected] * (
        len(setups) + len(contexts))
    assert setups and contexts
