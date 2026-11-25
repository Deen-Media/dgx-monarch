"""Every sampler entry carries the sequence tag to the route auto resolves.

The finding the tag answers: a Qwen-Image-Edit 2511 FP8 render with a prompt
and a negative of equal text rows and the file's default index_timestep_zero
references. Row 68's pair scope would send the pair to uly2, whose sharded
forward refuses those references; the tag keeps it on cfg2 row 68, which runs
them. Each entry here (KSampler, KSampler Advanced, the Pipeline sampler,
Sampler Custom behind either CFG guider, and the Gate node) reads the pair,
and the driver resolves the latent it hands on. Dropping the tag at any one
entry turns that entry's test red.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.nodes import common, guiders, samplers
from dgx_monarch.nodes.render_preflight import CFG_PAIR_SEQUENCE_REFUSED_METADATA_KEY as SEQ
from test_pair_scope_fallback_runs import _model, _pair, _refs, checkpoint_files

_PX = 1216  # 1.48 MP, inside row 68's band


@pytest.fixture
def edit_2511(monkeypatch, tmp_path):
    """The model and an equal pair with the file's default references."""
    checkpoint_files(monkeypatch, tmp_path)("F8_E4M3", "__index_timestep_zero__")
    return (_model("qwen_image"), *_pair(_refs(), _refs()))


def _latent():
    return {"samples": torch.zeros(1, 16, _PX // 8, _PX // 8)}


def _keeps_row_68(model, latent):
    assert latent.get(SEQ) is True
    topology, _sage, reason = common._resolve_topology_for_latent(model, latent, 4.0, 2)
    assert topology.describe() == "cfg2" and "table row 68" in reason


@pytest.fixture
def rendered(monkeypatch):
    seen: list[dict] = []

    def run_render(model, request, latent, cfg_value, steps_hint):
        seen.append(latent)
        return {"samples": torch.zeros(1, 16, 8, 8)}

    monkeypatch.setattr(samplers, "run_render", run_render)
    monkeypatch.setattr(samplers, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(samplers, "krea2_ref_preflight_summary", lambda *_args: None)
    return seen


def test_the_ksampler_carries_the_tag(edit_2511, rendered):
    model, positive, negative = edit_2511
    samplers.DGXMonarchKSampler().sample(model, 0, 8, 4.0, "euler", "simple", positive,
                                         negative, _latent())
    (latent,) = rendered
    _keeps_row_68(model, latent)


def test_the_advanced_ksampler_carries_the_tag(edit_2511, rendered):
    model, positive, negative = edit_2511
    samplers.DGXMonarchKSamplerAdvanced().sample(
        model, "enable", 0, 8, 4.0, "euler", "simple", positive, negative, _latent(), 0, 10000,
        "disable")
    (latent,) = rendered
    _keeps_row_68(model, latent)


def test_the_pipeline_sampler_carries_the_tag(edit_2511, rendered, monkeypatch):
    model, positive, negative = edit_2511
    pushed: list[dict] = []

    class Pipeline:
        def __init__(self, depth, on_step=None):
            self.depth = depth

        def push(self, model, request, latent, cfg_value, steps_hint):
            pushed.append(latent)

        def drain(self):
            return [{"samples": torch.zeros(1, 16, 8, 8)} for _latent in pushed]

    monkeypatch.setattr(samplers, "auto_gate_required", lambda *_args: len(rendered) == 0)
    monkeypatch.setattr(samplers, "RenderPipeline", Pipeline)
    model = dataclasses.replace(model, mesh=dataclasses.replace(model.mesh, pipeline_depth=2))
    samplers.DGXMonarchKSamplerPipeline().sample(model, "1, 2", 8, 4.0, "euler", "simple",
                                                 positive, negative, _latent())
    assert rendered and pushed
    for latent in (*rendered, *pushed):
        _keeps_row_68(model, latent)


@pytest.mark.parametrize("guider", ["cfg", "dual"])
def test_sampler_custom_carries_the_tag_from_either_cfg_guider(edit_2511, rendered,
                                                               monkeypatch, guider):
    model, positive, negative = edit_2511
    monkeypatch.setattr(guiders, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(guiders, "krea2_ref_preflight_summary", lambda *_args: None)
    monkeypatch.setattr(samplers, "_canonical_noise_for_wire", lambda _noise: None)
    if guider == "cfg":
        (built,) = guiders.DGXMonarchCFGGuider().get_guider(model, positive, negative, 4.0)
    else:
        (built,) = guiders.DGXMonarchDualModelGuider().get_guider(model, positive, 4.0,
                                                                  negative=negative)
    samplers.DGXMonarchSamplerCustom().sample(SimpleNamespace(seed=0), built, object(),
                                              torch.tensor([1.0, 0.0]), _latent())
    (latent,) = rendered
    _keeps_row_68(model, latent)


def test_the_gate_node_carries_the_tag(edit_2511, monkeypatch):
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes.gate_node import DGXMonarchIdentityGate

    model, positive, negative = edit_2511
    proved: list[dict] = []

    def ceremony(model, request, latent, *_args, **_kwargs):
        proved.append(latent)
        return {"latent": latent, "verdict": "PASS"}

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    monkeypatch.setattr(gate_mod, "conditioning_for_wire", lambda value: value)
    DGXMonarchIdentityGate().gate(model, positive, negative, _latent(), 0, 8, 4.0, "euler",
                                  "simple")
    (latent,) = proved
    _keeps_row_68(model, latent)
