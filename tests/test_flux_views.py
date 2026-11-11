"""UltraTex view conditioning on the Flux family: where it runs, and the refusal elsewhere (#148).

ComfyUI 0.45.0's UltraTex Conditioning node puts ``view_grid``,
``token_masks`` and ``reference_views`` on the conditioning, and stock's Flux
forward reads them (adapters/flux_views.py). One GPU, uly2 and uly2+fsdp are
proven equal to stock against a real ComfyUI
(tests/test_flux_views_comfy_exactness.py); every other topology refuses class
P on the driver and on every Worker rank, and auto skips the cfg rows.

No comfy, no GPU, no mesh: the render, sniff and resolution seams are stubbed.
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import cfg_parallel, flux_views
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import common, model_patch_preflight, render_preflight, samplers
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.topology import Topology, choose_auto_topology, topology_from_preset

SRC = Path(__file__).resolve().parents[1] / "src"
VIEWS = flux_views.VIEWS_METADATA_KEY
GRID = [(0, 0), (0, 2), (1, 0), (0, 1), (1, 1), (1, 2)]


def _ultratex(**overrides) -> list:
    """A conditioning as UltraTex Conditioning builds it, packed or not."""
    extras = {"reference_latents": [torch.zeros(1, 4, 2, 3), torch.zeros(1, 4, 1, 1)],
              "reference_latents_method": "index", "view_grid": GRID,
              "token_masks": {"target": torch.ones(6, 8, 8), "refs": [torch.ones(1, 16, 24), None]},
              "reference_views": [None, 0], "background_latent": torch.zeros(1, 4, 1, 1)}
    extras.update(overrides)
    return [[torch.zeros(1, 7, 8), extras]]


def _plain() -> list:
    return [[torch.zeros(1, 7, 8), {"pooled_output": None}]]


def _request(positive=None, negative=None, **extra) -> dict:
    return {"kind": "ksampler", "positive": positive if positive is not None else _ultratex(),
            "negative": negative if negative is not None else _plain(), **extra}


# --- What counts as a view-conditioned render ------------------------------------------


@pytest.mark.parametrize("key", ["view_grid", "token_masks", "reference_views"])
def test_any_one_of_the_three_keys_marks_the_conditioning(key):
    entry = [[torch.zeros(1, 7, 8), {key: _ultratex()[0][1][key]}]]
    assert flux_views.conditioning_carries_views(entry)
    assert flux_views.conditioning_carries_views(_plain(), entry)


@pytest.mark.parametrize("conditioning", [
    None, [], "text", [[torch.zeros(1, 7, 8)]], [[None, "options"]], _plain(),
    [[torch.zeros(1, 7, 8), {"view_grid": None, "token_masks": None, "reference_views": None}]],
    [[torch.zeros(1, 7, 8), {"background_latent": torch.zeros(1, 4, 1, 1)}]],
])
def test_absent_none_or_malformed_carries_nothing(conditioning):
    # model_base.Flux.extra_conds reads each key only when it is not None.
    assert not flux_views.conditioning_carries_views(conditioning)


def test_the_request_scan_reads_both_prompts_and_the_guiders():
    assert flux_views.request_carries_views(_request())
    assert flux_views.request_carries_views(_request(positive=_plain(), negative=_ultratex()))
    assert not flux_views.request_carries_views(_request(positive=_plain()))
    custom = {"kind": "custom", "guider": {"kind": "cfg", "positive": _ultratex(), "negative": _plain()}}
    assert flux_views.request_carries_views(custom)
    assert not flux_views.request_carries_views({"kind": "custom", "guider": {}})
    assert not flux_views.request_carries_views("request")


# --- Which topologies are proven ---------------------------------------------------------


@pytest.mark.parametrize(("preset", "world", "proven"), [
    ("single", 1, True), ("uly2", 2, True), ("uly2+fsdp", 2, True),
    ("single", 2, False),  # dp2: the preset keeps its degree and dp takes the rest
    ("cfg2", 2, False), ("dp2", 2, False), ("ring2", 2, False), ("cfg2+fsdp", 2, False),
    ("dp2+fsdp", 2, False), ("uly2", 4, False),
    ("uly2+cfg2", 4, False), ("uly4", 4, False),
])
def test_one_gpu_and_ulysses_at_degree_two_are_the_proven_topologies(preset, world, proven):
    topo = topology_from_preset(preset, world)
    assert flux_views.runs_views(topo) is proven
    as_pushed = {"ulysses": topo.ulysses, "ring": topo.ring, "cfg": topo.cfg,
                 "dp": topo.dp, "fsdp": topo.fsdp}
    assert flux_views.runs_views(as_pushed) is proven  # a Worker's dict reads the same
    assert flux_views.describe(as_pushed) == topo.describe()


def test_the_cfg_slice_rule_would_cut_an_index_tensor_as_long_as_the_batch():
    """One more reason cfg2 refuses: the slice rule cuts every tensor whose
    first dimension equals the model-call batch, and token_keep's kept index
    list can be that long, which would drop kept tokens with no error."""
    keep = {"target": torch.arange(12), "refs": [None], "background": torch.zeros(1, 12, 4)}
    sliced = cfg_parallel._take_slice(keep, batch=12, world=2, rank=1)
    assert sliced["target"].tolist() == list(range(6, 12))
    assert sliced["background"] is keep["background"]


# --- The refusal --------------------------------------------------------------------------


def _refusal_text(family, topology, request, **kwargs) -> str:
    with pytest.raises(UnsupportedModelError) as caught:
        flux_views.refuse_unproven_views(family, topology, request, **kwargs)
    return str(caught.value)


@pytest.mark.parametrize("family", sorted(flux_views.VIEW_FAMILIES))
@pytest.mark.parametrize("preset", ["cfg2", "dp2", "ring2", "single", "cfg2+fsdp"])
def test_an_unproven_topology_refuses_class_p_with_the_working_ones(family, preset):
    topo = topology_from_preset(preset, 2)
    text = _refusal_text(family, topo, _request(), preset=preset)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS
    assert text.split("] ", 1)[1].startswith(f"{family}: this render's conditioning carries UltraTex")
    assert f"topology preset {preset!r} resolves {topo.describe()}" in text
    assert "Use topology uly2, or uly2+fsdp" in text and "gpus_per_host=1" in text
    assert text.endswith("See docs/TROUBLESHOOTING.md #148.")


@pytest.mark.parametrize("family", ["chroma", "flux3", "qwen_image", None])
def test_families_whose_forward_reads_no_view_input_pass(family):
    flux_views.refuse_unproven_views(family, topology_from_preset("cfg2", 2), _request())


@pytest.mark.parametrize("preset", ["uly2", "uly2+fsdp"])
def test_the_proven_topologies_and_a_plain_render_pass(preset):
    flux_views.refuse_unproven_views("flux2", topology_from_preset(preset, 2), _request())
    flux_views.refuse_unproven_views("flux2", Topology(world=1), _request())
    flux_views.refuse_unproven_views("flux2", topology_from_preset("cfg2", 2), _request(positive=_plain()))


# --- The driver preflight -----------------------------------------------------------------


def _model(preset: str):
    return SimpleNamespace(unet_name="flux-2-klein-base-9b_bf16.safetensors",
                           mesh=SimpleNamespace(topology_preset=preset))


def test_the_driver_refuses_before_dispatch_on_the_header_family(monkeypatch):
    reads: list = []
    monkeypatch.setattr(model_patch_preflight, "_sniffed_family", lambda model: reads.append(model) or "flux2")
    text = None
    with pytest.raises(UnsupportedModelError) as caught:
        model_patch_preflight.preflight_flux_views(_model("cfg2"), _request(), topology_from_preset("cfg2", 2))
    text = str(caught.value)
    assert "topology preset 'cfg2' resolves cfg2" in text and len(reads) == 1


def test_the_driver_reads_no_header_where_nothing_can_refuse(monkeypatch):
    def sniff(_model):
        raise AssertionError("the header was read")

    monkeypatch.setattr(model_patch_preflight, "_sniffed_family", sniff)
    model_patch_preflight.preflight_flux_views(_model("uly2"), _request(), topology_from_preset("uly2", 2))
    model_patch_preflight.preflight_flux_views(
        _model("cfg2"), _request(positive=_plain()), topology_from_preset("cfg2", 2))


def test_an_unreadable_header_passes_to_the_worker_belt(monkeypatch):
    monkeypatch.setattr(model_patch_preflight, "_sniffed_family", lambda _model: None)
    model_patch_preflight.preflight_flux_views(_model("dp2"), _request(), topology_from_preset("dp2", 2))


def test_render_submit_runs_the_preflight_after_the_topology_resolves():
    source = (SRC / "dgx_monarch/nodes/render_submit.py").read_text()
    resolved = source.index("resolved = _resolve_topology_for_latent(")
    call = source.index("model_patch_preflight.preflight_flux_views(model, request, topo)")
    dispatch = source.index("validate_render_topology(model, topo, latent[\"samples\"])")
    assert resolved < call < dispatch


# --- The Worker belt ----------------------------------------------------------------------


def test_every_worker_rank_refuses_on_its_own_topology_and_loaded_family():
    request = _request()
    for topology in ({"cfg": 2}, {"dp": 2}, {"ring": 2, "fsdp": True}, {"ulysses": 4}):
        text = _refusal_text("flux2", topology, request)
        assert "this render's topology resolves" in text
    flux_views.refuse_unproven_views("flux2", {"ulysses": 2, "fsdp": True}, request)
    flux_views.refuse_unproven_views("flux2", {}, request)  # a Fleet job's world-1 Worker


def test_run_sample_reads_the_request_after_the_readiness_exchange():
    """Beside the MOSS-Image bias scan: after the exchange every rank crosses
    alike, before the cfg equalization and both samplers."""
    tree = ast.parse((SRC / "dgx_monarch/actor/sample_protocol.py").read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_sample")
    body = [ast.unparse(statement) for statement in run.body]
    belt = ("refuse_unproven_views(getattr(resident, 'family', None), "
            "getattr(worker, 'topology', {}) or {}, request)")
    assert body.count(belt) == 1
    at = body.index(belt)
    exchange = next(i for i, line in enumerate(body) if "readiness.ready_or_raise()" in line)
    assert exchange < at
    for later in ("equalize_cond_lengths(", "run_ksampler(", "run_custom("):
        assert next(i for i, line in enumerate(body) if later in line) > at, later


# --- Auto ---------------------------------------------------------------------------------


def test_auto_skips_longcats_cfg_row_for_a_view_conditioned_render():
    plain = choose_auto_topology("longcat", "bf16", 1.0, 2, cfg_value=3.5, batch_size=6)
    assert plain.topology.cfg == 2
    viewed = choose_auto_topology("longcat", "bf16", 1.0, 2, cfg_value=3.5, batch_size=6, views=True)
    assert viewed.topology.describe() == "uly2" and flux_views.runs_views(viewed.topology)
    assert ("cfg2 table row 28 would split the batch that holds each UltraTex sample's views; "
            "uly2 is the proven topology for view conditioning") in viewed.reason


def test_auto_keeps_flux2s_ulysses_row_and_other_families_rows():
    viewed = choose_auto_topology("flux2", "bf16", 4.19, 2, cfg_value=1.0, batch_size=6, views=True)
    assert viewed.topology.describe() == "uly2" and "table row 27" in viewed.reason
    # Chroma's diffusion model reads no view input, so its cfg row stands.
    chroma = choose_auto_topology("chroma", "bf16", 1.0, 2, cfg_value=3.5, batch_size=6,
                                  pair_differs=True, views=True)
    assert chroma.topology.cfg == 2


@pytest.mark.parametrize("batch", [6, 3])
def test_auto_refuses_where_no_proven_topology_tiles_the_world(batch):
    # Batch 6 leaves uly2+dp2 on world 4; an odd batch folds dp into ring.
    with pytest.raises(UnsupportedModelError) as caught:
        choose_auto_topology("flux2", "bf16", 4.19, 4, cfg_value=1.0, batch_size=batch, views=True)
    text = str(caught.value)
    assert parse_leading_refusal_tag(text).refusal_class is RefusalClass.PHYSICS
    assert "on world 4 auto resolves" in text and "#148" in text
    assert choose_auto_topology("flux2", "bf16", 4.19, 4, cfg_value=1.0, batch_size=batch).topology


# --- The tag every resolving site reads ---------------------------------------------------


def test_the_sampler_latent_carries_the_views_tag_and_the_wire_drops_it():
    tagged = render_preflight.sampler_latent({"samples": torch.zeros(6, 4, 2, 2)}, "euler", False, True)
    assert tagged[VIEWS] is True
    plain = render_preflight.sampler_latent(tagged, "euler")
    assert VIEWS not in plain  # an inherited tag is dropped first
    assert VIEWS not in render_preflight.latent_without_topology_metadata(tagged)


def test_the_resolver_forwards_the_views_tag_only_when_set(monkeypatch):
    parameters = inspect.signature(common.resolve_topology).parameters
    assert all(name in parameters for _key, name in render_preflight.TOPOLOGY_TAGS)
    assert "views" in inspect.signature(choose_auto_topology).parameters
    calls: list[dict] = []
    monkeypatch.setattr(common, "resolve_topology", lambda *_args, **kwargs: calls.append(kwargs))
    for latent in ({"samples": None, VIEWS: True}, {"samples": None}, {"samples": None, VIEWS: False}):
        common._resolve_topology_for_latent("model", latent, 1.0, 2)
    assert calls == [{"views": True}, {}, {}]


def _capture(monkeypatch) -> list[dict]:
    seen: list[dict] = []

    def run_render(model, request, latent, cfg_value, steps_hint):
        seen.append(latent)
        return {"samples": torch.zeros(6, 4, 2, 2)}

    monkeypatch.setattr(samplers, "run_render", run_render)
    monkeypatch.setattr(samplers, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(samplers, "krea2_ref_preflight_summary", lambda *_args: None)
    return seen


def test_the_samplers_tag_a_view_conditioned_pair(monkeypatch):
    seen = _capture(monkeypatch)
    for positive, tagged in ((_ultratex(), True), (_plain(), False)):
        samplers.DGXMonarchKSampler().sample(
            "model", 0, 4, 1.0, "euler", "simple", positive, _plain(), {"samples": torch.zeros(6, 4, 2, 2)})
        samplers.DGXMonarchKSamplerAdvanced().sample(
            "model", "enable", 0, 4, 1.0, "euler", "simple", positive, _plain(),
            {"samples": torch.zeros(6, 4, 2, 2)}, 0, 10000, "disable")
        assert [latent.get(VIEWS, False) for latent in seen[-2:]] == [tagged, tagged]
    # The Pipeline tags once, before its first-use check and every pushed render.
    monkeypatch.setattr(samplers, "auto_gate_required", lambda *_args: True)
    monkeypatch.setattr(samplers, "_restore_requested_levers", lambda _model: None)
    pipeline_model = SimpleNamespace(mesh=SimpleNamespace(pipeline_depth=1))
    for positive, tagged in ((_ultratex(), True), (_plain(), False)):
        samplers.DGXMonarchKSamplerPipeline().sample(
            pipeline_model, "7", 4, 1.0, "euler", "simple", positive, _plain(),
            {"samples": torch.zeros(6, 4, 2, 2)})
        assert seen[-1].get(VIEWS, False) is tagged
    monkeypatch.setattr(samplers, "_canonical_noise_for_wire", lambda _noise: None)
    for spec, tagged in (({"positive": _ultratex(), "negative": _plain()}, True), ({}, False)):
        samplers.DGXMonarchSamplerCustom().sample(
            SimpleNamespace(seed=0), {"model": "model", "spec": spec, "cfg": 1.0}, object(),
            torch.tensor([1.0, 0.0]), {"samples": torch.zeros(6, 4, 2, 2)})
        assert seen[-1].get(VIEWS, False) is tagged


def test_the_gate_node_proves_the_render_with_the_views_tag(monkeypatch):
    from dgx_monarch.nodes import gate as gate_mod
    from dgx_monarch.nodes.gate_node import DGXMonarchIdentityGate

    proved: list[dict] = []

    def ceremony(model, request, latent, *_args, **_kwargs):
        proved.append(latent)
        return {"latent": latent, "verdict": "PASS"}

    monkeypatch.setattr(gate_mod, "run_identity_ceremony", ceremony)
    monkeypatch.setattr(gate_mod, "conditioning_for_wire", lambda value: value)
    model = SimpleNamespace(mesh=SimpleNamespace(attention="TORCH_FLASH"))
    DGXMonarchIdentityGate().gate(model, _ultratex(), _plain(), {"samples": torch.zeros(6, 4, 2, 2)},
                                  0, 4, 1.0, "euler", "simple")
    (latent,) = proved
    assert latent.get(VIEWS) is True
