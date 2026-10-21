"""The Skip Layer Guidance carrier and the SD3 shift guard, without ComfyUI.

The leaf (dgx_monarch/skip_layer_guidance.py) is the one predicate the driver
preflight, the Worker belt and the sweep matrix share, so these tests pin its
wire, its activity rule (ComfyUI's own ``re.findall(r"\\d+")`` parse), its
refusal precedence and both raise sites. tests/test_skip_layer_guidance_comfy.py
runs the Worker application against ComfyUI's real node.
"""
from __future__ import annotations

import ast
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import guidance_render
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from dgx_monarch.nodes import model_patch_preflight as preflight
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.model_sampling import DGXMonarchModelSamplingSD3
from dgx_monarch.nodes.skip_layer_guidance import DGXMonarchSkipLayerGuidance
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.skip_layer_guidance import (
    SkipLayerGuidanceSpec,
    freeze_skip_layer_guidance,
    normalize_skip_layer_guidance,
    skip_layer_guidance_active,
    skip_layer_guidance_refusal_reason,
)
from dgx_monarch.topology import Topology, topology_from_preset

REPO = Path(__file__).resolve().parents[1]
STOCK_DEFAULTS = {"double_layers": "7, 8, 9", "single_layers": "7, 8, 9",
                  "start_percent": 0.0, "end_percent": 1.0}
# The CFG 3 templates' widgets: skip single layer 22 at every step.
TEMPLATE = {"double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 1.0}
OFF = {"double_layers": "", "single_layers": "", "start_percent": 0.0, "end_percent": 1.0}


def _model(**kwargs) -> ModelSpec:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    return ModelSpec(mesh=mesh, unet_name="model.safetensors",
                     options={"weight_dtype": "default"}, **kwargs)


def _skipping(value=None) -> ModelSpec:
    patched, = DGXMonarchSkipLayerGuidance().patch(_model(), **(value or TEMPLATE))
    return patched


def _class_p(exc: BaseException, entry: int) -> str:
    text = str(exc)
    tag = parse_leading_refusal_tag(text)
    assert tag is not None and tag.refusal_class is RefusalClass.PHYSICS, text
    assert f"docs/TROUBLESHOOTING.md #{entry}" in text, text
    return text


# --- the wire -------------------------------------------------------------------

@pytest.mark.parametrize("value", [STOCK_DEFAULTS, TEMPLATE, OFF])
def test_the_wire_accepts_stock_defaults_and_the_template_values(value):
    normalized = normalize_skip_layer_guidance(value)
    assert normalized == value
    assert normalized is not value
    assert isinstance(normalized["start_percent"], float)


def test_the_wire_turns_json_integers_into_floats_and_passes_none():
    assert normalize_skip_layer_guidance(None) is None
    normalized = normalize_skip_layer_guidance({**TEMPLATE, "start_percent": 0, "end_percent": 1})
    assert normalized["start_percent"] == 0.0 and isinstance(normalized["start_percent"], float)
    assert normalized["end_percent"] == 1.0 and isinstance(normalized["end_percent"], float)
    spec = SkipLayerGuidanceSpec(**TEMPLATE)
    assert normalize_skip_layer_guidance(spec) == TEMPLATE


@pytest.mark.parametrize("value", [
    "22",
    [],
    {},
    {k: v for k, v in TEMPLATE.items() if k != "end_percent"},
    {**TEMPLATE, "extra": 1},
    {**TEMPLATE, "start_percent": True},
    {**TEMPLATE, "end_percent": False},
    {**TEMPLATE, "start_percent": -0.001},
    {**TEMPLATE, "end_percent": 1.001},
    {**TEMPLATE, "end_percent": float("nan")},
    {**TEMPLATE, "start_percent": float("inf")},
    {**TEMPLATE, "start_percent": "0.0"},
    {**TEMPLATE, "single_layers": 22},
    {**TEMPLATE, "double_layers": None},
    {**TEMPLATE, "single_layers": "1, " * 86},
])
def test_the_wire_refuses_malformed_values_and_names_the_field(value):
    with pytest.raises(ValueError, match="skip_layer_guidance"):
        normalize_skip_layer_guidance(value)


def test_a_layer_list_of_256_characters_is_the_longest_accepted():
    assert normalize_skip_layer_guidance({**TEMPLATE, "single_layers": "2" * 256})
    with pytest.raises(ValueError, match="skip_layer_guidance single_layers"):
        normalize_skip_layer_guidance({**TEMPLATE, "single_layers": "2" * 257})


@pytest.mark.parametrize("double, single", [
    ("", ""), (",", ""), ("", ","), (" , ", "abc"),
    ("", "22"), ("7, 8, 9", ""), (" 22 , 24", ""), ("x1", "y"),
])
def test_activity_is_comfyuis_own_digit_parse_on_either_list(double, single):
    value = {**TEMPLATE, "double_layers": double, "single_layers": single}
    expected = bool(re.findall(r"\d+", double) or re.findall(r"\d+", single))
    assert skip_layer_guidance_active(value) is expected
    assert (freeze_skip_layer_guidance(value) is None) is (not expected)


def test_activity_of_nothing_is_false():
    assert skip_layer_guidance_active(None) is False
    assert freeze_skip_layer_guidance(None) is None


# --- the node -------------------------------------------------------------------

def test_the_node_schema_is_stocks_widgets_in_stocks_order():
    assert DGXMonarchSkipLayerGuidance.INPUT_TYPES() == {"required": {
        "model": ("DGXM_MODEL",),
        "double_layers": ("STRING", {"default": "7, 8, 9"}),
        "single_layers": ("STRING", {"default": "7, 8, 9"}),
        "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
        "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
    }}
    assert list(DGXMonarchSkipLayerGuidance.INPUT_TYPES()["required"]) == [
        "model", "double_layers", "single_layers", "start_percent", "end_percent"]
    assert DGXMonarchSkipLayerGuidance.RETURN_TYPES == ("DGXM_MODEL",)
    assert DGXMonarchSkipLayerGuidance.RETURN_NAMES == ("model",)
    assert DGXMonarchSkipLayerGuidance.FUNCTION == "patch"
    assert NODE_CLASS_MAPPINGS["DGXMonarchSkipLayerGuidance"] is DGXMonarchSkipLayerGuidance
    assert NODE_DISPLAY_NAME_MAPPINGS["DGXMonarchSkipLayerGuidance"] == (
        "Skip Layer Guidance (DGX Monarch)")


def test_the_node_carries_a_frozen_spec_on_a_new_model_and_the_wire_round_trips():
    model = _model()
    patched = _skipping()
    assert patched is not model
    assert model.skip_layer_guidance is None
    assert patched.skip_layer_guidance == SkipLayerGuidanceSpec(**TEMPLATE)
    assert "skip_layer_guidance" not in patched.options
    request = patched.request_dict()
    assert request["skip_layer_guidance"] == TEMPLATE
    request["skip_layer_guidance"]["single_layers"] = "1"
    assert patched.request_dict()["skip_layer_guidance"] == TEMPLATE
    assert model.request_dict()["skip_layer_guidance"] is None


def test_a_node_with_no_digit_in_either_list_is_byte_for_byte_the_no_node_request():
    plain = _model().request_dict()
    for value in (OFF, {**OFF, "double_layers": " , "}, {**OFF, "start_percent": 0.5}):
        patched = _skipping(value)
        assert patched.skip_layer_guidance is None
        assert json.dumps(patched.request_dict(), sort_keys=True) == json.dumps(plain, sort_keys=True)


def test_a_lone_empty_node_returns_its_input_model_as_stock_does():
    # comfy_extras/nodes_slg.py SkipLayerGuidanceDiTSimple returns its input
    # model unchanged when both lists hold no digit.
    model = _model()
    patched, = DGXMonarchSkipLayerGuidance().patch(model, **OFF)
    assert patched is model
    assert patched.skip_layer_guidance is None


def test_an_empty_node_chained_after_an_active_one_keeps_the_earlier_skip():
    active = _skipping()
    for value in (OFF, {**OFF, "double_layers": " , "}, {**OFF, "start_percent": 0.5}):
        chained, = DGXMonarchSkipLayerGuidance().patch(active, **value)
        assert chained is active
        assert chained.skip_layer_guidance == SkipLayerGuidanceSpec(**TEMPLATE)
        assert chained.request_dict()["skip_layer_guidance"] == TEMPLATE


def test_an_active_node_chained_after_an_active_one_replaces_it():
    # Stock's second node sets the one calc_cond_batch function a model holds
    # (ModelPatcher.set_model_sampler_calc_cond_batch_function), so the later
    # skip is the only one that runs.
    later = {"double_layers": "7", "single_layers": "", "start_percent": 0.01, "end_percent": 0.15}
    chained, = DGXMonarchSkipLayerGuidance().patch(_skipping(), **later)
    assert chained.skip_layer_guidance == SkipLayerGuidanceSpec(**later)
    assert chained.request_dict()["skip_layer_guidance"] == later


def test_the_skip_survives_every_driver_model_transform_before_the_wire():
    """Each driver step rebuilds the ModelSpec with dataclasses.replace, never field by field."""
    from dgx_monarch.nodes.gate_identity import model_for_request, model_with_worker_overrides

    model = _skipping()
    steps = [
        model_with_worker_overrides(model, {"lora_low_rss": False}, handle=SimpleNamespace(world=2)),
        model_for_request(model, {"uncond_model": _model().request_dict()}),
        model_for_request(model, {}),
        model.with_lora("style.safetensors", 0.5),
        model.with_model_sampling({"kind": "sd3", "shift": 5.0}),
        model.with_qwen_image21_cache({"device": "off", "dtype": "default"}),
    ]
    for step in steps:
        assert step.request_dict()["skip_layer_guidance"] == TEMPLATE


def test_the_skip_stays_outside_the_combo_and_store_identity():
    from dgx_monarch.actor.model_store import ModelStore
    from dgx_monarch.mesh_safety import request_combo_key

    plain, skipping = _model().request_dict(), _skipping().request_dict()
    assert request_combo_key(plain) == request_combo_key(skipping)

    def keys(request):
        return ModelStore.make_keys(request["unet_name"], request["options"], request["loras"], "bf16")

    assert keys(plain) == keys(skipping)


# --- the shared predicate -------------------------------------------------------

def _reason(**overrides):
    facts = {"cfg": 1, "sp": 1, "family": "flux3", "family_honours": False, "dual_model": False}
    return skip_layer_guidance_refusal_reason(**{**facts, **overrides})


def test_the_reason_precedence_is_dual_model_then_cfg_then_family():
    dual = _reason(dual_model=True, cfg=2, sp=2)
    assert "two-checkpoint guider" in dual
    cfg = _reason(cfg=2, sp=2)
    assert "cfg-parallel can split only a call that carries both" in cfg
    family = _reason(sp=2)
    assert "flux3's sharded forward could render without the skip and report nothing" in family
    assert len({dual, cfg, family}) == 3


def test_the_cfg_remedy_names_only_the_topologies_that_keep_the_skip_on_the_family():
    honouring = _reason(cfg=2, family="moss_image", family_honours=True)
    assert honouring.endswith("set the Init topology to auto, uly2, ring2, dp2 or single to keep the skip.")
    other = _reason(cfg=2)
    assert other.endswith("set the Init topology to dp2 or single to keep the skip.")
    assert "uly2" not in other and "auto" not in other


def test_no_honouring_family_has_a_cfg_auto_row():
    """The cfg remedy offers auto to an honouring family, so auto must never pick cfg2 there."""
    from dgx_monarch.topology_table import AUTO_TABLE

    cfg_rows = {rule.family for rule in AUTO_TABLE if rule.topology.get("cfg", 1) != 1}
    assert not cfg_rows & preflight.honouring_families()


@pytest.mark.parametrize("facts", [
    {"sp": 2, "family": "moss_image", "family_honours": True},
    {"sp": 1},
    {"sp": 1, "family_honours": False},
])
def test_the_reason_admits_uly2_on_an_honouring_family_single_and_dp2(facts):
    assert _reason(**facts) is None


# --- the driver preflight -------------------------------------------------------

@pytest.fixture
def sniff(monkeypatch):
    """Header sniff and honoring set, both swappable per test."""
    state = {"family": "moss_image", "calls": 0, "honouring": frozenset({"moss_image"})}

    def fake_sniff(_model):
        state["calls"] += 1
        return state["family"]

    monkeypatch.setattr(preflight, "_sniffed_family", fake_sniff)
    monkeypatch.setattr(preflight, "honouring_families", lambda: state["honouring"])
    return state


@pytest.mark.parametrize("preset", ["cfg2", "cfg2+fsdp", "uly2+cfg2"])
def test_the_driver_refuses_an_active_skip_on_every_cfg_topology(sniff, preset):
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_skip_layer_guidance(
            _skipping(), topology_from_preset(preset, 2 if "uly2" not in preset else 4), {})
    text = _class_p(caught.value, 126)
    assert "cfg-parallel can split only a call that carries both" in text
    assert "set the Init topology to auto, uly2, ring2, dp2 or single" in text
    assert sniff["calls"] == 1, "the cfg arm reads the header to name the topologies"


@pytest.mark.parametrize("preset", ["cfg2", "cfg2+fsdp"])
def test_the_driver_offers_a_non_honouring_family_only_dp2_or_single_on_cfg(sniff, preset):
    sniff["family"] = "flux3"
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_skip_layer_guidance(_skipping(), topology_from_preset(preset, 2), {})
    text = _class_p(caught.value, 126)
    assert "set the Init topology to dp2 or single to keep the skip." in text
    assert "uly2" not in text


def test_the_driver_refuses_a_non_honouring_family_under_uly2(sniff):
    sniff["family"] = "flux3"
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_skip_layer_guidance(_skipping(), topology_from_preset("uly2", 2), {})
    assert "flux3's sharded forward could render without the skip" in _class_p(caught.value, 126)


def test_the_driver_refuses_a_two_checkpoint_guider(sniff):
    request = {"uncond_model": _model().request_dict()}
    for topo in (topology_from_preset("single", 1), topology_from_preset("uly2", 2)):
        with pytest.raises(UnsupportedModelError) as caught:
            preflight.preflight_skip_layer_guidance(_skipping(), topo, request)
        assert "two-checkpoint guider" in _class_p(caught.value, 126)


def test_the_driver_reads_a_skip_wired_on_the_uncond_model_too(sniff):
    request = {"uncond_model": _skipping().request_dict()}
    with pytest.raises(UnsupportedModelError, match="two-checkpoint guider"):
        preflight.preflight_skip_layer_guidance(_model(), topology_from_preset("single", 1), request)


@pytest.mark.parametrize("preset, world", [("uly2", 2), ("uly2+fsdp", 2), ("single", 1), ("dp2", 2)])
def test_the_driver_passes_moss_image_on_uly2_and_every_family_on_single_and_dp2(sniff, preset, world):
    preflight.preflight_skip_layer_guidance(_skipping(), topology_from_preset(preset, world), {})
    sniff["family"] = "flux3"
    if preset in ("single", "dp2"):
        preflight.preflight_skip_layer_guidance(_skipping(), topology_from_preset(preset, world), {})


def test_the_driver_does_nothing_without_an_active_skip(sniff):
    sniff["family"] = "flux3"
    for model in (_model(), _skipping(OFF)):
        for preset, world in (("cfg2", 2), ("uly2", 2)):
            preflight.preflight_skip_layer_guidance(model, topology_from_preset(preset, world), {})
    assert sniff["calls"] == 0


def test_a_sniff_error_disables_only_the_family_arm(monkeypatch):
    broken = types.ModuleType("folder_paths")

    def get_full_path(*_args):
        raise OSError("no models directory")

    broken.get_full_path = get_full_path
    monkeypatch.setitem(sys.modules, "folder_paths", broken)
    monkeypatch.setattr(preflight, "honouring_families", lambda: frozenset())
    preflight.preflight_skip_layer_guidance(_skipping(), topology_from_preset("uly2", 2), {})
    with pytest.raises(UnsupportedModelError, match="cfg-parallel"):
        preflight.preflight_skip_layer_guidance(_skipping(), topology_from_preset("cfg2", 2), {})


def test_no_family_honours_block_replacement_without_its_adapter_saying_so():
    from dgx_monarch.adapters import ADAPTERS

    declared = {a.family for a in ADAPTERS if getattr(a, "sp_honours_block_replace", False)}
    assert preflight.honouring_families() == frozenset(declared)
    assert declared <= {"moss_image"}, "a family joins only with its own end-to-end test"


def test_the_driver_runs_the_skip_check_then_the_shift_check_on_both_models(monkeypatch):
    calls = []
    monkeypatch.setattr(preflight, "preflight_skip_layer_guidance",
                        lambda model, topo, request: calls.append(("skip", model)))
    monkeypatch.setattr(preflight, "preflight_shift_node", lambda model: calls.append(("shift", model)))
    cond, uncond = _model(), {"unet_name": "neg.safetensors", "model_sampling": None}
    preflight.preflight_model_patches(cond, Topology(), {"uncond_model": uncond})
    assert calls == [("skip", cond), ("shift", cond), ("shift", uncond)]
    calls.clear()
    preflight.preflight_model_patches(cond, Topology(), {})
    assert calls == [("skip", cond), ("shift", cond)]


def _call_order(path: Path, function: str) -> list[str]:
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function)
    return [ast.unparse(call.func) for call in ast.walk(node) if isinstance(call, ast.Call)]


def test_render_submit_runs_the_patch_preflight_once_right_after_the_h3_topology_check():
    calls = _call_order(REPO / "src/dgx_monarch/nodes/render_submit.py", "_submit_render_guarded")
    index = calls.index("render_preflight.preflight_minimax_h3_topology")
    assert calls[index + 1] == "model_patch_preflight.preflight_model_patches"
    assert calls.count("model_patch_preflight.preflight_model_patches") == 1


def test_fleet_runs_the_shift_check_before_it_encodes_or_dispatches():
    """Fleet KSampler bypasses run_render, so it calls the shift check itself, first."""
    calls = _call_order(REPO / "src/dgx_monarch/nodes/fleet.py", "fleet")
    assert calls.count("model_patch_preflight.preflight_shift_node") == 1
    first = calls.index("model_patch_preflight.preflight_shift_node")
    assert first < calls.index("build_jobs") < calls.index("session.bind")


# --- the Worker belt ------------------------------------------------------------

class _Adapter:
    def __init__(self, family: str, honours: bool = False):
        self.family = family
        if honours:
            self.sp_honours_block_replace = True


def _worker(preset: str, world: int):
    topo = topology_from_preset(preset, world)
    return SimpleNamespace(topology={"ulysses": topo.ulysses, "ring": topo.ring, "cfg": topo.cfg,
                                     "dp": topo.dp, "fsdp": topo.fsdp}, world=world)


def _wire(model: ModelSpec = None, uncond: ModelSpec | None = None) -> dict:
    request = {"model": (model or _skipping()).request_dict()}
    if uncond is not None:
        request["uncond_model"] = uncond.request_dict()
    return request


class _Unused:
    """A patcher or adapter the inactive path must never touch."""

    def __getattr__(self, name):
        raise AssertionError(f"the inactive path read {name}")


def test_the_belt_returns_the_clone_untouched_and_looks_nothing_up_when_inactive(monkeypatch):
    def no_lookup(*_args, **_kwargs):
        raise AssertionError("adapter_for ran on a render without a skip")

    monkeypatch.setattr("dgx_monarch.adapters.adapter_for", no_lookup)
    patcher = _Unused()
    for request in (_wire(_model()), _wire(_skipping(OFF)), {"model": {"unet_name": "m"}}):
        assert guidance_render.apply_skip_layer_guidance(
            patcher, request, _worker("cfg2", 2)) is patcher


@pytest.mark.parametrize("preset, world, adapter, request_fn, fragment", [
    ("cfg2", 2, _Adapter("moss_image", True), lambda: _wire(), "cfg-parallel"),
    ("cfg2+fsdp", 2, _Adapter("moss_image", True), lambda: _wire(), "cfg-parallel"),
    ("uly2", 2, _Adapter("flux3"), lambda: _wire(), "flux3's sharded forward"),
    ("cfg2", 2, _Adapter("flux3"), lambda: _wire(), "to dp2 or single to keep the skip"),
    ("single", 1, _Adapter("moss_image", True),
     lambda: _wire(uncond=_model()), "two-checkpoint guider"),
    ("single", 1, _Adapter("moss_image", True),
     lambda: _wire(_model(), uncond=_skipping()), "two-checkpoint guider"),
])
def test_the_belt_refuses_what_the_driver_refuses(preset, world, adapter, request_fn, fragment):
    with pytest.raises(UnsupportedModelError) as caught:
        guidance_render.apply_skip_layer_guidance(_Unused(), request_fn(), _worker(preset, world), adapter)
    assert fragment in _class_p(caught.value, 126)


def test_the_belt_resolves_the_bound_adapter_itself_on_a_sequence_parallel_render(monkeypatch):
    """run_sample's own local adapter is None on uly2 and ring; the belt never reads it."""
    seen = []

    def adapter_for(model, override):
        seen.append((model, override))
        return _Adapter("krea2")

    monkeypatch.setattr("dgx_monarch.adapters.adapter_for", adapter_for)
    patcher = SimpleNamespace(model=object())
    worker = _worker("uly2", 2)
    worker.store = SimpleNamespace(family_override="krea2")
    with pytest.raises(UnsupportedModelError, match="krea2's sharded forward"):
        guidance_render.apply_skip_layer_guidance(patcher, _wire(), worker)
    assert seen == [(patcher.model, "krea2")]


def test_run_sample_hands_the_belt_no_adapter_right_after_the_shift_clone():
    tree = ast.parse((REPO / "src/dgx_monarch/actor/sample_protocol.py").read_text())
    run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run_sample")
    body = [ast.unparse(stmt) for stmt in ast.walk(run) if isinstance(stmt, ast.Assign)]
    clone = body.index("render_patcher = model_sampling_render_clone(patcher, primary_sampling)")
    assert body[clone + 1] == (
        "render_patcher = guidance_render.apply_skip_layer_guidance(render_patcher, request, worker)")


@pytest.mark.parametrize("preset, world, family, honours, request_fn", [
    ("cfg2", 2, "moss_image", True, lambda: _wire()),
    ("cfg2", 2, "flux3", False, lambda: _wire()),
    ("uly2", 2, "flux3", False, lambda: _wire()),
    ("single", 1, "moss_image", True, lambda: _wire(uncond=_model())),
])
def test_the_driver_and_the_worker_say_the_same_words(monkeypatch, preset, world, family, honours,
                                                     request_fn):
    request = request_fn()
    monkeypatch.setattr(preflight, "_sniffed_family", lambda _model: family)
    monkeypatch.setattr(preflight, "honouring_families",
                        lambda: frozenset({family}) if honours else frozenset())
    uncond = None if "uncond_model" not in request else _model()
    with pytest.raises(UnsupportedModelError) as driver:
        preflight.preflight_skip_layer_guidance(
            _skipping(), topology_from_preset(preset, world),
            {"uncond_model": uncond.request_dict()} if uncond is not None else {})
    with pytest.raises(UnsupportedModelError) as worker:
        guidance_render.apply_skip_layer_guidance(
            _Unused(), request, _worker(preset, world), _Adapter(family, honours))
    assert str(driver.value) == str(worker.value)


# --- the SD3 shift guard --------------------------------------------------------

def _sd3(model: ModelSpec) -> ModelSpec:
    return model.with_model_sampling({"kind": "sd3", "shift": 5.0})


# --- the order of the shift and skip nodes -------------------------------------

WINDOWED = {**TEMPLATE, "start_percent": 0.2}


def test_a_shift_wired_before_a_windowed_skip_passes_in_the_workers_order():
    shifted, = DGXMonarchModelSamplingSD3().patch(_model(), 3.0)
    patched, = DGXMonarchSkipLayerGuidance().patch(shifted, **WINDOWED)
    assert patched.model_sampling is not None
    assert patched.skip_layer_guidance == SkipLayerGuidanceSpec(**WINDOWED)


def test_a_shift_wired_after_a_windowed_skip_refuses_class_p():
    with pytest.raises(UnsupportedModelError) as caught:
        DGXMonarchModelSamplingSD3().patch(_skipping(WINDOWED), 3.0)
    text = _class_p(caught.value, 126)
    assert "step window runs from 0.2 to 1" in text
    assert "Put the shift node before the skip node" in text


@pytest.mark.parametrize("value", [TEMPLATE, STOCK_DEFAULTS, {**OFF, "start_percent": 0.2}],
                         ids=["whole-window", "stock-defaults", "inactive"])
def test_a_shift_after_a_skip_passes_when_the_window_cannot_move(value):
    shifted, = DGXMonarchModelSamplingSD3().patch(_skipping(value), 3.0)
    assert shifted.model_sampling is not None


@pytest.fixture
def configs(monkeypatch):
    """A folder_paths stub and a swappable config detection."""
    state = {"config": None, "calls": [], "path": "/home/user/models/diffusion_models/m.safetensors"}
    stub = types.ModuleType("folder_paths")
    stub.get_full_path = lambda _kind, _name: state["path"]
    monkeypatch.setitem(sys.modules, "folder_paths", stub)

    def detect(path):
        state["calls"].append(path)
        if isinstance(state["config"], Exception):
            raise state["config"]
        return state["config"]

    monkeypatch.setattr("dgx_monarch.render_memory_price.detect_comfy_model_config", detect)
    return state


def _config(name: str, settings):
    return type(name, (), {"sampling_settings": settings})()


def test_the_shift_guard_refuses_a_config_that_samples_at_multiplier_one(configs):
    configs["config"] = _config("MOSSImage", {"multiplier": 1.0, "shift": 5.0})
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_shift_node(_sd3(_model()))
    text = _class_p(caught.value, 128)
    assert "MOSSImage config for model.safetensors samples at multiplier 1.0" in text


@pytest.mark.parametrize("config", [
    _config("WAN21_T2V", {"shift": 8.0}),
    _config("Kandinsky5", {"shift": 10.0}),
    _config("Explicit", {"multiplier": 1000}),
    _config("Odd", {"multiplier": True}),
    _config("Text", {"multiplier": "1.0"}),
    _config("Empty", {}),
    _config("NoneSettings", None),
    _config("NotADict", ["multiplier"]),
    None,
    RuntimeError("Tensor.item() cannot be called on meta tensors"),
])
def test_the_shift_guard_fails_open_on_anything_but_an_explicit_one(configs, config):
    configs["config"] = config
    preflight.preflight_shift_node(_sd3(_model()))
    assert configs["calls"] == [configs["path"]]


def test_the_shift_guard_reads_no_header_without_the_node_or_a_path(configs):
    configs["config"] = _config("MOSSImage", {"multiplier": 1.0})
    preflight.preflight_shift_node(_model())
    preflight.preflight_shift_node({"unet_name": "m.safetensors", "model_sampling": None})
    configs["path"] = None
    preflight.preflight_shift_node(_sd3(_model()))
    assert configs["calls"] == []


def test_the_shift_guard_reads_an_uncond_model_request_too(configs):
    configs["config"] = _config("MOSSImage", {"multiplier": 1.0})
    uncond = _sd3(_model()).request_dict()
    with pytest.raises(UnsupportedModelError, match=r"multiplier 1\.0"):
        preflight.preflight_model_patches(_model(), Topology(), {"uncond_model": uncond})


def _committed_graphs() -> list[Path]:
    return (sorted((REPO / "example_workflows").glob("*.json"))
            + sorted((REPO / "tests/fixtures/workflows/generated").glob("*.json")))


def test_shipped_graphs_wire_the_sd3_node_only_to_wan_and_kandinsky5():
    """The node is noise on a multiplier-1.0 family, so only 1000-multiplier families carry it.

    tests/test_skip_layer_guidance_comfy.py reads the real configs of the files
    that are on disk; this half needs no file.
    """
    wired = {}
    for path in _committed_graphs():
        nodes = json.loads(path.read_text()).get("nodes", [])
        if not any(n.get("type") == "DGXMonarchModelSamplingSD3" for n in nodes):
            continue
        wired[path.name] = [
            (n.get("dgxm_widgets") or {}).get("unet_name") for n in nodes
            if n.get("type") in ("DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader")]
    assert wired, "no committed graph wires the SD3 node; retire this pin with the node"
    for name, files in wired.items():
        assert files, f"{name} wires the SD3 node with no DGX Monarch loader"
        for file in files:
            assert re.search(r"wan|kandinsky", str(file), re.IGNORECASE), (
                f"{name} wires Model Sampling SD3 (DGX Monarch) to {file}; check that its "
                "ComfyUI config does not sample at multiplier 1.0 (docs/TROUBLESHOOTING.md #128)")
