"""The Worker applies ComfyUI's own Skip Layer Guidance on the render clone (real ComfyUI, CPU).

``actor/guidance_render.apply_skip_layer_guidance`` must hand the request's
widgets to ComfyUI's ``SkipLayerGuidanceDiTSimple.execute`` unchanged, on the
per-render clone, so the window test, the options copy and the two model calls
are ComfyUI's code. This file checks that on a real ``ModelPatcher``: the
installed calc-cond-batch function, an inactive spec leaving the clone alone,
and a window read from the clone's own (shifted) sampling object. It also reads
ComfyUI's real model configs for the SD3 shift guard, and runs that guard on
every shipped graph that wires the SD3 node whose model files are on disk.

The fixture forces ``--cpu`` because comfy.cli_args parses argv on import, and
skips where ComfyUI is not importable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.actor import guidance_render
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import model_patch_preflight as preflight
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.skip_layer_guidance import DGXMonarchSkipLayerGuidance
from module_location_helpers import from_checkout

REPO = Path(__file__).resolve().parents[1]
COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
TEMPLATE = {"double_layers": "", "single_layers": "22", "start_percent": 0.0, "end_percent": 1.0}


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy():
    """Import real ComfyUI, then leave sys.modules as found."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before = list(sys.path)
    argv_before = sys.argv
    if os.path.isdir(COMFY_DIR) and COMFY_DIR not in sys.path:
        sys.path.insert(0, COMFY_DIR)
    try:
        sys.argv = ["pytest-skip-layer-guidance-comfy", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            model_patcher = pytest.importorskip("comfy.model_patcher")
            try:
                # Gated on comfy.* above: tests/test_release_surfaces.py
                # admits only comfy.* import gates outside the dev extra.
                import comfy_extras.nodes_slg as slg
            except ImportError:
                pytest.skip("this ComfyUI has no Skip Layer Guidance node")
            import comfy.model_sampling
            import comfy.samplers
            import comfy.supported_models
            import folder_paths
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(slg=slg, model_patcher=model_patcher,
                              model_sampling=comfy.model_sampling, samplers=comfy.samplers,
                              supported_models=comfy.supported_models, folder_paths=folder_paths)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _patcher(comfy):
    """A real ModelPatcher over a module that carries a flow sampling object."""
    module = torch.nn.Module()
    module.model_config = None
    module.model_sampling = comfy.model_sampling.ModelSamplingDiscreteFlow(None)
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(module, load_device=cpu, offload_device=cpu)


def _request(value=None, uncond=None) -> dict:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    model = ModelSpec(mesh=mesh, unet_name="model.safetensors")
    patched, = DGXMonarchSkipLayerGuidance().patch(model, **(value or TEMPLATE))
    request = {"model": patched.request_dict()}
    if uncond is not None:
        request["uncond_model"] = uncond
    return request


def _worker(ulysses=2, cfg=1):
    return SimpleNamespace(topology={"ulysses": ulysses, "ring": 1, "cfg": cfg, "dp": 1, "fsdp": False})


HONOURING = SimpleNamespace(family="moss_image", sp_honours_block_replace=True)


def test_the_worker_calls_comfyuis_execute_on_the_render_clone(comfy, monkeypatch):
    calls = []
    original = comfy.slg.SkipLayerGuidanceDiTSimple.execute

    def spy(cls, *args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(comfy.slg.SkipLayerGuidanceDiTSimple, "execute", classmethod(spy))
    render = _patcher(comfy)
    out = guidance_render.apply_skip_layer_guidance(render, _request(), _worker(), HONOURING)

    assert calls == [((render, 0.0, 1.0), {"double_layers": "", "single_layers": "22"})]
    assert out is not render
    assert "sampler_calc_cond_batch_function" in out.model_options
    assert "sampler_calc_cond_batch_function" not in render.model_options


@pytest.mark.parametrize("value", [
    {**TEMPLATE, "single_layers": ""},
    {**TEMPLATE, "single_layers": " , "},
])
def test_an_inactive_spec_leaves_the_clone_without_a_function(comfy, value):
    render = _patcher(comfy)
    assert guidance_render.apply_skip_layer_guidance(render, _request(value), _worker(cfg=2)) is render
    assert "sampler_calc_cond_batch_function" not in render.model_options


def _count_calls(comfy, monkeypatch, patcher, sigma: float) -> list[dict]:
    """Run the installed function once and return what each model call received."""
    seen = []

    def calc_cond_batch(model, conds, x, timestep, model_options):
        seen.append({"conds": conds, "options": model_options})
        return [torch.full_like(x, float(i)) for i in range(len(conds))]

    monkeypatch.setattr(comfy.samplers, "calc_cond_batch", calc_cond_batch)
    x = torch.zeros(1, 4, 2, 2)
    function = patcher.model_options["sampler_calc_cond_batch_function"]
    function({"conds": [[{"c": 1}], [{"c": 0}]], "input": x, "sigma": torch.tensor([sigma]),
              "model": patcher.model, "model_options": patcher.model_options})
    return seen


def test_inside_the_window_the_negative_call_carries_the_identity_skip(comfy, monkeypatch):
    out = guidance_render.apply_skip_layer_guidance(_patcher(comfy), _request(), _worker(), HONOURING)
    seen = _count_calls(comfy, monkeypatch, out, 0.5)
    assert len(seen) == 2
    assert seen[0]["conds"][1] is None and seen[1]["conds"][0] is None
    dit = seen[1]["options"]["transformer_options"]["patches_replace"]["dit"]
    assert set(dit) == {("single_block", 22)}
    probe = {"img": object()}
    assert dit[("single_block", 22)](probe, {}) is probe
    assert "patches_replace" not in seen[0]["options"].get("transformer_options", {})


def test_the_window_comes_from_the_clones_own_sampling_object(comfy, monkeypatch):
    """A shift override moves the window: Worker clone order is shift, then skip."""
    from dgx_monarch.actor.sampling import model_sampling_render_clone

    half = {**TEMPLATE, "start_percent": 0.0, "end_percent": 0.5}
    base = _patcher(comfy)
    stock = guidance_render.apply_skip_layer_guidance(
        model_sampling_render_clone(base, None), _request(half), _worker(), HONOURING)
    shifted = guidance_render.apply_skip_layer_guidance(
        model_sampling_render_clone(base, {"kind": "sd3", "shift": 5.0}), _request(half),
        _worker(), HONOURING)
    # end 0.5 maps to sigma 0.5 at shift 1 and to 5/6 at shift 5, so sigma 0.7
    # sits inside the stock window and outside the shifted one.
    assert len(_count_calls(comfy, monkeypatch, stock, 0.7)) == 2
    assert len(_count_calls(comfy, monkeypatch, shifted, 0.7)) == 1
    assert len(_count_calls(comfy, monkeypatch, shifted, 0.9)) == 2


def test_the_belt_refuses_cfg2_before_comfyuis_node_runs(comfy, monkeypatch):
    def never(*_args, **_kwargs):
        raise AssertionError("ComfyUI's node ran on a refused render")

    monkeypatch.setattr(comfy.slg.SkipLayerGuidanceDiTSimple, "execute", classmethod(never))
    with pytest.raises(UnsupportedModelError, match="cfg-parallel"):
        guidance_render.apply_skip_layer_guidance(
            _patcher(comfy), _request(), _worker(ulysses=1, cfg=2), HONOURING)


# --- the SD3 shift guard on ComfyUI's real configs -------------------------------

def _sd3_model(unet_name: str) -> ModelSpec:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    return ModelSpec(mesh=mesh, unet_name=unet_name).with_model_sampling({"kind": "sd3", "shift": 5.0})


@pytest.mark.parametrize("config_name, refuses", [
    ("MOSSImage", True),
    ("Krea2", True),
    ("QwenImage", True),
    ("WAN21_T2V", False),
    ("Kandinsky5", False),
    ("Kandinsky5Image", False),
])
def test_the_shift_guard_reads_comfyuis_real_config(comfy, monkeypatch, config_name, refuses):
    config_class = getattr(comfy.supported_models, config_name, None)
    if config_class is None:
        pytest.skip(f"this ComfyUI has no {config_name} config")
    config = config_class({})
    monkeypatch.setattr(comfy.folder_paths, "get_full_path", lambda _kind, name: f"/home/user/{name}")
    monkeypatch.setattr("dgx_monarch.render_memory_price.detect_comfy_model_config", lambda _path: config)
    if refuses:
        with pytest.raises(UnsupportedModelError, match=f"{config_name} config"):
            preflight.preflight_shift_node(_sd3_model("model.safetensors"))
    else:
        preflight.preflight_shift_node(_sd3_model("model.safetensors"))


def test_no_shipped_graph_on_disk_wires_the_sd3_node_to_a_multiplier_one_config(comfy):
    """Runs the real guard, real header detection included, on every file present.

    A detection that raises (some fp8 headers do) fails open by design, which
    this also exercises. Files absent here are covered by the name pin in
    tests/test_skip_layer_guidance.py.
    """
    graphs = (sorted((REPO / "example_workflows").glob("*.json"))
              + sorted((REPO / "tests/fixtures/workflows/generated").glob("*.json")))
    checked = 0
    for path in graphs:
        nodes = json.loads(path.read_text()).get("nodes", [])
        if not any(n.get("type") == "DGXMonarchModelSamplingSD3" for n in nodes):
            continue
        for node in nodes:
            if node.get("type") not in ("DGXMonarchUNETLoader", "DGXMonarchUncondUNETLoader"):
                continue
            name = (node.get("dgxm_widgets") or {}).get("unet_name")
            if not name or comfy.folder_paths.get_full_path("diffusion_models", name) is None:
                continue
            preflight.preflight_shift_node(_sd3_model(name))
            checked += 1
    if not checked:
        pytest.skip("no shipped SD3 graph's model file is on this host")
