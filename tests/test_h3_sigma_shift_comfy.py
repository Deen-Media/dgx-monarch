"""The Workers run ComfyUI's own MiniMax H3 shift node (real ComfyUI 0.44.0 or newer, CPU).

``actor/model_sampling_render.apply_model_sampling`` hands the request's two
widgets to ComfyUI's ``MiniMaxH3SigmaShift.execute`` (ModelSamplingMiniMaxH3) on
the per-render clone. This file checks what dgx-monarch adds against the real
thing: the carrier node copies the stock widgets; the Worker's clone carries
the same sampling object and transformer_options keys a stock graph gets, and
the resident patcher keeps neither; the keys are the ones the stock H3 forward
reads; and ``h3_sigma_shift.is_image_latent`` names the latent stock
``PH3ImageConditioning`` makes and no H3 video latent.

The fixture forces ``--cpu`` because comfy.cli_args parses argv on import, and
skips where ComfyUI is not importable or predates p-h3-image.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import h3_sigma_shift
from dgx_monarch.actor import model_sampling_render, sampling
from dgx_monarch.nodes.model_sampling import DGXMonarchModelSamplingMiniMaxH3
from module_location_helpers import from_checkout

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
H3_WIRE = {"kind": "minimax_h3", "shift_video": 1.0, "shift_audio": 1.0}


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy", "nodes", "node_helpers") or name.startswith(("comfy.", "comfy_"))


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
        sys.argv = ["pytest-h3-sigma-shift-comfy", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            model_patcher = pytest.importorskip("comfy.model_patcher")
            try:
                # Gated on comfy.* above: tests/test_release_surfaces.py
                # admits only comfy.* import gates outside the dev extra.
                from comfy_extras import nodes_minimax_h3
            except ImportError:
                pytest.skip("this ComfyUI has no MiniMax H3 nodes")
            if not hasattr(nodes_minimax_h3, "PH3ImageConditioning"):
                pytest.skip("this ComfyUI predates p-h3-image (ComfyUI 0.44.0 added it)")
            import comfy.ldm.minimax.model
            import comfy.model_sampling
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(
            nodes=nodes_minimax_h3, model_patcher=model_patcher,
            model_sampling=comfy.model_sampling, h3=comfy.ldm.minimax.model)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _resident(comfy):
    """A patcher over a stand-in H3 model: stock's config shifts and stock's sampling class."""
    class Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    module = torch.nn.Module()
    module.model_config = SimpleNamespace(sampling_settings={"shift": 12.0, "audio_shift": 3.0})
    module.model_sampling = Sampling(module.model_config)
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(module, load_device=cpu, offload_device=cpu)


def _sampling_facts(patcher):
    ms = patcher.get_model_object("model_sampling")
    options = patcher.model_options.get("transformer_options", {})
    return (type(ms).__mro__[1:3], ms.shift, ms.audio_shift, ms.audio_scale,
            ms.sigmas.tolist(), {key: options.get(key) for key in
                                 (h3_sigma_shift.VIDEO_KEY, h3_sigma_shift.AUDIO_KEY)})


def test_the_carrier_node_copies_the_stock_widgets(comfy):
    schema = comfy.nodes.MiniMaxH3SigmaShift.define_schema()
    assert schema.node_id == h3_sigma_shift.STOCK_NODE
    # The refusal names this path when the node is missing.
    assert h3_sigma_shift.STOCK_NODE_PATH == f"{comfy.nodes.__name__}.MiniMaxH3SigmaShift"
    assert schema.display_name == h3_sigma_shift.STOCK_DISPLAY
    stock = {item.id: item for item in schema.inputs}
    ours = DGXMonarchModelSamplingMiniMaxH3.INPUT_TYPES()["required"]
    assert list(ours) == list(stock)
    for name, constant in (("shift_video", h3_sigma_shift.SHIFT_VIDEO),
                           ("shift_audio", h3_sigma_shift.SHIFT_AUDIO)):
        item, (kind, spec) = stock[name], ours[name]
        assert kind == "FLOAT"
        assert (spec["default"], spec["min"], spec["max"], spec["step"]) == (
            item.default, item.min, item.max, item.step) == tuple(constant.values()), name


@pytest.mark.parametrize("shifts", [(1.0, 1.0), (12.0, 3.0), (5.5, 0.25)])
def test_the_worker_clone_carries_what_a_stock_graph_gets(comfy, shifts):
    resident = _resident(comfy)
    stock = comfy.nodes.MiniMaxH3SigmaShift.execute(resident.clone(), *shifts)[0]
    wire = {"kind": "minimax_h3", "shift_video": shifts[0], "shift_audio": shifts[1]}
    ours = sampling.model_sampling_render_clone(resident, wire)
    assert ours is not resident
    assert _sampling_facts(ours) == _sampling_facts(stock)
    assert ours.get_model_object("model_sampling").audio_scale == pytest.approx(shifts[0] / shifts[1])
    # The resident patcher the Worker keeps between renders holds neither the
    # shifted object nor the keys, so the next render starts from stock.
    assert resident.get_model_object("model_sampling").shift == 12.0
    assert h3_sigma_shift.VIDEO_KEY not in resident.model_options.get("transformer_options", {})
    plain = sampling.model_sampling_render_clone(resident, None)
    assert plain.get_model_object("model_sampling").shift == 12.0
    assert h3_sigma_shift.VIDEO_KEY not in plain.model_options.get("transformer_options", {})


def test_the_worker_calls_the_stock_node(comfy, monkeypatch):
    calls = []
    original = comfy.nodes.MiniMaxH3SigmaShift.execute

    def spy(cls, *args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(comfy.nodes.MiniMaxH3SigmaShift, "execute", classmethod(spy))
    render = _resident(comfy).clone()
    out = model_sampling_render.apply_model_sampling(render, H3_WIRE)
    assert calls == [((render, 1.0, 1.0), {})]
    assert out is not render
    assert model_sampling_render.h3_stock_node() is comfy.nodes.MiniMaxH3SigmaShift


def test_the_keys_are_the_ones_the_stock_forward_reads(comfy):
    """The stock outer forward reads both keys; the Worker's sharded forward reads the same."""
    forward = comfy.h3.MiniMaxH3Model._forward
    constants = {value for value in forward.__code__.co_consts if isinstance(value, str)}
    assert {h3_sigma_shift.VIDEO_KEY, h3_sigma_shift.AUDIO_KEY} <= constants
    out = comfy.nodes.MiniMaxH3SigmaShift.execute(_resident(comfy), 2.0, 4.0)[0]
    options = out.model_options["transformer_options"]
    assert (options[h3_sigma_shift.VIDEO_KEY], options[h3_sigma_shift.AUDIO_KEY]) == (2.0, 4.0)


class _Clip:
    def tokenize(self, prompt, images=None):
        return {"prompt": prompt, "images": images}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {"tokens": tokens}]]


@pytest.mark.parametrize("size", [(0, 0), (2912, 2912), (3360, 2496), (512, 768)])
def test_the_image_latent_is_the_one_ph3_image_conditioning_makes(comfy, size):
    cond, latent = comfy.nodes.PH3ImageConditioning.execute(_Clip(), "a red kite", *size)[:2]
    width, height = size if size != (0, 0) else (2912, 2912)
    video, audio = latent["samples"].tensors
    assert tuple(video.shape) == (1, 24, h3_sigma_shift.IMAGE_LATENT_T, height // 16, width // 16)
    assert tuple(audio.shape) == (1, 32, 2, h3_sigma_shift.IMAGE_AUDIO_T)
    assert h3_sigma_shift.is_image_latent(latent)
    assert cond[0][1]["tokens"]["prompt"] == "a red kite"


@pytest.mark.parametrize("length", [1, 5, 22, 124])
def test_no_h3_video_latent_reads_as_the_image_latent(comfy, length):
    latent = comfy.nodes.EmptyMiniMaxH3LatentAV.execute(256, 256, length)[0]
    assert latent["samples"].tensors[0].shape[2] >= 2
    assert not h3_sigma_shift.is_image_latent(latent)
