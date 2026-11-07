"""The Workers run ComfyUI's own MiniMax H3 context windows (real ComfyUI 0.44.0 or newer, CPU).

``actor/context_windows_render.apply`` hands the request's widgets to ComfyUI's
``MiniMaxH3ContextWindowsNode.execute`` on the per-render clone, so the
schedule, the per-window cond resizing and the blend are ComfyUI's code. This
file checks the three things dgx-monarch adds against the real thing: the
carrier node copies the stock widgets, the Worker installs the stock handler
on a clone, and the stdlib schedule and row count the sample deadline reads
(``h3_context_windows``) equal what the stock handler runs. It also drives the
stock handler over a packed H3 latent with a recording model call: each
window is one call on a shorter clip whose payload holds no layout for that
clip, so the sharded forward rebuilds the packed sequence per window
(adapters/minimax_h3.py), the contract the hardware compare then checks.

The fixture forces ``--cpu`` because comfy.cli_args parses argv on import, and
skips where ComfyUI is not importable or predates the node.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import h3_context_windows as windows
from dgx_monarch import h3_rows
from dgx_monarch.actor import context_windows_render
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.h3_context_windows import DGXMonarchMiniMaxH3ContextWindows
from module_location_helpers import from_checkout

COMFY_DIR = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
SETTINGS = [
    {"context_length": 124, "context_overlap": 39},
    {"context_length": 22, "context_overlap": 5},
    {"context_length": 73, "context_overlap": 56},
    {"context_length": 260, "context_overlap": 600},
]


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
        sys.argv = ["pytest-h3-context-windows-comfy", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            model_patcher = pytest.importorskip("comfy.model_patcher")
            try:
                # Gated on comfy.* above: tests/test_release_surfaces.py
                # admits only comfy.* import gates outside the dev extra.
                from comfy_extras.nodes_context_windows import MiniMaxH3ContextWindowsNode
                from comfy_extras.nodes_minimax_h3 import video_latent_t
            except ImportError:
                pytest.skip("this ComfyUI has no MiniMax H3 context windows (ComfyUI 0.44.0 added them)")
            import comfy.conds
            import comfy.context_windows
            import comfy.ldm.minimax.model
            import comfy.model_base
            import comfy.patcher_extension
            import comfy.utils
        finally:
            sys.argv = argv_before
        yield SimpleNamespace(
            node=MiniMaxH3ContextWindowsNode, video_latent_t=video_latent_t,
            model_patcher=model_patcher, conds=comfy.conds, cw=comfy.context_windows,
            h3=comfy.ldm.minimax.model, model_base=comfy.model_base,
            patcher_extension=comfy.patcher_extension, utils=comfy.utils)
    finally:
        gone = [name for name, module in list(sys.modules.items())
                if _is_comfy_module(name) or from_checkout(module, COMFY_DIR)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _wire(**changes) -> windows.ContextWindowsWire:
    value = {"context_length": 124, "context_overlap": 39, "fuse_method": "pyramid",
             "freenoise": True, "split_conds_to_windows": False}
    value.update(changes)
    wire = windows.normalize_context_windows(value)
    assert wire is not None
    return wire


def _patcher(comfy):
    module = torch.nn.Module()
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(module, load_device=cpu, offload_device=cpu)


def _request(**changes) -> dict:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    model = ModelSpec(mesh=mesh, unet_name="h3.safetensors")
    patched, = DGXMonarchMiniMaxH3ContextWindows().patch(model, **_wire(**changes))
    return {"model": patched.request_dict()}


def _stock_handler(comfy, wire):
    return comfy.node.execute(_patcher(comfy), wire["context_length"], wire["context_overlap"],
                              wire["fuse_method"], wire["freenoise"])[0].model_options["context_handler"]


def test_the_carrier_node_copies_the_stock_widgets(comfy):
    stock = {item.id: item for item in comfy.node.define_schema().inputs}
    ours = DGXMonarchMiniMaxH3ContextWindows.INPUT_TYPES()["required"]
    assert list(ours) == list(stock)
    for name in ("context_length", "context_overlap"):
        item, (kind, spec) = stock[name], ours[name]
        assert kind == "INT"
        assert (spec["default"], spec["min"], spec.get("max"), spec["step"]) == (
            item.default, item.min, item.max, item.step), name
    assert ours["fuse_method"][0] == list(stock["fuse_method"].options)
    assert ours["fuse_method"][1]["default"] == stock["fuse_method"].default
    for name in ("freenoise", "split_conds_to_windows"):
        assert ours[name][0] == "BOOLEAN"
        assert ours[name][1]["default"] == stock[name].default
        assert ours[name][1]["advanced"] is bool(stock[name].advanced)


def test_the_frame_arithmetic_matches_comfy(comfy):
    assert windows.FRAME_PER_TOKEN == tuple(comfy.h3.FRAME_PER_TOKEN)
    for frames in range(1, 700):
        assert h3_rows.latent_t_for_frames(frames) == comfy.video_latent_t(frames), frames


@pytest.mark.parametrize("setting", SETTINGS)
def test_the_window_schedule_is_the_one_the_stock_handler_runs(comfy, setting):
    wire = _wire(**setting)
    handler = _stock_handler(comfy, wire)
    assert (handler.context_length, handler.context_overlap) == windows.latent_window_settings(wire)
    assert handler.dim == 2 and handler.causal_window_fix is False
    assert handler.context_schedule.name == comfy.cw.ContextSchedules.STATIC_STANDARD
    for latent_t in range(1, 240):
        assert handler.context_schedule.func(latent_t, handler, {}) == windows.latent_windows(latent_t, wire)


@pytest.mark.parametrize("latent_t", [7, 37, 52, 107, 152])
def test_audio_spans_match_the_stock_modality_map(comfy, latent_t):
    wire = _wire()
    model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
    audio_t = h3_rows.audio_t_for_frames(h3_rows.frames_for_latent_t(latent_t))
    shapes = [torch.Size([1, 48, latent_t, 4, 6]), torch.Size([1, 64, 2, audio_t])]
    assert comfy.model_base.MiniMaxH3.context_window_modality_dims(model, shapes, 2) == [2, 3]
    for window in windows.latent_windows(latent_t, wire):
        video, audio = comfy.model_base.MiniMaxH3.map_context_window_to_modalities(model, window, shapes, 2)
        assert video == window
        assert len(audio) == windows.audio_span(window, latent_t, audio_t)


def test_the_worker_applies_the_stock_node_on_the_render_clone(comfy, monkeypatch):
    calls = []
    original = comfy.node.execute

    def spy(cls, *args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(comfy.node, "execute", classmethod(spy))
    render = _patcher(comfy)
    out = context_windows_render.apply(render, _request(fuse_method="flat"))
    assert calls == [((render, 124, 39, "flat", True), {"split_conds_to_windows": False})]
    assert out is not render and "context_handler" not in render.model_options
    # The freenoise wrapper raises on a render with no handler, so nothing may
    # reach the patcher the Worker keeps between renders.
    assert not any(key.startswith("ContextWindows_")
                   for kind in render.wrappers.values() for key in kind)
    handler = out.model_options["context_handler"]
    assert isinstance(handler, comfy.cw.IndexListContextHandler)
    assert handler.fuse_method.name == "flat" and handler.freenoise is True
    keys = comfy.patcher_extension.WrappersMP
    assert "ContextWindows_prepare_sampling" in out.wrappers[keys.PREPARE_SAMPLING]
    assert "ContextWindows_sampler_sample" in out.wrappers[keys.SAMPLER_SAMPLE]
    plain = context_windows_render.apply(render, _request(freenoise=False))
    assert "ContextWindows_sampler_sample" not in plain.wrappers.get(keys.SAMPLER_SAMPLE, {})
    assert context_windows_render.apply(render, {"model": {"unet_name": "h3.safetensors"}}) is render


def _h3_model(comfy):
    model = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
    torch.nn.Module.__init__(model)
    model.latent_format = SimpleNamespace(temporal_downscale_ratio=4)
    return model


def _run_handler(comfy, wire, latent_t, *, keyframes):
    """Drive the stock handler over a packed H3 latent; return the recorded model calls."""
    lat_h, lat_w, text_len = 4, 6, 9
    audio_t = h3_rows.audio_t_for_frames(h3_rows.frames_for_latent_t(latent_t))
    video = torch.randn(1, 48, latent_t, lat_h, lat_w)
    audio = torch.randn(1, 64, 2, audio_t)
    x_in, shapes = comfy.utils.pack_latents([video, audio])
    guides = [{"resolved_frame_index": 0, "latent": torch.randn(1, 48, 5, lat_h, lat_w)}] if keyframes else []
    layout = comfy.h3.PackedLayout(text_len, latent_t, lat_h, lat_w, audio_t, keyframes=guides or None)
    payload = {"keyframes": guides, "layout": layout, "audio_scale": 1.0, "seed": 0}
    cond = [{"model_conds": {"latent_shapes": comfy.conds.CONDConstant(shapes),
                             "minimax_payload": comfy.conds.CONDConstant(payload)}}]
    handler = _stock_handler(comfy, wire)
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    options = {"transformer_options": {"sample_sigmas": sigmas}}
    calls = []

    def calc_cond_batch(model, conds, x, timestep, model_options):
        sub_shapes = conds[0][0]["model_conds"]["latent_shapes"].cond
        calls.append((comfy.utils.unpack_latents(x, sub_shapes), conds[0][0]["model_conds"]["minimax_payload"].cond,
                      model_options["transformer_options"]["context_window"]))
        return [torch.zeros_like(x)]

    out = handler.execute(calc_cond_batch, _h3_model(comfy), [cond], x_in, sigmas[:1], options)
    assert out[0].shape == x_in.shape
    return calls, (text_len, lat_h, lat_w, audio_t)


@pytest.mark.parametrize("keyframes", [False, True])
def test_each_window_is_one_call_whose_payload_layout_the_forward_rebuilds(comfy, keyframes):
    wire, latent_t = _wire(), 107
    calls, (text_len, lat_h, lat_w, audio_t) = _run_handler(comfy, wire, latent_t, keyframes=keyframes)
    expected = windows.latent_windows(latent_t, wire)
    assert len(calls) == len(expected) > 1
    for (video, audio), payload, window in ((c[0], c[1], c[2]) for c in calls):
        index = window.index_list
        assert index in expected
        assert video.shape[2] == len(index)
        assert audio.shape[-1] == windows.audio_span(index, latent_t, audio_t)
        # The adapter keeps a layout only on an exact signature match
        # (adapters/minimax_h3.py), so a kept one must be the rebuilt one.
        layout = payload.get("layout")
        signature = (text_len, len(index), lat_h, lat_w, audio.shape[-1])
        if layout is not None and layout.signature == signature:
            fresh = comfy.h3.PackedLayout(*signature, keyframes=payload.get("keyframes") or None)
            assert layout.segments == fresh.segments
        if keyframes:
            assert all(0 <= kf["resolved_frame_index"] for kf in payload["keyframes"])
            assert len(payload["cond_video_latents"]) == len(payload["keyframes"])


def test_the_deadline_counts_the_rows_every_window_samples(comfy):
    wire, latent_t, fixed = _wire(context_length=73, context_overlap=22), 152, 123
    calls, (_text_len, lat_h, lat_w, audio_t) = _run_handler(comfy, wire, latent_t, keyframes=False)
    frame_rows = h3_rows.frame_rows(lat_h, lat_w)
    per_call = sum(fixed + video.shape[2] * frame_rows + 2 * audio.shape[-1] for (video, audio), _p, _w in calls)
    assert windows.windowed_rows(latent_t=latent_t, frame_rows=frame_rows, audio_t=audio_t,
                                 fixed_rows=fixed, wire=wire) == per_call
