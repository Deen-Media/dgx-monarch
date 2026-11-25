"""MiniMax H3 context windows: the carrier node, its wire value, both refusal sites and the deadline count.

No ComfyUI here: tests/test_h3_context_windows_comfy.py pins the widgets, the
schedule and the row count to ComfyUI's own node and handler.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import h3_activation, h3_rows
from dgx_monarch import h3_context_windows as windows
from dgx_monarch.actor import context_windows_render
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes import model_patch_preflight as preflight
from dgx_monarch.nodes import render_rows
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.h3_context_windows import DGXMonarchMiniMaxH3ContextWindows

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
STOCK = {"context_length": 124, "context_overlap": 39, "fuse_method": "pyramid",
         "freenoise": True, "split_conds_to_windows": False}


def _model(**windows_value) -> ModelSpec:
    mesh = MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                    attention="TORCH_FLASH", sync_ulysses=True)
    model = ModelSpec(mesh=mesh, unet_name="h3.safetensors")
    if not windows_value:
        return model
    return DGXMonarchMiniMaxH3ContextWindows().patch(model, **windows_value)[0]


# 1. The wire value


def test_the_node_schema_is_the_spec_modules_constants():
    required = DGXMonarchMiniMaxH3ContextWindows.INPUT_TYPES()["required"]
    assert list(required) == ["model", *STOCK]
    assert {name: spec[1]["default"] for name, spec in required.items() if name != "model"} == STOCK
    assert required["context_length"][1] == windows.CONTEXT_LENGTH
    assert required["context_overlap"][1] == windows.CONTEXT_OVERLAP
    assert tuple(required["fuse_method"][0]) == windows.FUSE_METHODS
    assert required["fuse_method"][1]["default"] == windows.FUSE_DEFAULT
    assert required["freenoise"][1]["default"] is windows.FREENOISE_DEFAULT
    assert required["split_conds_to_windows"][1]["default"] is windows.SPLIT_CONDS_DEFAULT


def test_a_graph_without_the_node_keeps_its_request_byte_for_byte():
    request = _model().request_dict()
    assert windows.FIELD not in request
    assert set(request) == {"unet_name", "options", "loras", "model_sampling", "skip_layer_guidance"}


def test_the_node_adds_one_validated_entry_to_the_model_request():
    request = _model(**STOCK).request_dict()
    assert request[windows.FIELD] == STOCK
    assert windows.normalize_context_windows(request[windows.FIELD]) == STOCK


@pytest.mark.parametrize("change, message", [
    ({"context_length": 21}, "context_length must be 22"),
    ({"context_length": windows.MAX_RESOLUTION + 1}, "context_length must be 22"),
    ({"context_overlap": 4}, "context_overlap must be 5 or more"),
    ({"context_length": True}, "context_length must be an integer"),
    ({"context_overlap": 39.0}, "context_overlap must be an integer"),
    ({"fuse_method": "average"}, "fuse_method must be one of"),
    ({"freenoise": 1}, "freenoise must be true or false"),
    ({"split_conds_to_windows": "no"}, "split_conds_to_windows must be true or false"),
])
def test_the_wire_value_refuses_what_comfy_validation_would(change, message):
    with pytest.raises(ValueError, match=message):
        windows.normalize_context_windows({**STOCK, **change})


@pytest.mark.parametrize("value", [{**STOCK, "extra": 1}, {k: v for k, v in STOCK.items() if k != "freenoise"},
                                   ["pyramid"]])
def test_the_wire_value_is_exact_key_and_fail_closed(value):
    with pytest.raises(ValueError):
        windows.normalize_context_windows(value)


def test_the_spec_round_trips_and_none_means_no_windows():
    spec = windows.freeze_context_windows(STOCK)
    assert spec is not None and spec.to_wire() == STOCK
    assert windows.normalize_context_windows(spec) == STOCK
    assert windows.freeze_context_windows(None) is None
    assert windows.context_windows_field(None) == {}


# 2. The schedule and the row count


def test_the_15_second_clip_runs_four_windows_of_37_latents():
    latent_t = h3_rows.latent_t_for_frames(362)
    assert windows.latent_window_settings(STOCK) == (37, 12)
    assert [(w[0], len(w)) for w in windows.latent_windows(latent_t, STOCK)] == [
        (0, 37), (25, 37), (50, 37), (70, 37)]


def test_a_clip_no_longer_than_one_window_is_one_call_over_the_whole_clip():
    assert windows.latent_windows(37, STOCK) == [list(range(37))]
    assert windows.latent_windows(2, STOCK) == [[0, 1]]


def test_the_overlap_always_leaves_a_chunk_of_new_frames():
    for length in range(22, 400, 17):
        for overlap in range(5, 600, 17):
            wire = {**STOCK, "context_length": length, "context_overlap": overlap}
            size, kept = windows.latent_window_settings(wire)
            assert size - kept >= windows.CHUNK_LATENTS


def test_every_window_covers_the_clip_and_the_audio_axis_is_split_without_gaps():
    latent_t, audio_t = 107, 603
    found = windows.latent_windows(latent_t, STOCK)
    assert set().union(*map(set, found)) == set(range(latent_t))
    assert windows.audio_span(list(range(latent_t)), latent_t, audio_t) == audio_t


def test_one_window_counts_the_whole_clips_rows():
    latent_t, frame_rows, audio_t, fixed = 37, 1008, 210, 404
    whole = latent_t * frame_rows + 2 * audio_t + fixed
    assert windows.windowed_rows(latent_t=latent_t, frame_rows=frame_rows, audio_t=audio_t,
                                 fixed_rows=fixed, wire=STOCK) == whole


def test_overlapping_windows_count_more_rows_than_the_whole_clip():
    latent_t, frame_rows, audio_t, fixed = 107, 1008, 603, 404
    whole = latent_t * frame_rows + 2 * audio_t + fixed
    counted = windows.windowed_rows(latent_t=latent_t, frame_rows=frame_rows, audio_t=audio_t,
                                    fixed_rows=fixed, wire=STOCK)
    assert whole == 109_466
    assert counted == 4 * fixed + 4 * 37 * frame_rows + 2 * 827
    assert counted > whole


# 3. The deadline reads the windowed count


class _Shape:
    def __init__(self, shape):
        self.shape = shape


def _install_h3_checkpoint(monkeypatch):
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: "/models/h3.safetensors"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_rows, "sniff_checkpoint", lambda _path: ("minimax_h3", "bf16"))


def test_the_sample_deadline_counts_every_windows_rows(monkeypatch):
    _install_h3_checkpoint(monkeypatch)
    request = {"positive": [[_Shape((1, 404, 4096)), {}]]}
    latent = {"samples": _Shape((1, 24, 107, 768 // 16, 1344 // 16))}
    assert render_rows.packed_rows_for_request(_model(), request, latent) == (109_466, "minimax_h3")
    windowed = render_rows.packed_rows_for_request(_model(**STOCK), request, latent)
    assert windowed == (4 * 404 + 4 * 37 * 1008 + 2 * 827, "minimax_h3")


# 4. The refusals


def test_the_family_constant_is_the_activation_preflights():
    assert windows.FAMILY == h3_activation.H3_FAMILY


def test_only_another_readable_family_has_a_reason():
    assert windows.family_refusal_reason("minimax_h3") is None
    assert windows.family_refusal_reason(None) is None
    assert "this checkpoint reads as wan" in windows.family_refusal_reason("wan")


def test_the_driver_refuses_another_family_before_dispatch(monkeypatch):
    monkeypatch.setattr(preflight, "_sniffed_family", lambda _model: "wan")
    with pytest.raises(UnsupportedModelError) as caught:
        preflight.preflight_context_windows(_model(**STOCK), {})
    assert str(caught.value).startswith("[dgxm:P] ")
    assert "#142" in str(caught.value) and "reads as wan" in str(caught.value)


@pytest.mark.parametrize("family", ["minimax_h3", None])
def test_the_driver_passes_h3_and_an_unreadable_header(monkeypatch, family):
    monkeypatch.setattr(preflight, "_sniffed_family", lambda _model: family)
    preflight.preflight_context_windows(_model(**STOCK), {})


def test_the_driver_reads_no_header_without_the_node(monkeypatch):
    def boom(_model):
        raise AssertionError("no header read without the node")

    monkeypatch.setattr(preflight, "_sniffed_family", boom)
    preflight.preflight_context_windows(_model(), {})


def test_the_driver_refuses_windows_on_a_guiders_negative_model():
    request = {"uncond_model": _model(**STOCK).request_dict()}
    with pytest.raises(UnsupportedModelError, match="two-checkpoint guider"):
        preflight.preflight_context_windows(_model(), request)


def test_the_model_patch_preflight_runs_the_windows_check(monkeypatch):
    seen = []
    monkeypatch.setattr(preflight, "preflight_context_windows", lambda model, request: seen.append(model))
    monkeypatch.setattr(preflight, "preflight_skip_layer_guidance", lambda *args: None)
    monkeypatch.setattr(preflight, "preflight_shift_node", lambda *args: None)
    monkeypatch.setattr(preflight, "preflight_lora_quant", lambda *args: None)
    model = _model(**STOCK)
    preflight.preflight_model_patches(model, SimpleNamespace(), {})
    assert seen == [model]


def _request(**windows_value) -> dict:
    return {"model": _model(**windows_value).request_dict()}


def test_the_worker_belt_passes_without_the_node_even_on_an_old_comfy(monkeypatch):
    monkeypatch.setattr(context_windows_render, "stock_node", lambda: None)
    context_windows_render.assert_supported(_request(), "wan")


@pytest.mark.parametrize("family", ["wan", "ltx"])
def test_the_worker_belt_refuses_another_family(monkeypatch, family):
    monkeypatch.setattr(context_windows_render, "stock_node", lambda: object)
    with pytest.raises(UnsupportedModelError, match=f"reads as {family}") as caught:
        context_windows_render.assert_supported(_request(**STOCK), family)
    assert str(caught.value).startswith("[dgxm:P] ")


def test_the_worker_belt_refuses_a_comfy_without_the_stock_node(monkeypatch):
    monkeypatch.setattr(context_windows_render, "stock_node", lambda: None)
    with pytest.raises(UnsupportedModelError, match=r"ComfyUI 0\.44\.0 or newer"):
        context_windows_render.assert_supported(_request(**STOCK), "minimax_h3")


def test_the_worker_belt_refuses_windows_on_a_guiders_negative_model(monkeypatch):
    monkeypatch.setattr(context_windows_render, "stock_node", lambda: object)
    request = {"model": _model().request_dict(), "uncond_model": _model(**STOCK).request_dict()}
    with pytest.raises(UnsupportedModelError, match="two-checkpoint guider"):
        context_windows_render.assert_supported(request, "minimax_h3")


def test_apply_hands_the_widgets_to_the_stock_node(monkeypatch):
    calls = []

    class Node:
        @classmethod
        def execute(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return ("windowed",)

    monkeypatch.setattr(context_windows_render, "stock_node", lambda: Node)
    render = object()
    assert context_windows_render.apply(render, _request(**STOCK)) == "windowed"
    assert calls == [((render, 124, 39, "pyramid", True), {"split_conds_to_windows": False})]
    assert context_windows_render.apply(render, _request()) is render


def test_apply_refuses_rather_than_render_the_whole_clip(monkeypatch):
    monkeypatch.setattr(context_windows_render, "stock_node", lambda: None)
    with pytest.raises(UnsupportedModelError, match="whole clip at once"):
        context_windows_render.apply(object(), _request(**STOCK))


# 5. Where run_sample calls the belt and the carrier


def _run_sample_body() -> ast.FunctionDef:
    tree = ast.parse((SRC / "actor" / "sample_protocol.py").read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_sample")


def _calls(node: ast.AST, attr: str) -> list[ast.Call]:
    return [call for call in ast.walk(node) if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute) and call.func.attr == attr
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "context_windows_render"]


def test_the_worker_belt_sits_inside_the_load_guard():
    body = _run_sample_body()
    guard = next(node for node in ast.walk(body) if isinstance(node, ast.Try)
                 and any(isinstance(h.type, ast.Name) and h.type.id == "RenderCancelledError"
                         for h in node.handlers))
    assert len(_calls(guard, "assert_supported")) == 1
    assert len(_calls(body, "assert_supported")) == 1


def test_the_windows_go_on_after_the_skip_and_before_the_sampler():
    source = (SRC / "actor" / "sample_protocol.py").read_text()
    skip = source.index("guidance_render.apply_skip_layer_guidance(render_patcher")
    apply = source.index("context_windows_render.apply(render_patcher, request)")
    sampler = source.index("samples, out = run_ksampler(")
    assert skip < apply < sampler
    assert len(_calls(_run_sample_body(), "apply")) == 1
