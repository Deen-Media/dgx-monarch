"""Price the latent grid that workers will sample after ComfyUI resizing.

Workers pass downscale_ratio_spacial and downscale_ratio_temporal to
comfy.sample.fix_empty_latent_channels. An empty latent tagged for a different
ratio renders at round(side * tag / ratio) cells. Pricing its original shape
underestimates a render when the tag exceeds the model's ratio.

latent_scale.rendered_latent_shape supplies the corrected shape to auto
megapixels, render memory pricing, MiniMax H3 row counts, and reference/pose
token estimates. Each test uses a mismatched-tag empty latent. Absent or matched
tags and populated latents must retain the original shape.

ComfyUI is stubbed here; tests/test_worker_latent_size_tags.py checks the
arithmetic against real comfy.sample.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import h3_activation, latent_scale, mesh_safety, render_memory_price
from dgx_monarch.nodes import common, render_preflight
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.sampling_contract import LATENT_SIZE_TAG_KEYS

SPATIAL, TEMPORAL = LATENT_SIZE_TAG_KEYS
UNET = "model.safetensors"


def _format(channels: int, spatial: int, temporal: int, dims: int):
    return SimpleNamespace(latent_channels=channels, spacial_downscale_ratio=spatial,
                           temporal_downscale_ratio=temporal, latent_dimensions=dims)


# The latent formats these tests use, as comfy's latent_formats declares them.
CHROMA_RADIANCE = _format(3, 1, 1, 2)   # pixel space
WAN21 = _format(16, 8, 4, 3)
H3_AV = _format(32, 16, 4, 3)           # MiniMaxH3AV


@pytest.fixture(autouse=True)
def _comfy_config(monkeypatch):
    """A comfy whose detection names one config; tests set its latent format."""
    config = SimpleNamespace(latent_format=None, memory_usage_factor=2.0)
    monkeypatch.setattr(render_memory_price, "detect_comfy_model_config",
                        lambda _path: config)
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: f"/models/{UNET}"
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    return config


def _with(tags: dict, samples: torch.Tensor) -> dict:
    return {"samples": samples, **tags}


def _auto_reason(monkeypatch, config, latent_format, family, quant, latent) -> str:
    config.latent_format = latent_format
    monkeypatch.setattr(common, "sniff_checkpoint_for_topology", lambda _path: (family, quant))
    monkeypatch.setattr(common, "_impl_warned", set())
    model = ModelSpec(
        mesh=MeshSpec(handle=SimpleNamespace(world=2), topology_preset="auto",
                      attention="TORCH_FLASH", sync_ulysses=True),
        unet_name=UNET, options={"weight_dtype": "default"})
    _topology, _sage, reason = common._resolve_topology_for_latent(model, latent, 4.0)
    return reason


def test_auto_reads_a_rescaled_empty_latent_at_its_rendered_size(monkeypatch, _comfy_config):
    # EmptyLatentImage at 1024x1024 (8x tag) on a pixel-space model: the
    # worker samples 1024x1024 pixels, not a 128x128 grid at one pixel each.
    reason = _auto_reason(monkeypatch, _comfy_config, CHROMA_RADIANCE, "chroma", "bf16",
                          _with({SPATIAL: 8}, torch.zeros(1, 4, 128, 128)))
    assert "chroma bf16 at 1.0 MP" in reason
    # An LTX latent (32x tag) on an 8x video model renders 4x wider and taller.
    reason = _auto_reason(monkeypatch, _comfy_config, WAN21, "wan", "fp8",
                          _with({SPATIAL: 32, TEMPORAL: 8}, torch.zeros(1, 128, 5, 32, 32)))
    assert "wan fp8 at 1.0 MP" in reason


@pytest.mark.parametrize("tags", [{}, {SPATIAL: 8, TEMPORAL: 4}, "populated"],
                         ids=["absent", "matched", "populated-mismatched"])
def test_auto_megapixels_are_unchanged_without_a_rescale(monkeypatch, _comfy_config, tags):
    if tags == "populated":
        latent = _with({SPATIAL: 32, TEMPORAL: 8}, torch.ones(1, 16, 5, 64, 64))
    else:
        latent = _with(tags, torch.zeros(1, 16, 5, 64, 64))
    reason = _auto_reason(monkeypatch, _comfy_config, WAN21, "wan", "fp8", latent)
    assert "wan fp8 at 0.3 MP" in reason   # 512x512, the grid as built at 8x


def _arm_preflight(monkeypatch, config, latent_format, family):
    config.latent_format = latent_format
    monkeypatch.setattr(render_preflight, "sniff_checkpoint", lambda _path: (family, "bf16"))
    monkeypatch.setattr(render_preflight, "_driver_footprint_preflight_for_request",
                        lambda *_a, **_k: None)
    for memo in ("_LAST_RENDERED_UNET", "_LAST_SUBMITTED_UNET", "_LAST_DRIVER_CHARGED_UNET"):
        monkeypatch.setattr(render_preflight, memo, None)
    monkeypatch.setattr(render_memory_price, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(render_memory_price, "mem_available_bytes", lambda: 2**50)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 2**50)


def _run_preflight(latent: dict, request: dict | None = None) -> None:
    mesh = SimpleNamespace(topology_preset="auto", worker_args={}, handle=None)
    model = SimpleNamespace(unet_name=UNET, mesh=mesh, options={})
    render_preflight.activation_footprint_preflight_for_request(
        model, request if request is not None else {"positive": []}, latent)


def _priced_area(monkeypatch, config, latent_format, latent) -> int:
    _arm_preflight(monkeypatch, config, latent_format, "chroma")
    seen: list = []
    real = render_memory_price.comfy_render_memory_bound

    def record(path, latent_shape):
        found = real(path, latent_shape)
        seen.append(found[1]["area"])
        return found

    monkeypatch.setattr(render_memory_price, "comfy_render_memory_bound", record)
    _run_preflight(latent)
    area, = seen
    return area


def test_render_memory_price_reads_the_rendered_grid(monkeypatch, _comfy_config):
    area = _priced_area(monkeypatch, _comfy_config, CHROMA_RADIANCE,
                        _with({SPATIAL: 8}, torch.zeros(1, 4, 16, 16)))
    assert area == 128 * 128   # 16x16 grid, 128x128 sampled


@pytest.mark.parametrize("tags", [{}, {SPATIAL: 1}, "populated"],
                         ids=["absent", "matched", "populated-mismatched"])
def test_render_memory_price_is_unchanged_without_a_rescale(monkeypatch, _comfy_config, tags):
    samples = torch.ones(1, 3, 16, 16) if tags == "populated" else torch.zeros(1, 3, 16, 16)
    latent = _with({SPATIAL: 8} if tags == "populated" else tags, samples)
    assert _priced_area(monkeypatch, _comfy_config, CHROMA_RADIANCE, latent) == 16 * 16


def _h3_rows(monkeypatch, config, latent) -> h3_activation.h3_rows.H3Rows:
    _arm_preflight(monkeypatch, config, H3_AV, h3_activation.H3_FAMILY)
    monkeypatch.setattr(h3_activation.driver_footprint, "file_size_bytes", lambda _path: 0)
    seen: list = []
    real = h3_activation.estimate_h3_activation

    def record(**kwargs):
        seen.append(kwargs["rows"])
        return real(**kwargs)

    monkeypatch.setattr(h3_activation, "estimate_h3_activation", record)
    _run_preflight(latent)
    rows, = seen
    return rows


def test_h3_rows_count_the_rendered_grid(monkeypatch, _comfy_config):
    # An LTX latent (32x tag) on H3's 16x format: twice as wide and tall.
    rows = _h3_rows(monkeypatch, _comfy_config,
                    _with({SPATIAL: 32}, torch.zeros(1, 128, 5, 24, 40)))
    assert (rows.frame_rows, rows.latent_t) == (24 * 40, 5)
    assert rows.video == 5 * 24 * 40


@pytest.mark.parametrize("tags", [{}, {SPATIAL: 16, TEMPORAL: 4}, "populated"],
                         ids=["absent", "matched", "populated-mismatched"])
def test_h3_rows_are_unchanged_without_a_rescale(monkeypatch, _comfy_config, tags):
    if tags == "populated":
        latent = _with({SPATIAL: 32, TEMPORAL: 8}, torch.ones(1, 32, 5, 48, 80))
    else:
        latent = _with(tags, torch.zeros(1, 32, 5, 48, 80))
    rows = _h3_rows(monkeypatch, _comfy_config, latent)
    assert (rows.frame_rows, rows.latent_t, rows.video) == (24 * 40, 5, 5 * 24 * 40)


def _ref_pose_video_shape(monkeypatch, config, latent) -> tuple:
    _arm_preflight(monkeypatch, config, WAN21, "wan_scail")
    seen: list = []
    monkeypatch.setattr(mesh_safety, "activation_footprint_preflight",
                        lambda *_a, **kwargs: seen.append(kwargs))
    ref = torch.zeros(1, 16, 1, 16, 16)
    _run_preflight(latent, {"positive": [[torch.zeros(1), {"reference_latents": [ref]}]]})
    call, = seen
    assert [tuple(shape) for shape in call["ref_shapes"]] == [(1, 16, 1, 16, 16)]
    return tuple(call["video_shape"])


def test_reference_and_pose_tokens_count_the_rendered_grid(monkeypatch, _comfy_config):
    shape = _ref_pose_video_shape(monkeypatch, _comfy_config,
                                  _with({SPATIAL: 32, TEMPORAL: 8}, torch.zeros(1, 128, 5, 16, 16)))
    assert shape == (1, 16, 10, 64, 64)
    assert mesh_safety._token_count_5d(shape) == 10 * 32 * 32


@pytest.mark.parametrize("tags", [{}, {SPATIAL: 8, TEMPORAL: 4}, "populated"],
                         ids=["absent", "matched", "populated-mismatched"])
def test_reference_and_pose_tokens_are_unchanged_without_a_rescale(
        monkeypatch, _comfy_config, tags):
    if tags == "populated":
        latent = _with({SPATIAL: 32, TEMPORAL: 8}, torch.ones(1, 16, 5, 16, 16))
    else:
        latent = _with(tags, torch.zeros(1, 16, 5, 16, 16))
    assert _ref_pose_video_shape(monkeypatch, _comfy_config, latent) == (1, 16, 5, 16, 16)


def test_the_rendered_shape_fails_open_to_the_shape_as_built(monkeypatch, _comfy_config):
    samples = torch.zeros(1, 4, 16, 16)
    _comfy_config.latent_format = None   # comfy names a config without a format
    assert latent_scale.rendered_latent_shape(UNET, samples, 8) == (1, 4, 16, 16)

    def absent(_path):
        raise ImportError("no comfy on this process")

    monkeypatch.setattr(render_memory_price, "detect_comfy_model_config", absent)
    assert latent_scale.rendered_latent_shape(UNET, samples, 8) == (1, 4, 16, 16)
