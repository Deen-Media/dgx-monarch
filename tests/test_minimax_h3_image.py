"""MiniMax-H3-Image adapter pieces that need no ComfyUI: the modulation plan, the grid
check, the input refusals and the driver's cfg/dp preflight.

The forward against stock lives in tests/test_minimax_h3_image_comfy.py.
"""
from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import detect as detect_mod
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.minimax_h3_image import (
    FAMILY,
    MINIMAX_H3_IMAGE_BATCH_CAPPED_MESSAGE,
    _reject_unsupported_image_features,
)
from dgx_monarch.adapters.minimax_h3_packing import (
    H3_IMAGE_SEGMENT_KINDS,
    H3_SEG_MODALITY,
    assert_image_grid,
    h3_image_modulation_plan,
)
from dgx_monarch.nodes.render_preflight import preflight_minimax_h3_topology
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag
from dgx_monarch.topology import Topology
from preflight_fails_open_helpers import install_fake_folder_paths

VISUAL = 0.999  # ComfyUI's VISUAL_COND_TIMESTEP; seam 52 holds it


def _layout(*spans):
    segments, offset = [], 0
    for kind, rows in spans:
        segments.append((offset, offset + rows, kind))
        offset += rows
    return SimpleNamespace(segments=segments, seq_len=offset)


def _stock_mod_segments(layout, t, tags=None):
    """A transcription of stock MiniMaxH3ImageModel._forward's seg_t and mod_segments."""
    seg_t = {"text": t, "video": t, "ref_img": max(t, VISUAL)}
    unique_t = sorted({seg_t[k] for _, _, k in layout.segments})
    t_row = {v: i for i, v in enumerate(unique_t)}
    rows = []
    for a, b, kind in layout.segments:
        base = t_row[seg_t[kind]] * 3
        for i in range(a, b):
            rows.append(base + (int(tags[i - a]) if kind == "text" and tags is not None
                                else H3_SEG_MODALITY[kind]))
    return unique_t, rows, t_row[t]


@pytest.mark.parametrize("t", [0.0, 0.25, 0.999, 1.0])
@pytest.mark.parametrize("refs", [0, 1, 3])
def test_the_plan_numbers_rows_as_stock(t, refs):
    layout = _layout(("text", 5), *[("ref_img", 4)] * refs, ("video", 6))
    unique_t, rows, target = h3_image_modulation_plan(layout, {}, t, VISUAL)
    stock_t, stock_rows, stock_target = _stock_mod_segments(layout, t)
    assert unique_t == stock_t and rows.tolist() == stock_rows and target == stock_target


def test_a_reference_takes_the_constant_never_the_payloads_augmentation():
    layout = _layout(("text", 2), ("ref_img", 2), ("video", 2))
    plain = h3_image_modulation_plan(layout, {}, 0.3, VISUAL)
    augmented = h3_image_modulation_plan(layout, {"visual_cond_noise_aug": 0.5}, 0.3, VISUAL)
    assert plain[0] == augmented[0] == [0.3, VISUAL]
    assert plain[1].tolist() == augmented[1].tolist() == [1, 1, 3, 3, 0, 0]


def test_presentation_tags_split_the_text_rows_as_stock():
    layout = _layout(("text", 6), ("video", 3))
    tags = torch.tensor([[1, 1, 0, 0, 1, 1]])
    _, rows, _ = h3_image_modulation_plan(layout, {"text_token_tags": tags}, 0.5, VISUAL)
    assert rows.tolist() == _stock_mod_segments(layout, 0.5, tags[0])[1]


def test_no_audio_time_joins_the_set():
    layout = _layout(("text", 2), ("video", 2))
    unique_t, rows, target = h3_image_modulation_plan(layout, {}, 0.4, VISUAL)
    assert unique_t == [0.4] and target == 0 and rows.tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize("kind", ["audio", "cond", "cond_audio", "ref_audio", "brand_new"])
def test_any_other_segment_kind_refuses_class_p(kind):
    layout = _layout(("text", 2), (kind, 2), ("video", 2))
    with pytest.raises(UnsupportedModelError) as caught:
        h3_image_modulation_plan(layout, {}, 0.5, VISUAL)
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "has not vetted" in str(caught.value)
    assert {"text", "ref_img", "video"} == H3_IMAGE_SEGMENT_KINDS


def _head(blocks, dim=96):
    weight = SimpleNamespace(shape=(dim * blocks, 8))
    return SimpleNamespace(video_out=SimpleNamespace(weight=weight), video_dim=dim)


@pytest.mark.parametrize("blocks, points, refuses", [
    (6, 7, False), (1, 7, False), (1, 2, False), (6, 5, True), (6, 8, True), (2, 2, True)])
def test_the_grid_check_refuses_only_where_stock_fails(blocks, points, refuses):
    grid = [1.0 - k / max(points - 1, 1) for k in range(points)]
    if not refuses:
        assert assert_image_grid(_head(blocks), grid) is None
        return
    with pytest.raises(UnsupportedModelError) as caught:
        assert_image_grid(_head(blocks), grid)
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert f"{blocks} blocks" in str(caught.value) and f"{points} points" in str(caught.value)


def test_a_batch_above_one_refuses_class_p():
    with pytest.raises(UnsupportedModelError) as caught:
        _reject_unsupported_image_features(torch.zeros(2, 24, 1, 4, 4), {}, {})
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "batch of 2" in str(caught.value)


@pytest.mark.parametrize("name", ["attention_mask", "attn_mask"])
def test_an_attention_mask_refuses_class_p(name):
    with pytest.raises(UnsupportedModelError, match=name):
        _reject_unsupported_image_features(torch.zeros(1, 24, 1, 4, 4), {}, {name: torch.ones(1)})


def test_a_dit_block_replacement_refuses_class_p_and_a_denoise_mask_does_not():
    x = torch.zeros(1, 24, 1, 4, 4)
    with pytest.raises(UnsupportedModelError, match="block replacement"):
        _reject_unsupported_image_features(
            x, {"patches_replace": {"dit": {("double_block", 0): object()}}}, {})
    # Stock's image forward ignores a mask (seam 53); so does this one.
    assert _reject_unsupported_image_features(x, {}, {"denoise_mask": torch.zeros(1)}) is None


def _sniffed(monkeypatch, family):
    install_fake_folder_paths(monkeypatch, "/models/minimax-h3-image.safetensors")
    monkeypatch.setattr(detect_mod, "sniff_checkpoint", lambda path: (family, "bf16"))


@pytest.mark.parametrize("topo", [Topology(cfg=2, world=2), Topology(dp=2, world=2)])
def test_the_driver_refuses_cfg2_and_dp2_with_the_image_sentence(monkeypatch, topo):
    _sniffed(monkeypatch, FAMILY)
    with pytest.raises(UnsupportedModelError) as caught:
        preflight_minimax_h3_topology(types.SimpleNamespace(unet_name="m.safetensors"), topo)
    assert str(caught.value) == MINIMAX_H3_IMAGE_BATCH_CAPPED_MESSAGE
    assert "minimax_h3_image:" in str(caught.value) and "audio" not in str(caught.value)


def test_the_driver_lets_uly2_through_for_the_image_family(monkeypatch):
    _sniffed(monkeypatch, FAMILY)
    preflight_minimax_h3_topology(types.SimpleNamespace(unet_name="m.safetensors"),
                                  Topology(ulysses=2, world=2))


def test_the_h3_sampling_node_refuses_this_family_naming_the_grid():
    from dgx_monarch import h3_sigma_shift

    reason = h3_sigma_shift.family_refusal_reason(FAMILY)
    assert reason is not None and "pdd_sigmas grid" in reason and "does not read" in reason
    assert h3_sigma_shift.family_refusal_reason("minimax_h3") is None


def test_the_h3_context_windows_refuse_this_family():
    from dgx_monarch import h3_context_windows

    assert h3_context_windows.family_refusal_reason(FAMILY) is not None
    assert h3_context_windows.family_refusal_reason("minimax_h3") is None


def test_the_worker_sampling_guard_refuses_class_p_on_this_family():
    from dgx_monarch.actor import model_sampling_render

    request = {"model": {"model_sampling": {"kind": "minimax_h3", "shift_video": 1.0,
                                            "shift_audio": 1.0}}}
    with pytest.raises(UnsupportedModelError) as caught:
        model_sampling_render.assert_supported(request, FAMILY)
    assert parse_leading_refusal_tag(str(caught.value)).refusal_class is RefusalClass.PHYSICS
    assert "pdd_sigmas grid" in str(caught.value)
