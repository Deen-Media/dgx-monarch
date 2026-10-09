"""Typed refusals for unsafe parallel routes.

Covers PixelDiT quant-under-SP and CFG exclusions, the pixel-space pad-mask
refusal, and FSDP load-transient capacity checks. These guards prevent wrong
results, stock mask-upscaling errors, and cold-load memory exhaustion.
"""
from __future__ import annotations

import pytest

from dgx_monarch.adapters import cfg_parallel
from dgx_monarch.adapters.base import UnsupportedModelError


class _Adapter:
    family = "probe"


def test_sp_quant_default_is_orthogonal():
    cfg_parallel.assert_sp_quant_supported(_Adapter(), 2, "mxfp8")


def test_sp_quant_declared_set_refuses_unvalidated():
    adapter = _Adapter()
    adapter.sp_validated_quants = frozenset({"bf16"})
    adapter.sp_quant_refusal_reason = "probe measured wrong"
    cfg_parallel.assert_sp_quant_supported(adapter, 2, "bf16")
    cfg_parallel.assert_sp_quant_supported(adapter, 1, "mxfp8")  # sp=1 exempt
    with pytest.raises(UnsupportedModelError, match="probe measured wrong"):
        cfg_parallel.assert_sp_quant_supported(adapter, 2, "mxfp8")


def test_pixeldit_declares_both_measured_exclusions():
    from dgx_monarch.adapters.pixeldit_comfy import PixelDiTAdapter

    assert PixelDiTAdapter.cfg_parallel_supported is False
    assert PixelDiTAdapter.sp_validated_quants == frozenset({"bf16"})
    assert "0.206" in PixelDiTAdapter.cfg_parallel_refusal_reason
    assert "0.629" in PixelDiTAdapter.sp_quant_refusal_reason
    with pytest.raises(UnsupportedModelError, match=r"0\.206"):
        cfg_parallel.assert_cfg_parallel_supported(PixelDiTAdapter, 2)
    with pytest.raises(UnsupportedModelError, match=r"0\.629"):
        cfg_parallel.assert_sp_quant_supported(PixelDiTAdapter, 2, "mxfp8")


def test_cfg_refusal_keeps_the_architectural_default_text():
    adapter = _Adapter()
    adapter.cfg_parallel_supported = False
    with pytest.raises(UnsupportedModelError, match="separate calls"):
        cfg_parallel.assert_cfg_parallel_supported(adapter, 2)


def test_pixel_space_pad_mask_refuses_with_the_remedy():
    with pytest.raises(UnsupportedModelError) as excinfo:
        cfg_parallel.refuse_pixel_space_pad_mask("chroma")
    message = str(excinfo.value)
    assert "pixel-space" in message
    assert "equal-length prompts" in message
    assert "Zero Out" in message


def test_equalizer_routes_pixel_space_to_the_refusal(monkeypatch):
    import torch

    from dgx_monarch.actor import sampling

    class _ChromaLike:
        family = "chroma"
        cfg_cond_padding = "pad+mask"

    cond = [[torch.zeros(1, 20, 8), {}]]
    uncond = [[torch.zeros(1, 8, 8), {}]]
    pixel_latent = torch.zeros(1, 3, 64, 64)
    with pytest.raises(UnsupportedModelError, match="pixel-space"):
        sampling.equalize_cond_lengths(_ChromaLike(), cond, uncond, pixel_latent)
    # The 16-channel latent path still equalizes.
    latent = torch.zeros(1, 16, 64, 64)
    _pos, neg = sampling.equalize_cond_lengths(_ChromaLike(), cond, uncond, latent)
    assert neg[0][0].shape[1] == 24  # 20 text + 1024 image keys align up to 1048 (a multiple of 8): 24 text rows
    assert "attention_mask" in neg[0][1]


def test_fsdp_load_transient_pricing(tmp_path, monkeypatch):
    from dgx_monarch.adapters import fsdp
    from dgx_monarch.mesh_safety import StockLoadCapacityError
    from dgx_monarch.refusal import RefusalClass, parse_refusal_tag

    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"x" * 1000)
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: True)
    bf16_required = fsdp.fsdp_required_bytes(1000, 2, "bf16")
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: bf16_required)
    fsdp.fsdp_load_capacity_check(
        str(checkpoint), "model.safetensors", {}, world=2,
        checkpoint_kind="bf16")
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: bf16_required - 1)
    with pytest.raises(StockLoadCapacityError) as excinfo:
        fsdp.fsdp_load_capacity_check(
            str(checkpoint), "model.safetensors", {}, world=2,
            checkpoint_kind="bf16")
    tag = parse_refusal_tag(str(excinfo.value))
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY

    # An unreadable header (no kind) keeps the conservative direct-wrap price.
    unknown_required = fsdp.fsdp_required_bytes(1000, 2)
    assert unknown_required > bf16_required
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: unknown_required)
    fsdp.fsdp_load_capacity_check(str(checkpoint), "model.safetensors", {}, world=2)
    monkeypatch.setattr(fsdp, "mem_available_bytes", lambda: unknown_required - 1)
    with pytest.raises(StockLoadCapacityError) as excinfo:
        fsdp.fsdp_load_capacity_check(str(checkpoint), "model.safetensors", {}, world=2)
    tag = parse_refusal_tag(str(excinfo.value))
    assert tag is not None and tag.refusal_class is RefusalClass.CAPACITY

    # dtype casts and discrete devices stay excluded.
    fsdp.fsdp_load_capacity_check(str(checkpoint), "m", {"dtype": "bf16"})
    monkeypatch.setattr(fsdp, "gpu_is_integrated", lambda: False)
    fsdp.fsdp_load_capacity_check(str(checkpoint), "m", {})
