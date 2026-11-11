"""vae_item_batches.py: one image a VAE batch, a batch rule that does not read memory.

The rule is not installed anywhere (it waits on a decision). These tests hold
what it would do: on real ComfyUI's ``VAE`` (CPU, random weights; skipped
where ComfyUI is not importable) stock's latent moves with the free-memory
reading, and the rule's does not.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import vae_item_batches
from module_location_helpers import from_checkout


def test_only_a_batch_of_images_for_an_image_vae_is_split():
    image_vae = SimpleNamespace(latent_dim=2, extra_1d_channel=None)
    assert vae_item_batches.takes_items(image_vae, torch.zeros(6, 8, 8, 3))
    assert not vae_item_batches.takes_items(image_vae, torch.zeros(1, 8, 8, 3))   # one image
    assert not vae_item_batches.takes_items(SimpleNamespace(latent_dim=3), torch.zeros(6, 8, 8, 3))  # video
    assert not vae_item_batches.takes_items(
        SimpleNamespace(latent_dim=2, extra_1d_channel=1), torch.zeros(6, 8, 8, 3))  # audio
    assert not vae_item_batches.takes_items(image_vae, torch.zeros(6, 8))  # not an image batch


def test_each_item_goes_through_the_original_alone_and_in_order():
    seen = []

    def encode(self, tensor, scale=1.0):
        seen.append(tuple(tensor.shape))
        return tensor.sum(dim=(1, 2, 3), keepdim=True) * scale

    wrapped = vae_item_batches.by_item(encode)
    vae = SimpleNamespace(latent_dim=2, extra_1d_channel=None)
    pixels = torch.arange(6 * 4 * 4 * 3, dtype=torch.float32).reshape(6, 4, 4, 3)
    out = wrapped(vae, pixels, scale=2.0)
    assert seen == [(1, 4, 4, 3)] * 6
    assert torch.equal(out, pixels.sum(dim=(1, 2, 3), keepdim=True) * 2.0)
    seen.clear()
    video = SimpleNamespace(latent_dim=3)
    assert wrapped(video, pixels).shape[0] == 6 and seen == [(6, 4, 4, 3)]  # left whole


def test_off_by_default_with_a_trial_switch_and_an_off_switch_that_wins(monkeypatch):
    assert vae_item_batches.ON_BY_DEFAULT is False  # hardware decides; see the module's first lines
    assert vae_item_batches.wanted({}) is False
    assert vae_item_batches.wanted({vae_item_batches.ENABLE_ENV: "1"}) is True
    assert vae_item_batches.wanted({vae_item_batches.ENABLE_ENV: "0"}) is False
    assert vae_item_batches.wanted({vae_item_batches.ENABLE_ENV: "1", vae_item_batches.DISABLE_ENV: "1"}) is False
    monkeypatch.setattr(vae_item_batches, "ON_BY_DEFAULT", True)
    assert vae_item_batches.wanted({}) is True
    assert vae_item_batches.wanted({vae_item_batches.DISABLE_ENV: "1"}) is False


def test_the_driver_hook_installs_nothing_by_default_and_never_raises(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(vae_item_batches, "install", lambda: calls.append("install"))
    monkeypatch.delenv(vae_item_batches.ENABLE_ENV, raising=False)
    monkeypatch.delenv(vae_item_batches.DISABLE_ENV, raising=False)
    assert vae_item_batches.install_driver() == "off" and calls == []
    monkeypatch.setenv(vae_item_batches.ENABLE_ENV, "1")
    with caplog.at_level("INFO"):
        assert vae_item_batches.install_driver() == "installed" and calls == ["install"]
    assert "one image a batch" in caplog.text and vae_item_batches.DISABLE_ENV in caplog.text

    def broken():
        raise ImportError("No module named 'comfy'")

    monkeypatch.setattr(vae_item_batches, "install", broken)
    assert vae_item_batches.install_driver() == "no_comfy"


@pytest.fixture(scope="module")
def comfy_sd():
    """Real ComfyUI's comfy.sd with --cpu, restored afterwards."""
    def is_comfy(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved = {name: module for name, module in sys.modules.items() if is_comfy(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-vae-item-batches", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
            options.enable_args_parsing()
            sd = pytest.importorskip("comfy.sd")
        finally:
            sys.argv = argv_before
        yield sd
    finally:
        if "comfy.sd" in sys.modules:
            vae_item_batches.uninstall()
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


def _toy_vae(sd):
    """ComfyUI's plain AutoencoderKL layout, named by one key; every weight random."""
    vae = sd.VAE(sd={"decoder.conv_in.weight": torch.zeros(512, 4, 3, 3)})
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in vae.first_stage_model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return vae


def _under_readings(vae, call, items):
    """``call()`` with the VAE's free-memory reading faked to fit 1, 2, 3 and all ``items`` a batch."""
    stock_reading = vae.patcher.get_free_memory
    results = {}
    try:
        for batch in (1, 2, 3, items):
            vae.patcher.get_free_memory = lambda device, batch=batch: batch * call.need + call.need // 2
            results[batch] = call().float().cpu()
    finally:
        vae.patcher.get_free_memory = stock_reading
    return results


def test_stocks_encode_follows_the_reading_and_the_rules_does_not(comfy_sd):
    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd)
    pixels = torch.rand(6, 64, 64, 3, generator=torch.Generator().manual_seed(6))

    def encode():
        return vae.encode(pixels)

    encode.need = vae.memory_used_encode(tuple(pixels.movedim(-1, 1).shape), vae.vae_dtype)
    stock = _under_readings(vae, encode, 6)
    # Stock: the latent depends on the split the reading chose (CPU float32 here; on the GPU
    # the V1 probe of 2026-11-11 found 56 to 64 percent of BF16 values apart).
    assert any(not torch.equal(stock[1], stock[batch]) for batch in (2, 3, 6))

    vae_item_batches.install()
    try:
        assert getattr(comfy_sd.VAE.encode, "_dgxm_item_batches", False) is True
        vae_item_batches.install()  # a second install does not stack
        assert not getattr(comfy_sd.VAE.encode.__wrapped__, "_dgxm_item_batches", False)
        ruled = _under_readings(vae, encode, 6)
    finally:
        vae_item_batches.uninstall()
    assert not getattr(comfy_sd.VAE.encode, "_dgxm_item_batches", False)
    # The rule: one tensor whatever memory reads, and it is stock's own one-image-a-batch result.
    for batch in (2, 3, 6):
        assert torch.equal(ruled[1], ruled[batch]), batch
    assert torch.equal(ruled[1], stock[1])
    # It adds nothing of its own: image i's latent is stock's encode of image i alone.
    alone = torch.cat([vae.encode(pixels[index:index + 1]).float().cpu() for index in range(6)], dim=0)
    assert torch.equal(ruled[1], alone)
    # And it is not what a stock box with room for the whole batch gives: that is said in the docs.
    assert not torch.equal(ruled[1], stock[6])


def test_install_imports_nothing_of_comfyui_itself(monkeypatch):
    monkeypatch.delitem(sys.modules, "comfy.sd", raising=False)
    monkeypatch.setenv(vae_item_batches.ENABLE_ENV, "1")
    monkeypatch.delenv(vae_item_batches.DISABLE_ENV, raising=False)
    assert vae_item_batches.install_driver() == "no_comfy"
    assert "comfy.sd" not in sys.modules


def test_under_the_worker_assisted_decode_wrapper_an_image_batch_still_runs_one_item_a_batch(comfy_sd, monkeypatch):
    """vae_decode_route wraps VAE.decode outside this rule once the Init node arms it.

    For an image VAE it declines the plain decode entry and calls on, so the
    rule's result is unchanged: one tensor whatever memory reads.
    """
    import comfy.utils

    from dgx_monarch import vae_decode_route

    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd)
    latents = torch.randn(4, 4, 8, 8, generator=torch.Generator().manual_seed(4))

    def decode():
        return vae.decode(latents)

    decode.need = vae.memory_used_decode(tuple(latents.shape), vae.vae_dtype)
    stock = {name: getattr(comfy_sd.VAE, name) for name in ("decode", "decode_tiled")}
    stock_tiled = comfy.utils.tiled_scale_multidim
    vae_item_batches.install()
    try:
        alone = _under_readings(vae, decode, 4)
        vae_decode_route.install()
        monkeypatch.setattr(vae_decode_route, "_ARMED", (lambda: None, "no mesh"))
        assert getattr(comfy_sd.VAE.decode, "_dgxm_vae_decode_route", False) is True
        assert getattr(comfy_sd.VAE.decode.__wrapped__, "__wrapped__", None) is stock["decode"]
        under_route = _under_readings(vae, decode, 4)
    finally:
        for name, function in stock.items():
            setattr(comfy_sd.VAE, name, function)
        comfy.utils.tiled_scale_multidim = stock_tiled
    for batch in (1, 2, 3, 4):
        assert torch.equal(under_route[batch], alone[1]), batch


def test_the_rules_decode_is_one_tensor_whatever_memory_reads(comfy_sd):
    torch.set_num_threads(2)
    vae = _toy_vae(comfy_sd)
    latents = torch.randn(4, 4, 8, 8, generator=torch.Generator().manual_seed(4))

    def decode():
        return vae.decode(latents)

    decode.need = vae.memory_used_decode(tuple(latents.shape), vae.vae_dtype)
    vae_item_batches.install()
    try:
        ruled = _under_readings(vae, decode, 4)
    finally:
        vae_item_batches.uninstall()
    stock_one = _under_readings(vae, decode, 4)[1]
    assert ruled[1].shape[0] == 4
    for batch in (2, 3, 4):
        assert torch.equal(ruled[1], ruled[batch]), batch
    assert torch.equal(ruled[1], stock_one)
