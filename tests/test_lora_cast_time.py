"""A LoRA bakes to the weight stock ComfyUI's default loader computes at cast time (#145).

Stock ComfyUI's DynamicVRAM applies a LoRA when a layer casts its weight: it
dequantizes to the forward's compute dtype, patches in that dtype, and
requantizes a quantized weight with stochastic rounding seeded by the module
name. ComfyUI's classic patcher, which a Worker on stock residency loads
through, bakes in the LoRA compute dtype with fp32 intermediates and a seed
from the full key, which on an INT8 ConvRot weight is another draw of the
requantization noise. These tests run real ComfyUI on CPU (skipped where it is
not importable) on a small synthetic INT8 ConvRot weight and a BF16 weight,
and compare with comfy.ops' own post_cast through the canary's helpers.
"""
from __future__ import annotations

import os
import sys

import pytest

from module_location_helpers import from_checkout as module_from_checkout

KEYS = ("diffusion_model.q.weight", "diffusion_model.p.weight")


@pytest.fixture(scope="module")
def comfy_cpu():
    """Real ComfyUI with --cpu, as the bake-equivalence test imports it; restored afterwards."""
    def is_comfy_module(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved = {name: module for name, module in sys.modules.items() if is_comfy_module(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path, original_argv = list(sys.path), sys.argv
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR", "../ComfyUI"))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "canary"))
    try:
        sys.argv = ["pytest-lora-cast-time", "--cpu"]
        options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
        options.enable_args_parsing()
        pytest.importorskip("comfy.lora")
        pytest.importorskip("comfy.quant_ops")
        import bake_equivalence_canary as canary

        yield canary
    finally:
        sys.argv = original_argv
        from dgx_monarch.actor import lora_cast_time

        if "comfy.model_patcher" in sys.modules:
            lora_cast_time.uninstall()
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy_module(name) or module_from_checkout(module, comfy_dir)
                or name == "bake_equivalence_canary"]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _patched(canary, strength=0.8):
    import comfy.model_patcher
    import torch

    root, patches = canary.tiny_cast_time_model()
    cpu = torch.device("cpu")
    patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
    patcher.add_patches(patches, strength)
    return root, patcher, cpu


@pytest.mark.parametrize("key", KEYS, ids=["int8_convrot", "bf16"])
def test_the_bake_equals_stocks_cast_and_the_classic_bake_does_not(comfy_cpu, key):
    from dgx_monarch.actor import lora_cast_time

    root, patcher, cpu = _patched(comfy_cpu)
    stock = comfy_cpu.stock_cast_time_weight(root, patcher, key)
    classic = patcher.patch_weight_to_device(key, device_to=cpu, return_weight=True)
    ours = lora_cast_time.cast_time_weight(patcher, key, cpu)
    assert ours is not None
    assert comfy_cpu.same_weight(ours, stock)
    # The classic recipe draws other bits: the gap this module closes.
    assert not comfy_cpu.same_weight(classic, stock)


def test_int8_convrot_keeps_its_layout_and_stays_quantized(comfy_cpu):
    from comfy.quant_ops import QuantizedTensor

    from dgx_monarch.actor import lora_cast_time

    root, patcher, cpu = _patched(comfy_cpu)
    pristine = root.diffusion_model.q.weight
    ours = lora_cast_time.cast_time_weight(patcher, KEYS[0], cpu)
    assert isinstance(ours, QuantizedTensor)
    assert ours._layout_cls == pristine._layout_cls
    assert ours._params.convrot and ours._params.convrot_groupsize == 256
    assert ours.dtype == pristine.dtype and tuple(ours.shape) == tuple(pristine.shape)
    assert root.diffusion_model.q.weight is pristine  # never written in place


def test_the_installed_patcher_bakes_stocks_weight_and_unpatches_to_the_file(comfy_cpu):
    import comfy.utils

    from dgx_monarch.actor import lora_cast_time

    root, patcher, cpu = _patched(comfy_cpu)
    pristine = {key: comfy.utils.get_attr(root, key) for key in KEYS}
    stock = {key: comfy_cpu.stock_cast_time_weight(root, patcher, key) for key in KEYS}
    assert lora_cast_time.install() is True
    try:
        assert lora_cast_time.install() is False  # idempotent
        patcher.load(cpu, full_load=True)
        for key in KEYS:
            assert comfy_cpu.same_weight(comfy.utils.get_attr(root, key), stock[key]), key
        patcher.unpatch_model(cpu)
        for key in KEYS:
            assert comfy_cpu.same_weight(comfy.utils.get_attr(root, key), pristine[key]), key
    finally:
        assert lora_cast_time.uninstall() is True
    assert lora_cast_time.uninstall() is False


def test_the_lazy_swaps_bake_and_verify_agree_with_stock(comfy_cpu):
    import comfy.utils

    from dgx_monarch.actor import lora_cast_time, store_bake

    root, patcher, _cpu = _patched(comfy_cpu)
    stock = {key: comfy_cpu.stock_cast_time_weight(root, patcher, key) for key in KEYS}
    lora_cast_time.install()
    try:
        store = type("Store", (), {"verify_failures": 0, "_bake_key": staticmethod(store_bake.bake_key)})()
        for key in KEYS:
            store_bake.verify_key(store, patcher, key)  # bakes, then compares with the reference
            assert comfy_cpu.same_weight(comfy.utils.get_attr(root, key), stock[key]), key
        assert store.verify_failures == 0
    finally:
        lora_cast_time.uninstall()


def test_keys_stock_bakes_classically_keep_the_classic_bake(comfy_cpu):
    import comfy.lora
    import comfy.model_patcher
    import comfy.ops
    import torch

    from dgx_monarch.actor import lora_cast_time

    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tiny = ops.Linear(16, 16, bias=False)       # 512 bytes: stock force-loads it
            self.wide = ops.Linear(256, 64, bias=False)       # stored fp32, computed in bf16

        def get_dtype_inference(self):
            return torch.bfloat16

    root = _Root()
    root.tiny.weight = torch.nn.Parameter(torch.randn(16, 16).to(torch.bfloat16), requires_grad=False)
    root.wide.weight = torch.nn.Parameter(torch.randn(64, 256), requires_grad=False)
    lora, key_map = {}, {}
    for name, (o, i) in (("tiny", (16, 16)), ("wide", (64, 256))):
        lora[f"{name}.lora_up.weight"] = torch.randn(o, 2) * 0.01
        lora[f"{name}.lora_down.weight"] = torch.randn(2, i) * 0.01
        key_map[name] = f"{name}.weight"
    cpu = torch.device("cpu")
    patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
    patcher.add_patches(comfy.lora.load_lora(lora, key_map), 1.0)
    for key in ("tiny.weight", "wide.weight"):
        assert lora_cast_time.cast_time_weight(patcher, key, cpu) is None, key
    classic = patcher.patch_weight_to_device("wide.weight", device_to=cpu, return_weight=True)
    lora_cast_time.install()
    try:
        via = patcher.patch_weight_to_device("wide.weight", device_to=cpu, return_weight=True)
    finally:
        lora_cast_time.uninstall()
    assert via.dtype == classic.dtype and torch.equal(via, classic)


def test_a_dynamic_patcher_keeps_stocks_own_method(comfy_cpu, monkeypatch):
    import comfy.model_patcher

    from dgx_monarch.actor import lora_cast_time

    _root, patcher, cpu = _patched(comfy_cpu)
    classic = patcher.patch_weight_to_device(KEYS[0], device_to=cpu, return_weight=True)
    monkeypatch.setattr(comfy.model_patcher.ModelPatcher, "is_dynamic", lambda self: True)
    lora_cast_time.install()
    try:
        via = patcher.patch_weight_to_device(KEYS[0], device_to=cpu, return_weight=True)
    finally:
        lora_cast_time.uninstall()
    assert comfy_cpu.same_weight(via, classic)


def test_the_bake_takes_cast_time_on_exactly_the_keys_stocks_own_load_does(comfy_cpu):
    """Stock's ModelPatcherDynamic.load chooses; the bake must choose the same keys,
    across the 16 KiB cut-off for BF16 and INT8 ConvRot, a bias and a resize."""
    assert comfy_cpu.check_stock_choice() == 17
