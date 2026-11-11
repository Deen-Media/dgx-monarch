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


# A partial load. ComfyUI's classic patcher gives every patched module it leaves
# off the device a cast-time patch, the small ones included; stock's DynamicVRAM
# load force-loads a module of 16 KiB or less through the classic bake. With the
# LoRA compute dtype GB10 reports (FP16) the two recipes round a BF16 norm
# weight with a diff apart, so the tests pin that dtype on this CPU.

_NORM_DIM = 1024  # 2 KiB of bf16: a module stock force-loads


def _partial_rig():
    import comfy.lora
    import comfy.model_patcher
    import comfy.ops
    import torch

    ops = comfy.ops.disable_weight_init

    class _Root(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = ops.Linear(256, 64, bias=False, dtype=torch.bfloat16)  # 32 KiB: a cast-time key
            self.norm = ops.RMSNorm(_NORM_DIM, dtype=torch.bfloat16)

        def get_dtype_inference(self):
            return torch.bfloat16

    gen = torch.Generator().manual_seed(11)
    pristine = {"lin.weight": (torch.randn(64, 256, generator=gen) * 0.05).to(torch.bfloat16),
                "norm.weight": (1.0 + 0.2 * torch.randn(_NORM_DIM, generator=gen)).to(torch.bfloat16)}
    lora = {"lin.lora_up.weight": torch.randn(64, 4, generator=gen) * 0.01,
            "lin.lora_down.weight": torch.randn(4, 256, generator=gen) * 0.01,
            "lin.alpha": torch.tensor(4.0),
            "norm.diff": (torch.randn(_NORM_DIM, generator=gen) * 0.02).to(torch.bfloat16)}
    patches = comfy.lora.load_lora(lora, {"lin": "lin.weight", "norm": "norm.weight"})
    cpu = torch.device("cpu")

    def build():
        root = _Root()
        root.lin.weight = torch.nn.Parameter(pristine["lin.weight"].clone(), requires_grad=False)
        root.norm.weight = torch.nn.Parameter(pristine["norm.weight"].clone(), requires_grad=False)
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.5)
        return root, patcher

    return build, pristine, cpu


@pytest.fixture
def gb10_lora_dtype(comfy_cpu):
    """The LoRA compute dtype GB10 reports, pinned for this CPU and put back."""
    import comfy.model_management as mm
    import torch

    cpu = torch.device("cpu")
    before = dict(mm.LORA_COMPUTE_DTYPES)
    mm.LORA_COMPUTE_DTYPES[cpu] = torch.float16
    try:
        yield
    finally:
        mm.LORA_COMPUTE_DTYPES.clear()
        mm.LORA_COMPUTE_DTYPES.update(before)


def _cast_weight(module):
    """The weight the layer's forward computes with, through ComfyUI's own cast."""
    import comfy.ops
    import torch

    probe = torch.zeros(1, dtype=torch.bfloat16)
    weight, _bias, stream = comfy.ops.cast_bias_weight(module, probe, offloadable=True)
    out = weight.clone()
    comfy.ops.uncast_bias_weight(module, weight, None, stream)
    return out


@pytest.mark.parametrize("route", ["partial load", "partial unload"])
def test_a_small_module_left_off_the_device_computes_with_stocks_classic_bake(
        comfy_cpu, gb10_lora_dtype, route):
    import torch

    from dgx_monarch.actor import lora_cast_time

    build, pristine, cpu = _partial_rig()
    twin, reference = build()
    # What stock's DynamicVRAM load gives each key: the force-load bake of the
    # small module, the cast-time weight of the large one.
    stock_norm = reference.patch_weight_to_device("norm.weight", device_to=cpu, return_weight=True, force_cast=True)
    stock_lin = comfy_cpu.stock_cast_time_weight(twin, reference, "lin.weight")

    def offload(patcher):
        if route == "partial load":
            patcher.load(cpu, lowvram_model_memory=1, full_load=False)
        else:
            patcher.load(cpu, full_load=True)
            patcher.partially_unload(cpu, memory_to_free=1e12)

    # ComfyUI's classic patcher alone: the small module takes the cast-time
    # recipe and misses stock's weight. This is the gap.
    root, patcher = build()
    offload(patcher)
    assert root.norm.weight_function and root.lin.weight_function
    assert not torch.equal(_cast_weight(root.norm), stock_norm)
    patcher.unpatch_model(cpu)

    root, patcher = build()
    assert lora_cast_time.install() is True
    try:
        offload(patcher)
        assert getattr(root.norm.weight_function[0], "_dgxm_full_load_weight", False)
        assert not getattr(root.lin.weight_function[0], "_dgxm_full_load_weight", False)
        assert torch.equal(_cast_weight(root.norm), stock_norm)
        assert torch.equal(_cast_weight(root.lin), stock_lin)
        # The stored weights are untouched, and an unpatch leaves the file's values.
        assert torch.equal(root.norm.weight, pristine["norm.weight"])
        patcher.unpatch_model(cpu)
        assert torch.equal(root.norm.weight, pristine["norm.weight"])
        assert torch.equal(root.lin.weight, pristine["lin.weight"])
        assert not root.norm.weight_function
        # A full load of the same stack bakes the same two weights.
        patcher.load(cpu, full_load=True)
        assert torch.equal(root.norm.weight, stock_norm)
        assert torch.equal(root.lin.weight, stock_lin)
    finally:
        assert lora_cast_time.uninstall() is True
    import comfy.model_patcher as comfy_mp

    for name in ("load", "partially_unload", "patch_weight_to_device"):
        assert not getattr(getattr(comfy_mp.ModelPatcher, name), "_dgxm_cast_time", False)


@pytest.mark.parametrize("route", ["partial load", "partial unload"])
def test_a_quantized_layer_left_off_the_device_computes_with_the_full_loads_weight(comfy_cpu, route):
    """An INT8 ConvRot layer with a LoRA, left off the device. ComfyUI's classic
    cast hands its cast-time patch the dequantized weight and never requantizes,
    so the layer computes with another weight than stock's cast and than a full
    load's bake. With the route installed the cast returns stock's requantized
    weight, and the layer's output is the full load's, bit for bit; the BF16
    layer beside it was already the same."""
    import comfy.model_patcher
    import torch

    from dgx_monarch.actor import lora_cast_time

    cpu = torch.device("cpu")
    x = torch.randn(3, 512, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)

    def build():
        root, patches = comfy_cpu.tiny_cast_time_model()
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.8)
        return root, patcher

    def offload(patcher):
        if route == "partial load":
            patcher.load(cpu, lowvram_model_memory=1, full_load=False)
        else:
            patcher.load(cpu, full_load=True)
            patcher.partially_unload(cpu, memory_to_free=1e12)

    def outputs(root):
        with torch.no_grad():
            return root.diffusion_model.q(x), root.diffusion_model.p(x)

    assert lora_cast_time.install() is True
    try:
        full_root, full = build()
        full.load(cpu, full_load=True)
        full_q, full_p = outputs(full_root)
        root, patcher = build()
        offload(patcher)
        assert getattr(root.diffusion_model.q.weight_function[0], "_dgxm_full_load_weight", False)
        assert not getattr(root.diffusion_model.p.weight_function[0], "_dgxm_full_load_weight", False)
        part_q, part_p = outputs(root)
        assert torch.equal(part_q, full_q) and torch.equal(part_p, full_p)
    finally:
        assert lora_cast_time.uninstall() is True
    # ComfyUI's classic patcher alone, against the same full-load outputs: the
    # quantized layer misses, which is the gap; the BF16 layer does not.
    root, patcher = build()
    offload(patcher)
    alone_q, alone_p = outputs(root)
    assert not torch.equal(alone_q, full_q)
    assert torch.equal(alone_p, full_p)


def _offload(patcher, cpu, route):
    if route == "partial load":
        patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    else:
        patcher.load(cpu, full_load=True)
        patcher.partially_unload(cpu, memory_to_free=1e12)


@pytest.mark.parametrize("store", ["float8_e4m3fn", "float32"])
@pytest.mark.parametrize("route", ["partial load", "partial unload"])
def test_a_plain_weight_stored_in_another_dtype_left_off_the_device_computes_with_the_full_loads_weight(
        comfy_cpu, gb10_lora_dtype, store, route):
    """A plain Linear above 16 KiB stored in another dtype than the forward (a plain FP8
    file under a BF16 manual cast, an FP32 island). A full load bakes it classically, since
    cast_time_weight declines it; ComfyUI's classic patch on the cast weight does not."""
    import comfy.lora
    import comfy.model_patcher
    import comfy.ops
    import torch

    from dgx_monarch.actor import lora_cast_time

    dtype, cpu = getattr(torch, store), torch.device("cpu")

    class _Root(torch.nn.Module):
        manual_cast_dtype = torch.bfloat16

        def __init__(self):
            super().__init__()
            self.lin = comfy.ops.manual_cast.Linear(512, 64, bias=False, dtype=dtype)

        def get_dtype_inference(self):
            return torch.bfloat16

    gen = torch.Generator().manual_seed(5)
    pristine = (torch.randn(64, 512, generator=gen) * 0.05).to(dtype)
    lora = {"lin.lora_up.weight": torch.randn(64, 4, generator=gen) * 0.01,
            "lin.lora_down.weight": torch.randn(4, 512, generator=gen) * 0.01,
            "lin.alpha": torch.tensor(4.0)}
    patches = comfy.lora.load_lora(lora, {"lin": "lin.weight"})

    def build():
        root = _Root()
        root.lin.weight = torch.nn.Parameter(pristine.clone(), requires_grad=False)
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.8)
        return root, patcher

    assert lora_cast_time.install() is True
    try:
        full_root, full = build()
        full.load(cpu, full_load=True)
        full_weight = _cast_weight(full_root.lin)
        root, patcher = build()
        _offload(patcher, cpu, route)
        assert torch.equal(_cast_weight(root.lin), full_weight)
    finally:
        assert lora_cast_time.uninstall() is True
    root, patcher = build()
    _offload(patcher, cpu, route)
    assert not torch.equal(_cast_weight(root.lin), full_weight)  # ComfyUI's classic patcher alone


@pytest.mark.parametrize("route", ["partial load", "partial unload"])
def test_a_small_quantized_layer_left_off_the_device_computes_with_the_full_loads_weight(
        comfy_cpu, route):
    """An INT8 ConvRot Linear of 16 KiB or less. Stock force-loads it through the classic
    bake (the quantized setter), and so does a full load; ComfyUI's classic patch hands the
    layer the dequantized weight plus the LoRA, never requantized."""
    import comfy.model_patcher
    import torch

    from dgx_monarch.actor import lora_cast_time

    cpu = torch.device("cpu")
    x = torch.randn(3, 256, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)

    def build():
        root, patches = comfy_cpu.tiny_cast_time_model(out_features=16, in_features=256)  # q: 4 KiB
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.8)
        return root, patcher

    def output(root):
        with torch.no_grad():
            return root.diffusion_model.q(x)

    assert lora_cast_time.install() is True
    try:
        full_root, full = build()
        full.load(cpu, full_load=True)
        expected = output(full_root)
        root, patcher = build()
        _offload(patcher, cpu, route)
        assert torch.equal(output(root), expected)
    finally:
        assert lora_cast_time.uninstall() is True
    root, patcher = build()
    _offload(patcher, cpu, route)
    assert not torch.equal(output(root), expected)


@pytest.mark.parametrize("out_features, in_features", [(64, 512), (16, 256)], ids=["32 KiB", "4 KiB"])
@pytest.mark.parametrize("route", ["partial load", "partial unload"])
def test_a_quantized_layer_on_a_full_precision_matmul_left_off_the_device_matches_the_full_load(
        comfy_cpu, route, out_features, in_features):
    """An INT8 ConvRot Linear whose layer config asks for a full-precision matmul.

    ``cast_time_weight`` declines it at any size, so a full load bakes it with
    ComfyUI's classic quantized setter and the forward dequantizes that weight.
    Left off the device, the swapped patch runs the same bake when the layer
    casts; ComfyUI's own patch hands the forward the dequantized weight plus
    the LoRA, which was never through the setter's rounding.
    """
    import comfy.model_patcher
    import torch

    from dgx_monarch.actor import lora_cast_time

    cpu = torch.device("cpu")
    x = torch.randn(3, in_features, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)

    def build():
        root, patches = comfy_cpu.tiny_cast_time_model(out_features=out_features, in_features=in_features)
        root.diffusion_model.q._full_precision_mm = True
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.8)
        return root, patcher

    def output(root):
        with torch.no_grad():
            return root.diffusion_model.q(x)

    assert lora_cast_time.install() is True
    try:
        full_root, full = build()
        full.load(cpu, full_load=True)
        expected = output(full_root)
        root, patcher = build()
        _offload(patcher, cpu, route)
        assert getattr(root.diffusion_model.q.weight_function[0], "_dgxm_full_load_weight", False)
        assert torch.equal(output(root), expected)
    finally:
        assert lora_cast_time.uninstall() is True
    root, patcher = build()
    _offload(patcher, cpu, route)
    assert not torch.equal(output(root), expected)
