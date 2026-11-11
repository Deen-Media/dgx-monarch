"""actor/lora_kept_bake.py: a full load's baked weight kept on a module ComfyUI leaves off the device.

A layer of a format that also quantizes its input (FP8 here) computes the full
load's output only while it carries no cast-time patch, so the weight is baked
once and kept on the host beside ComfyUI's backup. Where the host has no room
for that copy a weight-only format falls back to the patch at cast, and an
input-quantizing layer has no exact fallback: the render is refused, class C,
before its first forward. Real ComfyUI on CPU (skipped where it is not
importable); the rig is the reviewer's FP8 probe of 2026-11-11.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch.actor import lora_kept_bake
from dgx_monarch.capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES
from dgx_monarch.refusal import RefusalClass, RefusalTag, parse_leading_refusal_tag
from module_location_helpers import from_checkout

OUT, IN, RANK = 64, 512, 4
LAYOUTS = {"float8_e4m3fn": "TensorCoreFP8E4M3Layout", "int8_tensorwise": "TensorWiseINT8Layout"}


def _device(kind):
    return SimpleNamespace(load_device=SimpleNamespace(type=kind, index=0))


def test_nothing_is_kept_where_the_load_device_shares_the_hosts_memory(monkeypatch):
    """A kept bake is a second copy of the layer; on a shared pool it would come from the render's room."""
    import psutil
    import torch

    for patcher in (_device("cpu"), _device("mps"), SimpleNamespace()):
        budget = lora_kept_bake.Budget(patcher)
        assert budget.separate is False and budget.left == 0 and budget.take(1) is False

    def properties(integrated):
        return lambda device: SimpleNamespace(is_integrated=integrated)

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties(1))  # GB10 in a DGX Spark
    assert lora_kept_bake.separate_host_memory(_device("cuda")) is False
    assert lora_kept_bake.Budget(_device("cuda")).take(1) is False
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties(0))  # a GPU with memory of its own
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=ABSOLUTE_HOST_FLOOR_BYTES + 1000))
    budget = lora_kept_bake.Budget(_device("cuda"))
    assert budget.separate is True and budget.left == 1000  # what the host has above the pack's floor
    assert budget.take(600) is True and budget.take(600) is False and budget.take(400) is True

    def no_answer(device):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", no_answer)
    assert lora_kept_bake.separate_host_memory(_device("cuda")) is False  # unknown is treated as shared


@pytest.fixture
def own_memory(monkeypatch):
    """Stand this CPU in for a GPU with memory of its own, where a host copy may be kept."""
    monkeypatch.setattr(lora_kept_bake, "separate_host_memory", lambda patcher: True)


@pytest.fixture(scope="module")
def comfy_cpu():
    """Real ComfyUI with --cpu, restored afterwards."""
    def is_comfy(name):
        return name == "folder_paths" or name == "comfy" or name.startswith(("comfy.", "comfy_"))

    preserved = {name: module for name, module in sys.modules.items() if is_comfy(name)}
    for name in preserved:
        sys.modules.pop(name, None)
    path_before, argv_before = list(sys.path), sys.argv
    comfy_dir = os.path.abspath(os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR") or "../ComfyUI")
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    try:
        sys.argv = ["pytest-lora-kept-bake", "--cpu"]
        try:
            options = pytest.importorskip("comfy.options", reason="no ComfyUI on sys.path")
            options.enable_args_parsing()
            pytest.importorskip("comfy.lora")
            pytest.importorskip("comfy.quant_ops")
        finally:
            sys.argv = argv_before
        yield
    finally:
        from dgx_monarch.actor import lora_cast_time

        if "comfy.model_patcher" in sys.modules:
            lora_cast_time.uninstall()
        gone = [name for name, module in list(sys.modules.items())
                if is_comfy(name) or from_checkout(module, comfy_dir)]
        for name in gone:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = path_before


@pytest.fixture
def installed(comfy_cpu):
    from dgx_monarch.actor import lora_cast_time

    assert lora_cast_time.install() is True
    try:
        yield
    finally:
        assert lora_cast_time.uninstall() is True


def _rig(formats=("float8_e4m3fn",)):
    """A root with one patched quantized Linear per format, each with a bias, and its patcher."""
    import comfy.lora
    import comfy.model_patcher
    import comfy.ops
    import torch
    from comfy.quant_ops import QuantizedTensor

    cpu = torch.device("cpu")
    ops = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16)
    names = [f"q{index}" for index in range(len(formats))]

    class _Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name in names:
                setattr(self, name, ops.Linear(IN, OUT, bias=True))

    class _Root(torch.nn.Module):
        manual_cast_dtype = torch.bfloat16

        def __init__(self):
            super().__init__()
            self.diffusion_model = _Block()

        def get_dtype_inference(self):
            return torch.bfloat16

    gen = torch.Generator().manual_seed(7)
    sd, lora, key_map = {}, {}, {}
    for name, fmt in zip(names, formats, strict=True):
        w = (torch.randn(OUT, IN, generator=gen) * 0.05).to(torch.bfloat16)
        if fmt == "float8_e4m3fn":
            qt, conf = QuantizedTensor.from_float(w, LAYOUTS[fmt]), {"format": fmt}
        else:
            qt = QuantizedTensor.from_float(w, LAYOUTS[fmt], per_channel=True, convrot=True,
                                            convrot_groupsize=256, is_weight=True)
            conf = {"format": fmt, "convrot": True, "convrot_groupsize": 256}
        unit = f"diffusion_model.{name}"
        sd[f"{unit}.weight"] = qt._qdata
        sd[f"{unit}.weight_scale"] = qt._params.scale
        sd[f"{unit}.comfy_quant"] = torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)
        sd[f"{unit}.bias"] = torch.randn(OUT, generator=gen).to(torch.bfloat16)
        lora[f"{unit}.lora_up.weight"] = torch.randn(OUT, RANK, generator=gen) * 0.01
        lora[f"{unit}.lora_down.weight"] = torch.randn(RANK, IN, generator=gen) * 0.01
        lora[f"{unit}.alpha"] = torch.tensor(float(RANK))
        key_map[unit] = f"{unit}.weight"
    patches = comfy.lora.load_lora(lora, key_map)

    def build():
        root = _Root()
        root.load_state_dict({key: value.clone() for key, value in sd.items()}, strict=True)
        patcher = comfy.model_patcher.ModelPatcher(root, cpu, cpu)
        patcher.add_patches(patches, 0.8)
        return root, patcher

    x = torch.randn(3, IN, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)

    def outputs(root):
        with torch.no_grad():
            return [getattr(root.diffusion_model, name)(x) for name in names]

    return build, outputs, cpu


def _offload(patcher, cpu, route):
    if route == "partial load":
        patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    else:
        patcher.load(cpu, full_load=True)
        patcher.partially_unload(cpu, memory_to_free=1e12)


def _full_load(build, outputs, cpu):
    root, patcher = build()
    patcher.load(cpu, full_load=True)
    return outputs(root)


def test_which_layers_quantize_their_input(comfy_cpu):
    build, _outputs, _cpu = _rig(("float8_e4m3fn", "int8_tensorwise"))
    root, _patcher = build()
    fp8, int8 = root.diffusion_model.q0, root.diffusion_model.q1
    assert lora_kept_bake.quantizes_input(fp8) is True
    assert lora_kept_bake.quantizes_input(int8) is False  # a weight-only format
    fp8._full_precision_mm = True
    assert lora_kept_bake.quantizes_input(fp8) is False
    assert lora_kept_bake.quantizes_input(SimpleNamespace(weight=None)) is False


@pytest.mark.parametrize("route", ["partial load", "partial unload"])
def test_an_fp8_layer_left_off_the_device_computes_the_full_loads_output_from_a_kept_bake(
        installed, own_memory, route):
    """The gap no cast-time patch closes: with the patch attached the forward leaves the input unquantized."""
    import torch

    build, outputs, cpu = _rig()
    (expected,) = _full_load(build, outputs, cpu)
    root, patcher = build()
    _offload(patcher, cpu, route)
    q = root.diffusion_model.q0
    assert q.weight_function == [] and q.comfy_patched_weights is True and getattr(q, lora_kept_bake.KEPT)
    assert "diffusion_model.q0.weight" in patcher.backup
    assert torch.equal(outputs(root)[0], expected)
    assert lora_kept_bake.inexact(patcher) == []
    lora_kept_bake.check(patcher)  # nothing to refuse


@pytest.mark.parametrize("where", ["a shared pool", "a full host"])
@pytest.mark.parametrize("route", ["partial load", "partial unload"])
def test_with_no_room_for_the_bake_the_fp8_layer_differs_and_the_render_is_refused(installed, monkeypatch, route, where):
    """On a shared pool (a Spark, and this CPU) nothing is kept; on a GPU of its own the host may be full."""
    import torch

    build, outputs, cpu = _rig()
    (expected,) = _full_load(build, outputs, cpu)
    if where == "a full host":
        monkeypatch.setattr(lora_kept_bake, "separate_host_memory", lambda patcher: True)
        monkeypatch.setattr(lora_kept_bake.Budget, "take", lambda self, nbytes: False)
    root, patcher = build()
    _offload(patcher, cpu, route)
    q = root.diffusion_model.q0
    assert getattr(q.weight_function[0], "_dgxm_full_load_weight", False)
    assert not torch.equal(outputs(root)[0], expected)  # the weight is the full load's; the input is not quantized
    short = lora_kept_bake.inexact(patcher)
    assert [key for key, _nbytes in short] == ["diffusion_model.q0.weight"]
    with pytest.raises(lora_kept_bake.OffloadedInputQuantError) as raised:
        lora_kept_bake.check(patcher)
    message = str(raised.value)
    assert parse_leading_refusal_tag(message) == RefusalTag(
        refusal_class=RefusalClass.CAPACITY, guard="partial_load_divergence", waivable=False)
    for words in ("float8_e4m3fn", "diffusion_model.q0.weight", "nothing was sampled", "lora_low_rss on",
                  "an INT8 or BF16 checkpoint"):
        assert words in message, words
    if where == "a shared pool":
        assert "this GPU shares the host's memory" in message and "more free host memory" not in message
    else:
        assert "GiB more free host memory" in message and "is available above the 5 GiB floor" in message
    # On a GPU of its own with room on the host, the next load keeps the bake and the render goes on.
    monkeypatch.undo()
    monkeypatch.setattr(lora_kept_bake, "separate_host_memory", lambda patcher: True)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    assert q.weight_function == [] and torch.equal(outputs(root)[0], expected)
    lora_kept_bake.check(patcher)


def test_a_kept_module_survives_every_move_comfyui_makes(installed, own_memory):
    """Unload again, load whole, unpatch: no phantom free, no second bake, the file's values back."""
    import torch

    build, outputs, cpu = _rig()
    (expected,) = _full_load(build, outputs, cpu)
    full_root, full = build()
    full.load(cpu, full_load=True)
    baked = full_root.diffusion_model.q0.weight
    pristine_root, _unused = build()
    pristine = pristine_root.diffusion_model.q0.weight

    def same(a, b):
        return torch.equal(a.dequantize(), b.dequantize()) and torch.equal(a._qdata, b._qdata)

    root, patcher = build()
    q = root.diffusion_model.q0
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    assert same(q.weight, baked) and same(patcher.backup["diffusion_model.q0.weight"].weight, pristine)
    # A further partial unload passes over it: it is off the device already and frees nothing.
    loaded_before = patcher.model.model_loaded_weight_memory
    assert patcher.partially_unload(cpu, memory_to_free=1e12) == 0
    assert patcher.model.model_loaded_weight_memory == loaded_before
    assert q.comfy_patched_weights is True and same(q.weight, baked)
    assert torch.equal(outputs(root)[0], expected)
    # Loaded whole, it is moved, not baked a second time.
    patcher.load(cpu, full_load=True)
    assert same(q.weight, baked) and not hasattr(q, "prev_comfy_cast_weights")
    assert torch.equal(outputs(root)[0], expected)
    lora_kept_bake.forget_moved(patcher)
    assert not hasattr(q, lora_kept_bake.KEPT)
    # Offloaded once more through ComfyUI's own unload, it is restored, moved and kept again.
    patcher.partially_unload(cpu, memory_to_free=1e12)
    assert q.weight_function == [] and same(q.weight, baked)
    assert torch.equal(outputs(root)[0], expected)
    # A full unpatch restores the file's weight and leaves no mark behind for the next load.
    patcher.unpatch_model(cpu)
    assert same(q.weight, pristine) and not patcher.backup and not hasattr(q, "comfy_patched_weights")
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    assert same(q.weight, baked) and torch.equal(outputs(root)[0], expected)


def test_layers_that_quantize_their_input_take_the_room_first(installed, own_memory, monkeypatch):
    """Room for one module: the FP8 layer keeps its bake, the INT8 layer computes its weight at cast."""
    import torch

    from dgx_monarch.actor import lora_cast_time

    build, outputs, cpu = _rig(("int8_tensorwise", "float8_e4m3fn"))  # the INT8 layer comes first in the model
    expected = _full_load(build, outputs, cpu)
    root, patcher = build()
    int8, fp8 = root.diffusion_model.q0, root.diffusion_model.q1
    one = lora_cast_time._module_bytes(fp8)

    def room_for_one(self, patcher):
        self.separate, self.left = True, one

    monkeypatch.setattr(lora_kept_bake.Budget, "__init__", room_for_one)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    assert fp8.weight_function == [] and getattr(fp8, lora_kept_bake.KEPT)
    assert getattr(int8.weight_function[0], "_dgxm_full_load_weight", False)
    got = outputs(root)
    assert torch.equal(got[0], expected[0]) and torch.equal(got[1], expected[1])  # both exact
    lora_kept_bake.check(patcher)


def test_the_single_rank_guard_rides_prepare_sampling_once(installed, monkeypatch):
    """partial_load_guard.install puts the check on a one-rank patcher; it runs after the load."""
    import comfy.patcher_extension as pe

    from dgx_monarch.actor import partial_load_guard

    build, _outputs, cpu = _rig()
    _root, patcher = build()
    partial_load_guard.install(patcher, 1)
    partial_load_guard.install(patcher, None)  # a second install adds nothing
    wrappers = pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, lora_kept_bake.WRAPPER_KEY, patcher.model_options, is_model_options=True)
    assert len(wrappers) == 1
    assert not pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, partial_load_guard.WRAPPER_KEY, patcher.model_options,
        is_model_options=True)  # the cross-rank guard is for a world above one
    # A whole load passes; so does a partial one whose bake is kept (a GPU of its own).
    patcher.load(cpu, full_load=True)
    assert wrappers[0](lambda *args, **kwargs: ("real model", "conds", []), patcher, (1, 4, 8, 8), {}) == (
        "real model", "conds", [])
    patcher.unpatch_model(cpu)
    monkeypatch.setattr(lora_kept_bake, "separate_host_memory", lambda patcher: True)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    assert wrappers[0](lambda *args, **kwargs: ("real model", "conds", []), patcher, (1, 4, 8, 8), {})[0] == "real model"
    # On a shared pool the same partial load is refused before the first forward.
    monkeypatch.undo()
    patcher.unpatch_model(cpu)
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)
    with pytest.raises(lora_kept_bake.OffloadedInputQuantError):
        wrappers[0](lambda *args, **kwargs: ("real model", "conds", []), patcher, (1, 4, 8, 8), {})


def test_a_bake_that_fails_leaves_the_module_as_comfyui_had_it(installed):
    """Every bake is computed before anything is stored: a failure on the bias must not leave a baked weight.

    A weight baked and stored with no ``comfy_patched_weights`` beside it would
    be baked a second time by ComfyUI's next load.
    """
    import comfy.model_patcher as comfy_mp
    import torch

    build, outputs, cpu = _rig(("int8_tensorwise",))
    (expected,) = _full_load(build, outputs, cpu)
    root, patcher = build()
    q = root.diffusion_model.q0
    patcher.load(cpu, lowvram_model_memory=1, full_load=False)  # a shared pool: the patch computes at cast
    pristine = q.weight._qdata.clone()
    weight_patch = q.weight_function[0]
    # A second cast-time patch, on the bias, whose bake fails.
    bias_key = "diffusion_model.q0.bias"
    patcher.patches[bias_key] = patcher.patches["diffusion_model.q0.weight"]
    q.bias_function = [comfy_mp.LowVramPatch(bias_key, patcher.patches)]
    bake = patcher.patch_weight_to_device

    def fails_on_the_bias(key, *args, **kwargs):
        if key == bias_key:
            raise RuntimeError("no bake for the bias")
        return bake(key, *args, **kwargs)

    patcher.patch_weight_to_device = fails_on_the_bias
    try:
        room = SimpleNamespace(take=lambda nbytes: True)
        assert lora_kept_bake.keep(patcher, q, room) is False
    finally:
        del patcher.patch_weight_to_device
    assert torch.equal(q.weight._qdata, pristine)
    assert q.weight_function == [weight_patch] and len(q.bias_function) == 1
    assert not getattr(q, "comfy_patched_weights", False) and not hasattr(q, lora_kept_bake.KEPT)
    assert not patcher.backup
    del patcher.patches[bias_key]
    q.bias_function = []
    assert torch.equal(outputs(root)[0], expected)  # still exact at cast, for a weight-only format
    # With both bakes working and room on the host, the same module is kept.
    assert lora_kept_bake.keep(patcher, q, room) is True
    assert q.weight_function == [] and q.comfy_patched_weights is True
    assert torch.equal(outputs(root)[0], expected)

