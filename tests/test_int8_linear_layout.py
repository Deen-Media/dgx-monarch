from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import int8_linear_layout as layout


class _Linear(torch.nn.Module):
    quant_format = "int8_tensorwise"
    layout_type = "TensorWiseINT8Layout"

    def __init__(self):
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def _forward(self, input, weight, bias):
        self.calls.append(input)
        return input * 3 + bias


class _Tree(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = _Linear()

    def _forward(self, x, timestep, context):
        return x + timestep + context


def _weight(*, convrot=False, layout_name="TensorWiseINT8Layout"):
    return SimpleNamespace(_layout_cls=layout_name, _params=SimpleNamespace(convrot=convrot))


def test_tensorwise_int8_normalizes_noncontiguous_effective_input_to_native_rows():
    tree = _Tree()
    assert layout.install(tree) == 1
    assert layout.install(tree) == 0
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    assert not value.is_contiguous()
    with torch.no_grad():
        expected = value.contiguous() * 3 + 2
        actual = tree.linear._forward(value, _weight(), torch.tensor(2.0))
    assert tree.linear.calls[-1].is_contiguous()
    assert torch.equal(actual, expected)
    assert torch.equal(tree.linear.calls[-1], value)


def test_root_model_forward_with_model_arguments_is_untouched():
    tree = _Tree()
    original = tree._forward
    layout.install(tree)
    assert tree._forward.__func__ is original.__func__
    assert torch.equal(tree._forward(torch.tensor(1), torch.tensor(2), torch.tensor(3)), torch.tensor(6))


def test_only_nonconvrot_tensorwise_quantized_effective_weights_change_layout():
    tree = _Tree()
    layout.install(tree)
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    for weight in (_weight(convrot=True), _weight(layout_name="TensorCoreFP8Layout"), torch.ones(1)):
        tree.linear._forward(value, weight, torch.tensor(0.0))
        assert tree.linear.calls[-1] is value


def test_current_comfy_qtensor_flinear_cpu_fake_backend_matches_inference_reference():
    """Exercise current Comfy's QT F.linear dispatcher without a CUDA kernel."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    comfy_dir = os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR")
    if not comfy_dir:
        pytest.skip("set COMFY_DIR or COMFYUI_DIR to run the optional current-Comfy canary")
    source = Path(__file__).resolve().parents[1] / "src"
    code = r'''
import sys
sys.argv = ["int8-layout-test", "--cpu"]
import comfy.options
comfy.options.args_parsing = True
import torch
from comfy.quant_ops import QuantizedTensor, get_layout_class
from comfy_kitchen.tensor import base
from dgx_monarch.adapters import int8_linear_layout as layout

layout_cls = get_layout_class("TensorWiseINT8Layout")
weight = torch.nn.Parameter(QuantizedTensor(torch.ones((1, 4), dtype=torch.int8), "TensorWiseINT8Layout",
                                           layout_cls.Params(scale=torch.ones(1, dtype=torch.float32),
                                                             orig_dtype=torch.bfloat16, orig_shape=(1, 4))),
                            requires_grad=False)
def matrix(q, input):
    w = q.dequantize().float()
    return w if input.shape[-1] == w.shape[0] else w.t()
def linear_handler(_qt, args, _kwargs):
    input, qweight, bias = args
    return (input.float() @ matrix(qweight, input) + bias.float()).to(torch.bfloat16)
def addmm_handler(_qt, args, _kwargs):
    bias, input, qweight = args[:3]
    return (input.float() @ matrix(qweight, input) + bias.float()).to(torch.bfloat16)
def mm_handler(_qt, args, _kwargs):
    input, qweight = args[:2]
    return (input.float() @ matrix(qweight, input)).to(torch.bfloat16)
base._LAYOUT_DISPATCH_TABLE[torch.ops.aten.linear.default][layout_cls] = linear_handler
base._LAYOUT_DISPATCH_TABLE[torch.ops.aten.addmm.default][layout_cls] = addmm_handler
base._LAYOUT_DISPATCH_TABLE[torch.ops.aten.mm.default][layout_cls] = mm_handler
class Linear(torch.nn.Module):
    quant_format = "int8_tensorwise"; layout_type = "TensorWiseINT8Layout"
    def __init__(self): super().__init__(); self.seen = None
    def _forward(self, input, effective_weight, bias):
        self.seen = input
        return torch.nn.functional.linear(input, effective_weight, bias)
class Tree(torch.nn.Module):
    def __init__(self): super().__init__(); self.linear = Linear()
tree = Tree()
# Noncontiguous rows are [256,1,0,0] and [0,0,0,0]. A fused FP32+bias
# result is 1; the split mm result rounds 257 to BF16 256 before adding -256.
value = torch.tensor([[[256, 0], [1, 0], [0, 0], [0, 0]]], dtype=torch.bfloat16).transpose(1, 2)
bias = torch.tensor([-256], dtype=torch.bfloat16)
assert not value.is_contiguous()
with torch.inference_mode():
    native = tree.linear._forward(value.clone(), weight, bias)
with torch.no_grad():
    before = tree.linear._forward(value, weight, bias)
assert not torch.equal(before, native)
assert layout.install(tree) == 1
with torch.no_grad(): fixed = tree.linear._forward(value, weight, bias)
assert torch.equal(fixed, native) and tree.linear.seen.is_contiguous()
# NVFP4 uses the pack's direct native `aten.linear` call and must return a
# normal tensor rather than a QuantizedTensor wrapper.
nv = get_layout_class("TensorCoreNVFP4Layout")
def nv_linear(_qt, _args, _kwargs): return torch.tensor([[9]], dtype=torch.bfloat16)
base._LAYOUT_DISPATCH_TABLE[torch.ops.aten.linear.default][nv] = nv_linear
nv_params = lambda: nv.Params(scale=torch.ones((), dtype=torch.float32),
                               block_scale=torch.ones((1, 1), dtype=torch.float8_e4m3fn),
                               orig_dtype=torch.bfloat16, orig_shape=(1, 4))
nv_input = QuantizedTensor(torch.zeros((1, 4), dtype=torch.uint8), "TensorCoreNVFP4Layout", nv_params())
nv_weight = QuantizedTensor(torch.zeros((1, 4), dtype=torch.uint8), "TensorCoreNVFP4Layout", nv_params())
class NvLinear(torch.nn.Module):
    quant_format = "nvfp4"; layout_type = "TensorCoreNVFP4Layout"
    def _forward(self, input, weight, bias): raise AssertionError("must dispatch natively")
nv_tree = torch.nn.Module(); nv_tree.linear = NvLinear()
assert layout.install(nv_tree) == 1
nv_out = nv_tree.linear._forward(nv_input, nv_weight, None)
assert type(nv_out) is torch.Tensor and torch.equal(nv_out, torch.tensor([[9]], dtype=torch.bfloat16))
'''
    env = os.environ | {"PYTHONPATH": f"{comfy_dir}:{source}"}
    completed = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                               capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr

class _Nvfp4Input:
    _layout_cls = "TensorCoreNVFP4Layout"


class _Nvfp4Weight:
    _layout_cls = "TensorCoreNVFP4Layout"
    calls = []

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        cls.calls.append((func, types, args, kwargs))
        return torch.tensor([7.0])


class _Nvfp4Linear(torch.nn.Module):
    quant_format = "nvfp4"
    layout_type = "TensorCoreNVFP4Layout"

    def _forward(self, *_args):
        raise AssertionError("NVFP4 native dispatch should bypass F.linear lowering")


def test_nvfp4_effective_qtensor_pair_uses_current_comfy_native_linear_dispatch():
    tree = torch.nn.Module()
    tree.linear = _Nvfp4Linear()
    assert layout.install(tree) == 1
    _Nvfp4Weight.calls.clear()
    result = tree.linear._forward(_Nvfp4Input(), _Nvfp4Weight(), object())
    assert torch.equal(result, torch.tensor([7.0]))
    assert len(_Nvfp4Weight.calls) == 1
    func, types, args, kwargs = _Nvfp4Weight.calls[0]
    assert func is torch.ops.aten.linear.default
    assert types == (_Nvfp4Input, _Nvfp4Weight)
    assert isinstance(args[0], _Nvfp4Input) and isinstance(args[1], _Nvfp4Weight)
    assert kwargs == {}
