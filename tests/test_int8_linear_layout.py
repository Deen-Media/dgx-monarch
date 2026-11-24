from __future__ import annotations

import json
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


def test_tensorwise_int8_hands_a_noncontiguous_effective_input_over_as_native_rows():
    tree = _Tree()
    assert layout.install(tree) == 1
    assert layout.install(tree) == 0
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(1, 2)
    assert not value.is_contiguous()
    with torch.no_grad():
        expected = value.contiguous() * 3 + 2
        actual = tree.linear._forward(value, _weight(), torch.tensor(2.0))
    rows = tree.linear.calls[-1]
    assert rows.shape == (8, 3) and rows.is_contiguous()
    assert torch.equal(rows, value.reshape(8, 3))
    assert actual.shape == (2, 4, 3) and torch.equal(actual, expected)


def test_root_model_forward_with_model_arguments_is_untouched():
    tree = _Tree()
    original = tree._forward
    layout.install(tree)
    assert tree._forward.__func__ is original.__func__
    assert torch.equal(tree._forward(torch.tensor(1), torch.tensor(2), torch.tensor(3)), torch.tensor(6))


def test_every_weight_only_layout_gets_rows_and_other_weights_keep_the_input():
    """ConvRot included: a Worker's 3-D view can send any weight-only layout
    down the dequantizing expand route, so each one gets 2-D rows. An FP8 or
    NVFP4 weight quantizes its input in ComfyUI's own 2-D reshape, and a plain
    tensor is ComfyUI's dequantized LoRA path; those keep the input as it is,
    and so does a 2-D input."""
    tree = _Tree()
    layout.install(tree)
    chained = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    view = torch.zeros(2, 7, 4)[:, :3]
    weight_only = (_weight(), _weight(convrot=True),
                   *(_weight(layout_name=name) for name in sorted(layout.WEIGHT_ONLY_LAYOUTS)))
    assert len(weight_only) == 5
    for value in (chained, view, chained.transpose(1, 2)):
        for weight in weight_only:
            out = tree.linear._forward(value, weight, torch.tensor(0.0))
            assert tree.linear.calls[-1].ndim == 2 and tree.linear.calls[-1].is_contiguous()
            assert out.shape == value.shape and torch.equal(out, value * 3)
        for weight in (_weight(layout_name="TensorCoreFP8Layout"),
                       _weight(layout_name="TensorCoreNVFP4Layout"), torch.ones(1)):
            tree.linear._forward(value, weight, torch.tensor(0.0))
            assert tree.linear.calls[-1] is value
    flat = chained[0]
    for weight in weight_only:
        tree.linear._forward(flat, weight, torch.tensor(0.0))
        assert tree.linear.calls[-1] is flat


# Every weight-only format by name, with the layout ComfyUI loads it as, what
# comfy-kitchen's quantizer takes for it and the bytes a packed row of 256
# inputs holds. The grouped formats share one layout class and differ in code
# width, so each has its own row here and in the canary seam.
_FORMATS = {
    "int8_tensorwise": ("TensorWiseINT8Layout", {"is_weight": True, "per_channel": True}, 256),
    "convrot_w4a4": ("TensorCoreConvRotW4A4Layout", {}, 128),
    "w3a8_int8": ("AsymW4A8Int8Layout", {"bits": 3}, 96),
    "asym_w4a8_int8": ("AsymW4A8Int8Layout", {"bits": 4}, 128),
    "w5a8_int8": ("AsymW4A8Int8Layout", {"bits": 5}, 160),
    "w6a8_int8": ("AsymW4A8Int8Layout", {"bits": 6}, 192),
}


def test_every_format_of_the_table_has_a_row_by_name():
    assert {name: row[0] for name, row in _FORMATS.items()} == dict(layout.WEIGHT_ONLY_FORMATS)
    assert layout.WEIGHT_ONLY_LAYOUTS == frozenset(layout.WEIGHT_ONLY_FORMATS.values())


def test_a_format_ahead_of_a_release_is_in_the_table_and_names_its_comfy_commit():
    """The seam excuses a listed format only where ComfyUI does not know the
    name and this table names the ComfyUI commit that adds it."""
    assert set(layout.FORMATS_AHEAD_OF_A_RELEASE) == {"w3a8_int8", "w5a8_int8"}
    for name, commit in layout.FORMATS_AHEAD_OF_A_RELEASE.items():
        assert name in layout.WEIGHT_ONLY_FORMATS
        assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")


@pytest.mark.parametrize("quant_format", sorted(_FORMATS))
def test_each_weight_only_format_wraps_once(quant_format):
    tree = _Tree()
    tree.linear.quant_format, tree.linear.layout_type = quant_format, _FORMATS[quant_format][0]
    assert layout.install(tree) == 1
    assert layout.install(tree) == 0


@pytest.mark.parametrize("quant_format,layout_type", [
    ("int8_tensorwise", "TensorCoreConvRotW4A4Layout"),
    ("float8_e4m3fn", "TensorCoreFP8E4M3Layout"),
    ("mxfp8", "TensorCoreMXFP8Layout"),
    (None, None),
])
def test_a_mismatched_or_activation_quantized_format_is_not_wrapped(quant_format, layout_type):
    tree = _Tree()
    tree.linear.quant_format, tree.linear.layout_type = quant_format, layout_type
    assert layout.install(tree) == 0


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

def test_comfy_int8_convrot_linear_outside_inference_mode_runs_its_int8_gemm():
    """ComfyUI's own mixed-precision Linear, loaded with an INT8 ConvRot weight.

    Outside inference mode, as on a Worker's sampling thread: the rows
    ``at_stock_rows`` keeps reach comfy-kitchen's INT8 GEMM through ``mm``, and
    so does the stale batch-1 view the old form returned once this module's
    wrapper hands it over as rows, never through ``expand``. Each equals the
    inference-mode result, stock's own, bit for bit. Whether that view
    unwrapped still dequantizes the weight through ``expand`` depends on the
    torch build, so the case only reports it.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    comfy_dir = os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR")
    if not comfy_dir:
        pytest.skip("set COMFY_DIR or COMFYUI_DIR to run the optional current-Comfy case")
    source = Path(__file__).resolve().parents[1] / "src"
    code = r"""
import json, sys
sys.argv = ["int8-convrot-route-test", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
from comfy.quant_ops import QuantizedTensor
from dgx_monarch.adapters import int8_linear_layout as layout
from dgx_monarch.adapters.flux3_adaln_parity import at_stock_rows

seen = []
stock_dispatch = QuantizedTensor.__torch_dispatch__.__func__
def spy(cls, func, types, args=(), kwargs=None):
    seen.append(func.overloadpacket.__name__)
    return stock_dispatch(cls, func, types, args, kwargs)
QuantizedTensor.__torch_dispatch__ = classmethod(spy)

torch.manual_seed(5)
width, out_features, rows, stock_rows = 256, 64, 5, 9
packed = QuantizedTensor.from_float((torch.randn(out_features, width) * 0.02).to(torch.bfloat16),
                                    "TensorWiseINT8Layout", is_weight=True, per_channel=True,
                                    convrot=True, convrot_groupsize=256)
conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
def build():
    linear = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Linear(
        width, out_features, bias=False)
    linear.load_state_dict({"weight": packed._qdata.clone(), "weight_scale": packed._params.scale.clone(),
                            "comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)},
                           strict=False)
    assert linear.quant_format == "int8_tensorwise" and linear.weight._params.convrot
    return linear
def run(linear, x):
    seen.clear()
    out = linear(x)
    return out, list(seen)

plain, wrapped = build(), build()
assert layout.install(wrapped) == 1
piece = torch.randn(1, rows, width).to(torch.bfloat16)
with torch.inference_mode():
    stock, _route = run(plain, piece.clone())
with torch.no_grad():
    kept = at_stock_rows(lambda call: call * 1, piece, stock_rows)
    assert kept.stride() == (rows * width, width, 1), kept.stride()
    out, route = run(plain, kept)
    assert "mm" in route and "expand" not in route, route
    assert torch.equal(out, stock)
    call = torch.zeros(1, stock_rows, width, dtype=torch.bfloat16)
    call[:, :rows] = piece
    stale = call[:, :rows].contiguous()
    _out, unwrapped = run(plain, stale)
    out, route = run(wrapped, stale)
    assert "mm" in route and "expand" not in route, route
    assert out.shape == stock.shape and torch.equal(out, stock)
    if "expand" not in unwrapped:
        print(f"this torch already folds the unwrapped stale view ({unwrapped}); "
              "the wrapper has no expand route to close here")
"""
    env = os.environ | {"PYTHONPATH": f"{comfy_dir}:{source}", "CUDA_VISIBLE_DEVICES": ""}
    completed = subprocess.run([sys.executable, "-c", code], env=env, text=True,
                               capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    if completed.stdout.strip():
        print(completed.stdout.strip())


_ABSENT = "FORMAT-ABSENT-FROM-THIS-COMFYUI"


@pytest.mark.parametrize("quant_format", sorted(_FORMATS))
def test_comfy_linear_of_each_weight_only_format_matches_stock_through_the_wrapper(quant_format):
    """Each format by name: comfy-kitchen packs the weight at the format's own
    width, ComfyUI's loader loads it under the format's name, and the wrapped
    Linear, outside inference mode, gives stock's inference-mode bits through
    ``mm`` for 3-D views whose strides do not chain. The same views unwrapped
    are only reported: whether they take the dequantizing ``expand`` depends
    on the torch build.

    A format of ``FORMATS_AHEAD_OF_A_RELEASE`` skips on a ComfyUI that does
    not list it, and only then: the child process says so in a line of its
    own and exits clean, so a crash is never a skip.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    comfy_dir = os.environ.get("COMFY_DIR") or os.environ.get("COMFYUI_DIR")
    if not comfy_dir:
        pytest.skip("set COMFY_DIR or COMFYUI_DIR to run the optional current-Comfy case")
    source = Path(__file__).resolve().parents[1] / "src"
    layout_type, pack, packed_width = _FORMATS[quant_format]
    code = r"""
import json, sys
name, layout_type, pack, packed_width, absent = json.loads(sys.argv[1])
sys.argv = ["weight-only-format-row", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor
from dgx_monarch.adapters import int8_linear_layout as layout

if name not in QUANT_ALGOS:
    print(absent)
    sys.exit(0)
assert QUANT_ALGOS[name].get("quantize_input", True) is False, QUANT_ALGOS[name]
assert QUANT_ALGOS[name]["comfy_tensor_layout"] == layout_type, QUANT_ALGOS[name]

seen = []
stock_dispatch = QuantizedTensor.__torch_dispatch__.__func__
def spy(cls, func, types, args=(), kwargs=None):
    seen.append(func.overloadpacket.__name__)
    return stock_dispatch(cls, func, types, args, kwargs)
QuantizedTensor.__torch_dispatch__ = classmethod(spy)

torch.manual_seed(5)
width, out_features, rows = 256, 64, 5
packed = QuantizedTensor.from_float((torch.randn(out_features, width) * 0.02).to(torch.bfloat16),
                                    layout_type, **pack)
assert tuple(packed._qdata.shape) == (out_features, packed_width), packed._qdata.shape
params = packed._params
state = {"weight": packed._qdata,
         "comfy_quant": torch.tensor(list(json.dumps({"format": name}).encode()), dtype=torch.uint8)}
if layout_type == "AsymW4A8Int8Layout":
    state["weight_s_rel"], state["weight_s_channel"] = params.scale, params.s_channel
    if params.codebook is not None:
        state["weight_codebook"] = params.codebook
else:
    state["weight_scale"] = params.scale
def build():
    linear = comfy.ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Linear(
        width, out_features, bias=False)
    linear.load_state_dict({key: value.clone() for key, value in state.items()}, strict=False)
    assert (linear.quant_format, linear.layout_type) == (name, layout_type)
    assert isinstance(linear.weight, QuantizedTensor)
    assert tuple(linear.weight._qdata.shape) == (out_features, packed_width)
    return linear
def run(linear, x):
    seen.clear()
    out = linear(x)
    return out, list(seen)

plain, wrapped = build(), build()
assert layout.install(wrapped) == 1, f"the wrapper does not take a {name} Linear"
wide = torch.randn(2, 9, width).to(torch.bfloat16)
views = {"prefix slice": wide[:, :rows], "stepped rows": wide[:, ::2],
         "transposed": wide.transpose(1, 2).contiguous().transpose(1, 2),
         "expanded batch": wide[:1, :rows].expand(2, rows, width)}
for what, view in views.items():
    assert not view.is_contiguous(), what
    with torch.inference_mode():
        stock, _route = run(plain, view.clone())
    with torch.no_grad():
        bare, unwrapped = run(plain, view)
        out, route = run(wrapped, view)
    assert "mm" in route and "expand" not in route, (name, what, route)
    assert out.shape == stock.shape and torch.equal(out, stock), (name, what)
    print(f"{name} {what}: unwrapped {unwrapped}, equal to stock {torch.equal(bare, stock)}")
"""
    # The parent's own path entries follow, so a comfy-kitchen put ahead of the
    # installed one for this run (the release ComfyUI master pins) reaches the child too.
    inherited = os.environ.get("PYTHONPATH")
    child_path = f"{comfy_dir}:{source}" + (f":{inherited}" if inherited else "")
    env = os.environ | {"PYTHONPATH": child_path, "CUDA_VISIBLE_DEVICES": ""}
    argument = json.dumps([quant_format, layout_type, pack, packed_width, _ABSENT])
    completed = subprocess.run([sys.executable, "-c", code, argument], env=env, text=True,
                               capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    if _ABSENT in lines:
        assert quant_format in layout.FORMATS_AHEAD_OF_A_RELEASE, (
            f"this ComfyUI does not list {quant_format}, and FORMATS_AHEAD_OF_A_RELEASE does not excuse it")
        pytest.skip(f"this ComfyUI does not list {quant_format}; ComfyUI commit "
                    f"{layout.FORMATS_AHEAD_OF_A_RELEASE[quant_format][:9]} adds it")
    assert len(lines) >= 4 and all(line.startswith(quant_format) for line in lines[-4:]), completed.stdout
    print(completed.stdout.strip())


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
