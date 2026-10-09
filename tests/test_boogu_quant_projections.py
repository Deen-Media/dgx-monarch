"""Real ComfyUI fp8 instruct Linears run on every instruct row (pure Ulysses).

The shipped Boogu fp8_scaled file loads every instruct Linear through comfy's
mixed-precision Linear with the ``float8_e4m3fn`` format and no
``input_scale``; ``instruct_to_k`` and ``instruct_to_v`` also carry
``full_precision_matrix_mult``. comfy quantizes the input of such a Linear
against the checkpoint's ``input_scale``, or the constant 1.0 when there is
none, so a full-row call quantizes every row as stock does. Here the instruct
Linears of ComfyUI's own ``BooguTransformer2DModel`` (toy size, bfloat16) load
from that state-dict shape; every other layer loads unquantized. At bfloat16
this CPU runs comfy_kitchen's fp8 GEMM itself (``_fp8_scaled_mm``), so the
tests record the rows each fp8 GEMM call receives, beside each instruct
Linear's operand and the output. This CPU's fp8 GEMM gives each row the same
bits at either row count, at this width and at Boogu's 3360 and 13568, so the
operand and GEMM-row checks, not the output, are what fail without the
selection. Layouts whose input scale is not a per-tensor
constant keep the shard. The fixtures come from
tests/test_boogu_comfy_exactness.py and skip where ComfyUI is not importable.
"""
from __future__ import annotations

import json
import sys
import threading

import pytest
import torch

from dgx_monarch.adapters import boogu_ulysses, chroma_text
from dgx_monarch.adapters.base import InjectionContext, make_usp_attention
from dgx_monarch.adapters.boogu import BooguAdapter
from test_boogu_comfy_exactness import (  # noqa: F401  (fixtures re-exported)
    CHANNELS,
    FEATURES,
    HEADS,
    HIDDEN,
    KV_HEADS,
    WORLD,
    _ComfyHarness,
    _inputs,
    _recorder,
    _sdpa,
    _WholeAxisUSP,
    comfy,
    one_thread,
)
from test_flux_pad_exclusion import _LOCAL, _install_stubs, _rank

PATHS = ("img_instruct_attn.processor.instruct_to_q", "img_instruct_attn.processor.instruct_to_k",
         "img_instruct_attn.processor.instruct_to_v", "img_instruct_attn.processor.instruct_out",
         "instruct_feed_forward.linear_1", "instruct_feed_forward.linear_3",
         "instruct_feed_forward.linear_2")
FULL_PRECISION = ("instruct_to_k", "instruct_to_v")
# Odd and even instruct rows, with and without a reference stream.
CASES = [(5, (3, 3), ()), (4, (2, 3), (3,))]


@pytest.fixture
def real(request):
    """comfy's ops, Boogu model and attention modules, from the module fixture
    of tests/test_boogu_comfy_exactness.py."""
    return request.getfixturevalue("comfy")


def _module(block, path):
    module = block
    for name in path.split("."):
        module = getattr(module, name)
    return module


def _quant_conf(path):
    conf = {"format": "float8_e4m3fn"}
    if path.rsplit(".", 1)[-1] in FULL_PRECISION:
        conf["full_precision_matrix_mult"] = True
    return torch.tensor(list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8)


def _fp8_state(state, input_scale=None):
    """The fp8_scaled file's shape: each instruct weight in float8_e4m3fn with
    an F32 per-tensor ``weight_scale`` and a ``comfy_quant`` entry, every other
    tensor in bfloat16."""
    out = {}
    for key, value in state.items():
        layer = key.rsplit(".", 1)[0]
        if key.endswith(".weight") and layer.endswith(PATHS):
            scale = value.abs().amax().float() / torch.finfo(torch.float8_e4m3fn).max
            out[key] = (value.float() / scale).to(torch.float8_e4m3fn)
            out[f"{layer}.weight_scale"] = scale
            out[f"{layer}.comfy_quant"] = _quant_conf(layer)
            if input_scale is not None:
                out[f"{layer}.input_scale"] = torch.tensor(input_scale)
        else:
            out[key] = value.to(torch.bfloat16) if value.is_floating_point() else value
    return out


def _mixed(real):
    return real[0].mixed_precision_ops({}, torch.bfloat16)


def _model(real, state):
    _ops, boogu, _attention = real
    model = boogu.BooguTransformer2DModel(
        patch_size=2, in_channels=CHANNELS, hidden_size=HIDDEN, num_layers=2,
        num_double_stream_layers=2, num_refiner_layers=1, num_attention_heads=HEADS,
        num_kv_heads=KV_HEADS, multiple_of=16, axes_dim_rope=(2, 2, 4), axes_lens=(64, 64, 64),
        instruction_feat_dim=FEATURES, timestep_scale=1000.0, dtype=torch.bfloat16,
        device="cpu", operations=_mixed(real))
    model.load_state_dict(state)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _dense_state(real):
    ops, boogu, _attention = real
    torch.manual_seed(7)
    dense = boogu.BooguTransformer2DModel(
        patch_size=2, in_channels=CHANNELS, hidden_size=HIDDEN, num_layers=2,
        num_double_stream_layers=2, num_refiner_layers=1, num_attention_heads=HEADS,
        num_kv_heads=KV_HEADS, multiple_of=16, axes_dim_rope=(2, 2, 4), axes_lens=(64, 64, 64),
        instruction_feat_dim=FEATURES, timestep_scale=1000.0, dtype=torch.float32,
        device="cpu", operations=ops.disable_weight_init)
    for parameter in dense.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    return dense.state_dict()


def _who():
    return "stock" if threading.current_thread() is threading.main_thread() else _rank()


def _spy(model, sink):
    """Record every instruct Linear's operand, picked by path rather than by
    the selector, so a lost selection reads as a row count."""
    for index, block in enumerate(model.double_stream_layers):
        for path in PATHS:
            module = _module(block, path)
            original = module.forward

            def spy(value, original=original, key=(index, path)):
                sink.append((_who(), key, tuple(value.shape), value.stride(),
                             value.detach().clone()))
                return original(value)
            module.forward = spy


def _gemm_spy(monkeypatch, sink):
    """Record the rows every comfy_kitchen fp8 GEMM call receives."""
    fp8 = sys.modules["comfy_kitchen.tensor.fp8"]
    original = fp8._fp8_scaled_mm

    def record(a, b, *args, **kwargs):
        sink.append((_who(), tuple(a.shape), a.dtype))
        return original(a, b, *args, **kwargs)
    monkeypatch.setattr(fp8, "_fp8_scaled_mm", record)


def _run(monkeypatch, real, text_rows, grid, refs, batch):
    sdpa = _sdpa(real)
    state = _fp8_state(_dense_state(real))
    x, timesteps, context, ref_latents = _inputs(text_rows, grid, refs, batch)
    x, context = x.to(torch.bfloat16), context.to(torch.bfloat16)
    ref_latents = None if ref_latents is None else [r.to(torch.bfloat16) for r in ref_latents]
    operands: list = []
    gemms: list = []
    _gemm_spy(monkeypatch, gemms)
    stock = _model(real, state)
    _spy(stock, operands)
    expected = stock.forward(x, timesteps, context, text_rows, ref_latents=ref_latents,
                             transformer_options={"optimized_attention_override":
                                                  _recorder(sdpa, {0: []})})

    harness = _ComfyHarness(WORLD, sdpa)
    _install_stubs(monkeypatch, harness)
    monkeypatch.setattr(sys.modules["xfuser.core.long_ctx_attention"],
                        "xFuserLongContextAttention", _WholeAxisUSP)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: WORLD)
    kernel = make_usp_attention("TORCH_FLASH")
    models = [_model(real, state) for _ in range(WORLD)]
    for model in models:
        _spy(model, operands)
        BooguAdapter().inject_usp(model, InjectionContext(
            topology_sp=WORLD, usp_attention=kernel, pure_ulysses=True))
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD
    replicated: dict[int, list] = {rank: [] for rank in range(WORLD)}

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank].forward(
                x, timesteps, context, text_rows, ref_latents=ref_latents,
                transformer_options={"optimized_attention_override": _recorder(sdpa, replicated)})
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for error in errors:
        if error is not None:
            raise error
    return expected, outputs, operands, gemms, models


def _real_strides(stride, shape):
    return tuple(step for step, size in zip(stride, shape, strict=True) if size != 1)


def _by(records, who):
    return [record[1:] for record in records if record[0] == who]


@pytest.mark.usefixtures("one_thread")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,refs", CASES)
def test_fp8_instruct_linears_see_every_instruct_row_in_stock_layout(
    monkeypatch, real, text_rows, grid, refs, batch
):
    _expected, _outputs, operands, _gemms, models = _run(
        monkeypatch, real, text_rows, grid, refs, batch)
    # The shipped file's two entry shapes: fp8 with no input_scale, and the
    # same with full_precision_matrix_mult for instruct_to_k and _to_v.
    for model in models:
        for block in model.double_stream_layers:
            for path in PATHS:
                module = _module(block, path)
                assert module.layout_type == "TensorCoreFP8E4M3Layout", path
                assert getattr(module, "input_scale", None) is None, path
                assert module._full_precision_mm is path.endswith(FULL_PRECISION), path
    stock = _by(operands, "stock")
    assert len(stock) == 2 * len(PATHS)
    image = grid[0] * grid[1] + sum(refs)
    for rank in range(WORLD):
        got = _by(operands, rank)
        assert [key for key, *_ in got] == [key for key, *_ in stock], rank
        for (key, shape, stride, value), (_key, want_shape, want_stride, want) in zip(
                got, stock, strict=True):
            assert shape == want_shape and shape[:2] == (batch, text_rows), (rank, key)
            # A size-1 batch axis has no memory step, so its stride is free.
            assert _real_strides(stride, shape) == _real_strides(want_stride, shape), (rank, key)
            if key[1].endswith("instruct_out"):
                assert stride == ((text_rows + image) * HIDDEN, HIDDEN, 1), (rank, key)
            assert torch.equal(value, want), (rank, key)


@pytest.mark.usefixtures("one_thread")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,refs", CASES)
def test_fp8_forward_equals_stock_and_each_fp8_gemm_takes_stock_rows(
    monkeypatch, real, text_rows, grid, refs, batch
):
    """The five instruct Linears without ``full_precision_matrix_mult`` reach
    the fp8 GEMM, at stock's ``batch * text_rows`` rows on every rank; the
    other layers are unquantized here, so no other fp8 GEMM runs."""
    expected, outputs, _operands, gemms, _models = _run(
        monkeypatch, real, text_rows, grid, refs, batch)
    stock = _by(gemms, "stock")
    assert len(stock) == 5 * 2
    for shape, dtype in stock:
        assert shape[0] == batch * text_rows and dtype == torch.float8_e4m3fn
    for rank in range(WORLD):
        assert _by(gemms, rank) == stock, rank
        assert not torch.isnan(outputs[rank]).any()
        assert outputs[rank].shape == expected.shape
        assert (outputs[rank].float() - expected.float()).abs().max().item() == 0.0, rank
        assert torch.equal(outputs[rank], expected), rank


def _entry(fmt, out_features, in_features):
    """A ``fmt`` state-dict entry with no ``input_scale``, in the shape comfy's
    loader reads (``_load_quantized_module``)."""
    tensors = {
        "nvfp4": {"weight": torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
                  "weight_scale": torch.ones(out_features, in_features // 16).to(
                      torch.float8_e4m3fn),
                  "weight_scale_2": torch.tensor(1.0)},
        "mxfp8": {"weight": torch.zeros(out_features, in_features).to(torch.float8_e4m3fn),
                  "weight_scale": torch.full((out_features, in_features // 32), 127,
                                             dtype=torch.uint8)},
        "int8_tensorwise": {"weight": torch.zeros(out_features, in_features, dtype=torch.int8),
                            "weight_scale": torch.tensor(1.0)},
    }[fmt]
    tensors["comfy_quant"] = torch.tensor(list(json.dumps({"format": fmt}).encode("utf-8")),
                                          dtype=torch.uint8)
    return tensors


def _fp8_block(real, input_scale=None, others=()):
    """A real double block loaded through comfy's loader, its instruct Linears
    in fp8; each ``(path, format)`` in ``others`` loads in that format."""
    ops, boogu, _attention = real

    def build(operations, dtype):
        return boogu.BooguDoubleStreamBlock(HIDDEN, HEADS, KV_HEADS, 16, None, 1e-5,
                                            dtype=dtype, device="cpu", operations=operations)
    dense = build(ops.disable_weight_init, torch.float32)
    for parameter in dense.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    state = _fp8_state(dense.state_dict(), input_scale=input_scale)
    for path, fmt in others:
        for key in [key for key in state if key.startswith(f"{path}.")]:
            del state[key]
        linear = _module(dense, path)
        state.update({f"{path}.{name}": value for name, value in
                      _entry(fmt, linear.out_features, linear.in_features).items()})
    block = build(_mixed(real), torch.bfloat16)
    block.load_state_dict(state)
    return block


@pytest.mark.parametrize("input_scale", [None, 0.05])
def test_real_fp8_block_selects_all_seven_instruct_linears(real, input_scale):
    block = _fp8_block(real, input_scale)
    modules = [_module(block, path) for path in PATHS]
    for path, module in zip(PATHS, modules, strict=True):
        assert not isinstance(module, torch.nn.Linear), path
        assert module.layout_type == "TensorCoreFP8E4M3Layout", path
        assert (getattr(module, "input_scale", None) is None) is (input_scale is None), path
        assert module._full_precision_mm is (path.rsplit(".", 1)[-1] in FULL_PRECISION), path
    selected = boogu_ulysses.instruct_projections(block)
    assert [module for module, _ in selected] == modules
    assert [joint for _, joint in selected] == [False, False, False, True, False, False, False]


def test_layouts_whose_input_scale_is_not_a_constant_keep_the_shard(real):
    """Every comfy format's layout, set on a real loaded fp8 Linear: only the
    two fp8 formats are selected. NVFP4 takes an amax over the whole input when
    it has no ``input_scale``, MXFP8 scales per block, and a layout the house
    table does not name is never selected."""
    quant_algos = sys.modules["comfy.quant_ops"].QUANT_ALGOS
    layouts = {name: conf["comfy_tensor_layout"] for name, conf in quant_algos.items()}
    assert {"float8_e4m3fn", "float8_e5m2", "nvfp4"} <= set(layouts)
    admitted = {"float8_e4m3fn", "float8_e5m2"}
    for name, layout in [*layouts.items(), ("unknown", "SomeFutureLayout")]:
        block = _fp8_block(real)
        target = _module(block, PATHS[0])
        target.layout_type = layout
        kept = [module for module, _ in boogu_ulysses.instruct_projections(block)]
        assert (target in kept) is (name in admitted), name
        assert kept[-6:] == [_module(block, path) for path in PATHS[1:]], name


def test_real_dynamic_scale_layouts_keep_the_shard(real):
    """NVFP4 with no ``input_scale`` takes an amax over the whole input, so a
    full-row call would change its scale. It, MXFP8 and int8 load here through
    comfy's loader beside the fp8 Linears, and only the fp8 ones are selected."""
    quant_algos = sys.modules["comfy.quant_ops"].QUANT_ALGOS
    others = tuple((path, fmt) for path, fmt in (
        (PATHS[0], "nvfp4"), (PATHS[4], "mxfp8"), (PATHS[6], "int8_tensorwise"))
        if fmt in quant_algos)
    assert others[0] == (PATHS[0], "nvfp4")
    block = _fp8_block(real, others=others)
    for path, fmt in others:
        module = _module(block, path)
        assert module.layout_type == quant_algos[fmt]["comfy_tensor_layout"], fmt
        assert getattr(module, "input_scale", None) is None, fmt
    kept = [module for module, _ in boogu_ulysses.instruct_projections(block)]
    skipped = {path for path, _fmt in others}
    assert kept == [_module(block, path) for path in PATHS if path not in skipped]
