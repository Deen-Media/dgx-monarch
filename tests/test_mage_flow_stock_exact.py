"""Real ComfyUI Mage Flow: a two-rank pure-Ulysses forward produces exactly the stock output.

This builds comfy's own `MageFlowTransformer2DModel` at toy sizes in float32,
runs its stock `_forward` once, then runs the adapter's bound forward on two
rank threads through the real `base.make_usp_attention` dispatch and the real
full-axis call. Only the all-to-all, the gather and the kernel are stubbed (the
harness in tests/test_flux_pad_exclusion.py). The stub kernel calls the same
attention function stock's override hook handed it, on operands laid out with
stock's strides, so a difference anywhere upstream shows as a difference here.

Four things must hold for every shape: each attention kernel call receives
exactly stock's q, k and v (values, key order and shape); each wrapped text
Linear receives exactly stock's input (all text rows, stock's strides, the
same values); each block makes one text gather per wrapped input, four when
all six text Linears are wrapped, because the three `add_*` projections share
one input; and the output equals stock's with a max abs difference of 0.0.

This CPU's float32 GEMM gives the same bits at half the rows, so the output
check alone cannot see the text-row departure; the Linear operand check is what
fails without the full-row text projections. The image-stream Linears still
run on this rank's rows until a GPU probe shows whether that is exact; the test
pins it, so wrapping them must update the test. The same checks run on a model
built from comfy's mixed-precision Linear, the class a quantized file loads
every layer through: a layer it holds unquantized takes the full-row path, and
one comfy's loader gives a quantized `layout_type` keeps this rank's rows. Skips
where ComfyUI is not importable; comfy.cli_args parses argv on import, so the
fixture forces ``--cpu``.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import pytest
import torch

from dgx_monarch.adapters import base, chroma_text, mage_flow
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.mage_flow import MageFlowAdapter
from test_flux_pad_exclusion import _LOCAL, _Harness, _install_stubs, _rank

WORLD = 2
TEXT_LINEARS = ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
                "attn.to_add_out", "txt_mlp.net.0.proj", "txt_mlp.net.2")
SHARDED_IMAGE = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                 "img_mlp.net.0.proj", "img_mlp.net.2")
# Reference latent grids: one gives 6 extra image rows, two give 15.
REFERENCE_GRIDS = {0: [], 1: [(2, 3)], 2: [(2, 3), (3, 3)]}


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_mage():
    """Import real `comfy.ops` and the Mage model, then undo it."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-mage-flow-stock-exact", "--cpu"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            model = pytest.importorskip("comfy.ldm.mage_flow.model")
        finally:
            sys.argv = original_argv
        yield ops, model
    finally:
        for name in [n for n in sys.modules if _is_comfy_module(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


def _recording_ops(ops, family):
    """comfy's Linear family with a Linear that records what it is handed.

    The record sits in the class forward, which is what the full-row wrapper
    calls with the gathered rows, so it sees the GEMM's real operand. "dense"
    is disable_weight_init, the bf16 checkpoint's family; "mixed" is
    mixed_precision_ops with no quantization config, which loads every layer
    the way it loads the unquantized layers of a quantized checkpoint.
    """
    family_ops = (ops.disable_weight_init if family == "dense"
                  else ops.mixed_precision_ops({}, torch.float32))

    class Ops(family_ops):
        class Linear(family_ops.Linear):
            def forward(self, value, *args, **kwargs):
                self.seen.append((tuple(value.shape), value.stride(), value.detach().clone()))
                return super().forward(value, *args, **kwargs)
    return Ops


def _build(mage, operations):
    return mage.MageFlowTransformer2DModel(
        in_channels=4, out_channels=4, num_layers=2, attention_head_dim=8,
        num_attention_heads=2, joint_attention_dim=16, axes_dims_rope=(2, 2, 4),
        dtype=torch.float32, device=torch.device("cpu"), operations=operations)


def _state(comfy, quantized=()):
    """Seeded float32 weights; each block path in `quantized` stored as an fp8
    file stores it: an e4m3 weight, its scale, a static input scale and the
    `comfy_quant` entry comfy's loader reads to set `layout_type`. The static
    input scale makes the layer's input quantization independent of its row
    count, so the forward can still be compared with stock's output for exact equality."""
    ops, mage = comfy
    seed = _build(mage, ops.disable_weight_init)
    generator = torch.Generator().manual_seed(20261006)
    with torch.no_grad():
        for parameter in seed.parameters():
            parameter.copy_(0.1 * torch.randn(parameter.shape, generator=generator))
    state = dict(seed.state_dict())
    config = torch.tensor(list(json.dumps({"format": "float8_e4m3fn"}).encode()), dtype=torch.uint8)
    for index in range(len(seed.transformer_blocks)):
        for path in quantized:
            prefix = f"transformer_blocks.{index}.{path}."
            weight = state[prefix + "weight"]
            scale = weight.abs().amax() / 448.0
            state[prefix + "weight"] = (weight / scale).to(torch.float8_e4m3fn)
            state[prefix + "weight_scale"] = scale
            state[prefix + "input_scale"] = torch.tensor(16.0 / 448.0)
            state[prefix + "comfy_quant"] = config
    return state


def _model(comfy, family, state):
    ops, mage = comfy
    recording = _recording_ops(ops, family)
    model = _build(mage, recording)
    model.load_state_dict(state)
    for module in model.modules():
        if isinstance(module, recording.Linear):
            module.seen = []
    return model


def _real_strides(shape, stride):
    """A size-1 axis keeps the padded gather's stride, which no GEMM reads."""
    return tuple(step for step, size in zip(stride, shape, strict=True) if size != 1)


def _linear(block, path):
    module = block
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


class _StockKernel(_Harness):
    """Attends with stock's own function on operands laid out as stock's were."""

    def __init__(self, world, func, stock_calls):
        super().__init__(world, strict=False)
        self.func, self.stock_calls = func, stock_calls
        self.lock = threading.Lock()
        self.seen: dict[int, list] = {}

    def attend(self, entry, q, k, v):
        rank = _rank()
        with self.lock:
            calls = self.seen.setdefault(rank, [])
            index = len(calls)
        reference = self.stock_calls[index]
        heads = q.shape[2]
        operands = []
        for value, stock in zip((q, k, v), reference, strict=True):
            laid = torch.empty_strided(stock.shape, stock.stride(), dtype=value.dtype)
            laid.copy_(value.transpose(1, 2))
            operands.append(laid)
        calls.append((entry, *operands))
        out = self.func(*operands, heads, None, skip_reshape=True)
        return out.reshape(q.shape[0], q.shape[1], heads, q.shape[3])


def _inputs(text_rows, grid, batch, references, mask):
    generator = torch.Generator().manual_seed(text_rows * 100 + grid[0] * 10 + grid[1] + batch)
    x = torch.randn(batch, 4, *grid, generator=generator)
    timestep = torch.tensor([0.25, 0.75])[:batch]
    context = torch.randn(batch, text_rows, 16, generator=generator)
    refs = [torch.randn(batch, 4, *size, generator=generator)
            for size in REFERENCE_GRIDS[references]] or None
    kwargs = {"ref_latents": refs}
    if mask:
        # The all-valid additive form stock accepts: every key stays visible.
        kwargs["attention_mask"] = torch.zeros(batch, text_rows)
    return x, timestep, context, kwargs


def _stock(comfy, family, inputs, state):
    model = _model(comfy, family, state)
    calls, kernel = [], []

    def record(func, *args, **kwargs):
        kernel.append(func)
        calls.append(tuple(value.detach().clone() for value in args[:3]))
        return func(*args, **kwargs)

    x, timestep, context, kwargs = inputs
    out = model._forward(x, timestep, context,
                         transformer_options={"optimized_attention_override": record}, **kwargs)
    return model, out.detach(), calls, kernel[0]


def _sharded(monkeypatch, comfy, family, inputs, state, func, stock_calls):
    monkeypatch.setattr(mage_flow, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: WORLD)
    harness = _StockKernel(WORLD, func, stock_calls)
    harness.gathers = []

    def counted(value, length, dim=1):
        harness.gathers.append(_rank())
        return base.sp_gather(value, length, dim=dim)
    # mage_flow imports sp_gather by value; every gather the forward makes,
    # its text gathers included, goes through that name.
    monkeypatch.setattr(mage_flow, "sp_gather", counted)
    _install_stubs(monkeypatch, harness)
    attention = base.make_usp_attention("TORCH_FLASH")
    models = [_model(comfy, family, state) for _ in range(WORLD)]
    for model in models:
        MageFlowAdapter().inject_usp(model, InjectionContext(
            topology_sp=WORLD, usp_attention=attention, pure_ulysses=True))
    x, timestep, context, kwargs = inputs
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            _LOCAL.rank = rank
            with torch.no_grad():
                outputs[rank] = models[rank]._forward(x, timestep, context, **kwargs)
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
    return models, outputs, harness


def _check(monkeypatch, comfy, text_rows, grid, batch, references, *, mask=False,
           family="dense", quantized=()):
    inputs = _inputs(text_rows, grid, batch, references, mask)
    state = _state(comfy, quantized)
    with torch.no_grad():
        stock, expected, stock_calls, func = _stock(comfy, family, inputs, state)
    models, outputs, harness = _sharded(
        monkeypatch, comfy, family, inputs, state, func, stock_calls)
    image_rows = grid[0] * grid[1] + sum(h * w for h, w in REFERENCE_GRIDS[references])

    # Every kernel call: stock's q, k, v, in stock's order, laid out as stock's.
    blocks = len(stock.transformer_blocks)
    assert len(stock_calls) == blocks
    for rank in range(WORLD):
        calls = harness.seen[rank]
        assert len(calls) == blocks
        for (entry, *operands), reference in zip(calls, stock_calls, strict=True):
            assert entry == "exact"
            for got, want in zip(operands, reference, strict=True):
                assert got.shape == want.shape
                assert torch.equal(got, want)

    # One text gather per block for the add_* input its wrapped projections
    # share, one for each other wrapped text Linear, then the output gather.
    wrapped = [path for path in TEXT_LINEARS if path not in quantized]
    shared = any(path.startswith("attn.add_") for path in wrapped)
    per_block = shared + sum(not path.startswith("attn.add_") for path in wrapped)
    assert sorted(harness.gathers) == sorted(list(range(WORLD)) * (per_block * blocks + 1))

    # Every wrapped text Linear: stock's full rows, strides and values. A
    # quantized one: this rank's rows, with stock's values on its real rows.
    local = -(-text_rows // WORLD)
    for index, stock_block in enumerate(stock.transformer_blocks):
        for rank, model in enumerate(models):
            block = model.transformer_blocks[index]
            selected = [module for module, _ in mage_flow._text_modules(block)]
            for path in TEXT_LINEARS:
                module = _linear(block, path)
                assert isinstance(module, torch.nn.Linear) is (family == "dense"), path
                assert "forward" not in vars(module), (index, path)
                (want_shape, want_stride, want), = _linear(stock_block, path).seen
                (got_shape, got_stride, got), = module.seen
                if path in quantized:
                    assert module.layout_type == "TensorCoreFP8E4M3Layout", path
                    assert all(other is not module for other in selected), (index, path)
                    real = max(0, min(local, text_rows - rank * local))
                    assert got_shape[1] == local, (index, path)
                    assert torch.equal(got[:, :real], want[:, rank * local:rank * local + real])
                    continue
                assert getattr(module, "layout_type", None) is None, path
                assert any(other is module for other in selected), (index, path)
                assert got_shape == want_shape, (index, path)
                assert _real_strides(got_shape, got_stride) == _real_strides(
                    want_shape, want_stride), (index, path)
                assert torch.equal(got, want), (index, path)
            # Held for a GPU probe: these still see this rank's rows.
            for path in SHARDED_IMAGE:
                (got_shape, _stride, _value), = _linear(block, path).seen
                assert got_shape[1] == -(-image_rows // WORLD), (index, path)

    for out in outputs:
        assert out.shape == expected.shape
        assert (out - expected).abs().max().item() == 0.0
        assert torch.equal(out, expected)


@pytest.mark.parametrize("references", [0, 1, 2])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid", [(3, 5), (4, 4)])
@pytest.mark.parametrize("text_rows", [5, 6])
def test_two_rank_forward_equals_stock_bit_for_bit(
    monkeypatch, comfy_mage, text_rows, grid, batch, references
):
    """Odd and even text (5, 6) and image (15, 16) rows; references take the
    image stream to 21, 22, 30 or 31 rows, so every stream meets both parities."""
    _check(monkeypatch, comfy_mage, text_rows, grid, batch, references)


@pytest.mark.parametrize("batch", [1, 2])
def test_an_all_valid_mask_matches_stock_on_this_cpu(monkeypatch, comfy_mage, batch):
    """Stock hands its kernel an all-zero additive bias; the adapter drops it.

    The keys and the output still match here because this CPU's SDPA returns
    the same bits with and without a zero bias. A GPU picks another kernel when
    any mask is present, so this case says nothing about hardware; comfy's Mage
    text encoder drops an all-valid mask before the model sees it.
    """
    _check(monkeypatch, comfy_mage, 5, (3, 5), batch, 1, mask=True)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,references", [(5, (3, 5), 1), (6, (4, 4), 0)])
def test_unquantized_layers_of_a_quantized_checkpoint_match_stock(
    monkeypatch, comfy_mage, text_rows, grid, references, batch
):
    """comfy's mixed-precision Linear, the class an MXFP8, NVFP4 or FP8 mixed
    file loads its BF16 layers through, takes the same full-row path."""
    _check(monkeypatch, comfy_mage, text_rows, grid, batch, references, family="mixed")


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,references", [(5, (3, 5), 1), (6, (4, 4), 0)])
def test_a_quantized_text_linear_keeps_its_shard(
    monkeypatch, comfy_mage, text_rows, grid, references, batch
):
    """comfy's loader reads the fp8 `comfy_quant` entry and names the layout.
    Those two Linears keep this rank's rows and make no gather; `add_q_proj`
    and `add_v_proj` still share one gather, and the other text Linears of
    the same mixed-precision class run on full rows."""
    _check(monkeypatch, comfy_mage, text_rows, grid, batch, references, family="mixed",
           quantized=("attn.add_k_proj", "txt_mlp.net.2"))
