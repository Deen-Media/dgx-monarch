"""Two-rank pure-Ulysses Lens equals stock Lens on the real ComfyUI model.

This builds comfy's own `LensTransformer2DModel` at a toy width in float32 on
the CPU, runs its stock `_forward` once, and runs the adapter's bound forward on
two rank threads. Only the sequence-parallel leaves are stubbed: the head
all-to-all (a real head scatter), the sequence all-gather, and xFuser's kernel
(a maskless torch SDPA, which is what xFuser runs on the rig). Everything else
is shipped code, comfy's included.

One recorder sits over `comfy.ops.scaled_dot_product_attention`, the function
stock's `attention_pytorch` calls. The test asserts that every attention call on
each rank reaches that function with exactly stock's queries, keys, values and
bias for that rank's heads, in stock's key order, and that the output equals
stock's with a maximum absolute difference of 0.0. Comfy picks
`attention_pytorch` on an NVIDIA box with torch 2 or later unless it picks
another attention backend: sage, flash, xformers, split, quad or Comfy
Kitchen. On a CPU it needs ``--use-pytorch-cross-attention``, which the
fixture passes with ``--cpu``.

The streams take odd and even row counts each, at batch 1 and 2, with no mask,
with the all-ones keep-mask the GPT-OSS encoder ships for one prompt, and with
a keep-mask whose last two text rows are masked, which stock attends under the
float32 minimum. The image-stream GEMMs still run on half the rows, an
untreated departure; a GPU run shows whether it changes bits. The two sharded
tests therefore need a BLAS that gives the same bits per row at either row
count for the toy widths; a probe checks that first and skips, naming the
departure, where it does not hold. The seam test needs no such BLAS: it pins
the stock behavior the bias treatment copies, an additive bias on every joint
attention, so the comfy-canary job runs it on any runner. The file skips where
ComfyUI is not importable.
"""
from __future__ import annotations

import os
import sys
import threading
import types

import pytest
import torch

import test_flux_pad_exclusion as harness_module
from dgx_monarch.adapters import base, chroma_text, lens
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.lens import LensAdapter

WORLD = 2
HEADS = 4
# (image grid, text rows): nothing pads, odd text, odd image, both odd.
SHAPES = [((2, 4), 4), ((2, 4), 5), ((3, 3), 4), ((3, 3), 5)]
_WHO = threading.local()


def _is_comfy_module(name: str) -> bool:
    return name in ("folder_paths", "comfy") or name.startswith(("comfy.", "comfy_"))


@pytest.fixture(scope="module")
def comfy_lens():
    """Import real `comfy.ops` and `comfy.ldm.lens.model`, then undo it."""
    preserved = {n: m for n, m in sys.modules.items() if _is_comfy_module(n)}
    for name in preserved:
        sys.modules.pop(name, None)
    original_path = list(sys.path)
    comfy_dir = os.path.abspath(os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ComfyUI")))
    if os.path.isdir(comfy_dir) and comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    original_argv = sys.argv
    try:
        sys.argv = ["pytest-lens-comfy-exactness", "--cpu", "--use-pytorch-cross-attention"]
        try:
            options = pytest.importorskip(
                "comfy.options", reason="no ComfyUI on sys.path (comfy-canary job covers it)")
            options.enable_args_parsing()
            ops = pytest.importorskip("comfy.ops")
            attention = pytest.importorskip("comfy.ldm.modules.attention")
            lens_model = pytest.importorskip("comfy.ldm.lens.model")
        finally:
            sys.argv = original_argv
        assert lens_model.optimized_attention is attention.attention_pytorch
        yield ops, lens_model
    finally:
        for name in [n for n in sys.modules if _is_comfy_module(n)]:
            sys.modules.pop(name, None)
        sys.modules.update(preserved)
        sys.path[:] = original_path


# (in, out) of the Linears a shard runs on half the rows: the image stream's
# img_qkv, to_out.0 and img_mlp, and in the negative control the unwrapped text
# stream's txt_qkv, to_add_out and txt_mlp, which share those widths.
_HALF_ROW_LINEARS = ((32, 96), (32, 32), (32, 85), (85, 32))


def _slice_of_joint(rows: torch.Tensor, extra: int) -> torch.Tensor:
    """`rows` as a leading slice of a longer joint tensor, the layout to_out.0 reads."""
    joint = rows.new_zeros(rows.shape[0], rows.shape[1] + extra, rows.shape[2])
    joint[:, :rows.shape[1]] = rows
    return joint[:, :rows.shape[1]]


@pytest.fixture(scope="module")
def row_invariant_blas():
    """Skip unless this CPU's GEMM gives each row the same bits at M and 2M rows.

    Each side keeps its own layout class, as in the model: a contiguous input
    against a contiguous one, and a slice of a joint tensor against a slice.
    """
    generator = torch.Generator().manual_seed(5)
    for width_in, width_out in _HALF_ROW_LINEARS:
        weight = torch.randn(width_out, width_in, generator=generator)
        bias = torch.randn(width_out, generator=generator)
        for batch in (1, 2):
            for full in (4, 5, 8, 9):
                local = -(-full // WORLD)
                rows = torch.randn(batch, full, width_in, generator=generator)
                pairs = ((rows, rows[:, :local].contiguous()),
                         (_slice_of_joint(rows, 5), _slice_of_joint(rows[:, :local], 3)))
                for whole_in, half_in in pairs:
                    whole = torch.nn.functional.linear(whole_in, weight, bias)
                    half = torch.nn.functional.linear(half_in, weight, bias)
                    if not torch.equal(whole[:, :local], half):
                        pytest.skip(
                            "this CPU's BLAS gives a row different bits at "
                            f"{local} and {full} rows for a {width_in}x{width_out} "
                            "Linear: the untreated image-GEMM departure, which "
                            "the bit-for-bit operand proof cannot run past")


def _build(ops, lens_model, state=None):
    model = lens_model.LensTransformer2DModel(
        patch_size=2, in_channels=8, out_channels=2, num_layers=2, attention_head_dim=8,
        num_attention_heads=HEADS, enc_hidden_dim=12, axes_dims_rope=(2, 2, 4),
        multi_layer_encoder_feature=True, selected_layer_index=(0, 1),
        dtype=torch.float32, device="cpu", operations=ops.disable_weight_init)
    if state is None:
        generator = torch.Generator().manual_seed(1009)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(0.3 * torch.randn(parameter.shape, generator=generator))
    else:
        model.load_state_dict(state)
    return model


class _Calls:
    """Every kernel call, per caller, in call order."""

    def __init__(self):
        self.lock = threading.Lock()
        self.by_who: dict[object, list[tuple]] = {}

    def add(self, kind, q, k, v, mask):
        who = getattr(_WHO, "who", "stock")
        entry = (kind, q.detach().clone(), k.detach().clone(), v.detach().clone(),
                 None if mask is None else mask.detach().clone())
        with self.lock:
            self.by_who.setdefault(who, []).append(entry)


def _install(monkeypatch, ops, calls):
    """Record comfy's SDPA, and stub the three sequence-parallel leaves."""
    fabric = harness_module._Fabric(WORLD)
    original_sdpa = ops.scaled_dot_product_attention

    def recording_sdpa(q, k, v, *args, **kwargs):
        calls.add("sdpa", q, k, v, args[0] if args else kwargs.get("attn_mask"))
        return original_sdpa(q, k, v, *args, **kwargs)

    def head_scatter(tensor):
        # (B, L_local, H, D) -> (B, L, H / world, D): this rank's heads, every row.
        rank, parts = harness_module._rank(), fabric.exchange(tensor)
        width = tensor.shape[2] // WORLD
        return torch.cat([part[:, :, rank * width:(rank + 1) * width] for part in parts], dim=1)

    def head_gather(tensor):
        # (B, L, H / world, D) -> (B, L_local, H, D): every head, this rank's rows.
        rank, parts = harness_module._rank(), fabric.exchange(tensor)
        rows = tensor.shape[1] // WORLD
        return torch.cat([part[:, rank * rows:(rank + 1) * rows] for part in parts], dim=2)

    def flash(q, k, v):
        # xFuser's TORCH_FLASH takes no mask; (B, L, H, D) in and out.
        calls.add("xfuser", q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None)
        out = torch.nn.functional.scaled_dot_product_attention(
            *(t.transpose(1, 2) for t in (q, k, v)))
        return out.transpose(1, 2)

    class _StubUSP:
        ulysses_pg = ring_pg = attn_type = attn_processor = None
        q_descale = k_descale = v_descale = None

        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, attn, q_l, k_l, v_l, **kwargs):
            return head_gather(flash(*(head_scatter(t) for t in (q_l, k_l, v_l))))

        @staticmethod
        def ring_attn_fn(q, k, v, **kwargs):
            return flash(q, k, v)

    class _StubAllToAll:
        @staticmethod
        def apply(_group, tensor, scatter_idx, gather_idx):
            return head_scatter(tensor) if (scatter_idx, gather_idx) == (2, 1) else head_gather(
                tensor)

    class _StubSpGroup:
        @staticmethod
        def all_gather(tensor, dim):
            return torch.cat(fabric.exchange(tensor), dim=dim)

    modules = {name: types.ModuleType(name) for name in (
        "xfuser", "xfuser.core", "xfuser.core.distributed",
        "xfuser.core.long_ctx_attention", "yunchang", "yunchang.kernels",
        "yunchang.comm", "yunchang.comm.all_to_all")}
    modules["xfuser.core.distributed"].get_ring_parallel_world_size = lambda: 1
    modules["xfuser.core.distributed"].get_sp_group = lambda: _StubSpGroup()
    modules["xfuser.core.long_ctx_attention"].xFuserLongContextAttention = _StubUSP
    modules["yunchang.kernels"].AttnType = {"TORCH_FLASH": "TORCH_FLASH"}
    modules["yunchang.comm.all_to_all"].SeqAllToAll4D = _StubAllToAll
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ops, "scaled_dot_product_attention", recording_sdpa)
    for module in (base, chroma_text):
        monkeypatch.setattr(module, "sp_world", lambda: WORLD)
    for module in (base, chroma_text, lens):
        monkeypatch.setattr(module, "sp_rank", harness_module._rank)
    return fabric


# No mask; the int64 keep-mask of ones the GPT-OSS encoder ships for one
# prompt; and the same mask with its last two text rows masked.
MASKS = ["none", "ones", "trailing"]


def _inputs(grid, text_rows, batch, masked):
    generator = torch.Generator().manual_seed(31 * text_rows + 7 * batch + grid[0])
    x = torch.randn(batch, 8, *grid, generator=generator)
    timestep = torch.tensor([0.3, 0.7][:batch])
    context = torch.randn(batch, text_rows, 24, generator=generator)
    mask = None
    if masked != "none":
        mask = torch.ones(batch, text_rows, dtype=torch.long)
        if masked == "trailing":
            mask[:, -2:] = 0
    return x, timestep, context, mask


def _sharded(monkeypatch, ops, lens_model, stock, inputs, *, pure_ulysses):
    calls = _Calls()
    fabric = _install(monkeypatch, ops, calls)
    models = [_build(ops, lens_model, stock.state_dict()) for _ in range(WORLD)]
    for model in models:
        LensAdapter().inject_usp(model, InjectionContext(
            topology_sp=WORLD, usp_attention=base.make_usp_attention("TORCH_FLASH"),
            pure_ulysses=pure_ulysses))
    x, timestep, context, mask = inputs
    outputs: list = [None] * WORLD
    errors: list = [None] * WORLD

    def body(rank):
        try:
            harness_module._LOCAL.rank = _WHO.who = rank
            with torch.no_grad():
                outputs[rank] = models[rank]._forward(x, timestep, context, attention_mask=mask)
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(WORLD)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    for error in errors:
        if error is not None:
            raise error
    return outputs, calls


def _stock(ops, lens_model, inputs):
    """Stock's output and its kernel calls, recorded at comfy's SDPA."""
    model, calls = _build(ops, lens_model), _Calls()
    original = ops.scaled_dot_product_attention

    def recording_sdpa(q, k, v, *args, **kwargs):
        calls.add("sdpa", q, k, v, args[0] if args else kwargs.get("attn_mask"))
        return original(q, k, v, *args, **kwargs)

    x, timestep, context, mask = inputs
    ops.scaled_dot_product_attention = recording_sdpa
    try:
        with torch.no_grad():
            expected = model._forward(x, timestep, context, attention_mask=mask)
    finally:
        ops.scaled_dot_product_attention = original
    stock_calls = calls.by_who["stock"]
    assert len(stock_calls) == len(model.transformer_blocks)
    assert all(kind == "sdpa" and bias is not None for kind, *_qkv, bias in stock_calls)
    return model, expected, stock_calls


def _rank_major(image_rows, text_rows):
    """Stock row indices in the order a rank-major gather lists them, pads left out."""
    image_local, text_local = -(-image_rows // WORLD), -(-text_rows // WORLD)
    order = []
    for rank in range(WORLD):
        order += range(rank * image_local, min((rank + 1) * image_local, image_rows))
        order += range(image_rows + rank * text_local,
                       image_rows + min((rank + 1) * text_local, text_rows))
    return order


@pytest.mark.parametrize("masked", MASKS)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_stock_hands_comfy_sdpa_the_joint_bias_the_forward_rebuilds(
    comfy_lens, grid, text_rows, batch, masked
):
    """Stock builds its joint key bias on every call, all zeros for an unmasked
    prompt, and casts it to the query dtype; that keeps torch's SDPA off flash.
    The sharded forward rebuilds it, so a stock change here must fail a test."""
    ops, lens_model = comfy_lens
    inputs = _inputs(grid, text_rows, batch, masked)
    _model, _expected, stock_calls = _stock(ops, lens_model, inputs)
    mask, image_rows = inputs[3], grid[0] * grid[1]
    for _kind, q, _k, _v, bias in stock_calls:
        rebuilt = lens._joint_key_bias(mask, batch, image_rows, text_rows, "cpu").to(q.dtype)
        assert bias.dtype == q.dtype and torch.equal(bias, rebuilt)
    assert bool(stock_calls[0][4].any()) is (masked == "trailing")


@pytest.mark.parametrize("masked", MASKS)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("grid,text_rows", SHAPES)
def test_two_ranks_reproduce_stock_operands_and_output_bit_for_bit(
    monkeypatch, comfy_lens, row_invariant_blas, grid, text_rows, batch, masked
):
    ops, lens_model = comfy_lens
    inputs = _inputs(grid, text_rows, batch, masked)
    stock_model, expected, stock_calls = _stock(ops, lens_model, inputs)
    outputs, sharded = _sharded(monkeypatch, ops, lens_model, stock_model, inputs,
                                pure_ulysses=True)
    width = HEADS // WORLD
    for rank in range(WORLD):
        rank_calls = sharded.by_who[rank]
        assert len(rank_calls) == len(stock_calls)
        for block, (got, want) in enumerate(zip(rank_calls, stock_calls, strict=True)):
            kind, q, k, v, bias = got
            _kind, q_s, k_s, v_s, bias_s = want
            heads = slice(rank * width, (rank + 1) * width)
            # Stock's function, and stock's operands for this rank's heads in
            # stock's key order: every image row, then every text row.
            assert kind == "sdpa", (rank, block)
            for name, value, stock in (("q", q, q_s), ("k", k, k_s), ("v", v, v_s)):
                assert value.dtype == stock.dtype and torch.equal(value, stock[:, heads]), (
                    rank, block, name)
            assert bias is not None and bias.dtype == bias_s.dtype
            assert torch.equal(bias, bias_s), (rank, block)
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0].shape == expected.shape
    assert (outputs[0] - expected).abs().max().item() == 0.0


@pytest.mark.parametrize("grid,text_rows", [((2, 4), 4), ((3, 3), 5)])
def test_the_negative_control_without_pure_ulysses_hands_xfuser_rank_major_keys(
    monkeypatch, comfy_lens, row_invariant_blas, grid, text_rows
):
    """Ring and hybrid keep the maskless xFuser kernel and rank-major keys, so
    the test above would fail on a forward that treated neither."""
    ops, lens_model = comfy_lens
    inputs = _inputs(grid, text_rows, 1, "ones")
    stock_model, _expected, stock_calls = _stock(ops, lens_model, inputs)
    _outputs, sharded = _sharded(monkeypatch, ops, lens_model, stock_model, inputs,
                                 pure_ulysses=False)
    width = HEADS // WORLD
    order = _rank_major(grid[0] * grid[1], text_rows)
    assert order != sorted(order)
    for rank in range(WORLD):
        rank_calls = sharded.by_who[rank]
        assert rank_calls and all(kind == "xfuser" and bias is None
                                  for kind, _q, _k, _v, bias in rank_calls)
        # Block 0's keys are stock's keys in rank-major order; later blocks
        # differ in value as well, since the order changes the rounding.
        keys, stock_keys = rank_calls[0][2], stock_calls[0][2][:, rank * width:(rank + 1) * width]
        assert torch.equal(keys, stock_keys[:, :, order])
        assert not torch.equal(keys, stock_keys)
