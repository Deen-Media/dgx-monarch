"""adapters/flux3_adaln_parity.py: each span piece's fused norm at stock's row count.

comfy-kitchen's fused AdaLN picks its CUDA kernel by a call's kernel rows
(batch times rows), and the kernels round a few outputs one bf16 step apart.
The CPU half pins the padding arithmetic, the stock row count each clipped
span carries, and the bound forward, with a stand-in block whose forward makes
stock's norm calls through a module global here and a fake op that records
each call's rows. The GPU half is the seam test: on a CUDA device it shows the
short call takes the other kernel and that the padded call returns stock's
bits. tests/test_flux3_comfy_exactness.py runs the same route through
ComfyUI's own FLUX.3 blocks.
"""
from __future__ import annotations

from importlib import metadata
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.flux3_adaln_parity import (
    at_stock_rows,
    bind_adaln_parity,
    loads_norm_global,
    pad_rows,
)
from dgx_monarch.adapters.flux3_packing import shard_stream, stock_norm_rows

D = 8
CALLS: list[int] = []


def modulated_norm(x, norm, scale, shift, modulation_dims=None):
    """The stand-in for stock's fused norm: row-local, and it records its rows."""
    CALLS.append(x.shape[1])
    return x * (1 + scale) + shift


class _Block(torch.nn.Module):
    """Stock ``Block.forward``'s norm calls and gate writes, nothing else."""

    def forward(self, x, spans, pe, transformer_options=None):
        mod = spans[0][2]
        x_mod = modulated_norm(x, None, mod.scale, mod.shift)
        for start, end, mod in spans[1:]:
            x_mod[:, start:end] = modulated_norm(x[:, start:end], None, mod.scale, mod.shift)
        for start, end, mod in spans:
            x[:, start:end].addcmul_(x_mod[:, start:end], mod.gate)
        return x


class _ForeignNormBlock(_Block):
    """A block that also normalizes with a modulation no span carries."""

    def forward(self, x, spans, pe, transformer_options=None):
        x = super().forward(x, spans, pe, transformer_options)
        stray = torch.ones(1, 1, x.shape[-1])
        return modulated_norm(x, None, stray, stray)


class _NoNormBlock(torch.nn.Module):
    def forward(self, x, spans, pe, transformer_options=None):
        return x


def _mod(value: float) -> SimpleNamespace:
    tensor = torch.full((1, 1, D), value)
    return SimpleNamespace(scale=tensor * 0.5, shift=tensor * 0.25, gate=tensor)


def _spans_and_input(seed: int = 0):
    """Rank 1 of the uly2 one-reference edit, at 1/1024 of its rows."""
    torch.manual_seed(seed)
    spans = [(0, 0, _mod(0.1)), (0, 4, _mod(0.2)), (4, 6, _mod(0.3))]
    return spans, torch.randn(2, 6, D)


@pytest.fixture(autouse=True)
def _clear_calls():
    CALLS.clear()
    yield
    CALLS.clear()


def test_pad_rows_fills_a_piece_to_stocks_call_and_refuses_a_piece_it_cannot_hold():
    assert pad_rows(40, 4096) == 4056
    assert pad_rows(552, 1104) == 552
    assert pad_rows(4096, 4096) == 0
    for piece, stock in ((0, 10), (11, 10), (-1, 10)):
        with pytest.raises(ValueError, match="cannot sit"):
            pad_rows(piece, stock)


def test_a_piece_runs_in_a_stock_sized_call_of_zero_rows_and_keeps_its_own_rows():
    seen: list[torch.Tensor] = []

    def op(call):
        seen.append(call.clone())
        return call * 3 + 1

    base_rows = torch.randn(2, 9, D)
    piece = base_rows[:, 2:5]          # a strided view at batch 2, as stock slices a span
    assert not piece.is_contiguous()
    out = at_stock_rows(op, piece, 7)
    (call,) = seen
    assert call.shape == (2, 7, D)
    assert torch.equal(call[:, :3], piece)
    assert torch.count_nonzero(call[:, 3:]) == 0
    assert out.shape == piece.shape and out.is_contiguous()
    assert torch.equal(out, piece * 3 + 1)


def test_a_piece_of_stocks_size_is_handed_to_the_op_as_it_is():
    seen: list[torch.Tensor] = []
    x = torch.randn(1, 5, D)
    out = at_stock_rows(lambda call: seen.append(call) or call, x, 5)
    assert seen[0] is x and out is x


@pytest.mark.parametrize("batch", [1, 2])
def test_the_kept_rows_have_a_fresh_tensors_strides_at_every_batch(batch):
    """At batch 1 the kept rows are a prefix of the call. Torch counts that
    prefix as contiguous, so ``.contiguous()`` keeps the call's leading stride,
    and a linear after it cannot fold the rows into one matrix. The result
    carries a fresh ``[B, n, D]`` tensor's strides instead: a view of the op's
    output at batch 1, one copy at batch 2."""
    outputs: list[torch.Tensor] = []

    def op(call):
        outputs.append(call * 2)
        return outputs[-1]

    piece = torch.randn(batch, 3, D)
    out = at_stock_rows(op, piece, 7)
    assert out.shape == (batch, 3, D) and out.stride() == (3 * D, D, 1)
    assert torch.equal(out, piece * 2)
    shared = out.untyped_storage().data_ptr() == outputs[0].untyped_storage().data_ptr()
    assert shared is (batch == 1)
    stale = outputs[0][:, :3].contiguous()
    assert stale.stride(0) == (7 * D if batch == 1 else 3 * D), (
        "torch now gives a batch-1 prefix slice fresh strides on .contiguous(); "
        "this test no longer shows the layout the reshape avoids")


def _dispatch_spy(monkeypatch, quantized_tensor) -> list[str]:
    """The aten ops ``quantized_tensor`` dispatches, recorded as it runs them."""
    seen: list[str] = []
    stock = quantized_tensor.__torch_dispatch__.__func__

    def spy(cls, func, types, args=(), kwargs=None):
        seen.append(func.overloadpacket.__name__)
        return stock(cls, func, types, args, kwargs)

    monkeypatch.setattr(quantized_tensor, "__torch_dispatch__", classmethod(spy))
    return seen


@pytest.mark.parametrize("batch", [1, 2])
def test_an_int8_convrot_linear_after_the_kept_rows_runs_its_int8_gemm(monkeypatch, batch):
    """Outside inference mode, as on a Worker's sampling thread, F.linear on an
    INT8 ConvRot weight reaches comfy-kitchen's INT8 GEMM through ``mm`` when the
    activation's leading strides chain, and dequantizes the weight through
    ``expand`` when they do not. The kept rows take ``mm`` and match the
    inference-mode result, which is stock's route, bit for bit. The old
    ``.contiguous()`` form at batch 1 is the control: it takes ``expand``."""
    _comfy_kitchen()
    from comfy_kitchen.tensor import QuantizedTensor

    torch.manual_seed(3)
    width, rows, stock_rows = 256, 5, 9
    weight = QuantizedTensor.from_float(
        (torch.randn(64, width) * 0.02).to(torch.bfloat16), "TensorWiseINT8Layout",
        is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256)
    seen = _dispatch_spy(monkeypatch, QuantizedTensor)
    piece = torch.randn(batch, rows, width).to(torch.bfloat16)
    with torch.inference_mode():
        stock = torch.nn.functional.linear(piece.clone(), weight)
    with torch.no_grad():
        kept = at_stock_rows(lambda call: call * 1, piece, stock_rows)
        seen.clear()
        out = torch.nn.functional.linear(kept, weight)
        assert seen == ["t", "mm"], seen
        assert torch.equal(out, stock)
        if batch == 1:
            call = torch.zeros(batch, stock_rows, width, dtype=torch.bfloat16)
            call[:, :rows] = piece
            seen.clear()
            torch.nn.functional.linear(call[:, :rows].contiguous(), weight)
            assert seen == ["t", "expand"], (
                f"the stale batch-1 layout now dispatches {seen}; the control no longer "
                "shows the route the reshape avoids")


def test_stock_norm_rows_names_stocks_call_for_each_span_of_the_uly2_edit():
    """1024 square, one 1 MP reference, 80 text rows: the stream holds 8272."""
    spans = [(0, 4096, "img"), (4096, 8192, "ref"), (8192, 8272, "txt")]
    # Rank 0: the image, then 40 reference rows that stock normalizes in its
    # 4096-row reference call.
    assert stock_norm_rows(spans, 0, 4136, 8272) == (8272, 4096)
    # Rank 1: the empty image span still names the whole stream's call.
    assert stock_norm_rows(spans, 4136, 4136, 8272) == (8272, 4096, 80)
    # The image phase: one span, its call the image rows.
    assert stock_norm_rows([(0, 4096, "early")], 2048, 2048, 4096) == (4096,)


@pytest.mark.parametrize("rank", [0, 1])
def test_a_shard_carries_one_stock_row_count_for_each_clipped_span(monkeypatch, rank):
    monkeypatch.setattr(base, "sp_world", lambda: 2)
    monkeypatch.setattr(base, "sp_rank", lambda: rank)
    spans = [(0, 9, "img"), (9, 15, "ref"), (15, 19, "txt")]   # 19 rows pad to 20
    shard = shard_stream(torch.randn(1, 19, D), torch.zeros(19, 4), spans,
                         lambda ids: ids.unsqueeze(0), "test")
    assert len(shard.norm_rows) == len(shard.spans)
    assert shard.norm_rows == ((19, 6) if rank == 0 else (19, 6, 4))


def test_the_bound_forward_launches_every_norm_call_at_stocks_rows():
    spans, x = _spans_and_input()
    expected = _Block()(x.clone(), spans, None)
    assert CALLS == [6, 4, 2]                     # the rank's own call sizes
    CALLS.clear()
    block = _Block()
    assert bind_adaln_parity(block)
    out = block(x.clone(), spans, None, {}, stock_norm_rows=(12, 8, 2))
    # Stock's sizes: the stream, the 8-row reference span, the 2-row text span.
    assert CALLS == [12, 8, 2]
    assert torch.equal(out, expected)


def test_without_stock_row_counts_the_bound_forward_is_the_class_forward():
    spans, x = _spans_and_input()
    block = _Block()
    bind_adaln_parity(block)
    block(x.clone(), spans, None, {})
    assert CALLS == [6, 4, 2]


def test_a_repeat_bind_replaces_and_leaves_class_and_module_untouched():
    spans, x = _spans_and_input()
    class_forward, module_norm = vars(_Block)["forward"], globals()["modulated_norm"]
    block = _Block()
    for _ in range(3):
        bind_adaln_parity(block)
    block(x.clone(), spans, None, {}, stock_norm_rows=(12, 8, 2))
    assert CALLS == [12, 8, 2]                    # one route, not three stacked
    assert vars(_Block)["forward"] is class_forward
    assert globals()["modulated_norm"] is module_norm
    assert "forward" not in vars(_Block())        # only the bound instance changed


def test_a_norm_call_no_span_carries_raises_instead_of_passing_through():
    spans, x = _spans_and_input()
    block = _ForeignNormBlock()
    assert bind_adaln_parity(block)
    with pytest.raises(AssertionError, match="not one of its spans' calls"):
        block(x.clone(), spans, None, {}, stock_norm_rows=(12, 8, 2))


def test_counts_that_do_not_match_the_spans_raise():
    spans, x = _spans_and_input()
    block = _Block()
    bind_adaln_parity(block)
    with pytest.raises(ValueError, match="3 spans and 2 stock row counts"):
        block(x.clone(), spans, None, {}, stock_norm_rows=(12, 8))


def test_a_forward_that_reads_no_norm_global_runs_unrouted():
    spans, x = _spans_and_input()
    block = _NoNormBlock()
    assert not loads_norm_global(_NoNormBlock.forward)
    assert not bind_adaln_parity(block)
    assert torch.equal(block(x, spans, None, {}, stock_norm_rows=(12, 8, 2)), x)


@pytest.mark.parametrize("block_class,warned", [(_Block, 0), (_NoNormBlock, 1)])
def test_the_adapter_binds_every_sharded_block_and_warns_once_when_unroutable(
        monkeypatch, block_class, warned):
    from dgx_monarch.adapters import flux3
    from dgx_monarch.adapters.base import InjectionContext

    warnings: list[str] = []
    monkeypatch.setattr(flux3.log, "warning", lambda message, *args: warnings.append(message))
    model = torch.nn.Module()
    model.content_mode_blocks = torch.nn.ModuleDict({
        "image": torch.nn.ModuleList([block_class() for _ in range(2)]),
        "image_cond": torch.nn.ModuleList()})
    model.txt_mode_blocks = torch.nn.ModuleList()
    model.single_blocks = torch.nn.ModuleList([block_class() for _ in range(3)])
    model.single_blocks[0].num_heads = 24
    flux3.Flux3Adapter().inject_usp(model, InjectionContext(
        topology_sp=2, usp_attention=None, pure_ulysses=True))
    sharded = [*model.content_mode_blocks["image"], *model.single_blocks]
    assert all("forward" in vars(block) for block in sharded)
    assert len([m for m in warnings if "parity route is off" in m]) == warned


# --- The GPU seam ----------------------------------------------------------------
#
# The op probe's inputs: one generator, FLUX.3 Klein's 3072 hidden width, and
# the same draw order, so the rows below are the rows the probe measured.

WIDTH, EPS, PROBE_ROWS, SPAN = 3072, 1e-6, 8272, slice(4096, 8192)


def _probe_inputs(batch: int):
    generator = torch.Generator(device="cuda").manual_seed(1234)

    def draw(*shape, scale=1.0, offset=0.0):
        values = torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)
        return (values * scale + offset).to(torch.bfloat16)

    x = draw(1, PROBE_ROWS, WIDTH, scale=3.0, offset=0.25)
    scale, shift = draw(1, 1, WIDTH, scale=0.2), draw(1, 1, WIDTH, scale=0.2)
    if batch == 2:
        x = torch.cat([x, draw(1, PROBE_ROWS, WIDTH, scale=3.0, offset=-0.1)], dim=0)
        scale, shift = draw(2, 1, WIDTH, scale=0.2), draw(2, 1, WIDTH, scale=0.2)
    return x[:, SPAN], scale, shift


def _comfy_kitchen():
    return pytest.importorskip("comfy_kitchen", reason="the seam is comfy-kitchen's kernel")


def _report(label: str, batch: int, rows: int, candidate, reference) -> int:
    mismatched = int((candidate != reference).sum())
    print(f"{label}: batch {batch}, {rows} rows a sample ({batch * rows} kernel rows), "
          f"{mismatched} of {candidate.numel()} outputs differ from stock's call")
    return mismatched


# (batch, piece start in the span, piece rows): rank 0's 40 reference rows of
# the uly2 edit, a 1024-row piece at the branch itself, the same piece at an
# odd offset, and base T512's batch-2 image piece of 1024 kernel rows.
SEAM_CASES = [(1, 0, 40), (1, 0, 1024), (1, 1531, 1024), (2, 0, 512)]
SHORT_CASES = [(1, 0, 40), (1, 0, 1024), (2, 0, 512)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel branch needs a CUDA device")
@pytest.mark.parametrize("batch,start,rows", SEAM_CASES)
def test_a_piece_padded_to_its_spans_rows_gets_stocks_bits_on_the_gpu(batch, start, rows):
    ck = _comfy_kitchen()
    span, scale, shift = _probe_inputs(batch)
    stock = ck.adaln(span, scale, shift, EPS)[:, start:start + rows]
    piece = span[:, start:start + rows].contiguous()
    padded = at_stock_rows(lambda call: ck.adaln(call, scale, shift, EPS), piece,
                           span.shape[1])
    print(f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
          f"comfy-kitchen {metadata.version('comfy-kitchen')}")
    assert _report("padded to the span", batch, rows, padded, stock) == 0
    assert torch.equal(padded, stock)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel branch needs a CUDA device")
@pytest.mark.parametrize("batch,start,rows", SHORT_CASES)
def test_the_unpadded_short_call_takes_the_other_kernel_on_this_gpu(batch, start, rows):
    """The branch the padding closes exists here. If it fails, comfy-kitchen
    no longer splits its kernels at these rows; the padded test above still
    holds, and the docs' account of the branch needs updating."""
    ck = _comfy_kitchen()
    span, scale, shift = _probe_inputs(batch)
    stock = ck.adaln(span, scale, shift, EPS)[:, start:start + rows]
    short = ck.adaln(span[:, start:start + rows].contiguous(), scale, shift, EPS)
    assert _report("unpadded", batch, rows, short, stock) > 0
