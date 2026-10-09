"""CPU contracts for Boogu's full-row instruct projections (pure Ulysses).

Stock runs the seven instruct-stream Linears of each double block on every
instruct row. A shard runs them on half, and the BLAS can pick another kernel
for another row count. The forward wraps them through chroma_text's shared
helper, so each sees all instruct rows in stock's layout and its output is
re-sharded. The first tests drive the helper on one block; the later ones
drive the real bound forward through the harness of
tests/test_boogu_ulysses_order.py.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import boogu_ulysses, chroma_text
from dgx_monarch.adapters.base import pad_seq_to_multiple
from test_boogu_ulysses_order import (  # noqa: F401  (fixture re-exported)
    DIM,
    SHAPES,
    WORLD,
    _matches,
    _ToyDouble,
    comfy_leaves,
    drive,
)

NAMES = ("to_q", "to_k", "to_v", "out", "linear_1", "linear_3", "linear_2")


def _linear():
    return torch.nn.Linear(DIM, DIM, bias=False, dtype=torch.float64)


def _namespaces():
    processor = SimpleNamespace(instruct_to_q=_linear(), instruct_to_k=_linear(),
                                instruct_to_v=_linear(), instruct_out=_linear())
    feed_forward = SimpleNamespace(linear_1=_linear(), linear_3=_linear(), linear_2=_linear())
    return processor, feed_forward


def _block():
    processor, feed_forward = _namespaces()
    return SimpleNamespace(img_instruct_attn=SimpleNamespace(processor=processor),
                           instruct_feed_forward=feed_forward)


def _module(block, name):
    if name.startswith("linear_"):
        return getattr(block.instruct_feed_forward, name)
    return getattr(block.img_instruct_attn.processor, f"instruct_{name}")


def _real_strides(stride, shape):
    return tuple(step for step, size in zip(stride, shape, strict=True) if size != 1)


def _record(block):
    seen = {name: [] for name in NAMES}
    for name in NAMES:
        module = _module(block, name)
        original = module.forward

        def spy(value, name=name, original=original):
            seen[name].append((value.shape[1], value.stride(), value.is_contiguous(),
                               tuple(value.shape)))
            return original(value)
        module.forward = spy
    return seen


def _run_helper(monkeypatch, block, source, world, image_rows):
    """Each rank's output of every projection, joined and trimmed to full rows."""
    padded, text_rows = pad_seq_to_multiple(source, world, dim=1)
    chunks = list(torch.chunk(padded, world, dim=1))
    outputs = {name: [] for name in NAMES}
    for rank in range(world):
        def gather(value, length, dim=1, rank=rank):
            assert length == text_rows and dim == 1 and torch.equal(value, chunks[rank])
            return padded[:, :length]
        monkeypatch.setattr(chroma_text, "sp_gather", gather)
        monkeypatch.setattr(chroma_text, "sp_rank", lambda rank=rank: rank)
        monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
        with chroma_text.full_row_text_projections(
                block, text_rows, image_rows, pure_ulysses=True,
                select=boogu_ulysses.instruct_projections):
            for name in NAMES:
                outputs[name].append(_module(block, name)(chunks[rank]))
    return {name: torch.cat(parts, dim=1)[:, :text_rows] for name, parts in outputs.items()}


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("rows", [1, 33, 55, 66, 109, 110])
def test_every_projection_sees_full_rows_in_stock_layout(monkeypatch, batch, world, rows):
    """66 and 109 are the template's uncond and cond instruct lengths."""
    torch.manual_seed(batch * 1000 + rows)
    block, image_rows = _block(), 17
    originals = {name: _module(block, name).forward for name in NAMES}
    seen = _record(block)
    source = torch.randn(batch, rows, DIM, dtype=torch.float64)
    actual = _run_helper(monkeypatch, block, source, world, image_rows)
    stock_joint = torch.empty(batch, rows + image_rows, DIM, dtype=torch.float64)
    stock_view = stock_joint[:, :rows]
    stock_view.copy_(source)
    for name in NAMES:
        stock_input = stock_view if name == "out" else source
        assert torch.equal(actual[name], originals[name](stock_input)), name
        assert len(seen[name]) == world, name
        for seen_rows, stride, contiguous, seen_shape in seen[name]:
            assert seen_rows == rows, name
            if name == "out":
                # Stock slices instruct rows out of the joint attention output,
                # so the batch stride spans the image rows too.
                assert stride == stock_view.stride() and contiguous is stock_view.is_contiguous()
            else:
                assert contiguous, name
                assert _real_strides(stride, seen_shape) == _real_strides(
                    source.stride(), source.shape), name


def test_selection_covers_the_seven_instruct_linears_and_skips_non_linears():
    block = _block()
    selected = boogu_ulysses.instruct_projections(block)
    assert [module for module, _ in selected] == [_module(block, name) for name in NAMES]
    assert [joint for _, joint in selected] == [False, False, False, True, False, False, False]

    class MixedPrecisionLinear(torch.nn.Module):
        """Stands in for comfy's mixed_precision_ops Linear."""
        def forward(self, value):
            return value
    block.img_instruct_attn.processor.instruct_to_k = MixedPrecisionLinear()
    block.instruct_feed_forward.linear_2 = MixedPrecisionLinear()
    kept = [module for module, _ in boogu_ulysses.instruct_projections(block)]
    assert kept == [_module(block, name) for name in ("to_q", "to_v", "out", "linear_1", "linear_3")]
    assert boogu_ulysses.instruct_projections(SimpleNamespace()) == ()


class _SpyLinear(torch.nn.Linear):
    def __init__(self):
        super().__init__(DIM, DIM, bias=False, dtype=torch.float64)
        self.seen = []

    def forward(self, value):
        self.seen.append((value.shape[1], value.stride(), value.is_contiguous()))
        return super().forward(value)


class _LinearDouble(_ToyDouble):
    """The toy double block plus Boogu's seven instruct Linears, each called
    once per block on the operand stock hands it."""

    def __init__(self, index):
        super().__init__(index)
        processor = SimpleNamespace(instruct_to_q=_SpyLinear(), instruct_to_k=_SpyLinear(),
                                    instruct_to_v=_SpyLinear(), instruct_out=_SpyLinear())
        feed_forward = SimpleNamespace(linear_1=_SpyLinear(), linear_3=_SpyLinear(),
                                       linear_2=_SpyLinear())
        self.img_instruct_attn = SimpleNamespace(processor=processor)
        self.instruct_feed_forward = feed_forward

    def __call__(self, img, instruct, *args, **kwargs):
        for name in ("to_q", "to_k", "to_v", "linear_1", "linear_3", "linear_2"):
            _module(self, name)(instruct)
        return super().__call__(img, instruct, *args, **kwargs)

    def instruct_output(self, rows):
        _module(self, "out")(rows)


@pytest.mark.usefixtures("comfy_leaves")
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,refs", SHAPES)
def test_forward_wraps_every_double_block_under_pure_ulysses_only(
    monkeypatch, text_rows, grid, refs, batch
):
    image = grid[0] * grid[1] + sum(refs)
    for pure, expected_rows in ((True, text_rows), (False, -(-text_rows // WORLD))):
        _outputs, harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                                  pure_ulysses=pure, batch=batch, strict=pure,
                                  double_cls=_LinearDouble)
        for model in harness.models:
            for block in model.double_stream_layers:
                for name in NAMES:
                    module = _module(block, name)
                    assert [rows for rows, *_ in module.seen] == [expected_rows], (name, pure)
                    assert "forward" not in vars(module)
                    if pure and name == "out":
                        # Stock's joint output spans every instruct and image row.
                        assert module.seen[0][1] == ((text_rows + image) * DIM, DIM, 1)


@pytest.mark.usefixtures("comfy_leaves")
@pytest.mark.parametrize("text_rows,grid,refs", [(5, (3, 3), ()), (4, (3, 3), (2,))])
def test_the_wrapped_forward_still_matches_one_rank(monkeypatch, text_rows, grid, refs):
    reference, _ = drive(monkeypatch, text_rows, grid, refs, world=1, pure_ulysses=False,
                         double_cls=_LinearDouble)
    outputs, _harness = drive(monkeypatch, text_rows, grid, refs, world=WORLD,
                              pure_ulysses=True, double_cls=_LinearDouble)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])
