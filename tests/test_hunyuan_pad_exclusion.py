"""Hunyuan divisibility pad rows are not attended (pure Ulysses).

`shard_seq` zero-pads an odd stream, and after the norm and the modulation
shift a zero row is a real key at RoPE 0. Stock one-GPU attention has no such
key. The double loop pads text and image apart; the single loop pads the tail
of the joined stream. Each loop names its own rows as `drop_rows`, and the
full-axis path removes them after it restores stock order. The harness is the
one in tests/test_hunyuan_order.py, run strict: its recording kernel fails the
test if a pad row (tag 0) reaches attention or a real row is missing.
"""
from __future__ import annotations

import pytest
import torch

from test_hunyuan_order import (  # noqa: F401  (autouse fixture re-exported)
    SHAPES,
    WORLD,
    _inputs,
    _matches,
    _run,
    _stock_tags,
    comfy_layers_stub,
)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("qwen_rows,byt5_rows,grid", SHAPES)
def test_sharded_render_reproduces_the_one_rank_render_row_for_row(
    monkeypatch, qwen_rows, byt5_rows, grid, batch
):
    inputs = _inputs(qwen_rows, byt5_rows, grid, batch)
    total = qwen_rows + byt5_rows + grid[0] * grid[1]
    reference, _ = _run(monkeypatch, inputs, total, world=1, pure_ulysses=False, strict=True)
    outputs, harness = _run(monkeypatch, inputs, total, world=WORLD, pure_ulysses=True,
                            strict=True)
    stock = _stock_tags(total)
    for kind in ("double", "single"):
        calls = [call for call in harness.calls if call[1] == kind]
        assert len(calls) == WORLD * 2
        # The strict recorder already rejects pad tags; this pins the exact keys.
        assert all(tags == stock for *_head, tags in calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("ref_grid", [(1, 3), (2, 2)])
def test_reference_rows_count_as_image_rows_for_the_pad(monkeypatch, ref_grid, batch):
    """Reference tokens join the image stream before the shard, so the pad
    follows image plus reference rows: 3 + 6 pads a row where 6 alone does not,
    and 4 + 6 pads nothing."""
    inputs = _inputs(3, 2, (2, 3), batch, ref_grid=ref_grid)
    total = 5 + ref_grid[0] * ref_grid[1] + 6
    reference, _ = _run(monkeypatch, inputs, total, world=1, pure_ulysses=False, strict=True)
    outputs, harness = _run(monkeypatch, inputs, total, world=WORLD, pure_ulysses=True,
                            strict=True)
    assert harness.calls and all(tags == _stock_tags(total) for *_head, tags in harness.calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("qwen_rows,byt5_rows,grid,double_drop,single_drop", [
    # Local rows per rank are [text_local, image_local]; a stream's pad is its
    # last padded row, held by rank 1. Worked by hand:
    (4, 2, (2, 4), [], []),           # text 3, image 4 per rank; joint 14
    (3, 2, (2, 4), [9], [13]),        # text 3 + image 4 = 7: 7 + 2; joint 7 + 6
    (4, 2, (3, 3), [15], [15]),       # text 3 + image 5 = 8: 8 + 3 + 4; joint 8 + 7
    (3, 2, (3, 3), [10, 15], []),     # 8 + 2, 8 + 3 + 4; joint 14 divides
])
def test_each_loop_names_its_own_pad_rows(
    monkeypatch, qwen_rows, byt5_rows, grid, double_drop, single_drop
):
    """The double loop names text and image pads in rank-major coordinates; the
    single loop names the joined stream's tail. One options dict cannot carry
    both, so each loop has its own."""
    inputs = _inputs(qwen_rows, byt5_rows, grid)
    total = qwen_rows + byt5_rows + grid[0] * grid[1]
    _outputs, harness = _run(monkeypatch, inputs, total, world=WORLD, pure_ulysses=True)
    assert harness.drops and all(
        drop == (double_drop if kind == "double" else single_drop)
        for kind, drop in harness.drops)
    assert len(harness.drops) == WORLD * 4
