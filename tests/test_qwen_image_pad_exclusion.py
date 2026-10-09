"""Qwen-Image divisibility pad rows are not attended (pure Ulysses).

`shard_seq` zero-pads each stream, and after LayerNorm and the modulation shift a
zero row is a real key at RoPE (0, 0, 0). Stock one-GPU attention has no such
key. The forward names those rows as `drop_rows`, and the full-axis path removes
them after it restores stock order. The harness is the one in
tests/test_qwen_image_order.py; in strict mode its recording kernel fails the
test if a pad row (tag 0) reaches attention or a real row goes missing.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch.adapters import base
from test_qwen_image_order import (
    SHAPES,
    WORLD,
    _matches,
    _rank_major_tags,
    _references,
    _run,
    _streams,
)

# (text rows, image grid, reference rows). The image stream the forward shards
# is image plus references, so its pad follows that total, not the image alone:
# 8 + 3 = 11 pads a row while the image alone does not, and 9 + 3 = 12 pads
# nothing while the image alone would.
REFERENCE_CASES = [
    (4, (2, 4), [3]),
    (5, (2, 4), [3]),
    (4, (3, 3), [3]),
    (5, (3, 3), [2, 3]),
    (6, (2, 4), [4, 1]),
]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_sharded_render_reproduces_the_one_rank_render_row_for_row(
    monkeypatch, text_rows, grid, batch
):
    text, image = _streams(text_rows, grid, batch)
    reference, _ = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False, strict=True)
    outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True,
                            strict=True)
    stock = [float(t) for t in range(1, text_rows + image.shape[1] + 1)]
    # The strict recorder already rejects pad tags; this pins the exact keys.
    assert harness.calls and all(tags == stock for *_head, tags in harness.calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid,rank_major,stock", [
    # Local pair per rank is [text_local, image_local]; the pad is each
    # stream's last padded row, held by rank 1. Worked by hand:
    (4, (2, 4), [], []),             # nothing pads
    (5, (2, 4), [9], [5]),           # text 3+4 per rank: 7 + 2
    (4, (3, 3), [13], [13]),         # text 2, image 5 per rank: 7 + 2 + 4
    (5, (3, 3), [10, 15], [5, 15]),  # text 3, image 5 per rank: 8 + 2, 8 + 3 + 4
])
def test_drop_rows_name_the_rank_major_pad_coordinates(
    monkeypatch, text_rows, grid, rank_major, stock
):
    """The dispatcher gets rank-major pad coordinates, and the descriptor maps
    them to stock's [all text, all image] positions."""
    text, image = _streams(text_rows, grid)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True)
    assert harness.drops and all(drops == rank_major for drops in harness.drops)
    order = harness.orders[0]
    gathered = WORLD * (order.text_rows + order.image_rows)
    probe = torch.arange(gathered, dtype=torch.float32).reshape(1, -1, 1, 1)
    _q, _k, _v, canonical, _kv, _perm = order.canonicalize(
        probe, probe, probe, rank_major, None, None)
    assert canonical == stock


@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_ring_and_hybrid_keep_attended_pads_and_never_refuse(monkeypatch, text_rows, grid):
    """Without pure Ulysses no drop rows are named, so the ring pad guard, which
    refuses named rows, cannot fire, and the rank-major path attends the pads."""
    text, image = _streams(text_rows, grid)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=False)
    assert all(drops is None for drops in harness.drops)
    assert all(order is None for order in harness.orders)
    expected = _rank_major_tags(text_rows, image.shape[1])
    assert all(tags == expected for _e, kind, _i, tags in harness.calls if kind == "double")


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("method", ["index", "offset", "negative_index"])
@pytest.mark.parametrize("text_rows,grid,ref_rows", REFERENCE_CASES)
def test_reference_tokens_count_as_image_rows_for_the_pad(
    monkeypatch, text_rows, grid, ref_rows, method, batch
):
    """The forward shards image plus references, and the pad follows that length.

    The strict recorder rejects a pad tag and a lost real tag, so a pad
    coordinate computed from the image alone (`num_embeds`) either leaves the
    pad attended or drops a reference token. The canonical drop set is also
    worked out here from the stream lengths, independent of the forward.
    """
    text, image = _streams(text_rows, grid, batch)
    refs = _references(text_rows, image.shape[1], ref_rows, batch)
    kwargs = {"forward_kwargs": {"ref_latents_method": method}}
    reference, _ = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False,
                        strict=True, refs=refs, **kwargs)
    outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True,
                            strict=True, refs=refs, **kwargs)
    image_total = image.shape[1] + sum(ref_rows)
    stock = [float(t) for t in range(1, text_rows + image_total + 1)]
    assert harness.calls and all(tags == stock for *_head, tags in harness.calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])
    order = harness.orders[0]
    padded_text, padded_image = WORLD * order.text_rows, WORLD * order.image_rows
    assert padded_image - image_total == (-image_total) % WORLD
    gathered = padded_text + padded_image
    probe = torch.arange(gathered, dtype=torch.float32).reshape(1, -1, 1, 1)
    rank_major = harness.drops[0]
    *_head, canonical, _kv, _perm = order.canonicalize(probe, probe, probe, rank_major, None, None)
    assert canonical == (list(range(text_rows, padded_text))
                         + list(range(padded_text + image_total, gathered)))


def test_the_ring_pad_guard_the_gate_avoids_still_refuses(monkeypatch):
    """The ring pad guard still refuses named pad rows; gating `drop_rows` on pure
    Ulysses keeps the forward clear of it.

    This exercises `base` only. The forward-level proof that ring and hybrid
    name no drop rows is test_ring_and_hybrid_keep_attended_pads_and_never_refuse.
    """
    with pytest.raises(base.UnsupportedModelError):
        base.assert_ulysses_only_padding(2, 1)
