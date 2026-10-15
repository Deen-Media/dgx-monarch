"""adapters/flux3_packing.py on the CPU, with no ComfyUI import.

Stock FLUX.3 hands every block ``(start, end, ModulationOut)`` spans that
partition its rows: the first span's modulation is applied to every row and
each later span redoes its own, and each gate writes its span's rows only. A
rank holding a contiguous window must therefore see the spans clipped to its
window, with the first span kept even when empty. These cases pin that, the
shard geometry, the pad rows, the layout asserts and the two refusals.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

from dgx_monarch import accuracy_waiver
from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import UnsupportedModelError, padded_row_indices
from dgx_monarch.adapters.flux3_packing import (
    assert_joint_layout,
    joint_pad_count,
    local_spans,
    refuse_indivisible_heads,
    refuse_padded_ring,
    shard_stream,
)
from dgx_monarch.refusal import RefusalClass, parse_leading_refusal_tag


def _class(exc: BaseException) -> RefusalClass | None:
    tag = parse_leading_refusal_tag(str(exc))
    return tag.refusal_class if tag is not None else None


def _rows(spans) -> list[tuple[int, int, str]]:
    return [(start, end, mod) for start, end, mod in spans]


def test_a_span_inside_across_and_outside_the_window():
    spans = [(0, 4, "img"), (4, 10, "ref"), (10, 12, "txt")]
    # Window [3, 7): img clipped to its last row, ref clipped, txt dropped.
    assert _rows(local_spans(spans, 3, 4)) == [(0, 1, "img"), (1, 4, "ref")]
    # Window [5, 9): inside ref only; the empty img span is kept, txt dropped.
    assert _rows(local_spans(spans, 5, 4)) == [(0, 0, "img"), (0, 4, "ref")]
    # Window [8, 12): img kept empty, ref's tail, txt whole.
    assert _rows(local_spans(spans, 8, 4)) == [(0, 0, "img"), (0, 2, "ref"), (2, 4, "txt")]


def test_the_uly2_one_reference_edit_example_exactly():
    """uly2, edit at 1024 square, one 1 MP reference, 80 text rows: N = 8272."""
    spans = [(0, 4096, "img"), (4096, 8192, "ref"), (8192, 8272, "txt")]
    rows = 4136
    assert _rows(local_spans(spans, 0, rows)) == [(0, 4096, "img"), (4096, 4136, "ref")]
    assert _rows(local_spans(spans, rows, rows)) == [
        (0, 0, "img"), (0, 4056, "ref"), (4056, 4136, "txt")]


def _check_partition(total: int, world: int, spans) -> None:
    rows = -(-total // world)
    seen: list[int] = []
    for rank in range(world):
        row0 = rank * rows
        local = local_spans(spans, row0, rows)
        assert local[0][2] == spans[0][2], "spans[0] must lead on every rank"
        real = min(rows, max(0, total - row0))
        covered = 0
        edge = None
        for start, end, mod in local:
            if start == end:
                continue
            assert 0 <= start < end <= real
            if edge is not None:
                assert start == edge, "local spans must stay contiguous"
            edge = end
            covered += end - start
            seen.extend(range(row0 + start, row0 + end))
            # The span's modulation is the stock span that owns those rows.
            owner = next(m for s, e, m in spans if s <= row0 + start < e)
            assert owner == mod
        assert covered == real
        # A later span with no local row is dropped, never kept empty.
        assert all(start != end for start, end, _mod in local[1:])
    assert sorted(seen) == list(range(total))


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("lengths", [(9, 4, 4), (16, 0, 4), (25, 12, 80), (4, 6, 3), (2, 0, 3)])
def test_every_real_row_is_covered_exactly_once_over_the_ranks(world, lengths):
    image, refs, text = lengths
    spans, start = [], 0
    for rows, mod in ((image, "img"), (refs, "ref"), (text, "txt")):
        if rows:
            spans.append((start, start + rows, mod))
            start += rows
    _check_partition(start, world, spans)


def test_world_four_tiny_stream_puts_pads_on_two_ranks_and_one_rank_holds_only_pads(monkeypatch):
    """N = 5 at world 4: chunks of 2 over 8 rows. Rank 2 holds one real row and
    one pad, rank 3 holds two pads and keeps only the empty first span."""
    spans = [(0, 3, "img"), (3, 5, "txt")]
    _check_partition(5, 4, spans)
    assert _rows(local_spans(spans, 6, 2)) == [(0, 0, "img")]
    monkeypatch.setattr(base, "sp_world", lambda: 4)
    assert padded_row_indices([(5, 2)]) == [5, 6, 7]


@pytest.fixture
def ranks(monkeypatch):
    box = {"world": 2, "rank": 0}
    monkeypatch.setattr(base, "sp_world", lambda: box["world"])
    monkeypatch.setattr(base, "sp_rank", lambda: box["rank"])
    return box


def _rope(ids):
    # Stock's layout [1, n, 1, 64, 2, 2] is not needed here: a per-row map
    # that keeps the row count and returns a contiguous tensor, as stock's does.
    return (ids * 2.0).unsqueeze(0).contiguous()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("world,total", [(2, 9), (2, 10), (4, 5)])
def test_shard_stream_geometry(ranks, batch, world, total):
    ranks["world"] = world
    tokens = torch.arange(batch * total * 3, dtype=torch.float32).reshape(batch, total, 3)
    ids = torch.arange(total * 4, dtype=torch.float32).reshape(total, 4)
    spans = [(0, total - 2, "a"), (total - 2, total, "b")]
    rows = -(-total // world)
    for rank in range(world):
        ranks["rank"] = rank
        shard = shard_stream(tokens, ids, spans, _rope, "test")
        assert shard.tokens.shape == (batch, rows, 3)
        assert shard.tokens.is_contiguous()
        real = min(rows, max(0, total - rank * rows))
        assert torch.equal(shard.tokens[:, :real], tokens[:, rank * rows:rank * rows + real])
        assert torch.equal(shard.tokens[:, real:], torch.zeros_like(shard.tokens[:, real:]))
        assert shard.pe.shape[1] == rows
        assert torch.equal(shard.pe[0, :real], ids[rank * rows:rank * rows + real] * 2.0)
        assert shard.spans == local_spans(spans, rank * rows, rows)
        assert shard.drop_rows == padded_row_indices([(total, rows)])
        assert shard.orig == total
    assert (shard.drop_rows == []) == (total % world == 0)


def test_joint_pad_count_follows_the_world(ranks):
    assert joint_pad_count(8272) == 0
    assert joint_pad_count(8110) == 0
    ranks["world"] = 4
    assert joint_pad_count(8110) == 2
    assert joint_pad_count(12125) == 3


def _parts(image=4, refs=6, text=2, batch=1):
    parts = [(torch.zeros(batch, image, 3), "image", torch.zeros(image, 4))]
    if refs:
        parts.append((torch.zeros(batch, refs, 3), "image_cond", torch.zeros(refs, 4)))
    parts.append((torch.zeros(batch, text, 3), "txt", torch.zeros(text, 4)))
    return parts


def _spans(parts):
    spans, start = [], 0
    for tokens, mode, _ids in parts:
        spans.append((start, start + tokens.shape[1], mode))
        start += tokens.shape[1]
    return spans


def test_assert_joint_layout_holds_stocks_layout():
    parts = _parts()
    ids = torch.cat([p[2] for p in parts])
    assert_joint_layout(parts, ids, _spans(parts), 1, [(2, 3)])
    no_refs = _parts(refs=0)
    assert_joint_layout(no_refs, torch.cat([p[2] for p in no_refs]), _spans(no_refs), 1, None)


@pytest.mark.parametrize("broken", [
    "refs without shapes", "shapes without refs", "batch", "part ids", "joint ids",
    "span gap", "span end"])
def test_assert_joint_layout_raises_on_each_broken_invariant(broken):
    parts = _parts(refs=0 if broken == "shapes without refs" else 6)
    ref_shapes = None if broken == "refs without shapes" else [(2, 3)]
    if broken == "shapes without refs":
        ref_shapes = [(2, 3)]
    if broken == "batch":
        parts[-1] = (torch.zeros(2, 2, 3), "txt", parts[-1][2])
    if broken == "part ids":
        parts[0] = (parts[0][0], "image", torch.zeros(5, 4))
    ids = torch.cat([p[2] for p in parts])
    if broken == "joint ids":
        ids = ids[:-1]
    spans = _spans(parts)
    if broken == "span gap":
        spans[1] = (spans[1][0] + 1, spans[1][1], spans[1][2])
    if broken == "span end":
        spans[-1] = (spans[-1][0], spans[-1][1] - 1, spans[-1][2])
    with pytest.raises(AssertionError):
        assert_joint_layout(parts, ids, spans, 1, ref_shapes)


@pytest.fixture
def ring(monkeypatch):
    box = {"world": 2}
    distributed = types.ModuleType("xfuser.core.distributed")
    distributed.get_ring_parallel_world_size = lambda: box["world"]
    for name, module in (("xfuser", types.ModuleType("xfuser")),
                         ("xfuser.core", types.ModuleType("xfuser.core")),
                         ("xfuser.core.distributed", distributed)):
        monkeypatch.setitem(sys.modules, name, module)
    return box


@pytest.mark.parametrize("pads", [(1, 0), (0, 2), (1, 3)])
def test_a_padded_ring_refuses_class_k_and_a_granted_waiver_passes(ring, monkeypatch, pads):
    with pytest.raises(UnsupportedModelError, match="ring/hybrid") as caught:
        refuse_padded_ring(*pads)
    tag = parse_leading_refusal_tag(str(caught.value))
    assert tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
    assert tag.guard == base.RING_PAD_GUARD and tag.waivable
    monkeypatch.setattr(accuracy_waiver, "waived", lambda guard: guard == base.RING_PAD_GUARD)
    refuse_padded_ring(*pads)


def test_an_unpadded_ring_and_a_padded_ulysses_render_pass(ring):
    refuse_padded_ring(0, 0)
    ring["world"] = 1
    refuse_padded_ring(1, 3)


def test_unpadded_render_never_reads_the_ring_degree(monkeypatch):
    """No pad, no question: the guard is not even imported."""
    monkeypatch.setitem(sys.modules, "xfuser.core.distributed", None)
    refuse_padded_ring(0, 0)


@pytest.mark.parametrize("degree", [5, 7, 16])
def test_a_ulysses_degree_that_does_not_divide_24_heads_refuses_class_p(degree):
    with pytest.raises(UnsupportedModelError, match=f"Ulysses degree {degree}") as caught:
        refuse_indivisible_heads(24, degree)
    assert _class(caught.value) is RefusalClass.PHYSICS


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_the_degrees_that_divide_24_heads_pass(degree):
    refuse_indivisible_heads(24, degree)
