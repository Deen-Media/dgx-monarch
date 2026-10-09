"""Krea2's text path runs on every text row under pure Ulysses.

Stock runs `txtfusion` (two layerwise blocks, the projector, two refiner
blocks) and `txtmlp` on the whole text stream. A shard runs their Linears and
norms on half the rows, and the BLAS can pick another kernel for another row
count, which can change the bits. The conditioning is replicated on every
rank, so under pure Ulysses the forward runs stock's text path on all rows and
shards only its output. These tests drive the real bound forward over the toy
of tests/test_krea2_order.py and read which rows each text stage was handed,
and which mask and options `txtfusion` got: stock passes no mask and the
caller's options. Ring and hybrid keep the shard-first path, refiner attention
through USP included, and the last test pins it.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch.adapters import base
from test_krea2_order import (  # noqa: F401  (fixture re-exported)
    DIM,
    SHAPES,
    WORLD,
    _local,
    _matches,
    _run,
    _streams,
    krea2_comfy_stub,
)

pytestmark = pytest.mark.usefixtures("krea2_comfy_stub")

# A marker the forward must pass through to txtfusion unchanged.
CALLER_OPTIONS = {"caller_marker": "krea2-text-rows"}


def _refiner(calls):
    return [call for call in calls if call[1] == "refiner"]


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_pure_ulysses_runs_every_text_stage_on_every_text_row(monkeypatch, text_rows, grid, batch):
    text, latent = _streams(text_rows, grid, batch)
    # Krea2 ignores attention_mask, so a non-None one must not reach txtfusion.
    caller = {"attention_mask": torch.ones(batch, text_rows), "options": CALLER_OPTIONS}
    reference, _ = _run(monkeypatch, text, latent, world=1, pure_ulysses=False, **caller)
    outputs, harness = _run(monkeypatch, text, latent, world=WORLD, pure_ulysses=True, **caller)
    stock_input = text.reshape(batch, text_rows, 1, DIM)
    for model in harness.models:
        fusion = model.txtfusion
        # One call, on stock's unsharded and contiguous (B, T, layers, D) input.
        assert len(fusion.inputs) == 1
        assert torch.equal(fusion.inputs[0], stock_input)
        assert fusion.inputs[0].stride() == stock_input.stride()
        # Stock's call: no mask, and the caller's options rather than USP ones.
        assert len(fusion.masks) == 1 and fusion.masks[0] is None
        assert fusion.options == [CALLER_OPTIONS]
        assert [spy.seen for spy in fusion.layerwise_blocks] == [
            [(batch * text_rows, 1, DIM)]] * 2
        assert fusion.projector.seen == [(batch, text_rows, DIM, 1)]
        assert model.txtmlp.seen == [(batch, text_rows, DIM)]
    # Native refiner attention over all text keys in stock order, never USP.
    stock_text = [float(t) for t in range(1, text_rows + 1)]
    refiner = _refiner(harness.calls)
    assert len(refiner) == WORLD * 2
    assert all(entry == "native" and tags == stock_text for entry, _k, _i, tags in refiner)
    assert not [kind for kind, _order in harness.orders if kind == "refiner"]
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_without_pure_ulysses_the_text_path_keeps_its_shard(monkeypatch, text_rows, grid, batch):
    """Ring and hybrid: the shard-first path with refiner USP attention."""
    text, latent = _streams(text_rows, grid, batch)
    _outputs, harness = _run(monkeypatch, text, latent, world=WORLD, pure_ulysses=False)
    local = _local(text_rows)
    for model in harness.models:
        fusion = model.txtfusion
        assert fusion.inputs == []
        assert [spy.seen for spy in fusion.layerwise_blocks] == [[(batch * local, 1, DIM)]] * 2
        assert fusion.projector.seen == [(batch, local, DIM, 1)]
        assert model.txtmlp.seen == [(batch, local, DIM)]
    stock_text = [float(t) for t in range(1, text_rows + 1)]
    padded = text_rows % WORLD != 0
    refiner = _refiner(harness.calls)
    assert len(refiner) == WORLD * 2
    assert all(entry == ("exact" if padded else "xfuser") and tags == stock_text
               for entry, _k, _i, tags in refiner)
    text_drop = tuple(base.padded_row_indices([(text_rows, local)]))
    assert {tuple(rows or ()) for kind, rows in harness.drops if kind == "refiner"} == {text_drop}
