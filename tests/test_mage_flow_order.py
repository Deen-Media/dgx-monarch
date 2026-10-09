"""Mage Flow double blocks attend stock's keys in stock's order (pure Ulysses).

Each rank holds `[text_r, image_r]`, so Ulysses' head all-to-all hands the
kernel `[text0, image0, text1, image1]` where stock attends `[text0, text1,
image0, image1]`, and each stream's divisibility pad is a zero row that is a
real key after modulation. This file drives the real bound Mage forward through
the real `base.make_usp_attention` dispatch and the real full-axis call, with
only the all-to-all, the gather and the kernel stubbed (the harness
tests/test_flux_pad_exclusion.py builds). Every key row carries a tag, so the
kernel names the exact rows and order it was handed; a pad row carries tag 0.
No ComfyUI import: tests/test_mage_flow_stock_exact.py runs the real model.
"""
from __future__ import annotations

import threading

import pytest
import torch

from dgx_monarch.adapters import base, chroma_text, mage_flow
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.mage_flow import MageFlowAdapter
from test_flux_pad_exclusion import (
    _LOCAL,
    DIM,
    _channel_map,
    _Harness,
    _install_stubs,
    _matches,
    _Modulation,
    _rank,
)
from test_qwen_image_order import WORLD, _QwenToyBlock, _rank_major_tags, _references, _streams

# (text rows, image grid): even, odd text, odd image, both odd.
SHAPES = [(4, (2, 4)), (5, (2, 4)), (4, (3, 3)), (5, (3, 3))]
# The image stream Mage shards is target plus references, so its pad follows
# that total: 8 + 3 pads a row, 9 + 3 does not.
REFERENCE_CASES = [(4, (2, 4), [3]), (5, (3, 3), [3]), (5, (3, 3), [2, 3]), (6, (2, 4), [4, 1])]


class _MageToyBlock(_QwenToyBlock):
    """Mage's keyword block call over the Qwen toy's elementwise attention."""

    def __call__(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                 temb, image_rotary_emb, transformer_options):
        return super().__call__(hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                                temb, image_rotary_emb, None, transformer_options)


class _ToyMage:
    """Mage-shaped where it matters: unpatched tokens, ids from process_img."""

    out_channels = DIM
    img_in = staticmethod(lambda value: value)
    txt_norm = staticmethod(lambda value: value)
    txt_in = staticmethod(lambda value: value)
    norm_out = staticmethod(lambda value, temb: value)
    pe_embedder = staticmethod(lambda ids: ids)

    def __init__(self, tokens, grid, blocks=2, block_cls=_MageToyBlock, refs=()):
        self.tokens, self.grid = tokens, grid
        self.refs, self.ref_calls = list(refs), 0
        self.transformer_blocks = [block_cls(index) for index in range(blocks)]

    def process_img(self, x, index=0):
        # Ids continue the tags, so a misaligned id shard shows in the query.
        first = 1
        if index == 0:
            tokens = self.tokens
        else:
            tokens = self.refs[self.ref_calls]
            first += self.tokens.shape[1] + sum(ref.shape[1] for ref in self.refs[:self.ref_calls])
            self.ref_calls += 1
        batch, count = tokens.shape[:2]
        ids = torch.arange(first, first + count, dtype=torch.float64).reshape(1, count, 1).repeat(batch, 1, 3)
        return tokens, ids, self.grid

    @staticmethod
    def time_text_embed(timestep, hidden):
        return _Modulation(1)

    @staticmethod
    def proj_out(value):
        return _channel_map(value, 9)


def _drive(harness, text, image, grid, *, pure_ulysses, blocks=2, block_cls=_MageToyBlock,
           refs=(), forward_kwargs=None):
    """Run the real bound forward once per rank, one model per rank thread."""
    models = [_ToyMage(image, grid, blocks, block_cls, refs) for _ in range(harness.world)]
    kernel = base.make_usp_attention("TORCH_FLASH")
    orders, drops = [], []

    def attn(*args, **kwargs):
        orders.append(kwargs.get("sequence_order"))
        drops.append(kwargs.get("drop_rows"))
        return kernel(*args, **kwargs)

    for model in models:
        MageFlowAdapter().inject_usp(model, InjectionContext(
            topology_sp=harness.world, usp_attention=attn, pure_ulysses=pure_ulysses))
    batch = text.shape[0]
    x = torch.zeros(batch, DIM, *grid, dtype=torch.float64)
    forward_kwargs = dict(forward_kwargs or {})
    if refs:
        forward_kwargs["ref_latents"] = [torch.zeros(batch, DIM, 1, 1, dtype=torch.float64)
                                         for _ in refs]
    outputs: list = [None] * harness.world
    errors: list = [None] * harness.world

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank]._forward(
                x, torch.zeros(batch, dtype=torch.float64), text, **forward_kwargs)
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(harness.world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    for error in errors:
        if error is not None:
            raise error
    harness.orders, harness.drops, harness.models = orders, drops, models
    return outputs


def _run(monkeypatch, text, image, grid, *, world, pure_ulysses, strict=False, refs=(), **drive):
    # mage_flow imports sp_rank by value; the projection helper reads both.
    monkeypatch.setattr(mage_flow, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
    harness = _Harness(world, strict=strict)
    rows = text.shape[1] + image.shape[1] + sum(ref.shape[1] for ref in refs)
    harness.expected_tags = [float(tag) for tag in range(1, rows + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, text, image, grid, pure_ulysses=pure_ulysses, refs=refs, **drive), harness


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_keys_arrive_in_stock_order_without_pads_and_outputs_match_one_rank(
    monkeypatch, text_rows, grid, batch
):
    text, image = _streams(text_rows, grid, batch)
    reference, _ = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True,
                            strict=True)
    stock = [float(t) for t in range(1, text_rows + image.shape[1] + 1)]
    assert len(harness.calls) == WORLD * 2
    # [all text, all image], no tag 0, through the full-axis call.
    assert all(entry == "exact" and tags == stock for entry, _kind, _index, tags in harness.calls)
    local = {(order.text_rows, order.image_rows) for order in harness.orders}
    assert local == {(-(-text_rows // WORLD), -(-image.shape[1] // WORLD))}
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid,ref_rows", REFERENCE_CASES)
def test_reference_tokens_follow_the_image_in_stock_order(
    monkeypatch, text_rows, grid, ref_rows, batch
):
    text, image = _streams(text_rows, grid, batch)
    refs = _references(text_rows, image.shape[1], ref_rows, batch)
    reference, _ = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False, refs=refs)
    outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True,
                            strict=True, refs=refs)
    stock = [float(t) for t in range(1, text_rows + image.shape[1] + sum(ref_rows) + 1)]
    assert harness.calls and all(tags == stock for *_head, tags in harness.calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_without_pure_ulysses_the_keys_stay_rank_major(monkeypatch, text_rows, grid):
    """The negative control: no descriptor, so the kernel sees rank-major keys.

    Mage names its pad rows on every topology. The stub reports a ring degree
    of 1, so they are dropped here; on a real ring the base guard refuses them.
    """
    text, image = _streams(text_rows, grid)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=False)
    expected = [tag for tag in _rank_major_tags(text_rows, image.shape[1]) if tag]
    assert all(order is None for order in harness.orders)
    assert expected != sorted(expected)
    assert harness.calls and all(tags == expected for *_head, tags in harness.calls)
