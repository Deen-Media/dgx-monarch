"""Qwen-Image double blocks attend in stock key order (pure Ulysses).

Each rank holds `[text_r, image_r]`, so Ulysses' head all-to-all hands the
kernel `[text0, image0, text1, image1]` where stock attends `[text0, text1,
image0, image1]`, and flash rounding depends on key order. This file drives the
real bound Qwen forward through the real `base.make_usp_attention` dispatch and
the real full-axis call, with only the all-to-all, the gather and the kernel
stubbed. Every key row carries a tag, so the kernel names the exact order it was
handed. The harness is the one tests/test_flux_pad_exclusion.py builds.
"""
from __future__ import annotations

import threading

import pytest
import torch

from dgx_monarch.adapters import base, chroma_text, qwen_image
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.qwen_image import QwenImageAdapter
from test_flux_pad_exclusion import (
    _LOCAL,
    _TAG,
    BATCH,
    DIM,
    _attention,
    _channel_map,
    _Harness,
    _install_stubs,
    _matches,
    _Modulation,
    _rank,
    _ToyBlock,
)

WORLD = 2
# (text rows, image grid): even, odd text, odd image, both odd.
SHAPES = [(4, (2, 4)), (5, (2, 4)), (4, (3, 3)), (5, (3, 3))]


class _QwenToyBlock(_ToyBlock):
    """Qwen's block signature over the flux toy block, elementwise apart from attention."""

    def __call__(self, hidden_states, encoder_hidden_states, encoder_hidden_states_mask,
                 temb, image_rotary_emb, timestep_zero_index, transformer_options):
        img_q, img_k, img_v = self.qkv(hidden_states, temb)
        txt_q, txt_k, txt_v = self.qkv(encoder_hidden_states, temb)
        attended = _attention(torch.cat((txt_q, img_q), dim=1),
                              torch.cat((txt_k, img_k), dim=1),
                              torch.cat((txt_v, img_v), dim=1),
                              image_rotary_emb, transformer_options)
        rows = encoder_hidden_states.shape[1]
        return (self.residual(encoder_hidden_states, attended[:, :rows], temb),
                self.residual(hidden_states, attended[:, rows:], temb))


class _ToyQwen:
    patch_size = 2
    default_ref_method = "index"
    img_in = staticmethod(lambda value: value)
    txt_norm = staticmethod(lambda value: value)
    txt_in = staticmethod(lambda value: value)
    norm_out = staticmethod(lambda value, temb: value)
    pe_embedder = staticmethod(lambda ids: ids)

    def __init__(self, tokens, grid, blocks=2, block_cls=_QwenToyBlock, refs=()):
        self.tokens, self.grid = tokens, grid
        self.refs, self.ref_calls = list(refs), 0
        self.transformer_blocks = [block_cls(index) for index in range(blocks)]

    def process_img(self, x, index=0, h_offset=0, w_offset=0):
        # The forward calls this once for the image (index 0) and once per
        # reference latent, whose index is never 0 for any reference method.
        first = 1
        if index == 0:
            tokens = self.tokens
        else:
            tokens = self.refs[self.ref_calls]
            first += self.tokens.shape[1] + sum(ref.shape[1] for ref in self.refs[:self.ref_calls])
            self.ref_calls += 1
        batch, count = tokens.shape[:2]
        ids = torch.arange(first, first + count, dtype=torch.float64).reshape(1, count, 1).repeat(batch, 1, 3)
        rows, cols = self.grid
        return tokens, ids, (batch, 1, 1, 2 * rows, 2 * cols)

    @staticmethod
    def time_text_embed(timesteps, hidden, additional_t_cond=None):
        return _Modulation(1)

    @staticmethod
    def proj_out(value):
        return _channel_map(value, 9)


def _streams(text_rows, grid, batch=1):
    """Tagged text rows 1..T and image rows T+1..T+I, with distinct features.

    Items after the first keep every tag but get a different feature body, so a
    permutation that mishandled the batch axis would show in the outputs.
    """
    image_rows = grid[0] * grid[1]
    text = torch.zeros(BATCH, text_rows, DIM, dtype=torch.float64)
    image = torch.zeros(BATCH, image_rows, DIM, dtype=torch.float64)
    text[0, :, _TAG] = torch.arange(1, text_rows + 1, dtype=torch.float64)
    image[0, :, _TAG] = torch.arange(text_rows + 1, text_rows + image_rows + 1, dtype=torch.float64)
    for column in range(1, DIM):
        text[0, :, column] = 0.1 * column + 0.03 * torch.arange(text_rows, dtype=torch.float64)
        image[0, :, column] = 0.2 * column - 0.05 * torch.arange(image_rows, dtype=torch.float64)
    text, image = text.repeat(batch, 1, 1), image.repeat(batch, 1, 1)
    for item in range(1, batch):
        text[item, :, _TAG + 1:] *= 1.0 + 0.1 * item
        image[item, :, _TAG + 1:] *= 1.0 - 0.07 * item
    return text, image


def _references(text_rows, image_rows, ref_rows, batch=1):
    """Tagged reference tokens that continue the image tags, one tensor per reference.

    The tags follow the text rows and the image rows, so a reference token sorts
    after every image token, as it does after `torch.cat` in the forward.
    """
    refs, start = [], text_rows + image_rows
    for count in ref_rows:
        body = torch.zeros(BATCH, count, DIM, dtype=torch.float64)
        body[0, :, _TAG] = torch.arange(start + 1, start + count + 1, dtype=torch.float64)
        for column in range(1, DIM):
            body[0, :, column] = 0.3 * column + 0.04 * torch.arange(count, dtype=torch.float64)
        body = body.repeat(batch, 1, 1)
        for item in range(1, batch):
            body[item, :, _TAG + 1:] *= 1.0 + 0.05 * item
        refs.append(body)
        start += count
    return refs


def _drive(harness, text, image, grid, *, pure_ulysses, blocks=2, block_cls=_QwenToyBlock,
           refs=(), forward_kwargs=None):
    """Run the real bound forward once per rank and return what each returned.

    Each rank thread gets its own model and blocks, as each rank process does,
    so a per-block patch never races with the other rank.
    """
    models = [_ToyQwen(image, grid, blocks, block_cls, refs) for _ in range(harness.world)]
    kernel = base.make_usp_attention("TORCH_FLASH")
    orders, drops = [], []

    def attn(*args, **kwargs):
        orders.append(kwargs.get("sequence_order"))
        drops.append(kwargs.get("drop_rows"))
        return kernel(*args, **kwargs)

    for model in models:
        QwenImageAdapter().inject_usp(model, InjectionContext(
            topology_sp=harness.world, usp_attention=attn, pure_ulysses=pure_ulysses))
    batch = text.shape[0]
    x = torch.zeros(batch, 1, 1, 2 * grid[0], 2 * grid[1], dtype=torch.float64)
    forward_kwargs = dict(forward_kwargs or {})
    if refs:
        forward_kwargs["ref_latents"] = [torch.zeros(batch, 1, 1, 2, 2, dtype=torch.float64)
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
    monkeypatch.setattr(qwen_image, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
    harness = _Harness(world, strict=strict)
    rows = text.shape[1] + image.shape[1] + sum(ref.shape[1] for ref in refs)
    harness.expected_tags = [float(tag) for tag in range(1, rows + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, text, image, grid, pure_ulysses=pure_ulysses, refs=refs, **drive), harness


def _rank_major_tags(text_rows, image_rows):
    """The key order a rank-major gather hands the kernel, pad rows as tag 0."""
    text_local, image_local = -(-text_rows // WORLD), -(-image_rows // WORLD)
    order = []
    for rank in range(WORLD):
        for start, local, total, offset in ((rank * text_local, text_local, text_rows, 0),
                                            (rank * image_local, image_local, image_rows, text_rows)):
            order += [float(offset + start + i + 1) if start + i < total else 0.0
                      for i in range(local)]
    return order


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_keys_arrive_in_stock_order_and_outputs_return_to_their_rank(
    monkeypatch, text_rows, grid, batch
):
    text, image = _streams(text_rows, grid, batch)
    reference, _ = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True)
    stock = [float(t) for t in range(1, text_rows + image.shape[1] + 1)]
    calls = [call for call in harness.calls if call[1] == "double"]
    assert len(calls) == WORLD * 2
    # [all text rows, then all image rows], pad rows gone, through the
    # full-axis path rather than xfuser's own call.
    assert all(entry == "exact" and tags == stock for entry, _kind, _index, tags in calls)
    local = {(order.text_rows, order.image_rows) for order in harness.orders}
    assert local == {(-(-text_rows // WORLD), -(-image.shape[1] // WORLD))}
    # The inverse permutation returned each row to its rank: every rank holds
    # the whole image, and it is the one-rank answer for each batch item.
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid", [(4, (2, 4)), (6, (2, 4))])
def test_an_evenly_sharded_render_takes_the_full_axis_path_and_matches_one_rank(
    monkeypatch, text_rows, grid
):
    """No pad anywhere: restoring the order is the whole difference.

    Every pure-Ulysses render, padded or not, names an order, so the dispatcher
    sends it through the full-axis call and never reaches xfuser's own.
    """
    text, image = _streams(text_rows, grid)
    reference, ref_harness = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=True)
    stock = [float(t) for t in range(1, text_rows + image.shape[1] + 1)]
    assert all(tags == stock for *_head, tags in ref_harness.calls)
    assert harness.calls and all(
        entry == "exact" and tags == stock for entry, _kind, _index, tags in harness.calls)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_without_pure_ulysses_the_keys_stay_rank_major(monkeypatch, text_rows, grid):
    """A ring or hybrid worker context never receives the descriptor."""
    text, image = _streams(text_rows, grid)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=WORLD, pure_ulysses=False)
    expected = _rank_major_tags(text_rows, image.shape[1])
    assert all(order is None for order in harness.orders)
    # The negative control: rank-major differs from stock, so the test above
    # would fail on a harness that never reordered.
    assert [t for t in expected if t] != sorted(t for t in expected if t)
    for _entry, kind, _index, tags in harness.calls:
        if kind == "double":
            assert tags == expected
