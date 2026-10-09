"""Ideogram4 attends stock's keys in stock's order, with no pad row (pure Ulysses).

The forward packs `[text, image]` into one stream and shards it in contiguous
chunks, so the Ulysses head all-to-all hands the kernel the keys in stock's
order with no reordering. A stream the sequence-parallel degree does not divide
ends in zero pad rows. Ideogram4's modulation only scales and RMSNorm keeps a
zero row at zero, but a pad row is still a key: its logit of 0 takes softmax
weight in the first block, and from the second block on it carries the mean of
the values. Under pure Ulysses the forward names the pad rows as `drop_rows`; ring
and hybrid name nothing.

This drives the real bound forward through the real `base.make_usp_attention`
dispatch with the harness in tests/test_flux_pad_exclusion.py: only the
all-to-all, the sequence gather and the kernel are stubs, and every key row
carries a tag, so the kernel names the rows it was handed and their order. The
two comfy leaves the forward imports are stubbed, so no ComfyUI is needed;
tests/test_ideogram4_stock_equivalence.py runs comfy's real model.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest
import torch

from dgx_monarch.actor.attention_context import injection_context
from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.pixeldit import Ideogram4Adapter
from dgx_monarch.topology import PRESETS
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
    _ToyBlock,
)

LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR = 3, 2
# (text rows, image grid). None is the unconditional model's image-only call.
# The packed total decides the pad: 4 + 8 and 5 + 9 divide two, 5 + 8 and
# 4 + 9 pad one row, and an image-only call pads when the grid is odd.
SHAPES = [
    (4, (2, 4)), (5, (2, 4)), (4, (3, 3)), (5, (3, 3)),
    (None, (2, 4)), (None, (3, 3)),
]
PADDED = [(5, (2, 4)), (4, (3, 3)), (None, (3, 3))]


@pytest.fixture
def comfy_stub(monkeypatch):
    """The two comfy modules the forward imports inside its body.

    RoPE is replaced by the row's own position ids, which the toy block folds
    into its query, so a position shard misaligned with the token shard shows
    in the output.
    """
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    package = types.ModuleType("comfy.ldm.ideogram4")
    model = types.ModuleType("comfy.ldm.ideogram4.model")
    text_encoders = types.ModuleType("comfy.text_encoders")
    llama = types.ModuleType("comfy.text_encoders.llama")
    model.LLM_TOKEN_INDICATOR = LLM_TOKEN_INDICATOR
    model.OUTPUT_IMAGE_INDICATOR = OUTPUT_IMAGE_INDICATOR
    model._split_half_rope_matrix = lambda freqs: freqs
    llama.precompute_freqs_cis = (
        lambda head_dim, ids, theta, rope_dims=None, interleaved_mrope=False, device=None:
        ids.transpose(0, 1).to(torch.float64) * 0.01)
    comfy.ldm, comfy.text_encoders = ldm, text_encoders
    ldm.ideogram4, package.model, text_encoders.llama = package, model, llama
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm),
                         ("comfy.ldm.ideogram4", package),
                         ("comfy.ldm.ideogram4.model", model),
                         ("comfy.text_encoders", text_encoders),
                         ("comfy.text_encoders.llama", llama)):
        monkeypatch.setitem(sys.modules, name, module)


class _Recording:
    """An identity map that records the row count and layout it was handed."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, bool]] = []

    def __call__(self, value):
        self.seen.append((value.shape[1], value.is_contiguous()))
        return value


class _ToyLayer(_ToyBlock):
    """Ideogram4's block signature over the flux toy's elementwise attention."""

    def __call__(self, h, attn_mask, freqs_cis, adaln_input, transformer_options):
        assert attn_mask is None
        # The time embedding is per batch, never per token, so nothing pads it.
        assert adaln_input.shape == (h.shape[0], 1, DIM)
        mod = _Modulation(self.index)
        q, k, v = self.qkv(h, mod)
        return self.residual(h, _attention(q, k, v, freqs_cis, transformer_options), mod)


class _ToyIdeogram:
    """The attributes the bound forward reads, with tags carried on channel 0.

    `input_proj` and `llm_cond_proj` are identities, so the image rows keep
    their tags through the image mask and the text rows take theirs from the
    context. The indicator embedding leaves channel 0 at zero.
    """

    head_dim, rope_theta, mrope_section = DIM, 10000.0, (1, 1, 0)

    def __init__(self, image, blocks=2):
        self.image = image
        self.layers = [_ToyLayer(index) for index in range(blocks)]
        self.input_proj = _Recording()
        self.llm_cond_norm = _Recording()
        self.llm_cond_proj = _Recording()

    @staticmethod
    def t_embedding(timesteps, dtype):
        return torch.zeros(timesteps.shape[0], DIM, dtype=dtype)

    @staticmethod
    def adaln_proj(value):
        return value

    @staticmethod
    def embed_image_indicator(indices, out_dtype):
        embedded = torch.zeros(*indices.shape, DIM, dtype=out_dtype)
        embedded[..., 1:] = 0.05 * indices.unsqueeze(-1).to(out_dtype)
        return embedded

    def _img_to_tokens(self, x):
        assert x.shape[-2] * x.shape[-1] == self.image.shape[1]
        return self.image

    @staticmethod
    def _image_position_ids(gh, gw, device):
        rows = torch.arange(gh, device=device).view(-1, 1).expand(gh, gw).reshape(-1)
        cols = torch.arange(gw, device=device).view(1, -1).expand(gh, gw).reshape(-1)
        return torch.stack([torch.zeros_like(rows), rows, cols], dim=1) + 50

    @staticmethod
    def final_layer(h, adaln_input):
        return _channel_map(h, 9)

    @staticmethod
    def _tokens_to_img(tokens, gh, gw):
        return tokens


def _streams(text_rows, grid, batch=1):
    """Tagged text rows 1..T and image rows T+1..T+I; later batch items differ."""
    image_rows = grid[0] * grid[1]
    first = text_rows or 0
    image = torch.zeros(BATCH, image_rows, DIM, dtype=torch.float64)
    image[0, :, _TAG] = torch.arange(first + 1, first + image_rows + 1, dtype=torch.float64)
    for column in range(1, DIM):
        image[0, :, column] = 0.2 * column - 0.05 * torch.arange(image_rows, dtype=torch.float64)
    image = image.repeat(batch, 1, 1)
    for item in range(1, batch):
        image[item, :, _TAG + 1:] *= 1.0 - 0.07 * item
    if text_rows is None:
        return None, image
    text = torch.zeros(BATCH, text_rows, DIM, dtype=torch.float64)
    text[0, :, _TAG] = torch.arange(1, text_rows + 1, dtype=torch.float64)
    for column in range(1, DIM):
        text[0, :, column] = 0.1 * column + 0.03 * torch.arange(text_rows, dtype=torch.float64)
    text = text.repeat(batch, 1, 1)
    for item in range(1, batch):
        text[item, :, _TAG + 1:] *= 1.0 + 0.1 * item
    return text, image


def _run(monkeypatch, text, image, grid, *, world, pure_ulysses, strict=True):
    """Run the bound forward once per rank thread; return outputs and harness."""
    harness = _Harness(world, strict=strict)
    rows = image.shape[1] + (0 if text is None else text.shape[1])
    harness.expected_tags = [float(tag) for tag in range(1, rows + 1)]
    _install_stubs(monkeypatch, harness)
    kernel = base.make_usp_attention("TORCH_FLASH")
    drops: list = []

    def attn(*args, **kwargs):
        drops.append(kwargs.get("drop_rows"))
        return kernel(*args, **kwargs)

    models = [_ToyIdeogram(image) for _ in range(world)]
    for model in models:
        Ideogram4Adapter().inject_usp(model, InjectionContext(
            topology_sp=world, usp_attention=attn, pure_ulysses=pure_ulysses))
    batch = image.shape[0]
    x = torch.zeros(batch, 1, *grid, dtype=torch.float64)
    outputs: list = [None] * world
    errors: list = [None] * world

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank]._forward(
                x, torch.zeros(batch, dtype=torch.float64), text,
                attention_mask=None, transformer_options={})
        except BaseException as exc:  # re-raised in the test body
            errors[rank] = exc
            harness.fabric.barrier.abort()

    threads = [threading.Thread(target=body, args=(rank,)) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=90)
    for error in errors:
        if error is not None:
            raise error
    harness.drops, harness.models = drops, models
    return outputs, harness


def _padded_total(rows, world):
    return -(-rows // world) * world


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_keys_arrive_in_stock_order_without_the_pad_row(
    monkeypatch, comfy_stub, text_rows, grid, batch, world
):
    """The whole claim: stock's keys, stock's order, the one-rank answer."""
    text, image = _streams(text_rows, grid, batch)
    reference, _ = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, text, image, grid, world=world, pure_ulysses=True)
    rows = image.shape[1] + (text_rows or 0)
    stock = [float(tag) for tag in range(1, rows + 1)]
    padded = rows % world != 0
    assert len(harness.calls) == world * 2
    # A padded stream takes the full-axis call, which drops the tail; an
    # unpadded one keeps xfuser's own call, because this family names no order.
    assert all(entry == ("exact" if padded else "xfuser") and tags == stock
               for entry, _kind, _index, tags in harness.calls)
    assert all(drops == list(range(rows, _padded_total(rows, world)))
               for drops in harness.drops)
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert outputs[0].shape == (batch, image.shape[1], DIM)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_ring_and_hybrid_keep_attended_pads_and_never_refuse(
    monkeypatch, comfy_stub, text_rows, grid
):
    """Without pure Ulysses no drop rows are named, so the ring pad guard
    cannot fire and the kernel still attends any pad row."""
    text, image = _streams(text_rows, grid)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=2,
                             pure_ulysses=False, strict=False)
    rows = image.shape[1] + (text_rows or 0)
    expected = [float(tag) for tag in range(1, rows + 1)] + [0.0] * (rows % 2)
    assert harness.drops and all(drops is None for drops in harness.drops)
    assert all(entry == "xfuser" and tags == expected
               for entry, _kind, _index, tags in harness.calls)


@pytest.mark.parametrize("text_rows,grid", PADDED)
def test_an_attended_pad_row_changes_the_answer(monkeypatch, comfy_stub, text_rows, grid):
    """The negative control: without it this file would pass on a toy whose
    pad key wins no softmax weight. The attended pad must move the real rows
    materially, not in the last bit."""
    text, image = _streams(text_rows, grid)
    reference, _ = _run(monkeypatch, text, image, grid, world=1, pure_ulysses=False)
    outputs, harness = _run(monkeypatch, text, image, grid, world=2,
                            pure_ulysses=False, strict=False)
    assert any(0.0 in tags for *_head, tags in harness.calls)
    assert (outputs[0] - reference[0]).norm() / reference[0].norm() > 1e-4


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("text_rows,grid", SHAPES)
def test_embeddings_run_on_every_row_before_the_shard(
    monkeypatch, comfy_stub, text_rows, grid, batch
):
    """No row-count-dependent text GEMM runs on a shard, so none is wrapped.

    `llm_cond_proj` projects the text on all text rows and `input_proj` the
    packed stream on all rows, both contiguous, before `shard_seq`, as stock's
    `_backbone` does, so these embeddings need no full-row projection. The
    per-block Linears still run on shard rows, untreated.
    """
    text, image = _streams(text_rows, grid, batch)
    _outputs, harness = _run(monkeypatch, text, image, grid, world=2, pure_ulysses=True)
    rows = image.shape[1] + (text_rows or 0)
    for model in harness.models:
        assert model.input_proj.seen == [(rows, True)]
        if text_rows is None:
            assert model.llm_cond_norm.seen == model.llm_cond_proj.seen == []
        else:
            assert model.llm_cond_norm.seen == [(text_rows, True)]
            assert model.llm_cond_proj.seen == [(text_rows, True)]


def test_the_ring_pad_guard_the_gate_avoids_still_refuses():
    """Pins the ring pad guard in base.py, which gating `drop_rows` on pure
    Ulysses keeps clear of; the forward-level proof is the ring test above."""
    with pytest.raises(base.UnsupportedModelError):
        base.assert_ulysses_only_padding(2, 1)


@pytest.mark.parametrize("preset,pure", [("uly2+fsdp", True), ("ring2+fsdp", False)])
def test_fsdp_presets_follow_their_sequence_topology(preset, pure):
    """FSDP shards weights, not the sequence, so the gate reads only the
    Ulysses and ring degrees: uly2+fsdp drops the pad row as uly2 does, and
    ring2+fsdp attends it as ring2 does (docs/ADAPTERS.md says both)."""
    topology = dict(PRESETS[preset])
    context = injection_context(topology, 2, object())
    assert context.pure_ulysses is pure
