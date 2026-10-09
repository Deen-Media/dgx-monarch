"""Hunyuan double blocks attend in stock key order under pure Ulysses.

Each double block joins `[text_r, image_r]` per rank, so Ulysses' head
all-to-all hands the kernel `[text0, image0, text1, image1]` where stock
attends `[all text, all image]`, and flash rounding depends on key order. The
single loop shards one joined stream in contiguous chunks, which gathers in
stock order already.

This file drives the real bound `forward_orig` through the real
`base.make_usp_attention` dispatch and full-axis call. Only the all-to-all, the
gather and the kernel are stubbed, by the harness in
tests/test_flux_pad_exclusion.py. Every key row carries a tag, so the kernel
names the rows it was handed; a pad row carries tag 0. The pad rows themselves
are tests/test_hunyuan_pad_exclusion.py's claim, so this file reads order with
them left out.
"""
from __future__ import annotations

import sys
import threading
import types
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch.adapters import base, chroma_text, hunyuan
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.hunyuan import HunyuanAdapter
from test_flux_pad_exclusion import (
    _CONTEXT,
    _LOCAL,
    _TAG,
    DIM,
    _channel_map,
    _Harness,
    _install_stubs,
    _matches,
    _Modulation,
    _rank,
    _ToyDoubleBlock,
    _ToySingleBlock,
)

WORLD = 2
# (Qwen text rows, byt5 rows, image grid). The text stream the forward shards is
# Qwen plus byt5, so its parity is the sum: 3 + 2 is odd, 4 + 2 even. The
# joint stream in the single loop is text plus image, which pads when exactly
# one of the two is odd.
SHAPES = [
    (4, 2, (2, 4)),   # text 6, image 8, joint 14: nothing pads
    (3, 2, (2, 4)),   # text 5, image 8, joint 13: text and joint pad
    (4, 2, (3, 3)),   # text 6, image 9, joint 15: image and joint pad
    (3, 2, (3, 3)),   # text 5, image 9, joint 14: text and image pad
    (5, 0, (2, 3)),   # no glyph tokens: text 5, image 6, joint 11
]


@pytest.fixture(autouse=True)
def comfy_layers_stub(monkeypatch):
    """The one comfy import the forward makes, with Hunyuan's keyword."""
    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    flux_pkg = types.ModuleType("comfy.ldm.flux")
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.timestep_embedding = lambda t, dim, **_kwargs: torch.zeros(*t.shape, dim)
    comfy.ldm, ldm.flux, flux_pkg.layers = ldm, flux_pkg, layers
    for name, module in (("comfy", comfy), ("comfy.ldm", ldm),
                         ("comfy.ldm.flux", flux_pkg),
                         ("comfy.ldm.flux.layers", layers)):
        monkeypatch.setitem(sys.modules, name, module)


class _HunyuanDouble(_ToyDoubleBlock):
    """Hunyuan's double-block keywords over the flux toy block."""

    def __call__(self, img, txt, vec, pe, attn_mask=None, modulation_dims_img=None,
                 modulation_dims_txt=None, transformer_options=None):
        return super().__call__(img, txt, vec, pe, attn_mask, transformer_options)


class _HunyuanSingle(_ToySingleBlock):
    def __call__(self, img, vec, pe, attn_mask=None, modulation_dims=None,
                 transformer_options=None):
        return super().__call__(img, vec, pe, attn_mask, transformer_options)


class _ToyHunyuan:
    """The HunyuanImage 2.1 path: patch 1, byt5 after text, no cond types."""

    patch_size = [1, 1]
    out_channels = DIM
    time_r_in = vector_in = cond_type_embedding = vision_in = None
    params = SimpleNamespace(guidance_embed=False, vec_in_dim=None, meanflow_sum=False)
    time_in = staticmethod(lambda embedded: _Modulation(1))
    txt_in = staticmethod(lambda txt, timesteps, mask, transformer_options=None: txt)
    byt5_in = staticmethod(lambda byt5: byt5)
    pe_embedder = staticmethod(lambda ids: ids)

    def __init__(self, double_cls=_HunyuanDouble, doubles=2, singles=2):
        self.double_blocks = [double_cls(index) for index in range(doubles)]
        self.single_blocks = [_HunyuanSingle(10 + index) for index in range(singles)]

    @staticmethod
    def img_in(latent):
        return latent.flatten(2).transpose(1, 2)

    @staticmethod
    def img_ids(ref_latent):
        rows = ref_latent.shape[-2] * ref_latent.shape[-1]
        ids = 100.0 + torch.arange(rows, dtype=torch.float64)
        return ids.reshape(1, rows, 1).repeat(ref_latent.shape[0], 1, DIM - 1)

    @staticmethod
    def final_layer(img, vec, modulation_dims=None):
        return _channel_map(img, 9)


def _tagged(count, first, batch, scale):
    """`count` rows tagged first..first+count-1, a distinct body per batch item."""
    rows = torch.zeros(1, count, DIM, dtype=torch.float64)
    rows[0, :, _TAG] = torch.arange(first, first + count, dtype=torch.float64)
    for column in range(1, DIM):
        rows[0, :, column] = scale * column + 0.03 * torch.arange(count, dtype=torch.float64)
    rows = rows.repeat(batch, 1, 1)
    for item in range(1, batch):
        rows[item, :, _TAG + 1:] *= 1.0 + 0.1 * item
    return rows


def _inputs(qwen_rows, byt5_rows, grid, batch=1, ref_grid=None):
    """Forward arguments whose tags run text, byt5, reference, image in stock order."""
    text_rows = qwen_rows + byt5_rows
    ref_rows = ref_grid[0] * ref_grid[1] if ref_grid else 0
    image_rows = grid[0] * grid[1]
    txt = _tagged(qwen_rows, 1, batch, 0.1)
    byt5 = _tagged(byt5_rows, qwen_rows + 1, batch, 0.15) if byt5_rows else None
    image = _tagged(image_rows, text_rows + ref_rows + 1, batch, 0.2)
    latent = image.transpose(1, 2).reshape(batch, DIM, *grid)
    kwargs = {}
    if ref_grid:
        ref = _tagged(ref_rows, text_rows + 1, batch, 0.3)
        kwargs["ref_latent"] = ref.transpose(1, 2).reshape(batch, DIM, *ref_grid)
    txt_ids = torch.arange(1, qwen_rows + 1, dtype=torch.float64).reshape(1, -1, 1).repeat(
        batch, 1, DIM - 1)
    img_ids = (200.0 + torch.arange(image_rows, dtype=torch.float64)).reshape(1, -1, 1).repeat(
        batch, 1, DIM - 1)
    return dict(img=latent, img_ids=img_ids, txt=txt, txt_ids=txt_ids, txt_mask=None,
                timesteps=torch.zeros(batch, dtype=torch.float64), txt_byt5=byt5, **kwargs)



def _drive(harness, inputs, *, pure_ulysses, double_cls=_HunyuanDouble):
    """Run the real bound forward once per rank, each rank on its own model."""
    models = [_ToyHunyuan(double_cls) for _ in range(harness.world)]
    kernel = base.make_usp_attention("TORCH_FLASH")
    orders, drops = [], []

    def attn(*args, **kwargs):
        kind = getattr(_CONTEXT, "where", (None, None))[0]
        orders.append((kind, kwargs.get("sequence_order")))
        drops.append((kind, kwargs.get("drop_rows")))
        return kernel(*args, **kwargs)

    for model in models:
        HunyuanAdapter().inject_usp(model, InjectionContext(
            topology_sp=harness.world, usp_attention=attn, pure_ulysses=pure_ulysses))
    outputs: list = [None] * harness.world
    errors: list = [None] * harness.world

    def body(rank):
        try:
            _LOCAL.rank = rank
            outputs[rank] = models[rank].forward_orig(**inputs)
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


def _run(monkeypatch, inputs, total_rows, *, world, pure_ulysses, strict=False, **drive):
    monkeypatch.setattr(hunyuan, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_rank", _rank)
    monkeypatch.setattr(chroma_text, "sp_world", lambda: world)
    harness = _Harness(world, strict=strict)
    harness.expected_tags = [float(tag) for tag in range(1, total_rows + 1)]
    _install_stubs(monkeypatch, harness)
    return _drive(harness, inputs, pure_ulysses=pure_ulysses, **drive), harness


def _stock_tags(rows):
    return [float(tag) for tag in range(1, rows + 1)]


def _real(tags):
    """The tags of real rows; a pad row carries 0."""
    return [tag for tag in tags if tag]


def _rank_major(text_rows, image_rows):
    """The double-block key order without the treatment, pad rows as tag 0."""
    text_local, image_local = -(-text_rows // WORLD), -(-image_rows // WORLD)
    order = []
    for rank in range(WORLD):
        for start, local, total, offset in ((rank * text_local, text_local, text_rows, 0),
                                            (rank * image_local, image_local, image_rows,
                                             text_rows)):
            order += [float(offset + start + i + 1) if start + i < total else 0.0
                      for i in range(local)]
    return order


def _padded_tail(rows):
    return _stock_tags(rows) + [0.0] * ((-rows) % WORLD)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("qwen_rows,byt5_rows,grid", SHAPES)
def test_double_block_keys_arrive_in_stock_order(
    monkeypatch, qwen_rows, byt5_rows, grid, batch
):
    inputs = _inputs(qwen_rows, byt5_rows, grid, batch)
    text_rows, image_rows = qwen_rows + byt5_rows, grid[0] * grid[1]
    _outputs, harness = _run(monkeypatch, inputs, text_rows + image_rows, world=WORLD,
                             pure_ulysses=True)
    stock = _stock_tags(text_rows + image_rows)
    doubles = [call for call in harness.calls if call[1] == "double"]
    singles = [call for call in harness.calls if call[1] == "single"]
    assert len(doubles) == len(singles) == WORLD * 2
    # [all text, then all image], through the full-axis call.
    assert all(entry == "exact" and _real(tags) == stock
               for entry, _kind, _index, tags in doubles)
    # The joined stream gathers in stock order without a descriptor.
    assert all(_real(tags) == stock for *_head, tags in singles)
    text_local, image_local = -(-text_rows // WORLD), -(-image_rows // WORLD)
    named = {(kind, order.text_rows, order.image_rows)
             for kind, order in harness.orders if order is not None}
    assert named == {("double", text_local, image_local)}
    assert all(order is None for kind, order in harness.orders if kind == "single")


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("qwen_rows,byt5_rows,grid", [(4, 2, (2, 4)), (6, 2, (2, 2))])
def test_an_unpadded_render_matches_one_rank(monkeypatch, qwen_rows, byt5_rows, grid, batch):
    """Nothing pads, so the order is the only difference, and every
    pure-Ulysses double block still takes the full-axis call."""
    inputs = _inputs(qwen_rows, byt5_rows, grid, batch)
    total = qwen_rows + byt5_rows + grid[0] * grid[1]
    reference, ref_harness = _run(monkeypatch, inputs, total, world=1, pure_ulysses=False,
                                  strict=True)
    outputs, harness = _run(monkeypatch, inputs, total, world=WORLD, pure_ulysses=True,
                            strict=True)
    stock = _stock_tags(total)
    assert all(tags == stock for *_head, tags in ref_harness.calls)
    assert harness.calls and all(tags == stock for *_head, tags in harness.calls)
    assert {entry for entry, kind, *_rest in harness.calls if kind == "double"} == {"exact"}
    assert all(torch.equal(out, outputs[0]) for out in outputs)
    assert _matches(outputs[0], reference[0])


@pytest.mark.parametrize("qwen_rows,byt5_rows,grid", SHAPES)
def test_without_pure_ulysses_nothing_changes(monkeypatch, qwen_rows, byt5_rows, grid):
    """Ring and hybrid name no order and no drop rows, so the ring pad guard
    cannot fire and the kernel gets the rank-major keys, pads included."""
    inputs = _inputs(qwen_rows, byt5_rows, grid)
    text_rows, image_rows = qwen_rows + byt5_rows, grid[0] * grid[1]
    _outputs, harness = _run(monkeypatch, inputs, text_rows + image_rows, world=WORLD,
                             pure_ulysses=False)
    assert all(order is None for _kind, order in harness.orders)
    assert all(drop is None for _kind, drop in harness.drops)
    double = _rank_major(text_rows, image_rows)
    single = _padded_tail(text_rows + image_rows)
    for entry, kind, _index, tags in harness.calls:
        assert entry == "xfuser"
        assert tags == (double if kind == "double" else single)
    # The negative control: rank-major order is not stock order here.
    assert _real(double) != sorted(_real(double))


@pytest.mark.parametrize("pure_ulysses", [True, False])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("qwen_rows,byt5_rows,grid,ref_grid,img_slice", [
    (3, 2, (2, 4), None, [5, 13]),      # joint 13 shards to 7 rows per rank
    (3, 2, (2, 3), (1, 3), [5, 14]),    # 3 reference rows join the image stream
])
def test_single_blocks_see_stocks_full_img_slice(
    monkeypatch, qwen_rows, byt5_rows, grid, ref_grid, img_slice, batch, pure_ulysses
):
    """Stock sets `img_slice` to [text, text + image] over the joined stream.
    A replaced single block or an attention patch reads it, and the block's
    rows are a shard, so the shard length would name the wrong split."""
    inputs = _inputs(qwen_rows, byt5_rows, grid, batch, ref_grid=ref_grid)
    seen = []

    def replace(args, extra):
        seen.append((_rank(), list(args["transformer_options"]["img_slice"])))
        return extra["original_block"](args)

    inputs["transformer_options"] = {"patches_replace": {"dit": {("single_block", 0): replace}}}
    _run(monkeypatch, inputs, img_slice[1], world=WORLD, pure_ulysses=pure_ulysses)
    assert sorted(seen) == [(rank, img_slice) for rank in range(WORLD)]


def test_a_render_with_no_text_rows_keeps_working(monkeypatch):
    """With no text the gathered order is already stock order, and the
    descriptor refuses an empty text segment, so none is built."""
    inputs = _inputs(0, 0, (2, 3))
    reference, _ = _run(monkeypatch, inputs, 6, world=1, pure_ulysses=False, strict=True)
    outputs, harness = _run(monkeypatch, inputs, 6, world=WORLD, pure_ulysses=True, strict=True)
    assert all(order is None for _kind, order in harness.orders)
    assert all(tags == _stock_tags(6) for *_head, tags in harness.calls)
    assert _matches(outputs[0], reference[0])
