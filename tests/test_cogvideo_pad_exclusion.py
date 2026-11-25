"""CogVideoX divisibility pad rows: excluded under Ulysses, refused on ring (real Comfy, CPU).

A CogVideoX block attends one joint [text, image] sequence. Each rank holds
``[text_local, image_local]``, and ``shard_seq`` pads each stream to the
sequence-parallel degree, so an odd text or image count leaves a pad row in
that stream's tail on the last rank. After the block's LayerNormZero a pad row
is a real key, so stock one-GPU attention has no such key and a sharded render
that attends it differs from stock. The forward names both streams' pad rows
on every topology: pure Ulysses drops them at its full-axis point and restores
stock's key order there; ring and hybrid refuse class K unless the ring_pad
waiver is granted.

Both CogVideoX layouts run: 1.5 (temporal patch 2, image RoPE that takes the
image shard) and 1.0 (spatial patch only, positional table added before the
shard). The harness is tests/pad_rows_comfy_helpers.py: comfy's own model at
toy size in float64, its stock class forward against the bound forward on two
rank threads, the pad rows filled with NaN so any that reaches a real row
shows.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.cogvideo import CogVideoXAdapter
from pad_rows_comfy_helpers import (
    WORLD,
    ScaleRecorder,
    assert_ring_pad_refusal,
    assert_scales_match,
    fabric_max,
    fill_parameters,
    grant_ring_pad_waiver,
    install,
    nan_pads,
    plain,
    real_comfy,
    replicas,
    run_ranks,
    same,
)

CHANNELS, TEXT_DIM = 4, 12
# (text rows, latent T, H, W). For 1.5 the temporal patch is 2, so T=1 and T=3
# patch to one and two frames; the spatial patch is 2 for both layouts.
SHAPES = {
    "1.5": [(4, (3, 6, 6)),     # 4 text, 18 image: nothing pads
            (5, (3, 6, 6)),     # odd text
            (4, (1, 6, 10)),    # 15 image rows: odd image
            (5, (1, 6, 10))],   # both odd
    "1.0": [(4, (2, 6, 10)),    # 30 image rows: nothing pads
            (5, (2, 6, 10)),
            (4, (1, 6, 10)),
            (5, (1, 6, 10))],
}
CASES = [(variant, *shape) for variant, shapes in SHAPES.items() for shape in shapes]


def _image_rows(variant, latent):
    frames, height, width = latent
    patched_frames = (frames + 1) // 2 if variant == "1.5" else frames
    return patched_frames * (height // 2) * (width // 2)


PADDED = [case for case in CASES
          if case[1] % WORLD or _image_rows(case[0], case[2]) % WORLD]


@pytest.fixture(scope="module")
def comfy_cogvideo():
    with real_comfy("cogvideo-pad-exclusion", "comfy.ldm.cogvideo.model") as modules:
        yield modules


def _toy(comfy_cogvideo, variant):
    ops, cog = comfy_cogvideo
    # 1.5's RoPE splits a head into quarter and three-eighths parts, so 16; the
    # 1.0 positional table splits the model width into thirds, so 2 x 24.
    layout = ({"patch_size_t": 2, "use_rotary_positional_embeddings": True,
               "attention_head_dim": 16}
              if variant == "1.5" else
              {"patch_size_t": None, "use_rotary_positional_embeddings": False,
               "attention_head_dim": 24})
    model = cog.CogVideoXTransformer3DModel(
        num_attention_heads=2, in_channels=CHANNELS,
        out_channels=CHANNELS, time_embed_dim=16, text_embed_dim=TEXT_DIM, num_layers=2,
        patch_size=2, dtype=torch.float64, device="cpu",
        operations=ops.disable_weight_init, **layout)
    fill_parameters(model, seed=7)
    return model


def _inputs(text_rows, latent, batch):
    generator = torch.Generator().manual_seed(1000 * batch + 100 * text_rows + sum(latent))
    x = torch.randn(batch, CHANNELS, *latent, generator=generator, dtype=torch.float64)
    timestep = torch.tensor([400.0, 700.0], dtype=torch.float64)[:batch]
    context = torch.randn(batch, text_rows, TEXT_DIM, generator=generator, dtype=torch.float64)
    return x, timestep, context


def _stock(monkeypatch, comfy_cogvideo, model, inputs):
    """Stock's class forward, recording each attention call's operands."""
    _ops, cog = comfy_cogvideo
    kernel = cog.optimized_attention
    calls: list[tuple] = []

    def record(q, k, v, heads, **kwargs):
        q, k, v = (plain(t) for t in (q, k, v))
        calls.append((q.clone(), k.clone(), v.clone()))
        return kernel(q, k, v, heads)

    monkeypatch.setattr(cog, "optimized_attention", record)
    try:
        with torch.no_grad():
            out = type(model)._forward(model, *inputs, transformer_options={})
    finally:
        monkeypatch.setattr(cog, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, model, inputs, kernel, *, pure_ulysses=True, ring_world=1):
    harness = install(monkeypatch, kernel, ring_world=ring_world)
    models = replicas(model)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        CogVideoXAdapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=pure_ulysses))
    outputs = run_ranks(harness, models,
                        lambda m: m._forward(*inputs, transformer_options={}))
    return outputs, harness


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("variant,text_rows,latent", CASES)
def test_pure_ulysses_attends_stock_keys_and_returns_stock_output(
    monkeypatch, comfy_cogvideo, variant, text_rows, latent, batch
):
    model = _toy(comfy_cogvideo, variant)
    inputs = _inputs(text_rows, latent, batch)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_cogvideo, model, inputs)
    nan_pads(monkeypatch)
    outputs, harness = _sharded(monkeypatch, model, inputs, kernel)

    for rank in range(WORLD):
        seen = harness.seen[rank]
        assert len(seen) == len(stock_calls) == 2
        for (entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_calls, strict=True):
            # Stock's key set in stock's order: no pad row, every real row.
            assert not torch.isnan(k).any() and not torch.isnan(v).any()
            assert k.shape == stock_k.shape, (entry, k.shape, stock_k.shape)
            assert same(k, stock_k) and same(v, stock_v) and same(q, stock_q)
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.isnan(outputs[0]).any()
    assert same(outputs[0], expected)


def _stock_row_of_each_key(k, stock_k):
    """For each key row, the stock key row it equals to rounding.

    Each rank runs a block's norm and q, k, v Linears on its own rows, and a
    CPU BLAS may round another row count apart (x86 does), so the order is read
    by matching rows, never by bits: every row of the random toy lies far from
    every other row, and each must match exactly one stock row to rounding.
    """
    assert k.shape == stock_k.shape and k.shape[0] == 1
    distances = (k[0][:, None] - stock_k[0][None]).norm(dim=-1)  # exact near zero
    nearest = distances.argmin(dim=1)
    assert sorted(nearest.tolist()) == list(range(k.shape[1]))  # one stock row each
    assert same(k, stock_k[:, nearest])
    second = distances.scatter(1, nearest[:, None], float("inf")).min(dim=1).values
    assert bool((second > 1e3 * distances.min(dim=1).values.clamp_min(1e-12)).all())
    return nearest


def _rank_major(text_rows, image_rows):
    """Stock's row indices in the order xfuser's gather lays them out."""
    text, image = torch.arange(text_rows), text_rows + torch.arange(image_rows)
    halves = [part.chunk(WORLD) for part in (text, image)]
    return torch.cat([piece for rank in range(WORLD) for piece in (halves[0][rank],
                                                                   halves[1][rank])])


@pytest.mark.parametrize("variant", ["1.5", "1.0"])
def test_at_an_even_length_the_order_descriptor_restores_stocks_key_order(
    monkeypatch, comfy_cogvideo, variant
):
    """What the order descriptor does where nothing pads. Without it the joint
    attention would take xfuser's own call, whose gathered keys run rank-major,
    [text 0, image 0, text 1, image 1]: stock's key set in another order, so a
    flash kernel sums each softmax in another order. With it the call goes to
    the full-axis point, which puts the keys back in stock's order: every
    block's key row i is stock's key row i, to rounding."""
    from dgx_monarch.adapters import cogvideo

    text_rows, latent = SHAPES[variant][0]
    model = _toy(comfy_cogvideo, variant)
    inputs = _inputs(text_rows, latent, 1)
    _expected, stock_calls, kernel = _stock(monkeypatch, comfy_cogvideo, model, inputs)
    rows = text_rows + _image_rows(variant, latent)
    _outputs, harness = _sharded(monkeypatch, model, inputs, kernel)
    assert {entry for entry, *_ in harness.seen[0]} == {"exact"}
    for (_e, _q, k, _v), (_sq, stock_k, _sv) in zip(harness.seen[0], stock_calls, strict=True):
        assert torch.equal(_stock_row_of_each_key(k, stock_k), torch.arange(rows))
    monkeypatch.setattr(cogvideo, "RankMajorJointOrder", lambda *_args: None)
    _outputs, harness = _sharded(monkeypatch, model, inputs, kernel)
    assert {entry for entry, *_ in harness.seen[0]} == {"xfuser"}
    order = _rank_major(text_rows, _image_rows(variant, latent))
    assert not torch.equal(order, torch.arange(rows))
    for (_e, _q, k, _v), (_sq, stock_k, _sv) in zip(harness.seen[0], stock_calls, strict=True):
        assert torch.equal(_stock_row_of_each_key(k, stock_k), order)


@pytest.mark.parametrize("variant,text_rows,latent", PADDED)
def test_a_padded_ring_render_refuses_class_k_by_default(
    monkeypatch, comfy_cogvideo, variant, text_rows, latent
):
    model = _toy(comfy_cogvideo, variant)
    inputs = _inputs(text_rows, latent, 1)
    _expected, _calls, kernel = _stock(monkeypatch, comfy_cogvideo, model, inputs)
    with pytest.raises(base.UnsupportedModelError) as caught:
        _sharded(monkeypatch, model, inputs, kernel, pure_ulysses=False, ring_world=2)
    assert_ring_pad_refusal(caught.value)


def test_a_waived_ring_render_attends_the_pad_and_differs_from_stock(monkeypatch, comfy_cogvideo):
    """Under the waiver the render goes ahead with the zero pad rows attended,
    which is the known-wrong math the waiver card names."""
    model = _toy(comfy_cogvideo, "1.5")
    inputs = _inputs(5, (1, 6, 10), 1)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_cogvideo, model, inputs)
    grant_ring_pad_waiver(monkeypatch)
    outputs, harness = _sharded(monkeypatch, model, inputs, kernel, pure_ulysses=False,
                                ring_world=2)
    keys = harness.seen[0][0][2]
    assert keys.shape[1] == stock_calls[0][1].shape[1] + 2   # one pad row per stream
    assert torch.isfinite(outputs[0]).all()
    assert not same(outputs[0], expected)


@pytest.mark.parametrize("variant,text_rows,latent", CASES)
def test_the_shared_nvfp4_scale_counts_only_real_rows(
    monkeypatch, comfy_cogvideo, variant, text_rows, latent
):
    """Every Linear quantizes as NVFP4 with the shared scale; each call's
    reduced amax is the one GPU's amax over the whole tensor, the pad rows
    (NaN here) left out."""
    model = _toy(comfy_cogvideo, variant)
    inputs = _inputs(text_rows, latent, 1)
    recorder = ScaleRecorder(monkeypatch)
    stock = replicas(model, 1)[0]
    assert ScaleRecorder.install(stock) > 0
    with torch.no_grad():
        type(stock)._forward(stock, *inputs, transformer_options={})
    expected = recorder.take()[0]
    nan_pads(monkeypatch)
    _o, _stock_calls, kernel = _stock(monkeypatch, comfy_cogvideo, model, inputs)
    harness = install(monkeypatch, kernel)
    models = replicas(model)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        CogVideoXAdapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=True))
        ScaleRecorder.install(rank_model, fabric_max(harness))
    run_ranks(harness, models, lambda m: m._forward(*inputs, transformer_options={}))
    assert_scales_match(expected, recorder.take(), WORLD)


@pytest.mark.parametrize("variant", ["1.5", "1.0"])
def test_an_unpadded_ring_render_needs_no_waiver(monkeypatch, comfy_cogvideo, variant):
    text_rows, latent = SHAPES[variant][0]
    model = _toy(comfy_cogvideo, variant)
    inputs = _inputs(text_rows, latent, 1)
    expected, _calls, kernel = _stock(monkeypatch, comfy_cogvideo, model, inputs)
    outputs, _harness = _sharded(monkeypatch, model, inputs, kernel, pure_ulysses=False,
                                 ring_world=2)
    assert same(outputs[0], expected)
