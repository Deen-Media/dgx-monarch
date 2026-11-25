"""Kandinsky5 visual pad rows: excluded under Ulysses, refused on ring (real Comfy, CPU).

The sharded forward runs the text blocks whole on every rank, then shards the
flattened visual stream and its 3D RoPE rows for the visual block loop. An odd
visual token count leaves a pad row at the stream's tail on the last rank. Its
RoPE row is zero, so its key is a zero vector, which still takes softmax weight
in every real row's self-attention. The forward names the pad rows on every
topology: pure Ulysses drops them at its full-axis point, and ring or hybrid
refuses class K unless the ring_pad waiver is granted.

One stream chunked in order gathers in stock's key order, so each visual
self-attention call must receive stock's q, k and v. The harness is
tests/pad_rows_comfy_helpers.py: comfy's own model at toy size in float32 (the
widest dtype comfy-kitchen's RoPE kernel takes), NaN pad rows.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext
from dgx_monarch.adapters.kandinsky5 import Kandinsky5Adapter
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

CHANNELS, TEXT_DIM, POOLED_DIM, TEXT_ROWS = 4, 12, 10, 5
# (latent T, H, W) with patch (1, 2, 2): 18 tokens, then 45, 9 and 15.
LATENTS = [(2, 6, 6), (3, 6, 10), (1, 6, 6), (1, 6, 10)]


@pytest.fixture(scope="module")
def comfy_kandinsky5():
    with real_comfy("kandinsky5-pad-exclusion", "comfy.ldm.kandinsky5.model") as modules:
        yield modules


def _toy(comfy_kandinsky5):
    ops, k5 = comfy_kandinsky5
    model = k5.Kandinsky5(
        in_visual_dim=CHANNELS, out_visual_dim=CHANNELS, in_text_dim=TEXT_DIM,
        in_text_dim2=POOLED_DIM, time_dim=16, model_dim=32, ff_dim=64, visual_embed_dim=16,
        patch_size=(1, 2, 2), num_text_blocks=1, num_visual_blocks=2, axes_dims=(4, 6, 6),
        dtype=torch.float32, device="cpu", operations=ops.disable_weight_init)
    fill_parameters(model, seed=5)
    return model


def _tokens(latent):
    return latent[0] * (latent[1] // 2) * (latent[2] // 2)


def _inputs(latent, batch):
    generator = torch.Generator().manual_seed(10 * batch + sum(latent))
    x = torch.randn(batch, CHANNELS, *latent, generator=generator)
    timestep = torch.tensor([0.4, 0.7])[:batch]
    context = torch.randn(batch, TEXT_ROWS, TEXT_DIM, generator=generator)
    pooled = torch.randn(batch, POOLED_DIM, generator=generator)
    return x, timestep, context, pooled


def _stock(monkeypatch, comfy_kandinsky5, model, inputs, tokens):
    """Stock's class forward, recording each visual self-attention's operands."""
    _ops, k5 = comfy_kandinsky5
    kernel = k5.optimized_attention
    calls: list[tuple] = []

    def record(q, k, v, heads, skip_reshape=False, **options):
        # Stock's wrapper hands (B, H, L, D) under skip_reshape; the harness
        # records comfy's (B, L, H*D) layout.
        q, k, v = (plain(t) for t in (q, k, v))
        flat = [t.transpose(1, 2).reshape(t.shape[0], t.shape[2], -1) if skip_reshape else t
                for t in (q, k, v)]
        if flat[0].shape[1] == flat[1].shape[1] == tokens:
            calls.append(tuple(t.clone() for t in flat))
        return kernel(q, k, v, heads, skip_reshape=skip_reshape)

    monkeypatch.setattr(k5, "optimized_attention", record)
    try:
        with torch.no_grad():
            out = type(model)._forward(model, *inputs, transformer_options={})
    finally:
        monkeypatch.setattr(k5, "optimized_attention", kernel)
    return out, calls, kernel


def _sharded(monkeypatch, model, inputs, kernel, *, pure_ulysses=True, ring_world=1):
    harness = install(monkeypatch, kernel, ring_world=ring_world)
    models = replicas(model)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        Kandinsky5Adapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=pure_ulysses))
    outputs = run_ranks(harness, models, lambda m: m._forward(*inputs, transformer_options={}))
    return outputs, harness


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("latent", LATENTS)
def test_pure_ulysses_attends_stock_keys_and_returns_stock_output(
    monkeypatch, comfy_kandinsky5, latent, batch
):
    model = _toy(comfy_kandinsky5)
    inputs = _inputs(latent, batch)
    expected, stock_calls, kernel = _stock(monkeypatch, comfy_kandinsky5, model, inputs,
                                           _tokens(latent))
    nan_pads(monkeypatch)
    outputs, harness = _sharded(monkeypatch, model, inputs, kernel)

    assert len(stock_calls) == 2
    for rank in range(WORLD):
        seen = harness.seen[rank]
        assert len(seen) == len(stock_calls)
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_calls, strict=True):
            assert k.shape == stock_k.shape
            assert same(k, stock_k) and same(v, stock_v) and same(q, stock_q)
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.isnan(outputs[0]).any()
    assert same(outputs[0], expected)


@pytest.mark.parametrize("latent", [*LATENTS, (9, 2, 2)])
def test_the_shared_nvfp4_scale_counts_only_real_rows(monkeypatch, comfy_kandinsky5, latent):
    """Every Linear quantizes as NVFP4 with the shared scale; each call reduces
    to the one GPU's amax. Nine visual tokens shard to five local rows, the
    text's own row count, and the text's last row is made the loudest, so a
    scale that cut the replicated cross-attention keys and values as if they
    were the visual stream would lose it."""
    model = _toy(comfy_kandinsky5)
    inputs = _inputs(latent, 1)
    inputs[2][:, -1] *= 100.0
    recorder = ScaleRecorder(monkeypatch)
    stock = replicas(model, 1)[0]
    assert ScaleRecorder.install(stock) > 0
    with torch.no_grad():
        type(stock)._forward(stock, *inputs, transformer_options={})
    expected = recorder.take()[0]
    nan_pads(monkeypatch)
    _o, _calls, kernel = _stock(monkeypatch, comfy_kandinsky5, model, inputs, _tokens(latent))
    harness = install(monkeypatch, kernel)
    models = replicas(model)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        Kandinsky5Adapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=True))
        ScaleRecorder.install(rank_model, fabric_max(harness))
    run_ranks(harness, models, lambda m: m._forward(*inputs, transformer_options={}))
    assert_scales_match(expected, recorder.take(), WORLD)


@pytest.mark.parametrize("latent", [(3, 6, 10), (1, 6, 6)])
def test_a_padded_ring_render_refuses_class_k_by_default(monkeypatch, comfy_kandinsky5, latent):
    model = _toy(comfy_kandinsky5)
    inputs = _inputs(latent, 1)
    _expected, _calls, kernel = _stock(monkeypatch, comfy_kandinsky5, model, inputs,
                                       _tokens(latent))
    with pytest.raises(base.UnsupportedModelError) as caught:
        _sharded(monkeypatch, model, inputs, kernel, pure_ulysses=False, ring_world=2)
    assert_ring_pad_refusal(caught.value)


def test_a_waived_ring_render_attends_the_pad_and_differs_from_stock(
    monkeypatch, comfy_kandinsky5
):
    model = _toy(comfy_kandinsky5)
    inputs = _inputs((3, 6, 10), 1)
    tokens = _tokens((3, 6, 10))
    expected, _calls, kernel = _stock(monkeypatch, comfy_kandinsky5, model, inputs, tokens)
    grant_ring_pad_waiver(monkeypatch)
    outputs, harness = _sharded(monkeypatch, model, inputs, kernel, pure_ulysses=False,
                                ring_world=2)
    assert harness.seen[0][0][2].shape[1] == tokens + 1
    assert torch.isfinite(outputs[0]).all()
    assert not same(outputs[0], expected)


def test_an_unpadded_ring_render_needs_no_waiver(monkeypatch, comfy_kandinsky5):
    model = _toy(comfy_kandinsky5)
    inputs = _inputs((2, 6, 6), 1)
    expected, _calls, kernel = _stock(monkeypatch, comfy_kandinsky5, model, inputs,
                                      _tokens((2, 6, 6)))
    outputs, _harness = _sharded(monkeypatch, model, inputs, kernel, pure_ulysses=False,
                                 ring_world=2)
    assert same(outputs[0], expected)
