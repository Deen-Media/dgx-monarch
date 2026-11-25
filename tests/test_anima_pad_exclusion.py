"""Anima pad rows: excluded under Ulysses, refused on ring (real Comfy, CPU).

Anima's sharded forward splits the (B, T, H, W, D) grid on H, so an odd
patched height pads whole H rows: in every frame, the last rank's chunk ends
in rows of W zero tokens. Flattened in comfy's t-major order those rows sit in
the middle of the rank's tokens, once per frame. After the block's adaLN a
zero row is a real key that stock one-GPU attention never sees. The forward
names those rows on every topology, as one segment per frame: pure Ulysses
drops them at its full-axis point, and ring or hybrid refuses class K unless
the ring_pad waiver is granted.

The model is comfy's cosmos ``MiniTrainDIT``, which ``Anima`` subclasses for
its text adapter alone; the denoise forward the adapter replaces and the
self-attention it reroutes are the base class's. Toy size, float32 (the widest
dtype comfy-kitchen's RoPE kernel takes), NaN pad rows: the harness is
tests/pad_rows_comfy_helpers.py. One frame chunked by H rows gathers in stock's
key order, so there each self-attention call must receive stock's q, k and v;
with several frames the gathered order is rank-major, so those runs compare
the key count and the output.
"""
from __future__ import annotations

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.anima import AnimaAdapter
from dgx_monarch.adapters.base import InjectionContext
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

CHANNELS, TEXT_DIM, TEXT_ROWS, HEADS, HEAD_DIM = 4, 12, 5, 2, 16
# (latent T, H, W) with spatial patch 2: patched heights 4 (nothing pads), 3
# and 5 (one pad row per frame), over one or two frames.
LATENTS = [(1, 8, 10), (1, 6, 10), (1, 10, 6), (2, 6, 10)]


@pytest.fixture(scope="module")
def comfy_cosmos():
    with real_comfy("anima-pad-exclusion", "comfy.ldm.cosmos.predict2") as modules:
        yield modules


def _toy(comfy_cosmos):
    ops, predict2 = comfy_cosmos
    model = predict2.MiniTrainDIT(
        max_img_h=64, max_img_w=64, max_frames=8, in_channels=CHANNELS,
        out_channels=CHANNELS, patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
        model_channels=HEADS * HEAD_DIM, num_blocks=2, num_heads=HEADS,
        crossattn_emb_channels=TEXT_DIM, pos_emb_cls="rope3d", pos_emb_learnable=True,
        pos_emb_interpolation="crop", use_adaln_lora=True, adaln_lora_dim=16,
        extra_per_block_abs_pos_emb=False, dtype=torch.float32, device="cpu",
        operations=ops.disable_weight_init)
    fill_parameters(model, seed=3, std=0.1)
    return model


def _tokens(latent):
    return latent[0] * (latent[1] // 2) * (latent[2] // 2)


def _inputs(latent, batch):
    generator = torch.Generator().manual_seed(10 * batch + sum(latent))
    x = torch.randn(batch, CHANNELS, *latent, generator=generator)
    timesteps = torch.tensor([0.4, 0.7])[:batch]
    context = torch.randn(batch, TEXT_ROWS, TEXT_DIM, generator=generator)
    return x, timesteps, context


def _flat(t):
    """Cosmos's (B, S, H, D) attention operands in comfy's (B, S, H*D) layout."""
    return t.reshape(t.shape[0], t.shape[1], -1)


def _kernel(predict2):
    """Stock's own op, from comfy's (B, L, H*D) layout the harness hands it."""
    def kernel(q, k, v, heads):
        shaped = [t.reshape(t.shape[0], t.shape[1], heads, -1) for t in (q, k, v)]
        return predict2.torch_attention_op(*shaped)
    return kernel


def _stock(comfy_cosmos, model, inputs):
    """Stock's class forward on a copy, recording each self-attention's operands."""
    _ops, predict2 = comfy_cosmos
    stock = replicas(model, 1)[0]
    calls: list[tuple] = []

    def record(q, k, v, transformer_options=None, preferred_attention=None):
        q, k, v = (plain(t) for t in (q, k, v))
        calls.append((_flat(q).clone(), _flat(k).clone(), _flat(v).clone()))
        return predict2.torch_attention_op(q, k, v)

    for block in stock.blocks:
        block.self_attn.attn_op = record
    with torch.no_grad():
        out = type(stock)._forward(stock, *inputs, transformer_options={})
    return out, calls


def _sharded(monkeypatch, comfy_cosmos, model, inputs, *, pure_ulysses=True, ring_world=1):
    harness = install(monkeypatch, _kernel(comfy_cosmos[1]), ring_world=ring_world)
    models = replicas(model)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        AnimaAdapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=pure_ulysses))
    outputs = run_ranks(harness, models,
                        lambda m: m._forward(*inputs, transformer_options={}))
    return outputs, harness


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("latent", LATENTS)
def test_pure_ulysses_attends_stock_keys_and_returns_stock_output(
    monkeypatch, comfy_cosmos, latent, batch
):
    model = _toy(comfy_cosmos)
    inputs = _inputs(latent, batch)
    expected, stock_calls = _stock(comfy_cosmos, model, inputs)
    nan_pads(monkeypatch)
    outputs, harness = _sharded(monkeypatch, comfy_cosmos, model, inputs)

    assert len(stock_calls) == 2
    for rank in range(WORLD):
        seen = harness.seen[rank]
        assert len(seen) == len(stock_calls)
        for (_entry, q, k, v), (stock_q, stock_k, stock_v) in zip(seen, stock_calls, strict=True):
            assert k.shape == stock_k.shape == (batch, _tokens(latent), HEADS * HEAD_DIM)
            assert not torch.isnan(k).any() and not torch.isnan(v).any()
            if latent[0] == 1:
                assert same(k, stock_k) and same(v, stock_v) and same(q, stock_q)
    assert torch.equal(outputs[0], outputs[1])
    assert not torch.isnan(outputs[0]).any()
    assert same(outputs[0], expected)


@pytest.mark.parametrize("latent", LATENTS)
def test_the_shared_nvfp4_scale_counts_only_real_rows(monkeypatch, comfy_cosmos, latent):
    """Every Linear quantizes as NVFP4 with the shared scale; each call reduces
    to the one GPU's amax, the padded H rows (NaN here) left out of both the
    flattened attention inputs and the MLP's grid."""
    model = _toy(comfy_cosmos)
    inputs = _inputs(latent, 1)
    recorder = ScaleRecorder(monkeypatch)
    stock = replicas(model, 1)[0]
    assert ScaleRecorder.install(stock) > 0
    with torch.no_grad():
        type(stock)._forward(stock, *inputs, transformer_options={})
    expected = recorder.take()[0]
    nan_pads(monkeypatch)
    harness = install(monkeypatch, _kernel(comfy_cosmos[1]))
    models = replicas(model)
    dispatch = base.make_usp_attention("TORCH_FLASH")
    for rank_model in models:
        AnimaAdapter().inject_usp(rank_model, InjectionContext(
            topology_sp=WORLD, usp_attention=dispatch, pure_ulysses=True))
        ScaleRecorder.install(rank_model, fabric_max(harness))
    run_ranks(harness, models, lambda m: m._forward(*inputs, transformer_options={}))
    assert_scales_match(expected, recorder.take(), WORLD)


@pytest.mark.parametrize("latent", [(1, 6, 10), (2, 10, 6)])
def test_a_padded_ring_render_refuses_class_k_by_default(monkeypatch, comfy_cosmos, latent):
    model = _toy(comfy_cosmos)
    with pytest.raises(base.UnsupportedModelError) as caught:
        _sharded(monkeypatch, comfy_cosmos, model, _inputs(latent, 1), pure_ulysses=False,
                 ring_world=2)
    assert_ring_pad_refusal(caught.value)


def test_a_waived_ring_render_attends_the_pad_and_differs_from_stock(monkeypatch, comfy_cosmos):
    model = _toy(comfy_cosmos)
    inputs = _inputs((1, 6, 10), 1)
    expected, _calls = _stock(comfy_cosmos, model, inputs)
    grant_ring_pad_waiver(monkeypatch)
    outputs, harness = _sharded(monkeypatch, comfy_cosmos, model, inputs, pure_ulysses=False,
                                ring_world=2)
    # One padded H row of five tokens.
    assert harness.seen[0][0][2].shape[1] == _tokens((1, 6, 10)) + 5
    assert torch.isfinite(outputs[0]).all()
    assert not same(outputs[0], expected)


def test_an_unpadded_ring_render_needs_no_waiver(monkeypatch, comfy_cosmos):
    model = _toy(comfy_cosmos)
    inputs = _inputs((1, 8, 10), 1)
    expected, _calls = _stock(comfy_cosmos, model, inputs)
    outputs, _harness = _sharded(monkeypatch, comfy_cosmos, model, inputs, pure_ulysses=False,
                                 ring_world=2)
    assert same(outputs[0], expected)
