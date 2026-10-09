"""Compare query-group attention bias with a dense mask on CPU.

Separate attention calls for query groups, each with its own additive bias,
must equal one call under the composite (T, T) mask. This checks the math of
ComfyUI's _attention_with_guide_mask in ldm/lightricks/model.py. An emulated
two-rank head scatter also checks the Ulysses layout.

Float64 inputs and a direct softmax reference isolate the algebra from kernel
accumulation differences. No ComfyUI, CUDA or xFuser is used.
"""
from __future__ import annotations

import math

import pytest
import torch

import dgx_monarch.adapters.base as base
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.usp_query_bias import (
    extend_bias_keys,
    query_group_attention,
    validate_query_bias_groups,
)

# comfy runs its masked guide calls with low_precision_attention=False, which
# sends attention_sage back to attention_pytorch and torch SDPA
# (ldm/modules/attention.py). Passing SDPA keeps this file free of comfy.
SDPA = torch.nn.functional.scaled_dot_product_attention

BATCH, HEADS, DIM = 1, 4, 8


def _qkv(tokens: int, *, heads: int = HEADS, seed: int = 0):
    """(B, L, H, D) inputs, the layout the Ulysses head scatter produces."""
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        torch.randn(BATCH, tokens, heads, DIM, generator=generator,
                    dtype=torch.float64)
        for _ in range(3)
    )


def _dense_attention(q, k, v, mask):
    """One attention over the whole query axis under one dense (T, T) mask."""
    q_h, k_h, v_h = (t.transpose(1, 2) for t in (q, k, v))
    scores = q_h @ k_h.transpose(-1, -2) / math.sqrt(q_h.shape[-1])
    if mask is not None:
        scores = scores + mask
    return (torch.softmax(scores, dim=-1) @ v_h).transpose(1, 2)


def _guide_masks(tokens: int, guide_start: int, weights: torch.Tensor):
    """comfy's two rectangles, built exactly as GuideAttentionMask builds them.

    Read from ldm/lightricks/model.py:377-393: log-space weights, zero for a
    weight of 1.0, finfo.min where a weight is 0.
    """
    finfo = torch.finfo(weights.dtype)
    positive = weights > 0
    log_w = torch.full_like(weights, finfo.min)
    log_w[positive] = torch.log(weights[positive].clamp(min=finfo.tiny))
    tracked = weights.shape[0]

    noisy = torch.zeros((1, 1, 1, tokens), dtype=weights.dtype)
    noisy[:, :, :, guide_start:guide_start + tracked] = log_w.view(1, 1, 1, -1)
    tracked_mask = torch.zeros((1, 1, tracked, tokens), dtype=weights.dtype)
    tracked_mask[:, :, :, :guide_start] = log_w.view(1, 1, -1, 1)
    return noisy, tracked_mask


def _dense_composite(tokens: int, guide_start: int, weights: torch.Tensor):
    """The (1, 1, T, T) mask comfy never materializes."""
    noisy, tracked_mask = _guide_masks(tokens, guide_start, weights)
    dense = torch.zeros((1, 1, tokens, tokens), dtype=weights.dtype)
    dense[:, :, :guide_start, :] = noisy
    dense[:, :, guide_start:guide_start + weights.shape[0], :] = tracked_mask
    return dense


def _groups(tokens: int, guide_start: int, weights: torch.Tensor):
    """The adapter's group set for this guide layout (adapters/ltx.py `_guide_bias`)."""
    noisy, tracked_mask = _guide_masks(tokens, guide_start, weights)
    tracked_end = guide_start + weights.shape[0]
    groups = [(0, guide_start, noisy), (guide_start, tracked_end, tracked_mask)]
    if tracked_end < tokens:
        groups.append((tracked_end, tokens, None))
    return tuple(groups)


@pytest.mark.parametrize("strength", [0.0, 0.3, 0.7, 0.999, 1.5, 4.0])
def test_the_query_split_equals_one_dense_composite_mask(strength):
    """One softmax per query group over the same keys equals one softmax under one mask."""
    tokens, guide_start = 12, 8
    weights = torch.full((tokens - guide_start,), strength, dtype=torch.float64)
    q, k, v = _qkv(tokens)

    split = query_group_attention(q, k, v, _groups(tokens, guide_start, weights),
                                  sdpa=SDPA)
    dense = _dense_attention(q, k, v, _dense_composite(tokens, guide_start, weights))

    assert torch.allclose(split, dense, atol=1e-12), (
        f"the query split diverges from the dense mask at strength {strength}: "
        f"max {(split - dense).abs().max().item()}")


def test_a_per_token_pixel_mask_rides_the_same_split():
    """A spatial guide mask gives each guide token its own weight.

    That is the second setting comfy builds a bias for, and the tracked
    rectangle carries it as one weight per row.
    """
    tokens, guide_start = 14, 9
    weights = torch.tensor([0.2, 0.8, 0.0, 1.0, 0.55], dtype=torch.float64)
    q, k, v = _qkv(tokens, seed=3)

    split = query_group_attention(q, k, v, _groups(tokens, guide_start, weights),
                                  sdpa=SDPA)
    dense = _dense_attention(q, k, v, _dense_composite(tokens, guide_start, weights))

    assert torch.allclose(split, dense, atol=1e-12)


def test_strength_one_is_the_unbiased_attention_bit_for_bit():
    """A weight of 1.0 must leave attention unchanged.

    comfy builds no bias when every weight is 1.0, but a strength-1.0 guide
    beside a biased one still gets weights of 1.0. A weight of 1.0 is zero in
    log space, so all-ones weights must give plain attention.
    """
    tokens, guide_start = 10, 6
    weights = torch.ones(tokens - guide_start, dtype=torch.float64)
    q, k, v = _qkv(tokens, seed=7)

    split = query_group_attention(q, k, v, _groups(tokens, guide_start, weights),
                                  sdpa=SDPA)
    plain = _dense_attention(q, k, v, None)

    assert torch.equal(split, plain) or torch.allclose(split, plain, atol=1e-13)


def test_a_zero_strength_guide_is_ignored_by_the_noisy_tokens():
    """Weight 0 becomes finfo.min, so a noisy query must not read a guide key."""
    tokens, guide_start = 9, 6
    weights = torch.zeros(tokens - guide_start, dtype=torch.float64)
    q, k, v = _qkv(tokens, seed=11)

    biased = query_group_attention(q, k, v, _groups(tokens, guide_start, weights),
                                   sdpa=SDPA)
    # The same render with the guide tokens absent from the key set.
    noisy_only = _dense_attention(q[:, :guide_start], k[:, :guide_start],
                                  v[:, :guide_start], None)

    assert torch.allclose(biased[:, :guide_start], noisy_only, atol=1e-12)


def test_a_head_split_reproduces_the_whole_attention():
    """Emulate the Ulysses head scatter at world 2.

    Each rank runs the same groups over the same full token axis for half the
    heads. The halves, concatenated, must equal the unsharded result; the bias
    applies per rank because it broadcasts over heads.
    """
    tokens, guide_start = 16, 11
    weights = torch.full((tokens - guide_start,), 0.7, dtype=torch.float64)
    groups = _groups(tokens, guide_start, weights)
    q, k, v = _qkv(tokens, heads=4, seed=5)

    whole = query_group_attention(q, k, v, groups, sdpa=SDPA)
    halves = [
        query_group_attention(q[:, :, lo:hi], k[:, :, lo:hi], v[:, :, lo:hi],
                              groups, sdpa=SDPA)
        for lo, hi in ((0, 2), (2, 4))
    ]

    assert torch.allclose(torch.cat(halves, dim=2), whole, atol=1e-12)


def test_pad_keys_extend_the_bias_with_zeros_and_join_the_unbiased_group():
    """A bias narrower than the key axis must put no bias on the extra keys.

    The extra queries must land in the trailing unbiased group, where comfy's
    third branch puts anything past the guides.
    """
    declared, pad, guide_start = 12, 2, 8
    weights = torch.full((declared - guide_start,), 0.7, dtype=torch.float64)
    padded = declared + pad
    groups = [*_groups(declared, guide_start, weights), (declared, padded, None)]
    q, k, v = _qkv(padded, seed=13)

    out = query_group_attention(q, k, v, tuple(groups), sdpa=SDPA)

    dense = torch.zeros((1, 1, padded, padded), dtype=torch.float64)
    dense[:, :, :declared, :declared] = _dense_composite(declared, guide_start, weights)
    assert torch.allclose(out, _dense_attention(q, k, v, dense), atol=1e-12)


def test_a_padded_stream_drops_its_pads_and_still_biases_its_own_rows():
    """Dropping the tail pads leaves each group biasing the tokens it named.

    The groups cover the stream's own rows, and its divisibility pads sit past
    every group boundary. Kept rows must match the dense composite over the
    unpadded sequence to 1e-12, and pad rows must come back as zeros.
    """
    from dgx_monarch.adapters.usp_pad_exclusion import attend_without_pads

    declared, pad, guide_start = 12, 2, 8
    weights = torch.full((declared - guide_start,), 0.7, dtype=torch.float64)
    groups = _groups(declared, guide_start, weights)
    q, k, v = _qkv(declared, seed=23)
    padded = [torch.cat([t, t.new_zeros(BATCH, pad, HEADS, DIM)], dim=1)
              for t in (q, k, v)]
    drop = list(range(declared, declared + pad))

    out = attend_without_pads(
        *padded, drop_rows=drop, kv_drop_rows=drop, groups=groups,
        attend=lambda q_r, k_r, v_r, g: query_group_attention(
            q_r, k_r, v_r, g, sdpa=SDPA))

    dense = _dense_attention(q, k, v,
                             _dense_composite(declared, guide_start, weights))
    assert torch.allclose(out[:, :declared], dense, atol=1e-12)
    assert torch.equal(out[:, declared:], torch.zeros_like(out[:, declared:]))


def test_extend_bias_keys_only_grows_and_only_with_zeros():
    bias = torch.full((1, 1, 3, 4), -2.0)
    grown = extend_bias_keys(bias, 7)
    assert grown.shape == (1, 1, 3, 7)
    assert torch.equal(grown[..., :4], bias)
    assert torch.equal(grown[..., 4:], torch.zeros(1, 1, 3, 3))
    assert extend_bias_keys(bias, 4) is bias
    assert extend_bias_keys(bias, 2) is bias


@pytest.mark.parametrize("groups,reason", [
    (((0, 4, None), (6, 10, None)), "gap"),
    (((0, 6, None), (4, 10, None)), "overlap"),
    (((0, 6, None),), "short"),
    (((0, 12, None),), "past the end"),
    ((), "empty"),
])
def test_a_broken_partition_refuses_before_any_kernel_call(groups, reason):
    with pytest.raises(UnsupportedModelError) as raised:
        validate_query_bias_groups(groups, 10, 10)
    assert "[dgxm:P]" in str(raised.value), reason


@pytest.mark.parametrize("bias", [
    torch.zeros(1, 1, 10),          # three axes, not (B, H, Q, K)
    torch.zeros(1, 1, 3, 10),       # neither 1 nor the group's query count
    torch.zeros(1, 1, 1, 40),       # more keys than the sequence carries
])
def test_a_bias_that_does_not_broadcast_over_its_group_refuses(bias):
    with pytest.raises(UnsupportedModelError):
        validate_query_bias_groups(((0, 10, bias),), 10, 10)


def test_an_unbiased_partition_is_plain_attention():
    """The group machinery must be inert when nothing asks for a bias."""
    q, k, v = _qkv(8, seed=17)
    out = query_group_attention(q, k, v, ((0, 8, None),), sdpa=SDPA)
    assert torch.allclose(out, _dense_attention(q, k, v, None), atol=1e-13)


# The topology guard is the last check before a bias could reach a ring kernel,
# which cannot read it. The LTX adapter refuses ring first (adapters/ltx.py) and
# the Lens adapter passes its bias only under pure Ulysses (adapters/lens.py);
# this guard is the backstop, with its own raise path.

def test_pure_ulysses_is_the_one_topology_the_guard_admits():
    """Degree 1 on the ring axis: every rank holds every key.

    The guard takes no pad count, so it admits a padded biased stream too. A
    biased stream shards alone, which puts its pads at the tail of the token
    axis, past every group boundary. test_usp_pad_exact.py proves the pad drop and
    test_a_padded_stream_drops_its_pads_and_still_biases_its_own_rows proves
    its composition with a bias.
    """
    assert base.assert_ulysses_only_query_bias(1) is None


@pytest.mark.parametrize("ring_world", [2, 4, 8])
def test_a_ring_topology_refuses_the_bias_and_names_ulysses(ring_world):
    """No ring rank holds the whole key axis a bias spans, so there is nothing
    to waive: the guard raises class P and says what does work."""
    with pytest.raises(UnsupportedModelError) as raised:
        base.assert_ulysses_only_query_bias(ring_world)

    message = str(raised.value)
    assert message.startswith("[dgxm:P]")
    assert "uly" in message and "single" in message
    assert "key axis" in message


def test_the_guard_is_not_compiled_into_one_render_s_decision():
    """A topology change must be read on every call, never compiled into a graph."""
    assert getattr(base.assert_ulysses_only_query_bias, "_torchdynamo_disable", False)
