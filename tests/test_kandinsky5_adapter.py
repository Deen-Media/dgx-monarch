"""CPU contracts for Kandinsky5 padding, RoPE sharding, binding, and detection.

CFG padding trims uniform trailing rows and refuses ragged batches because
this adapter has no text mask. USP binds forward_orig and visual
self-attention directly: decoder cross-attention shares transformer_options
and must stay stock, as must the text blocks. CFG padding binds _forward.
"""
import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.kandinsky5 import Kandinsky5Adapter, _trim_context_or_raise


def test_trim_uniform_pad_is_exact():
    ctx = torch.ones(1, 6, 8)
    ctx[:, 4:] = 0.0  # driver-appended rows
    out = _trim_context_or_raise(ctx)
    assert out.shape[1] == 4
    assert torch.equal(out, ctx[:, :4])


def test_trim_noop_without_padding():
    ctx = torch.ones(2, 5, 8)
    out = _trim_context_or_raise(ctx)
    assert out.shape[1] == 5
    assert torch.equal(out, ctx)


def test_trim_conditioning_zero_out_untouched():
    # An all-zero conditioning is a real uncond, not padding: never trimmed.
    ctx = torch.zeros(1, 6, 8)
    out = _trim_context_or_raise(ctx)
    assert out.shape[1] == 6


def test_trim_ragged_batch_raises_typed():
    # When one call holds different real lengths, krea2 falls back to an additive
    # bias; kandinsky5 has no mask, so it raises a typed error and never attends a pad.
    ctx = torch.ones(2, 8, 8)
    ctx[0, 5:] = 0.0
    ctx[1, 3:] = 0.0
    with pytest.raises(UnsupportedModelError, match="maskless"):
        _trim_context_or_raise(ctx)


def test_zero_pad_rows_are_not_neutral_through_text_embeddings():
    # Why the pad is trimmed: TextEmbeddings is Linear(bias=True) then an affine
    # LayerNorm, so a zero pad row embeds to LayerNorm(bias), not zero, and the
    # text blocks and every cross-attention would attend to it.
    lin = torch.nn.Linear(4, 4)
    with torch.no_grad():
        lin.weight.zero_()
        lin.bias.copy_(torch.arange(4.0))
        embedded = torch.nn.LayerNorm(4)(lin(torch.zeros(1, 4)))
    assert float(embedded.abs().sum()) > 0.0


def test_freqs_batch1_broadcast_survives_sharding(sp):
    # Visual freqs are (1, L, 1, head_dim/2, 2, 2) while tokens are (B, L, D);
    # both shard on dim=1, so the batch-1 broadcast must survive chunking and
    # each rank's freqs slice must cover exactly its token positions.
    length = 10
    freqs = torch.arange(length).view(1, length, 1, 1, 1, 1).float().expand(1, length, 1, 4, 2, 2)
    tokens = torch.arange(length).view(1, length, 1).float().expand(2, length, 3)
    shards = []
    for rank in (0, 1):
        sp(2, rank)
        f_local, _ = base.shard_seq(freqs, dim=1)
        t_local, t_orig = base.shard_seq(tokens, dim=1)
        assert f_local.shape[0] == 1  # broadcast batch intact
        assert f_local.shape[1] == t_local.shape[1]
        # Positions align: the freqs slice is the token slice's position run.
        assert torch.equal(f_local[0, :, 0, 0, 0, 0], t_local[0, :, 0])
        shards.append(f_local)
    assert torch.equal(torch.cat(shards, dim=1).narrow(1, 0, t_orig), freqs)


def test_freqs_divisibility_pad_is_zero_and_bookkept(sp):
    # Odd length: freqs zero-pad to the SP multiple as the tokens do, and the
    # token shard's orig_len is what sp_gather trims back to.
    length = 7
    freqs = torch.randn(1, length, 1, 4, 2, 2)
    padded, orig = base.pad_seq_to_multiple(freqs, 2, dim=1)
    assert orig == length and padded.shape[1] == 8
    assert float(padded[:, length:].abs().sum()) == 0.0
    sp(2, 1)
    f_local, _ = base.shard_seq(freqs, dim=1)
    assert torch.equal(f_local, padded.chunk(2, dim=1)[1])
    _, token_orig = base.shard_seq(torch.randn(2, length, 3), dim=1)
    assert token_orig == length


_STOCK = object()


class _Mod:
    def __init__(self):
        self.forward = _STOCK


class _DecoderBlock:
    def __init__(self):
        self.self_attention = _Mod()
        self.cross_attention = _Mod()


class _EncoderBlock:
    def __init__(self):
        self.self_attention = _Mod()


class _FakeK5:
    def __init__(self, visual=3, text=2):
        self.visual_transformer_blocks = [_DecoderBlock() for _ in range(visual)]
        self.text_transformer_blocks = [_EncoderBlock() for _ in range(text)]
        self.forward_orig = _STOCK
        self._forward = _STOCK


def test_inject_usp_binds_visual_self_attention_only():
    model = _FakeK5()
    Kandinsky5Adapter().inject_usp(model, InjectionContext(topology_sp=2, usp_attention=object()))
    for block in model.visual_transformer_blocks:
        assert block.self_attention.forward is not _STOCK  # USP-routed
        assert block.cross_attention.forward is _STOCK     # stock: local q, full text k/v
    for block in model.text_transformer_blocks:
        assert block.self_attention.forward is _STOCK      # replicated text path stays stock
    assert model.forward_orig is not _STOCK
    assert model._forward is _STOCK  # pro rope rescale sees the full pre-shard grid


def test_inject_cfg_pad_forward_binds_forward():
    model = _FakeK5()
    Kandinsky5Adapter().inject_cfg_pad_forward(model)
    assert model._forward is not _STOCK
    assert model.forward_orig is _STOCK  # pure-cfg keeps the stock block loop


# The matches() trio (video and image exact-type accept, typed refusal of an
# unvetted subclass, foreign-model decline) is tabled in
# tests/test_adapter_matches.py.


def test_state_dict_keys_never_mis_detect_as_another_family():
    # kandinsky5 checkpoint keys must not match another family's signature:
    # wan's "text_embedding." and "self_attn." needles are near-misses of
    # kandinsky's plural, longer names. detect._SIGNATURES holds a kandinsky5
    # row, so detect answers "kandinsky5"; "unknown" would name no other family.
    from dgx_monarch.adapters.detect import detect_family_from_keys

    keys = [
        "model.diffusion_model.text_embeddings.in_layer.weight",
        "model.diffusion_model.pooled_text_embeddings.in_layer.weight",
        "model.diffusion_model.visual_embeddings.in_layer.weight",
        "model.diffusion_model.time_embeddings.in_layer.weight",
        "model.diffusion_model.text_transformer_blocks.0.self_attention.to_query.weight",
        "model.diffusion_model.text_transformer_blocks.0.text_modulation.out_layer.weight",
        "model.diffusion_model.visual_transformer_blocks.0.self_attention.to_query.weight",
        "model.diffusion_model.visual_transformer_blocks.0.cross_attention.to_key.weight",
        "model.diffusion_model.visual_transformer_blocks.0.visual_modulation.out_layer.weight",
        "model.diffusion_model.out_layer.out_layer.weight",
    ]
    assert detect_family_from_keys(keys) in ("unknown", "kandinsky5")
