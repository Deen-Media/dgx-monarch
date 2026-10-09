"""CogVideoX adapter layout invariants (CPU, no distributed init).

CogVideoX applies RoPE to image tokens only, from a precomputed per-token
(cos, sin) table. There are no position ids to re-embed locally, so the
adapter shards the table itself. These tests pin what makes that exact: the
table takes the same divisibility pad and chunk as the image tokens (chunking
the unpadded table gives a later rank a table that does not match its token
shard), and the block's local [text, image] slice lines up with the sharded
table for every length parity.
"""
import types

import pytest
import torch

from dgx_monarch.adapters import base
from dgx_monarch.adapters.base import InjectionContext, UnsupportedModelError
from dgx_monarch.adapters.cogvideo import CogVideoXAdapter, _shard_rope


def _tokens(start, length):
    # Distinct per-position values: rope-table row i must land on token i, or
    # the mismatch shows.
    return torch.arange(start, start + length).view(1, length, 1).float()


@pytest.mark.parametrize("img_len", [8, 7])  # divisible and pad-needing lengths
def test_rope_table_shards_like_the_image_tokens(sp, img_len):
    img = _tokens(200, img_len)
    cos = img[0].clone()          # (L, d) table, row i carries token i's value
    sin = -img[0].clone()         # distinct table; -0 == 0 keeps the pad-row check exact
    for rank in (0, 1):
        sp(2, rank)
        img_local, _ = base.shard_seq(img, dim=1)
        cos_local, sin_local = _shard_rope((cos, sin))

        # Same pad, same chunk: rotation i stays on token i on every rank,
        # and the divisibility-pad rows carry cos=sin=0.
        assert cos_local.shape[0] == img_local.shape[1]
        assert torch.equal(cos_local, img_local[0])
        assert torch.equal(sin_local, -img_local[0])


def test_unpadded_chunk_misaligns_the_later_rank(sp):
    # The bug _shard_rope's pad rule prevents: chunking the unpadded table
    # gives the later rank a table that does not match its token shard.
    img = _tokens(200, 7)
    cos = img[0].clone()
    sp(2, 1)
    img_local, _ = base.shard_seq(img, dim=1)
    naive = torch.chunk(cos, 2, dim=0)[1]
    assert naive.shape[0] != img_local.shape[1]

    cos_local, _ = _shard_rope((cos, cos))
    assert cos_local.shape[0] == img_local.shape[1]


@pytest.mark.parametrize("txt_len", [4, 3])
def test_block_local_slice_recovers_the_image_shard(sp, txt_len):
    txt = _tokens(100, txt_len)
    img = _tokens(200, 7)
    for rank in (0, 1):
        sp(2, rank)
        txt_local, _ = base.shard_seq(txt, dim=1)
        img_local, _ = base.shard_seq(img, dim=1)

        # The block re-reads text_seq_length from its local text stream and slices
        # the joint [text, image] sequence with it (CogVideoXBlock.forward in comfy):
        # the slice recovers exactly this rank's image shard ...
        joint = torch.cat((txt_local, img_local), dim=1)
        text_seq_length = txt_local.shape[1]
        assert torch.equal(joint[:, text_seq_length:], img_local)

        # ... whose row count is exactly what the sharded rope table holds.
        cos_local, _ = _shard_rope((img[0], img[0]))
        assert cos_local.shape[0] == joint.shape[1] - text_seq_length


def test_image_shard_gather_round_trips(sp):
    # Emulated all-gather: concatenating every rank's chunk and trimming the
    # divisibility pad recovers the original image sequence bit-exactly.
    img = torch.randn(1, 7, 4)
    chunks = []
    for rank in (0, 1):
        sp(2, rank)
        local, orig = base.shard_seq(img, dim=1)
        chunks.append(local)
    assert torch.equal(torch.cat(chunks, dim=1).narrow(1, 0, orig), img)


def test_attention_mask_raises_typed():
    adapter = CogVideoXAdapter()
    fake = types.SimpleNamespace(blocks=[])
    adapter.inject_usp(fake, InjectionContext(topology_sp=2, usp_attention=object()))
    with pytest.raises(UnsupportedModelError, match="attention mask"):
        fake._forward(torch.zeros(1, 16, 1, 4, 4), torch.zeros(1), torch.zeros(1, 226, 8),
                      attention_mask=torch.ones(1, 226))

# The matches() trio (exact-type accept, typed reject on an unvetted
# subclass, foreign-model decline) lives in tests/test_adapter_matches.py.
