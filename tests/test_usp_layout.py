"""Sequence-parallel layout invariants on CPU, without distributed init.

Embedding each rank's local ID shards must match its [txt_local, img_local]
token layout. A contiguous chunk of the embedded full [txt, img] sequence does
not preserve that alignment (docs/ADAPTERS.md, "RoPE alignment under sharding").
"""
import pytest
import torch

from dgx_monarch.adapters import base


def _ids(start, length):
    # Distinct per-position values stand in for positional embeddings: an
    # identity "embedder" makes alignment directly comparable.
    return torch.arange(start, start + length).view(1, length, 1).float()


@pytest.mark.parametrize("txt_len", [3, 4])  # odd and even text lengths
def test_local_id_shards_match_the_block_layout(sp, txt_len):
    txt_ids, img_ids = _ids(100, txt_len), _ids(200, 8)
    for rank in (0, 1):
        sp(2, rank)
        txt_local, _ = base.shard_seq(txt_ids)
        img_local, _ = base.shard_seq(img_ids)
        block_layout = torch.cat((txt_local, img_local), dim=1)

        # The fix: embed this rank's own id shards, in block order.
        pe_new = torch.cat((base.shard_seq(txt_ids)[0], base.shard_seq(img_ids)[0]), dim=1)
        assert torch.equal(pe_new, block_layout)

        # The bug: a contiguous chunk of the embedded full sequence hands the
        # rank a run of the wrong positions (for every text-length parity).
        txt_p, _ = base.pad_seq_to_multiple(txt_ids, 2)
        img_p, _ = base.pad_seq_to_multiple(img_ids, 2)
        pe_old = torch.chunk(torch.cat((txt_p, img_p), dim=1), 2, dim=1)[rank]
        assert not torch.equal(pe_old, block_layout)


def test_joint_stream_shard_matches_sharded_ids(sp):
    # Single-block path: the [txt, img] re-cat stream and its concatenated
    # ids shard identically, so embedding the id shard stays token-aligned
    # even when the text length does not divide the SP degree.
    txt_len = 3  # odd: the left-edge-artifact parity
    txt = _ids(100, txt_len)
    img = _ids(200, 8)
    joint = torch.cat((txt, img), dim=1)
    joint_ids = torch.cat((txt, img), dim=1)
    for rank in (0, 1):
        sp(2, rank)
        stream_local, _ = base.shard_seq(joint)
        ids_local, _ = base.shard_seq(joint_ids)
        assert torch.equal(stream_local, ids_local)


def test_shard_then_gather_round_trips(sp):
    t = torch.randn(1, 7, 4)
    chunks = []
    for rank in (0, 1):
        sp(2, rank)
        chunk, orig = base.shard_seq(t)
        chunks.append(chunk)
    assert orig == 7
    padded = torch.cat(chunks, dim=1)
    assert padded.shape[1] == 8
    assert torch.equal(padded[:, :orig], t)
    assert torch.equal(padded[:, orig:], torch.zeros_like(padded[:, orig:]))


def test_strict_adapter_can_reject_nondivisible_sequence(sp):
    sp(2, 0)
    with pytest.raises(base.UnsupportedModelError, match="requires exact divisibility"):
        base.shard_seq(torch.randn(1, 7, 4), allow_padding=False)
