"""CPU checks for Lens USP pad-row exclusion.

No xfuser, comfy or distributed backend. For every rank of a simulated SP
world, shard both toy streams (`hidden_states`, `encoder_hidden_states`) and
the real `_local_pe_ids` id table the way `LensAdapter.inject_usp`'s
`usp_forward` does, rebuild the Ulysses head-scatter's rank-interleaved
gathered layout by hand, and compute
`joint_drop = padded_row_indices([img_seg, txt_seg])` in Lens's image-first
order. Dropping those rows must leave every real image and text token exactly
once, in the token stream and in the id table (RoPE positions), and every
dropped row must be a shard_seq zero pad. Order does not matter: attention is
permutation-invariant over keys when q, k and v share the permutation, so only
the set of values must match.

test_property_hardening.py proves the general N-segment case
(`test_shard_seq_and_real_sp_gather_round_trip`,
`test_padded_row_indices_match_the_rank_interleaved_marker_layout`); this file
pins the two-segment wiring `usp_forward` performs.
"""
from __future__ import annotations

from unittest.mock import patch

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from dgx_monarch.adapters import base
from dgx_monarch.adapters.lens import _local_pe_ids

# Disjoint, nonzero value ranges for image and text rows, so a misrouted row
# shows as a wrong integer, not only a wrong shape, and neither range meets
# shard_seq's zero pad.
_IMG_BASE = 1
_TXT_BASE = 1000


def _simulate(img_len: int, text_len: int, world: int):
    """Run every rank's shard_seq and _local_pe_ids, rebuild the gathered
    [seg0_r0, seg1_r0, seg0_r1, seg1_r1, ...] layout of the token stream and
    the id table, and compute the drop rows usp_forward passes to usp_options."""
    hidden = torch.arange(_IMG_BASE, _IMG_BASE + img_len, dtype=torch.float64).reshape(1, img_len, 1)
    encoder = torch.arange(_TXT_BASE, _TXT_BASE + text_len, dtype=torch.float64).reshape(1, text_len, 1)
    img_ids = hidden.repeat(1, 1, 3)
    txt_ids = encoder.repeat(1, 1, 3)

    joint_chunks = []
    id_chunks = []
    img_orig_len = txt_orig_len = 0
    img_local_len = txt_local_len = 0
    with patch.object(base, "sp_world", return_value=world):
        for rank in range(world):
            with patch.object(base, "sp_rank", return_value=rank):
                img_local, img_orig_len = base.shard_seq(hidden, dim=1)
                txt_local, txt_orig_len = base.shard_seq(encoder, dim=1)
                ids_local = _local_pe_ids(img_ids, txt_ids)
            assert img_local is not None and txt_local is not None
            img_local_len = img_local.shape[1]
            txt_local_len = txt_local.shape[1]
            # Lens's block joins [img, txt], image first (lens.py module
            # docstring and `_local_pe_ids`).
            joint_chunks.append(torch.cat((img_local, txt_local), dim=1))
            id_chunks.append(ids_local)

        gathered = torch.cat(joint_chunks, dim=1)
        gathered_ids = torch.cat(id_chunks, dim=1)
        img_seg = (img_orig_len, img_local_len)
        txt_seg = (txt_orig_len, txt_local_len)
        joint_drop = base.padded_row_indices([img_seg, txt_seg])
    return gathered, gathered_ids, joint_drop


_LENGTHS = st.integers(1, 12)
_WORLDS = st.sampled_from([1, 2, 4])


@given(img_len=_LENGTHS, text_len=_LENGTHS, world=_WORLDS)
@settings(max_examples=60)
def test_real_tokens_survive_the_shard_pad_gather_round_trip(img_len, text_len, world):
    # The gathered layout is rank-interleaved (`base.padded_row_indices`), not
    # stream-contiguous, so only the set of values is compared (module docstring).
    gathered, _, joint_drop = _simulate(img_len, text_len, world)
    drop_set = set(joint_drop)
    total = gathered.shape[1]
    keep = [r for r in range(total) if r not in drop_set]

    expected = torch.cat((
        torch.arange(_IMG_BASE, _IMG_BASE + img_len, dtype=torch.float64),
        torch.arange(_TXT_BASE, _TXT_BASE + text_len, dtype=torch.float64),
    ))
    actual = gathered[0, keep, 0]
    assert sorted(actual.tolist()) == sorted(expected.tolist())


@given(img_len=_LENGTHS, text_len=_LENGTHS, world=_WORLDS)
@settings(max_examples=60)
def test_id_table_survives_the_round_trip_identically_rope_alignment(img_len, text_len, world):
    _, gathered_ids, joint_drop = _simulate(img_len, text_len, world)
    drop_set = set(joint_drop)
    total = gathered_ids.shape[1]
    keep = [r for r in range(total) if r not in drop_set]

    expected = torch.cat((
        torch.arange(_IMG_BASE, _IMG_BASE + img_len, dtype=torch.float64),
        torch.arange(_TXT_BASE, _TXT_BASE + text_len, dtype=torch.float64),
    ))
    actual = gathered_ids[0, keep, 0]
    assert sorted(actual.tolist()) == sorted(expected.tolist())


@given(img_len=_LENGTHS, text_len=_LENGTHS, world=_WORLDS)
@settings(max_examples=60)
def test_dropped_rows_are_exactly_shard_seq_zero_pad_sentinels(img_len, text_len, world):
    gathered, gathered_ids, joint_drop = _simulate(img_len, text_len, world)
    total = gathered.shape[1]

    # Every dropped row is shard_seq's zero pad, never a real token (both value
    # ranges exclude zero).
    for row in joint_drop:
        assert float(gathered[0, row, 0]) == 0.0
        assert float(gathered_ids[0, row, 0]) == 0.0

    # Nothing is dropped twice and nothing real is missing: the pad count is
    # exactly the gap between the gathered total and the real token count.
    assert len(joint_drop) == len(set(joint_drop))
    assert total - len(joint_drop) == img_len + text_len


def test_world1_never_pads_lens_streams():
    gathered, gathered_ids, joint_drop = _simulate(img_len=7, text_len=5, world=1)
    assert joint_drop == []
    assert gathered.shape[1] == 12
    assert gathered_ids.shape[1] == 12
