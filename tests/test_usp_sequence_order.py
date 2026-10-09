from __future__ import annotations

import pytest
import torch

from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.adapters.usp_sequence_order import RankMajorJointOrder


def test_rank_major_joint_order_round_trips_unequal_segments_and_pad_rows():
    order = RankMajorJointOrder(3, 5)
    q = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1)
    canonical_q, _canonical_k, _canonical_v, drop, kv_drop, permutation = order.canonicalize(
        q, q, q, [7, 15], [2, 11], None)
    assert torch.equal(
        canonical_q[:, :, 0, 0],
        torch.tensor([[0, 1, 2, 8, 9, 10, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]], dtype=torch.float32),
    )
    assert drop == [10, 15] and kv_drop == [2, 11]
    assert torch.equal(order.restore(canonical_q, permutation), q)


def test_permutation_lists_every_rank_text_then_every_rank_image():
    """The index build is tensor arithmetic; this pins it to the definition."""
    for text_rows, image_rows, world in ((1, 1, 2), (3, 5, 2), (128, 2048, 2), (7, 2, 4)):
        local = text_rows + image_rows
        want = ([rank * local + row for rank in range(world) for row in range(text_rows)]
                + [rank * local + text_rows + row for rank in range(world) for row in range(image_rows)])
        got = RankMajorJointOrder(text_rows, image_rows).permutation(world * local, torch.device("cpu"))
        assert got.dtype == torch.long and got.tolist() == want


def test_rank_major_joint_order_remaps_independent_equal_self_attention_pads():
    order = RankMajorJointOrder(2, 4)
    q = torch.zeros(1, 12, 1, 1)
    _q, _k, _v, drop, kv_drop, _permutation = order.canonicalize(
        q, q, q, [1, 11], [2, 10], None)
    assert drop == [1, 11] and kv_drop == [4, 10]


def test_rank_major_joint_order_refuses_groups_and_invalid_shapes():
    order = RankMajorJointOrder(2, 4)
    q = torch.zeros(1, 12, 1, 1)
    with pytest.raises(UnsupportedModelError, match="query-bias"):
        order.canonicalize(q, q, q, None, None, [(0, 2)])
    with pytest.raises(UnsupportedModelError, match="equal"):
        order.canonicalize(q, q[:, :-1], q, None, None, None)


def test_pure_ulysses_descriptor_keeps_cfg_parallel_outside_its_sequence_axis():
    # CFG is a separate process-group dimension: pure Ulysses is ulysses > 1 and
    # ring 1 at any cfg degree (actor/attention_context.py). Injection hands the
    # descriptor only to pure Ulysses, and the attention callable that
    # base._make_usp_attention_callable builds drops it under ring.
    order = RankMajorJointOrder(2, 4)
    q = torch.zeros(1, 12, 1, 1)
    _q, _k, _v, _drop, _kv_drop, permutation = order.canonicalize(
        q, q, q, None, None, None)
    assert permutation.numel() == 12


def test_joint_sequence_order_is_a_descriptor_for_pure_ulysses_only():
    from types import SimpleNamespace

    from dgx_monarch.adapters.usp_sequence_order import joint_sequence_order

    order = joint_sequence_order(SimpleNamespace(pure_ulysses=True), 3, 5)
    assert order == RankMajorJointOrder(3, 5)
    assert joint_sequence_order(SimpleNamespace(pure_ulysses=False), 3, 5) is None
