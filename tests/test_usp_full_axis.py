from __future__ import annotations

import sys
import types

import torch

from dgx_monarch.adapters.usp_full_axis import run_full_axis
from dgx_monarch.adapters.usp_sequence_order import RankMajorJointOrder


def _install_all_to_all(monkeypatch, gathered, calls):
    class AllToAll:
        @staticmethod
        def apply(_group, tensor, scatter, gather):
            calls.append((scatter, gather, tensor.detach().clone()))
            if (scatter, gather) == (2, 1):
                return gathered.to(device=tensor.device, dtype=tensor.dtype)
            assert (scatter, gather) == (1, 2)
            return tensor

    module = types.ModuleType("yunchang.comm.all_to_all")
    module.SeqAllToAll4D = AllToAll
    monkeypatch.setitem(sys.modules, "yunchang", types.ModuleType("yunchang"))
    monkeypatch.setitem(sys.modules, "yunchang.comm", types.ModuleType("yunchang.comm"))
    monkeypatch.setitem(sys.modules, "yunchang.comm.all_to_all", module)


class _USP:
    ulysses_pg = ring_pg = attn_type = attn_processor = None
    q_descale = k_descale = v_descale = None

    def __init__(self, seen):
        self.seen = seen

    def ring_attn_fn(self, q, k, v, **_kwargs):
        self.seen["kernel_q"] = q.detach().clone()
        self.seen["kernel_k"] = k.detach().clone()
        self.seen["kernel_v"] = v.detach().clone()
        return q


def _canonical_attention(q, k, v, _groups):
    scores = torch.einsum("blhd,bmhd->bhlm", q, k) / q.shape[-1] ** 0.5
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("bhlm,bmhd->blhd", weights, v)


def test_full_axis_kernel_sees_canonical_rows_and_inverse_all_to_all_gets_rank_major(monkeypatch):
    calls, seen = [], {}
    rank_major = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1, 1)
    _install_all_to_all(monkeypatch, rank_major, calls)
    order = RankMajorJointOrder(2, 2)
    output = run_full_axis(_USP(seen), rank_major[:, :4], rank_major[:, :4],
                           rank_major[:, :4], [3], [7], None, order,
                           _canonical_attention)
    expected_canonical = torch.tensor([0, 1, 4, 5, 2, 3, 6, 7], dtype=torch.float64)
    # Query pad old row 3 maps to canonical row 5; key pad old row 7 stays 7.
    assert torch.equal(seen["kernel_q"][:, :, 0, 0], expected_canonical[[0, 1, 2, 3, 4, 6, 7]].unsqueeze(0))
    assert torch.equal(seen["kernel_k"][:, :, 0, 0], expected_canonical[[0, 1, 2, 3, 4, 5, 6]].unsqueeze(0))
    inverse_sent = calls[-1][2]
    assert torch.equal(inverse_sent[:, :, 0, 0], torch.tensor(
        [[0, 1, 2, 0, 4, 5, 6, 7]], dtype=torch.float64))
    assert output.shape == rank_major.shape


def test_full_axis_order_and_pad_composition_matches_direct_canonical_float64(monkeypatch):
    torch.manual_seed(4)
    rank_major = torch.randn(1, 8, 2, 3, dtype=torch.float64)
    calls, seen = [], {}
    _install_all_to_all(monkeypatch, rank_major, calls)
    order = RankMajorJointOrder(2, 2)
    class AttentionUSP(_USP):
        def ring_attn_fn(self, q, k, v, **_kwargs):
            self.seen["kernel_q"] = q.detach().clone()
            return _canonical_attention(q, k, v, None)

    result = run_full_axis(AttentionUSP(seen), rank_major[:, :4], rank_major[:, :4],
                           rank_major[:, :4], [3], [7], None, order,
                           _canonical_attention)
    canonical, _k, _v, drop, kv_drop, permutation = order.canonicalize(
        rank_major, rank_major, rank_major, [3], [7], None)
    keep_q = [index for index in range(8) if index not in drop]
    keep_k = [index for index in range(8) if index not in kv_drop]
    direct = _canonical_attention(canonical[:, keep_q], canonical[:, keep_k],
                                  canonical[:, keep_k], None)
    restored = canonical.new_zeros(*canonical.shape)
    restored[:, keep_q] = direct
    expected_rank_major = order.restore(restored, permutation)
    assert torch.allclose(calls[-1][2], expected_rank_major, atol=1e-12, rtol=1e-12)
    assert torch.equal(result, expected_rank_major)


def test_no_descriptor_keeps_the_existing_rank_major_full_axis_path(monkeypatch):
    calls, seen = [], {}
    rank_major = torch.arange(8, dtype=torch.float64).reshape(1, 8, 1, 1)
    _install_all_to_all(monkeypatch, rank_major, calls)
    run_full_axis(_USP(seen), rank_major[:, :4], rank_major[:, :4], rank_major[:, :4],
                  [], [], None, None, _canonical_attention)
    assert torch.equal(seen["kernel_q"], rank_major)
