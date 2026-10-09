"""The Ulysses full-axis attention call, with or without a sequence order."""
from __future__ import annotations

import torch

from .usp_pad_exclusion import attend_without_pads


def run_full_axis(usp_attn, q_l, k_l, v_l, drop_rows, kv_drop_rows, groups,
                  sequence_order, query_group_attention):
    """Gather sequence, optionally canonicalize it, attend, then restore."""
    from yunchang.comm.all_to_all import SeqAllToAll4D

    def attend(q_r, k_r, v_r, bias_groups):
        if bias_groups:
            return query_group_attention(q_r, k_r, v_r, bias_groups)
        out = usp_attn.ring_attn_fn(
            q_r, k_r, v_r, dropout_p=0.0, softmax_scale=None, causal=False,
            window_size=(-1, -1), alibi_slopes=None, deterministic=False,
            return_attn_probs=False, group=usp_attn.ring_pg,
            attn_type=usp_attn.attn_type, attn_processor=usp_attn.attn_processor,
            attn_layer=None, joint_tensor_key=None, joint_tensor_value=None,
            joint_strategy="none", q_descale=usp_attn.q_descale,
            k_descale=usp_attn.k_descale, v_descale=usp_attn.v_descale)
        return out[0] if isinstance(out, tuple) else out

    q_r, k_r, v_r = (
        SeqAllToAll4D.apply(usp_attn.ulysses_pg, value, 2, 1)
        for value in (q_l, k_l, v_l))
    permutation = None
    if sequence_order is not None:
        q_r, k_r, v_r, drop_rows, kv_drop_rows, permutation = (
            sequence_order.canonicalize(q_r, k_r, v_r, drop_rows, kv_drop_rows, groups))
    out = attend_without_pads(q_r, k_r, v_r, drop_rows=drop_rows,
                              kv_drop_rows=kv_drop_rows, groups=groups, attend=attend)
    if permutation is not None:
        out = sequence_order.restore(out, permutation)
    if not isinstance(out, torch.Tensor):
        raise TypeError("Ulysses full-axis attention returned a non-tensor")
    return SeqAllToAll4D.apply(usp_attn.ulysses_pg, out, 1, 2)
