"""Per-query-group additive attention bias, for the full-sequence point.

Some families bias attention between two token groups instead of hiding keys.
ComfyUI expresses that by splitting the query axis and running attention once
per group against the whole key axis, each group carrying its own additive
bias, rather than by handing one dense (T, T) mask to a kernel (LTX guide
attenuation, comfy ldm/lightricks/model.py ``_attention_with_guide_mask``).
Lens joint attention under pure Ulysses comes here too, as one group carrying
stock's key bias (adapters/lens.py ``_stock_key_bias``).

That shape re-expresses exactly under Ulysses: after the head scatter a rank
holds the whole token axis for a subset of heads, so the query groups and the
key axis are both complete, and every bias here is head-independent. The caller
owns the exchange; this module owns the math, and takes comfy only through a
lazy import of its SDPA wrapper, so a test can pass its own kernel and run the
whole thing on a CPU box.

ComfyUI runs its own biased groups through
``optimized_attention(..., low_precision_attention=False)``, so ``attention_sage``
falls back to ``attention_pytorch`` and ``comfy.ops.scaled_dot_product_attention``
with an additive ``attn_mask`` (comfy ldm/modules/attention.py). A biased group
here takes that same kernel, as stock does for the same call, and only calls
that carry bias groups take it.
"""
from __future__ import annotations

from typing import NoReturn

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

# One group: queries ``[start, end)`` and the additive bias they carry, or
# ``None`` for an unbiased group. A bias broadcasts to (B, H, end - start, keys)
# and may declare fewer keys than the kernel sees; the rest carry no bias.
QueryBiasGroup = tuple[int, int, "torch.Tensor | None"]

# These contracts break only when dgx-monarch builds a group set wrong, never
# on a workflow setting. The text still crosses to the driver, so it carries a
# class and names the topology that runs none of this machinery.
_GROUP_CONTRACT_ALTERNATIVE = (
    "This is a dgx-monarch fault, not a workflow setting: report the shapes. "
    "Use topology 'single' (mode=local with gpus_per_host=1), where stock "
    "attention applies the bias itself, until it is fixed."
)


def _sdpa():
    """comfy's SDPA wrapper: the kernel stock runs its own biased groups on."""
    from comfy.ops import scaled_dot_product_attention

    return scaled_dot_product_attention


def _contract_fault(detail: str) -> NoReturn:
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"query bias groups are malformed: {detail}. "
        f"{_GROUP_CONTRACT_ALTERNATIVE}",
    ))


def validate_query_bias_groups(
    groups: tuple[QueryBiasGroup, ...], queries: int, keys: int,
) -> None:
    """Refuse a group set that does not partition the query axis in order.

    A gap would leave output rows uninitialized and an overlap would let one
    group overwrite another's, so both fail before any kernel call rather than
    returning a plausible tensor.
    """
    cursor = 0
    for start, end, bias in groups:
        if start != cursor or end < start or end > queries:
            _contract_fault(
                f"group [{start}, {end}) does not continue the partition of "
                f"[0, {queries}) at row {cursor}")
        cursor = end
        if bias is None:
            continue
        if bias.ndim != 4 or bias.shape[2] not in (1, end - start):
            _contract_fault(
                f"a bias shaped {tuple(bias.shape)} does not broadcast over "
                f"the {end - start} query rows of its group as (B, H, Q, K)")
        if bias.shape[3] > keys:
            _contract_fault(
                f"a bias declares {bias.shape[3]} keys over a sequence of "
                f"{keys}")
    if cursor != queries:
        _contract_fault(
            f"the groups cover {cursor} of {queries} query rows")


def extend_bias_keys(bias: torch.Tensor, keys: int) -> torch.Tensor:
    """Zero-extend a bias to ``keys`` columns.

    A zero column adds no bias, so a key the bias does not declare is as visible
    as it is to an unbiased query. Callers drop divisibility pad keys before the
    kernel (``usp_pad_exclusion.attend_without_pads``), so pads never reach this.
    """
    short = keys - bias.shape[3]
    if short <= 0:
        return bias
    tail = bias.new_zeros((*bias.shape[:3], short))
    return torch.cat([bias, tail], dim=3)


def query_group_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    groups: tuple[QueryBiasGroup, ...], *, sdpa=None,
) -> torch.Tensor:
    """Attend each query group against the full keys under its own bias.

    ``q``/``k``/``v`` arrive as (B, L, H, D), the layout the Ulysses head
    scatter produces, and the result keeps it.
    """
    validate_query_bias_groups(groups, q.shape[1], k.shape[1])
    attend = sdpa if sdpa is not None else _sdpa()
    keys = k.shape[1]

    q_h, k_h, v_h = (t.transpose(1, 2) for t in (q, k, v))
    out = torch.empty_like(q_h)
    for start, end, bias in groups:
        if end == start:
            continue
        mask = None if bias is None else extend_bias_keys(bias, keys)
        out[:, :, start:end] = attend(
            q_h[:, :, start:end], k_h, v_h, attn_mask=mask)
    return out.transpose(1, 2)
