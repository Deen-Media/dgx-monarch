"""What ring and hybrid attention cannot do exactly, refused before they try.

Ring and hybrid layouts shard the key axis, so the pack has no point at which
a rank holds the whole sequence and can leave divisibility pad rows out of the
softmax, or apply a bias over the key axis where the kernel reads it. Wan's
TORCH_FLASH ring2 path does gather the whole K and V onto each rank for one
flash call, but the exclusion is not wired there (docs/MODELS.md), so it
refuses as the others do. Pure Ulysses does both exactly. The adapters call
these from their forwards (re-exported through ``adapters/base.py``).
"""
from __future__ import annotations

import torch

from .. import accuracy_waiver
from ..refusal import RefusalClass, refusal

RING_PAD_GUARD = "ring_pad"


@torch.compiler.disable
def assert_ulysses_only_padding(ring_world: int, n_pad_rows: int,
                                remedy: str | None = None) -> None:
    """Refuse pad rows that ring or hybrid attention cannot exclude.

    Ring has no full-sequence exclusion point, and attended pads produced
    measured left-edge corruption. A default-off class-K waiver is checked on
    every call so grants and revocations apply per render; compilation is
    disabled so one decision cannot persist into another render. ``remedy``,
    where a family can name the shapes that divide, replaces the generic one.
    """
    if ring_world > 1 and n_pad_rows:
        if accuracy_waiver.waived(RING_PAD_GUARD):
            return
        from .base import UnsupportedModelError

        raise UnsupportedModelError(refusal(
            RefusalClass.KNOWN_WRONG,
            "this sequence length needs divisibility padding, and the exact "
            "pad exclusion is ulysses-only; ring/hybrid attention would "
            "attend the synthetic pad rows (left-edge corruption, "
            "measured 2026-07-10). "
            + (remedy or "Use a uly* or single topology, or a resolution and prompt "
               "lengths whose token counts divide the SP degree")
            + " (docs/ADAPTERS.md).",
            guard=RING_PAD_GUARD,
            waivable=True,
            panel_action=accuracy_waiver.panel_action(RING_PAD_GUARD),
            troubleshooting=21,
        ) + accuracy_waiver.card_tail(RING_PAD_GUARD))


@torch.compiler.disable
def assert_ulysses_only_query_bias(ring_world: int) -> None:
    """Refuse a query-group bias that ring or hybrid attention cannot apply.

    Ulysses hands each rank the whole key axis, so a bias over that axis applies
    exactly. Ring never does: its ranks see the keys one block at a time, and
    the kernel that walks those blocks takes no bias argument (yunchang
    ring_flash_attn_func). There is nothing to waive, because no ring rank can
    be handed the weights.

    Divisibility padding is allowed with a bias. A biased stream shards alone,
    which puts its pads at the tail of the token axis, past every group
    boundary, so dropping them leaves each group biasing the tokens it named
    (adapters/usp_pad_exclusion.assert_pads_are_a_tail holds that layout).
    """
    if ring_world == 1:
        return
    from .base import UnsupportedModelError

    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        "this render biases attention between token groups, and ring or hybrid "
        "attention shards the key axis, so the weights cannot be applied where "
        "the kernel reads them. Use a pure-ulysses topology (uly2 on a two-rank "
        "fleet), or topology 'single' (mode=local with gpus_per_host=1) for a "
        "stock one-GPU run.",
        troubleshooting=80,
    ))
