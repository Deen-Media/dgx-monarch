"""Shard helpers the flux-family forwards share with the adapters that reuse
their control-span remap (hunyuan, lens, mage_flow, qwen_image)."""
from __future__ import annotations

import torch

from .base import padded_row_indices, shard_seq

# The named probe that vouches the flux-family divisibility pad-row exclusion.
# flux, flux2 and longcat share one forward (the same shard, the same two drop
# sets, the same Ulysses full-sequence point), so one matrix with a leg per
# family supplies the evidence for all three. A family whose leg fails
# sets its own `usp_pad_exclusion_probe = None` and refuses an indivisible stream.
PAD_FIDELITY_PROBE = "benchmark/reports/flux_family_pad_fidelity_matrix.toml"


def _control_span_overlap(row0: int, rows: int, span_start: int, span_len: int) -> tuple[int, int, int, int] | None:
    """Map a stock global-span ControlNet add onto one rank's row window.

    Stock Flux adds control tensors over global sequence rows: the double loop
    covers img rows [0, add_len), the single loop covers joint rows [txt_len,
    txt_len + add_len). Under sequence sharding this rank holds only rows
    [row0, row0 + rows) of the divisible global stream, and the add may be
    shorter than the stream it targets (Kontext ref tokens sit past controlnet
    coverage), so re-sharding the add tensor cannot align it; compute the
    overlapping interval instead.

    Returns (local_lo, local_hi, add_lo, add_hi) slice bounds for
    `local[:, local_lo:local_hi] += add[:, add_lo:add_hi]`, or None when this
    rank's window misses the span entirely.
    """
    lo = max(row0, span_start)
    hi = min(row0 + rows, span_start + span_len)
    if lo >= hi:
        return None
    return lo - row0, hi - row0, lo - span_start, hi - span_start


def _family_shard(t: torch.Tensor, family: str, name: str, *,
                  pad_vouched: bool) -> tuple[torch.Tensor, int]:
    """Shard a stream, allowing padding only with family-specific evidence.

    A family with a passed padding probe excludes synthetic rows before every
    kernel. Without that evidence, reject an indivisible stream rather than
    attending added keys (docs/TROUBLESHOOTING.md #77).
    """
    return shard_seq(t, dim=1, allow_padding=pad_vouched, name=f"{family} {name}")


def _exact_shard(t: torch.Tensor, family: str, name: str) -> tuple[torch.Tensor, int]:
    """The strict shard, kept for out-of-tree callers only.

    Every in-tree forward calls `_family_shard`, which reads the calling
    adapter's own probe attribute; a new caller should too.
    """
    return _family_shard(t, family, name, pad_vouched=False)


def _family_drop_rows(pad_vouched: bool, segments: list[tuple[int, int]]) -> list[int] | None:
    """Return gathered pad coordinates for a family with a passed padding probe.

    Return ``None`` without a passed probe; the sharding helper rejects any
    indivisible stream in that case. With a passed probe, a divisible stream
    returns ``[]``, which the attention path treats as unpadded.
    """
    return padded_row_indices(segments) if pad_vouched else None


def _assert_ids_shard_with_tokens(*pairs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Assert that position IDs and tokens use identical shard lengths.

    Both lengths derive from the same stream, so a mismatch is a caller bug,
    not a workflow refusal.
    """
    for ids, tokens in pairs:
        assert ids.shape[1] == tokens.shape[1], (
            f"position id shard carries {ids.shape[1]} rows against "
            f"{tokens.shape[1]} token rows")
