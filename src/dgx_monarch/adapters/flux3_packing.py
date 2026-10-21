"""FLUX.3 Klein sequence helpers: span clipping, stream shards and pad rows.

Stock FLUX.3 runs every block over a row stream with a list of
``(start, end, ModulationOut)`` spans: the first span's modulation is applied
to every row, each later span redoes its own rows, and each span's gate writes
only its own rows (comfy/ldm/flux3/model.py, ``Block.forward``). Every part of
a block apart from attention is row-local, so a rank holding a contiguous
window of rows computes each of them exactly once its spans are clipped to
that window and each fused-norm call runs at stock's row count, which
``stock_norm_rows`` names and adapters/flux3_adaln_parity.py applies. These
helpers do the clipping, cut one stream into this rank's window with its ids
and rotary table, and name the window's divisibility pad rows. They import no
ComfyUI module, so the CPU suite reaches them without one.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from ..refusal import RefusalClass, refusal
from . import base
from .base import UnsupportedModelError, padded_row_indices, shard_seq
from .flux_shared import _assert_ids_shard_with_tokens, _control_span_overlap


def _clip(spans: list, row0: int, rows: int) -> Iterator[tuple[int, int, int, Any]]:
    """``(index, local start, local end, mod)`` for each span a window keeps."""
    for index, (start, end, mod) in enumerate(spans):
        overlap = _control_span_overlap(row0, rows, start, end - start)
        if overlap is not None:
            yield index, overlap[0], overlap[1], mod
        elif index == 0:
            yield index, 0, 0, mod


def local_spans(spans: list, row0: int, rows: int) -> list:
    """Stock's ``(start, end, mod)`` spans clipped to rows ``[row0, row0 + rows)``.

    ``spans[0]`` is kept even when no local row falls in it, as ``(0, 0, mod)``:
    ``Block.forward`` reads its modulation for the pass over every local row,
    and its empty gate slice writes nothing. A later span with no local row is
    dropped, which changes no row and hands the fused norm no empty launch.
    """
    return [(start, end, mod) for _index, start, end, mod in _clip(spans, row0, rows)]


def stock_norm_rows(spans: list, row0: int, rows: int, stream_rows: int) -> tuple[int, ...]:
    """Stock's rows a sample in the fused-norm call behind each span ``local_spans`` keeps.

    Stock's first call normalizes the whole stream, so the first span carries
    ``stream_rows``; each later span carries the rows of the stock span it was
    clipped from. adapters/flux3_adaln_parity.py launches each piece at this size.
    """
    return tuple(stream_rows if index == 0 else spans[index][1] - spans[index][0]
                 for index, _start, _end, _mod in _clip(spans, row0, rows))


@dataclass(frozen=True)
class StreamShard:
    """One rank's window of a row stream, ready for stock blocks."""

    tokens: torch.Tensor   # this rank's rows, contiguous, pads at the tail
    pe: torch.Tensor       # stock's rotary layout for exactly these rows
    spans: list            # stock's spans clipped to the window
    drop_rows: list[int]   # the stream's pad rows in Ulysses' gathered order
    orig: int              # the pre-pad length, which sp_gather restores
    norm_rows: tuple[int, ...] = ()  # stock's fused-norm call rows, one a span


def shard_stream(tokens: torch.Tensor, ids: torch.Tensor, spans: list,
                 rope: Any, name: str) -> StreamShard:
    """Cut ``tokens`` ``[B, N, D]`` and ``ids`` ``[N, 4]`` into this rank's window.

    The ids take the same pad and chunk as the tokens and are embedded by
    stock's own ``rope`` on this rank. The token chunk is copied contiguous:
    ``shard_seq`` returns a strided view at batch 2, and ``Block.forward``
    writes its gate residual in place on its input, so the copy keeps every
    kernel input in stock's layout.
    """
    local, orig = shard_seq(tokens, dim=1, name=f"{name}-token")
    ids_local, _ = shard_seq(ids, dim=0, name=f"{name}-position")
    _assert_ids_shard_with_tokens((ids_local.unsqueeze(0), local))
    rows = local.shape[1]
    row0 = base.sp_rank() * rows
    clipped = local_spans(spans, row0, rows)
    covered = sum(end - start for start, end, _ in clipped)
    assert covered == min(rows, max(0, orig - row0)), (
        f"{name}: clipped spans cover {covered} of this rank's real rows")
    return StreamShard(tokens=local.contiguous(), pe=rope(ids_local), spans=clipped,
                       drop_rows=padded_row_indices([(orig, rows)]), orig=orig,
                       norm_rows=stock_norm_rows(spans, row0, rows, orig))


def assert_joint_layout(parts: list, ids: torch.Tensor, spans: list, batch: int,
                        ref_shapes: Any) -> None:
    """Stock's invariants for the joint stream, as plain asserts.

    ``parts`` is ``[(tokens, mode, part_ids), ...]`` in stock order. Each is a
    stock invariant no workflow can reach (the ids and tokens of one stream
    disagree only through a caller bug), so none is a typed refusal.
    """
    has_refs = any(mode == "image_cond" for _tokens, mode, _ids in parts)
    assert has_refs == (ref_shapes is not None), (
        "flux3: reference states and reference shapes must arrive together")
    for tokens, mode, part_ids in parts:
        assert tokens.shape[0] == batch, (
            f"flux3: the {mode} stream carries batch {tokens.shape[0]} against {batch}")
        assert part_ids.shape[0] == tokens.shape[1], (
            f"flux3: the {mode} stream has {tokens.shape[1]} rows and "
            f"{part_ids.shape[0]} position ids")
    total = sum(tokens.shape[1] for tokens, _mode, _ids in parts)
    assert ids.shape[0] == total, f"flux3: {ids.shape[0]} joint ids for {total} rows"
    edge = 0
    for start, end, _mod in spans:
        assert start == edge and end >= start, f"flux3: span ({start}, {end}) is not contiguous"
        edge = end
    assert edge == total, f"flux3: the spans end at {edge} of {total} rows"


def joint_pad_count(rows: int) -> int:
    """The divisibility pad rows ``shard_seq`` adds to a stream of ``rows``."""
    return -rows % base.sp_world()


def refuse_padded_ring(image_pads: int, joint_pads: int) -> None:
    """Refuse a padded ring or hybrid render once, before the first block.

    A FLUX.3 pad row is a real key after the first adaLN, and only Ulysses has
    a full-sequence point where it can leave attention. The generic guard and
    its ``ring_pad`` waiver apply; its remedy (``uly*`` or ``single``) fits
    this family, because ``single`` runs it.
    """
    if not image_pads and not joint_pads:
        return
    from xfuser.core.distributed import get_ring_parallel_world_size

    base.assert_ulysses_only_padding(get_ring_parallel_world_size(),
                                     image_pads + joint_pads)


def ulysses_degree() -> int:
    """The active Ulysses degree, or 1 where model parallel is not initialized."""
    try:
        from xfuser.core.distributed import get_ulysses_parallel_world_size

        return int(get_ulysses_parallel_world_size())
    except Exception:
        return 1


def refuse_indivisible_heads(heads: int, degree: int, *, family: str = "flux3",
                             divisors: str = "2, 4 or 8 for the 24-head FLUX.3 Klein files") -> None:
    """Refuse a Ulysses degree that does not divide the attention heads.

    Ulysses hands each rank whole heads. Every shipped FLUX.3 Klein file has
    24, which degrees 2, 4 and 8 divide; ring has no head constraint.
    MOSS-Image passes its own ``family`` and ``divisors`` (adapters/moss_image.py).
    """
    if degree > 1 and heads % degree:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"{family}: this checkpoint has {heads} attention heads, which the Ulysses "
            f"degree {degree} does not divide, and Ulysses hands each rank whole heads. "
            f"Use a Ulysses degree that divides the head count ({divisors}), a ring "
            "topology, or topology 'single' (mode=local with gpus_per_host=1) "
            "(docs/MODELS.md).",
        ))

