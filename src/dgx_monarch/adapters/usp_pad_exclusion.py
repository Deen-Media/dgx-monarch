"""Drop divisibility pad rows at the Ulysses full-sequence point.

``shard_seq`` zero-pads a stream up to the sequence-parallel degree. After
modulation a zero embedding is not a zero key, so a maskless kernel that sees
those rows attends synthetic keys and can corrupt the left edge of the output.
Remove those rows before the kernel and restore zeros in their positions to
reproduce the unsharded computation.

Two streams need two row sets. A self-attention takes its queries and its keys
from the same padded stream, so one set serves both. An LTX audio-to-video
attention does not: its queries are video rows and its keys are audio rows, and
each stream padded on its own. Passing one set there would drop real audio keys
at video coordinates, so the query side and the key side are separate arguments
throughout.

The caller owns the all-to-all and the kernel; this leaf owns the row algebra
and takes no project import beyond the refusal vocabulary and its index
cache, so the float64 oracle in tests/test_usp_pad_exact.py runs the whole
thing on a CPU box.

The keep index is built on the device from the integer row list, one
``arange`` per run of kept rows, and cached per layout. The device never hands
a value back to the host here: a ``nonzero`` or a copy from host memory would
make the host wait for the stream, several times per attention call.
"""
from __future__ import annotations

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError
from .index_cache import BoundedIndexCache, device_key

# One render uses a few keep layouts (each stream, cond and uncond); the bound
# leaves room for several renders' worth before the oldest goes.
_KEEP_INDEX: BoundedIndexCache[torch.Tensor] = BoundedIndexCache(32)


def _kept_rows(rows: int, drop_rows, device: torch.device) -> torch.Tensor:
    """Ascending indices of the rows not in ``drop_rows``, built on ``device``."""
    runs, start = [], 0
    for dropped in sorted(set(drop_rows)):
        if dropped > start:
            runs.append(torch.arange(start, dropped, dtype=torch.long, device=device))
        start = dropped + 1
    if start < rows:
        runs.append(torch.arange(start, rows, dtype=torch.long, device=device))
    if not runs:
        return torch.empty(0, dtype=torch.long, device=device)
    return runs[0] if len(runs) == 1 else torch.cat(runs)


def keep_index(rows: int, drop_rows, device) -> torch.Tensor | None:
    """Row indices attention keeps, or ``None`` when it keeps every row.

    These indices reach a CUDA scatter, so one that addresses a row the axis
    does not have raises a device-side assert, which aborts the worker process
    and with it the fleet. The range check costs a comparison on a short list
    and turns the same mistake into a message naming both numbers.

    The result is shared by every call with the same rows, row list and
    device, so callers must only read it.
    """
    if not drop_rows:
        return None
    device = device_key(device)
    key = (rows, tuple(drop_rows), device)
    cached = _KEEP_INDEX.get(key)
    if cached is not None:
        return cached
    highest = max(drop_rows)
    if highest >= rows or min(drop_rows) < 0:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"a divisibility pad row at index {highest} was handed to an "
            f"attention whose axis carries {rows} rows. A cross attention "
            "between two sharded streams takes one row set per side, and this "
            "one got the other side's set. This is a dgx-monarch fault, not a "
            "workflow setting: report the shapes. Use topology 'single' "
            "(mode=local with gpus_per_host=1) until it is fixed.",
        ))
    # Comfy samples under inference mode; build a normal tensor so a later
    # call outside it can still use the shared index.
    with torch.inference_mode(False):
        index = _kept_rows(rows, drop_rows, device)
    return _KEEP_INDEX.put(key, index)


def assert_pads_are_a_tail(drop_rows, rows: int) -> None:
    """Refuse a query-group bias whose pad rows are not the sequence tail.

    Group boundaries are coordinates in the sequence, so dropping rows from the
    middle of it would slide every later boundary and bias the wrong tokens.
    Every stream that carries a bias today shards alone, which puts its pads at
    the tail and leaves the boundaries exact. A layout that ever breaks that has
    to be measured, not assumed, so it stops here rather than rendering.
    """
    if not drop_rows:
        return
    tail = range(rows - len(drop_rows), rows)
    if list(drop_rows) != list(tail):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"a query-group bias met {len(drop_rows)} divisibility pad row(s) "
            f"away from the tail of its {rows}-row sequence, where the group "
            "boundaries stop being exact. This is a dgx-monarch fault, not a "
            "workflow setting: report the shapes. Use topology 'single' "
            "(mode=local with gpus_per_host=1) until it is fixed.",
        ))


def attend_without_pads(q, k, v, *, drop_rows, kv_drop_rows, groups, attend):
    """Run ``attend`` over the real rows and restore zeros where the pads sat.

    ``q``/``k``/``v`` arrive as (B, L, H, D) over the whole token axis, which is
    what the Ulysses head scatter hands each rank. ``attend(q, k, v, groups)``
    is the caller's kernel. The result keeps the query layout it was given, so
    ``sp_gather`` still trims the pad rows the caller sharded in.
    """
    if groups:
        assert_pads_are_a_tail(drop_rows, q.shape[1])
    query_index = keep_index(q.shape[1], drop_rows, q.device)
    if kv_drop_rows is drop_rows and k.shape[1] == q.shape[1] and k.device == q.device:
        key_index = query_index   # a self-attention drops the same rows on both sides
    else:
        key_index = keep_index(k.shape[1], kv_drop_rows, k.device)
    out = attend(
        q if query_index is None else q.index_select(1, query_index),
        k if key_index is None else k.index_select(1, key_index),
        v if key_index is None else v.index_select(1, key_index),
        groups,
    )
    if query_index is None:
        return out
    full = q.new_zeros(*q.shape)
    full[:, query_index] = out
    return full
