"""Distribute separate conditioning calls at ComfyUI's CALC_COND_BATCH wrapper.

When conditionings do not concatenate, ``cfg_parallel`` has no batch to split.
Give each CFG rank a contiguous group of condition indices and mask the rest
with ``None``. Stock ``_calc_cond_batch`` initializes each output to zeros,
so masked indices remain zero while active calls retain single-GPU semantics.
Gather outputs back in condition order.

Each step costs one all-gather of ``world * ceil(conds / world)`` latent-sized
tensors plus the local stack. ``multigpu_clones`` and ``context_handler`` remain
unchanged. Use ComfyUI's ``can_concat_cond`` and each condition object's
``can_concat`` rules for shapes, repeats, and constants rather than inferring
compatibility from text length alone. See docs/ADAPTERS.md.
"""
from __future__ import annotations

import contextlib
import inspect
import math
from collections.abc import Iterator

import torch

from ..log import get_logger
from .base import Adapter, cfg_rank, cfg_world

log = get_logger(__name__)

# Comfy applies this seam itself for every family, so no adapter declares it.
CFG_DISPATCH_SEAM = "calc_cond_batch"

# No value for the family's constant: a unique object, so it never equals a published value.
_ABSENT = object()

# Keys comfy tests by presence before it compares two conds: they narrow
# ``input_x``, or drop the cond for this step. The rule below models none, so a
# cond carrying one keeps the slice path, the path every family was measured on.
_UNMODELLED_COND_KEYS = ("default", "area", "mask", "gligen",
                         "timestep_start", "timestep_end")

CFG_PATHS = ("slice", "dispatch")
_last_path: str | None = None


def cfg_path_taken() -> str | None:
    """The path the most recent cond batch took. The gate ledger records
    identity verdicts, not per-render traces, so the path is kept here."""
    return _last_path


def _record_path(path: str) -> None:
    global _last_path
    if path == _last_path:
        return
    _last_path = path
    log.info("cfg-parallel cond batch takes the %s path", path)


_dispatching = False


def dispatch_in_progress() -> bool:
    """Whether this rank is inside its own per-cond call. The slice wrapper
    reads it and passes such a call through: the rank owns every row already.
    Compute is serialized on one thread, so a flag suffices."""
    return _dispatching


@contextlib.contextmanager
def _dispatch_scope() -> Iterator[None]:
    global _dispatching
    previous, _dispatching = _dispatching, True
    try:
        yield
    finally:
        _dispatching = previous


# Whether this render's original (pre-pad) text lengths fold, which decides
# nvfp4's cfg scale (docs/VALIDATION.md). Like `_dispatching`, a
# module global set from a rank-replicated input and read with no collective.
_cfg_pair_folds = True


def cross_attn_pair_folds(a: int, b: int) -> bool:
    """Comfy's `CONDCrossAttn.can_concat` length rule (comfy/conds.py): equal
    folds; unequal folds only at an LCM repeat of 4 or less. Models the length
    axis alone, not that method's shape/device checks."""
    if a == b:
        return True
    lo, hi = (a, b) if a < b else (b, a)
    if lo <= 0:
        return False
    return math.lcm(lo, hi) // lo <= 4


def record_cfg_pair_lengths(lengths: list[int]) -> None:
    """Set `cfg_pair_folds` from this render's original text lengths, every
    length against the first as `conds_fold` does. Called once per render
    (`sample_protocol.run_sample`, as the default) and, where
    `equalize_cond_lengths` runs (pure cfg, and only when
    `cfg_dispatches_per_cond` is false), again by it: with the real lengths for
    a family whose cfg-pad forward restores the stock call, with an empty list
    (fold) otherwise."""
    global _cfg_pair_folds
    _cfg_pair_folds = (
        all(cross_attn_pair_folds(lengths[0], length) for length in lengths[1:])
        if lengths else True)


def cfg_pair_folds() -> bool:
    """The last `record_cfg_pair_lengths` answer; true (fold) by default."""
    return _cfg_pair_folds


def cond_group_bounds(count: int, world: int, rank: int) -> tuple[int, int]:
    """The half-open cond-index group this rank runs. Contiguous, so the
    gathered order is the original order: slot ``j`` is global index
    ``rank * size + j``. A rank past the end owns nothing."""
    size = -(-count // world)
    start = min(rank * size, count)
    return start, min(start + size, count)


def _present_conds(conds: list) -> list[dict]:
    """Every conditioning dict this step will run, flattened over the slots."""
    return [item for entry in conds or [] if entry for item in entry
            if isinstance(item, dict)]


def _constant_values(entries: list[dict], key: str) -> list:
    """Every value the entries publish for one ``CONDConstant`` key. A
    constant holds its value on ``cond``, which reads with no comfy import."""
    values = []
    for item in entries:
        found = (item.get("model_conds") or {}).get(key)
        values.append(getattr(found, "cond", _ABSENT) if found is not None else _ABSENT)
    return values


def _constants_agree(entries: list[dict], key: str) -> bool:
    """The declared constant, read on its own and first: a family publishing
    the caption's real length has that one value decide the whole question. It
    is the comparison ``CONDConstant.can_concat`` makes."""
    values = _constant_values(entries, key)
    return all(value is values[0] or value == values[0] for value in values)


def _readable(model_conds) -> bool:
    """Whether every cond object here answers the two methods comfy calls."""
    return isinstance(model_conds, dict) and all(
        hasattr(value, "process_cond") and hasattr(value, "can_concat")
        for value in model_conds.values())


def _conditionings_concat(first: dict, other: dict, batch: int) -> bool:
    """comfy's ``cond_equal_size``: same keys, then each cond's own
    ``can_concat``, over the processed conds comfy itself compares."""
    if first.keys() != other.keys():
        return False
    for name, value in first.items():
        left = value.process_cond(batch_size=batch, area=None)
        right = other[name].process_cond(batch_size=batch, area=None)
        if not left.can_concat(right):
            return False
    return True


def conds_fold(conds: list, x_in: torch.Tensor, key: str | None = None) -> bool:
    """Whether comfy can batch this step's conditionings into one model call.

    Every conditioning is taken against the first, the group comfy's own loop
    builds before it looks at anything else. The declared constant answers
    first and alone; after it, a cond this rule does not model keeps the slice.

    Rank-replicated by construction: the cond tensors and constants, the latent
    shape and the family's key, all shipped identically to every rank. Comfy's
    loop reads free memory too, which differs per rank, but that only decides
    how many concatenable conds share one call, never whether they can concat.
    """
    entries = _present_conds(conds)
    if len(entries) < 2:
        return True
    if key is not None and not _constants_agree(entries, key):
        return False
    if any(name in item for item in entries for name in _UNMODELLED_COND_KEYS):
        return True
    # Comfy groups by hook group first, and a rank running one group alone
    # would patch weights the other did not. Read by value: the key can be None.
    if any(item.get("hooks") is not None for item in entries):
        return True
    first = entries[0]
    # comfy compares control by identity: one shared object concatenates.
    if any(item.get("control") is not first.get("control") for item in entries):
        return True
    conditionings: list = [item.get("model_conds") for item in entries]
    if not all(_readable(value) for value in conditionings):
        return True
    batch = int(x_in.shape[0])
    return all(_conditionings_concat(conditionings[0], other, batch)
               for other in conditionings[1:])


def _present_cond_indices(conds: list) -> list[int]:
    """Slots carrying work: at CFG 1.0 comfy passes ``[cond, None]``."""
    return [index for index, entry in enumerate(conds) if entry]


def _gather_cond_outputs(local: list, count: int, world: int,
                         start: int, stop: int, like: torch.Tensor) -> list:
    """Reassemble the sampler's output list from every rank's group. Each
    output is ``zeros_like(x_in)``, so the entries share one shape whatever the
    conditionings looked like and the stack is always legal."""
    from .cfg_parallel import _all_gather_cfg

    size = -(-count // world)
    slots = [local[index] for index in range(start, stop)]
    slots.extend(torch.zeros_like(like) for _ in range(size - len(slots)))
    gathered = _all_gather_cfg(torch.stack(slots))
    return [gathered[index] for index in range(count)]


def make_cond_dispatch_wrapper(adapter: Adapter):
    """Build the keyed CALC_COND_BATCH wrapper for `adapter`'s family."""
    key = getattr(adapter, "cfg_batch_constant", None)

    def cond_dispatch_wrapper(executor, *args, **kwargs):
        bound = inspect.signature(executor.original).bind_partial(*args, **kwargs)
        conds = bound.arguments["conds"]
        x_in = bound.arguments["x_in"]
        world = cfg_world()
        # Decided before any collective, from facts every rank holds: the cond
        # list the sampler built and cfg_world(). Both ranks reach the same
        # branch, so neither enters a gather the other skipped.
        if world <= 1 or len(_present_cond_indices(conds)) < 2 \
                or conds_fold(conds, x_in, key):
            _record_path("slice")
            return executor(*bound.args, **bound.kwargs)

        _record_path("dispatch")
        count = len(conds)
        start, stop = cond_group_bounds(count, world, cfg_rank())
        bound.arguments["conds"] = [
            entry if start <= index < stop else None
            for index, entry in enumerate(conds)
        ]
        with _dispatch_scope():
            local = executor(*bound.args, **bound.kwargs)
        return _gather_cond_outputs(local, count, world, start, stop, x_in)

    return cond_dispatch_wrapper


def install_cond_dispatch_wrapper(adapter: Adapter, model_options: dict) -> str:
    """Install per-condition dispatch for every family on a CFG topology.

    ComfyUI's concatenation rule decides eligibility without a family opt-in.
    Pairs that concatenate pass through to batch slicing. Check the existing
    key first because ``add_wrapper_with_key`` appends rather than replaces.
    """
    import comfy.patcher_extension as pe

    from ..constants import CFG_DISPATCH_WRAPPER_KEY

    if not pe.get_wrappers_with_key(
            pe.WrappersMP.CALC_COND_BATCH, CFG_DISPATCH_WRAPPER_KEY,
            model_options, is_model_options=True):
        pe.add_wrapper_with_key(
            pe.WrappersMP.CALC_COND_BATCH, CFG_DISPATCH_WRAPPER_KEY,
            make_cond_dispatch_wrapper(adapter), model_options, is_model_options=True)
    return pe.WrappersMP.CALC_COND_BATCH
