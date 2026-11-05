"""Make every rank batch its conds the way the leanest rank does.

ComfyUI decides how many conds share one model call from the free memory of
the box it runs on: ``comfy/samplers.py`` ``_calc_cond_batch`` runs a group
whole when ``model.memory_required(...) * 1.5`` is under
``model.current_patcher.get_free_memory(...)`` and splits it otherwise. On a
pair of Sparks each rank reads its own box (``actor/comfy_bridge.py`` reports
the host's MemAvailable), so a short head can run cond and uncond as two calls
while the sibling runs them as one. Under Ulysses, ring, FSDP or cfg-parallel
every model call is a set of collectives whose sizes follow the batch, and the
ranks then wait on collectives that do not match until the sample deadline.
On 2026-11-04 the Wan 2.2 text-to-video gate stalled for 600 s with rank 1's
all-to-all and all-gather carrying exactly twice rank 0's elements.

A ``CALC_COND_BATCH`` wrapper reads this rank's free memory once per call,
takes the minimum over the default process group and lets ComfyUI's own rule
run against that one figure, so every rank takes the leanest rank's batch.
It costs one 8-byte all-reduce per model call. World 1, and every Fleet job
(each a world-1 mesh), pass straight through. ``partial_load_guard.install``
registers it beside the partial-load guard, on every base either slot loads,
so every rank of a sample enters the exchange the same number of times.
"""
from __future__ import annotations

from typing import Any

from ..log import get_logger

log = get_logger(__name__)

WRAPPER_KEY = "dgx_monarch_cond_batch_agreement"
_METHOD = "get_free_memory"
_ABSENT = object()
_announced = False  # process lifetime: the journal names the agreement once


def _world() -> int:
    from .partial_load_guard import _distributed_world

    return _distributed_world()


def leanest_free_memory(local: int) -> int:
    """MIN-reduce one byte count over the default group; the local count off-group."""
    local = max(int(local), 0)
    if _world() <= 1:
        return local
    import torch
    import torch.distributed as dist

    device: str | torch.device = "cpu"
    try:
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
    except Exception:
        device = "cpu"
    value = torch.tensor([local], dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return int(value.item())


def _announce(local: int, leanest: int) -> None:
    global _announced
    if _announced:
        return
    _announced = True
    log.info("cond batch agreement: ComfyUI's cond batch rule reads the leanest rank's "
             "free memory on every rank (%.1f GiB here, %.1f GiB on the leanest rank "
             "at the first model call)", local / 2 ** 30, leanest / 2 ** 30)


def make_wrapper():
    """Build the ``CALC_COND_BATCH`` wrapper that pins the batch decision."""

    def wrapper(executor, *args, **kwargs):
        model = args[0] if args else kwargs.get("model")
        x_in = args[2] if len(args) > 2 else kwargs.get("x_in")
        patcher = getattr(model, "current_patcher", None)
        if patcher is None or x_in is None or _world() <= 1:
            return executor(*args, **kwargs)
        local = int(patcher.get_free_memory(x_in.device))
        leanest = leanest_free_memory(local)
        _announce(local, leanest)
        held = patcher.__dict__.get(_METHOD, _ABSENT)
        patcher.get_free_memory = lambda *_args, **_kwargs: leanest
        try:
            return executor(*args, **kwargs)
        finally:
            if held is _ABSENT:
                patcher.__dict__.pop(_METHOD, None)
            else:
                patcher.get_free_memory = held

    return wrapper


def install(base_patcher: Any, world: int | None) -> None:
    """Register the wrapper on a freshly loaded base patcher, once, above world 1.

    Read the key before writing it: comfy appends on a second add under one
    key, so a second install would run the exchange twice per call.
    """
    if not world or int(world) <= 1:
        return
    import comfy.patcher_extension as pe

    options = base_patcher.model_options
    if pe.get_wrappers_with_key(
        pe.WrappersMP.CALC_COND_BATCH, WRAPPER_KEY, options, is_model_options=True
    ):
        return
    pe.add_wrapper_with_key(
        pe.WrappersMP.CALC_COND_BATCH, WRAPPER_KEY, make_wrapper(),
        options, is_model_options=True,
    )
