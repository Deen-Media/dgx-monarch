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
takes the minimum over the ranks that share the model call and lets ComfyUI's
own rule run against that one figure, so each of them takes the leanest one's
batch. It costs one 8-byte all-reduce per model call. ``partial_load_guard.install``
registers it beside the partial-load guard, on every base either slot loads,
so every rank of a sample enters the exchange the same number of times, and
only where the topology shards a model call (Ulysses times ring above 1, cfg
above 1, or FSDP). A pure data-parallel model call runs no collective and its
ranks need not agree, and there the exchange would only make a rank that
stopped early leave its peer waiting until the stall budget. World 1, and
every Fleet job (each a world-1 mesh), never install it.

The same holds between data-parallel groups when a layout also shards the
call (uly2+dp2 or cfg2+dp2 at world 4): each group samples its own slice of
the batch, and an adaptive-step sampler such as ``dpm_adaptive`` sizes its
steps from that slice, so the groups can make different numbers of calls.
There the exchange runs only over the ranks of one data-parallel group, which
hold every Ulysses, ring and cfg collective of the call, and over a group the
call's own collectives already use where one has exactly those ranks: the
sequence-parallel group when cfg is 1 (every sharded forward ends in its
all-gather, ``adapters/base.sp_gather``), the cfg group when Ulysses times
ring is 1 (the cfg split's all-gather, ``adapters/cfg_parallel``). Only a
layout with both above 1 (uly2+cfg2+dp2 at world 8) takes xFuser's model
replica group, which nothing else runs over. xFuser builds all of them once at
setup, on every rank in the same order (``actor/worker_env.py``), so no group
is made here. Everywhere else the exchange stays on the default group: with
one data-parallel group it is the same ranks, and FSDP gathers each call's
weights over the whole world, so its data-parallel ranks share every call
(the driver refuses an adaptive-step sampler there, ``nodes/adaptive_steps.py``).
The pair never leaves the default group: at world 2 a layout with two
data-parallel groups is pure data parallel or ``dp2+fsdp``.

The figure stays pinned through the whole model call, the forward included, so
code inside the forward that reads ``current_patcher.get_free_memory`` sees the
leanest rank's figure from the start of the call, before the call's own
activations. Qwen Image 2.1's prefix-cache placement reads it there
(``comfy/ldm/qwen_image21/model.py``); ``adapters/qwen_image21.agree_prefix_cache``
keeps the ranks' choices equal. The pin cannot be narrowed to the batching pass
without losing agreement on the later batch decisions of the same call.
"""
from __future__ import annotations

from collections.abc import Mapping
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


def leanest_free_memory(local: int, group: Any = None) -> int:
    """MIN-reduce one byte count over ``group`` (the default group when None);
    the local count off-group."""
    local = max(int(local), 0)
    if _world() <= 1:
        return local
    import torch
    import torch.distributed as dist

    device: str | torch.device = "cpu"
    try:
        if dist.get_backend(group) == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
    except Exception:
        device = "cpu"
    value = torch.tensor([local], dtype=torch.int64, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN, group=group)
    return int(value.item())


def _call_group(kind: str) -> Any:
    """The torch group ``kind`` names for this rank, as xFuser built it at setup:
    ``"sp"``, ``"cfg"`` or ``"replica"`` (this rank's data-parallel group).

    Read on every call, never kept: a handle kept past a teardown would name a
    destroyed group. A lookup that fails raises; falling back to the default
    group would bring back the wait across groups.
    """
    from xfuser.core.distributed import get_cfg_group, get_model_replica_group, get_sp_group

    coordinator = {"sp": get_sp_group, "cfg": get_cfg_group,
                   "replica": get_model_replica_group}[kind]()
    return coordinator.device_group


def _announce(local: int, leanest: int) -> None:
    global _announced
    if _announced:
        return
    _announced = True
    log.info("cond batch agreement: ComfyUI's cond batch rule reads the leanest rank's "
             "free memory on every rank (%.1f GiB here, %.1f GiB on the leanest rank "
             "at the first model call)", local / 2 ** 30, leanest / 2 ** 30)


def make_wrapper(group_kind: str | None = None):
    """Build the ``CALC_COND_BATCH`` wrapper that pins the batch decision.

    ``group_kind`` names the group the exchange runs over in place of the
    default group (``exchange_group_kind`` says which).
    """

    def wrapper(executor, *args, **kwargs):
        model = args[0] if args else kwargs.get("model")
        x_in = args[2] if len(args) > 2 else kwargs.get("x_in")
        patcher = getattr(model, "current_patcher", None)
        if patcher is None or x_in is None or _world() <= 1:
            return executor(*args, **kwargs)
        local = int(patcher.get_free_memory(x_in.device))
        if group_kind is not None:
            leanest = leanest_free_memory(local, _call_group(group_kind))
        else:
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


def shards_model_calls(topology: Any) -> bool:
    """Whether a model call under ``topology`` runs collectives sized by its batch."""
    if not isinstance(topology, Mapping):
        return False
    try:
        sp = int(topology.get("ulysses", 1) or 1) * int(topology.get("ring", 1) or 1)
        return sp > 1 or int(topology.get("cfg", 1) or 1) > 1 or bool(topology.get("fsdp"))
    except (TypeError, ValueError):
        return False


def model_call_spans_world(topology: Any) -> bool:
    """Whether every rank of the world shares each model call under ``topology``.

    False only with more than one data-parallel group and no FSDP: then each
    group samples its own slice and shares no collective with another.
    """
    if not isinstance(topology, Mapping):
        return True
    try:
        return int(topology.get("dp", 1) or 1) <= 1 or bool(topology.get("fsdp"))
    except (TypeError, ValueError):
        return True


def exchange_group_kind(topology: Any) -> str | None:
    """The group the exchange runs over under ``topology``: None for the default
    group, else the smallest xFuser group that holds the ranks sharing the call
    and that the call's own collectives already use, if one does."""
    if model_call_spans_world(topology):
        return None
    sp = int(topology.get("ulysses", 1) or 1) * int(topology.get("ring", 1) or 1)
    cfg = int(topology.get("cfg", 1) or 1)
    if cfg <= 1:
        return "sp"
    if sp <= 1:
        return "cfg"
    return "replica"


def install(base_patcher: Any, world: int | None, topology: Any = None) -> None:
    """Register the wrapper on a freshly loaded base patcher, once, where a
    world above 1 shards its model calls.

    Read the key before writing it: comfy appends on a second add under one
    key, so a second install would run the exchange twice per call.
    """
    if not world or int(world) <= 1 or not shards_model_calls(topology):
        return
    import comfy.patcher_extension as pe

    options = base_patcher.model_options
    if pe.get_wrappers_with_key(
        pe.WrappersMP.CALC_COND_BATCH, WRAPPER_KEY, options, is_model_options=True
    ):
        return
    pe.add_wrapper_with_key(
        pe.WrappersMP.CALC_COND_BATCH, WRAPPER_KEY,
        make_wrapper(exchange_group_kind(topology)),
        options, is_model_options=True,
    )
