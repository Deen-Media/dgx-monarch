"""Refuse multi-rank sampling when any model is only partially resident.

ComfyUI decides placement during ``prepare_sampling -> load_models_gpu``.
Ranks with different free memory can choose different residency: partially
loaded weights are streamed and cast per step, which can change their math
relative to fully resident weights. The LTX 2.5 promotion evidence in
docs/VALIDATION.md records the resulting cross-rank latent drift.

A ``PREPARE_SAMPLING`` wrapper checks completeness after loading and before
the first forward, including Ulysses, Ring, FSDP, and CFG collectives. Exchange
the local flag over the default process group before deciding, so every rank
raises the same typed refusal and no healthy peer enters a collective alone.

The guard refuses rather than silently changing the requested residency.
"""
from __future__ import annotations

from typing import Any

from ..log import get_logger
from ..refusal import RefusalClass, refusal

log = get_logger(__name__)

GUARD = "partial_load_divergence"
WRAPPER_KEY = "dgx_monarch_partial_load_guard"
TROUBLESHOOTING = 79


class PartialLoadDivergenceError(RuntimeError):
    """Typed cross-rank refusal: the ranks disagree about weight residency.

    Never a ``StockLoadCapacityError``: that family routes to the residency
    rescue ladder and to the ceremony's "cannot load under stock" verdict, and
    neither is true here. The weights loaded. What failed is an agreement
    between ranks, and it is knowable only after an exchange.
    """


# Build refusal text at the raise, never during driver-side exception imports:
# a long-lived driver's refusal vocabulary may differ from newly deployed workers.
# MIN reduction carries agreement, not identity, so omit hosts and ranks here;
# each worker journal records its own cause.
REFUSAL_TEXT = (
    "the ranks disagree about weight residency: this render needs every rank to "
    "hold the whole model, and at least one holds it only partially, because "
    "ComfyUI offloaded part of its weights under memory pressure. Several ranks "
    "can be short at once, at different splits, and that is the same fault; the "
    "check reads only whether every rank is whole, not which rank is short. A "
    "partially held rank streams and casts those weights per block, so the ranks "
    "would compute different weights for the same step and the render would "
    "diverge without an error. Nothing was sampled and no rank went on. "
    "Each worker journal names how much its own box loaded and offloaded. What "
    "would fit: an int8 or fp8 checkpoint in place of bf16, "
    "lora_low_rss on so the LoRA stack bakes and ComfyUI's weight backup is "
    "freed, or the same render on a single box."
)


def local_full_load(model: Any) -> bool:
    """Whether ComfyUI reports this rank's weights fully resident.

    ``model_lowvram`` is comfy's own flag, set beside the "loaded completely /
    loaded partially" journal line; worker status reports the same signal as
    ``full_load``.
    """
    return not bool(getattr(model, "model_lowvram", False))


def _distributed_world() -> int:
    """World size of the live default process group, or 1 when there is none."""
    try:
        import torch.distributed as dist
    except Exception:
        return 1
    try:
        if not dist.is_available() or not dist.is_initialized():
            return 1
        return int(dist.get_world_size())
    except Exception:
        return 1


def _all_ranks_full(local_full: bool) -> bool:
    """MIN-reduce one flag over the default group; True only if every rank is full.

    One element, on the group's own device, over the group every rank already
    belongs to. Never a sub-group (sp, cfg, dp): a dp rank that computes a
    divergent slice still corrupts the concatenated batch, so the agreement has
    to span the whole world.
    """
    import torch
    import torch.distributed as dist

    device: str | torch.device = "cpu"
    try:
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
    except Exception:
        device = "cpu"
    flag = torch.tensor([1 if local_full else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(int(flag.item()))


def all_ranks_true(local_flag: bool) -> bool:
    """MIN-reduce one boolean over the default group; True only if every rank is True.

    Off-group (world 1, no process group) it returns the local flag. Reuses the
    guard's MIN all_reduce, the deadlock-safe exchange that crosses each rank's
    own fact before any rank decides. ``rank_readiness.exchange_ready`` wraps it
    for the post-load readiness exchange and for dm-cfg2 residency.
    """
    if _distributed_world() <= 1:
        return local_flag
    return _all_ranks_full(local_flag)


def check(model: Any, additional: tuple = ()) -> None:
    """Exchange this rank's load completeness, then raise on every rank or none.

    Single-rank runs never raise: one rank cannot disagree with itself, and a
    partial load there is comfy's ordinary, correct behavior. ``additional``
    covers every other model the same prepare loaded (dual-model guidance
    passes the uncond checkpoint through here): a partially held second model
    diverges the ranks exactly like a partially held first.
    """
    world = _distributed_world()
    if world <= 1:
        return
    local_full = local_full_load(model) and all(
        local_full_load(getattr(extra, "model", extra)) for extra in additional)
    # Logged before the exchange, on every rank: MIN erases which box was
    # short, so the journals are the only place that fact survives, and only
    # this line tells a render that passed from a guard that never ran.
    log.info("partial-load guard: local full_load=%s world=%d", local_full, world)
    if _all_ranks_full(local_full):
        return
    raise PartialLoadDivergenceError(refusal(
        RefusalClass.CAPACITY, REFUSAL_TEXT, guard=GUARD, waivable=False,
        troubleshooting=TROUBLESHOOTING,
    ))


def make_prepare_sampling_guard():
    """Build the ``PREPARE_SAMPLING`` wrapper that runs the check after the load."""

    def wrapper(executor, *args, **kwargs):
        result = executor(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) and result else None
        if model is None and args:
            model = getattr(args[0], "model", None)
        additional = (
            tuple(result[2]) if isinstance(result, tuple) and len(result) > 2
            and isinstance(result[2], (list, tuple)) else ())
        check(model, additional=additional)
        return result

    return wrapper


def install(base_patcher: Any, world: int | None) -> None:
    """Register the guard on a freshly loaded base patcher, once, above world 1.

    Read the key before writing it: comfy appends on a second add under one
    key, so a second install would run the exchange twice per sample.
    """
    if not world or int(world) <= 1:
        return
    import comfy.patcher_extension as pe

    options = base_patcher.model_options
    if pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, options, is_model_options=True
    ):
        return
    pe.add_wrapper_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, make_prepare_sampling_guard(),
        options, is_model_options=True,
    )
