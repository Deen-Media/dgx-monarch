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

import json
from itertools import accumulate
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
    between ranks, and it is knowable only after an exchange. Every rank raises
    it alike, so it speaks for the fleet and carries no rank's mark
    (``failure.mark_rank``): the driver names no box ahead of it.
    """

    fleet_wide = True


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


def _all_ranks_min(values: list[int]) -> list[int]:
    """MIN-reduce a few integers over the default group in ONE collective; every rank gets the same list.

    The guard's own flag rides first and the pair's backup agreement after it
    (``backup_agreement``), so agreeing costs no collective a render did not
    already make. Every rank must send the same count, which the callers fix
    from the topology and the call's shape, never from a rank's own state.
    """
    import torch
    import torch.distributed as dist

    device: str | torch.device = "cpu"
    try:
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
    except Exception:
        device = "cpu"
    sent = torch.tensor([int(value) for value in values], dtype=torch.int32, device=device)
    dist.all_reduce(sent, op=dist.ReduceOp.MIN)
    return [int(value) for value in sent.tolist()]


# The most one rank's words may take in ``_all_ranks_texts``: a refusal's
# figures as one JSON line, which ``fit_line`` keeps inside it.
TEXT_BYTES = 1024


def fit_line(facts: dict, key: str, mark: str) -> str:
    """``facts`` as one JSON line of at most ``TEXT_BYTES``, its ``key`` text cut to fit; "{}" if none fits.

    ``json.dumps`` escapes to ASCII, one byte a character, so the line's length is its size on the
    wire and the gather never cuts it: it always parses. A text too long once escaped (quotes,
    backslashes, newlines and every character outside ASCII grow, an emoji to twelve) keeps as
    many whole characters as fit and ends with ``mark``. Each character escapes on its own, so the
    sizes add up, and each takes at least one byte.
    """
    text, line = str(facts.get(key, "")), json.dumps(facts)
    if len(line) > TEXT_BYTES:
        left = TEXT_BYTES - len(json.dumps({**facts, key: mark}))
        sizes = accumulate(len(json.dumps(char)) - 2 for char in text[:max(0, left)])
        line = json.dumps({**facts, key: text[:sum(1 for size in sizes if size <= left)] + mark})
    return line if len(line) <= TEXT_BYTES else "{}"


def _all_ranks_texts(text: str) -> list[str]:
    """Gather one short text from every rank, in rank order, in ONE collective.

    MIN carries agreement and not identity, so a refusal decided from it cannot
    say which rank was short or wrong, or why. Only a refusal path that every
    rank takes alike, from the same MIN answer, calls this
    (``backup_agreement.settle``), so a render that samples makes no collective
    more. Every rank sends ``TEXT_BYTES`` bytes of UTF-8, the same count on
    every rank, zero-padded.
    """
    import torch
    import torch.distributed as dist

    device: str | torch.device = "cpu"
    try:
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
    except Exception:
        device = "cpu"
    data = list(text.encode("utf-8")[:TEXT_BYTES])
    sent = torch.zeros(TEXT_BYTES, dtype=torch.uint8, device=device)
    if data:
        sent[:len(data)] = torch.tensor(data, dtype=torch.uint8, device=device)
    got = [torch.zeros_like(sent) for _ in range(int(dist.get_world_size()))]
    dist.all_gather(got, sent)
    return [bytes(item.tolist()).rstrip(b"\0").decode("utf-8", "ignore") for item in got]


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


def _rank() -> int:
    try:
        import torch.distributed as dist

        return int(dist.get_rank())
    except Exception:
        return 0


def check(model: Any, additional: tuple = (), agree: tuple[tuple, tuple[Any, Any]] | None = None) -> None:
    """Exchange this rank's load completeness, then raise on every rank or none.

    Single-rank runs never raise: one rank cannot disagree with itself, and a
    partial load there is comfy's ordinary, correct behavior. ``additional``
    covers every other model the same prepare loaded (dual-model guidance
    passes the uncond checkpoint through here): a partially held second model
    diverges the ranks exactly like a partially held first.

    ``agree`` is (the patchers this prepare loaded, its latent shape and
    conds) on a topology where every rank holds the whole model. The same
    exchange then carries the pair's backup agreement (``backup_agreement``):
    the ranks free ComfyUI's weight backup together or not at all. Whether it
    is given is fixed by the topology and the call's shape, the same on every
    rank, so every rank sends the same count of integers.
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
    if agree is None:
        if _all_ranks_full(local_full):
            return
    else:
        from . import backup_agreement

        mine = backup_agreement.vote(agree[0], agree[1], _rank())
        answer = _all_ranks_min([1 if local_full else 0, *mine.elements])
        if answer[0] == 1:
            backup_agreement.settle(mine, answer[1:], _all_ranks_min, _all_ranks_texts)
            return
    raise PartialLoadDivergenceError(refusal(
        RefusalClass.CAPACITY, REFUSAL_TEXT, guard=GUARD, waivable=False,
        troubleshooting=TROUBLESHOOTING,
    ))


def make_prepare_sampling_guard(whole_model: bool = False):
    """Build the ``PREPARE_SAMPLING`` wrapper that runs the check after the load.

    ``whole_model`` (every rank holds the whole checkpoint: not FSDP) adds the
    pair's backup agreement to the same exchange.
    """

    def wrapper(executor, *args, **kwargs):
        result = executor(*args, **kwargs)
        model = result[0] if isinstance(result, tuple) and result else None
        if model is None and args:
            model = getattr(args[0], "model", None)
        additional = (
            tuple(result[2]) if isinstance(result, tuple) and len(result) > 2
            and isinstance(result[2], (list, tuple)) else ())
        agree = None
        if whole_model:
            patcher = args[0] if args else kwargs.get("model")
            shape = args[1] if len(args) > 1 else kwargs.get("noise_shape")
            conds = args[2] if len(args) > 2 else kwargs.get("conds")
            if patcher is not None and shape is not None and conds is not None:
                agree = ((patcher, *additional), (shape, conds))
        if agree is None:
            check(model, additional=additional)  # the call dev made, for a caller that wraps it
        else:
            check(model, additional=additional, agree=agree)
        return result

    return wrapper


def install(base_patcher: Any, world: int | None, topology: Any = None) -> None:
    """Register the guard on a freshly loaded base patcher, once, above world 1.

    Read the key before writing it: comfy appends on a second add under one
    key, so a second install would run the exchange twice per sample. The
    cond batch agreement rides the same install where ``topology`` shards a
    model call: it answers the same fault, ranks that decide from their own
    box's free memory, one call later.
    """
    if not world or int(world) <= 1:
        # One rank cannot disagree with itself, and a partial load there is
        # exact for every patched layer but one kind, which has its own check.
        from . import lora_partial_load

        lora_partial_load.install_guard(base_patcher)
        return
    import comfy.patcher_extension as pe

    from . import cond_batch_agreement

    cond_batch_agreement.install(base_patcher, world, topology)

    options = base_patcher.model_options
    if pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, options, is_model_options=True
    ):
        return
    # Every rank of a topology that is not FSDP holds the whole checkpoint; there the pair's backup
    # agreement rides the guard's exchange. Read from the topology alone, so every rank reads it alike.
    fsdp = bool(topology.get("fsdp")) if isinstance(topology, dict) else False
    pe.add_wrapper_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, make_prepare_sampling_guard(whole_model=not fsdp),
        options, is_model_options=True,
    )
