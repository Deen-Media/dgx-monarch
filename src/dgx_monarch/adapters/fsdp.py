"""FSDP capacity mode (DESIGN.md §5.4): fit models that do not fit resident.

FSDP is a capacity tool, never a speed tool. Each rank holds 1/world of the
weights and all-gathers per block during forward. The main use case is
Wan 2.2 14B on 121 GiB UMA boxes.

Measured evidence for the deadlock fix and the byte-identical capacity result:
docs/VALIDATION.md.
Launch constraints (typed errors, not silent degradation):
  * a bf16 or fp16 checkpoint with a matching denoising core. fp32 tensors
    are admitted as islands inside a bf16 core (fsdp_islands.py: at most 5%
    of the bytes, replicated through ``ignored_params``, never sharded);
    fp8 (per-tensor) and int8 (tensorwise, with or without convrot) files
    shard as stored bytes through fsdp_quant.py; block-scaled layouts (mxfp8,
    nvfp4) refuse, and a live cast to a quant kind is refused because it has
    no file-backed bytes,
  * a LoRA stack on a sharded model needs ``lora_low_rss`` on: the bake is
    the shard-aware full-then-slice path (actor/fsdp_lora.py), never comfy's
    load and backup machinery,
  * a LoRA stack also needs the checkpoint itself admitted for that bake
    (fsdp_lora_admission.py): a comfy-kitchen quantized checkpoint refuses
    typed before any load, whatever ``lora_low_rss`` or ``auto_gate`` says;
    a plain fp8-dtype file (Qwen-Image's e4m3fn) stays admitted. fp32
    islands are left to the bake's own per-key check.
Modules a family runs outside the denoise path (Anima's ``llm_adapter``, H3's
``condition_proj`` and ``token_refiner``, LTXAV's caption projections and
connectors) are ignored rather than wrapped (fsdp_islands.py says why).
Operational rule carried as config: never set NCCL_PROTO=LL. It is an
FSDP-killer (docs/VALIDATION.md). The shipped fabric profiles leave NCCL_PROTO
unset.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any

from ..capacity_fit import (
    ABSOLUTE_HOST_FLOOR_BYTES,
    FSDP_MATERIALIZE_FACTOR,  # noqa: F401 - compatibility re-export
    FSDP_STREAM_TRANSIENT_FRACTION,  # noqa: F401 - compatibility re-export
    fsdp_materialize_factor,
    fsdp_required_bytes,
)
from ..log import get_logger
from ..mesh_safety import (
    StockLoadCapacityError,
    gpu_is_integrated,
    mem_available_bytes,
)
from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError
from .fsdp_islands import (
    ALL_FP16_PROFILE,
    FP32_ISLANDS_MAX_BYTES,
    FP32_ISLANDS_PROFILE,
    auxiliary_module_parameters,
    classify_fp32_islands,
    island_parameters,
)
from .fsdp_wrap_trace import _allocator_pool_trim, _log_wrap_memory

log = get_logger(__name__)

ALL_BF16_PROFILE = "all_bf16"
WAN_FP32_INGRESS_PROFILE = "wan_fp32_patch_embedding_v1"
FSDP_ADMITTED_QUANT_KINDS = frozenset({"fp8", "int8"})
FSDP_ADMITTED_CHECKPOINT_KINDS = frozenset({"bf16", "fp16", *FSDP_ADMITTED_QUANT_KINDS})
# The live kind each admitted profile is built from; the live rescan before
# sharding must still read this kind, whatever the header-combined kind says.
_PROFILE_LIVE_KIND = {
    ALL_BF16_PROFILE: "bf16",
    ALL_FP16_PROFILE: "fp16",
    WAN_FP32_INGRESS_PROFILE: "bf16",
    FP32_ISLANDS_PROFILE: "bf16",
    "uniform_or_quantized_fp8": "fp8",
    "uniform_or_quantized_int8": "int8",
}
_WAN_FP32_INGRESS_PARAMETERS = frozenset({
    "patch_embedding.weight",
    "patch_embedding.bias",
})
_WAN_FP32_INGRESS_MAX_BYTES = 4 * 1024 * 1024
_WAN_FP32_INGRESS_MAX_FRACTION = 1e-3

_BLOCK_LIST_ATTRS = (
    "blocks", "transformer_blocks", "double_blocks", "single_blocks", "layers",
    # Omnigen2 and Boogu staged stacks, and PixelDiT's pixel stage. The wrap
    # finds block lists by attribute name, not by class.
    "double_stream_layers", "single_stream_layers",
    "context_refiner", "noise_refiner", "ref_image_refiner", "pixel_blocks",
    # Kandinsky5's text (encoder) and visual (decoder) block lists, visual
    # holding most parameters, and PixelDiT's patch stage, the MMDiTBlockT2I
    # stack that holds most of its parameters. Without these names Kandinsky5
    # hits the sharded_lists == 0 refusal below, and PixelDiT's patch_blocks
    # fall through to the root fully_shard(dm) instead of a wrap per block.
    "visual_transformer_blocks", "text_transformer_blocks", "patch_blocks",
)


def _is_audited_wan_fp32_ingress(
    base_model,
    diffusion_model,
    fp32_parameters: dict[str, Any],
    total_parameter_bytes: int,
    fp32_bytes: int,
) -> bool:
    """Recognize only Comfy's intentional FP32 Wan input convolution."""
    import torch
    try:
        from comfy import model_base, ops
        from comfy.ldm.wan.model import WanModel
        audited_base_types = (
            model_base.WAN21,
            model_base.WAN22,
            model_base.WAN21_FlowRVS,
        )
        audited_conv_types = (
            ops.disable_weight_init.Conv3d,
            ops.manual_cast.Conv3d,
        )
    except (AttributeError, ImportError):
        return False
    from .wan_animate2 import is_exact_animate2_pair
    animate2 = is_exact_animate2_pair(base_model, diffusion_model)
    if (
        (type(base_model) not in audited_base_types and not animate2)
        or (type(diffusion_model) is not WanModel and not animate2)
        or set(fp32_parameters) != _WAN_FP32_INGRESS_PARAMETERS
    ):
        return False
    patch_embedding = getattr(diffusion_model, "patch_embedding", None)
    if patch_embedding is None or type(patch_embedding) not in audited_conv_types:
        return False
    weight: torch.Tensor = fp32_parameters["patch_embedding.weight"]
    bias: torch.Tensor = fp32_parameters["patch_embedding.bias"]
    patch_size = tuple(getattr(diffusion_model, "patch_size", ()))
    dim = getattr(diffusion_model, "dim", None)
    in_dim = getattr(diffusion_model, "in_dim", None)
    if (
        weight is not getattr(patch_embedding, "weight", None)
        or bias is not getattr(patch_embedding, "bias", None)
        or weight.ndim != 5
        or bias.ndim != 1
        or weight.shape[0] != bias.numel()
        or weight.shape[0] != patch_embedding.out_channels
        or weight.shape[1] != patch_embedding.in_channels
        or patch_size != tuple(patch_embedding.kernel_size)
        or patch_size != tuple(patch_embedding.stride)
        or (patch_embedding.groups, tuple(patch_embedding.dilation),
            tuple(patch_embedding.padding), patch_embedding.padding_mode)
        != (1, (1, 1, 1), (0, 0, 0), "zeros")
        or tuple(weight.shape[2:]) != patch_size
        or patch_embedding.out_channels != dim
        or patch_embedding.in_channels != in_dim
    ):
        return False
    return (
        0 < fp32_bytes <= _WAN_FP32_INGRESS_MAX_BYTES
        and total_parameter_bytes > 0
        and fp32_bytes / total_parameter_bytes <= _WAN_FP32_INGRESS_MAX_FRACTION
    )


def classify_audited_live_precision(
    base_model,
    diffusion_model,
    fp32_parameters: dict[str, Any],
    total_parameter_bytes: int,
) -> tuple[str, int, int] | None:
    """Return adapter-owned evidence for one audited mixed-live layout."""
    fp32_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in fp32_parameters.values()
    )
    if _is_audited_wan_fp32_ingress(
        base_model,
        diffusion_model,
        fp32_parameters,
        total_parameter_bytes,
        fp32_bytes,
    ):
        return WAN_FP32_INGRESS_PROFILE, len(fp32_parameters), fp32_bytes
    return classify_fp32_islands(fp32_parameters, total_parameter_bytes)


def fsdp_precision_profile_is_admitted(
    profile: object,
    auxiliary_parameter_count: object,
    auxiliary_parameter_bytes: object,
    checkpoint_kind: object,
) -> bool:
    """Validate one public precision profile without duplicating family policy."""
    if (
        type(profile) is not str
        or type(auxiliary_parameter_count) is not int
        or type(auxiliary_parameter_bytes) is not int
        or type(checkpoint_kind) is not str
        or checkpoint_kind not in FSDP_ADMITTED_CHECKPOINT_KINDS
    ):
        return False
    uniform = auxiliary_parameter_count == 0 and auxiliary_parameter_bytes == 0
    if profile in ("uniform_or_quantized_fp8", "uniform_or_quantized_int8"):
        # Quantized bytes shard as stored (fsdp_quant.py); the row's checkpoint
        # kind must be that same file kind, never a live cast.
        return uniform and checkpoint_kind == profile.rsplit("_", 1)[1]
    if profile == ALL_BF16_PROFILE:
        # A bf16 core is admitted from either full-precision file: Comfy casts
        # an fp16 file into bf16 parameters on this hardware, and the
        # checkpoint kind on the row keeps saying which file it was. Never from
        # a quantized file: that core would be a dequantized copy, not the
        # stored bytes the quantized shard path (fsdp_quant.py) attests.
        return uniform and checkpoint_kind in ("bf16", "fp16")
    if profile == ALL_FP16_PROFILE:
        return uniform and checkpoint_kind == "fp16"
    if checkpoint_kind != "bf16":
        return False
    if profile == WAN_FP32_INGRESS_PROFILE:
        return (
            auxiliary_parameter_count == 2
            and 0 < auxiliary_parameter_bytes <= _WAN_FP32_INGRESS_MAX_BYTES
        )
    if profile == FP32_ISLANDS_PROFILE:
        return (
            auxiliary_parameter_count >= 1
            and 0 < auxiliary_parameter_bytes <= FP32_ISLANDS_MAX_BYTES
        )
    return False


def validate_fsdp_launch_quant(quant_kind: str) -> None:
    """Enforce the coarse FSDP launch contract on the checkpoint kind."""
    if quant_kind not in FSDP_ADMITTED_CHECKPOINT_KINDS:
        if quant_kind in {"mxfp8", "nvfp4"}:
            remediation = (
                "Block-scaled layouts have no registered shard layout; fp8 and "
                "int8 files shard as stored bytes."
            )
        elif quant_kind == "fp32":
            remediation = (
                "fp32 tensors are admitted only as islands "
                "inside a bf16 core, at most 5% of the "
                "checkpoint bytes."
            )
        else:
            remediation = "The detector could not prove the checkpoint's precision."
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "FSDP capacity mode supports bf16, fp16, fp8 and int8 checkpoints at "
            f"launch (got quant={quant_kind}). {remediation} Use a checkpoint of "
            "one of those kinds, or a resident topology for this file."
        ))


def validate_fsdp_live_precision(patcher, quant_kind: str, evidence) -> None:
    """Validate detector-owned live precision instead of trusting the quant_kind string alone."""
    from ..actor.store_detect import LivePrecisionEvidence

    validate_fsdp_launch_quant(quant_kind)
    if not isinstance(evidence, LivePrecisionEvidence):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "FSDP live precision evidence is missing or malformed. Use a readable "
            "safetensors checkpoint, or a resident topology for this file."
        ))
    if evidence.quant_kind != quant_kind:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "FSDP live precision evidence does not match the detected quant kind. "
            "Use a resident topology for this file, or reload it so the detector "
            "reads it again."
        ))
    if (
        evidence.checkpoint_kind not in FSDP_ADMITTED_CHECKPOINT_KINDS
        or evidence.checkpoint_kind != quant_kind
    ):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "FSDP precision evidence is not bound to a checkpoint of the "
            "detected kind (bf16, fp16, fp8 or int8). Use a checkpoint of one of "
            "those kinds, or a resident topology for this file."
        ))
    from ..actor.store_detect import _detect_live_precision

    # "bf16" asks the detector for the live scan; the kind it reads is the
    # core the profile was built from, not the header-combined kind.
    observed = _detect_live_precision(patcher, "bf16")
    if (
        observed.quant_kind != _PROFILE_LIVE_KIND.get(evidence.live_dtype_profile)
        or observed.live_dtype_profile != evidence.live_dtype_profile
        or observed.auxiliary_parameter_count
        != evidence.auxiliary_parameter_count
        or observed.auxiliary_parameter_bytes != evidence.auxiliary_parameter_bytes
    ):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "FSDP live precision evidence no longer matches the loaded model. Use "
            "a resident topology, or reload the model so the scan runs again."
        ))
    if fsdp_precision_profile_is_admitted(
        evidence.live_dtype_profile,
        evidence.auxiliary_parameter_count,
        evidence.auxiliary_parameter_bytes,
        evidence.checkpoint_kind,
    ):
        return
    if evidence.live_dtype_profile in (
            ALL_BF16_PROFILE, ALL_FP16_PROFILE,
            "uniform_or_quantized_fp8", "uniform_or_quantized_int8"):
        if evidence.auxiliary_parameter_count or evidence.auxiliary_parameter_bytes:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "FSDP uniform-precision evidence carries unexpected auxiliary "
                "parameters. Use a resident topology for this file."
            ))
    if evidence.live_dtype_profile == WAN_FP32_INGRESS_PROFILE:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "FSDP admits the Wan fp32-ingress profile on a bf16 checkpoint only "
            f"(got a {evidence.checkpoint_kind} checkpoint with "
            f"{evidence.auxiliary_parameter_count} ingress parameters, "
            f"{evidence.auxiliary_parameter_bytes} bytes). Use a resident "
            "topology for this file, or a bf16 checkpoint."
        ))
    if evidence.live_dtype_profile == FP32_ISLANDS_PROFILE:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "FSDP admits fp32 islands inside a bf16 core only, at most 5% of the "
            f"checkpoint bytes (got a {evidence.checkpoint_kind} checkpoint with "
            f"{evidence.auxiliary_parameter_count} island parameters, "
            f"{evidence.auxiliary_parameter_bytes} bytes). Use a resident "
            "topology for this file, or a bf16 checkpoint."
        ))
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        "FSDP live precision profile is not an audited full-precision core "
        f"layout (got {evidence.live_dtype_profile!r}). Use a bf16 checkpoint, "
        "or a resident topology for this file."
    ))


def refuse_fsdp_live_cast(weight_dtype: str) -> None:
    """A live cast to a quantized kind has no file-backed bytes to shard."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"FSDP shards quantized files as stored; weight_dtype={weight_dtype} would "
        "cast this checkpoint at load time, and a cast has no file-backed bytes "
        "to verify the shards against. Use weight_dtype=default with "
        "the file's own precision, or use a checkpoint stored in that format."))


def validate_fsdp_launch_loras(lora_stack, lora_low_rss: bool | None = None) -> None:
    """Admit a LoRA stack on a sharded model only under lora_low_rss.

    ``None`` means the caller cannot see the worker policy (the driver's LoRA
    loader); the worker's own check is the authority then.
    """
    if lora_stack and lora_low_rss is False:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "LoRA on FSDP-wrapped models needs lora_low_rss on: the sharded bake "
            "reads pristine weights from the checkpoint and never uses comfy's "
            "backup (actor/fsdp_lora.py). Use lora_low_rss=on (Init node or "
            "cluster.toml worker_args), or use a resident topology for this stack."))


def validate_fsdp_explicit_bf16_source(path: str, weight_dtype: str) -> None:
    """Refuse an explicit bf16 cast of a non-safetensors file: no header proves its precision."""
    if (
        weight_dtype == "bf16"
        and Path(path).suffix.lower() not in {".safetensors", ".sft"}
    ):
        validate_fsdp_launch_quant("unknown")


def apply_fsdp_capacity_mode(
    patcher,
    quant_kind: str,
    lora_stack,
    precision_evidence=None,
    prefetch_depth: int = 1,
    lora_low_rss: bool | None = None,
) -> None:
    """Shard the diffusion model parameters across the world group in place.

    ``prefetch_depth`` 1 keeps FSDP2's default one-ahead all-gather overlap;
    2 or more names that many blocks ahead through explicit forward prefetch,
    which trades reserved memory for speed (worker_args ``fsdp_prefetch_depth``).
    """
    validate_fsdp_launch_quant(quant_kind)
    validate_fsdp_launch_loras(lora_stack, lora_low_rss=lora_low_rss)

    from torch.distributed.fsdp import fully_shard

    dm = patcher.model.diffusion_model

    # The worker has bound the adapter and prepared any compile. Rescan the
    # live patcher here, right before the first parameter mutation, so a
    # replaced model or a dtype change cannot reach sharding on stale detector
    # evidence.
    validate_fsdp_live_precision(patcher, quant_kind, precision_evidence)
    # The lever check above passed; this refuses a checkpoint property no LoRA
    # bake admits, whatever lora_low_rss says.
    if lora_stack:
        from .fsdp_lora_admission import refuse_unless_fsdp_lora_admits

        refuse_unless_fsdp_lora_admits(precision_evidence)

    for p in dm.parameters():
        p.requires_grad = False

    wrapped_quant = 0
    if quant_kind in FSDP_ADMITTED_QUANT_KINDS:
        # Quantized bytes shard as stored: every QuantizedTensor parameter
        # becomes a wrapper FSDP2 can chunk, gather, and rebuild (fsdp_quant.py).
        from .fsdp_quant import wrap_quantized_parameters

        wrapped_quant = wrap_quantized_parameters(dm)

    # Replicated, never sharded: fp32 islands (the live scan above admitted
    # them under a profile), the modules a family runs off the denoise path
    # (fsdp_islands.py), and every zero-dimensional parameter. FSDP2 refuses a
    # scalar outright: it has no dim 0 to chunk, and replicating it costs
    # only a few bytes (for example, PiD's `log_alpha`). Build this set after
    # the quantized wrappers so it holds the objects the wrap will see.
    islands = island_parameters(dm)
    auxiliary = auxiliary_module_parameters(dm, patcher.model)
    scalars = {p for p in dm.parameters() if p.ndim == 0}
    ignored = islands | auxiliary | scalars
    wrap_kwargs: dict[str, Any] = {"ignored_params": ignored} if ignored else {}

    trim_pool = _allocator_pool_trim()
    sharded_lists = 0
    wrapped: list = []
    # Every host row reaches the device through one pinned bounce buffer, so
    # the driver never pins the checkpoint's own pages (fsdp_shard_build.py).
    from .fsdp_shard_build import DeviceCopier, build_shards, stage_on_device

    copier = DeviceCopier()

    def wrap_blocks() -> None:
        nonlocal sharded_lists
        for attr in _BLOCK_LIST_ATTRS:
            block_list = getattr(dm, attr, None)
            if block_list is None:
                continue
            for block in block_list:
                stage_on_device(block, copier, ignored=ignored)
                fully_shard(block, **wrap_kwargs)
                wrapped.append(block)
                _log_wrap_memory(len(wrapped))
            sharded_lists += 1
        if sharded_lists == 0:
            raise UnsupportedModelError(
                f"FSDP: no shardable block lists found on {type(dm).__name__} "
                f"(looked for {_BLOCK_LIST_ATTRS})."
            )
        # FSDP2's default keeps the root group unsharded after the first
        # forward, a training heuristic: backward would re-gather it. This path
        # never runs backward, so without the flag the text ingress, final
        # layer, modulation stacks and timestep embedding stay full size on
        # every rank: about 1 GiB per rank over a 16.2 GiB shard (32.5 GiB bf16
        # HunyuanImage). Blocks keep the default. The gain is idle
        # capacity, not render peak: the root group re-gathers each forward, at
        # one more exposed all-gather per step.
        stage_on_device(dm, copier, ignored=ignored)
        fully_shard(dm, reshard_after_forward=True, **wrap_kwargs)
        wrapped.append(dm)

    # Wrap on meta, then move only this rank's rows (fsdp_shard_build.py), so
    # the build puts no full-size block on the device. Quantized checkpoints
    # take the direct wrap: their QuantizedTensor wrappers carry FSDP2's own
    # gather hooks (fsdp_quant.py) and were proven on that path, and the
    # streaming build slices plain tensors only, so a quantized file pays the
    # full-size transient.
    materialized = 0
    try:
        if wrapped_quant:
            wrap_blocks()
        else:
            materialized = build_shards(dm, wrap_blocks, ignored, copier=copier)
    finally:
        # A build that raises leaves its frame alive in the traceback, and the
        # capacity refusal is the one path where the pinned bounce buffer must
        # not outlive the build.
        copier.close()
    gc.collect()
    if trim_pool is not None:
        trim_pool()
    _log_wrap_memory(len(wrapped))

    # Depth 1 leaves FSDP2's explicit prefetch list empty, torch's default
    # (tests/test_fsdp_capacity.py checks it). Depth 2 or more names that many
    # blocks ahead, so torch issues those all-gathers earlier from the CPU and
    # holds several groups at once. The world-group and Ulysses collectives
    # pair correctly because worker setup sets NCCL_LAUNCH_ORDER_IMPLICIT
    # (actor/worker_env.py), not because of this list.
    depth = max(int(prefetch_depth), 1)
    blocks = wrapped[:-1]  # the root is last and prefetches nothing
    for index, module in enumerate(wrapped):
        if not hasattr(module, "set_modules_to_forward_prefetch"):
            continue
        ahead = blocks[index + 1:index + 1 + depth] if depth > 1 and module is not dm else []
        module.set_modules_to_forward_prefetch(ahead)
    if depth > 1:
        log.info("FSDP forward prefetch: %d block(s) ahead (worker_args fsdp_prefetch_depth)",
                 depth)

    # Update both ComfyUI size estimates: the patcher's cached size and the
    # per-module sum used during loading. DTensor.nbytes reports global bytes;
    # leaving either estimate unchanged can force partial loading of a fitting
    # local shard.
    from ..actor import comfy_shard_size
    from .fsdp_quant import local_nbytes

    comfy_shard_size.install()
    local_bytes = 0
    for p in dm.parameters():
        local_bytes += local_nbytes(p)
    if hasattr(patcher, "size"):
        patcher.size = local_bytes

    import torch.distributed as dist

    world = dist.get_world_size()
    island_bytes = sum(p.numel() * p.element_size() for p in islands)
    # The auxiliary bytes are the ones a rank holds whole where a shard would
    # have been 1/world of them, so the line states them beside the count.
    auxiliary_bytes = sum(p.numel() * p.element_size() for p in auxiliary)
    log.info(
        "FSDP capacity mode: sharded %s across %d ranks, local weights %.1f GiB "
        "(%d parameters materialized from host rows, %d wraps; %d fp32 island "
        "parameters replicated, %.3f GiB; %d auxiliary-module parameters "
        "ignored, %.3f GiB; %d zero-dimensional parameters replicated; %d "
        "quantized parameters sharded as stored bytes)",
        type(dm).__name__, world, local_bytes / 2**30, materialized, len(wrapped),
        len(islands), island_bytes / 2**30, len(auxiliary),
        auxiliary_bytes / 2**30, len(scalars), wrapped_quant,
    )
    # Publish only after every fully_shard call, the local bookkeeping and the
    # distributed-state read succeed. The model store reads _dgxm_fsdp; the
    # gate's reload attestation (actor/gate_fsdp_cycle.py) reads
    # _dgxm_fsdp_ready.
    patcher._dgxm_fsdp = True
    patcher._dgxm_fsdp_ready = True
    # Offload stays on the compute device, as slab's does: a hot-swap's new
    # patch uuid makes comfy unpatch to the offload device, a full host copy of
    # every shard that frees nothing on unified memory.
    if getattr(patcher, "load_device", None) is not None:
        patcher.offload_device = patcher.load_device

def fsdp_load_capacity_check(
    path: str, unet_name: str, model_options: dict, world: int | None = None,
    checkpoint_kind: str | None = None,
) -> None:
    """Refuse an FSDP load whose kind-aware transient cannot fit."""
    if model_options.get("dtype") is not None or not gpu_is_integrated():
        return
    avail = mem_available_bytes()
    if avail is None:
        return
    size = os.path.getsize(path) if os.path.exists(path) else 0
    required = fsdp_required_bytes(size, world, checkpoint_kind)
    if required <= avail:
        return
    factor = fsdp_materialize_factor(world, checkpoint_kind)
    if checkpoint_kind in FSDP_ADMITTED_QUANT_KINDS:
        load_kind = "the direct wrapper keeps the full checkpoint live"
    elif checkpoint_kind in {"bf16", "fp16"}:
        load_kind = "the streaming build commits this rank's shards plus one block"
    else:
        load_kind = "the checkpoint kind is unproven, so the conservative price keeps the full checkpoint live"
    raise StockLoadCapacityError(refusal(
        RefusalClass.CAPACITY,
        f"the FSDP launch cannot load {unet_name}: {load_kind}, so it needs "
        f"near {factor:.2f}x the {size / 2**30:.1f} GiB checkpoint, or "
        f"{required / 2**30:.1f} GiB including the "
        f"{ABSOLUTE_HOST_FLOOR_BYTES / 2**30:.1f} GiB host floor, but only {avail / 2**30:.1f} GiB of "
        "unified memory is available, so the load would be killed out of "
        "memory before any refusal could classify it. Use a smaller checkpoint, or free more "
        "unified memory; a block-scaled quantized file (mxfp8, nvfp4) needs a resident topology. "
        "A supported fp8 or int8 file helps only by being smaller: its shards build on the direct path, which holds the whole file at once.",
        guard="stock_load_preflight", waivable=False,
    ))
