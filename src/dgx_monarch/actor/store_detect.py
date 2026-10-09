"""Checkpoint and live-precision detection for ModelStore."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..loader_options import weight_dtype_from_options
from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call
from .fsdp_checkpoint_pin import (
    PinnedFsdpCheckpoint,
    assert_fsdp_checkpoint_identity,
    fsdp_reuse_matches_proof,
    pin_fsdp_checkpoint,
)

__all__ = (
    "PinnedFsdpCheckpoint",
    "assert_fsdp_checkpoint_identity",
    "fsdp_reuse_matches_proof",
    "pin_fsdp_checkpoint",
)

log = get_logger(__name__)


@dataclass(frozen=True)
class LivePrecisionEvidence:
    """Sanitized detector-owned evidence for one materialized model."""

    quant_kind: str
    live_dtype_profile: str
    auxiliary_parameter_count: int = 0
    auxiliary_parameter_bytes: int = 0
    checkpoint_kind: str | None = None
    # Whether a quantized parameter carries a comfy-kitchen QuantizedTensor
    # wrapper (scaled fp8, fp8mixed, int8), as opposed to a plain-dtype tensor
    # at the same quant_kind (Qwen-Image's literal float8_e4m3fn weights carry
    # no wrapper and keep rendering). Read by adapters.fsdp_lora_admission.
    quantized_shards: bool = False

    def bind_checkpoint(
        self,
        checkpoint_kind: str | None,
        effective_kind: str,
    ) -> LivePrecisionEvidence:
        return LivePrecisionEvidence(
            quant_kind=effective_kind,
            live_dtype_profile=self.live_dtype_profile,
            auxiliary_parameter_count=self.auxiliary_parameter_count,
            auxiliary_parameter_bytes=self.auxiliary_parameter_bytes,
            checkpoint_kind=checkpoint_kind,
            quantized_shards=self.quantized_shards,
        )

    def public(self) -> dict[str, object]:
        return {
            "live_dtype_profile": self.live_dtype_profile,
            "auxiliary_parameter_count": self.auxiliary_parameter_count,
            "auxiliary_parameter_bytes": self.auxiliary_parameter_bytes,
            "checkpoint_precision": self.checkpoint_kind,
        }


def _to_comfy_model_options(options: dict | None) -> dict:
    """Map the loader's widget options to comfy load_diffusion_model options.

    ``bf16`` is an explicit full-precision request for capacity-sensitive
    workflows.
    """
    import torch

    out: dict[str, Any] = {}
    weight_dtype = weight_dtype_from_options(options)
    if weight_dtype == "bf16":
        out["dtype"] = torch.bfloat16
    elif weight_dtype == "fp8_e4m3fn":
        out["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        out["dtype"] = torch.float8_e4m3fn
        out["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        out["dtype"] = torch.float8_e5m2
    return out


def _quant_kind_from_options(options: dict | None) -> str:
    weight_dtype = weight_dtype_from_options(options)
    if weight_dtype.startswith("fp8"):
        return "fp8"
    return "bf16"


def validate_fsdp_request_checkpoint(
    path: str,
    options: dict | None,
    lora_stack,
) -> Any:
    """Prove one worker-local FSDP request; return the proof bound to its inode."""
    from ..adapters.detect import sniff_fsdp_launch_quant_proof
    from ..adapters.fsdp import (
        validate_fsdp_explicit_bf16_source,
        validate_fsdp_launch_loras,
        validate_fsdp_launch_quant,
    )

    validate_fsdp_launch_loras(lora_stack)
    weight_dtype = weight_dtype_from_options(options)
    validate_fsdp_explicit_bf16_source(path, weight_dtype)
    option_kind = _quant_kind_from_options(options)
    if option_kind != "bf16":
        from ..adapters.fsdp import refuse_fsdp_live_cast

        refuse_fsdp_live_cast(weight_dtype)
    proof = sniff_fsdp_launch_quant_proof(path)
    proven_kind = "unknown" if proof.quant_kind is None else proof.quant_kind
    validate_fsdp_launch_quant(proven_kind)
    if proof.file_identity is None:
        from ..mesh_safety import ArtifactBindingError

        raise ArtifactBindingError(
            "worker-local FSDP checkpoint proof has no exact file identity"
        )
    return proof


# The one live shape whose FP32 reading is a second look at the header's own
# bytes rather than independent evidence: a BF16 core carrying auxiliary FP32
# tensors. A copying load takes those in as BF16 and an assigning load leaves
# them FP32, so the reported kind would otherwise change with the residency.
BF16_CORE_FP32_AUXILIARY_PROFILE = "bf16_core_with_fp32_auxiliaries"


def _detect_checkpoint_kind(path: str, option_kind: str) -> str | None:
    """Classify safetensors storage before Comfy may cast it at load time.

    Non-safetensors formats retain the loaded-model detector as their source of
    truth. A safetensors header that this process cannot classify is ``unknown``
    rather than ``bf16`` so the literal-BF16 FSDP grant fails closed; stock
    Comfy loading remains available for non-FSDP paths.
    """
    if option_kind != "bf16":
        return option_kind
    if not path.lower().endswith((".safetensors", ".sft")):
        return None

    from ..adapters.detect import CheckpointSniffError, sniff_checkpoint

    try:
        _family, kind = sniff_checkpoint(path)
    except CheckpointSniffError as exc:
        safe_call(
            log.warning,
            "could not classify safetensors precision for %s (%s); reporting as unknown",
            Path(path).name,
            failure_summary(exc),
        )
        return "unknown"
    return kind


def file_dtypes_preserved(patcher) -> bool:
    """Whether Comfy made the checkpoint's own tensors this model's parameters.

    ``load_model_weights(..., assign=model_patcher.is_dynamic())`` in
    comfy/sd.py is Comfy's switch: a dynamic patcher assigns the loaded tensors
    in place of the parameters the model built at its compute dtype, while a
    classic one copies into those parameters and takes their dtype. Under the
    assigning branch a live dtype is a second reading of the file, not evidence
    about how the model computes. A BF16 checkpoint that carries auxiliary FP32
    tensors is one shape the branches read differently, and the only one that
    defers to the header (``BF16_CORE_FP32_AUXILIARY_PROFILE``).
    """
    is_dynamic = getattr(patcher, "is_dynamic", None)
    if not callable(is_dynamic):
        return False
    try:
        return bool(is_dynamic())
    except Exception as exc:  # a detector must never fail a load
        safe_call(
            log.warning,
            "could not read Comfy's dynamic-patcher flag (%s); reading live dtypes "
            "as a copying load",
            failure_summary(exc),
        )
        return False


def _combine_checkpoint_and_live_kinds(
    checkpoint_kind: str | None,
    live_kind: str,
    *,
    live_dtypes_are_the_file: bool = False,
    live_profile: str | None = None,
) -> str:
    """Resolve storage/live observations without weakening either one.

    Runtime quant metadata is the most precise source. Header quant markers
    come next, then any non-BF16 full-precision observation. An unreadable or
    unsupported header never becomes a BF16 grant because Comfy materialized
    BF16 tensors.

    ``live_dtypes_are_the_file`` says the load assigned the file's tensors
    (``file_dtypes_preserved``) and ``live_profile`` says what that reading
    saw. Only ``BF16_CORE_FP32_AUXILIARY_PROFILE`` defers to the header, which
    keeps one checkpoint's reported precision the same under every residency.

    Nothing else defers: the header's BF16 verdict is a compatibility default
    (``adapters.detect.detect_quant_from_header`` returns it for any header
    with no FP8, INT8 or F16 marker, an all-FP32 one included). It holds only
    for an export whose live reading shows a BF16 core, so a checkpoint with
    no BF16 core keeps its FP32 reading here.
    """
    if _is_quantized_kind(live_kind) and live_kind != "unknown":
        return live_kind
    if checkpoint_kind is not None and _is_quantized_kind(checkpoint_kind) \
            and checkpoint_kind != "unknown":
        return checkpoint_kind
    header_kind = (checkpoint_kind if live_dtypes_are_the_file
                   and checkpoint_kind is not None
                   and checkpoint_kind != "unknown"
                   and live_profile == BF16_CORE_FP32_AUXILIARY_PROFILE else None)
    for full_precision_kind in ("fp16", "fp32"):
        if checkpoint_kind == full_precision_kind:
            return full_precision_kind
        if live_kind == full_precision_kind and header_kind is None:
            return full_precision_kind
    if header_kind is not None:
        return header_kind
    if checkpoint_kind == "unknown" or live_kind == "unknown":
        return "unknown"
    return live_kind


def _detect_live_precision(patcher, option_kind: str) -> LivePrecisionEvidence:
    """Detect quantized checkpoints loaded through default options.

    ComfyUI exposes converted quantization through per-layer metadata and the
    model config. Order: a layer's ``quant_format`` wins, then an fp8 or int8
    parameter dtype, then the config's ``quant_config`` or
    ``custom_operations``; only then do full-precision dtypes decide. Keep
    ``fp16`` distinct: the FSDP launch gate admits it only as a uniform core.
    """
    def evidence(
        kind: str, profile: str | None = None, *, quantized_shards: bool = False,
    ) -> LivePrecisionEvidence:
        return LivePrecisionEvidence(
            kind, profile or f"uniform_or_quantized_{kind}",
            quantized_shards=quantized_shards)

    if option_kind != "bf16":
        return evidence(option_kind)
    import torch

    model = getattr(patcher, "model", None)
    model_config = getattr(model, "model_config", None)

    diffusion_model = getattr(model, "diffusion_model", None)
    saw_bf16 = False
    saw_fp16 = False
    fp32_parameters: dict[str, Any] = {}
    saw_unsupported_dtype = False
    total_parameter_bytes = 0
    if diffusion_model is not None:
        modules = getattr(diffusion_model, "modules", None)
        named_parameters = getattr(diffusion_model, "named_parameters", None)
        if not callable(modules) or not callable(named_parameters):
            return evidence("unknown")
        # Operation metadata is authoritative because wrapper dtypes can still
        # report the original BF16 storage kind.
        _FMT_KIND = {"float8_e4m3fn": "fp8", "float8_e5m2": "fp8",
                     "int8_tensorwise": "int8", "mxfp8": "mxfp8",
                     "nvfp4": "nvfp4"}
        for m in modules():
            fmt = getattr(m, "quant_format", None)
            if fmt:
                kind = _FMT_KIND.get(fmt)
                if kind is None:
                    log.warning("unknown quant_format %r; reporting as fp8", fmt)
                    kind = "fp8"
                return evidence(kind, quantized_shards=True)
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for name, p in named_parameters():
            total_parameter_bytes += p.numel() * p.element_size()
            if p.dtype in fp8_dtypes:
                # A plain fp8 tensor with no quant_format wrapper (Qwen-Image's
                # literal float8_e4m3fn weights): quantized_shards stays False.
                return evidence("fp8")
            if p.dtype == torch.int8:
                # No known checkpoint stores a bare, un-wrapped int8 tensor;
                # every int8 file seen here carries a comfy-kitchen scale
                # sidecar, so this is conservatively quantized too.
                return evidence("int8", quantized_shards=True)
            if p.dtype == torch.bfloat16:
                saw_bf16 = True
            elif p.dtype == torch.float16:
                # A later quantized parameter is more specific than auxiliary F16.
                saw_fp16 = True
            elif p.dtype == torch.float32:
                fp32_parameters[name] = p
            else:
                saw_unsupported_dtype = True

    if model_config is not None:
        if getattr(model_config, "quant_config", None):
            return evidence("fp8", quantized_shards=True)  # metadata is authoritative
        custom_ops = getattr(model_config, "custom_operations", None)
        if custom_ops is not None:
            ops_name = str(custom_ops).lower()
            if "int8" in ops_name:
                return evidence("int8", quantized_shards=True)
            if "fp8" in ops_name:
                return evidence("fp8", quantized_shards=True)
            # Unknown custom operations prevent a BF16 inference.
            return evidence("unknown", "unrecognized_custom_operations")
    if saw_fp16:
        if (
            saw_bf16 or fp32_parameters or saw_unsupported_dtype
            or total_parameter_bytes <= 0
        ):
            return evidence("fp16", "mixed_or_uniform_fp16")
        from ..adapters.fsdp_islands import ALL_FP16_PROFILE

        return evidence("fp16", ALL_FP16_PROFILE)
    if fp32_parameters and saw_bf16 and not saw_fp16 and not saw_unsupported_dtype:
        from ..adapters.fsdp import classify_audited_live_precision

        audited = classify_audited_live_precision(
            model,
            diffusion_model,
            fp32_parameters,
            total_parameter_bytes,
        )
        if audited is not None:
            profile, auxiliary_count, auxiliary_bytes = audited
            log.info(
                "live precision: BF16 core with audited auxiliary profile %s "
                "(%d auxiliary bytes)", profile, auxiliary_bytes,
            )
            return LivePrecisionEvidence(
                "bf16",
                profile,
                auxiliary_parameter_count=auxiliary_count,
                auxiliary_parameter_bytes=auxiliary_bytes,
            )
    if fp32_parameters:
        # Only a BF16 core with FP32 auxiliaries may defer to the header
        # (BF16_CORE_FP32_AUXILIARY_PROFILE); a model with no BF16 core keeps
        # its FP32 reading on every rung. Count the auxiliaries either way: once
        # the reported kind stops naming them, the evidence row is the only
        # place they stay visible. No profile outside the audited set is
        # admissible, so carrying the numbers cannot widen any grant.
        profile = (BF16_CORE_FP32_AUXILIARY_PROFILE if saw_bf16
                   else "mixed_or_uniform_fp32")
        return LivePrecisionEvidence(
            "fp32",
            profile,
            auxiliary_parameter_count=len(fp32_parameters),
            auxiliary_parameter_bytes=sum(
                p.numel() * p.element_size() for p in fp32_parameters.values()),
        )
    if saw_unsupported_dtype or not saw_bf16:
        return evidence("unknown")
    return evidence("bf16", "all_bf16")


def _detect_quant_kind(patcher, option_kind: str) -> str:
    """Compatibility reader for callers that need only the coarse kind."""
    return _detect_live_precision(patcher, option_kind).quant_kind


def _is_quantized_kind(kind: str) -> bool:
    """Whether ``kind`` needs quantized-weight hot-swap invalidation.

    Unknown kinds count as quantized. FP16 and FP32 must stay excluded: neither
    is quantized, and neither should inherit the measured fp8/int8
    conversion-spike workaround.
    """
    return kind not in {"bf16", "fp16", "fp32"}


def _detect_family(patcher) -> str:
    from ..adapters import detect_family

    return detect_family(patcher.model)


def detected_family_or_none(patcher) -> str | None:
    """What detection would have said, for a log line beside a forced family.

    Detection raises on the derivative subclasses a family override exists to
    admit, so this reports and never decides. Never let it fail a load: the
    forced bind governs, and ``adapter_for`` checks it separately.
    """
    try:
        return _detect_family(patcher)
    except Exception:
        return None
