"""Reject checkpoint formats the FSDP LoRA bake cannot restore exactly.

FSDP accepts several storage kinds, but LoRA additionally requires low-RSS
mode and the full-then-slice bake in ``actor.fsdp_lora``. That bake rejects
comfy-kitchen QuantizedTensor wrappers. Plain FP8-dtype weights without a
wrapper, such as Qwen-Image's float8_e4m3fn weights, remain eligible.

The driver checks header evidence before loading; the worker checks live
evidence before baking. Both raise the same typed refusal regardless of
``auto_gate`` or ``lora_low_rss`` settings.

FP32 islands need a separate per-key check. A header or coarse precision
profile cannot tell whether ComfyUI preserves an island's file dtype or
casts it to the core dtype. For example, Krea2 RAW's island is cast to bf16,
whereas ChromaRadiance's nerf-embedder islands remain fp32. The bake's
``_file_tensor`` check refuses live/file mismatches.
"""
from __future__ import annotations

from typing import NoReturn

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

_CAUSES = {
    "quantized_shards": (
        "this checkpoint's weights are comfy-kitchen quantized tensors "
        "(scaled fp8, fp8mixed, or int8, with or without convrot); "
        "actor/fsdp_lora.py refuses a quantized key outright, whatever "
        "lora_low_rss is set to"
    ),
}


def refuse_fsdp_lora_checkpoint_property(property_name: str) -> NoReturn:
    """Typed class P refusal for a checkpoint property no LoRA bake admits.

    Raised before any load reaches ``actor/fsdp_lora.py``'s in-bake checks, on
    every ``auto_gate`` setting: this is a property of the checkpoint's own
    bytes, never something ``lora_low_rss`` can fix.
    """
    cause = _CAUSES.get(property_name)
    if cause is None:
        raise ValueError(f"unknown FSDP LoRA admission property {property_name!r}")
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        "LoRA on an FSDP-wrapped model is not admitted for this checkpoint: "
        f"{cause}. Use a resident topology for this file, or a bf16, fp16, "
        "or plain-dtype fp8 checkpoint under FSDP."))


def live_admission_property(evidence: object) -> str | None:
    """Name the checkpoint property from already-detected live evidence.

    ``evidence`` is a ``store_detect.LivePrecisionEvidence`` (duck-typed here
    to keep this adapter importable without the actor package): its
    ``quantized_shards`` flag is the live QuantizedTensor evidence
    ``actor.store_detect._detect_live_precision`` already computed. Returns
    ``None`` when it is unset, which is also the answer for every fp32-island
    checkpoint (the module docstring says why).
    """
    if getattr(evidence, "quantized_shards", False):
        return "quantized_shards"
    return None


def refuse_unless_fsdp_lora_admits(evidence: object) -> None:
    """Worker-side backstop: refuse before the bake, from live evidence alone.

    The driver-side header preflight (``adapters/detect.py``
    ``fsdp_lora_admission_property``, wired into the LoRA loader node) is the
    first thing an operator meets for an explicit FSDP topology. Every fresh
    FSDP load with a LoRA stack also crosses this call
    (``apply_fsdp_capacity_mode``, before the block-list wrap and before any
    LoRA bake), so a caller that bypasses the node graph still meets the
    refusal before the bake for a comfy-kitchen quantized checkpoint.
    """
    property_name = live_admission_property(evidence)
    if property_name is not None:
        refuse_fsdp_lora_checkpoint_property(property_name)
