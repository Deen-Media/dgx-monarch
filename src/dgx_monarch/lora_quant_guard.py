"""Refuse a LoRA on the quantized files whose cluster bake renders wrong (#129).

One predicate and one refusal, read by the driver before dispatch
(nodes/model_patch_preflight.py and Fleet KSampler) and by every Worker's
model store before any LoRA build or bake (actor/model_store.py for a
resident base, actor/store_load.py for a fresh load). Each side reads facts
every rank holds alike, the family and the detected quantization kind, so
every rank refuses alike. docs/TROUBLESHOOTING.md #129 holds the measurement
and the remedy. This module imports no ComfyUI.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any

from .refusal import RefusalClass, refusal

# The detected quantization kinds that refuse any LoRA, by family. Widening
# the refusal to another family is one entry here, with its evidence in #129.
LORA_REFUSED_QUANTS: Mapping[str, frozenset[str]] = MappingProxyType({
    "moss_image": frozenset({"int8"}),
})


def lora_refused(family: str | None, quant_kind: str | None,
                 lora_stack: Sequence[Any] | None) -> bool:
    """Whether a ``family`` file of ``quant_kind`` refuses this LoRA stack."""
    return bool(lora_stack) and quant_kind in LORA_REFUSED_QUANTS.get(family or "", ())


def assert_lora_quant_supported(family: str | None, quant_kind: str | None,
                                lora_stack: Sequence[Any] | None) -> None:
    """Raise class P when ``lora_refused``; an empty stack never refuses."""
    if not lora_refused(family, quant_kind, lora_stack):
        return
    from .adapters.base import UnsupportedModelError

    names = ", ".join(str(entry.get("name") if isinstance(entry, Mapping) else entry)
                      for entry in lora_stack or ())
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family}: an {str(quant_kind).upper()} file with a LoRA ({names}) rendered far "
        "from stock ComfyUI on the cluster on 2026-10-21, and the cause is under "
        "investigation, so this release refuses the pair on every topology before any "
        "weight moves: each one bakes the LoRA the same way, mode=local on one GPU "
        "included. Render this graph in stock ComfyUI on one GPU instead, or remove the "
        "LoRA; a BF16 file with a LoRA is not refused.",
        troubleshooting=129))
