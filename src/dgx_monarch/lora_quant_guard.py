"""Refuse a LoRA on the quantized files whose LoRA render is wrong or unmeasured (#129).

Two grounds, by family and kind. A kind whose render with a baked LoRA was
measured far from stock refuses on that measurement (MOSS-Image INT8). A kind
of the same family that no render has measured with a LoRA refuses as
unmeasured, in words that claim no measurement (MOSS-Image ``w4a8`` and
``w4a4``): the family's one measured quantized kind rendered wrong, and
nothing shows these do better.

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
    "moss_image": frozenset({"int8", "w4a8", "w4a4"}),
})
# Of those, the kinds no render has measured with a LoRA on that family. They
# refuse because the family's measured quantized kind rendered wrong and
# nothing shows these do better, and their refusal says so: it states no
# measurement of its own. w4a8 is ComfyUI's grouped INT8 formats and w4a4 its
# ConvRot W4A4 (actor/store_detect.py); a Worker read a file of either as
# int8 or fp8 until it named them, so the int8 refusal used to reach some of
# them by accident.
LORA_UNMEASURED_QUANTS: Mapping[str, frozenset[str]] = MappingProxyType({
    "moss_image": frozenset({"w4a8", "w4a4"}),
})
# What works instead, the end of both refusals (docs/TROUBLESHOOTING.md #129).
WORKING_ALTERNATIVE = (
    "Render this graph in stock ComfyUI on one GPU instead, with its default DynamicVRAM "
    "loader, or remove the LoRA; a BF16 file with a LoRA is not refused.")


def lora_refused(family: str | None, quant_kind: str | None,
                 lora_stack: Sequence[Any] | None) -> bool:
    """Whether a ``family`` file of ``quant_kind`` refuses this LoRA stack."""
    return bool(lora_stack) and quant_kind in LORA_REFUSED_QUANTS.get(family or "", ())


def header_kind_is_open(family: str | None, quant_kind: str | None) -> bool:
    """Whether a header's ``int8`` leaves this family's refusal between two texts.

    A ConvRot W4A4 file's header reads ``int8`` as an INT8 file's does. Where
    the family refuses ``w4a4`` as unmeasured and ``int8`` on a measurement,
    the driver has to learn which file it holds before it describes it
    (adapters/layer_records.py ``name_int8_header_kind``).
    """
    return quant_kind == "int8" and "w4a4" in LORA_UNMEASURED_QUANTS.get(family or "", ())


def assert_lora_quant_supported(family: str | None, quant_kind: str | None,
                                lora_stack: Sequence[Any] | None, *,
                                int8_or_w4a4: bool = False) -> None:
    """Raise class P when ``lora_refused``; an empty stack never refuses.

    ``int8_or_w4a4`` is the driver's word that ``quant_kind`` is a header's
    ``int8`` and that the file's layer records could not say whether it is an
    INT8 or a ConvRot W4A4 file. The refusal then describes both, and states
    the measurement of the INT8 file only as that file's.
    """
    if not lora_refused(family, quant_kind, lora_stack):
        return
    from .adapters.base import UnsupportedModelError

    names = ", ".join(str(entry.get("name") if isinstance(entry, Mapping) else entry)
                      for entry in lora_stack or ())
    if int8_or_w4a4 and header_kind_is_open(family, quant_kind):
        why = (
            f"{family}: an INT8 or ConvRot W4A4 file with a LoRA ({names}) is refused. Its "
            "header reads int8, as both kinds do, and the file carries no layer record that "
            "says which it is. This family's INT8 file with a LoRA rendered far from stock "
            "ComfyUI on the cluster on 2026-10-21, and no render has measured a ConvRot W4A4 "
            "file with one, so this release refuses the pair on every topology before any "
            "weight moves, mode=local on one GPU included.")
    elif quant_kind in LORA_UNMEASURED_QUANTS.get(family or "", ()):
        why = (
            f"{family}: no render has measured a LoRA ({names}) on a {str(quant_kind).upper()} "
            "file of this family. Its INT8 file with a LoRA rendered far from stock ComfyUI on "
            "the cluster on 2026-10-21, and nothing shows this kind does better, so this "
            "release refuses the pair on every topology before any weight moves, mode=local on "
            "one GPU included, until a render measures it.")
    else:
        why = (
            f"{family}: an {str(quant_kind).upper()} file with a LoRA ({names}) rendered far "
            "from stock ComfyUI on the cluster on 2026-10-21. dgx-monarch baked the LoRA into "
            "the quantized weights as stock's classic loader does, and stock with DynamicVRAM "
            "off rendered the Kromoss INT8 file with its DMD2 LoRA as far from its own default. "
            "The Workers now bake stock's cast-time weight instead (#145), which no hardware run "
            "has measured on this file yet, so this release still refuses the pair on every "
            "topology before any weight moves, mode=local on one GPU included.")
    # One raise for the three texts, each ending in the same remedy, bound at module
    # level so the class rule in tests/test_refusal_classes.py reads it.
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS, why + " " + WORKING_ALTERNATIVE, troubleshooting=129))
