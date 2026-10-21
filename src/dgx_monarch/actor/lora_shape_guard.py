"""Refuse a LoRA patch that changes a weight's shape, before any bake (#127).

ComfyUI can apply such a patch (a "set" or padded "diff" patch of another shape,
for example the MOSS-Image color and depth control adapter, which widens
``video_embedder`` from 128 to 384 input channels). dgx-monarch's bake, un-bake,
slab and FSDP paths keep the checkpoint's shapes, and the bake's
``comfy.utils.copy_to_param`` would fail untyped on the first such key. The test
is ComfyUI's own predicate (``calculate_shape(...) != weight.shape``), and a
DTensor, slab view or meta parameter reports its logical shape, so the guard
holds under every residency mode.
"""
from __future__ import annotations

from typing import Any

from ..refusal import RefusalClass, refusal


def assert_patch_shapes_fit(active_patcher: Any) -> None:
    """Raise class P when any patched key would leave its file shape."""
    patches = getattr(active_patcher, "patches", None)
    if not patches:
        return
    import comfy.lora
    import comfy.utils

    for key in patches:
        try:
            weight = comfy.utils.get_attr(active_patcher.model, key)
        except AttributeError:
            continue  # ComfyUI's patcher skips a key it cannot resolve as well
        shape = getattr(weight, "shape", None)
        if shape is None:
            continue
        patched = comfy.lora.calculate_shape(patches[key], weight, key)
        if patched == shape:
            continue
        from ..adapters.base import UnsupportedModelError

        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"a LoRA patch replaces {key} of shape {tuple(shape)} with one of shape "
            f"{tuple(patched)}, as a control adapter that widens the model's input does "
            "(the MOSS-Image color and depth control LoRA widens video_embedder from 128 "
            "to 384 input channels). dgx-monarch's bake, un-bake, slab and FSDP paths keep "
            "the checkpoint's shapes, so this release refuses it before any weight moves. "
            "Remove that LoRA from the DGX Monarch LoRA loader, or run it in plain ComfyUI "
            "on one GPU instead.",
            troubleshooting=127))
