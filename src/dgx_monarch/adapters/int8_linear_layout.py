"""Preserve TensorWise INT8 and NVFP4 linear semantics across Comfy residency modes."""
from __future__ import annotations

import functools
from typing import Any

_INT8_LAYOUT = "TensorWiseINT8Layout"
_NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
_MARK = "_dgxm_tensorwise_int8_layout_forward"


def _is_affected_weight(weight: Any) -> bool:
    """Whether Comfy will dispatch this effective weight through TensorWise INT8.

    Reads the *argument* supplied to ``_forward``, not the module's stored
    parameter: Comfy hands a dequantized tensor here on LoRA and
    weight-function paths, which must keep their native behavior.
    """
    if getattr(weight, "_layout_cls", None) != _INT8_LAYOUT:
        return False
    return not bool(getattr(getattr(weight, "_params", None), "convrot", False))


def _is_nvfp4_pair(input: Any, weight: Any) -> bool:
    return (getattr(input, "_layout_cls", None) == _NVFP4_LAYOUT
            and getattr(weight, "_layout_cls", None) == _NVFP4_LAYOUT)


def _wrap(module: Any) -> bool:
    # Model classes also expose `_forward`, with model arguments such as
    # (x, timestep, context). Only a loaded Comfy mixed-precision Linear has
    # both of these stable format descriptors.
    quant_format = getattr(module, "quant_format", None)
    layout_type = getattr(module, "layout_type", None)
    eligible = ((quant_format == "int8_tensorwise" and layout_type == _INT8_LAYOUT)
                or (quant_format == "nvfp4" and layout_type == _NVFP4_LAYOUT))
    if not eligible or getattr(module, _MARK, False) or not callable(getattr(module, "_forward", None)):
        return False
    original = module._forward

    @functools.wraps(original)
    def normalized(input, weight, bias):
        if _is_nvfp4_pair(input, weight):
            # ComfyUI has native NVFP4 linear/mm dispatch but
            # no addmm handler. Call its existing linear dispatcher before
            # no-grad F.linear can lower to dequantized addmm.
            import torch
            return type(weight).__torch_dispatch__(
                torch.ops.aten.linear.default, (type(input), type(weight)),
                (input, weight, bias), {})
        # ComfyUI's QuantizedTensor F.linear route decomposes
        # differently for a non-contiguous activation outside inference mode.
        # TensorWise INT8 expects logical rows, so materialize them once before
        # dispatch. Other layouts, ConvRot and dequantized weights keep Comfy's route.
        if _is_affected_weight(weight) and not input.is_contiguous():
            input = input.contiguous()
        return original(input, weight, bias)

    module._forward = normalized
    setattr(module, _MARK, True)
    return True


def install(diffusion_model: Any) -> int:
    """Wrap existing TensorWise-INT8/NVFP4 mixed-precision linears once; return count.

    The wrapper belongs to the module object. ModelPatcher clones share that
    model and do not stack wrappers; a fresh model receives a fresh wrapper.
    """
    walk = getattr(diffusion_model, "modules", None)
    if not callable(walk):
        return 0
    return sum(_wrap(module) for module in walk())
