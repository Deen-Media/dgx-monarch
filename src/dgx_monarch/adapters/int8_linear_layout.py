"""Keep ComfyUI's weight-only quantized linears on their kernels, and NVFP4 on its linear.

ComfyUI quantizes no activation for the formats whose ``quantize_input`` is
False (INT8 tensorwise with or without ConvRot, ConvRot W4A4, and the grouped
W4A8 and W6A8 formats). Its mixed-precision Linear then hands ``F.linear`` the
activation as it arrives, a 3-D one included. Under ``torch.inference_mode()``,
as in stock's prompt thread, that reaches the layout's ``aten.linear`` handler.
A Worker's sampling thread runs outside inference mode, where ``aten.linear``
decomposes: a 3-D activation whose leading strides chain folds into ``mm`` or
``addmm``, which the layouts handle with the same kernel, and any other one
expands the weight for ``bmm``. No layout handles ``expand``, so the weight is
dequantized and the linear runs unquantized with a different result. A
sharded forward can hand a linear such a view, so every 3-D or wider input to
these layouts is flattened to rows here and the output is viewed back.
"""
from __future__ import annotations

import functools
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

_NVFP4_LAYOUT = "TensorCoreNVFP4Layout"
_MARK = "_dgxm_tensorwise_int8_layout_forward"
# ComfyUI's quantization formats with ``quantize_input`` False, and the layout
# each one loads as. This module imports no ComfyUI, so the table is copied;
# the "weight-only linear dispatch" seam in tests/canary/comfy_seam_contracts.py
# holds it equal to ComfyUI's own QUANT_ALGOS.
WEIGHT_ONLY_FORMATS: Mapping[str, str] = MappingProxyType({
    "int8_tensorwise": "TensorWiseINT8Layout",
    "convrot_w4a4": "TensorCoreConvRotW4A4Layout",
    "asym_w4a8_int8": "AsymW4A8Int8Layout",
    "w6a8_int8": "AsymW4A8Int8Layout",
})
WEIGHT_ONLY_LAYOUTS = frozenset(WEIGHT_ONLY_FORMATS.values())


def _is_weight_only_quantized(weight: Any) -> bool:
    """Whether ComfyUI will dispatch this effective weight through a weight-only layout.

    Reads the *argument* supplied to ``_forward``, not the module's stored
    parameter: ComfyUI hands a dequantized tensor here on LoRA and
    weight-function paths, which must keep their native behavior.
    """
    return getattr(weight, "_layout_cls", None) in WEIGHT_ONLY_LAYOUTS


def _is_nvfp4_pair(input: Any, weight: Any) -> bool:
    return (getattr(input, "_layout_cls", None) == _NVFP4_LAYOUT
            and getattr(weight, "_layout_cls", None) == _NVFP4_LAYOUT)


def _wrap(module: Any) -> bool:
    # Model classes also expose `_forward`, with model arguments such as
    # (x, timestep, context). Only a loaded Comfy mixed-precision Linear has
    # both of these stable format descriptors.
    quant_format = getattr(module, "quant_format", None)
    layout_type = getattr(module, "layout_type", None)
    weight_only = WEIGHT_ONLY_FORMATS.get(quant_format) if isinstance(quant_format, str) else None
    eligible = ((weight_only is not None and layout_type == weight_only)
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
        if _is_weight_only_quantized(weight) and input.ndim > 2:
            # Contiguous 2-D rows always reach the layout's mm or addmm
            # handler; the module docstring says why a 3-D view may not. The
            # INT8 handlers copy a strided input to contiguous rows themselves,
            # so for them the copy here costs nothing more.
            out = original(input.reshape(-1, input.shape[-1]).contiguous(), weight, bias)
            return out.view(*input.shape[:-1], out.shape[-1])
        return original(input, weight, bias)

    module._forward = normalized
    setattr(module, _MARK, True)
    return True


def install(diffusion_model: Any) -> int:
    """Wrap existing weight-only quantized and NVFP4 mixed-precision linears once; return count.

    The wrapper belongs to the module object. ModelPatcher clones share that
    model and do not stack wrappers; a fresh model receives a fresh wrapper.
    """
    walk = getattr(diffusion_model, "modules", None)
    if not callable(walk):
        return 0
    return sum(_wrap(module) for module in walk())
