"""Bake a LoRA with the math stock ComfyUI's default loader runs at cast time.

Stock ComfyUI loads a model through DynamicVRAM by default, and DynamicVRAM
never merges a LoRA at load: ``ModelPatcherDynamic.load`` attaches a
``LowVramPatch`` to each patched weight, and ``comfy.ops`` runs it when the
layer casts its weight for a forward. That cast dequantizes (or casts) the
weight to the forward's compute dtype, runs ``calculate_weight`` in that
dtype, and requantizes a quantized weight with stochastic rounding seeded by
the module's name. It writes the result back on the first forward, so the
layer computes with that one weight from then on.

A Worker on stock residency loads through ComfyUI's classic patcher, whose
bake is the recipe stock uses only with DynamicVRAM off: the LoRA compute
dtype (fp16 on GB10), fp32 intermediates and a seed from the key with its
``.weight`` suffix. On an INT8 ConvRot file that is another draw of the
requantization noise, larger than many LoRAs' own delta; on a BF16 file it is
another rounding (docs/TROUBLESHOOTING.md #145).

``cast_time_weight`` returns stock's cast-time weight for one key, or
``None`` where stock bakes that key classically itself, or where stock's
cast does not settle on one weight (``_reason``); those keys keep the classic
bake. ``install`` routes ComfyUI's classic ``patch_weight_to_device``
through it in a Worker process: the sample-time load, the ``lora_low_rss``
merge, the clone hot swap and the ambient verify's reference all call that
method. The lazy swap and the FSDP bake call ``cast_time_weight`` directly. A
dynamic patcher (comfy-managed residency) keeps stock's own method. ComfyUI
is imported only inside functions.
"""
from __future__ import annotations

import collections
from typing import Any, cast

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)

# ModelPatcherDynamic.load force-loads a module this small through the classic
# bake instead of attaching a cast-time patch.
SMALL_MODULE_BYTES = 16 * 1024
_MARK = "_dgxm_cast_time"
_said: set[str] = set()  # process lifetime: one INFO line per outcome kind
# The record stock's patch_weight_to_device keeps for an unpatch.
_Backup = collections.namedtuple("_Backup", ["weight", "inplace_update"])


def _say(kind: str, msg: str, *args: Any) -> None:
    if kind not in _said:
        _said.add(kind)
        safe_call(log.info, msg, *args)


def _is_quantized(tensor: Any) -> bool:
    from comfy.quant_ops import QuantizedTensor

    return isinstance(tensor, QuantizedTensor)


def _module_bytes(module: Any) -> int:
    """``comfy.model_management.module_size`` in element counts.

    Counting ``numel() * element_size()`` gives a DTensor's whole logical size,
    so an FSDP-sharded module takes the same branch as its resident twin.
    """
    return sum(int(t.numel()) * int(t.element_size()) for t in module.state_dict().values())


def compute_dtype(model: Any, weight: Any) -> Any:
    """The dtype a patched layer's forward casts its weight to.

    Stock casts to the forward's input dtype, and a ComfyUI model feeds its
    network ``get_dtype_inference()``: the manual cast dtype, else the model's
    own. A bare module (tests, canaries) computes in its weight's dtype.
    """
    inference = getattr(model, "get_dtype_inference", None)
    if callable(inference):
        return inference()
    manual = getattr(model, "manual_cast_dtype", None)
    return manual if manual is not None else weight.dtype


def _reason(patcher: Any, key: str, module: Any, param: str, weight: Any) -> str | None:
    """Why stock does not cast-time patch ``key`` as one weight, or ``None``."""
    if param not in ("weight", "bias") or not hasattr(module, "comfy_cast_weights"):
        return "stock patches it at load (not a cast weight)"
    import comfy.lora
    import comfy.ops

    patches = patcher.patches[key]
    if key in getattr(patcher, "weight_wrapper_patches", {}):
        return "it carries a weight wrapper"
    if _module_bytes(module) <= SMALL_MODULE_BYTES and not getattr(patcher.model, "comfy_writable", False):
        return "stock force-loads a module this small"
    if comfy.lora.calculate_shape(patches, weight, key) != weight.shape:
        return "the LoRA resizes the weight"
    if _is_quantized(weight):
        if not (isinstance(module, comfy.ops.MixedPrecisionOp) and type(module).__name__ == "Linear"):
            return "a quantized weight outside a Linear"
        if (getattr(module, "layout_type", None) is None or getattr(module, "_full_precision_mm", False)
                or getattr(module, "comfy_force_cast_weights", False)):
            return "the layer runs a full-precision matmul, where stock's first cast differs from later ones"
        return None
    if weight.dtype != compute_dtype(patcher.model, weight):
        return "the weight is stored in another dtype than the forward, where stock's first cast differs"
    return None


def cast_time_weight(patcher: Any, key: str, device: Any, source: Any = None) -> Any:
    """Stock's cast-time weight for ``key``, computed on ``device``, or ``None``.

    ``source`` stands in for the module's current weight (the FSDP bake passes
    the pristine full weight). The weight is never modified in place.
    """
    if key not in patcher.patches:
        return None
    import comfy.model_management as mm
    import comfy.model_patcher as comfy_mp
    import comfy.utils

    module_key, _, param = key.rpartition(".")
    module = patcher.model
    try:
        for name in module_key.split(".") if module_key else ():
            module = getattr(module, name)  # comfy.utils.get_attr's walk
    except AttributeError:
        return None  # the classic bake's get_key_weight walks the same path
    weight = source if source is not None else getattr(module, param, None)
    if weight is None:
        return None
    why = _reason(patcher, key, module, param, weight)
    if why is not None:
        _say(why, "LoRA bake: %s keeps ComfyUI's classic bake (%s); later keys like it log nothing",
             key, why)
        return None
    dtype = compute_dtype(patcher.model, weight)
    if _is_quantized(weight):
        orig = weight.to(device=device)
        x = orig.to(dtype=dtype).dequantize()
    else:
        orig = None
        x = mm.cast_to_device(weight, device, dtype, copy=True)
    # The same callable stock attaches; it computes in x's dtype.
    x = comfy_mp.LowVramPatch(key, patcher.patches)(x)
    if orig is None:
        _say("plain", "LoRA bake: stock's cast-time math, first plain key %s in %s", key, dtype)
        return x
    seed = comfy.utils.string_to_seed(module_key)
    out = orig.requantize_from_float(x, scale="recalculate", stochastic_rounding=seed)
    _say("quantized", "LoRA bake: stock's cast-time math, first quantized key %s dequantized to %s "
         "and requantized with the module's seed", key, dtype)
    return out.to(dtype=weight.dtype)


def store(model: Any, key: str, out: Any, in_place: bool) -> None:
    """Write a cast-time weight: in place for a plain weight when asked, else a new Parameter."""
    import comfy.utils
    import torch

    if in_place and not _is_quantized(out):
        comfy.utils.copy_to_param(model, key, out)
        return
    module_key, _, param = key.rpartition(".")
    module = comfy.utils.get_attr(model, module_key) if module_key else model
    # MixedPrecisionOps' set_weight ends the same way for its requantized weight.
    setattr(module, param, torch.nn.Parameter(out, requires_grad=False))


def _patch_weight_to_device(stock: Any) -> Any:
    def patch_weight_to_device(self, key, device_to=None, inplace_update=False, return_weight=False,
                               force_cast=False):
        out = None
        if key in self.patches and not self.is_dynamic():
            import comfy.model_patcher as comfy_mp

            weight, _set_func, _convert = comfy_mp.get_key_weight(self.model, key)
            out = cast_time_weight(self, key, device_to if device_to is not None else self.load_device)
            if out is not None and device_to is None:
                out = out.to(device=weight.device)
        if out is None:
            return stock(self, key, device_to=device_to, inplace_update=inplace_update,
                         return_weight=return_weight, force_cast=force_cast)
        inplace_update = self.weight_inplace_update or inplace_update
        if key not in self.backup and not return_weight:
            self.backup[key] = _Backup(weight.to(device=self.offload_device, copy=inplace_update), inplace_update)
        if return_weight:
            return out
        store(self.model, key, out, inplace_update)
        return None

    marked = cast(Any, patch_weight_to_device)
    marked._dgxm_cast_time = True
    marked._dgxm_stock = stock
    return patch_weight_to_device


def install() -> bool:
    """Route the classic patcher's LoRA bake through stock's cast-time math.

    Idempotent. Returns whether this call installed it. A Worker process runs
    it when its residency is not comfy-managed (``comfy_dynamic.stage_b``).
    """
    try:
        import comfy.model_patcher as comfy_mp

        current = comfy_mp.ModelPatcher.patch_weight_to_device
    except (ImportError, AttributeError) as exc:
        safe_call(log.warning, "LoRA bake: ComfyUI's patcher is not importable here (%s); "
                  "a LoRA would bake with the classic math", failure_summary(exc))
        return False
    if getattr(current, _MARK, False):
        return False
    comfy_mp.ModelPatcher.patch_weight_to_device = _patch_weight_to_device(current)
    safe_call(log.info, "LoRA bake: stock ComfyUI's cast-time math (its DynamicVRAM default) for "
              "patched weights on this worker (docs/TROUBLESHOOTING.md #145)")
    return True


def uninstall() -> bool:
    """Put stock's method back; for tests and canaries. Returns whether it changed."""
    import comfy.model_patcher as comfy_mp

    current = comfy_mp.ModelPatcher.patch_weight_to_device
    if not getattr(current, _MARK, False):
        return False
    comfy_mp.ModelPatcher.patch_weight_to_device = current._dgxm_stock
    return True


__all__ = ["SMALL_MODULE_BYTES", "cast_time_weight", "compute_dtype", "install", "store", "uninstall"]
