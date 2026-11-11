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
cast does not settle on one weight; those keys keep the classic bake.
``stock_patches_at_cast_time`` is stock's own load-time choice, which the
canary checks against ComfyUI's real ``ModelPatcherDynamic.load``.
``install`` routes ComfyUI's classic ``patch_weight_to_device`` through
``cast_time_weight`` in a Worker process: the sample-time load, the
``lora_low_rss`` merge, the clone hot swap and the ambient verify's
reference all call that method. The lazy swap and the FSDP bake call ``cast_time_weight`` directly. A
dynamic patcher (comfy-managed residency) keeps stock's own method. ComfyUI
is imported only inside functions.

``install`` also runs ``keep_full_load_weights_on_offloaded_modules`` after the
classic patcher's ``load`` and ``partially_unload``. Those two give every
patched module they leave off the device a cast-time patch that runs on the
dequantized weight. A plain key stock bakes classically then takes the classic
recipe when its layer casts, and a quantized key takes ``cast_time_weight``,
so a partial load and a full load compute with the same weights.
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


def _stock_classic_reason(patcher: Any, key: str, module: Any, param: str, weight: Any) -> str | None:
    """Why ``ModelPatcherDynamic.load`` bakes ``key`` classically itself, or ``None``.

    Stock attaches a cast-time patch only to the weight or bias of a cast
    module larger than 16 KiB, and only when the LoRA keeps the weight's
    shape; it force-loads every other patched parameter through the classic
    bake. The bake-equivalence canary runs stock's own load to hold this.
    """
    if param not in ("weight", "bias") or not hasattr(module, "comfy_cast_weights"):
        return "stock patches it at load (not a cast weight)"
    import comfy.lora

    if _module_bytes(module) <= SMALL_MODULE_BYTES and not getattr(patcher.model, "comfy_writable", False):
        return "stock force-loads a module this small"
    if comfy.lora.calculate_shape(patcher.patches[key], weight, key) != weight.shape:
        return "the LoRA resizes the weight"
    return None


def _one_weight_reason(patcher: Any, key: str, module: Any, weight: Any) -> str | None:
    """Why stock's cast-time patch on ``key`` is not one weight, or ``None``."""
    import comfy.ops

    if key in getattr(patcher, "weight_wrapper_patches", {}):
        return "it carries a weight wrapper"
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


def _locate(patcher: Any, key: str, source: Any = None) -> tuple[Any, str, Any] | None:
    """The module, parameter name and weight ``key`` names, or ``None``."""
    module_key, _, param = key.rpartition(".")
    module = patcher.model
    try:
        for name in module_key.split(".") if module_key else ():
            module = getattr(module, name)  # comfy.utils.get_attr's walk
    except AttributeError:
        return None  # the classic bake's get_key_weight walks the same path
    weight = source if source is not None else getattr(module, param, None)
    return None if weight is None else (module, param, weight)


def stock_patches_at_cast_time(patcher: Any, key: str) -> bool:
    """Whether stock's DynamicVRAM load attaches a cast-time patch to ``key``."""
    found = _locate(patcher, key) if key in patcher.patches else None
    return found is not None and _stock_classic_reason(patcher, key, *found) is None


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

    found = _locate(patcher, key, source)
    if found is None:
        return None
    module, param, weight = found
    module_key = key.rpartition(".")[0]
    why = (_stock_classic_reason(patcher, key, module, param, weight)
           or _one_weight_reason(patcher, key, module, weight))
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


def _full_load_weight_at_cast(patcher: Any, key: str, module: Any, param: str, classic: bool) -> Any:
    """A lowvram patch that gives ``key`` the weight a full load bakes, each time its layer casts.

    ``classic`` is a plain key stock bakes classically: the recipe of
    ``ModelPatcher.patch_weight_to_device`` on the module's stored weight.
    Otherwise the key is a quantized weight: ``cast_time_weight``, the
    requantized weight stock's cast computes, in place of the dequantized float
    weight ComfyUI's classic cast hands the patch.
    """
    import comfy.float
    import comfy.lora
    import comfy.model_management as mm
    import comfy.model_patcher as comfy_mp
    import comfy.utils

    class FullLoadWeightAtCast(comfy_mp.LowVramPatch):
        _dgxm_full_load_weight = True

        def __call__(self, weight):
            if not classic:
                out = cast_time_weight(patcher, self.key, weight.device)
                return out if out is not None else comfy_mp.LowVramPatch.__call__(self, weight)
            stored = getattr(module, param)
            patches = self.prepared_patches if self.prepared_patches is not None else self.patches[self.key]
            temp = mm.cast_to_device(stored, weight.device, mm.lora_compute_dtype(weight.device), copy=True)
            out = comfy.lora.calculate_weight(patches, temp, self.key)
            out = comfy.float.stochastic_rounding(out, stored.dtype, seed=comfy.utils.string_to_seed(self.key))
            return out.to(dtype=weight.dtype)

    return FullLoadWeightAtCast(key, patcher.patches)


def keep_full_load_weights_on_offloaded_modules(patcher: Any) -> int:
    """Give each offloaded patched key the weight a full load bakes, at cast time.

    When ComfyUI's classic patcher leaves a patched module off the device (a
    partial load, or a partial unload under memory pressure), it attaches a
    cast-time patch to its weight and bias, and its cast runs that patch on the
    weight in the forward's dtype, dequantized. For a plain weight above 16 KiB
    that is stock's cast-time weight, the one a full load bakes. For two kinds
    of key it is not, and this swaps their patch:

    - a plain key stock bakes classically (a module of 16 KiB or less, which
      stock's DynamicVRAM load force-loads): on GB10 the two recipes round a
      BF16 norm weight with a diff apart on about 7 percent of its values. The
      swapped patch runs the classic recipe.
    - a quantized weight ``cast_time_weight`` serves: ComfyUI's classic cast
      computes with the dequantized weight plus the LoRA and never requantizes,
      which on an INT8 ConvRot layer is another weight than stock's (a layer's
      output about 0.01 NRMS apart). The swapped patch returns
      ``cast_time_weight``.

    A quantized key ``cast_time_weight`` does not serve keeps ComfyUI's own
    patch. Returns how many patches it swapped.
    """
    swapped = 0
    for module in patcher.model.modules():
        for param in ("weight", "bias"):
            functions = getattr(module, param + "_function", None)
            weight = getattr(module, param, None)
            if not functions or weight is None:
                continue
            for index, function in enumerate(functions):
                key = getattr(function, "key", None)
                if (not isinstance(key, str) or not getattr(function, "is_lowvram_patch", False)
                        or getattr(function, "_dgxm_full_load_weight", False) or key not in patcher.patches):
                    continue
                classic_why = _stock_classic_reason(patcher, key, module, param, weight)
                if _is_quantized(weight):
                    if classic_why is not None or _one_weight_reason(patcher, key, module, weight) is not None:
                        continue
                    why = "stock requantizes it when the layer casts"
                elif classic_why is None:
                    continue
                else:
                    why = classic_why
                functions[index] = _full_load_weight_at_cast(
                    patcher, key, module, param, classic=not _is_quantized(weight))
                swapped += 1
                _say("offloaded " + why, "LoRA bake: %s is off the device after a partial load; its "
                     "cast-time patch computes the weight a full load bakes (%s); later keys like it "
                     "log nothing", key, why)
    return swapped


def _then_keep_full_load_weights(stock: Any) -> Any:
    """``stock`` (ModelPatcher.load or partially_unload), then the swap above."""
    def method(self, *args, **kwargs):
        out = stock(self, *args, **kwargs)
        if self.patches and not self.is_dynamic():
            keep_full_load_weights_on_offloaded_modules(self)
        return out

    marked = cast(Any, method)
    marked._dgxm_cast_time = True
    marked._dgxm_stock = stock
    return method


# The classic patcher's methods that leave a patched module off the device.
_OFFLOADING_METHODS = ("load", "partially_unload")


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
    for name in _OFFLOADING_METHODS:
        method = getattr(comfy_mp.ModelPatcher, name, None)
        if method is None:  # a ComfyUI without it offloads nothing through it; the canary names the change
            safe_call(log.warning, "LoRA bake: ComfyUI's patcher has no %s here; a partial load "
                      "keeps ComfyUI's own cast-time patch on small modules", name)
            continue
        setattr(comfy_mp.ModelPatcher, name, _then_keep_full_load_weights(method))
    safe_call(log.info, "LoRA bake: stock ComfyUI's cast-time math (its DynamicVRAM default) for "
              "patched weights on this worker (docs/TROUBLESHOOTING.md #145)")
    return True


def uninstall() -> bool:
    """Put stock's methods back; for tests and canaries. Returns whether it changed."""
    import comfy.model_patcher as comfy_mp

    current = comfy_mp.ModelPatcher.patch_weight_to_device
    if not getattr(current, _MARK, False):
        return False
    comfy_mp.ModelPatcher.patch_weight_to_device = current._dgxm_stock
    for name in _OFFLOADING_METHODS:
        method = getattr(comfy_mp.ModelPatcher, name)
        if getattr(method, _MARK, False):
            setattr(comfy_mp.ModelPatcher, name, method._dgxm_stock)
    return True


__all__ = ["SMALL_MODULE_BYTES", "cast_time_weight", "compute_dtype", "install",
           "keep_full_load_weights_on_offloaded_modules", "stock_patches_at_cast_time", "store", "uninstall"]
