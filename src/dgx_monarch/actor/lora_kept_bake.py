"""Keep a full load's baked weight on a patched module ComfyUI leaves off the device.

``lora_cast_time.keep_full_load_weights_on_offloaded_modules`` gives an
offloaded patched key the full load's weight each time its layer casts. That
is exact for the weight and it has two costs:

- A layer of a format that also quantizes its input (FP8, NVFP4, MXFP8) then
  computes another output than a fully loaded one. ComfyUI's forward
  quantizes the input only while the layer carries no cast-time patch, and
  the classic patcher has no other slot for one. No patch at cast time can
  close that.
- The bake runs on every cast. On an INT8 ConvRot layer that is a
  dequantize, the LoRA and a requantization each step, where a full load, and
  stock's DynamicVRAM after its first forward, cast a weight baked once.

Both end the same way: bake the weight once and store it on the offloaded
module, with ComfyUI's own backup of the pristine weight, and take the patch
off. The layer then is what a fully loaded layer is, kept on the host: it
casts a baked weight, carries no patch, and so quantizes its input. This is
ComfyUI's ``force_patch_weights`` state for one module, with the bake run on
the load device as every other bake is.

It costs one host copy of each kept weight: the baked weight beside its
backup. Where that copy comes from decides whether it may be taken:

- On a GPU with memory of its own the copy is host RAM, which is not what
  ran short. A module is kept while the host has the room above a floor.
- On a GPU that shares the host's memory (a DGX Spark) the copy comes from
  the pool ComfyUI just held back for the render: it left the layer off the
  device because a second copy of it did not fit beside the render's working
  memory. Nothing is kept there.

A module that is not kept carries the cast-time patch, which is exact for a
weight-only format. For a layer that quantizes its input there is no exact
fallback: ``check`` refuses the render, class C, before its first forward,
and names what would fit. On a Spark that is ``lora_low_rss`` on, its
default, which bakes under a full load and keeps no backup.

ComfyUI's classic patcher has no state for "baked and off the device", so the
kept module borrows one: it carries ``comfy_patched_weights`` as a loaded
module does, which makes a later ``load`` move it without patching it again.
``partially_unload`` would read the same flag as "on the device" and count
memory it did not free, so the flag is hidden from that one call (``hide``).
A full unpatch restores the backup as for any baked weight.

Held on CPU against ComfyUI's own patcher. The kept path runs only on a GPU
with memory of its own, which no hardware record of this project covers.
"""
from __future__ import annotations

from typing import Any

from ..capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)

_GIB = 2 ** 30
KEPT = "_dgxm_kept_bake"  # on a module whose baked weight this module stored off the device
WRAPPER_KEY = "dgx_monarch_offloaded_input_quant_guard"
# The guard partial_load_guard raises across ranks: a partial load that would not compute what a
# full load computes. Spelled out here because the refusal ledger reads a guard from its own module.
GUARD = "partial_load_divergence"
TROUBLESHOOTING = 145


class OffloadedInputQuantError(RuntimeError):
    """A patched layer that quantizes its input is off the device and its bake could not be kept."""


def separate_host_memory(patcher: Any) -> bool:
    """Whether the load device has memory of its own, so a host copy is not taken from it.

    Only a CUDA device that does not report itself integrated. The CPU, an
    integrated GPU, and any device this cannot ask share the host's pool.
    """
    device = getattr(patcher, "load_device", None)
    if getattr(device, "type", None) != "cuda":
        return False
    try:
        import torch

        return not bool(getattr(torch.cuda.get_device_properties(device), "is_integrated", 0))
    except Exception:
        return False


class Budget:
    """Host bytes kept bakes may take in one pass.

    Nothing where the load device shares the host's memory. Elsewhere, what
    the host has available now above the pack's host floor.
    """

    def __init__(self, patcher: Any) -> None:
        self.separate = separate_host_memory(patcher)
        self.left = 0
        if self.separate:
            import psutil

            self.left = int(psutil.virtual_memory().available) - ABSOLUTE_HOST_FLOOR_BYTES

    def take(self, nbytes: int) -> bool:
        if nbytes > self.left:
            return False
        self.left -= nbytes
        return True


def quantizes_input(module: Any) -> bool:
    """Whether ComfyUI's forward quantizes this layer's input when it carries no cast-time patch."""
    from . import lora_cast_time

    if not lora_cast_time._is_quantized(getattr(module, "weight", None)):
        return False
    if (getattr(module, "layout_type", None) is None or getattr(module, "_full_precision_mm", False)
            or getattr(module, "comfy_force_cast_weights", False)):
        return False
    import comfy.ops

    # The lookup ComfyUI's Linear.forward makes; a format it does not list quantizes its input.
    return bool(comfy.ops.QUANT_ALGOS.get(getattr(module, "quant_format", None), {}).get("quantize_input", True))


def _off_device(module: Any) -> bool:
    return hasattr(module, "prev_comfy_cast_weights")  # ComfyUI's own mark of a module it left off the device


def _patched_params(patcher: Any, module: Any) -> list[tuple[str, str, Any]] | None:
    """``(param, key, weight)`` for each cast-time patch on the module, or ``None`` if it cannot be kept.

    A module is kept whole or not at all, because ``comfy_patched_weights`` is
    one flag for its weight and its bias. A weight wrapper after the patch, a
    patch whose key left the stack, or a quantized weight outside a Linear
    (whose setter this module does not vouch for) leaves it to the cast.
    """
    import comfy.ops

    from . import lora_cast_time

    found = []
    for param in ("weight", "bias"):
        functions = getattr(module, param + "_function", None)
        if not functions:
            continue
        weight = getattr(module, param, None)
        if weight is None or len(functions) != 1:
            return None
        key = getattr(functions[0], "key", None)
        if (not isinstance(key, str) or not getattr(functions[0], "is_lowvram_patch", False)
                or key not in patcher.patches or key in getattr(patcher, "weight_wrapper_patches", {})):
            return None
        if lora_cast_time._is_quantized(weight) and not (
                isinstance(module, comfy.ops.MixedPrecisionOp) and type(module).__name__ == "Linear"):
            return None
        found.append((param, key, weight))
    return found or None


def keep(patcher: Any, module: Any, budget: Budget) -> bool:
    """Bake the module's patched weights once and store them on it, off the device. True when kept.

    Every bake is computed before anything is stored, so a failure leaves the
    module as ComfyUI had it, with its cast-time patches.
    """
    from . import lora_cast_time

    if not _off_device(module):
        return False
    params = _patched_params(patcher, module)
    if params is None or not budget.take(lora_cast_time._module_bytes(module)):
        return False
    try:
        baked = [(param, key, weight,
                  patcher.patch_weight_to_device(key, device_to=patcher.load_device, return_weight=True)
                  .to(device=weight.device))
                 for param, key, weight in params]
    except Exception as exc:
        safe_call(log.warning, "LoRA bake: could not bake %s to keep it off the device (%s); it keeps "
                  "its cast-time patch", params[0][1], failure_summary(exc))
        return False
    in_place = bool(getattr(patcher, "weight_inplace_update", False))
    for param, key, weight, out in baked:
        patcher.unpin_weight(key)
        if key not in patcher.backup:
            patcher.backup[key] = lora_cast_time._Backup(
                weight.to(device=patcher.offload_device, copy=in_place), in_place)
        lora_cast_time.store(patcher.model, key, out, in_place)
        setattr(module, param + "_function", [])
        patcher.model.lowvram_patch_counter = max(0, int(getattr(patcher.model, "lowvram_patch_counter", 0)) - 1)
        patcher.pin_weight_to_device(key)
    module.comfy_patched_weights = True
    setattr(module, KEPT, True)
    return True


def _kept(patcher: Any) -> list[Any]:
    return [module for module in patcher.model.modules()
            if getattr(module, KEPT, False) and _off_device(module)
            and getattr(module, "comfy_patched_weights", False) is True]


def hide(patcher: Any) -> list[Any]:
    """Before ``partially_unload``: make it pass over the kept modules, which are off the device already."""
    kept = _kept(patcher)
    for module in kept:
        module.comfy_patched_weights = False
    return kept


def unhide(kept: list[Any]) -> None:
    for module in kept:
        if getattr(module, KEPT, False) and _off_device(module):
            module.comfy_patched_weights = True


def forget_moved(patcher: Any) -> None:
    """Drop the mark from a module ComfyUI has since loaded or unpatched; it is an ordinary module again."""
    for module in patcher.model.modules():
        if getattr(module, KEPT, False) and not (
                _off_device(module) and getattr(module, "comfy_patched_weights", False) is True):
            delattr(module, KEPT)


def inexact(patcher: Any) -> list[tuple[str, int]]:
    """``(key, bytes)`` of each off-device layer that quantizes its input and still carries a cast-time patch."""
    from . import lora_cast_time

    out = []
    for module in patcher.model.modules():
        functions = getattr(module, "weight_function", None)
        if not functions or not quantizes_input(module):
            continue
        key = next((getattr(f, "key", None) for f in functions if getattr(f, "is_lowvram_patch", False)), None)
        if isinstance(key, str):
            out.append((key, lora_cast_time._module_bytes(module)))
    return out


def check(patcher: Any) -> None:
    """Refuse, before the first forward, a render whose patched input-quantizing layers could not be kept."""
    model = getattr(patcher, "model", None)
    if model is None or not getattr(model, "model_lowvram", False) or patcher.is_dynamic():
        return
    short = inexact(patcher)
    if not short:
        return
    need = sum(nbytes for _key, nbytes in short)
    budget = Budget(patcher)
    formats = sorted({str(getattr(module, "quant_format", "")) for module in model.modules()
                      if quantizes_input(module)} - {""})
    if budget.separate:
        room = max(0, budget.left)
        why = (f"keeping those bakes needs {need / _GIB:.1f} GiB of host memory where {room / _GIB:.1f} GiB "
               f"is available above the {ABSOLUTE_HOST_FLOOR_BYTES / _GIB:.0f} GiB floor")
        more = f"{max(0, need - room) / _GIB:.1f} GiB more free host memory; "
    else:
        why = (f"this GPU shares the host's memory, so the {need / _GIB:.1f} GiB those bakes need would "
               "come out of the room ComfyUI held back for the render when it left the layers off the device")
        more = ""
    raise OffloadedInputQuantError(refusal(
        RefusalClass.CAPACITY,
        f"a LoRA patches {len(short)} layer(s) that ComfyUI left off the device (the model is loaded "
        f"partly), in a checkpoint format that also quantizes a layer's input ({', '.join(formats)}). Such "
        f"a layer computes the full load's output only from a weight baked once and kept, and {why}. With "
        "the patch applied at cast time the layer's input stays unquantized and the render would differ "
        f"from a full load without an error, so nothing was sampled; the first such key is {short[0][0]}. "
        "What would fit: lora_low_rss on (the default on a DGX Spark), which bakes the LoRA under a full "
        f"load and keeps no backup; {more}an INT8 or BF16 checkpoint of the same model, whose layers do "
        "not quantize their input; or the render without the LoRA.",
        guard=GUARD, waivable=False, troubleshooting=TROUBLESHOOTING))


def install_guard(base_patcher: Any) -> None:
    """Run ``check`` after ComfyUI's ``prepare_sampling`` load on this patcher, once.

    A process with no ComfyUI to import has no ComfyUI load to check, and is left alone.
    """
    try:
        import comfy.patcher_extension as pe
    except ImportError:
        return

    options = base_patcher.model_options
    if pe.get_wrappers_with_key(pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, options, is_model_options=True):
        return

    def wrapper(executor: Any, *args: Any, **kwargs: Any) -> Any:
        result = executor(*args, **kwargs)
        # prepare_sampling(model, ...): its first argument is the patcher it just loaded.
        patcher = args[0] if args else kwargs.get("model")
        if hasattr(patcher, "is_dynamic") and hasattr(patcher, "model"):
            check(patcher)
        return result

    pe.add_wrapper_with_key(pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, wrapper, options, is_model_options=True)


__all__ = ["KEPT", "Budget", "OffloadedInputQuantError", "check", "forget_moved", "hide", "inexact",
           "install_guard", "keep", "quantizes_input", "separate_host_memory", "unhide"]
