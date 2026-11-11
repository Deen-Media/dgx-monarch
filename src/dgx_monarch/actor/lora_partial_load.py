"""A one-GPU partial load with a LoRA stack: make it whole, or say why it cannot be.

With ``lora_low_rss`` off, ComfyUI's classic patcher may load a patched model
only partly. ``lora_cast_time`` then gives each patched key it leaves off the
device the full load's weight when its layer casts. That is exact for the
weight and it has two costs:

- The bake runs on every cast: on an INT8 ConvRot layer a dequantize, the
  LoRA and a requantization each step, where a full load casts a weight baked
  once.
- A layer of a format that also quantizes its input (FP8, NVFP4, MXFP8) still
  computes another output than a loaded one, because ComfyUI's forward
  quantizes the input only while the layer carries no cast-time patch. No
  patch at cast time can close that.

One route has neither cost, and it is the one the default Spark configuration
already takes: bake under a full load, then free ComfyUI's backup and the
patches (``store_bake._merge_and_free``, what ``lora_low_rss`` does). After
ComfyUI's ``prepare_sampling`` load, ``check`` takes that route for the load
by itself when it can:

- The load device must share the host's memory. There a partial load is
  ComfyUI's budget and its backup at work, not a model that cannot fit, and
  moving the rest of the model onto the device moves bytes within one pool. A
  GPU with memory of its own was loaded partly because the model does not fit
  it; nothing can make that load whole.
- The host must have room, above the pack's host floor, for the backups of
  the patched weights still off the device. ComfyUI keeps them while it bakes
  and they are freed with every other backup when the bake is done, so the
  model ends with less memory held than the partial load had.

The slot is then what a low-RSS slot is: baked, with no backup, so a later
stack change reloads from disk (``StoredModel.base_baked``).

Where the route cannot be taken the cast-time weights stand, which is exact
for every weight-only format. A render that would run a patched
input-quantizing layer off the device is refused, class C, before its first
forward, with what was tried and what would fit.

Nothing is cleared before the full load has succeeded, so a refusal or a
failure leaves the model as ComfyUI had it.
"""
from __future__ import annotations

from typing import Any

from ..capacity_floor import ABSOLUTE_HOST_FLOOR_BYTES
from ..log import get_logger
from ..refusal import RefusalClass, refusal

log = get_logger(__name__)

_GIB = 2 ** 30
WRAPPER_KEY = "dgx_monarch_partial_load_with_lora"
# The guard partial_load_guard raises across ranks: a partial load that would not compute what a
# full load computes. Spelled out here because the refusal ledger reads a guard from its own module.
GUARD = "partial_load_divergence"
TROUBLESHOOTING = 145
STORED = "_dgxm_stored"  # on a model: a weak reference to the store slot that holds it (stored_model.py)


class OffloadedInputQuantError(RuntimeError):
    """A patched layer that quantizes its input is off the device and the load could not be made whole."""


def shares_host_memory(patcher: Any) -> bool:
    """Whether the load device draws on the host's own memory: the CPU, or an integrated CUDA device."""
    device = getattr(patcher, "load_device", None)
    kind = getattr(device, "type", None)
    if kind == "cpu":
        return True
    if kind != "cuda":
        return False
    try:
        import torch

        return bool(getattr(torch.cuda.get_device_properties(device), "is_integrated", 0))
    except Exception:
        return False


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


def _patched_off_device(patcher: Any) -> list[tuple[str, Any]]:
    """``(key, module)`` for each module that carries a cast-time LoRA patch on its weight or bias."""
    out = []
    for module in patcher.model.modules():
        for param in ("weight", "bias"):
            key = next((getattr(f, "key", None) for f in getattr(module, param + "_function", None) or ()
                        if getattr(f, "is_lowvram_patch", False)), None)
            if isinstance(key, str):
                out.append((key, module))
                break
    return out


def inexact(patcher: Any) -> list[tuple[str, int]]:
    """``(key, bytes)`` of each off-device layer that quantizes its input and carries a cast-time patch."""
    from . import lora_cast_time

    return [(key, lora_cast_time._module_bytes(module)) for key, module in _patched_off_device(patcher)
            if quantizes_input(module)]


def _host_room() -> int:
    import psutil

    return int(psutil.virtual_memory().available) - ABSOLUTE_HOST_FLOOR_BYTES


def make_whole(patcher: Any) -> str | None:
    """Take the low-RSS route for this partial load. ``None`` when taken, else why it was not."""
    from . import lora_cast_time, store_bake

    slot = getattr(patcher.model, STORED, None)
    stored = slot() if callable(slot) else None
    if stored is None:
        return "this model is not one the Worker's store holds"
    if not shares_host_memory(patcher):
        return ("this GPU has memory of its own, and ComfyUI loaded the model partly because it does "
                "not fit there whole")
    pending = _patched_off_device(patcher)
    need = sum(lora_cast_time._module_bytes(module) for _key, module in pending)
    room = _host_room()
    if need > room:
        return (f"baking the {len(pending)} patched layer(s) still off the device needs {need / _GIB:.1f} GiB "
                f"for ComfyUI's backup of them while it bakes, and {max(0, room) / _GIB:.1f} GiB is available "
                f"above the {ABSOLUTE_HOST_FLOOR_BYTES / _GIB:.0f} GiB floor")
    store_bake._merge_and_free(patcher)  # a full load bakes the rest; then the backup and the patches go
    active = getattr(stored, "active_patcher", None)
    if active is not None and active is not patcher:
        active.patches.clear()  # the same stack on the store's own patcher: it is baked in now
    stored.base_baked, stored.unbake = True, None
    log.info("LoRA bake: ComfyUI loaded this model partly with a LoRA stack and lora_low_rss off, so the "
             "Worker took the low-RSS route for this load: baked %d more patched layer(s) under a full "
             "load and freed ComfyUI's backup (%.1f GiB of room was needed for it while baking). A stack "
             "change on this model reloads it from disk", len(pending), need / _GIB)
    return None


def check(patcher: Any) -> None:
    """After ComfyUI's load and before the first forward: make a partial load whole, or refuse what would differ."""
    model = getattr(patcher, "model", None)
    if (model is None or not getattr(model, "model_lowvram", False) or patcher.is_dynamic()
            or not getattr(patcher, "patches", None)):
        return
    try:
        why_not = make_whole(patcher)
    except Exception as exc:
        # Nothing was cleared: ComfyUI's backup and the cast-time patches are as its own load left them.
        why_not = f"the full load it needs failed ({type(exc).__name__}: {exc})"
    if why_not is None:
        return
    short = inexact(patcher)
    if not short:
        log.info("LoRA bake: this model stays partly loaded, with the full load's weights computed "
                 "when each off-device layer casts: %s", why_not)
        return
    formats = sorted({str(getattr(module, "quant_format", "")) for module in model.modules()
                      if quantizes_input(module)} - {""})
    raise OffloadedInputQuantError(refusal(
        RefusalClass.CAPACITY,
        f"a LoRA patches {len(short)} layer(s) that ComfyUI left off the device (the model is loaded "
        f"partly), in a checkpoint format that also quantizes a layer's input ({', '.join(formats)}). With "
        "the patch applied at cast time such a layer's input stays unquantized, so the render would "
        "differ from a full load without an error. The Worker tried to bake the whole stack under a full "
        f"load, as lora_low_rss does, and could not: {why_not}. Nothing was sampled, and the model is as "
        f"ComfyUI loaded it; the first such key is {short[0][0]}. What would fit: lora_low_rss on (the "
        "default on a DGX Spark), which bakes at load time before anything else holds the memory; more "
        "free memory on this box; an INT8 or BF16 checkpoint of the same model, whose layers do not "
        "quantize their input; or the render without the LoRA.",
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


__all__ = ["STORED", "OffloadedInputQuantError", "check", "inexact", "install_guard", "make_whole",
           "quantizes_input", "shares_host_memory"]
