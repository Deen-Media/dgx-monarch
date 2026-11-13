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

Nothing is cleared before the full load has succeeded, so a route that is not
taken leaves the model as ComfyUI loaded it. A full load that fails does not:
ComfyUI undoes a load that raises (``ModelPatcher.partially_load`` calls
``detach()``), which leaves the model unpatched and unloaded. Sampling it
would render without the LoRA, so any failure of the route refuses the
render, class C, whatever the format. And after a route that reports success
``check`` reads the model itself (patched, whole, counted as loaded, still
held by ComfyUI) and refuses if it is not what a full load leaves.

The route loads the rest of the model through the model's own patcher, as
ComfyUI does when a loaded model may use more memory, and not through
``load_models_gpu``: that would free every other model the sampling loaded
(a dual-model guider's other checkpoint) and unpatch its LoRA. The check
runs on every model ``prepare_sampling`` loaded.

A WHOLE load keeps ComfyUI's backup too: the host copy of every weight the
stack patches stands through the whole render (the Worker held 17.7 GiB of
anonymous memory beside a 15.5 GiB model on record, most of it that backup).
ComfyUI's whole-load condition leaves only its reserve for what the sampler
takes beyond ComfyUI's own estimate, and that grows with the latent, so at
the tightest reading that still loads whole a long clip runs the host under
its floor with the backup standing by unused. So where a patched model is
loaded whole and the room above the floor is under the ask (``_sampling_ask``,
1.5 times ComfyUI's full estimate for this latent), ``check`` takes the
route's last step alone: the backup and the patches go, no weight moves, and
the slot is a low-RSS slot. With more room than that the backup stays, and a
stack change swaps in place as before. The weights are the classic bake's
either way. What can differ is ComfyUI's own choice, made from free memory at
each step, to run cond and uncond in one call: on a graph with two conds on
one model the memory given back can turn two calls into one, as it would with
that memory free for any other reason, and the float order then differs.
"""
from __future__ import annotations

from typing import Any

from .. import mesh_safety
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
# ``DGXM_DISABLE_BACKUP_FREE=1`` on a Worker: a whole classic load keeps ComfyUI's backup through the
# render however little room is left, as before this step existed (``free_backup_for_sampling``). The
# partial-load route is not touched by it. For diagnosis, and to read one build with and without.
BACKUP_FREE_DISABLE_ENV = "DGXM_DISABLE_BACKUP_FREE"


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


def _host_copies_are_file_pages() -> bool:
    """Whether the weights off the device are file pages (the mmap loader) and not the Worker's own copy.

    The Worker's pread loader marks ``safetensors.safe_open`` (the flag worker_status reports). Without
    it a checkpoint is memory-mapped, and moving a layer onto the device takes new memory for it.
    """
    try:
        import safetensors

        return not getattr(safetensors.safe_open, "_dgxm_pread", False)
    except Exception:
        return True


def _room_needed(patcher: Any, pending: list[tuple[str, Any]]) -> int:
    """Host bytes the full load needs beyond what the partial load holds."""
    from . import lora_cast_time

    need = sum(lora_cast_time._module_bytes(module) for _key, module in pending)  # ComfyUI's backup of each
    if _host_copies_are_file_pages():
        need += max(0, int(patcher.model_size()) - int(patcher.loaded_size()))  # and every layer it moves
    return need


def _layer_class(module: Any) -> tuple[str, int]:
    """How the classic bake's price reads one patched layer, from the live module: (class, stored bytes).

    An INT8 ConvRot layer and a scaled FP8 one are told by ComfyUI's own
    ``quant_format``; an FP8 weight with no quantization layout is plain FP8;
    a float weight is full precision. Anything else, and a module that cannot
    be read, is unclassed and takes the line's share, the dearer side.
    """
    from .. import capacity_classic_bake as price

    try:
        from . import lora_cast_time

        weight = getattr(module, "weight", None)
        nbytes = lora_cast_time._module_bytes(module)
        if lora_cast_time._is_quantized(weight):
            fmt = str(getattr(module, "quant_format", "") or "")
            if fmt.startswith("int8"):
                return price.INT8_QUANTIZED, nbytes
            return (price.FP8_QUANTIZED if fmt.startswith("float8") else price.OTHER), nbytes
        dtype = str(getattr(weight, "dtype", ""))
        if "float8" in dtype:
            return price.PLAIN_FP8, nbytes
        if dtype in ("torch.bfloat16", "torch.float16", "torch.float32", "torch.float64"):
            return price.FULL_PRECISION, nbytes
        return price.OTHER, nbytes
    except Exception:
        return price.OTHER, 0


def _bake_room(pending: list[tuple[str, Any]]) -> int:
    """What baking the pending layers ComfyUI's classic way adds on the device beside them.

    The stock price's own charge for a stack (``capacity_classic_bake``),
    worked on the layers still to bake and by each layer's own format: 2 GiB,
    and for each layer 7.5 MiB where it is scaled FP8 or full precision, 4 MiB
    where it is INT8 ConvRot, nothing where it is plain FP8. The line for
    every format asked 4.7 GiB for 371 INT8 layers on 2026-11-12 where the
    bake of such layers adds nothing on record (0.06 GiB for 1261 patched
    keys), and the route was declined for 0.3 GiB with the backup then kept
    through the render. Not the price's bytes guard, and not its exception
    for a reserve: this is the route's room, not a price.
    """
    from .. import capacity_classic_bake as price

    return price.term_from_weights((_layer_class(module) for _key, module in pending), guard=False)


def _registered(patcher: Any) -> bool:
    """Whether ComfyUI's model manager holds this patcher's model as loaded."""
    import comfy.model_management as mm

    return any(getattr(loaded.model, "model", None) is patcher.model for loaded in mm.current_loaded_models)


def make_whole(patcher: Any) -> str | None:
    """Take the low-RSS route for this partial load. ``None`` when taken, else why it was not.

    It raises if the full load raises. ComfyUI undoes a load that fails
    (``ModelPatcher.partially_load`` calls ``detach()``), so the caller must
    refuse the render then.
    """
    slot = getattr(patcher.model, STORED, None)
    stored = slot() if callable(slot) else None
    if stored is None:
        return "this model is not one the Worker's store holds"
    if not shares_host_memory(patcher):
        return ("this GPU has memory of its own, and ComfyUI loaded the model partly because it does "
                "not fit there whole")
    pending = _patched_off_device(patcher)
    need = _room_needed(patcher, pending) + _bake_room(pending)
    room = _host_room()
    if need > room:
        return (f"baking the {len(pending)} patched layer(s) still off the device needs {need / _GIB:.1f} GiB "
                f"while it bakes, and {max(0, room) / _GIB:.1f} GiB is available above the "
                f"{ABSOLUTE_HOST_FLOOR_BYTES / _GIB:.0f} GiB floor")
    # Load the rest of this one model through its own patcher, as ComfyUI's LoadedModel does when a
    # loaded model may use more memory (model_use_more_vram): no free_memory pass, so no other model
    # this sampling loaded (a dual-model guider's other checkpoint) is unloaded. Then the tail of
    # store_bake._merge_and_free: the backup and the patches go.
    patcher.partially_load(patcher.load_device, extra_memory=1e32)
    returned = _free_backup(patcher, stored)
    log.info("LoRA bake: ComfyUI loaded this model partly with a LoRA stack and lora_low_rss off, so the "
             "Worker took the low-RSS route for this load: baked %d more patched layer(s) under a full "
             "load and freed ComfyUI's backup (%.1f GiB of room was needed while baking). A stack "
             "change on this model reloads it from disk. %s", len(pending), need / _GIB, returned)
    return None


def _free_backup(patcher: Any, stored: Any) -> str:
    """The tail of ``store_bake._merge_and_free`` on a model that is whole and baked: the backup and the patches go.

    Returns what the box got back, for the caller's line: MemAvailable before the free, and after it
    and the host trim (``host_trim``: without one, weights under the C library's mapping threshold
    stay in its heap once freed, from a process's second load of them on).
    """
    import gc

    from . import host_trim

    before = _host_room() + ABSOLUTE_HOST_FLOOR_BYTES
    patcher.backup.clear()
    patcher.backup_buffers.clear()
    patcher.patches.clear()
    # A clone made between the store's patcher and the sampler's (the render clone, a sampling-shift
    # node's clone) copied the stack: no patcher of this model may keep it once it is baked in.
    parent = getattr(patcher, "parent", None)
    while parent is not None and getattr(parent, "model", None) is patcher.model:
        parent.patches.clear()
        parent = getattr(parent, "parent", None)
    gc.collect()
    active = getattr(stored, "active_patcher", None)
    if active is not None and active is not patcher:
        active.patches.clear()  # the same stack on the store's own patcher: it is baked in now
    stored.base_baked, stored.unbake = True, None
    ran, seconds, _at_trim, after = host_trim.trim()
    trimmed = f"host trim {seconds:.2f} s" if ran else "no host trim on this C library"
    return (f"MemAvailable read {before / _GIB:.1f} GiB before the free and {after / _GIB:.1f} GiB "
            f"after it ({trimmed})")


# ComfyUI runs cond and uncond in one call when this many times its full estimate is free
# (``comfy/samplers.py`` ``_calc_cond_batch``), and a batched call takes more than the estimate.
ASK_SHARE_OF_FULL_ESTIMATE = 1.5


def _sampling_ask(patcher: Any, noise_shape: Any, conds: Any) -> int | None:
    """What the sampler may ask for this latent, in bytes; ``None`` if ComfyUI's estimate cannot be had.

    1.5 times the first of ``comfy.sampler_helpers.estimate_memory``'s two
    figures, the one worked on the doubled batch (cond and uncond in one
    call). ComfyUI batches from that same 1.5, against free memory and not
    against the pack's floor: an ask of the full figure alone left a band
    where the backup was kept, ComfyUI batched, and the batched call (about
    1.15 times the estimate on the single-batch records, 81 and 241 frames of
    Wan-Animate 2, 2026-11-12) ran the host under its floor. Asking the
    threshold itself closes it and errs toward freeing, whose cost is a slower
    next stack change.
    """
    try:
        import comfy.sampler_helpers

        full, _minimum = comfy.sampler_helpers.estimate_memory(patcher, noise_shape, conds)
        return int(ASK_SHARE_OF_FULL_ESTIMATE * full)
    except Exception:
        return None


def free_backup_for_sampling(patcher: Any, noise_shape: Any, conds: Any) -> bool:
    """On a whole classic load, free ComfyUI's backup where the room left is under what sampling will ask.

    ``True`` when it freed it. The model must be whole and patched, hold a
    backup, sit on a device that shares the host's memory and belong to a
    slot of the Worker's store; the room is the host's available memory above
    its floor, the ask ``_sampling_ask`` (1.5 times ComfyUI's full estimate).
    No weight moves: the weights on the
    device are the baked ones already, and ComfyUI needs the backup only to
    change the stack in place, which the slot then does by a reload from disk
    (``StoredModel.base_baked``), as after the partial-load route and as with
    ``lora_low_rss`` on. ``BACKUP_FREE_DISABLE_ENV`` set on the Worker keeps the
    backup and logs what the step would have weighed.
    """
    model = patcher.model
    if not getattr(patcher, "backup", None) or getattr(model, "current_weight_patches_uuid", None) is None:
        return False
    slot = getattr(model, STORED, None)
    stored = slot() if callable(slot) else None
    if stored is None or not shares_host_memory(patcher):
        return False
    ask = _sampling_ask(patcher, noise_shape, conds)
    if ask is None:
        return False
    room = _host_room()
    if room >= ask:
        return False
    figures = (max(0, room) / _GIB, ABSOLUTE_HOST_FLOOR_BYTES / _GIB, ask / _GIB)
    weighed = ("LoRA bake: ComfyUI loaded this model whole with a LoRA stack and lora_low_rss off, and "
               "%.1f GiB is available above the %.0f GiB floor where the sampler may ask %.1f GiB (1.5 "
               "times ComfyUI's estimate for this latent)")
    if mesh_safety.env_enabled(BACKUP_FREE_DISABLE_ENV):
        kept = (f"{weighed}. ComfyUI's weight backup is kept through the render: the step that frees it "
                f"is off ({BACKUP_FREE_DISABLE_ENV}=1 on the Worker)")
        log.info(kept, *figures)
        return False
    returned = _free_backup(patcher, stored)
    freed = (f"{weighed}, so the Worker freed ComfyUI's weight backup before the first step (the weights "
             "on the device are the baked ones; none moved). A stack change on this model reloads it "
             f"from disk. {returned}")
    log.info(freed, *figures)
    return True


def _what_is_wrong_after_the_route(patcher: Any, was_registered: bool) -> str | None:
    """A second wall, read from the model and not from the route: what it lacks to sample with its LoRA."""
    model = patcher.model
    if getattr(model, "current_weight_patches_uuid", None) is None:
        return "ComfyUI holds its weights as unpatched"
    if getattr(model, "model_lowvram", False) or _patched_off_device(patcher):
        return "part of it is still off the device with a cast-time patch"
    if not getattr(model, "model_loaded_weight_memory", 0):
        return "ComfyUI counts none of it as loaded"
    if was_registered and not _registered(patcher):
        return "ComfyUI no longer holds it as a loaded model"
    return None


def check(patcher: Any, sampling: tuple[Any, Any] | None = None) -> None:
    """After ComfyUI's load and before the first forward: make a partial load whole, or refuse what would differ.

    ``sampling`` is ``prepare_sampling``'s latent shape and its conds. With
    them a WHOLE load is weighed too, and gives up ComfyUI's backup where the
    sampler will need the room (``free_backup_for_sampling``). Without them a
    whole load is left as it is.
    """
    model = getattr(patcher, "model", None)
    if model is None or patcher.is_dynamic() or not getattr(patcher, "patches", None):
        return
    partly = bool(getattr(model, "model_lowvram", False))
    if not partly and sampling is None:
        return
    was_registered = _registered(patcher)
    if not partly:
        if sampling is None or not free_backup_for_sampling(patcher, *sampling):
            return
        why_not: str | None = None  # whole and baked, its backup gone: the state the route ends in
    else:
        try:
            why_not = make_whole(patcher)
        except Exception as exc:
            # ComfyUI undoes a load that fails (ModelPatcher.partially_load calls detach()): the model is
            # no longer the partial load prepare_sampling left, and sampling it would run without its LoRA.
            raise OffloadedInputQuantError(refusal(
                RefusalClass.CAPACITY,
                "a LoRA stack is on a model ComfyUI loaded only partly, and the full load the Worker ran "
                    f"to bake it whole failed ({type(exc).__name__}: {exc}). ComfyUI unloads a model whose "
                    "load fails, so nothing was sampled. What would fit: lora_low_rss on (the default on a "
                    "DGX Spark), which bakes at load time before anything else holds the memory; more free "
                    "memory on this box; or the render without the LoRA.",
                guard=GUARD, waivable=False, troubleshooting=TROUBLESHOOTING)) from exc
    if why_not is None:
        wrong = _what_is_wrong_after_the_route(patcher, was_registered)
        if wrong is not None:
            did = ("ComfyUI loaded only partly, and after the Worker loaded the rest to bake it whole" if partly
                   else "ComfyUI loaded whole, and after the Worker freed its weight backup for the sampler")
            raise OffloadedInputQuantError(refusal(
                RefusalClass.CAPACITY,
                f"a LoRA stack is on a model {did} the model is not as a full load leaves it: {wrong}. "
                    "Sampling it could run without its LoRA, so nothing was sampled. Reset the attached "
                    "mesh so the model loads again, and turn lora_low_rss on (the default on a DGX Spark), "
                    "which bakes at load time.",
                guard=GUARD, waivable=False, troubleshooting=TROUBLESHOOTING))
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
    """Run ``check`` after ComfyUI's ``prepare_sampling`` load, once, on every model it loaded.

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
        # prepare_sampling(model, ...) returns (real model, conds, the models it loaded). Its first
        # argument is the patcher it loaded; a dual-model guider's other checkpoint is among the rest,
        # and a later load can leave an earlier model partly loaded after that model's own check.
        loaded = [args[0] if args else kwargs.get("model")]
        if isinstance(result, tuple) and len(result) > 2 and isinstance(result[2], list | tuple):
            loaded.extend(result[2])
        # The latent's shape and the conds, as prepare_sampling got them: what a whole load is weighed by.
        shape = args[1] if len(args) > 1 else kwargs.get("noise_shape")
        conds = args[2] if len(args) > 2 else kwargs.get("conds")
        sampling = (shape, conds) if shape is not None and conds is not None else None
        seen: set[int] = set()
        for patcher in loaded:
            if id(patcher) in seen or not (hasattr(patcher, "is_dynamic") and hasattr(patcher, "model")):
                continue
            seen.add(id(patcher))
            check(patcher, sampling)
        return result

    pe.add_wrapper_with_key(pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, wrapper, options, is_model_options=True)


__all__ = ["STORED", "OffloadedInputQuantError", "check", "free_backup_for_sampling", "inexact", "install_guard",
           "make_whole", "quantizes_input", "shares_host_memory"]
