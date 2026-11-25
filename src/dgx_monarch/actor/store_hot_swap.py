"""The classic LoRA hot swap for ModelStore, outside ``lora_low_rss``.

``ModelStore.ensure`` calls ``hot_swap_transition`` for a stack change on a
resident whose base the low-RSS bake did not consume: it clones the pristine
base, applies the new stack, and lets ComfyUI's next load patch the weights.
``copies_off_the_gpu`` sends a change that would first copy weights off the GPU
to a fresh load, and a quantized resident has ComfyUI's patches taken off
before the next load, in place where that moves no other weight
(``unpatch_in_place``). ``store_bake`` re-exports every public name here, and
the store's methods are reached through the store it is handed.
"""
from __future__ import annotations

import time
from typing import Any

from ..constants import TRANSITION_HOT_SWAP
from . import store_identity


def invalidate_gpu_weights(old_active) -> None:
    """Unload every model so the next load rebuilds patched quantized GPU weights from CPU."""
    import comfy.model_management as mm

    del old_active
    mm.unload_all_models()
    mm.soft_empty_cache()


def unpatch_in_place(mm, patchers: tuple) -> bool:
    """Take ComfyUI's weight patches off a model it holds whole, moving no other weight.

    ComfyUI's own unload restores each patched key from its host backup and then
    moves the whole model to the offload device, the host: on unified memory a
    copy of every weight through pageable memory, 83 to 100 s for a 12 GiB
    quantized model on the pair, to change 40 weights. Here each clone of the
    model points its offload device at its load device for the unload, as a
    copy-free drop does (``resident_ledger.unload_dropped``), so the unpatch
    puts the backed-up keys back on the device and leaves every other weight
    where it is; the offload devices are restored after, so the next patch
    keeps its backup on the host as before. The model stays, pristine, and
    ComfyUI's next load applies the new stack to the patched keys only.
    Returns False when ComfyUI holds no entry for the model (nothing ran).
    """
    model = getattr(patchers[0], "model", None)
    loaded = getattr(mm, "current_loaded_models", None)
    if model is None or not isinstance(loaded, list):
        return False
    entries = [entry for entry in list(loaded)
               if getattr(getattr(entry, "model", None), "model", None) is model]
    if not entries:
        return False
    clones: dict[int, Any] = {}
    for patcher in (*patchers, *(entry.model for entry in entries)):
        if patcher is not None and getattr(patcher, "load_device", None) is not None:
            clones.setdefault(id(patcher), patcher)
    saved = [(patcher, patcher.offload_device) for patcher in clones.values()]
    try:
        for patcher, _device in saved:
            patcher.offload_device = patcher.load_device
        for entry in entries:
            if entry.model_unload():
                for index in range(len(loaded) - 1, -1, -1):
                    if loaded[index] is entry:
                        loaded.pop(index)
    finally:
        for patcher, device in saved:
            patcher.offload_device = device
    mm.soft_empty_cache()
    return True


def _unpatch_quantized_in_place(store, stored, slot: str) -> bool:
    """The quantized hot swap's unpatch without a host copy, where that is proven safe.

    The gates are the copy-free drop's (``resident_ledger.detaches_alone``): a
    plain stock resident on unified memory, which ComfyUI holds whole, and
    whose model the other slot does not share. Anything else keeps the global
    unload. A model ComfyUI no longer holds needs nothing when it names no
    patch uuid; when it still names one, the global unload runs. The swap
    itself leaves the other slot alone; with smart memory off, as on a Worker,
    ComfyUI's next load that does not use that slot still unloads it.
    """
    import comfy.model_management as mm

    from . import resident_ledger, store_residency

    base = stored.base_patcher
    model = getattr(base, "model", None)
    other = store._stored("cond" if slot == "uncond" else "uncond")
    if (model is None or getattr(model, "model_lowvram", False)
            or not resident_ledger.detaches_alone(stored)
            or store_residency.describe(stored)["residency"] != "stock"
            or (other is not None and getattr(getattr(other, "base_patcher", None), "model", None) is model)):
        return False
    if unpatch_in_place(mm, (base, stored.active_patcher)):
        return True
    return getattr(model, "current_weight_patches_uuid", None) is None


def swaps_in_place(stored, lora_low_rss: bool) -> bool:
    """Whether a LoRA change on this resident is made from this stock copy of the base.

    Mirrors the two branches ``ModelStore.ensure`` takes before a fresh load:
    the low-RSS lazy swap restores and re-bakes in place, and the classic hot
    swap clones the pristine base and re-applies the stack. Neither places a
    second copy of the base. A classic change that ``copies_off_the_gpu`` sends
    to a fresh load instead drops this copy first, which returns its GPU copy,
    and the Worker's own wall prices that load whole after the drop, so the
    credit holds there too. A pending slab retry reloads, and so does a base
    the low-RSS bake consumed with no un-bake record. The fleet's capacity
    quote reads this from the store snapshot (``capacity_lora_bake``).
    """
    if stored is None or getattr(stored, "slab_auto_retry", False):
        return False
    if lora_low_rss:
        return stored.unbake is not None or not stored.lora_sig
    return not stored.base_baked


def _placed_bytes(model) -> int:
    """What ComfyUI's ledger says it placed of this model on the GPU; 0 when unknown."""
    value = getattr(model, "model_loaded_weight_memory", 0)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def copies_off_the_gpu(store, stored, slot: str, active) -> tuple[int, int]:
    """Weights an in-place swap would first copy off the GPU, as (uncovered, patched).

    ComfyUI applies a new stack to a placed model by unpatching it to the host
    (``ModelPatcher.unpatch_model``): each weight its backup holds goes back
    as is, and every other weight is copied from the GPU into fresh host memory
    before the new patches apply. From a bare base that backup is empty, so
    the whole model is copied. On the 2026-11-05 hardware leg the BF16
    Wan-Animate 2 swap spent 230 s in that copy, while a fresh load of base
    and LoRA took 13.1 s and its patch 4.0 s. A fresh load drops the slot
    without a host copy (``resident_ledger.unload_dropped``) and the
    sampler patches from the file's own host copy, at the same price.

    So ``uncovered`` counts the weights the new stack patches that the backup
    the clones share does not hold, and the store reloads when it is not 0.
    It stays 0, so the swap stays in place, unless every gate holds: classic
    mode (the caller), a stock resident (slab, FSDP and comfy-managed keep their
    own paths), a model ComfyUI placed on the GPU, and an empty other slot.
    That last gate dates from the drop's global unload, which pushed the other
    slot's model to the host the same slow way. On unified memory a stock drop
    no longer does (``resident_ledger.unload_dropped``); the gate stays until
    the in-place route beside a held slot is measured on hardware.
    """
    from . import store_residency

    patches = getattr(active, "patches", None)
    backup = getattr(stored.base_patcher, "backup", None)
    other = store._stored("cond" if slot == "uncond" else "uncond")
    if (not isinstance(patches, dict) or not isinstance(backup, dict) or other is not None
            or store_residency.describe(stored)["residency"] != "stock"
            or getattr(stored.base_patcher, "_dgxm_fsdp", False)
            or _placed_bytes(getattr(stored.base_patcher, "model", None)) <= 0):
        return 0, len(patches) if isinstance(patches, dict) else 0
    return len(set(patches) - set(backup)), len(patches)


def hot_swap_transition(
    store,
    stored,
    slot: str,
    unet_name: str,
    lora_stack: list[dict] | None,
    base_key: tuple,
    requested_identity: dict,
    *,
    request_artifact_identity,
    logger,
) -> tuple[Any, str] | None:
    """Clone the pristine base, re-apply the lora stack, and adopt the result.

    Returns ``None`` when the swap would first copy weights off the GPU
    (``copies_off_the_gpu``): the caller then loads base and stack fresh.
    """
    from . import in_place_fallback
    from .model_store import _is_quantized_kind, lora_signature

    t0 = time.perf_counter()
    active = store._build_active(stored.base_patcher, lora_stack)
    t_build = time.perf_counter() - t0
    uncovered, patched = copies_off_the_gpu(store, stored, slot, active)
    if uncovered:
        logger.info(
            "model store [%s]: a LoRA change on %s would copy the model off the GPU first "
            "(no host copy of %d of the %d weights the new stack patches), so it is not "
            "hot-swapped: base and stack load fresh, or bake in place", slot, unet_name, uncovered, patched)
        # The store holds what this build read (the patches, not the model) so the fresh load's walls price a fact.
        in_place_fallback.note_stack_read(store, slot, lora_stack, getattr(active, "patches", None))
        return None
    store_identity.assert_stable_identity(
        request_artifact_identity(unet_name, lora_stack), requested_identity,
        "applying a hot-swap")
    # `stored.quant_kind` is the detected kind, not the one the request's
    # options imply: a scaled-fp8 checkpoint loads under default options and
    # still pays the conversion spike. FP16/FP32 are distinct full-precision
    # kinds for the FSDP safety gate, not quantized formats, so they keep the
    # ordinary hot-swap path.
    effective_quant = stored.quant_kind
    # A change between stacks that patch no weight (a LoRA whose keys match
    # nothing in this model) leaves the placed weights exactly right, so the
    # global unload, which copies the whole model to the host, is skipped.
    weights_change = not (_patches_nothing(active) and _patches_nothing(stored.active_patcher))
    note = ""
    if _is_quantized_kind(effective_quant) and not weights_change:
        note = " (quant: no weight changes, GPU weights kept)"
    elif _is_quantized_kind(effective_quant):
        # The patched quantized weights rebuild from ComfyUI's host backup at
        # the next load. Where the resident detaches alone, only those keys
        # move; elsewhere every model is unloaded, the other slot included.
        t1 = time.perf_counter()
        if _unpatch_quantized_in_place(store, stored, slot):
            note = (f" (quant: patched weights restored in place, no host copy; build {t_build:.1f}s, "
                    f"unpatch {time.perf_counter() - t1:.1f}s)")
        else:
            store._invalidate_gpu_weights(stored.active_patcher)
            store._mark_other_slot_evicted(slot)  # the global unload hit the other slot
            note = (f" (quant: GPU weights invalidated; build {t_build:.1f}s, "
                    f"unload {time.perf_counter() - t1:.1f}s)")
    stored.active_patcher = active
    stored.lora_sig = lora_signature(lora_stack)
    stored.request_key = (*base_key, stored.lora_sig, effective_quant)
    stored.artifact_identity = requested_identity
    logger.info(
        "model store [%s]: %s -- %s loras=%s in %.1fs%s",
        slot, TRANSITION_HOT_SWAP, unet_name, stored.lora_sig,
        time.perf_counter() - t0, note,
    )
    return active, TRANSITION_HOT_SWAP


def _patches_nothing(patcher) -> bool:
    """Whether this patcher's stack patches no weight; unknown reads as patching."""
    patches = getattr(patcher, "patches", None)
    return isinstance(patches, dict) and not patches
