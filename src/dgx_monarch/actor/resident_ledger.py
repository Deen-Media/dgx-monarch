"""Record resident slab, FSDP, and stock weights through ComfyUI's loader.

Slab and FSDP builds place weights directly, leaving ComfyUI's
``model_loaded_weight_memory`` and ``model.device`` at their empty values.
At sample time, ComfyUI can then request free memory for weights already
resident, choose partial loading, and trigger the cross-rank divergence guard.

Call ``partially_load(load_device, 1e32)`` when the weights become resident,
using the same full-load path as ``LoadedModel.model_load``. A repeated call
returns early without moving weights. ``store_load`` also declares eligible
stock residents here.

Using the loader preserves its device walk, per-module patched-weight flags,
weight functions, and hook handling. Patchers with pending weight or wrapper
patches are excluded to avoid baking a model-sized backup during construction.
Slab and FSDP LoRA builds require low-RSS mode and clear their patches before
reaching this point. The pack registers no ``ON_LOAD`` callbacks to move earlier.

The declaration log and ComfyUI's ``loaded completely`` line provide build-
time evidence; the sample-time partial-load guard confirms residency. ComfyUI
can still partially unload weights if the render working set does not fit.
This declaration also does not reduce the identity gate's model-copy peak.
"""
from __future__ import annotations

import gc
import time
from typing import Any

from .. import mesh_safety
from ..log import get_logger
from ..transfer_utils import failure_summary

log = get_logger(__name__)

# ``DGXM_DISABLE_COPY_FREE_DROP=1`` on a Worker: every drop goes back to
# ComfyUI's global unload, and the swap credit's guard for a partly loaded
# resident and a held other slot stands again (``drop_copies_nothing``).
COPY_FREE_DROP_DISABLE_ENV = "DGXM_DISABLE_COPY_FREE_DROP"

# ComfyUI's own spelling of "load all of it": LoadedModel.model_load turns a
# lowvram budget of 0 into this before it calls partially_load.
FULL_LOAD_EXTRA = 1e32
# A host trim after a drop is said only where it returned this much.
_TRIM_WORTH_A_LINE = 256 * 2**20


def skip_reason(patcher: Any) -> str | None:
    """Why this patcher must not be full-loaded here, or None when it may be."""
    model = getattr(patcher, "model", None)
    if model is None:
        return "the patcher holds no model"
    if getattr(patcher, "load_device", None) is None:
        return "the patcher names no load device"
    if getattr(patcher, "patches", None):
        return "weight patches are still pending"
    if getattr(patcher, "weight_wrapper_patches", None):
        return "weight wrapper patches are still pending"
    if getattr(model, "model_lowvram", False):
        return "comfy already holds part of these weights"
    if int(getattr(model, "model_loaded_weight_memory", 0) or 0) > 0:
        return "comfy's ledger already counts these weights"
    return None


def declare(patcher: Any, residency: str) -> str | None:
    """Full-load `patcher` through comfy so its ledger names the resident bytes.

    Returns ``None`` when the ledger was written, and the reason it was not
    otherwise. The reason is a journal line either way: a rank that reads
    `loaded partially` at sample time is read against this line.
    """
    reason = skip_reason(patcher)
    if reason is not None:
        log.info("resident ledger: %s weights left undeclared (%s)", residency, reason)
        return reason
    model = patcher.model
    t0 = time.perf_counter()
    declared = int(patcher.partially_load(patcher.load_device, FULL_LOAD_EXTRA) or 0)
    log.info(
        "resident ledger: declared %.2f GiB of %s weights to comfy in %.1fs; "
        "comfy now reads %.2f GiB loaded, full_load=%s, so its sample-time load "
        "prices the working set instead of the whole model",
        declared / 2**30, residency, time.perf_counter() - t0,
        int(getattr(model, "model_loaded_weight_memory", 0) or 0) / 2**30,
        not bool(getattr(model, "model_lowvram", False)),
    )
    return None


def detaches_alone(stored: Any) -> bool:
    """Whether a drop of this resident leaves every other model ComfyUI holds loaded.

    A plain stock resident on unified memory does, unless
    ``DGXM_DISABLE_COPY_FREE_DROP=1``. On a discrete GPU the global unload
    stays: an offload there frees device memory the next load may need. A
    slab and a shard build keep it too: their owners close only after every
    model that could reference their memory is detached.
    """
    return (not mesh_safety.env_enabled(COPY_FREE_DROP_DISABLE_ENV)
            and mesh_safety.gpu_is_integrated()
            and getattr(stored, "slab", None) is None
            and getattr(stored, "fsdp_checkpoint_pin", None) is None
            and not getattr(getattr(stored, "base_patcher", None), "_dgxm_fsdp", False))


def stays_on_device(stored: Any) -> bool:
    """Whether a held slot can be pinned so that no later unload copies it to the host.

    Asked only on unified memory (``detaches_alone`` is the gate), where an
    offload is a copy inside one pool that frees nothing. ``None``, an empty
    slot, has nothing to copy. A held one qualifies when ComfyUI holds it
    whole and the store placed it: a stock resident, or a slab or shard base,
    which is pinned at load already. A partly loaded model would have its host
    part copied to the device by the pin, and a comfy-managed one is ComfyUI's
    to move.
    """
    if stored is None:
        return True
    from . import store_residency

    model = getattr(getattr(stored, "base_patcher", None), "model", None)
    return (model is not None and not getattr(model, "model_lowvram", False)
            and store_residency.describe(stored)["residency"] in ("stock", "slab"))


def drop_copies_nothing(stored: Any, other: Any) -> bool:
    """Whether dropping ``stored`` beside ``other``, the slot that stays, moves no weight to the host.

    One answer for the drop and for the price. ``unload_dropped`` does what
    this says, and the quote endpoint reports it, so the swap credit counts a
    drop only where it is true (``capacity_lora_bake.credit_stock_swap``):

    - the dropped model is detached in place, whole or partly loaded;
    - the slot that stays is not touched by the drop;
    - and it is pinned to its device, so the unload ComfyUI runs at the next
      load that does not use it moves nothing either. On a Worker smart memory
      is off, and ``load_models_gpu`` then unloads every other loaded model.
    """
    return detaches_alone(stored) and stays_on_device(other)


def unload_dropped(mm: Any, stored: Any, other: Any, everything: bool = False) -> bool:
    """Unload the slot the store is dropping. True when the other slot was left loaded.

    ``everything`` is the store-wide unload (``ModelStore.unload_all``), which
    must leave ComfyUI holding no model at all and so keeps the global unload.
    Otherwise only the dropped model's records leave ComfyUI's loaded list; a
    model both slots share stays.

    - A model ComfyUI holds whole is pinned to its load device and unloaded by
      ComfyUI's own ``model_unload``, so its detach moves nothing.
    - A partly loaded model cannot be pinned (the pin would copy its host part
      to the device), and ComfyUI's detach would copy its device part to the
      host. Both are waste for weights about to be thrown away. It leaves the
      list by the step ComfyUI itself takes when it replaces a clone in
      ``load_models_gpu``: ``detach(unpatch_all=False)`` and the finalizer's
      detach, which move no weight. Its pinned-memory records and its backup
      are released here, since no unpatch runs.

    ``other``, the slot that stays, is pinned where ``stays_on_device`` allows.
    A resident that does not detach alone takes ``unload_without_offload``.
    """
    patchers = (stored.base_patcher, stored.active_patcher)
    if everything or not detaches_alone(stored):
        unload_without_offload(mm, patchers)
        return False
    kept = tuple(patcher for patcher in (getattr(other, "base_patcher", None),
                                         getattr(other, "active_patcher", None)) if patcher is not None)
    if kept and stays_on_device(other):
        pin_to_load_device(mm, kept)
    pin_to_load_device(mm, patchers)
    shared = {id(getattr(patcher, "model", None)) for patcher in kept}
    dropped = {id(patcher.model) for patcher in patchers
               if getattr(patcher, "model", None) is not None} - shared
    loaded = getattr(mm, "current_loaded_models", [])
    for index in range(len(loaded) - 1, -1, -1):
        entry = loaded[index]
        patcher: Any = getattr(entry, "model", None)
        model = getattr(patcher, "model", None)
        if model is None or id(model) not in dropped:
            continue
        if not getattr(model, "model_lowvram", False):
            if entry.model_unload():
                loaded.pop(index)
            continue
        loaded.pop(index)
        patcher.offload_device = patcher.load_device  # its option patches stay where they are
        patcher.detach(unpatch_all=False)
        unpin = getattr(patcher, "unpin_all_weights", None)
        if callable(unpin):
            unpin()
        backup = getattr(patcher, "backup", None)
        if isinstance(backup, dict):
            backup.clear()
        finalizer = getattr(entry, "model_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
        entry.model_finalizer = entry.real_model = None
    mm.soft_empty_cache()
    return True


def mark_after_drop(store: Any, slot: str, alone: bool) -> None:
    """Record what a finished drop of ``slot`` did to the other slot.

    A drop that detached alone left it loaded, so only ``slot``'s own mark is
    cleared. The global unload took the other slot's GPU weights with it.
    """
    if alone:
        store._gpu_evicted.discard(slot)
    else:
        store._mark_other_slot_evicted(slot)


def unload_without_offload(mm: Any, patchers: tuple) -> None:
    """Unload discarded models through ComfyUI's global unload, without a host copy of a whole load.

    The drop of a slab or a shard build, and every drop under
    ``DGXM_DISABLE_COPY_FREE_DROP=1``. ComfyUI detaches a clone by moving its
    weights to its offload device. For fully loaded models being discarded,
    first set every clone's offload device to its load device so detach moves
    nothing. Partially loaded models keep the default offload path here,
    because pinning them would first copy their remaining host weights to the
    device, and every other loaded model is unloaded with them.
    """
    pin_to_load_device(mm, patchers)
    mm.unload_all_models()
    mm.soft_empty_cache()


def pin_to_load_device(mm: Any, patchers: tuple) -> None:
    """Point every loaded clone of these models at its load device."""
    # A partially loaded model (model_lowvram) already holds some weights on the
    # host; pinning would copy those to the device just to discard them, so it
    # keeps comfy's default and only a fully loaded model is pinned.
    models = {id(p.model) for p in patchers
              if getattr(p, "model", None) is not None and not getattr(p.model, "model_lowvram", False)}
    loaded = [getattr(entry, "model", None) for entry in list(getattr(mm, "current_loaded_models", ()))]
    for patcher in (*patchers, *loaded):
        if (patcher is not None and id(getattr(patcher, "model", None)) in models
                and getattr(patcher, "load_device", None) is not None):
            patcher.offload_device = patcher.load_device


def pin_store_for_unload(*stored: Any) -> None:
    """Pin all slots before a store-wide unload starts.

    ComfyUI unloads globally. Pinning one slot at a time would let the first
    drop copy the other slot's model to host memory before it is pinned.
    """
    patchers = tuple(patcher for slot in stored if slot is not None
                     for patcher in (slot.base_patcher, slot.active_patcher) if patcher is not None)
    if patchers:
        import comfy.model_management as mm

        pin_to_load_device(mm, patchers)


def collect_dropped() -> None:
    """Collect the dropped model once the store has cleared its references."""
    gc.collect()


def flush_dropped(mm: Any, slot: str) -> None:
    """Return a dropped model's cached device blocks, and its freed host memory, to the box, best effort.

    The unload (``unload_dropped``) flushes while the store still holds the weights.
    After those references are cleared, flush again so ``MemAvailable`` and the
    next capacity quote reflect the released memory. The model is already
    dropped; a flush failure only leaves capacity quotes pessimistic.

    Then one host trim (``host_trim``): from a Worker's second load of a file
    on, the C library keeps the freed pages of its smaller weights, and a
    dropped model's memory would not reach the box (on record a hard clear
    left a Worker at 10.35 GiB with no model, and a load that fit was then
    refused 0.5 GiB short). Every drop of a slot ends here, and the Worker's
    hard clear is a drop of every slot. A reuse makes no drop, so this is no
    step of an ordinary render.
    """
    try:
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("model store [%s]: cache flush after the drop failed (%s); capacity "
                    "quotes may count the dropped weights until the next flush", slot, failure_summary(exc))
    from . import host_trim

    ran, seconds, before, after = host_trim.trim()
    if ran and after - before >= _TRIM_WORTH_A_LINE:
        log.info("model store [%s]: the host trim after the drop returned %.1f GiB to the box "
                 "(MemAvailable %.1f to %.1f GiB, %.2f s)", slot, (after - before) / 2**30,
                 before / 2**30, after / 2**30, seconds)
