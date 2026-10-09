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

from ..log import get_logger
from ..transfer_utils import failure_summary

log = get_logger(__name__)

# ComfyUI's own spelling of "load all of it": LoadedModel.model_load turns a
# lowvram budget of 0 into this before it calls partially_load.
FULL_LOAD_EXTRA = 1e32


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


def unload_without_offload(mm: Any, patchers: tuple) -> None:
    """Unload discarded models without creating a full host copy.

    ComfyUI detaches a clone by moving its weights to its offload device. For
    fully loaded models being discarded, first set every clone's offload device
    to its load device so detach moves nothing. This avoids a slow, potentially
    out-of-memory copy. Partially loaded models keep the default offload path
    because pinning them would first copy their remaining host weights to the
    device.
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
    """Return a dropped model's cached device blocks to the host, best effort.

    ``unload_without_offload`` flushes while the store still holds the weights.
    After those references are cleared, flush again so ``MemAvailable`` and the
    next capacity quote reflect the released memory. The model is already
    dropped; a flush failure only leaves capacity quotes pessimistic.
    """
    try:
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("model store [%s]: cache flush after the drop failed (%s); capacity "
                    "quotes may count the dropped weights until the next flush", slot, failure_summary(exc))
