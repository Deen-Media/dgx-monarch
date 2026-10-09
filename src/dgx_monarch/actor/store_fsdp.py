"""Topology-scoped FSDP and managed-residency checks for model loads.

Every load path crosses this module before ModelStore hot-swap returns. That
placement keeps ComfyUI's memory manager out of LoRA swap paths and binds
accuracy refusals to the checkpoint and dispatch that triggered them.
"""
from __future__ import annotations

from .. import residency_mode, upstream_gate
from ..adapters.base import UnsupportedModelError
from ..refusal import RefusalClass, refusal
from ..residency_mode import ComfyManagedResidencyError


def active(worker) -> bool:
    return bool(getattr(worker, "topology", {}).get("fsdp"))


def _assert_artifact_runnable(args, kwargs) -> None:
    """Refuse an artifact this release carries no forward for, before any load.

    The check lives in this funnel, not in ``store_load.fresh_load``: there
    ``store._drop(slot)`` has already evicted the resident model by the time a
    refusal could fire, and the baked hot-swap arm never reaches it at all
    (``tests/test_comfy_managed_residency.py::test_the_regression_that_made_the_yield_move``).
    The raise is rank symmetric because the predicate is a pure function of the
    file every rank opens.
    """
    unet_name = args[0] if args else kwargs.get("unet_name", "")
    try:
        from .comfy_bridge import resolve_model_path

        path = resolve_model_path("diffusion_models", unet_name)
    except Exception:  # unresolvable here: the load path's own error stands
        return
    if not upstream_gate.refuses(path):
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS, upstream_gate.ZIMAGE_L2P_REFUSAL,
        troubleshooting=upstream_gate.TROUBLESHOOTING))


def _assert_comfy_managed_supported(worker, args, kwargs) -> None:
    """Under comfy-managed residency, refuse a LoRA stack or FSDP before ComfyUI runs.

    Check LoRA first because its load-path hazard is more specific when both
    conditions apply.
    """
    if not residency_mode.active():
        return
    lora_stack = args[2] if len(args) > 2 else kwargs.get("lora_stack")
    count = len(lora_stack or [])
    if count:
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS,
            f"comfy-managed residency is on for this worker and this render carries "
            f"{count} LoRA(s). dgx-monarch keeps ComfyUI's memory manager out of every "
            "LoRA swap path: under pool pressure that manager un-patched a model mid-load "
            "and wrote a sentinel over a real weight (docs/VALIDATION.md, 2026-07-07), "
            "and comfy's ModelPatcherDynamic replaces exactly the patch, unpatch, load "
            "and backup methods that path depends on. No LoRA render has been proven "
            "under it on this hardware. Nothing was loaded and nothing was quarantined. "
            "What works instead: turn the Init node's comfy_managed widget off for "
            "graphs that use LoRAs and restart the Worker service (`dgxm restart`), "
            "which gives this box "
            "slab residency and the low-RSS lazy un-bake again, or render this graph "
            "without a LoRA stack.",
            troubleshooting=residency_mode.TROUBLESHOOTING))
    if active(worker):
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS, residency_mode.COMFY_MANAGED_FSDP_REFUSAL,
            troubleshooting=residency_mode.TROUBLESHOOTING))


def ensure(worker, *args, **kwargs):
    """Refuse, bind accuracy context, then load through ``ModelStore.ensure``.

    An active FSDP topology adds the strict local-header proof. Bind accuracy
    context only for the conditional checkpoint: a dual-model request loads its
    unconditional checkpoint second, but render guards apply to the conditional
    model. Rescue consent identifies the sample dispatch; other load paths
    disarm waivers so grants cannot survive thread reuse.
    """
    _assert_artifact_runnable(args, kwargs)
    from .. import accuracy_waiver

    _assert_comfy_managed_supported(worker, args, kwargs)
    if kwargs.get("slot", "cond") == "cond":
        accuracy_waiver.bind_model(
            args[0] if args else kwargs.get("unet_name", ""),
            args[1] if len(args) > 1 else kwargs.get("options"),
            args[2] if len(args) > 2 else kwargs.get("lora_stack"),
            getattr(worker, "topology", {}), getattr(worker, "world", 0),
            dispatch="rescue_consent" in kwargs)
    if active(worker):
        kwargs["fsdp_launch"] = True
    return worker.store.ensure(*args, **kwargs)


def validate_injection(
    worker,
    quant_kind: str,
    lora_stack,
) -> None:
    """Run cheap launch refusals before adapter/compile preparation."""
    if not active(worker):
        return
    from ..adapters.fsdp import (
        validate_fsdp_launch_loras,
        validate_fsdp_launch_quant,
    )

    validate_fsdp_launch_quant(quant_kind)
    store = getattr(worker, "store", None)
    low_rss = getattr(store, "lora_low_rss", None) if store is not None else None
    validate_fsdp_launch_loras(
        lora_stack, lora_low_rss=None if low_rss is None else bool(low_rss))
