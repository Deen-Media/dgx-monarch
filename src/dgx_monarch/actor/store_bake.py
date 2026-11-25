"""Weight bake, verification, and swap transitions for ModelStore.

For a LoRA stack change on a resident base, ModelStore.ensure calls one of the
two transition functions at the end; it reloads from disk when neither applies
or the lazy swap fails recoverably. Each reaches the store's methods through
the store it is handed, and model_store's module names through a call-time
import, so both resolve when the transition runs rather than when this module
loads. The classic hot swap lives in ``store_hot_swap`` and a slab resident's
settlement in ``store_slab_settle``; their public names are importable here.
"""
from __future__ import annotations

import time
from typing import Any

from ..constants import TRANSITION_HOT_SWAP
from ..log import get_logger
from ..transfer_utils import (
    failure_summary,
    raise_with_distinct_cause,
    reconcile_error,
    safe_call,
    safe_note,
)
from . import slab_lifetime, store_identity
from .store_hot_swap import (
    _placed_bytes,
    copies_off_the_gpu,
    hot_swap_transition,
    invalidate_gpu_weights,
    swaps_in_place,
    unpatch_in_place,
)
from .store_slab_settle import SwapSettle, slab_misfits

__all__ = ["SwapSettle", "_placed_bytes", "audit_swap_dtypes", "bake_key", "copies_off_the_gpu",
           "hot_swap_transition", "invalidate_gpu_weights", "lazy_swap", "lazy_swap_transition",
           "slab_misfits", "swaps_in_place", "unpatch_in_place", "verify_key"]

log = get_logger(__name__)


def _merge_and_free(active, base_path: str | None = None):
    """Bake the LoRA stack, then release ComfyUI's weight backup and patches.

    This reclaims unified memory without render-time work. When ``base_path``
    supports exact restoration, capture pristine checkpoint references before
    baking so later stack changes can restore in place. ComfyUI must still own
    the initial load and backup: bypassing that backup let a pressure-driven
    unpatch overwrite a live weight. See docs/VALIDATION.md for the restore
    contract. Returns the un-bake record, or ``None`` when later
    changes must reload from disk.
    """
    import gc

    import comfy.model_management as mm

    record = None
    t0 = time.perf_counter()
    if base_path is not None:
        from .unbake import capture_unbake_record

        try:
            record = capture_unbake_record(active.model, list(active.patches), base_path)
        except Exception as exc:
            record = None
            safe_call(
                log.warning,
                "unbake capture skipped (%s); stack changes will reload from disk",
                failure_summary(exc),
            )
    t_capture = time.perf_counter() - t0
    t0 = time.perf_counter()
    mm.load_models_gpu([active], force_full_load=True)   # bakes the LoRAs into the GPU weights
    t_bake = time.perf_counter() - t0
    active.backup.clear()
    active.backup_buffers.clear()
    active.patches.clear()
    gc.collect()
    log.info("merge-and-free: capture %.1fs, bake %.1fs", t_capture, t_bake)
    return record


def lazy_swap(store, stored, active, unet_name: str):
    """Fused restore + re-bake for a low_rss stack change.

    Stream pristine keys from disk while each restored key bakes on the GPU.
    The swap avoids ComfyUI's load manager so it cannot unpatch, evict, or
    double-count during the bake. Reuse verified record entries and capture
    only new keys before baking. Each key goes back to the resident's own
    storage as the swap finishes with it (``store_slab_settle.SwapSettle``).
    Return the new un-bake record, or ``None`` when complete coverage cannot be
    proven; the next stack change then reloads from disk.
    """
    settle = SwapSettle(stored)
    record = _restore_and_bake(store, stored, active, unet_name, settle)
    settle.finish()
    return record


def _restore_and_bake(store, stored, active, unet_name: str, settle: SwapSettle):
    import random

    from .unbake import UnbakeRecord, capture_unbake_record, restore_pristine

    old = stored.unbake
    patch_keys = list(active.patches)
    baked: set[str] = set()
    n_verify = len(patch_keys) if store.swap_verify < 0 else min(store.swap_verify, len(patch_keys))
    verify_keys = set(random.sample(patch_keys, n_verify)) if n_verify else set()

    def bake_one(key: str) -> None:
        if key in active.patches:
            if key in verify_keys:
                store._verify_key(active, key)
            else:
                store._bake_key(active, key)
            baked.add(key)
        settle(key)

    if old is not None:
        restore_pristine(stored.base_patcher.model, old, after_key=bake_one)
    if not patch_keys:
        active.patches.clear()
        return None
    covered = (set(old.mapped) | set(old.quant) | set(old.resident)) if old else set()
    novel = [k for k in patch_keys if k not in covered]
    fresh = None
    if novel:
        # New keys are pristine here and must be captured before baking.
        # Resolve through model_store to retain its injectable path seam.
        from . import model_store as _ms

        try:
            fresh = capture_unbake_record(
                active.model, novel, _ms.resolve_model_path("diffusion_models", unet_name))
        except Exception as exc:
            safe_call(
                log.warning,
                "unbake capture skipped (%s); the next stack change will reload "
                "from disk",
                failure_summary(exc),
            )
    for key in patch_keys:
        if key not in baked:
            bake_one(key)
    if verify_keys:
        log.info("ambient verify: %d/%d keys bit-matched comfy's own bake (%s)",
                 len(verify_keys), len(patch_keys), ", ".join(sorted(verify_keys)))
        from ..telemetry import emit

        emit("verify", checked=len(verify_keys), of=len(patch_keys))
    active.patches.clear()
    store._audit_swap_dtypes(stored.base_patcher.model, old, patch_keys)
    if novel and fresh is None:
        return None
    source = fresh if fresh is not None else old
    record = UnbakeRecord(path=source.path, file_size=source.file_size,
                          file_mtime_ns=source.file_mtime_ns,
                          file_dev=source.file_dev, file_ino=source.file_ino,
                          file_ctime_ns=source.file_ctime_ns)
    for key in patch_keys:
        for rec in (old, fresh):
            if rec is None:
                continue
            if key in rec.mapped:
                record.mapped[key] = rec.mapped[key]
                break
            if key in rec.quant:
                record.quant[key] = rec.quant[key]
                break
            if key in rec.resident:
                record.resident[key] = rec.resident[key]
                break
        else:
            return None  # no record covers this patched key
    return record


def audit_swap_dtypes(model, record, patch_keys) -> None:
    """Require each restored key to have its recorded live dtype.

    Refusing here routes drift through slot discard and clean reload instead of
    allowing a mid-render dtype failure.
    """
    from .unbake import UnbakeError, live_tensor

    if record is None:
        return
    for key, ft in record.mapped.items():
        expected = ft.cast_to or ft.dtype
        try:
            actual = live_tensor(model, key).dtype
        except AttributeError:
            continue
        if actual != expected:
            from ..telemetry import emit

            emit("audit_fail", key=key, actual=str(actual), expected=str(expected))
            raise UnbakeError(
                f"post-swap dtype audit: {key} is {actual}, expected {expected} "
                f"(cast_to={ft.cast_to}, patched={key in set(patch_keys)}); "
                "dropping the slot for a clean reload")


def verify_key(store, active, key: str) -> None:
    """Bake one key through both paths and require bit identity.

    ``patch_weight_to_device(..., return_weight=True)`` computes a reference
    without writing. Both paths use the same key-seeded rounding, so mismatch
    triggers slot discard and clean reload.
    """
    import torch

    from .unbake import UnbakeError, live_tensor

    # Compare at ``load_device`` so a residence-dependent compute dtype cannot
    # look like semantic divergence.
    device = getattr(active, "load_device", None) or live_tensor(active.model, key).device
    ref = active.patch_weight_to_device(key, device_to=device, return_weight=True)
    store._bake_key(active, key)
    ours = live_tensor(active.model, key)
    ref_q, ours_q = hasattr(ref, "dequantize"), hasattr(ours, "dequantize")
    if ref_q and ours_q:
        ref_d, ours_d = ref.dequantize(), ours.dequantize()
        same = torch.equal(ref_d.to(ours_d.device), ours_d)
    elif ref_q or ours_q:
        same = False  # Quantization on only one side is structural drift.
    else:
        same = ref.dtype == ours.dtype and torch.equal(ref.to(ours.device), ours)
    del ref
    if not same:
        store.verify_failures += 1
        raise UnbakeError(
            f"ambient bake verify MISMATCH on {key}: this swap's bake does not "
            "bit-match comfy's own computation. The worker drops the slot and "
            "reloads the model through comfy's stock path, so the output stays "
            "correct. This usually means a ComfyUI update changed bake semantics; "
            "report it with your ComfyUI commit."
        )


def bake_key(active, key: str) -> None:
    """Bake one key without invoking ComfyUI's load or backup manager.

    The caller supplies a pristine weight. A key stock patches at cast time
    takes stock's cast-time weight (actor/lora_cast_time.py, #145); any other
    key takes ComfyUI's classic bake: the patcher's public conversion and set
    functions, its LoRA compute dtype, and key-seeded rounding. Plain weights
    update in place to preserve tensor identity; quantized weights replace
    their wrapper parameter. Ambient verification (verify_key) checks a sample
    of keys per swap, two by default, against the reference bake.
    """
    import comfy.float
    import comfy.lora
    import comfy.model_management as mm
    import comfy.model_patcher as comfy_mp
    import comfy.utils

    from . import lora_cast_time

    patches = active.patches[key]
    weight, set_func, convert_func = comfy_mp.get_key_weight(active.model, key)
    # Pin bake math to ``load_device``. Using a temporarily offloaded weight's
    # device changes the compute dtype and caused deterministic image drift.
    bake_device = getattr(active, "load_device", None) or weight.device
    cast = lora_cast_time.cast_time_weight(active, key, bake_device)
    if cast is not None:
        lora_cast_time.store(active.model, key, cast, in_place=True)
        return
    temp = mm.cast_to_device(
        weight, bake_device, mm.lora_compute_dtype(bake_device), copy=True)
    if convert_func is not None:
        temp = convert_func(temp, inplace=True)
    out = comfy.lora.calculate_weight(patches, temp, key)
    if set_func is None:
        out = comfy.float.stochastic_rounding(
            out, weight.dtype, seed=comfy.utils.string_to_seed(key))
        comfy.utils.copy_to_param(active.model, key, out)
    else:
        set_func(out, inplace_update=False, seed=comfy.utils.string_to_seed(key))
    del temp, out


def _adopt_patch_uuid(stored, active) -> None:
    """Record the active patch UUID after an in-place stock LoRA swap.

    The weights now contain ``active``'s patches. Leaving the old UUID would
    make ComfyUI unpatch the model to the host and reload it on the next load.
    Slab and FSDP residents instead pin their offload device and remain unchanged.
    """
    if getattr(stored, "slab", None) is not None or getattr(stored.base_patcher, "_dgxm_fsdp", False):
        return
    model, uuid = getattr(active, "model", None), getattr(active, "patches_uuid", None)
    if model is not None and uuid is not None and hasattr(model, "current_weight_patches_uuid"):
        model.current_weight_patches_uuid = uuid


def _empty_record(previous, unet_name: str):
    """An un-bake record with no keys, bound to the checkpoint the resident came from.

    The bound is the previous record's file identity when there is one, else
    the file's own header identity now; restore checks it before it restores
    nothing, so a changed file still reloads. ``None`` when the file cannot be
    read, which keeps the reload contract.
    """
    from .unbake import UnbakeRecord, read_safetensors_header

    if previous is not None:
        return UnbakeRecord(path=previous.path, file_size=previous.file_size,
                            file_mtime_ns=previous.file_mtime_ns, file_dev=previous.file_dev,
                            file_ino=previous.file_ino, file_ctime_ns=previous.file_ctime_ns)
    from . import model_store as _ms

    try:
        path = _ms.resolve_model_path("diffusion_models", unet_name)
        _header, _metadata, parsed = read_safetensors_header(
            path, want_metadata=True, want_identity=True)
    except Exception as exc:
        safe_call(log.info, "no un-bake record for a stack that patches nothing (%s); "
                  "the next stack change reloads from disk", failure_summary(exc))
        return None
    return UnbakeRecord(path=path, file_size=parsed.file_size, file_mtime_ns=parsed.file_mtime_ns,
                        file_dev=parsed.file_dev, file_ino=parsed.file_ino,
                        file_ctime_ns=parsed.file_ctime_ns)


def lazy_swap_transition(
    store,
    slot: str,
    unet_name: str,
    lora_stack: list[dict] | None,
    base_key: tuple,
    requested_identity: dict,
    *,
    request_artifact_identity,
    logger,
) -> tuple[Any, str] | None:
    """Swap the lora stack in place on the resident `slot` holds.

    The resident is read off the store rather than passed in: the recovery path
    drops the slot, and a caller-side local would outlive that drop and keep the
    outgoing model alive across it. Returns (patcher, transition) on success,
    and ``None`` when the swap failed recoverably and the caller must fall
    through to a fresh load.
    """
    from ..adapters.base import UnsupportedModelError
    from .model_store import _best_effort, lora_signature
    from .unbake import UnbakeRefused

    stored = store._stored(slot)
    t0 = time.perf_counter()
    lazy_failed: BaseException | None = None
    # Publish refusal before restore/bake can mutate the resident. Any
    # interruption in recovery then leaves this exact slot blocked.
    store._drop_failed.add(slot)
    try:
        t1 = time.perf_counter()
        active = store._build_active(stored.base_patcher, lora_stack)
        t_build = time.perf_counter() - t1
        misfits = slab_misfits(stored, active)
        if misfits:
            # Nothing is written yet: the resident is intact, and the fresh
            # load that follows drops it and prices its own bake.
            store._drop_failed.discard(slot)
            _best_effort(
                logger.info, "model store [%s]: %d key(s) of this stack would bake to a payload "
                "their slab regions cannot hold (%s), so %s loads fresh and its bake is priced "
                "whole", slot, len(misfits), ", ".join(misfits[:4]), unet_name)
            return None
        t1 = time.perf_counter()
        previous = stored.unbake
        patched = len(getattr(active, "patches", None) or ())  # the swap clears them
        record = store._lazy_swap(stored, active, unet_name)
        if record is None and lora_stack and not patched:
            # A stack that patches no weight (its keys match nothing in this
            # model) leaves the base pristine. An empty record keeps the next
            # change lazy; without one that change would reload from disk.
            record = _empty_record(previous, unet_name)
        t_swap = time.perf_counter() - t1
        store_identity.assert_stable_identity(
            request_artifact_identity(unet_name, lora_stack), requested_identity,
            "applying a lazy hot-swap")
    except (UnsupportedModelError, UnbakeRefused):
        # A typed launch refusal, or a stack refused before any weight was
        # written: the resident still holds its old stack intact, and a fresh
        # load would only meet the same refusal after dropping it.
        store._drop_failed.discard(slot)
        raise
    except BaseException as exc:
        stored = None
        active = None
        lazy_failed = exc
    else:
        stored.active_patcher = active
        stored.unbake = record
        _adopt_patch_uuid(stored, active)
        # The swap re-baked into these weights, or restored them
        # pristine when the new stack is empty.
        stored.base_baked = stored.baked_in_place = bool(lora_stack)
        stored.lora_sig = lora_signature(lora_stack)
        stored.request_key = (*base_key, stored.lora_sig, stored.quant_kind)
        stored.artifact_identity = requested_identity
        store._drop_failed.discard(slot)
        _best_effort(
            logger.info, "model store [%s]: %s (lazy un-bake) -- %s loras=%s "
            "in %.1fs (lora build %.1fs, restore+bake pipeline %.1fs)", slot,
            TRANSITION_HOT_SWAP, unet_name, stored.lora_sig,
            time.perf_counter() - t0, t_build, t_swap)
        return active, TRANSITION_HOT_SWAP
    if lazy_failed is not None:
        _best_effort(
            logger.warning, "model store [%s]: lazy swap failed (%r); dropping "
            "the slot for a clean reload", slot, lazy_failed)
        _best_effort(slab_lifetime._clear_exception_frames, lazy_failed)
        _best_effort(setattr, lazy_failed, "__traceback__", None)
        try:
            store._drop(slot)
        except BaseException as discard_exc:
            store._drop_failed.add(slot)
            evidence = failure_summary(discard_exc)
            _best_effort(
                logger.warning, "model store [%s]: lazy-swap discard also failed "
                "(%s); slot poisoned", slot, evidence)
            safe_note(
                lazy_failed,
                "lazy-swap resident discard also failed; slot poisoned",
                evidence,
            )
            strongest, cause = reconcile_error(
                lazy_failed,
                discard_exc,
                "lazy-swap resident discard also failed",
            )
            if strongest is lazy_failed:
                _best_effort(
                    slab_lifetime._clear_exception_frames,
                    discard_exc,
                )
                _best_effort(setattr, discard_exc, "__traceback__", None)
            raise_with_distinct_cause(strongest, cause)
        store._drop_failed.discard(slot)
        if not isinstance(lazy_failed, Exception):
            raise lazy_failed from None
    return None
