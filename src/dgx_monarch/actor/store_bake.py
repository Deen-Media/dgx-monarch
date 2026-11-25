"""Weight bake, verification, and swap transitions for ModelStore.

For a LoRA stack change on a resident base, ModelStore.ensure calls one of the
two transition functions at the end; it reloads from disk when neither applies
or the lazy swap fails recoverably. Each reaches the store's methods through
the store it is handed, and model_store's module names through a call-time
import, so both resolve when the transition runs rather than when this module
loads.
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


_DIFFUSION_PREFIX = "diffusion_model."


class _SwapSettle:
    """Return each key a lazy swap touched to where the resident keeps its weights.

    A restore rebuilds a quantized wrapper from a fresh copy and ComfyUI's bake
    replaces it out of place, so on a slab resident every patched quantized
    key would otherwise sit twice in unified memory: once in its idle slab
    region and once in cudaMalloc, while every capacity figure counts the slab
    once. Settling the key the moment the swap is done with it (restored, then
    baked when the new stack patches it) puts it back in its own region, so at
    most the key in hand has a second copy. Plain keys restore and bake in
    place and stay where they are.
    """

    def __init__(self, stored):
        self.slab = getattr(stored, "slab", None)
        self.model = stored.base_patcher.model
        self.quantized = False
        self.moved = 0
        self.keys = 0

    def __call__(self, key: str) -> None:
        from .unbake import _is_quantized, live_tensor

        try:
            self.quantized = self.quantized or _is_quantized(live_tensor(self.model, key))
        except (AttributeError, TypeError):
            pass
        if self.slab is None or not key.startswith(_DIFFUSION_PREFIX):
            return
        self.keys += 1
        self.moved += self.slab.reabsorb_key(self.model.diffusion_model, key[len(_DIFFUSION_PREFIX):])

    def finish(self) -> None:
        """Give the freed copies back: the allocator otherwise keeps them as its own cache."""
        if self.slab is None and not self.quantized:
            return
        import comfy.model_management as mm

        mm.soft_empty_cache()
        if self.slab is not None:
            safe_call(
                log.info,
                "slab residency after the swap: %d key(s) checked, %.2f GiB returned to the slab%s",
                self.keys, self.moved / (1 << 30),
                f", quant strays {self.slab.stray_bytes / (1 << 30):.2f} GiB stay in cudaMalloc"
                if self.slab.stray_bytes else "")


def slab_misfits(stored, active) -> list[str]:
    """Keys this stack newly patches whose baked payload no slab region can take back.

    Such a key would stay in cudaMalloc beside its idle region, a second copy no
    price saw made, so the swap does not run: the store loads fresh, and the
    load's own walls price the bake whole (``store_slab_admit.admit_bake``). A
    key that already holds bytes outside the slab is not new: the fresh load
    priced that copy, and each swap replaces it with one of the same size.
    """
    slab = getattr(stored, "slab", None)
    patches = getattr(active, "patches", None)
    dm = getattr(getattr(getattr(stored, "base_patcher", None), "model", None), "diffusion_model", None)
    if (slab is None or dm is None or not isinstance(patches, dict)
            or not callable(getattr(slab, "baked_payload_fits", None))):
        return []
    misfits = []
    for key in patches:
        if not key.startswith(_DIFFUSION_PREFIX):
            continue
        name = key[len(_DIFFUSION_PREFIX):]
        if not slab.strays(name) and not slab.baked_payload_fits(dm, name):
            misfits.append(key)
    return misfits


def lazy_swap(store, stored, active, unet_name: str):
    """Fused restore + re-bake for a low_rss stack change.

    Stream pristine keys from disk while each restored key bakes on the GPU.
    The swap avoids ComfyUI's load manager so it cannot unpatch, evict, or
    double-count during the bake. Reuse verified record entries and capture
    only new keys before baking. Each key goes back to the resident's own
    storage as the swap finishes with it (``_SwapSettle``). Return the new
    un-bake record, or ``None`` when complete coverage cannot be proven; the
    next stack change then reloads from disk.
    """
    settle = _SwapSettle(stored)
    record = _restore_and_bake(store, stored, active, unet_name, settle)
    settle.finish()
    return record


def _restore_and_bake(store, stored, active, unet_name: str, settle: _SwapSettle):
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
