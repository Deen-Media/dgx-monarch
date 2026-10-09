"""Per-actor model store: request reuse, pristine-base LoRA swap, or full load.

The state machine is specified in DESIGN.md §5.5; every transition logs its
class (reuse | hot-swap | load).
"""
from __future__ import annotations

import os
import threading
from contextlib import suppress
from typing import Any

from ..adapters import fsdp_shard_build
from ..capacity_fit import SLAB_VOUCHED_FAMILIES as SLAB_VOUCHED_FAMILIES
from ..constants import TRANSITION_REUSE
from ..log import get_logger
from ..mesh_safety import stock_load_preflight  # noqa: F401 - read by store_load
from ..transfer_utils import failure_summary
from . import (
    resident_ledger,
    slab_lifetime,
    store_bake,
    store_detect,
    store_family,
    store_identity,
    store_load,
    store_residency,
)
from .comfy_bridge import resolve_model_path

# Re-exports: store_load and store_bake import most of these names from model_store at call
# time, so a test patch on model_store reaches those calls; the rest stay importable here.
from .store_bake import _merge_and_free  # noqa: F401 - compatibility re-export
from .store_detect import (
    _combine_checkpoint_and_live_kinds,  # noqa: F401 - compatibility re-export
    _detect_checkpoint_kind,  # noqa: F401 - compatibility re-export
    _detect_family,  # noqa: F401 - compatibility re-export
    _detect_live_precision,  # noqa: F401 - compatibility re-export
    _detect_quant_kind,  # noqa: F401 - compatibility re-export
    _is_quantized_kind,  # noqa: F401 - compatibility re-export
    _quant_kind_from_options,
    _to_comfy_model_options,  # noqa: F401 - compatibility re-export
)
from .store_load_ownership import FreshLoadOwnership  # noqa: F401 - compatibility re-export
from .stored_model import StoredModel

log = get_logger(__name__)

# Memoize families only after unforced loads, keyed by file identity. New bytes
# load as stock; later loads consult SLAB_VOUCHED_FAMILIES. Evict the oldest row
# from the largest family when full (store_family._eviction_victim). A miss
# requires a stock load or capacity refusal. Tests redirect DGXM_FAMILY_MEMO_PATH.
# The 1024-row limit exceeds the measured 92-row working set while keeping the
# memo below 80 kB. The 1 MiB file-size floor excludes tiny test fixtures.
_FAMILY_MEMO_PATH = os.environ.get("DGXM_FAMILY_MEMO_PATH") or os.path.expanduser("~/.cache/dgx-monarch/family_memo.json")
_FAMILY_MEMO_LOCK = threading.Lock()
_FAMILY_MEMO_LIMIT = 1024
_FAMILY_MEMO_MIN_BYTES = 1 << 20


def _slab_capable_path(path: str) -> bool:
    """Whether the checkpoint path is eligible for the safetensors slab loader."""
    return path.lower().endswith((".safetensors", ".sft"))


def _file_identity(path: str) -> str:
    return store_family.file_identity(path)


def memoized_family(path: str) -> str | None:
    """Previously detected family for these exact file bytes, if any."""
    return store_family.memoized_family(
        path, identity=_file_identity, memo_path=_FAMILY_MEMO_PATH, lock=_FAMILY_MEMO_LOCK)


def memoize_family(path: str, family: str) -> None:
    """Record a detected family atomically and best-effort across workers."""
    store_family.memoize_family(
        path, family, identity=_file_identity, memo_path=_FAMILY_MEMO_PATH,
        lock=_FAMILY_MEMO_LOCK, limit=_FAMILY_MEMO_LIMIT, logger=log,
        min_size=_FAMILY_MEMO_MIN_BYTES)


def normalize_options(options: dict | None) -> tuple:
    """Canonical, hashable form of loader options (weight dtype etc)."""
    if not options:
        return ()
    return tuple(sorted((str(k), str(v)) for k, v in options.items()))


def lora_signature(lora_stack: list[dict] | None) -> tuple:
    """Hashable signature of a lora stack: ((name, strength), ...) in order."""
    if not lora_stack:
        return ()
    return tuple((entry["name"], round(float(entry["strength"]), 6)) for entry in lora_stack)


def request_artifact_identity(unet_name: str, lora_stack: list[dict] | None) -> dict:
    return store_identity.build_request_artifact_identity(
        unet_name, lora_stack, resolve_model_path)


def _best_effort(call: Any, *args: Any) -> None:
    with suppress(BaseException):
        call(*args)


class ModelStore:
    """Owns the resident diffusion model of one GPUWorker."""

    def __init__(self) -> None:
        self.current: StoredModel | None = None
        self.uncond: StoredModel | None = None  # second model slot for the dual-model guider (Ideogram4)
        # comfy's unload_all_models() is global: dropping or quant-hot-swapping one
        # slot evicts the other slot's GPU weights too. Instance lifetime, keyed by
        # slot name; a reload or reuse of that slot clears it, and the reuse log says so.
        self._gpu_evicted: set[str] = set()
        # A failed global unload can leave comfy's process-wide model state partly
        # changed, so the slot keeps its resident and its slab and refuses loads.
        # Same lifetime and keys; a successful unload retry clears it.
        self._drop_failed: set[str] = set()
        self.lora_low_rss: bool = False
        # World size of the mesh this store loads for; the FSDP shard-build
        # price is per rank (actor/store_residency.resolve).
        self.world: int | None = None
        # Lazy-swap keys checked against comfy's bake: 0 disables, -1 checks all.
        self.swap_verify: int = 2
        self.verify_failures: int = 0
        # Zero-copy slab residency keeps weights in shared memory, with no cudaMalloc
        # copy and no CPU staging, and unload returns memory to the OS at once
        # (actor/slab.py). Set from the Init node's worker_args at setup.
        # False | True | "auto" ("auto" resolves per family at load time
        # against SLAB_VOUCHED_FAMILIES via the family memo).
        self.slab_weights: bool | str = False
        # Why an upstream gate collapsed residency to False, carried forward so
        # the ladder can name the blocker instead of refusing bare (class C).
        self.slab_blocked_reason: str = ""
        # The Init node's family_adapter widget, or None for auto detection.
        # Instance lifetime; every adapter decision reads it, so injection and the
        # render-time reads cannot disagree about the family. No setter: on a
        # change worker_env drops every resident, as it does for the compile policy.
        self.family_override: str | None = None

    def set_slab_mode(self, slab_weights: bool | str, reason: str = "") -> None:
        """Switch weight residency mode.

        A slab-backed model and a cudaMalloc-backed model have different storage
        and are never reused across a flip; store_residency.policy_keeps_resident
        decides whether a policy change keeps the resident.
        """
        mode: bool | str = "auto" if slab_weights == "auto" else bool(slab_weights)
        self.slab_blocked_reason = "" if mode else reason
        if mode == self.slab_weights:
            return
        previous = self.slab_weights
        self.slab_weights = mode
        for slot in ("cond", "uncond"):
            stored = self._stored(slot)
            if stored is None:
                continue
            if not store_residency.policy_keeps_resident(stored, mode):
                self._drop(slot)
                continue
            if mode is False:
                # Explicit stock cannot honour a vouched-auto promotion, and the
                # retry would load these bytes a second time.
                stored.slab_auto_retry = False
            log.info(
                "model store [%s]: adopt resident -- %s (%s residency held across "
                "slab policy %s -> %s)", slot, stored.base_key[0],
                store_residency.describe(stored)["residency"], previous, mode)
        log.info("weight residency mode: %s",
                 "auto (slab for ledger-vouched families)" if mode == "auto"
                 else "slab (zero-copy shared memory)" if mode
                 else "cudaMalloc (stock)")

    def set_lora_mode(self, low_rss: bool) -> None:
        """Switch lora residency mode.

        A resident holding unbaked lora patches also holds ComfyUI's full weight
        backup, so it is dropped rather than carried into low-RSS mode. A
        resident with no lora, or one already baked with its backup freed, has
        the same weights under either mode and stays.
        """
        low_rss = bool(low_rss)
        if low_rss == self.lora_low_rss:
            return
        previous = self.lora_low_rss
        self.lora_low_rss = low_rss
        for slot in ("cond", "uncond"):
            stored = self._stored(slot)
            if stored is None:
                continue
            if stored.lora_sig and not stored.base_baked:
                self._drop(slot)
                continue
            log.info("model store [%s]: adopt resident -- %s (lora residency held "
                     "across low_rss %s -> %s)", slot, stored.base_key[0],
                     previous, low_rss)
        log.info("lora residency mode: %s", "low_rss (merge-free)" if low_rss else "hot_swap (baked)")

    def _stored(self, slot: str) -> StoredModel | None:
        return self.uncond if slot == "uncond" else self.current

    def _mark_other_slot_evicted(self, slot: str) -> None:
        """After a global unload for `slot`: mark the other resident slot evicted (it
        reloads its GPU weights on next use) and clear `slot`, which is being replaced or emptied."""
        other = "cond" if slot == "uncond" else "uncond"
        if self._stored(other) is not None:
            self._gpu_evicted.add(other)
        self._gpu_evicted.discard(slot)

    @staticmethod
    def make_keys(unet_name: str, options: dict | None, lora_stack: list[dict] | None,
                  quant_kind: str) -> tuple[tuple, tuple]:
        base_key = (unet_name, normalize_options(options))
        request_key = (*base_key, lora_signature(lora_stack), quant_kind)
        return base_key, request_key

    def ensure(
        self,
        unet_name: str,
        options: dict | None,
        lora_stack: list[dict] | None,
        slot: str = "cond",
        on_base_loaded=None,
        fsdp_launch: bool = False,
        rescue_consent: dict | None = None,
    ) -> tuple[Any, str]:
        """Make `slot` hold (unet_name, options, loras); return (patcher, transition).

        on_base_loaded(base_patcher, quant_kind, lora_stack,
        precision_evidence) runs exactly once per fresh base load at the adapter
        injection point.
        """
        if slab_lifetime.cleanup_pending():
            raise RuntimeError(
                "a previous model load could not be cleaned up; explicitly unload "
                "or reset the Attached mesh before loading or reusing a model"
            )
        if slot in self._drop_failed:
            raise RuntimeError(
                f"model store [{slot}]: a previous unload failed; retry the unload "
                "or reset the Attached mesh before loading or reusing a model"
            )

        # Validate worker-bound loader options before artifact reads, strict
        # FSDP proof, store-key construction, or model lifecycle effects.
        quant_kind = _quant_kind_from_options(options)
        # The load path reads it through store_load; keeping it here means every
        # ensure, reuse included, fails on an unimportable comfy before it
        # touches store state.
        import comfy.sd as comfy_sd  # noqa: F401

        requested_identity = request_artifact_identity(unet_name, lora_stack)
        fsdp_preflight_path = None
        fsdp_checkpoint_proof = None
        if fsdp_launch:
            fsdp_preflight_path = resolve_model_path("diffusion_models", unet_name)
            fsdp_checkpoint_proof = store_detect.validate_fsdp_request_checkpoint(
                fsdp_preflight_path,
                options,
                lora_stack,
            )
            store_identity.assert_stable_identity(
                request_artifact_identity(unet_name, lora_stack),
                requested_identity,
                "proving the worker-local FSDP checkpoint header",
            )

        stored = self.uncond if slot == "uncond" else self.current
        base_key, request_key = self.make_keys(unet_name, options, lora_stack, quant_kind)
        authoritative_slab_retry = False

        if stored is not None and stored.base_key == base_key and (
            store_identity.base_artifact_identity(stored.artifact_identity)
            != store_identity.base_artifact_identity(requested_identity)
            or not store_detect.fsdp_reuse_matches_proof(stored, fsdp_checkpoint_proof)
        ):
            # Any bounded or exact identity drift routes stale state to a clean load.
            log.warning(
                "model store [%s]: checkpoint or ComfyUI identity changed for %s; "
                "discarding the resident base before reuse", slot, unet_name,
            )
            stored = None

        if stored is not None and stored.base_key == base_key:
            # Same checkpoint: detection already ran at load time, and the
            # option-derived kind can be weaker (scaled-fp8 checkpoints load
            # through default options). Compare with the detected kind so an
            # identical re-request reuses instead of hot-swapping forever.
            request_key = (*base_key, lora_signature(lora_stack), stored.quant_kind)
            if stored.slab_auto_retry:
                log.info("model store [%s]: reloading the first stock load into slab", slot)
                authoritative_slab_retry = True
                stored = None

        if stored is not None and stored.request_key == request_key \
                and stored.artifact_identity == requested_identity:
            reload_note = ""
            if slot in self._gpu_evicted:
                self._gpu_evicted.discard(slot)  # this reuse re-materializes it
                reload_note = " (GPU reload: a prior model change unloaded every model's GPU weights)"
            log.info("model store [%s]: %s%s -- %s", slot, TRANSITION_REUSE, reload_note, unet_name)
            return stored.active_patcher, TRANSITION_REUSE

        if stored is not None and stored.base_key == base_key and self.lora_low_rss \
                and (stored.unbake is not None or not stored.lora_sig):
            # Low-RSS hot-swap restores verified bytes and re-bakes in place. An
            # ordinary failure drops the slot and falls through to a full load; an
            # interruption drops it and propagates. A no-LoRA base needs no restore
            # (store_bake.lazy_swap).
            stored = None  # release ensure's pin; the transition reads the slot
            result = store_bake.lazy_swap_transition(
                self, slot, unet_name, lora_stack, base_key, requested_identity,
                request_artifact_identity=request_artifact_identity, logger=log)
            if result is not None:
                return result

        if stored is not None and stored.base_key == base_key and not self.lora_low_rss \
                and not stored.base_baked:
            # Hot-swap (clone the pristine base, re-apply loras). Skipped in
            # low_rss: _merge_and_free consumed the base (baked, backup freed), so
            # it cannot be re-cloned, and a base baked under an earlier low_rss
            # policy stays unclonable after the policy flips. A stack change then
            # falls through to a full reload from disk (exact), the low_rss recovery
            # contract when lazy un-bake is unavailable (an unsupported or ambiguous
            # quantized encoding, failed capture, or checkpoint changed on disk).
            return store_bake.hot_swap_transition(
                self, stored, slot, unet_name, lora_stack, base_key, requested_identity,
                request_artifact_identity=request_artifact_identity, logger=log)

        # Release ensure's local pin before dropping the old slot: USP injection
        # creates a model reference cycle, so a live local would defeat _drop's
        # gc.collect() and briefly retain two full DiTs.
        stored = None
        return store_load.fresh_load(
            self, slot, unet_name, options, lora_stack, quant_kind, requested_identity,
            base_key, on_base_loaded, fsdp_launch, fsdp_preflight_path,
            fsdp_checkpoint_proof, authoritative_slab_retry, rescue_consent,
            request_artifact_identity=request_artifact_identity,
            resolve_model_path=resolve_model_path, logger=log)

    def _lazy_swap(self, stored, active, unet_name: str):
        if getattr(stored.base_patcher, "_dgxm_fsdp", False):
            from . import fsdp_lora

            return fsdp_lora.lazy_swap(self, stored, active, unet_name)
        return store_bake.lazy_swap(self, stored, active, unet_name)

    _audit_swap_dtypes = staticmethod(store_bake.audit_swap_dtypes)

    def _verify_key(self, active, key: str) -> None:
        return store_bake.verify_key(self, active, key)

    _bake_key = staticmethod(store_bake.bake_key)

    def _build_active(self, base_patcher, lora_stack: list[dict] | None):
        """Clone the pristine base and apply the lora stack.

        Uses comfy's own lora merge math (fp32 intermediates + stochastic
        rounding), matching stock semantics (DESIGN.md §5.5). In
        low-RSS mode the build is the same; the caller then bakes to GPU
        and frees the backup via _merge_and_free (the residency saving), so the
        rendered weights are identical whether low_rss is on or off.
        """
        import comfy.sd as comfy_sd
        import comfy.utils as comfy_utils

        if lora_stack and getattr(base_patcher, "_dgxm_fsdp", False):
            from ..adapters.fsdp import validate_fsdp_launch_loras

            # comfy's merge cannot patch a DTensor; the shard-aware bake
            # (actor/fsdp_lora.py) can, and only under lora_low_rss.
            validate_fsdp_launch_loras(lora_stack, lora_low_rss=bool(self.lora_low_rss))

        active = base_patcher.clone()
        for entry in lora_stack or []:
            lora_path = resolve_model_path("loras", entry["name"])
            lora_sd = comfy_utils.load_torch_file(lora_path, safe_load=True)
            if self.lora_low_rss:
                # Pre-stage the adapter tensors on the compute device: the bake
                # otherwise uploads each lora's matrices host to device once per
                # patched key (~1900 small copies on a 7-lora full-coverage stack).
                # Values are unchanged, so the bake stays identical; the memory
                # is held only until patches.clear() at bake end.
                import comfy.model_management as mm

                dev = mm.get_torch_device()
                lora_sd = {k: v.to(dev) if hasattr(v, "to") else v for k, v in lora_sd.items()}
            active, _ = comfy_sd.load_lora_for_models(active, None, lora_sd, float(entry["strength"]), 0)
        return active

    _invalidate_gpu_weights = staticmethod(store_bake.invalidate_gpu_weights)

    def _drop(self, slot: str) -> None:
        stored = self.uncond if slot == "uncond" else self.current
        if stored is None:
            self._drop_failed.discard(slot)
            return
        import comfy.model_management as mm
        try:
            try:
                if stored.active_patcher is not None:
                    stored.active_patcher.cleanup()
            except Exception as exc:
                _best_effort(
                    log.warning, "model store [%s]: patcher cleanup failed during "
                    "drop: %s", slot, failure_summary(exc))
            fsdp_shard_build.drop_sharded_weights_in_place(
                (stored.base_patcher, stored.active_patcher),
                origin=f"model store [{slot}]")
            resident_ledger.unload_without_offload(mm, (stored.base_patcher, stored.active_patcher))
            self._drop_failed.add(slot)
            stored.base_patcher = stored.active_patcher = stored.unbake = None
            resident_ledger.collect_dropped()
            owner_kind = "slab" if stored.slab is not None else "resource" if stored.fsdp_checkpoint_pin is not None else ""
            if owner_kind == "slab":
                slab_lifetime.retain(stored)
            elif owner_kind:
                slab_lifetime._retain_resource(stored)
            setattr(self, "uncond" if slot == "uncond" else "current", None)
        except BaseException as exc:
            setattr(self, "uncond" if slot == "uncond" else "current", stored)
            self._drop_failed.add(slot)
            self._gpu_evicted.add(slot)
            other = "cond" if slot == "uncond" else "uncond"
            if self._stored(other) is not None:
                self._gpu_evicted.add(other)
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(
                f"model store [{slot}]: discard failed ({failure_summary(exc)}); the store "
                "keeps ownership of the resident and blocks the slot until cleanup succeeds"
            ) from exc
        if owner_kind == "slab":
            slab_lifetime.close_slab_or_retain_after_explicit_unload(stored, "model-store resources")
        elif owner_kind:
            slab_lifetime.close_or_retain_after_explicit_unload(stored, "model-store resources")
        self._drop_failed.discard(slot)
        self._mark_other_slot_evicted(slot)
        resident_ledger.flush_dropped(mm, slot)

    def cleanup_active(self) -> None:
        for stored in (self.current, self.uncond):
            if stored is not None:
                try:
                    stored.active_patcher.cleanup()
                except Exception:
                    pass

    def unload_all(self) -> None:
        resident_ledger.pin_store_for_unload(self.current, self.uncond)
        self._drop("uncond")
        self._drop("cond")
        self.release_retained_cleanup()

    def release_retained_cleanup(self) -> None:
        """Retry cleanup a failed load could not finish, keeping residents."""
        try:
            slab_lifetime.release_after_explicit_unload()
        except BaseException as exc:
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(
                "failed-load cleanup remains blocked; reset the Attached mesh"
            ) from exc

    def snapshot(self) -> dict:
        def describe(stored: StoredModel | None):
            if stored is None:
                return None
            return {
                "request_key": repr(stored.request_key),
                "family": stored.family,
                "quant": stored.quant_kind,
                **stored.precision_evidence.public(),
                "loras": list(stored.lora_sig),
                **store_residency.describe(stored),
            }

        return {
            "cond": describe(self.current),
            "uncond": describe(self.uncond),
            "cleanup_failed_slots": sorted(self._drop_failed),
            "failed_load_cleanup_pending": slab_lifetime.cleanup_pending(),
            "retained_failed_load_slabs": slab_lifetime.retained_count(),
            "retained_failed_load_resources": slab_lifetime.retained_resource_count(),
        }
