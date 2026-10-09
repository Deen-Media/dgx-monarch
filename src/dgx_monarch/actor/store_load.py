"""Fresh-load path of ModelStore.ensure: residency decision, load, adoption.

Reached when no resident can be reused or swapped. Runs the residency ladder,
loads through Comfy (slab or stock), detects the checkpoint's family and quant
kind, injects the adapter, then publishes the StoredModel into the slot under
FreshLoadOwnership. The names taken from model_store are imported inside the
function body so every lookup is late-bound; hoisting them to module scope would
make test and gate patches on model_store invisible to the load.
"""
from __future__ import annotations

import os
from typing import Any

from .. import residency_mode
from ..compile_policy import compile_dit_blocks_slab, compile_dit_is_known_noop
from ..constants import TRANSITION_LOAD
from ..transfer_utils import raise_with_distinct_cause, reconcile_error
from . import load_fault, store_detect, store_identity, store_residency


def fresh_load(
    store,
    slot: str,
    unet_name: str,
    options: dict | None,
    lora_stack: list[dict] | None,
    quant_kind: str,
    requested_identity: dict,
    base_key: tuple,
    on_base_loaded,
    fsdp_launch: bool,
    fsdp_preflight_path,
    fsdp_checkpoint_proof,
    authoritative_slab_retry: bool,
    rescue_consent: dict | None,
    *,
    request_artifact_identity,
    resolve_model_path,
    logger,
) -> tuple[Any, str]:
    """Drop `slot`, load `unet_name` into it, and return (patcher, transition)."""
    import time

    import comfy.sd as comfy_sd

    from .model_store import (
        SLAB_VOUCHED_FAMILIES,
        FreshLoadOwnership,
        StoredModel,
        _combine_checkpoint_and_live_kinds,
        _detect_checkpoint_kind,
        _detect_family,
        _detect_live_precision,
        _merge_and_free,
        _slab_capable_path,
        _to_comfy_model_options,
        lora_signature,
        memoize_family,
        memoized_family,
        stock_load_preflight,
    )

    family_override = getattr(store, "family_override", None)
    # The operator's uma_reserve_gb, in bytes, for the slab wall's floor. Both
    # worker_env seams set it on the store beside the residency policy, so this
    # wall and the driver's own quote charge one number. The default covers a
    # store no setup has touched, and it hides a missing copy rather than
    # raising over it, so the routing tests pin the figure that arrives here.
    reserve_bytes = int(float(getattr(store, "uma_reserve_gb", 0.0) or 0.0) * (1 << 30))
    store._drop(slot)
    t0 = time.perf_counter()
    path = fsdp_preflight_path or resolve_model_path(
        "diffusion_models",
        unet_name,
    )
    model_options = _to_comfy_model_options(options)
    checkpoint_quant = (
        fsdp_checkpoint_proof.quant_kind
        if fsdp_checkpoint_proof is not None
        else _detect_checkpoint_kind(path, quant_kind)
    )
    slab = None
    slab_dtype_fallback = False
    # The residency ladder (actor/store_residency.py) owns the whole decision:
    # comfy-managed, explicit widget, vouched auto, stock if it fits, class C
    # rescue offer, consented slab load.
    decision = store_residency.resolve(
        path=path, unet_name=unet_name, model_options=model_options, fsdp_launch=fsdp_launch,
        world=getattr(store, "world", None),
        slab_weights=store.slab_weights, lora_low_rss=store.lora_low_rss, rescue_consent=rescue_consent,
        blocked_reason=store.slab_blocked_reason, slab_capable_path=_slab_capable_path(path),
        authoritative_slab_retry=authoritative_slab_retry, memoized_family=memoized_family,
        vouched_families=SLAB_VOUCHED_FAMILIES, node_options=options, lora_stack=lora_stack,
        family_override=family_override, reserve_bytes=reserve_bytes,
    )
    fsdp_checkpoint_pin = None
    load_ownership = FreshLoadOwnership()
    pin_handoff = load_ownership.pin_handoff
    slab_handoff = load_ownership.slab_handoff
    base = active = None
    stored = None
    slot_attr = "uncond" if slot == "uncond" else "current"
    store._drop_failed.add(slot)
    try:
        store_detect.pin_fsdp_checkpoint(
            path, fsdp_checkpoint_proof, handoff=pin_handoff)
        fsdp_checkpoint_pin = pin_handoff[-1] if pin_handoff else None
        load_path = fsdp_checkpoint_pin.loader_path if fsdp_checkpoint_pin else path
        # This call sits above the ``decision.use_slab`` branch, so a slab load is
        # priced too (store_residency.preload_capacity_check says what each
        # rung pays).
        store_residency.preload_capacity_check(
            decision, load_path, unet_name, model_options,
            preflight=stock_load_preflight,
            fsdp_launch=fsdp_checkpoint_proof is not None,
            fsdp_checkpoint_kind=checkpoint_quant,
            fsdp_world=getattr(store, "world", None),
            reserve_bytes=reserve_bytes,
            lora_stack=lora_stack, lora_low_rss=bool(store.lora_low_rss))
        # The debug fault knob fires here and nowhere else: the fleet has
        # agreed, every local price is paid, and no weight has been read yet.
        load_fault.check()
        if decision.use_slab:
            from . import comfy_bridge

            base, slab, slab_dtype_fallback = comfy_bridge.load_diffusion_model_slab(
                load_path, model_options, comfy_sd.load_diffusion_model, unet_name,
                handoff=slab_handoff)
        else:
            import contextlib

            load_window: contextlib.AbstractContextManager[object]
            if fsdp_checkpoint_proof is not None:
                # The mmap window keeps the state dict file-backed
                # (pread_backend.mmap_load_window) and the assign window makes
                # those file tensors the parameters, so nothing is committed
                # before each block is sharded (actor/fsdp_streaming.py).
                from . import fsdp_streaming, pread_backend

                load_window = contextlib.ExitStack()
                load_window.enter_context(pread_backend.mmap_load_window())
                load_window.enter_context(fsdp_streaming.assign_load_window())
            else:
                load_window = contextlib.nullcontext()
            with load_window:
                base = comfy_sd.load_diffusion_model(load_path, model_options=model_options)
        if fsdp_checkpoint_proof is not None:
            store_detect.assert_fsdp_checkpoint_identity(
                path,
                fsdp_checkpoint_proof,
                "finishing the Comfy model load",
            )
        live_precision = _detect_live_precision(base, quant_kind)
        detected_quant = _combine_checkpoint_and_live_kinds(
            checkpoint_quant, live_precision.quant_kind,
            live_dtypes_are_the_file=store_detect.file_dtypes_preserved(base),
            live_profile=live_precision.live_dtype_profile)
        precision_evidence = live_precision.bind_checkpoint(
            checkpoint_quant,
            detected_quant,
        )
        if family_override:
            # Detection raises on the derivative subclasses an override exists
            # to admit, so it runs best effort here, for the log line only.
            # adapter_for validates the forced bind inside on_base_loaded below.
            family = family_override
            detected = store_detect.detected_family_or_none(base)
            logger.info(
                "family adapter: forced %s for %s (detection read %s); this load "
                "is not the named family's recorded evidence",
                family_override, unet_name, detected or "no supported family")
        else:
            family = _detect_family(base)
        # Apply Qwen's cache policy after every load route and family selection,
        # and before adapter injection can see this patcher
        # (qwen_image21_cache.apply_loaded_cache_policy says why).
        from .qwen_image21_cache import apply_loaded_cache_policy

        apply_loaded_cache_policy(base, family, options)
        # Header classification happens before Comfy opens the model. A
        # same-name atomic replacement in that interval must be rejected
        # before adapter injection (notably FSDP sharding), even when
        # Comfy casts the replacement's live parameters to BF16. The
        # later check remains necessary to cover LoRA/build-time drift.
        store_identity.assert_stable_identity(
            request_artifact_identity(unet_name, lora_stack), requested_identity,
            "loading base before adapter injection",
        )
        if not family_override:
            # A forced family is never memoized. The memo is keyed by file
            # identity and read by the plain `auto` residency path, so writing
            # a claimed family here would hand it to a later session that set
            # no override.
            memoize_family(path, family)
        compile_requested = os.environ.get("DGXM_COMPILE_DIT") == "1"
        auto_retry = decision.auto_retry_eligible and family in SLAB_VOUCHED_FAMILIES
        explicit_noop_retry = (
            store.slab_weights is True and compile_requested
            and compile_dit_is_known_noop(family)
            and not family_override and not fsdp_launch
            and not authoritative_slab_retry and _slab_capable_path(path)
            and decision.rung in (store_residency.RUNG_STOCK_FITS,
                                  store_residency.RUNG_EXPLICIT_STOCK)
        )
        slab_auto_retry = (slab is None and not slab_dtype_fallback
                           and not compile_dit_blocks_slab(family, compile_requested)
                           and (auto_retry or explicit_noop_retry))
        if slab_auto_retry:
            logger.info(
                "family %s qualifies for slab residency, but this first load of %s stayed "
                "stock because the family is known only after a load; the next "
                "load of the same file slab-loads", family, unet_name,
            )
        # Set from the load that ran, not the decision: a dtype fallback can
        # take the stock branch after a slab decision. The worker's injection
        # seam reads this marker before it can call the compiler.
        base._dgxm_slab_resident = slab is not None
        if on_base_loaded is not None:
            on_base_loaded(
                base,
                detected_quant,
                lora_stack,
                precision_evidence,
            )
        if fsdp_checkpoint_proof is not None:
            # Every block is sharded now, so nothing needs the mmap any more.
            # This runs before comfy's own load_models_gpu, which sizes what it
            # moves by the memory it can see.
            from . import fsdp_streaming

            fsdp_streaming.release_materialize_window(base, load_path)
        active = store._build_active(base, lora_stack)
        unbake_record = None
        base_baked = bool(store.lora_low_rss and lora_stack)
        if store.lora_low_rss and lora_stack:
            # Bake now and free the ~model-sized backup (the residency win); the
            # sampler's later load_models_gpu is then a no-op on a stable snapshot.
            # Capture is attempted for every quant: it verifies byte-identity
            # per key/component against the checkpoint and aborts itself when a
            # checkpoint cannot be lazily un-baked (e.g. a bf16 file cast to
            # fp8 at load); those keep the full-reload contract.
            if getattr(base, "_dgxm_fsdp", False):
                # Sharded weights: comfy's load never runs; the bake reads the
                # pristine full weight from disk and writes this rank's chunk.
                from . import fsdp_lora

                unbake_record = fsdp_lora.bake_stack(store, active, path)
            else:
                if slab is not None:
                    # The header estimate priced the ladder; this backstop
                    # re-checks against the exact patch set _build_active
                    # just resolved, one monarch-owned moment before the
                    # bake runs.
                    from .. import capacity_fit
                    from . import store_slab_admit
                    from .rescue_offer import host_name

                    size = os.path.getsize(path) if os.path.exists(path) else 0
                    store_slab_admit.admit_bake(active, unet_name, host_name(),
                                                capacity_fit.slab_floor_bytes(size, reserve_bytes))
                unbake_record = _merge_and_free(active, base_path=path)
        if slab is not None:
            # comfy's first-load bake (and the F32->live-dtype pre-cast)
            # replace some Parameters with cudaMalloc tensors; migrate them
            # back into slab memory so residency stays 1x in the slab.
            import comfy.model_management as mm

            stats = slab.reabsorb(base.model.diffusion_model)
            mm.soft_empty_cache()
            logger.info("slab residency: %.2f GiB slab, reabsorbed %s strays (%s GiB)%s",
                        slab.total_gib, stats.get("reabsorbed", 0),
                        stats.get("reabsorbed_gib", 0.0),
                        f", quant strays {stats['quant_stray_gib']} GiB stay in cudaMalloc"
                        if stats.get("quant_stray_gib") else "")
        fsdp = bool(getattr(base, "_dgxm_fsdp", False))
        stock = slab is None and not fsdp and decision.residency != residency_mode.MODE_COMFY_MANAGED
        if slab is not None or fsdp or stock:
            # Run comfy's own full load now, after any LoRA bake. Slab and FSDP
            # weights are already on the device, so the walk moves nothing and
            # comfy's ledger stops reading zero (resident_ledger says why that
            # matters). A stock resident is placed now too: comfy would size its
            # move at sample time while the loaded file still sits on the host,
            # and a fleet rank that partial-loads there meets the divergence
            # guard. Placing it now, at the peak the stock price was
            # just taken for, frees the host copy before any sampler sizes the
            # model. A stack still pending (baked hot-swap residency) keeps
            # comfy's sample-time load: declare skips it.
            from . import resident_ledger

            resident_ledger.declare(
                active, "slab" if slab is not None else "fsdp shard" if fsdp else "stock")
        store_identity.assert_stable_identity(
            request_artifact_identity(unet_name, lora_stack), requested_identity,
            "loading; refusing unstable weights")
        stored = StoredModel(
            base_key=base_key,
            request_key=(*base_key, lora_signature(lora_stack), detected_quant),
            base_patcher=base,
            active_patcher=active,
            quant_kind=detected_quant,
            precision_evidence=precision_evidence,
            family=family,
            artifact_identity=requested_identity,
            lora_sig=lora_signature(lora_stack),
            unbake=unbake_record,
            slab=slab,
            slab_auto_retry=slab_auto_retry,
            fsdp_checkpoint_pin=fsdp_checkpoint_pin,
            residency_rung=decision.rung,
            base_baked=base_baked,
        )
        load_ownership.prepare_adoption(store, slot_attr, stored)
        setattr(store, slot_attr, stored)
        load_ownership.disarm_and_confirm()
    except BaseException as load_exc:
        if stored is not None and getattr(store, slot_attr) is stored:
            try:
                load_ownership.disarm_and_confirm()
            except BaseException as confirm_exc:
                strongest, cause = reconcile_error(
                    load_exc,
                    confirm_exc,
                    "published fresh-load adoption confirmation also failed",
                )
                raise_with_distinct_cause(strongest, cause)
            store._drop_failed.discard(slot)
            raise
        cleaned = False
        fresh_cleanup_error: BaseException | None = None
        try:
            base = active = None
            if stored is not None:
                stored.base_patcher = stored.active_patcher = stored.unbake = None
            cleaned = load_ownership.cleanup(load_exc)
        except BaseException as caught:
            fresh_cleanup_error = caught
        if cleaned:
            store._drop_failed.discard(slot)
        if fresh_cleanup_error is not None:
            strongest, cause = reconcile_error(
                load_exc,
                fresh_cleanup_error,
                "fresh-load cleanup also failed",
            )
            raise_with_distinct_cause(strongest, cause)
        raise
    store._drop_failed.discard(slot)
    logger.info(
        "model store [%s]: %s -- %s (%s, quant=%s) in %.1fs [%s]",
        slot, TRANSITION_LOAD, unet_name, family, detected_quant, time.perf_counter() - t0,
        store_residency.summary_line(stored, decision),
    )
    from ..telemetry import emit

    emit("load", model=unet_name, quant=detected_quant,
         total_s=round(time.perf_counter() - t0, 1))
    return active, TRANSITION_LOAD
