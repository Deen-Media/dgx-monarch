"""The GPUWorker sample body: authorize, load, equalize, denoise, report.

Runs on the actor's GPU thread under worker.py's `_sample_impl`, which passes
in the six sampler and identity seams (its comment says why). `worker` is the
live GPUWorker.
"""
from __future__ import annotations

import socket
import time
from types import ModuleType
from typing import Any

from ..model_sampling import normalize_model_sampling
from . import dm_cfg2_residency, latent_outputs, rank_readiness, store_fsdp, worker_env
from .cancellation import raise_if_sample_cancelled


def _sniff_request_family(worker, spec) -> str | None:
    """The checkpoint family a request model spec names, from its header alone.

    Header-only, so both ranks read the same two files and reach the same
    dm-cfg2 decision without loading either checkpoint. Returns None on a
    missing path or unreadable header (the load path surfaces those).
    """
    if not spec or not spec.get("unet_name"):
        return None
    try:
        import folder_paths

        from ..adapters.detect import CheckpointSniffError, sniff_checkpoint
        from ..family_select import effective_family, override_for_worker

        path = folder_paths.get_full_path("diffusion_models", spec["unet_name"])
        if path is None:
            return None
        try:
            sniffed = sniff_checkpoint(str(path))[0]
        except CheckpointSniffError:
            return None
        return effective_family(sniffed, override_for_worker(worker))
    except Exception:
        return None


def _dual_model_cfg2_topology(worker) -> bool:
    """The raise-free, rank-identical predicate that admits the dm-cfg2 path.

    World 2, cfg 2, no sequence or data parallelism, no FSDP. Both ranks read the
    same driver-pushed topology, so both take the ready exchange in ``run_sample``
    or neither does; that exchange can never deadlock on a rank that branched
    away. dual_model_cfg2_slot gates on the same predicate.
    """
    topo = getattr(worker, "topology", {})
    sp = int(topo.get("ulysses", 1)) * int(topo.get("ring", 1))
    # Never under fsdp: slotting one checkpoint per rank would have each rank
    # fully_shard a different model, and the world-group all-gather would
    # exchange shards of two checkpoints.
    return (int(topo.get("cfg", 1)) == 2 and sp == 1 and int(topo.get("dp", 1)) == 1
            and int(getattr(worker, "world", 0) or 0) == 2
            and not bool(topo.get("fsdp")))


def _post_load_readiness(worker) -> ModuleType | None:
    """The module carrying this topology's post-load readiness exchange, or None.

    Both candidates expose ``ready_or_raise()`` and ``not_ready(exc)``. The
    module is returned, not its two functions, so tests can swap the exchange at
    the module attribute to pin which one ran; a caller holding the functions
    would keep the ones bound at import.

    Raise-free and rank-identical, so every rank of a fleet takes the exchange
    or none does. dm-cfg2 is asked first and keeps its own refusal, whose text
    names the two-checkpoint split only that topology has; every other
    multi-rank, non-FSDP topology takes the generic one.
    """
    if _dual_model_cfg2_topology(worker):
        return dm_cfg2_residency
    if rank_readiness.whole_model_topology(worker):
        return rank_readiness
    return None


def _cfg_parallel_with_fsdp(worker) -> bool:
    topo = getattr(worker, "topology", {})
    return int(topo.get("cfg", 1)) > 1 and bool(topo.get("fsdp"))


def dual_model_cfg2_slot(worker, request: dict) -> str | None:
    """The one checkpoint slot this cfg rank keeps resident under dm-cfg2.

    "cond" for cfg rank 0, "uncond" for cfg rank 1, or None when the render is
    not dm-cfg2 (a dual-model render then keeps both checkpoints resident). The
    refusals below read only the topology, the request and the two checkpoint
    headers, identical on every rank, so a bad wiring refuses on both ranks
    together. Where two boxes hold different bytes under one name the reads can
    diverge; outside FSDP, run_sample's ready exchange still turns that into a
    symmetric refusal.
    """
    from ..adapters import family_supports_dual_model_cfg
    from ..adapters.base import UnsupportedModelError, cfg_rank
    from ..refusal import RefusalClass, refusal

    if _cfg_parallel_with_fsdp(worker):
        cond_family = _sniff_request_family(worker, request.get("model"))
        if cond_family is not None and family_supports_dual_model_cfg(cond_family):
            # Request-decided and rank-symmetric, so class P: the driver retires
            # the sample lease consumed and the next queue reuses the same fleet.
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"{cond_family} runs dual-model guidance, which cfg-parallel under FSDP "
                "cannot carry: each rank would shard a different checkpoint and the "
                "sharded all-gather would exchange halves of two models. Use topology "
                "cfg2 (dm-cfg2, one checkpoint per rank, resident) or uly2+fsdp "
                "(both checkpoints sharded on both ranks) for this family."))
        return None
    if not _dual_model_cfg2_topology(worker):
        return None
    cond_family = _sniff_request_family(worker, request.get("model"))
    uncond = request.get("uncond_model")
    if cond_family is None or not family_supports_dual_model_cfg(cond_family):
        # cfg2 for a batched-cfg family such as krea2 is the ordinary
        # combined path; only a dm-cfg2 family reaches the split guider. A
        # dual-model guider fed a non-dm-cfg2 family under cfg2 is a wiring
        # mistake that would trip the batched wrapper on the uncond rank.
        if uncond is not None and cond_family is not None:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"{cond_family} does not run dual-model cfg-parallel (cfg2); its guidance "
                "is not a two-checkpoint split. Use topology uly2 for this family, or wire "
                "a family that supports dm-cfg2 (Ideogram4)."))
        return None
    if uncond is None:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"{cond_family} under cfg2 runs dual-model guidance and needs the unconditional "
            "model resident on the second rank. Connect DGXMonarchUncondUNETLoader to "
            "DGXMonarchDualModelGuider's model_negative input, or use topology uly2 instead."))
    uncond_family = _sniff_request_family(worker, uncond)
    if uncond_family != cond_family:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "dual-model cfg2 needs both checkpoints from the same family: the conditional "
            f"model is {cond_family} and the unconditional model is "
            f"{uncond_family or 'unreadable'}. Wire both loaders to checkpoints of the same "
            "dm-cfg2 family (Ideogram4), or use topology uly2 instead."))
    return "cond" if cfg_rank() == 0 else "uncond"


def run_sample(
    worker,
    request: dict,
    progress_port,
    cancel_event,
    *,
    equalize_cond_lengths,
    model_sampling_render_clone,
    request_artifact_identity,
    run_ksampler,
    run_custom,
    latent_signature,
) -> dict:
    if worker._setup_key is None:
        raise RuntimeError("sample before setup(): the Init node must run first")
    raise_if_sample_cancelled(cancel_event)

    model_spec = request["model"]
    model_sampling = normalize_model_sampling(model_spec.get("model_sampling"))
    uncond_sampling = None
    if request.get("uncond_model"):
        uncond_sampling = normalize_model_sampling(
            request["uncond_model"].get("model_sampling"))
    expected_artifacts = worker_env.verify_sample_artifact_authorization(
        worker, request, request_artifact_identity)
    worker_env.activate_accuracy_waivers(request)  # before any load or injection
    if request.get("sage_kernel"):
        # Resolve the per-render kernel (sage or sol; the request field is
        # named sage_kernel for wire compatibility) before a fresh model
        # injects its family attention. A Sage render must not construct the
        # Wan TORCH_FLASH treatment because setup's default was flash;
        # same-generation Sage unavailability still keeps the current
        # kernel through _AttentionDispatch.configure's fallback.
        worker._attn.configure(
            request["sage_kernel"],
            bool(request.get("sync_ulysses", True)),
            topology=worker.topology,
            world=worker.world,
            setup_generation=worker._setup_generation,
        )
    raise_if_sample_cancelled(cancel_event)
    from .sampling import RenderCancelledError

    # dm-cfg2 residency: cfg rank 0 keeps only the conditional checkpoint and cfg
    # rank 1 only the unconditional one, so each box holds a single ig4
    # checkpoint. The split guider runs one forward per rank against that
    # resident primary; the other checkpoint never loads on this rank.
    #
    # A load or identity failure can strike one rank alone: under dm-cfg2 because
    # the two ranks open different files, and on every other multi-rank topology
    # because the boxes reach the load with different memory, so a wall can fire
    # on one of them after the fleet agreed.
    # The slot decision and the per-rank load therefore run inside a guard: where
    # _post_load_readiness names an exchange (never under FSDP), a failure on
    # this rank still joins it, so a healthy peer refuses instead of waiting at
    # the first collective until the group timeout. Cancellation propagates
    # untouched: the driver cancels every rank and recycles, so a wait on a
    # one-sided cancel ends at the recycle, and routing a routine cancel through
    # the ready flag would make the peer refuse.
    readiness = _post_load_readiness(worker)
    primary_sampling = model_sampling
    from ..adapters import quant_activation_scale
    try:
        dm2_slot = dual_model_cfg2_slot(worker, request)
        if dm2_slot == "uncond":
            uspec = request["uncond_model"]
            patcher, transition = store_fsdp.ensure(
                worker, uspec["unet_name"], uspec.get("options"),
                uspec.get("loras"), slot="uncond", on_base_loaded=worker._inject_for_topology,
                rescue_consent=worker_env.sample_rescue_consent(request))
            worker_env.assert_resident_artifact_identity(
                worker, "uncond", expected_artifacts[1])
            primary_sampling = uncond_sampling
            raise_if_sample_cancelled(cancel_event)
            worker._check_uma_reserve()
            uncond_patcher = None
        else:
            patcher, transition = store_fsdp.ensure(
                worker, model_spec["unet_name"], model_spec.get("options"),
                model_spec.get("loras"), slot="cond", on_base_loaded=worker._inject_for_topology,
                rescue_consent=worker_env.sample_rescue_consent(request))
            worker_env.assert_resident_artifact_identity(
                worker, "cond", expected_artifacts[0])
            raise_if_sample_cancelled(cancel_event)

            worker._check_uma_reserve()
            uncond_patcher = None
            # Under dm-cfg2 (dm2_slot == "cond") cfg rank 0 skips the unconditional
            # load; the both-resident dual-model path (dm2_slot None) loads it as
            # the second model.
            if dm2_slot is None and request.get("uncond_model"):
                raise_if_sample_cancelled(cancel_event)
                uspec = request["uncond_model"]
                uncond_patcher, _uncond_transition = store_fsdp.ensure(
                    worker, uspec["unet_name"], uspec.get("options"),
                    uspec.get("loras"), slot="uncond", on_base_loaded=worker._inject_for_topology)
                worker_env.assert_resident_artifact_identity(
                    worker, "uncond", expected_artifacts[1])
        # Class K, and the driver retires the sample lease consumed: the bar
        # refuses a sharded nvfp4 render of a family measured past the fidelity
        # floor, the install's record is rank identical, and grants are live.
        # Inside the load guard, so a refusing rank's peer refuses too rather
        # than waiting out the group timeout; on the sample path, because a
        # class-K grant lives for one dispatch and a load carries none.
        if quant_activation_scale.plan_for_topology(
                worker.topology, dual_model_cfg=dm2_slot is not None):
            for loaded in (patcher, uncond_patcher):
                if loaded is not None:
                    quant_activation_scale.assert_shard_quant_scale_covered(
                        loaded.model.diffusion_model)
        if worker.topology.get("ring", 1) > 1:
            # cuDNN Ring prepares here, inside the load guard, before the exchange.
            worker._attn.prepare_native_attention()
    except RenderCancelledError:
        raise
    except Exception as exc:
        if readiness is None:
            raise
        # NoReturn: exchange this rank's not-ready flag, then re-raise the cause.
        readiness.not_ready(exc)
    if readiness is not None:
        # This rank loaded and matched; release the fleet or refuse if a peer did
        # not. Outside the try so its own refusal is never re-caught above.
        readiness.ready_or_raise()
    raise_if_sample_cancelled(cancel_event)

    # Opt-in proof only: full-content file identities are stable-read after
    # ModelStore has adopted and checked every requested resident, but before
    # any render-local patching or denoising. The returned structure is bounded
    # and path-free; ordinary renders skip this I/O and keep their result shape.
    from .. import adoption_evidence
    from .comfy_bridge import resolve_model_path

    # dm-cfg2 keeps one checkpoint per rank, so no single rank can prove both
    # slots adopted; the cross-rank identity gate is the residency proof there.
    resident_adoption_evidence = (
        None if dm2_slot is not None
        else adoption_evidence.build_worker_evidence(
            worker,
            request,
            expected_artifacts,
            resolver=resolve_model_path,
            identity_fn=request_artifact_identity,
        )
    )

    # cfg-parallel asymmetric-prompt equalization (per-family rule). Pure cfg
    # topologies only: the mask-honoring paths (chroma's cond mask, krea2's
    # pad-aware forward) run native attention, which sp > 1 replaces with
    # maskless USP kernels. Padding there would silently attend to zero rows,
    # so combined uly*+cfg* topologies skip it. Their asymmetric prompts fold
    # under comfy's own repeat rule where it allows; otherwise they take the
    # per-cond dispatch (adapters/cfg_dispatch.py), and they reach the cfg
    # wrapper's typed error only where that dispatch also declines.
    from ..adapters import adapter_for, cfg_dispatch
    from ..adapters.cfg_parallel import cfg_dispatches_per_cond
    from ..family_select import override_for_worker

    # Fold is this render's default. Where equalize_cond_lengths runs, it records
    # the real lengths in its place only for a family whose cfg pad restores the
    # stock call (cfg_pad_restores_stock_call).
    cfg_dispatch.record_cfg_pair_lengths([])
    family_override = override_for_worker(worker)
    sp_degree = int(worker.topology.get("ulysses", 1)) * int(worker.topology.get("ring", 1))
    adapter = (adapter_for(patcher.model, family_override)
               if int(worker.topology.get("cfg", 1)) > 1 and sp_degree == 1 else None)
    # A family that dispatches per cond is never padded: the pad could
    # never make its constant fold, and it would hand one rank a padded cond
    # whose forward can refuse where the other's returns, stranding its peer in
    # an all-gather. The skip reads a static attribute, so every rank takes it.
    if adapter is not None and not cfg_dispatches_per_cond(adapter):
        latent_samples = request["latent"]["samples"]
        if getattr(adapter, "cfg_cond_padding", None) == "pad+mask":
            # The one rule that reads the image grid gets the grid the sampler
            # will run, size tags applied, by shape arithmetic: no second fixed
            # latent beside the sampler's own, and no second fix.
            latent_samples = latent_outputs.fixed_latent_grid(patcher, request["latent"])
        if request.get("positive") is not None:
            request["positive"], request["negative"] = equalize_cond_lengths(
                adapter, request["positive"], request["negative"], latent_samples)
        else:
            # SamplerCustom path: conds live inside the guider spec.
            guider_spec = request.get("guider") or {}
            if guider_spec.get("kind") == "cfg":
                guider_spec["positive"], guider_spec["negative"] = equalize_cond_lengths(
                    adapter, guider_spec["positive"], guider_spec["negative"], latent_samples)

    from .comfy_bridge import gpu_load_seconds_reset

    render_patcher = model_sampling_render_clone(patcher, primary_sampling)
    render_uncond_patcher = (
        model_sampling_render_clone(uncond_patcher, uncond_sampling)
        if uncond_patcher is not None else None
    )
    gpu_load_seconds_reset()  # zero the accumulator for this request
    t0 = time.perf_counter()

    from .sampling import _dp_info

    # A non-dp render never consumes this (split_conditioning_for_dp and
    # split_guider_for_dp both short-circuit at dp_world <= 1).
    dp_cond_exempt_keys: frozenset[str] = frozenset()
    if _dp_info()[1] > 1:
        dp_cond_exempt_keys = adapter_for(
            patcher.model, family_override).dp_cond_exempt_keys

    raise_if_sample_cancelled(cancel_event)
    if request["kind"] in ("ksampler", "ksampler_advanced"):
        samples, out = run_ksampler(
            render_patcher, request, progress_port, cancel_event=cancel_event,
            dp_cond_exempt_keys=dp_cond_exempt_keys)
    elif request["kind"] == "custom":
        samples, out = run_custom(
            render_patcher, request, render_uncond_patcher, progress_port,
            cancel_event=cancel_event, dp_cond_exempt_keys=dp_cond_exempt_keys)
    else:
        raise ValueError(f"unknown sample kind {request['kind']!r}")
    elapsed = time.perf_counter() - t0
    gpu_load_s = gpu_load_seconds_reset()
    from ..transfer import is_nested_tensor

    stats_tensors = list(samples.unbind()) if is_nested_tensor(samples) else None
    dp_rank, _dp_world = _dp_info()
    result: dict[str, Any] = {
        "host": socket.gethostname(),
        "rank": worker.rank,
        # This rank's dp slice index; nodes/render_result.py orders leaders by
        # it, so the field must always be present.
        "dp_rank": dp_rank,
        "sample_s": round(elapsed, 2),
        "gpu_load_s": round(gpu_load_s, 2),
        "transition": transition,
        "latent": None,
        # Every rank reports an exact full-byte digest plus full-content,
        # noise-comparable projections, so the cross-rank identity gate
        # (DESIGN.md §5.3) tells bit identity from bounded numerical drift
        # without shipping each rank's tensor. Every packed modality counts:
        # a matching video part must not hide diverging LTX-AV audio.
        "latent_stats": ([latent_signature(part) for part in stats_tensors]
                         if stats_tensors is not None else latent_signature(samples)),
        # One entry per class-K guard that proceeded under a granted waiver
        # (empty unless one was spent); the driver writes the WAIVER `use` row
        # at the ledger's current protocol (gate_ledger.GATE_PROTOCOL_VERSION).
        worker_env.ACCURACY_WAIVER_RESULT_KEY: worker_env.accuracy_waiver_stamps(),
    }
    if resident_adoption_evidence is not None:
        result["resident_adoption_evidence"] = resident_adoption_evidence
    if out is not None:
        return worker._pack_latent_result(result, out, request)
    return result
