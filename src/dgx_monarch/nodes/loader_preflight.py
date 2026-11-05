"""Loader-side UMA footprint preflight and slab-residency rescue.

The check runs above ``ensure_live`` and charges two non-overlapping windows:
the pending DiT load, then the resident weights plus the driver stack still to
be built. It is a conservative DiT-axis classifier, not a full demand model.
The 2026-08-04 calibration separates a 61.7 GiB artifact that froze the host
from a 31.7 GiB artifact that completed with 0.7 GiB free.

Only a completed over-budget estimate refuses. Refusals name the charged terms,
budget, fitting options, bypass, and a slab rescue when that policy would fit.
No ComfyUI import at module scope: ``folder_paths`` and
``comfy.model_management`` are imported inside the functions that need them, so
a headless caller can import this leaf.
"""
from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .. import (
    capacity_fit,
    consent_pending,
    driver_footprint,
    family_select,
    gate_artifacts,
    mesh_safety,
    residency_mode,
    upstream_gate,
)
from ..adapters.base import UnsupportedModelError
from ..log import get_logger
from ..refusal import PanelAction, RefusalClass, refusal
from . import consent_projection, consent_rescue, graph_load_price, old_model_clear
from .graph_load_price import WINDOW_SETTLED, WINDOW_TRANSIENT, loader_refusal
from .loader_graph import (
    DriverStackTerms,
    activation_sp_degree,
    clear_memos,
    credited_terms,
    driver_stack_terms,
    note_stack_charged,
    remember,
    reset_charged_artifacts,
    unet_names,
)

log = get_logger(__name__)

_GIB = 2 ** 30

# Panel copy and headless controls come from the consent registry so the card
# and refusal stay aligned.
LOADER_GUARD = "loader_footprint_preflight"
RESCUE_KIND = "rescue-slab"

# Named windows distinguish a load spike from an oversized resident set. They
# and ``loader_refusal`` live with the refusal's words (graph_load_price) and
# are read through this module too.

# Session lifetime, keyed by the artifact and the world it was priced under,
# FIFO-evicted at loader_graph._MEMO_LIMIT. A stale entry can omit a charge but
# cannot create a false refusal. recycle_drain clears this and the render-site
# memos together. Write it only through loader_graph's `remember` and
# `clear_memos`, which hold the memo lock. A shard build and a whole-file load
# are two prices for one name: keyed on the name alone, a load admitted at a
# shard price would credit a later whole-file load with weights it never held.
_CLEARED_UNETS: dict[str, None] = {}


def _memo_key(unet_name: str, fsdp_world: int) -> str:
    """The priced pair, rendered as the one key ``remember`` can evict."""
    return f"{unet_name}\x00{int(fsdp_world)}"


# proc lifetime, keyed by family, never evicted: one warning each. Not a
# residency memo: a recycle does not hand a headless caller a graph, so
# `reset_memos` leaves this alone and the line is not repeated per recycle.
_UNPRICED_STACK_WARNED: set[str] = set()
_UNPRICED_STACK_LOCK = threading.Lock()


def _first_unpriced_stack_warn(family: str) -> bool:
    """True exactly once per family: locked test-and-add, so overlapping
    loads (pipeline depth > 1, fleet) cannot race the membership check."""
    with _UNPRICED_STACK_LOCK:
        if family in _UNPRICED_STACK_WARNED:
            return False
        _UNPRICED_STACK_WARNED.add(family)
        return True


# Re-exported: the loader node reaches it through this module.
record_rescue_row = consent_rescue.record_rescue_row


def reset_memos() -> None:
    """Drop every residency memo; `_UNPRICED_STACK_WARNED` is not one (see its comment)."""
    clear_memos(_CLEARED_UNETS)
    reset_charged_artifacts()
    consent_projection.reset_grant_audit()


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


def _slab_capable_path(path: str) -> bool:
    """Return whether the worker slab loader can open this checkpoint path.

    The predicate is local to avoid importing the torch-backed model store.
    """
    return path.lower().endswith((".safetensors", ".sft"))


# The one merge both loader checks and every consent read use
# (consent_rescue.mesh_worker_args says why there is only one).
effective_worker_args = consent_rescue.mesh_worker_args


def preflight_upstream_gated_artifact(unet_name: str) -> None:
    """Refuse an artifact this release carries no forward for, or whose weight
    scales ComfyUI would drop, before anything else.

    No topology, world, budget or latent shape changes this answer, so it runs
    above the comfy-managed policy card, the footprint card and ``ensure_live``.
    The worker owns the same refusal at its load funnel
    (``actor/store_fsdp._assert_artifact_runnable``) for the headless case
    (docs/TROUBLESHOOTING.md #85); both raise through
    ``upstream_gate.raise_refusal``, which builds the sentence at the raise,
    never at import (the comment above ``actor/partial_load_guard.REFUSAL_TEXT``
    says why). No mesh fact enters the decision, so ``mesh`` is not a parameter.
    """
    try:
        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", unet_name)
    except Exception:  # headless or unreadable models tree: never a refusal
        return
    if path is not None:
        upstream_gate.raise_refusal(str(path))


def preflight_comfy_managed_topology(mesh: Any) -> None:
    """Reject comfy-managed FSDP before evaluating capacity.

    This combination is unsupported regardless of memory availability. Check the
    requested mode and preset on the driver before a capacity card can offer slab,
    which cannot control comfy-managed placement. ``actor/store_fsdp`` repeats
    this check at the worker's load entry point.
    """
    from ..topology import declares_fsdp

    if not declares_fsdp(getattr(mesh, "topology_preset", "auto")):
        return
    if not residency_mode.requested(effective_worker_args(mesh)):
        return
    raise residency_mode.ComfyManagedResidencyError(refusal(
        RefusalClass.PHYSICS, residency_mode.COMFY_MANAGED_FSDP_REFUSAL,
        troubleshooting=residency_mode.TROUBLESHOOTING))


def _effective_slab(worker_args: Mapping[str, Any] | None, family: str) -> bool | str:
    """Resolve the residency mode the load will use.

    ``auto`` follows the worker's vouched-family set. An unvouched family pays
    the stock host copy, charged at 1.1x weights. Comfy-managed residency has
    neither that host copy nor slab ownership and is resolved from full policy.
    """
    if residency_mode.requested(worker_args):
        return residency_mode.MODE_COMFY_MANAGED
    requested = (worker_args or {}).get("slab_weights")
    if isinstance(requested, bool):
        return requested
    return family in capacity_fit.SLAB_VOUCHED_FAMILIES


def _fsdp_shard_world(mesh: Any, worker_args: Mapping[str, Any] | None,
                      quant_kind: str) -> int:
    """The world of the streaming shard build this load will run, or 0.

    0 is the whole-file price, so every unknown keeps the refusing number. Only
    the price moves: a refusal still reaches the same rescue, quarantine and
    consent arms.
    """
    try:
        from ..topology import declares_fsdp

        if not declares_fsdp(getattr(mesh, "topology_preset", "auto")):
            return 0
        if residency_mode.requested(worker_args):
            return 0  # comfy-managed owns the weights, and is refused above
        # Only a full-precision file takes the streaming build. A quantized
        # one takes FSDP2's direct wrap, where each block moves at full size,
        # and the block-scaled kinds shard nothing at all (adapters/fsdp.py),
        # so both keep the whole-file price. Slab policy does not enter this:
        # ``worker_env.slab_mode_effective`` is False under FSDP, so a sharded
        # rank takes no slab whatever it asked for. Two 61.7 GiB bf16 H3 legs at
        # uly2+fsdp built shards with slab_weights on (2026-09-05, 2026-09-07).
        from ..adapters.fsdp import (
            FSDP_ADMITTED_CHECKPOINT_KINDS,
            FSDP_ADMITTED_QUANT_KINDS,
        )

        if quant_kind not in FSDP_ADMITTED_CHECKPOINT_KINDS - FSDP_ADMITTED_QUANT_KINDS:
            return 0
        # bool is an int, so True must not read as a world; world 1 shards
        # nothing and pays the whole file.
        world = getattr(mesh, "world", None)
        if isinstance(world, bool) or not isinstance(world, int) or world < 2:
            return 0
        # And this host must hold exactly one of those shards, or the file is
        # the price again: rank 0's host runs `gpus_per_host` of them, and a
        # this_host() mesh runs the whole world in children of this process,
        # where the head holds every shard and the whole file is what it costs.
        per_host = getattr(getattr(mesh, "handle", None), "gpus_per_host", None)
        if isinstance(per_host, bool) or per_host != 1:
            return 0
        return world
    except Exception:  # every unknown keeps the whole-file price
        return 0


def _register_offer(descriptor: Mapping[str, Any]) -> None:
    """Publish a best-effort card without masking the typed refusal."""
    try:
        consent_pending.register_pending(dict(descriptor))
    except Exception as exc:  # Defensive: registration must not replace refusal.
        log.warning("pending consent not registered (%r)", exc)


def _combo_identity(unet_name: str, options: Mapping[str, Any] | None,
                    path: str) -> tuple[str, Any]:
    """Return the loader node's LoRA-free gate combination and artifact digest."""
    return consent_rescue.combo_identity(unet_name, options, path)


def memo_context(unet_name: str, options: Mapping[str, Any] | None) -> dict[str, str]:
    """Return the LoRA-free loader consent context.

    Adding a LoRA creates a different residency risk and requires new consent.
    """
    return consent_rescue.memo_context(unet_name, options)


def _estimate(
    profile: driver_footprint.DriverStackProfile, *, mem_available: int,
    weight_bytes: int, weights_resident: bool, co_resident: bool,
    slab_weights: bool | str | None, reserve: int, fsdp_world: int = 0,
) -> driver_footprint.DriverFootprintEstimate:
    """Estimate with the pending driver stack folded into the reserve.

    This keeps ``usable``, ``shortfall``, and ``dit_ceiling`` in one budget.
    """
    return driver_footprint.estimate_driver_footprint(
        profile=profile, mem_available=mem_available, weight_bytes=weight_bytes,
        weights_resident=weights_resident, co_resident=co_resident,
        slab_weights=slab_weights, reserve=reserve, fsdp_world=fsdp_world)


@dataclass(frozen=True, slots=True)
class _Shape:
    """Everything both windows are measured against, resolved once."""

    profile: driver_footprint.DriverStackProfile
    mem_available: int
    weight_bytes: int
    weights_resident: bool
    co_resident: bool
    floor_reserve: int
    stack_bytes: int
    fsdp_world: int = 0
    # The driver models a deferred load finds already loaded; 0 for an eager load.
    load_stack_bytes: int = 0


def _windows(shape: _Shape, *, slab: bool | str) -> tuple[
        driver_footprint.DriverFootprintEstimate,
        driver_footprint.DriverFootprintEstimate]:
    """Estimate the sequential transient-load and settled-resident windows.

    A stock load peaks near 2.1x the artifact, the weights plus the host copy
    they are placed from (``capacity_fit.STOCK_PLACEMENT_RATIO``, measured
    2026-10-01). The settled window charges weights at 1.0x, above the 0.87x
    measured on 2026-08-04, plus the unbuilt driver stack. Slab removes only
    the transient arena.

    A shard build takes one price in both windows: the build's own transient
    sits inside the shard charge, so the resident window carries it too and is
    conservative by that much rather than printing two numbers for one
    placement. Slab is not a lever there and the world governs both windows.
    """
    transient = _estimate(
        shape.profile, mem_available=shape.mem_available, weight_bytes=shape.weight_bytes,
        weights_resident=shape.weights_resident, co_resident=shape.co_resident,
        slab_weights=slab, reserve=shape.floor_reserve + shape.load_stack_bytes,
        fsdp_world=shape.fsdp_world)
    settled = _estimate(
        shape.profile, mem_available=shape.mem_available, weight_bytes=shape.weight_bytes,
        weights_resident=shape.weights_resident, co_resident=shape.co_resident,
        slab_weights=True, reserve=shape.floor_reserve + shape.stack_bytes,
        fsdp_world=shape.fsdp_world)
    return transient, settled


def _lead(binding: driver_footprint.DriverFootprintEstimate, window: str, shape: _Shape, *,
          unet_name: str, family: str, quant_kind: str, terms: DriverStackTerms,
          deferred: bool, prompt: object, slab: bool | str, mesh: Any,
          worker_args: Mapping[str, Any]) -> dict[str, str]:
    """The plain sentence, and what would fit, each alternative priced in both windows."""
    def fits(alt: _Shape) -> bool:
        return all(estimate.fits for estimate in _windows(alt, slab=slab))

    def stack_for(graph: object) -> DriverStackTerms:
        return credited_terms(driver_stack_terms(
            graph, family=family, sp_degree=activation_sp_degree(mesh, family)))

    lead = graph_load_price.plain_lead(
        binding, window=window, unet_name=unet_name, deferred=deferred,
        models=terms.models if shape.load_stack_bytes else ())
    return {"lead": lead, "fits": graph_load_price.alternatives(
        shape, fits=fits, unet_name=unet_name, family=family, quant_kind=quant_kind,
        terms=terms, deferred=deferred, prompt=prompt, stack_for=stack_for,
        pair=not residency_mode.requested(worker_args))}


def preflight_loader_footprint(
    mesh: Any, unet_name: str, options: Mapping[str, Any] | None,
    prompt: object = None,
) -> consent_rescue.RescueLoad | None:
    """Refuse an unaffordable eager load or apply an allowed slab rescue.

    This must run above ``ensure_live`` so refusal precedes fleet healing and
    rank mutation. It fails open except for a completed over-budget estimate
    and a forced family this checkpoint cannot carry.
    """
    try:
        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if path is None or not _slab_capable_path(str(path)):
            # Loader errors handle missing or slab-ineligible checkpoints.
            return None
        path = str(path)
        from ..adapters.detect import CheckpointSniffError, sniff_checkpoint

        forced_family = family_select.override_from_worker_args(
            getattr(mesh, "worker_args", None))
        quant_kind = ""
        try:
            sniffed, quant_kind = sniff_checkpoint(path)
            family = family_select.effective_family(sniffed, forced_family)
        except CheckpointSniffError:
            if forced_family is None:
                return None  # unreadable header degrades to a no-op, never a refusal
            # An unread kind keeps the whole-file price below, never the shard.
            family = forced_family
        if forced_family is not None:
            family_select.assert_override_admits_checkpoint(forced_family, path)
        # This warning sits above the capacity switches below: they decide
        # whether an estimate can refuse, and none of them may silence a report
        # about the architecture the file will build.
        family_select.warn_auto_admits_checkpoint(family, path)
        profile = driver_footprint.DRIVER_STACK_FAMILIES.get(family)
        if (profile is None or driver_footprint.preflight_disabled()
                or not mesh_safety.gpu_is_integrated()
                or driver_footprint.dtype_cast_requested(options)):
            # A cast's resident size differs from file size, so this estimate
            # cannot refuse safely.
            return None
        avail = mesh_safety.mem_available_bytes()
        if avail is None:
            return None  # off Linux: host MemAvailable bounds nothing here

        live_args = getattr(mesh, "worker_args", None)
        # Configured policy included, so the driver and the worker use the same
        # reserve and the same residency levers.
        worker_args = effective_worker_args(mesh)

        terms = credited_terms(driver_stack_terms(
            prompt, family=family,
            sp_degree=activation_sp_degree(mesh, family)))
        if not terms.resolved and _first_unpriced_stack_warn(family):
            # A direct node call carries no hidden prompt, so the estimator priced
            # only the weights. Say so once per family, so a post-mortem does
            # not have to infer it from a docstring.
            log.warning(
                "loader footprint preflight: no readable graph, headless call. "
                "This process charges only the checkpoint weights for %s; its "
                "driver stack (text encoder, VAEs, reference and guide encodes, "
                "render activations) stays unpriced. Gate capacity in the caller "
                "(docs/TROUBLESHOOTING.md #85).", family)
        floor_reserve = driver_footprint.reserve_bytes(worker_args)
        stack_bytes = terms.total
        weight_bytes = driver_footprint.file_size_bytes(path)
        co_resident = driver_footprint.rank0_co_resident(mesh)
        # A declared FSDP topology shards the checkpoint, so the whole file is
        # the price of a stock load this build never makes.
        fsdp_world = _fsdp_shard_world(mesh, worker_args, quant_kind)
        memo_key = _memo_key(unet_name, fsdp_world)
        # Under auto the load waits for the first render, by which time the
        # driver holds every model this graph loads (graph_load_price).
        deferred = graph_load_price.load_is_deferred(mesh)
        shape = _Shape(
            profile=profile, mem_available=avail, weight_bytes=weight_bytes,
            weights_resident=memo_key in _CLEARED_UNETS, co_resident=co_resident,
            floor_reserve=floor_reserve, stack_bytes=stack_bytes,
            fsdp_world=fsdp_world, load_stack_bytes=terms.models_bytes if deferred else 0)
        effective_slab = _effective_slab(worker_args, family)
        transient, settled = _windows(shape, slab=effective_slab)
        if transient.fits and settled.fits:
            remember(_CLEARED_UNETS, memo_key)
            note_stack_charged(terms)
            if fsdp_world:
                # The admitting path prints no card, so without this line the
                # shard price leaves no evidence in the journal.
                log.info(
                    "loader footprint preflight: FSDP shard build priced for %s "
                    "at world %d, charging %.2f of the file, %.1f GiB; load "
                    "window %.1f GiB usable, resident window %.1f GiB usable, "
                    "MemAvailable %.1f GiB", unet_name, fsdp_world,
                    settled.weight_share, settled.weight_bytes / _GIB,
                    transient.usable / _GIB, settled.usable / _GIB, avail / _GIB)
            return None

        # The window short by the most, the one the Old model card prices too.
        binding, window = max(((settled, WINDOW_SETTLED), (transient, WINDOW_TRANSIENT)),
                              key=lambda pair: (not pair[0].fits, pair[0].shortfall))
        words: dict[str, Any] = {
            "profile": profile, "floor_reserve": floor_reserve, "window": window,
            "load_models": shape.load_stack_bytes if deferred else None,
            **_lead(binding, window, shape, unet_name=unet_name, family=family,
                    quant_kind=quant_kind, terms=terms, deferred=deferred, prompt=prompt,
                    slab=effective_slab, mesh=mesh, worker_args=worker_args)}
        blocked = consent_projection.projection_blocker(worker_args)
        rescue_transient, rescue_settled = _windows(shape, slab=True)
        rescue = (rescue_settled if rescue_transient.fits and rescue_settled.fits
                  and effective_slab is not True and not blocked else None)
        try:
            combo, artifacts = _combo_identity(unet_name, options, path)
        except gate_artifacts.UnresolvedArtifactIdentityError as exc:
            # The load already failed capacity pricing. Preserve that refusal:
            # the outer estimator-error handler would otherwise admit it unpriced.
            raise driver_footprint.DriverFootprintCapacityError(refusal(
                RefusalClass.CAPACITY,
                loader_refusal(binding, terms, unet_name=unet_name, rescue=None, **words)
                + " " + graph_load_price.unstable_identity_clause(exc.names),
                guard=LOADER_GUARD, waivable=False)) from exc
        scope = consent_rescue.load_capability_context(worker_args)
        quarantine = consent_rescue.quarantine_decision_for_load(
            combo, artifacts, scope)
        if quarantine.state != "clear":
            quarantine_message = (
                loader_refusal(binding, terms, unet_name=unet_name, rescue=None, bypass=False,
                               **words)
                + f" Slab residency cannot be offered here either: {quarantine.reason}. ")
            if quarantine.state == "fail":
                raise driver_footprint.DriverFootprintCapacityError(refusal(
                    RefusalClass.KNOWN_WRONG,
                    quarantine_message + "There is no waiver for this refusal."))
            suffix = (
                "No slab-residency consent can be granted until the identity-gate "
                "evidence is readable." if quarantine.state == "unknown" else
                "There is no waiver for this refusal.")
            raise driver_footprint.DriverFootprintCapacityError(refusal(
                RefusalClass.CAPACITY,
                quarantine_message + suffix,
                guard=LOADER_GUARD, waivable=False))
        grant = consent_rescue.live_grant(
            path=path, unet_name=unet_name, options=options,
            combo=(combo, artifacts), scope=scope)
        if grant is not None and rescue is not None:
            consent_projection.project_scoped(
                mesh, live_args if isinstance(live_args, dict) else {}, grant, unet_name)
            consent_projection.audit_non_memo_grant(
                grant, combo_key=combo, artifacts=artifacts, unet_name=unet_name,
                path=path, context=memo_context(unet_name, options))
            remember(_CLEARED_UNETS, memo_key)
            note_stack_charged(terms)
            log.info("loader footprint preflight: consent %r present for %s, "
                     "loading slab-resident", RESCUE_KIND, unet_name)
            return consent_rescue.RescueLoad(
                combo_key=combo, artifacts=consent_rescue.artifacts_digest(artifacts),
                unet_name=unet_name,
                path=path, consent_id=str(getattr(grant, "id", "")),
                worker_args={**worker_args, "slab_weights": True},
                measured={
                    "probe": LOADER_GUARD,
                    "checkpoint_bytes": int(weight_bytes),
                    "mem_available_bytes": int(avail),
                    "driver_projection_bytes": int(settled.projected),
                    "headroom_bytes": int(rescue_settled.usable - rescue_settled.projected),
                })

        # The head's Worker shares this box: an old model it holds may be in the way.
        clear = old_model_clear.loader_offer(mesh, unet_name, options, windows=(transient, settled),
                                             keep=unet_names(prompt), window=window)
        if clear is not None and clear.helps:
            old_model_clear.refuse_at_loader(clear, loader_refusal(
                binding, terms, unet_name=unet_name, rescue=None, **words))
        message = loader_refusal(binding, terms, unet_name=unet_name, rescue=rescue, **words)
        if clear is not None:
            message = f"{message} {old_model_clear.held_text(clear)}"
        if blocked:
            # Report the existing residency constraint without offering override.
            # The merged copy above cannot name a session quarantine; the graph's
            # own dict can (gate_quarantine_scope.session_blocker).
            from .gate_quarantine_scope import session_blocker

            message = (f"{message} Slab residency is not offered here: "
                       f"{session_blocker(live_args) or blocked}, so this "
                       "refusal is about the setting on the graph, not about a consent.")
        elif grant is not None:
            # Consent cannot disable the guard when slab also exceeds capacity.
            message = (f"{message} A slab-residency consent is on file for this "
                       "checkpoint, but slab residency does not fit this graph either.")
        if rescue is None or grant is not None:
            # No applicable rescue remains, so omit the panel action.
            raise driver_footprint.DriverFootprintCapacityError(refusal(
                RefusalClass.CAPACITY, message, guard=LOADER_GUARD,
                waivable=False))
        spec = consent_pending.KIND_SPECS[RESCUE_KIND]
        try:
            _register_offer({
                "kind": RESCUE_KIND, "path": path, "memo_context": memo_context(unet_name, options),
                "combo_key": combo, "artifacts": consent_rescue.artifacts_digest(artifacts),
                "artifacts_legacy": artifacts.legacy,
                "artifacts_legacy_complete": artifacts.legacy_complete,
                "artifact": os.path.basename(unet_name), "unet_name": unet_name,
                "target_guard": LOADER_GUARD, "loras": 0, "scope": scope,
                "numbers": (f"needs {_gib(binding.projected)} GiB, "
                            f"{_gib(binding.usable)} GiB usable"),
            })
        except Exception as exc:  # the card is best effort, the refusal is not
            log.warning("consent card not built for %s (%r)", unet_name, exc)
        raise driver_footprint.DriverFootprintCapacityError(refusal(
            RefusalClass.CAPACITY, message, guard=LOADER_GUARD, waivable=True,
            panel_action=PanelAction(label=spec.primary_label, env=spec.env_var)))
    except (driver_footprint.DriverFootprintCapacityError, UnsupportedModelError):
        # Both are refusals, not estimator noise: the second is the family
        # override rejecting a checkpoint it cannot admit, and it must reach
        # the user instead of degrading to a no-op.
        raise
    except Exception:  # a broken estimator never blocks a load
        return None
    return None
