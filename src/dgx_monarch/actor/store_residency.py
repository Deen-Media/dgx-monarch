"""Resolve checkpoint memory residency through one ordered ladder.

Rungs, in the order `resolve` decides them: managed placement, explicit slab,
vouched auto-slab, the FSDP transient, then the stock fit, which yields
explicit stock or stock-when-it-fits, and on a miss the rescue offer or a
consented rescue. Explicit stock is terminal because it may represent identity
quarantine. Managed placement resolves first and cannot mix with slab or
low-RSS policy.

Consent grants availability, not accuracy. The driver normally projects a
grant into every rank's worker arguments; a directly injected grant remains
non-authoritative and does not update memos or vouching state. Rescue is offered
only when slab can run safely; otherwise a capacity refusal names the blocker.

``resolve`` is a wrapper: ``capacity_quote.price`` holds the arithmetic and the
rung vocabulary, and the raises stay here, so each refusal keeps its site and
sentence. The rest of this module is what the driver never needs: the
operator's text, the wire tail and the store's own evidence.

The module itself imports neither torch nor ComfyUI, so it stays CPU-testable.
Mixed worker and driver versions fail closed because an unknown consent
descriptor is inert text and a missing descriptor grants nothing.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .. import capacity_fit, consent_descriptor, mesh_safety, residency_mode
from ..capacity_fit import SlabFit, StockFit
from ..consent_descriptor import ConsentDescriptor, SlabResidencyRescueOffer
from ..gate_audit_evidence import measured_tag
from ..log import get_logger
from ..mesh_safety import StockLoadCapacityError
from ..refusal import PanelAction, RefusalClass, refusal, untagged
from ..residency_mode import PINNED_STAGING_FACTOR
from ..transfer_utils import failure_summary, safe_call
from . import capacity_quote, store_slab_admit

# One home for the rung words and the shared price helpers; a caller that
# spells them ``store_residency.X`` resolves the same object.
from .capacity_quote import (  # noqa: F401 - re-exported for the ladder's callers
    RUNG_COMFY_MANAGED,
    RUNG_CONSENTED_RESCUE,
    RUNG_EXPLICIT,
    RUNG_EXPLICIT_STOCK,
    RUNG_STOCK_FITS,
    RUNG_VOUCHED_AUTO,
    comfy_managed_active,
    compile_dit_active,
    default_resolve_lora_path,
    managed_required_bytes,
)
from .rescue_offer import _PANEL_ACTION, _RESCUE_SPEC, build_descriptor, memo_context
from .rescue_offer import host_name as _host

log = get_logger(__name__)

# Spelled as a literal, not imported: the refusal-class walker reads the guard
# off the module's own assignment, and a name it cannot resolve records a class
# C refusal that names no boundary.
RESCUE_GUARD = "stock_load_preflight"
_TROUBLESHOOTING = 53

# A journal label, never a decision rung: ``stock_fits`` means three different
# things (priced and fits, the probe did not apply, an FSDP transient check ran
# instead) and the record cannot tell them apart.
RUNG_STOCK_SKIPPED = "stock_skipped"


@dataclass(frozen=True, slots=True)
class ResidencyDecision:
    """Selected residency, deciding rung, and capacity evidence."""

    use_slab: bool
    rung: str
    reason: str
    auto_retry_eligible: bool
    fit: StockFit | None = None
    consent_id: str = ""
    slab_fit: SlabFit | None = None
    # The managed and FSDP rungs price in their own units, which no ``StockFit``
    # can hold: their wall's own block is kept here so the journal can quote it.
    measured: dict[str, Any] | None = None

    @property
    def residency(self) -> str:
        if self.rung == RUNG_COMFY_MANAGED:
            return residency_mode.MODE_COMFY_MANAGED
        return (residency_mode.MODE_SLAB if self.use_slab
                else residency_mode.MODE_STOCK)


def resolve(
    *,
    path: str,
    unet_name: str,
    model_options: dict,
    slab_weights: bool | str,
    slab_capable_path: bool,
    lora_low_rss: bool,
    fsdp_launch: bool,
    blocked_reason: str, world: int | None = None,
    authoritative_slab_retry: bool,
    memoized_family: Callable[[str], str | None],
    vouched_families: frozenset[str],
    rescue_consent: dict | None = None,
    fit_probe: Callable[[str, dict], StockFit] | None = None,
    file_identity: Callable[[str], str] | None = None,
    compile_dit: Callable[[], bool] | None = None,
    comfy_managed: Callable[[], bool] | None = None,
    resolve_lora_path: Callable[[str], str] | None = None,
    family_hint: str | None = None,
    family_override: str | None = None,
    node_options: dict | None = None,
    lora_stack: list | None = None,
    reserve_bytes: int = 0,
) -> ResidencyDecision:
    """Run the ladder: return a decision or raise the typed refusal.

    ``capacity_quote.price`` does the arithmetic. The driver's fleet agreement
    reads the same rows from the same function, so a rank's own wall and the
    pre-load agreement cannot price one load two ways. Probe defaults bind
    inside ``price`` at call time, so tests and callers can still replace them
    here.
    """
    row = capacity_quote.price(
        path=path, unet_name=unet_name, model_options=model_options,
        slab_weights=slab_weights, slab_capable_path=slab_capable_path,
        lora_low_rss=lora_low_rss, fsdp_launch=fsdp_launch,
        blocked_reason=blocked_reason, world=world,
        authoritative_slab_retry=authoritative_slab_retry,
        memoized_family=memoized_family, vouched_families=vouched_families,
        rescue_consent=rescue_consent, fit_probe=fit_probe,
        file_identity=file_identity, compile_dit=compile_dit,
        comfy_managed=comfy_managed, resolve_lora_path=resolve_lora_path,
        family_hint=family_hint,
        family_override=family_override, node_options=node_options,
        lora_stack=lora_stack, reserve_bytes=reserve_bytes,
    )
    fit = _stock_fit(row)
    slab_fit = _slab_fit(row)
    if row.verdict == capacity_quote.VERDICT_REFUSE and row.rung != RUNG_COMFY_MANAGED:
        # The managed rung is the exception: its wall is preload_capacity_check,
        # which prices the load again just before ComfyUI and raises the managed
        # card. A raise here would take the bare branch below, which drops the
        # managed card's staging cause, fixes and entry pointer.
        if row.use_slab and slab_fit is not None:
            store_slab_admit.admit(slab_fit, unet_name, _host(),
                                   reserve_bytes=reserve_bytes)
        if row.blocker is not None and fit is not None:
            raise StockLoadCapacityError(refusal(
                RefusalClass.CAPACITY, _no_rescue_text(unet_name, fit, row.blocker),
                guard=RESCUE_GUARD, waivable=False,
                troubleshooting=_TROUBLESHOOTING) + _wire_tail(fit, None))
        # The FSDP rung. Its own check built this sentence and this wrapper
        # re-raises it under the same class and guard, so the operator meets one
        # refusal with one boundary named, not an echo of another module's tag.
        raise StockLoadCapacityError(refusal(
            RefusalClass.CAPACITY, untagged(row.reason),
            guard=RESCUE_GUARD, waivable=False))
    if row.verdict == capacity_quote.VERDICT_RESCUE and fit is not None:
        raise SlabResidencyRescueOffer(refusal(
            RefusalClass.CAPACITY, _rescue_offer_text(row.descriptor, fit, row.reason),
            guard=RESCUE_GUARD, waivable=True, troubleshooting=_TROUBLESHOOTING,
            panel_action=PanelAction(label=_PANEL_ACTION, env=_RESCUE_SPEC.env_var),
        ) + _wire_tail(fit, row.descriptor))
    if row.rung == RUNG_CONSENTED_RESCUE and row.descriptor is not None:
        log.info("capacity rescue consented for %s (consent %s): loading slab-resident",
                 row.descriptor.unet_name, row.descriptor.consent_id)
    # A forced family never inherits vouching, and only the two stock rungs can
    # promote a later load: every other rung already placed what it decided.
    auto = slab_weights == "auto" and not family_override
    eligible = bool(auto and slab_capable_path and not authoritative_slab_retry
                    and not fsdp_launch)
    return ResidencyDecision(
        row.use_slab, row.rung, row.reason,
        eligible if row.rung in (RUNG_STOCK_FITS, RUNG_EXPLICIT_STOCK) else False,
        fit,
        row.descriptor.consent_id if row.descriptor is not None else "",
        slab_fit,
        row.measured,
    )


def _stock_fit(row: capacity_quote.RankPrice) -> StockFit | None:
    """Rebuild the stock probe's own result from the row it produced.

    The wire row carries the numbers; the decision and its refusal sentences
    want the dataclass. This is the inverse of ``measured()`` and not a second
    arithmetic: nothing here recomputes a price.
    """
    block = row.measured
    if block is None or block.get("probe") != capacity_fit.PROBE_STOCK_LOAD_FIT:
        return None
    return StockFit(bool(block["fits"]), bool(block["applies"]),
                    int(block["checkpoint_bytes"]), block["mem_available_bytes"],
                    "" if block["applies"] else row.reason)


def _slab_fit(row: capacity_quote.RankPrice) -> SlabFit | None:
    """The same inverse for the slab probe's block."""
    block = row.slab_measured
    if block is None or block.get("probe") != capacity_fit.PROBE_SLAB_LOAD_FIT:
        return None
    return SlabFit(bool(block["fits"]), bool(block["applies"]),
                   int(block["checkpoint_bytes"]), block["mem_available_bytes"],
                   int(block.get("floor_bytes") or 0),
                   "" if block["applies"] else row.reason,
                   int(block.get("lora_bytes") or 0))


def _no_rescue_text(unet_name: str, fit: StockFit, blocker: tuple[str, str]) -> str:
    why, options = blocker
    return (
        f"stock residency cannot load {unet_name} on {_host()}: the load needs "
        f"{fit.required_gib} GiB (a {fit.size_gib} GiB file, the host copy it is placed "
        f"from and the host floor) and only {fit.avail_gib} GiB of unified memory is "
        f"available, so it would be killed out of memory. Slab rescue is unavailable here "
        f"because {why}. What would fit: {options}. This refusal is capacity-protective "
        "(class C, StockLoadCapacityError family); nothing was loaded and nothing was quarantined."
    )


def _rescue_offer_text(descriptor: ConsentDescriptor, fit: StockFit,
                       cold_memo: str = "") -> str:
    """Build the body; ``refusal()`` owns panel and headless instructions.

    ``cold_memo`` is set only where the family memo was read on this box and
    came back empty, so a cold box reads as cold rather than as full.
    """
    clause = f" {cold_memo}" if cold_memo else ""
    return (
        f"capacity rescue available for {descriptor.unet_name} on {_host()}: stock residency "
        f"needs {fit.required_gib} GiB of unified memory for a {fit.size_gib} GiB file and "
        f"only {fit.avail_gib} GiB is available, so the stock load would be killed out of "
        f"memory. Zero-copy slab residency fits this checkpoint, and "
        f"{consent_descriptor.EVIDENCE_BYTE_VERIFIED}.{clause} This refusal is "
        "capacity-protective (StockLoadCapacityError family); nothing was loaded and "
        "nothing was quarantined."
    )


def _wire_tail(fit: StockFit, descriptor: ConsentDescriptor | None) -> str:
    """Append measured data and consent metadata to the refusal text.

    Monarch transport does not preserve exception attributes. Append metadata
    after the human message so descriptor bounds cannot consume its text limit.
    Encoding failure must not suppress the refusal.
    """
    parts = []
    try:
        parts.append(measured_tag(fit.measured()))
    except (TypeError, ValueError) as exc:
        safe_call(log.warning, "capacity refusal could not carry its measured numbers (%s)", failure_summary(exc))
    if descriptor is not None:
        try:
            parts.append(consent_descriptor.encode(descriptor))
        except (TypeError, ValueError) as exc:
            safe_call(log.warning, "consent descriptor could not be encoded (%s)", failure_summary(exc))
    return ("\n" + "\n".join(parts)) if parts else ""


def preload_capacity_check(
    decision: ResidencyDecision,
    path: str,
    unet_name: str,
    model_options: dict,
    *,
    preflight: Callable[[str, str, dict], None],
    pinned_staging: bool | None = None,
    fsdp_launch: bool = False,
    fsdp_checkpoint_kind: str | None = None,
    fsdp_world: int | None = None,
    reserve_bytes: int = 0,
    lora_stack: list[dict] | None = None,
    lora_low_rss: bool = False,
    resolve_lora_path: Callable[[str], str] | None = None,
) -> None:
    """Apply the capacity preflight for the selected rung.

    Every rung is priced here, including the slab ones: this is the last
    monarch-owned moment before ComfyUI is entered, so a refusal raised from it
    is still a refusal where nothing was loaded. Stock residency prices with
    ``capacity_fit.stock_load_fit`` and calls the injected ``preflight`` only
    when that probe does not apply; an FSDP launch re-prices its transient.
    Managed residency charges the artifact plus the absolute host floor the
    stock and slab prices charge, and no legacy load arena, because DynamicVRAM
    is what removes it. On top it charges the staging copy: with pinning
    disabled the weights page out of the checkpoint mapping and one resident
    copy is the whole cost, while a pinned host buffer adds a second copy that
    neither swaps nor evicts. Only the second case needs a multiplier, so the
    wall reads which one ComfyUI will do rather than assuming. Driver
    preflights enforce any larger operator reserve.
    """
    if decision.use_slab:
        # The second slab wall: memory moves between the rung's price and this
        # line. Same guard and typed error as the first wall, so a double
        # refusal is one refusal.
        store_slab_admit.admit(
            capacity_fit.slab_load_fit(
                path, model_options, reserve_bytes=reserve_bytes,
                lora_stack=lora_stack if lora_low_rss else None,
                resolve_lora_path=resolve_lora_path or default_resolve_lora_path),
            unet_name, _host(), reserve_bytes=reserve_bytes)
        return
    if decision.rung != RUNG_COMFY_MANAGED:
        if fsdp_launch:
            # resolve() already priced the FSDP transient; the stock full-copy
            # preflight is the wrong predicate for a sharded rank. Re-run the
            # exact shared price because memory can move after the first wall.
            if (model_options.get("dtype") is None
                    and mesh_safety.gpu_is_integrated()):
                size = os.path.getsize(path) if os.path.exists(path) else 0
                required = capacity_fit.fsdp_required_bytes(
                    size, fsdp_world, checkpoint_kind=fsdp_checkpoint_kind)
                available = mesh_safety.mem_available_bytes()
            else:
                available = None
            if available is not None and required > available:
                if fsdp_checkpoint_kind in {"bf16", "fp16"}:
                    load_kind = "the streaming build retains this rank's shards plus one block"
                elif fsdp_checkpoint_kind in {"fp8", "int8"}:
                    load_kind = "the direct wrapper keeps the full checkpoint live"
                else:
                    load_kind = (
                        "the checkpoint kind is unproven, so the conservative "
                        "price keeps the full checkpoint live")
                raise StockLoadCapacityError(refusal(
                    RefusalClass.CAPACITY,
                    f"the FSDP launch cannot load {unet_name}: {load_kind}; the load needs "
                    f"{required / 2**30:.1f} GiB including the "
                    f"{capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES / 2**30:.1f} GiB host floor, "
                    f"but only {available / 2**30:.1f} GiB is available.",
                    guard=RESCUE_GUARD, waivable=False) + "\n" + measured_tag(
                        capacity_fit.preflight_wall_measured(
                            size, available, required, False)))
            return
        fit = capacity_fit.stock_load_fit(path, model_options)
        if fit.applies:
            if not fit.fits:
                raise StockLoadCapacityError(refusal(
                    RefusalClass.CAPACITY,
                    f"stock residency cannot load {unet_name}: the load needs "
                    f"{fit.required_gib} GiB including the host copy it is placed from and the "
                    f"{capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES / 2**30:.1f} GiB host floor, "
                    f"but only {fit.avail_gib} GiB is available.",
                    guard=RESCUE_GUARD, waivable=False) + _wire_tail(fit, None))
            return
        preflight(path, unet_name, model_options)
        return
    # File size cannot predict a dtype cast, and host memory does not bound a
    # discrete GPU's model capacity.
    if model_options.get("dtype") is not None or not mesh_safety.gpu_is_integrated():
        return
    avail = mesh_safety.mem_available_bytes()
    if avail is None:
        return
    if pinned_staging is None:
        from . import comfy_dynamic

        pinned_staging = comfy_dynamic.pinned_staging_active()
    size = os.path.getsize(path) if os.path.exists(path) else 0
    floor = capacity_fit.ABSOLUTE_HOST_FLOOR_BYTES
    required = managed_required_bytes(size, pinned_staging)
    if required <= avail:
        return
    staging = (
        (f"ComfyUI is staging through a pinned host buffer here, so the charge is "
         f"{PINNED_STAGING_FACTOR:.2f}x the checkpoint, or {required / 2 ** 30:.1f} GiB "
         f"with the {floor / 2 ** 30:.1f} GiB host floor every residency on this box "
         "is priced over. That multiplier was measured on 2026-08-12: a pinned buffer is "
         "sized at twice the model and holds a second copy of every weight in the "
         "same unified pool the device computes from, and it neither swaps nor "
         "evicts. Turning pinned memory off (leave disable_pinned_memory alone, or "
         "set it true) drops the charge to one copy and lets the weights page "
         "straight out of the checkpoint mapping")
        if pinned_staging else
        (f"The charge is one resident copy plus the {floor / 2 ** 30:.1f} GiB host "
         f"floor every residency on this box is priced over, or {required / 2 ** 30:.1f} "
         "GiB. Weights page out of the checkpoint mapping here, so there is no second "
         "copy to remove and no arena to reclaim; a host with nothing left to page "
         "into cannot be rescued by lazy placement")
    )
    raise StockLoadCapacityError(refusal(
        RefusalClass.CAPACITY,
        f"comfy-managed residency cannot load {unet_name} on {_host()}: the checkpoint "
        f"is {size / 2 ** 30:.1f} GiB and only {avail / 2 ** 30:.1f} GiB of unified "
        f"memory is available, {(required - avail) / 2 ** 30:.1f} GiB short of what this "
        f"load needs. {staging}. What would fit: free unified memory on this host (drop "
        "caches, stop co-resident LLM servers, dgxm reap), use a pruned or quantized "
        "artifact, or turn the Init node's comfy_managed widget off and reset the "
        "attached mesh or restart the Worker service (`dgxm restart`), so the slab "
        "rescue ladder applies again. This refusal is capacity-protective (class C, "
        "StockLoadCapacityError family); nothing was loaded and nothing was quarantined.",
        guard=RESCUE_GUARD, waivable=False, troubleshooting=_TROUBLESHOOTING))


def _descriptor(
    *,
    unet_name: str,
    path: str,
    fit: StockFit,
    model_options: dict,
    file_identity: Callable[[str], str],
    family_hint: str | None,
    node_options: dict | None = None,
    lora_stack: list | None = None,
) -> ConsentDescriptor:
    """The ladder's seam onto ``rescue_offer.build_descriptor``.

    It passes this module's own ``memo_context`` and ``_host`` rather than the
    builder's defaults, so a diagnostic that replaces either name here still
    reaches the descriptor that a refusal carries.
    """
    return build_descriptor(
        unet_name=unet_name, path=path, fit=fit, model_options=model_options,
        file_identity=file_identity, family_hint=family_hint,
        node_options=node_options, lora_stack=lora_stack,
        context=memo_context, host=_host,
    )


def describe(stored: Any) -> dict[str, Any]:
    """Residency evidence for the model store's status snapshot.

    Load endpoints carry a slab certificate to the driver through
    ``ModelStore.snapshot``. A load inside ``sample`` or ``compute_sigmas``
    returns no snapshot, so the driver writes no row at load time. The worker
    still certifies that load and refuses a mismatch, and the identity gate
    reads its certificate through ``gate_swap_cycle`` (docs/DESIGN.md section 5.2).
    """
    if stored is None:
        return {"residency": None, "residency_rung": "", "slab_certificate": None}
    slab = getattr(stored, "slab", None)
    certificate = getattr(slab, "certificate", None) if slab is not None else None
    rung = getattr(stored, "residency_rung", "") or ""
    return {
        "residency": (residency_mode.MODE_COMFY_MANAGED if rung == RUNG_COMFY_MANAGED
                      else "slab" if slab is not None else "stock"),
        "residency_rung": rung,
        "slab_certificate": certificate.public() if certificate is not None else None,
    }


def policy_keeps_resident(stored: Any, slab_weights: bool | str) -> bool:
    """Whether a resident's storage class still matches a residency policy.

    A rung is evidence, not identity. ``stock_fits`` and ``explicit_stock``
    name one plain cudaMalloc residency, so a dispatch that respells stock
    policy keeps the resident copy. Unified memory does not free the first
    copy, so loading the same bytes again beside it is the capacity failure.
    Slab, comfy-managed, and stock storage stay distinct classes.
    """
    if stored is None:
        return True
    resident = describe(stored)["residency"]
    if resident == residency_mode.MODE_COMFY_MANAGED:
        return False
    if resident == residency_mode.MODE_SLAB:
        return slab_weights is not False
    return slab_weights is not True


def _gib(value: Any) -> float:
    return round(int(value or 0) / (1 << 30), 1)


def summary_line(stored: Any, decision: ResidencyDecision | None = None) -> str:
    """Summarize residency, pricing evidence, and the slab certificate for logs.

    Include whether the selected rung was priced and which probe ran. A rung
    name alone cannot distinguish a fitting load from a skipped probe.
    """
    evidence = describe(stored)
    certificate = evidence["slab_certificate"]
    tail = f" cert={certificate['summary']}" if certificate else ""
    rung = evidence["residency_rung"]
    priced = ""
    if decision is not None:
        fit: Any = decision.fit if decision.fit is not None else decision.slab_fit
        if rung == RUNG_STOCK_FITS and (decision.fit is None or not decision.fit.applies):
            rung = RUNG_STOCK_SKIPPED
        if fit is not None and fit.applies:
            priced = (f" priced={fit.required_gib}GiB against "
                      f"{fit.avail_gib}GiB by {fit.measured()['probe']}")
        elif decision.measured and decision.measured.get("probe") != capacity_quote.PROBE_REUSE:
            block = decision.measured
            priced = (f" priced={_gib(block['required_bytes'])}GiB against "
                      f"{_gib(block['mem_available_bytes'])}GiB "
                      f"by {block['probe']}")
        else:
            priced = " unpriced"
        priced += f" reason={decision.reason}"
    return f"residency={evidence['residency']} rung={rung}{priced}{tail}"
