"""Resolve and project capacity-rescue consent consistently across driver sites.
Class-K quarantine outranks consent; `consent_quarantine` owns the ledger read
that settles it. Worker args remain inside the capability context.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from .. import consent_pending
from ..log import get_logger
from .consent_quarantine import (  # noqa: F401  # Public consent_rescue API.
    QuarantineDecision,
    QuarantineQuery,
    quarantine_decision,
    quarantine_decision_for_load,
    quarantine_decisions,
    quarantine_query,
    quarantine_reason,
)

log = get_logger(__name__)

RESCUE_KIND = "rescue-slab"
QUARANTINE_LEVER = "slab_weights"
# Baked mode retains a full backup, so slab residency requires low-RSS. The
# worker otherwise collapses the request to stock and consumes extra capacity.
SLAB_PRECONDITION_LEVER = "lora_low_rss"


@dataclass(frozen=True, slots=True)
class RescueLoad:
    """Facts needed to join a consented load with its later certificate row."""

    combo_key: str
    artifacts: str
    unet_name: str
    path: str
    consent_id: str
    measured: dict[str, Any]
    worker_args: dict[str, Any] = field(default_factory=dict)


def _folded_certificate(load: RescueLoad, results: Sequence[Any], world: object,
                        slot: str) -> dict[str, Any] | None:
    """Every rank's byte-verify certificate for this slot, folded, or None."""
    from ..gate_audit import fold_rank_certificates

    if not _complete_rank_cohort(results, world):
        return None
    rows = [{"certificate": (row.get(slot) or {}).get("slab_certificate")}
            for row in results]
    return fold_rank_certificates(rows, cast(int, world))


def _complete_rank_cohort(results: Sequence[Any], world: object) -> bool:
    """Return whether these are exactly one well-formed response per rank."""
    if type(world) is not int or world <= 0 or len(results) != world:
        return False
    ranks: set[int] = set()
    for row in results:
        if not isinstance(row, Mapping):
            return False
        rank = row.get("rank")
        if type(rank) is not int or rank < 0 or rank >= world:
            return False
        ranks.add(rank)
    return ranks == set(range(world))


def record_rescue_row(load: RescueLoad | None, results: object, world: object = None,
                      slot: str = "cond", handle: Any = None) -> bool:
    """Write one driver-side certificate row for the loaded model slot.

    Each conditional or unconditional slot carries a separate certificate.
    Missing, partial, or wrong-world slab evidence fails closed after the load:
    drop the slot and raise a typed refusal. Confirmed stock fallback passed its
    own capacity check and owes no certificate: no slab bytes became resident.
    """
    if load is None or not isinstance(results, (list, tuple)) or not results:
        return False
    certificate: dict[str, Any] | None = None
    failure: Exception | None = None
    try:
        import folder_paths

        from .. import consent_store
        from ..gate_audit import record_rescue_certificate
        from ..gate_ledger import GateLedger, comfy_commit

        certificate = _folded_certificate(load, results, world, slot)
        if certificate is not None:
            return record_rescue_certificate(
                GateLedger(folder_paths.get_output_directory()), load.combo_key,
                load.artifacts, comfy_commit(),
                capability_context={"worker_args": dict(load.worker_args)},
                certificate=certificate, measured=dict(load.measured),
                consent_id=load.consent_id, unet_name=load.unet_name,
                file_identity=consent_store.file_identity(load.path))
    except Exception as exc:  # The load exists independently of its audit row.
        failure = exc
    if certificate is None:
        if _every_rank_took_stock(results, world, slot):
            log.warning("the consented load of %s fell back to stock residency, so no "
                        "capacity certificate exists and no CAPACITY_CERTIFIED row was written",
                        load.unet_name)
            return False
        raise _uncertified_slab_load(load, slot, handle)
    log.error("CAPACITY_CERTIFIED row not written for %s (%r)", load.unet_name, failure)
    return False


def _every_rank_took_stock(results: Sequence[Any], world: object, slot: str) -> bool:
    """Require one confirmed-stock report per expected worker rank and slot.

    Stock fallback needs no slab certificate, but missing or duplicate replies
    cannot establish fallback. Require callers to supply both world size and
    slot explicitly so outdated calls fail instead of silently skipping checks.
    """
    if not _complete_rank_cohort(results, world):
        return False
    for row in results:
        slot_state = row.get(slot)
        if (not isinstance(slot_state, Mapping)
                or slot_state.get("residency") != "stock"):
            return False
    return True


def _uncertified_slab_load(load: RescueLoad, slot: str, handle: Any) -> Exception:
    """Build the non-waivable refusal for an uncertified consented slab load.

    Missing certificate evidence is a class-P protocol failure, not measured
    class-K divergence. The message names a working alternative, as required
    by docs/DESIGN.md section 5.9.
    """
    from ..actor.slab_certificate import SlabCertificateError
    from ..refusal import RefusalClass, refusal

    _drop_uncertified_slot(load, handle)
    return SlabCertificateError(refusal(
        RefusalClass.PHYSICS,
        f"the consented pre-gate slab load of {load.unet_name} came back with no "
        f"byte-verify certificate for the {slot} slot, so its weights were never checked "
        "against the checkpoint. Every pre-gate slab load is certified or refused, and a "
        "certificate this driver cannot read is a protocol skew rather than measured "
        "wrongness, so no waiver can exist for it. The usual cause is a Worker service older "
        "than this driver. The driver asked the workers to unload their models. What "
        "works instead: restart the Worker service (`dgxm restart`) so both sides run one "
        "build, then queue again.",
        troubleshooting=54))


def _drop_uncertified_slot(load: RescueLoad, handle: Any) -> None:
    """Best-effort removal of an uncertified resident slot."""
    if handle is None:
        return
    try:
        handle.call_all("unload", timeout_s=600)
    except Exception as exc:  # The typed refusal remains the primary protection.
        log.error("uncertified slab load of %s could not be unloaded (%r); recycle the "
                  "fleet before the next render", load.unet_name, exc)


def memo_context(unet_name: str, options: Mapping[str, Any] | None,
                 loras: Sequence[Any] = ()) -> dict[str, str]:
    """Build the driver/worker residency consent key.

    The key excludes worker args, which the rescue changes, and topology, which
    changes with render shape. It matches ``actor/rescue_offer.memo_context``.
    """
    from ..gate_ledger import combo_key

    names = [str(entry.get("name", "")) for entry in (loras or ())
             if isinstance(entry, Mapping)]
    fields = {
        "combo_key": combo_key(unet_name, dict(options or {}), names),
        "weight_dtype": str((options or {}).get("weight_dtype", "default")),
    }
    try:
        # Use the registry's field set so readers and writers key memos alike.
        return dict(consent_pending.consent_context(RESCUE_KIND, **fields))
    except Exception as exc:  # A registry mismatch must not disable the check.
        log.warning("consent context rejected (%r)", exc)
        return fields


def combo_identity(unet_name: str, options: Mapping[str, Any] | None, path: str,
                   loras: Sequence[Any] = ()) -> tuple[str, Any]:
    """Build the gate ledger's combination key and artifact-set signature.

    Return the signature object so protocol-blind FAIL lookup can match its
    legacy spelling. Flattening it would hide legacy quarantine evidence.

    A sentinel identity raises here rather than composing a key: the lookups
    this feeds grant nothing on an error and everything on a digest that
    matches no row.
    """
    import folder_paths

    from ..gate_artifacts import refuse_unresolved
    from ..gate_ledger import artifact_set_signature, artifact_signature, combo_key

    names = [str(entry.get("name", "")) for entry in (loras or ())
             if isinstance(entry, Mapping)]
    labelled = [(f"diffusion_models/{unet_name}", artifact_signature(path))]
    for name in names:
        lora_path = folder_paths.get_full_path("loras", name)
        labelled.append(
            (f"loras/{name}", artifact_signature(lora_path) if lora_path else "missing"))
    refuse_unresolved(labelled)
    return (combo_key(unet_name, dict(options or {}), names),
            artifact_set_signature(signature for _name, signature in labelled))


def artifacts_digest(artifacts: Any) -> str:
    """The 64 hex artifact-set column value, from either spelling."""
    return str(getattr(artifacts, "current", artifacts) or "")


def load_capability_context(worker_args: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The capability a rescue would run under: this load, slab-resident.

    The same shape the consented load's own certificate row records, so the
    question the ledger is asked before the load and the scope the row lands in
    after it are one shape. ``None`` where the args carry a value the ledger
    cannot canonicalize, which is the contextless read to fall back on.
    """
    from .. import consent_store

    context = {"worker_args": {**dict(worker_args or {}), QUARANTINE_LEVER: True}}
    try:
        consent_store.canonical_context(context)
    except (TypeError, ValueError, RecursionError) as exc:
        log.warning("this load's capability context is not canonicalizable (%r)", exc)
        return None
    return context


def quarantine_reason_for_load(combo_key_value: str, artifacts: Any,
                               capability_context: dict[str, Any] | None) -> str:
    """The reason of ``quarantine_decision_for_load``, which documents its two reads."""
    return quarantine_decision_for_load(
        combo_key_value, artifacts, capability_context).reason


def live_grant(*, path: str, unet_name: str, options: Mapping[str, Any] | None,
               loras: Sequence[Any] = (), combo: tuple[str, str] | None = None,
               scope: dict[str, Any] | None = None) -> Any:
    """Resolve consent without raising; revoke a memo contradicted by FAIL.

    Use the load's capability context for both the preliminary quarantine check
    and grant resolution. Different scopes could repeatedly offer a card whose
    accepted memo is then refused. If the grant contains its own context, use
    that exact operator-approved context.
    """
    try:
        combo_key_value, artifacts = (
            combo if combo is not None else combo_identity(unet_name, options, path, loras))
        context = memo_context(unet_name, options, loras)
        grant = consent_pending.resolve(kind=RESCUE_KIND, path=path, context=context)
        if grant is None:
            return None
        granted_scope = getattr(grant, "capability_context", None)
        decision = quarantine_decision(combo_key_value, artifacts, granted_scope)
        grant_key = getattr(grant, "key", None)
        if decision.state == "fail" and isinstance(grant_key, str) and grant_key:
            # A gate-proven unsafe path cannot retain a live consent memo.
            _revoke_quarantined(path, context)
        if decision.state == "unknown" and granted_scope is None and scope is not None:
            decision = quarantine_decision(combo_key_value, artifacts, scope)
        if decision.state != "clear":
            return None
        return grant
    except Exception as exc:  # Missing evidence asks again and grants nothing.
        log.warning("consent lookup failed for %s (%r)", unet_name, exc)
        return None


def _revoke_quarantined(path: str, context: dict[str, str]) -> None:
    """Best-effort removal of a memo disproved by identity-gate FAIL."""
    try:
        from .. import consent_store
        from .consent_routes import revoke_memo_key

        key = consent_store.memo_key(RESCUE_KIND, path, dict(context))
        if revoke_memo_key(key):
            log.error("a persisted identity-gate FAIL revoked the slab-residency consent "
                      "for %s", path)
    except Exception as exc:
        log.warning("quarantined consent could not be revoked (%r)", exc)




def _graph_declares_fsdp(model: Any) -> bool:
    from ..topology import declares_fsdp

    return declares_fsdp(getattr(getattr(model, "mesh", None), "topology_preset", "auto"))


def unproven_quarantine_levers(model: Any) -> dict[str, bool]:
    """Choose residency settings after an incomplete identity check.

    Default to stock. Capacity-rescue consent can retain slab and its required
    low-RSS mode only without a LoRA swap path that still needs proof.

    FSDP LoRA has no stock fallback and requires ``lora_low_rss``, so retain that
    setting under an explicit FSDP topology. Disable ``slab_weights``; FSDP does
    not use slab storage.
    """
    levers: dict[str, bool] = {SLAB_PRECONDITION_LEVER: False, QUARANTINE_LEVER: False}
    if getattr(model, "loras", ()):
        if _graph_declares_fsdp(model):
            levers.pop(SLAB_PRECONDITION_LEVER)
        return levers  # no rescue for an unproven swap path
    if render_grant_is_live(model):
        levers.pop(QUARANTINE_LEVER)
        levers.pop(SLAB_PRECONDITION_LEVER)
        log.warning("auto-gate could not complete for %s; a capacity-rescue consent keeps "
                    "slab residency on, with the lora_low_rss lever it needs (the consent "
                    "lets the load fit; it proves nothing about accuracy)",
                    getattr(model, "unet_name", "?"))
    return levers


def revoke_consents_after_gate_fail(model: Any, levers: list[str]) -> str:
    """Best-effort revocation of consent disproved by identity-gate FAIL.

    Return the joined lever list for the quarantine event. Quarantine remains
    authoritative even if memo revocation fails.
    """
    joined = " + ".join(levers)
    if QUARANTINE_LEVER not in levers:
        return joined
    try:
        from .consent_routes import revoke_for_gate_fail

        revoked = revoke_for_gate_fail(getattr(model, "unet_name", None))
        if revoked:
            log.error("identity gate FAIL revoked %d slab-residency consent(s) for %s",
                      revoked, getattr(model, "unet_name", "?"))
    except Exception as exc:
        log.warning("consent revocation after an identity-gate FAIL failed (%r)", exc)
    return joined


def render_grant_is_live(model: Any) -> bool:
    """Return whether a live memo covers the render's non-quarantined checkpoint."""
    return render_grant(model) is not None


def render_consent_wire(model: Any) -> dict[str, str] | None:
    """Return the bounded wire stamp for a live rescue grant.

    Only the driver reads the store. The worker receives id, kind, and channel
    inside the dispatch envelope, authorizing only the rescue residency rung.
    """
    from .. import consent_pending

    grant = render_grant(model)
    return None if grant is None else consent_pending.grant_wire(grant)


def render_grant(model: Any) -> Any:
    """Return the render's live rescue grant; any lookup failure is a miss."""
    try:
        mesh = getattr(model, "mesh", None)
        unet_name = getattr(model, "unet_name", None)
        if getattr(mesh, "worker_args", None) is None or not isinstance(unet_name, str):
            return None

        import folder_paths

        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if not path:
            return None
        return live_grant(path=str(path), unet_name=unet_name,
                          options=getattr(model, "options", None) or {},
                          loras=getattr(model, "loras", ()) or ())
    except Exception as exc:
        log.warning("rescue consent lookup failed for a ceremony (%r)", exc)
        return None
