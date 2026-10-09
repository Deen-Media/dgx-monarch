"""Project granted capacity consent into worker residency policy.

Projection sets slab residency in the worker args published to every rank and
binds the grant to the render authorization envelope. It applies only to the
authorized checkpoint and only when residency is undecided; explicit policy and
quarantine outrank consent. Cached projections are removed before unrelated
loads. Standing and environment grants write audit rows when their bypass fires.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from .. import residency_mode
from ..gate_ledger import GateLedgerLookup
from ..log import get_logger
from . import gate_inconclusive
from .consent_rescue import (
    QUARANTINE_LEVER,
    RESCUE_KIND,
    SLAB_PRECONDITION_LEVER,
    RescueLoad,
    artifacts_digest,
    combo_identity,
    live_grant,
    memo_context,
)

log = get_logger(__name__)

# Record the projected checkpoint on the cached MeshSpec so unrelated loads can
# restore the previous policy.
_PROJECTION_ATTR = "_dgxm_rescue_projection"

# Session lifetime: record one non-memo WAIVER per kind, combination, and
# channel per fleet generation. Recycle clears the set for the next context.
_AUDITED_NON_MEMO_GRANTS: set[tuple[str, str, str]] = set()


def projection_blocker(worker_args: Mapping[str, Any]) -> str:
    """Return why this policy cannot accept projected rescue, or ``""``."""
    if residency_mode.requested(worker_args):
        return ("this worker runs comfy-managed residency, which places the weights "
                "through ComfyUI's own DynamicVRAM and forces slab_weights off; it is "
                "a bootstrap policy, so it cannot be turned off for one render "
                "(docs/TROUBLESHOOTING.md #62)")
    if worker_args.get(QUARANTINE_LEVER) is False:
        return ("slab residency is explicitly off for this graph (the Init node's "
                "slab_weights=off, or an identity-gate quarantine)")
    if worker_args.get(SLAB_PRECONDITION_LEVER) is False:
        return ("lora_low_rss is explicitly off for this graph and slab residency "
                "requires it")
    return ""


def project(worker_args: dict, grant: Any, unet_name: str) -> bool:
    """Project granted slab residency into shared worker args in place; return whether it changed."""
    if grant is None or not isinstance(worker_args, dict):
        return False
    if worker_args.get(QUARANTINE_LEVER) is True:
        return False
    blocker = projection_blocker(worker_args)
    if blocker:
        log.warning("a capacity-rescue consent for %s was not projected: %s",
                    unet_name, blocker)
        return False
    worker_args[QUARANTINE_LEVER] = True
    log.info("capacity rescue consent %s projects slab residency for %s",
             getattr(grant, "id", "?"), unet_name)
    return True


def _remember_projection(mesh: Any, unet_name: str, previous: Any) -> None:
    try:
        object.__setattr__(mesh, _PROJECTION_ATTR, (unet_name, previous))
    except Exception as exc:  # a mesh double that refuses attributes
        log.debug("rescue projection not recorded (%r)", exc)


def _recorded_projection(mesh: Any) -> tuple[str, Any] | None:
    record = getattr(mesh, _PROJECTION_ATTR, None)
    return record if isinstance(record, tuple) and len(record) == 2 else None


def project_scoped(mesh: Any, worker_args: dict, grant: Any, unet_name: str, *,
                   covered: frozenset[str] = frozenset()) -> bool:
    """Project one grant and restore policy before unrelated checkpoints.

    ``covered`` keeps a dual-model graph's first projection active while its
    second loader runs. Without restoration, the cached MeshSpec could apply one
    checkpoint's consent to later unrelated loads.
    """
    record = _recorded_projection(mesh)
    if grant is not None:
        previous = record[1] if record is not None else worker_args.get(QUARANTINE_LEVER, "auto")
        changed = project(worker_args, grant, unet_name)
        if changed or worker_args.get(QUARANTINE_LEVER) is True:
            _remember_projection(mesh, unet_name, previous)
        return changed
    if record is None:
        return False
    name, previous = record
    if name == unet_name or name in covered:
        return False
    if worker_args.get(QUARANTINE_LEVER) is True:
        worker_args[QUARANTINE_LEVER] = previous
        log.info("the slab-residency consent for %s does not cover %s; residency is back "
                 "to %r for this load", name, unet_name, previous)
    _remember_projection(mesh, "", None)
    return False


def _projection_grant(model_or_mesh: Any, unet_name: str, options: Any,
                      loras: Sequence[Any] = ()) -> tuple[Any, str, Any, str] | None:
    """(grant, combo key, artifacts, path) for one checkpoint, or None."""
    import folder_paths

    path = folder_paths.get_full_path("diffusion_models", unet_name)
    if not path:
        return None
    combo, artifacts = combo_identity(unet_name, options, str(path), loras)
    grant = live_grant(path=str(path), unet_name=unet_name, options=options,
                       loras=loras, combo=(combo, artifacts))
    return grant, combo, artifacts, str(path)


def project_for_render(model: Any) -> bool:
    """Project standing rescue before topology and quarantine resolution."""
    try:
        mesh = getattr(model, "mesh", None)
        worker_args = getattr(mesh, "worker_args", None)
        unet_name = getattr(model, "unet_name", None)
        if not isinstance(worker_args, dict) or not isinstance(unet_name, str):
            return False

        options = getattr(model, "options", None) or {}
        loras = getattr(model, "loras", ()) or ()
        resolved = _projection_grant(mesh, unet_name, options, loras)
        if resolved is None:
            return False
        grant, combo, artifacts, path = resolved
        changed = project_scoped(mesh, worker_args, grant, unet_name)
        if worker_args.get(QUARANTINE_LEVER) is True:
            audit_non_memo_grant(grant, combo_key=combo, artifacts=artifacts,
                                 unet_name=unet_name, path=path,
                                 context=memo_context(unet_name, options, loras),
                                 loras=len(loras))
        return changed
    except Exception as exc:  # never block a render on the consent store
        log.warning("rescue consent not projected (%r)", exc)
        return False


def project_for_loader(mesh: Any, unet_name: str, options: Any,
                       covered: frozenset[str] = frozenset()) -> RescueLoad | None:
    """Project standing rescue before eager load, or restore prior policy.

    Worker capacity checks can raise a card even when driver estimates do not.
    Projecting here ensures that consent reaches the immediate load. Return the
    evidence needed for a CAPACITY_CERTIFIED row when the grant is honored.
    """
    try:
        worker_args = getattr(mesh, "worker_args", None)
        if not isinstance(worker_args, dict) or not isinstance(unet_name, str):
            return None
        resolved = _projection_grant(mesh, unet_name, options)
        if resolved is None:
            return None
        grant, combo, artifacts, path = resolved
        project_scoped(mesh, worker_args, grant, unet_name, covered=covered)
        if grant is None or worker_args.get(QUARANTINE_LEVER) is not True:
            return None
        context = memo_context(unet_name, options)
        audit_non_memo_grant(grant, combo_key=combo, artifacts=artifacts,
                             unet_name=unet_name, path=path, context=context)
        return standing_rescue_load(
            mesh, grant, combo=combo, artifacts=artifacts, unet_name=unet_name, path=path)
    except Exception as exc:  # a consent-store fault never blocks a load
        log.warning("rescue consent not projected at the loader node (%r)", exc)
        return None


def standing_rescue_load(mesh: Any, grant: Any, *, combo: str, artifacts: Any,
                         unet_name: str, path: str) -> RescueLoad | None:
    """Build measurable evidence for a consented eager load."""
    from .. import driver_footprint, mesh_safety

    try:
        avail = mesh_safety.mem_available_bytes()
        checkpoint_bytes = int(driver_footprint.file_size_bytes(path))
        if avail is None or not checkpoint_bytes:
            return None  # off Linux, or an unreadable file: no numbers, no row
        worker_args = dict(getattr(mesh, "worker_args", None) or {})
        return RescueLoad(
            combo_key=combo, artifacts=artifacts_digest(artifacts), unet_name=unet_name,
            path=path, consent_id=str(getattr(grant, "id", "")),
            worker_args={**worker_args, QUARANTINE_LEVER: True},
            measured={
                "probe": "loader_footprint_preflight",
                "checkpoint_bytes": checkpoint_bytes,
                "mem_available_bytes": int(avail),
                "headroom_bytes": int(avail) - checkpoint_bytes,
            })
    except Exception as exc:  # the load still happens; the row is evidence
        log.warning("standing rescue row not prepared for %s (%r)", unet_name, exc)
        return None


def audit_non_memo_grant(grant: Any, *, combo_key: str, artifacts: Any, unet_name: str,
                         path: str, context: Mapping[str, Any], loras: int = 0) -> None:
    """Audit a standing or environment bypass when it first fires.

    A memo grant wrote its row at click time. Audit failure is logged but must
    not rewrite the authorized load to stock residency.
    """
    if grant is None or getattr(grant, "key", None):
        return
    source = str(getattr(grant, "consent_source", "") or "")
    kind = str(getattr(grant, "kind", "") or RESCUE_KIND)
    token = (kind, combo_key, source)
    if token in _AUDITED_NON_MEMO_GRANTS:
        return
    try:
        from .. import consent_store
        from ..consent_kinds import KIND_SPECS
        from .consent_routes import record_waiver

        spec = KIND_SPECS[kind]
        recorded = record_waiver(
            action="grant", kind=kind, target_guard=spec.default_guard,
            consent_id=str(getattr(grant, "id", "")), consent_source=source,
            reason=str(getattr(grant, "reason", "") or spec.risk),
            combo_key=combo_key, artifacts=artifacts_digest(artifacts),
            memo_context=dict(context), capability_context=None,
            unet_name=unet_name, file_identity=consent_store.file_identity(path),
            loras=loras)
    except Exception as exc:
        recorded = None
        log.error("a %s capacity bypass for %s could not be audited (%r)", source, unet_name, exc)
    if recorded is None:
        log.error("a %s capacity bypass for %s was not written to the identity-gate ledger; "
                  "check that the ComfyUI output directory is writable", source, unet_name)
        return
    _AUDITED_NON_MEMO_GRANTS.add(token)


def reset_grant_audit() -> None:
    """Forget which non-memo bypasses this process has already audited."""
    _AUDITED_NON_MEMO_GRANTS.clear()


def capacity_consent_authorization(
    model: Any,
    requested: dict,
    model_request: dict,
    identity: dict,
    combo_key: str,
    context: dict,
    lookup: GateLedgerLookup,
    token: tuple[str, ...],
    process_verdict: str | None,
) -> Any:
    """Preserve the class-U rescue exemption at the final residency authorization boundary.

    A gate cannot prove cross-residency identity when stock cannot load the
    checkpoint, so the ordinary no-PASS fallback must not erase an explicit
    capacity rescue. This exemption applies only when slab residency was already
    projected for this checkpoint, with no LoRA stack and no second model.

    The normal authorization envelope retains dispatch, driver, and worker
    rechecks. It carries no gate token or ledger authority: consent grants
    residency for one dispatch, never a verdict. Return ``None`` when stock
    fallback remains required.
    """
    from .. import __version__
    from ..mesh_residency import (
        CAPACITY_CONSENT_CONTRACT,
        CAPACITY_RESCUE_CONSENT_CAPABILITY,
        RESIDENCY_MODE_CAPACITY_CONSENT,
    )
    from .consent_rescue import render_consent_wire
    from .gate_identity import NormalRenderAuthorization

    wire = (render_consent_wire(model)
            if requested.get("slab_weights") is True and not model_request.get("loras")
            else None)
    state = lookup.state
    if wire is None:
        if gate_inconclusive.dispatch_grants_skip(lookup, token, process_verdict):
            log.info("normal render: the identity gate found nothing to gate for this "
                     "combination, so it keeps that verdict; this dispatch runs on stock "
                     "residency and the session's levers stay as configured")
            return None
        log.warning("normal render has no exact Gate PASS for the current %s state; "
                    "forcing stock residency for this dispatch", state)
        return None
    log.warning("normal render has no exact Gate PASS for the current %s state; a "
                "capacity-rescue consent keeps slab residency for this dispatch (the "
                "consent lets the load fit; it proves nothing about accuracy)", state)
    return NormalRenderAuthorization(
        requested,
        model_request,
        {
            "capability": CAPACITY_RESCUE_CONSENT_CAPABILITY,
            "consent_contract": CAPACITY_CONSENT_CONTRACT,
            "dgx_monarch": __version__,
            "consent": dict(wire),
            "comfy": identity["comfy"],
            "artifact_digest": identity["digest"],
            "combo_key": combo_key,
            "model_request": copy.deepcopy(model_request),
            "uncond_model_request": None,
            "artifact_sets": [copy.deepcopy(identity)],
            "capability_context": copy.deepcopy(context),
        },
        RESIDENCY_MODE_CAPACITY_CONSENT,
    )
