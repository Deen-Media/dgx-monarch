"""Resolve class-K accuracy waivers per dispatch and stamp affected output.

Waivers ride the request, not cached worker policy, and are re-resolved for each
dispatch. Gate ceremonies never receive them because waived known-wrong math
cannot earn PASS. A permanent use row is written only when a guard uses the
waiver; dispatch facts join to the result by render id.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .. import accuracy_waiver
from ..accuracy_waiver import KNOWN_WRONG_KINDS, REQUEST_KEY, RESULT_KEY, STAMPED_RESULT_KEY
from ..log import get_logger
from ..transfer_utils import safe_call

log = get_logger(__name__)

# Render lifetime: facts needed to join a dispatch to its WAIVER use row.
_PENDING_AUDIT: dict[str, dict[str, Any]] = {}
# Entries retire on every render close and on every abandoned submit. Nothing
# evicts a live one: that would lose a stamped result's mandatory use row.
_PENDING_AUDIT_LIMIT = 512


def topology_wire(topo: Any) -> dict[str, Any]:
    """Normalize topology to the shape shared by driver grants and workers."""
    if isinstance(topo, Mapping):
        return dict(topo)
    return {"ulysses": getattr(topo, "ulysses", 1), "ring": getattr(topo, "ring", 1),
            "cfg": getattr(topo, "cfg", 1), "dp": getattr(topo, "dp", 1),
            "fsdp": bool(getattr(topo, "fsdp", False))}


def _combo(unet_name: str, options: Any, path: str,
           loras: Sequence[Any]) -> tuple[str, Any]:
    from .consent_rescue import combo_identity

    return combo_identity(unet_name, options, path, loras)


def live_grants(*, path: str, combo_key: str, topo: Any, world: object
                ) -> dict[str, dict[str, str]]:
    """Return class-K grants for this context without writing or raising."""
    from .. import consent_pending

    context = accuracy_waiver.waiver_context(
        combo_key=combo_key, topology=topology_wire(topo), world=world)
    grants: dict[str, dict[str, str]] = {}
    for kind in KNOWN_WRONG_KINDS:
        try:
            grant = consent_pending.resolve(kind=kind, path=path, context=context)
        except Exception as exc:  # A store fault grants no waiver.
            log.warning("accuracy-waiver lookup failed for %s (%r)", kind, exc)
            continue
        if grant is not None:
            grants[kind] = consent_pending.grant_wire(grant)
    return grants


_FSDP_ABORT = "automatic FSDP clean-reload Gate aborted before PASS"


def ceremony_abort_reason(exc: BaseException) -> str:
    """Explain an automatic FSDP ceremony abort without raising.

    Ceremonies omit waivers, so a granted class-K guard can still refuse the
    proof. The returned reason must state that actionable boundary.
    """
    try:
        from ..refusal import RefusalClass, parse_refusal_tag

        seen: BaseException | None = exc
        for _ in range(8):  # Bound potentially cyclic cause chains.
            if seen is None:
                break
            tag = parse_refusal_tag(str(seen))
            if tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG:
                return (
                    f"{_FSDP_ABORT}: the class-K guard {tag.guard or 'unnamed'} "
                    "refused inside the ceremony. A ceremony renders without "
                    "accuracy waivers on purpose, so a granted waiver does not "
                    "clear it there and an fsdp topology cannot be proven for "
                    "this shape. Render it on a topology without fsdp, or, for ring_pad, at "
                    "token counts that need no divisibility padding, or, for sol_attn, with a "
                    "SAGE_* or TORCH_FLASH kernel (docs/TROUBLESHOOTING.md #55).")
            seen = (
                seen.__cause__
                if seen.__cause__ is not None
                else seen.__context__
            )
    except BaseException as exc2:  # Diagnostics cannot replace the refusal.
        safe_call(log.warning, "ceremony abort reason not read (%r)", exc2)
    return _FSDP_ABORT


def _gate_ceremony_active() -> bool:
    try:
        from .common import _AUTO_GATE_ACTIVE

        return bool(getattr(_AUTO_GATE_ACTIVE, "on", False))
    except Exception:  # Unknown activity is treated as an active ceremony.
        return True


# Process lifetime: the class-K guard the last first-use ceremony for one
# combination refused on, keyed by that combination. A ceremony carries no
# waiver, so a guard that stops one proof render stops every later one, and the
# residency check below reads this to name the blocker.
_CEREMONY_BLOCKERS: dict[str, str] = {}
_CEREMONY_BLOCKER_LIMIT = 64


def _walk_causes(exc: BaseException):
    """Yield an exception and its bounded cause chain, oldest raise last."""
    seen: BaseException | None = exc
    for _ in range(8):  # Bound potentially cyclic cause chains.
        if seen is None:
            return
        yield seen
        seen = seen.__cause__ if seen.__cause__ is not None else seen.__context__


def blocking_guard(exc: BaseException) -> str | None:
    """The waivable class-K guard an aborted ceremony refused on, or None.

    Read from the tag rather than the message, and checked against the guard
    vocabulary, so only a guard a consent can grant is named.
    """
    from ..consent_observe import refusal_text
    from ..refusal import GUARDS, RefusalClass, parse_refusal_tag

    for item in _walk_causes(exc):
        text = refusal_text(item)
        tag = parse_refusal_tag(text) if isinstance(text, str) else None
        guard = getattr(tag, "guard", None)
        spec = GUARDS.get(guard) if isinstance(guard, str) else None
        if (tag is not None and tag.refusal_class is RefusalClass.KNOWN_WRONG
                and tag.waivable and spec is not None and spec.waivable_now):
            return guard
    return None


def blocking_descriptor(exc: BaseException):
    """The consent descriptor a class-K ceremony abort carried, or None."""
    from .. import consent_descriptor
    from ..consent_observe import refusal_text

    for item in _walk_causes(exc):
        text = refusal_text(item)
        if not isinstance(text, str) or consent_descriptor.DESCRIPTOR_BEGIN not in text:
            continue
        parsed = consent_descriptor.parse(text)
        if parsed is not None and consent_descriptor.valid_kind(parsed.kind):
            return parsed
    return None


def remember_ceremony_blocker(combo_key: str, guard: str) -> None:
    """Record which class-K guard blocks this combination's ceremony."""
    if not combo_key or not guard:
        return
    if (combo_key not in _CEREMONY_BLOCKERS
            and len(_CEREMONY_BLOCKERS) >= _CEREMONY_BLOCKER_LIMIT):
        _CEREMONY_BLOCKERS.pop(next(iter(_CEREMONY_BLOCKERS)), None)
    _CEREMONY_BLOCKERS[combo_key] = guard


def ceremony_blocker(combo_key: object) -> str | None:
    """The remembered class-K blocker for one combination, or None."""
    return _CEREMONY_BLOCKERS.get(str(combo_key or ""))


def reset_ceremony_blockers() -> None:
    """Forget every remembered blocker (tests)."""
    _CEREMONY_BLOCKERS.clear()


def _comfy_managed(model: Any, handle: Any = None) -> bool:
    from .. import residency_mode
    from .gate_identity import resolve_effective_worker_args

    return residency_mode.requested(
        resolve_effective_worker_args(model, handle))


def _blocker_is_granted(descriptor: Any) -> bool:
    """Whether a live grant already answers the card this abort would offer."""
    import folder_paths

    from .. import consent_pending

    path = folder_paths.get_full_path("diffusion_models", descriptor.unet_name)
    if not path:
        return True  # No local identity, so there is no card to offer either.
    return consent_pending.resolve(
        kind=descriptor.kind, path=str(path),
        context=dict(descriptor.memo_context)) is not None


def settle_aborted_ceremony(model: Any, exc: BaseException) -> None:
    """Record the class-K blocker and surface it when no stock fallback exists.

    Identity checks omit accuracy waivers, so granting one cannot unblock the
    proof itself. Ordinary residency can fall back to stock and raise the guard
    again during the requested render. Comfy-managed residency cannot switch to
    stock in-process: raise the class-K refusal here before a generic class-P
    message incorrectly recommends retrying the same blocked proof.
    """
    try:
        guard = blocking_guard(exc)
        descriptor = None if guard is None else blocking_descriptor(exc)
        if guard is None or descriptor is None:
            return
        remember_ceremony_blocker(
            str(descriptor.memo_context.get("combo_key", "") or ""), guard)
        if not _comfy_managed(model) or _blocker_is_granted(descriptor):
            return
        name = descriptor.unet_name
    except Exception as diag:  # A diagnosis never replaces the caller's answer.
        log.warning("an aborted ceremony's class-K blocker was not read (%r)", diag)
        return
    log.error("the first-use ceremony for %s refuses on the class-K guard %s, and a "
              "ceremony never renders under a waiver, so this render gets that refusal "
              "and its card instead of the comfy-managed residency refusal",
              name, guard)
    raise exc


def assert_ceremony_not_blocked(model: Any, handle: Any, authorization: Any) -> None:
    """Refuse a comfy-managed render whose ceremony a class-K guard blocks.

    Class P: the residency stays unproven and no consent authorizes it. This
    site answers before the generic stock-mode refusal so the message names the
    guard, the waiver and the two settings that do work.
    """
    from .. import mesh_safety, residency_mode
    from ..refusal import RefusalClass, refusal

    if getattr(authorization, "residency_mode", "") != "stock":
        return
    try:
        if not _comfy_managed(model, handle):
            return
        guard = ceremony_blocker(
            mesh_safety.request_combo_key(authorization.model_request))
    except Exception as exc:  # The generic refusal still answers this render.
        log.warning("a comfy-managed ceremony blocker was not read (%r)", exc)
        return
    if guard is None:
        return
    raise residency_mode.ComfyManagedResidencyError(refusal(
        RefusalClass.PHYSICS,
        residency_mode.COMFY_MANAGED_KNOWN_WRONG_REFUSAL.format(
            guard=guard, headless=_headless_spelling(guard)),
        troubleshooting=residency_mode.TROUBLESHOOTING))


def _headless_spelling(guard: str) -> str:
    """The variable that grants this guard, for a card that has expired.

    A class-P refusal carries no panel sentence of its own, and the card
    the ceremony raised lives for half an hour, so the operator who comes back
    later needs the variable named in the text.
    """
    try:
        action = accuracy_waiver.panel_action(guard)
        return f"{action.env}={action.env_value} on the driver"
    except Exception as exc:  # An unknown guard leaves the panel as the route.
        log.debug("no headless spelling for %s (%r)", guard, exc)
        return "from the environment variable this guard's own card names"


def stamp_request(request: dict, model: Any, topo: Any, world: object,
                  render_id: str) -> int:
    """Stamp live waivers onto one dispatch and return the count.

    Store failure leaves the request unstamped so the worker guard still refuses.
    """
    request.pop(REQUEST_KEY, None)
    try:
        import folder_paths

        unet_name = getattr(model, "unet_name", None)
        if not isinstance(unet_name, str) or not unet_name:
            return 0
        path = folder_paths.get_full_path("diffusion_models", unet_name)
        if not path:
            return 0
        options = getattr(model, "options", None) or {}
        loras = getattr(model, "loras", ()) or ()
        combo, artifacts = _combo(unet_name, options, str(path), loras)
        grants = live_grants(path=str(path), combo_key=combo, topo=topo, world=world)
        if not grants:
            return 0
        if _gate_ceremony_active():
            log.warning("an accuracy waiver covers %s but a first-use ceremony is "
                        "running; the ceremony renders without it and the guard "
                        "still refuses", unet_name)
            return 0
        # Do not send an authorization the driver cannot audit if a guard uses
        # it. A full window refuses this dispatch's waiver; the worker's guard decides.
        if not _remember_audit(render_id, model, unet_name, str(path), combo, artifacts,
                             topo, world, len(loras)):
            log.error("accuracy waiver for render %s was not dispatched because the "
                      "%d-entry audit window is full; wait for an outstanding render "
                      "to close and queue again", render_id, _PENDING_AUDIT_LIMIT)
            return 0
        request[REQUEST_KEY] = dict(grants)
        log.warning("render %s carries %d accuracy waiver(s) for %s: output produced "
                    "under one will be stamped rendered-under-waiver",
                    render_id, len(grants), unet_name)
        return len(grants)
    except Exception as exc:  # Store failure cannot block an unwaived render.
        log.warning("accuracy waivers not stamped onto the dispatch (%r)", exc)
        request.pop(REQUEST_KEY, None)
        return 0


def _remember_audit(render_id: str, model: Any, unet_name: str, path: str,
                    combo: str, artifacts: Any, topo: Any, world: object,
                    loras: int) -> bool:
    from .. import consent_store
    from .consent_rescue import artifacts_digest

    key = str(render_id)
    if key not in _PENDING_AUDIT and len(_PENDING_AUDIT) >= _PENDING_AUDIT_LIMIT:
        return False
    _PENDING_AUDIT[key] = {
        "combo_key": combo,
        "artifacts": artifacts_digest(artifacts),
        "unet_name": unet_name,
        "file_identity": consent_store.file_identity(path),
        "loras": int(loras),
        "memo_context": accuracy_waiver.waiver_context(
            combo_key=combo, topology=topology_wire(topo), world=world),
        "capability_context": {"worker_topology": topology_wire(topo),
                               "world": int(world) if isinstance(world, int) else 0,
                               "attention": str(
                                   getattr(getattr(model, "mesh", None), "attention", ""))},
    }
    return True


def fold_result_stamps(results: object) -> list[dict[str, Any]]:
    """Union rank-reported waiver use, deduplicated by guard.

    Silent ranks do not disprove use because a guard may fire on only some ranks.
    """
    folded: dict[str, dict[str, Any]] = {}
    if not isinstance(results, (list, tuple)):
        return []
    for row in results:
        if not isinstance(row, Mapping):
            continue
        for entry in row.get(RESULT_KEY) or ():
            if not isinstance(entry, Mapping):
                continue
            guard = str(entry.get("guard", "") or "")
            if guard and guard not in folded:
                folded[guard] = dict(entry)
    return [folded[guard] for guard in sorted(folded)]


def validate_inherited_stamps(source: object) -> list[dict[str, Any]]:
    """Validate input-latent waiver provenance and mark it inherited.

    Inherited stamps disclose ancestry without claiming the current dispatch
    used a waiver or writing a duplicate use row.
    """
    if not isinstance(source, Mapping) or STAMPED_RESULT_KEY not in source:
        return []
    entries = source[STAMPED_RESULT_KEY]
    if type(entries) not in (list, tuple):
        raise RuntimeError(
            "inherited waiver provenance must be a list or tuple")
    carried: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise RuntimeError(
                f"inherited waiver provenance entry {index} must be a mapping")
        normalized = dict(entry)
        run_id, guard = normalized.get("run_id"), normalized.get("guard")
        if (not isinstance(run_id, str) or not run_id
                or not isinstance(guard, str) or not guard):
            raise RuntimeError(
                f"inherited waiver provenance entry {index} must carry nonempty "
                "string 'run_id' and 'guard'")
        identity = (run_id, guard)
        if identity in seen:
            continue
        seen.add(identity)
        normalized["inherited"] = True
        carried.append(normalized)
    return carried


def prepare_result_stamps(
    results: object, source: object = None, *, strict: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Validate and fold result provenance without publishing audit effects."""
    # Validate trust-bearing input provenance outside the best-effort row write;
    # malformed identity must refuse output rather than disappear in a warning.
    carried = validate_inherited_stamps(source)
    try:
        spent = fold_result_stamps(results)
        seen = {(str(e.get("run_id", "")), str(e.get("guard", ""))) for e in spent}
        carried = [entry for entry in carried
                   if (entry["run_id"], entry["guard"]) not in seen]
    except Exception as exc:  # Existing pixels do not depend on audit preparation.
        if strict:
            raise
        log.error("a waived render could not be stamped (%r)", exc)
        return None
    return spent, carried


def publish_result_stamps(
    out: dict,
    prepared: tuple[list[dict[str, Any]], list[dict[str, Any]]] | None,
) -> dict:
    """Publish prevalidated provenance and its permanent use rows."""
    if prepared is None:
        return out
    spent, carried = prepared
    try:
        if not spent and not carried:
            return out
        out[STAMPED_RESULT_KEY] = spent + carried
        for entry in spent:
            log.error("RENDERED UNDER WAIVER: %s (guard %s, consent %s, source %s)",
                      entry.get("stamp"), entry.get("guard"), entry.get("consent_id"),
                      entry.get("consent_source"))
            _record_use_row(entry)
        # Used waivers have their rows and can retire now. Every other outcome
        # retires on its close path (``retire_audit``).
        for retired in {str(entry.get("run_id", "") or "") for entry in spent}:
            _PENDING_AUDIT.pop(retired, None)
    except Exception as exc:  # Existing pixels do not depend on audit-row success.
        log.error("a waived render could not be stamped (%r)", exc)
    return out


def stamp_result(out: dict, results: object, source: object = None) -> dict:
    """Publish waiver provenance and write use rows for waivers used here."""
    return publish_result_stamps(out, prepare_result_stamps(results, source))


def _record_use_row(entry: Mapping[str, Any]) -> None:
    """Write one permanent row for a waiver this render used."""
    from ..consent_kinds import KIND_SPECS
    from .consent_routes import _reason_for, record_waiver

    run_id = str(entry.get("run_id", "") or "")
    facts = _PENDING_AUDIT.get(run_id)
    kind = str(entry.get("kind", "") or "")
    spec = KIND_SPECS.get(kind)
    if facts is None or spec is None:
        # Never guess a dispatch combination or unknown kind onto a permanent row.
        log.error("a waived render (%s, guard %s) could not be joined to its dispatch, "
                  "so no WAIVER use row was written; the render result still carries "
                  "the stamp", run_id or "no run id", entry.get("guard"))
        return
    recorded = record_waiver(
        action="use", kind=kind, target_guard=str(entry.get("guard", "") or None),
        consent_id=str(entry.get("consent_id", "") or ""),
        consent_source=str(entry.get("consent_source", "") or "panel"),
        # Grant and use rows share the exact card-reason builder.
        reason=_reason_for(spec, None),
        combo_key=str(facts["combo_key"]), artifacts=str(facts["artifacts"]),
        memo_context=dict(facts["memo_context"]),
        capability_context=dict(facts["capability_context"]),
        unet_name=str(facts["unet_name"]), file_identity=str(facts["file_identity"]),
        loras=int(facts["loras"]), stamp=str(entry.get("stamp", "") or ""),
        run_id=run_id)
    if recorded is None:
        log.error("a render under waiver %s was not written to the identity-gate "
                  "ledger; check that the ComfyUI output directory is writable",
                  entry.get("consent_id"))


def retire_audit(render_id: object) -> None:
    """Idempotently drop one render's audit join facts on every close path."""
    try:
        _PENDING_AUDIT.pop(str(render_id), None)
    except BaseException:  # Cleanup must not add a new failure.
        pass


def reset_pending_audit() -> None:
    """Forget the dispatch facts this process is holding (tests)."""
    _PENDING_AUDIT.clear()
