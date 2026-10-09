"""First-use identity-gate orchestration behind the ``nodes.common`` wrappers.

The process-local maps live in ``nodes/gate_process_state`` and are reached
through that module object, so the denial map's copy-on-write publish stays
visible to every later reader. Every other seam resolves against
``nodes.common`` per call: a module-top import would cycle, and a test's patch
on ``nodes.common`` must reach this module.
"""
from __future__ import annotations

import sys
from typing import Any, NoReturn

from ..transfer_utils import raise_with_distinct_cause, reconcile_error, safe_call
from . import gate_process_state as _state
from .gate_fsdp import (
    FsdpGateProofError,
    arm_fsdp_return_guard,
    cleanup_rejected_fsdp_pass,
    retract_process_gate_verdicts,  # noqa: F401  # published from here
    settled_refusal_tag,
    token_requires_fsdp_proof,
    token_resolves_fsdp,
    topology_requires_fsdp_proof,
)
from .gate_process_state import (  # noqa: F401 - compatibility re-export
    release_auto_gate_claim,
    restore_auto_gate_active,
)
from .gate_token_lifetimes import trim_session
from .render_validation import PACKED_TOPOLOGY_REFUSALS

# read-only after import: one reason per automatic-FSDP denial source
_FSDP_DENIAL_REASONS = {
    "process-local": "automatic FSDP execution denied by the process-local clean-reload Gate verdict {verdict}",
    "durable": "automatic FSDP execution denied by the durable clean-reload Gate FAIL",
    "completed": "automatic FSDP execution denied by the completed clean-reload Gate verdict {verdict}",
    "cached": "automatic FSDP execution denied by the cached clean-reload Gate verdict {verdict}",
    "claim": "automatic FSDP clean-reload Gate claim timed out",
}


def _deny_automatic_fsdp(source: str, verdict: str | None = None) -> NoReturn:
    """Raise the typed FSDP denial for one ``_FSDP_DENIAL_REASONS`` source."""
    template = _FSDP_DENIAL_REASONS.get(source)
    if template is None:
        raise ValueError(
            f"unknown automatic-FSDP denial source {source!r}: add its reason to "
            f"_FSDP_DENIAL_REASONS, which carries {sorted(_FSDP_DENIAL_REASONS)}"
        )
    reason = template.format(verdict=verdict)
    if verdict is None:
        raise FsdpGateProofError(reason)
    raise FsdpGateProofError(reason, verdict=verdict)


def _raise_cleanup_cancellation(primary: BaseException | None, cleanup: BaseException, label: str) -> None:
    if primary is None:
        raise cleanup
    winner, cause = reconcile_error(primary, cleanup, label)
    if winner is cleanup:
        raise_with_distinct_cause(winner, cause)

def record_process_gate_verdicts(
    tokens: list[tuple[str, ...]] | tuple[tuple[str, ...], ...],
    verdict: str, ceremony: dict | None = None,
) -> None:
    """Atomically publish one ceremony across normal/Fleet capabilities."""
    from .common import gate_inconclusive

    unique_tokens = tuple(dict.fromkeys(tokens))
    with _state._AUTO_GATE_LOCK:
        gate_inconclusive.remember_kind(unique_tokens, ceremony)
        denied = gate_inconclusive.session_denial_tokens(unique_tokens, ceremony)
        denials = dict(_state._PROCESS_GATE_DENIALS)
        if verdict in ("FAIL", "INCONCLUSIVE", "ERROR"):
            denials.update(dict.fromkeys(denied, verdict))
        else:
            for token in unique_tokens:
                denials.pop(token, None)
        # One copy-on-write store: an async interruption exposes all old or all
        # new revocations, never normal updated while Fleet remains stale.
        _state._PROCESS_GATE_DENIALS = denials
        for token in unique_tokens:
            if token in denied:
                _state._AUTO_GATE_SESSION[token] = verdict
            else:  # a sibling: its retest-guard denial lifts, its PASS is revoked
                denials.pop(token, None)
                _state._AUTO_GATE_SESSION.pop(token, None)
        trim_session(_state)  # the one capped store here (gate_token_lifetimes)


def process_gate_verdict(token: tuple[str, ...]) -> str | None:
    with _state._AUTO_GATE_LOCK:
        return process_gate_verdict_locked(token)


def process_gate_verdict_locked(token: tuple[str, ...]) -> str | None:
    """Read one verdict while the caller owns ``_AUTO_GATE_LOCK``."""
    return (
        _state._PROCESS_GATE_DENIALS.get(token)
        or _state._AUTO_GATE_SESSION.get(token)
    )


def auto_gate_context(
    model: Any,
    request_kind: str,
    latent: dict | None = None,
    cfg_value: float | None = None,
) -> tuple[str, tuple[str, ...]] | None:
    if getattr(model.mesh, "auto_gate", "off") != "first_use":
        return None
    if request_kind not in ("ksampler", "ksampler_advanced"):
        return None
    from ..gate_ledger import GateLedger, comfy_commit
    from .common import (
        _apply_persisted_quarantine,
        _bind_packed_render_model,
        _resolve_topology_for_latent,
        ensure_live,
        gate_inconclusive,
        log,
        resolve_sample_attention,
        topology_from_preset,
    )
    from .gate import (
        _combo_of,
        _effective_worker_args,
        _ledger_dir,
        gate_capability_context,
    )
    from .gate_identity import gate_verdict_token

    bound_handle = None
    if latent is not None:
        model, bound_handle = _bind_packed_render_model(
            model,
            latent,
            cfg_value,
            ensure_live_fn=ensure_live,
        )
    handle = (bound_handle if bound_handle is not None
              else ensure_live(model.mesh.handle))
    effective = _effective_worker_args(model, handle)
    resolved_topology = None
    resolved_attention = None
    if latent is not None:
        resolved_topology, sage, _reason = _resolve_topology_for_latent(
            model, latent, cfg_value, int(handle.world))
        resolved_attention = resolve_sample_attention(model.mesh.attention, sage)
    elif model.mesh.topology_preset != "auto":
        resolved_topology = topology_from_preset(
            model.mesh.topology_preset, int(handle.world))
        resolved_attention = resolve_sample_attention(model.mesh.attention, False)
    lora_risk = bool(model.loras) and effective.get("lora_low_rss") is not False
    # Omitted slab_weights means worker-side auto and remains risky.
    slab_risk = effective.get("slab_weights") is not False
    # comfy_managed forces lora_low_rss and slab_weights off, so without this
    # the render would read as risk-free here while authorization still demands
    # a PASS for the comfy-managed capability context. The rung is itself the
    # risky lever, so it triggers the ceremony.
    from .. import residency_mode

    comfy_risk = residency_mode.requested(effective)
    # Resolved no-LoRA FSDP is independently risky even though both residency levers
    # are required off: it needs its exact clean-reload proof before execution.
    fsdp_proof_required = topology_requires_fsdp_proof(
        resolved_topology,
        model.loras,
    )
    if (not lora_risk and not slab_risk and not comfy_risk
            and not fsdp_proof_required):
        return None
    from ..adapters.sol_attention import sol_ceremony_skip_reason

    skip = sol_ceremony_skip_reason(resolved_attention)
    if skip is not None:
        log.info("identity gate not run for %s: %s", model.unet_name, skip)
        return None
    try:
        key, artifacts = _combo_of(model)
        commit = comfy_commit()
        capability_context = gate_capability_context(
            model,
            handle,
            resolved_topology=resolved_topology,
            resolved_attention=resolved_attention,
        )
        if resolved_topology is None:
            # ``auto`` is shape-sensitive. Without the latent there is no safe
            # ledger lookup; concrete submission performs the exact lookup.
            state, entry = "unknown", None
        else:
            lookup = GateLedger(_ledger_dir()).lookup_with_integrity(
                key, artifacts, commit, capability_context)
            state, entry = gate_inconclusive.granted_state(lookup, fsdp_proof_required), lookup.entry
            # Keep canonical unknown/stale states in the public ledger API, but do
            # not let a positive cache suppress the one ceremony that can heal
            # unscoped damage or replace a non-PASS exact row.
            session_bridge_safe = (
                lookup.session_pass_safe
                and (entry is None or entry.get("verdict") == "PASS")
            )
            if not session_bridge_safe and state in ("unknown", "stale"):
                state = "error"
        if state == "fail":
            if fsdp_proof_required:
                log.error(
                    "FSDP clean-reload Gate previously FAILED for %s; denying "
                    "automatic FSDP execution",
                    model.unet_name,
                )
            else:
                levers = _apply_persisted_quarantine(model, entry)
                log.error(
                    "identity gate previously FAILED for %s; the ledger's FAIL row sets "
                    "%s=off on this graph",
                    model.unet_name,
                    "+".join(levers),
                )
        return state, gate_verdict_token(
            key, artifacts.current, commit, capability_context)
    except FsdpGateProofError:
        raise
    except Exception as exc:
        if fsdp_proof_required:
            raise FsdpGateProofError(
                "automatic FSDP Gate context/ledger lookup failed",
            ) from exc
        raise


def auto_gate_required(
    model: Any,
    request_kind: str = "ksampler",
    latent: dict | None = None,
    cfg_value: float | None = None,
) -> bool:
    """Whether a normal render must remain sequential for first-use gating."""
    from .common import (
        _auto_gate_context,
        _process_gate_verdict,
        _process_gate_verdict_locked,
        gate_inconclusive,
        log,
    )

    if getattr(_state._AUTO_GATE_ACTIVE, "on", False):
        return False
    try:
        context = _auto_gate_context(
            model, request_kind, latent, cfg_value)
    except FsdpGateProofError:
        raise
    except PACKED_TOPOLOGY_REFUSALS:
        raise
    except Exception as exc:
        safe_call(
            log.error,
            "auto-gate state lookup failed (%r); requiring fail-safe preflight", exc)
        return True
    if context is None:
        return False
    state, token = context
    fsdp_proof_required = token_requires_fsdp_proof(token)
    process_verdict = _process_gate_verdict(token)
    if process_verdict in ("FAIL", "INCONCLUSIVE", "ERROR"):
        if fsdp_proof_required:
            _deny_automatic_fsdp("process-local", process_verdict)
        return True
    if state == "fail" and fsdp_proof_required:
        _deny_automatic_fsdp("durable", "FAIL")
    if state in ("pass", "fail", gate_inconclusive.GRANTED_STATE):
        return False
    with _state._AUTO_GATE_CONDITION:
        if token in _state._AUTO_GATE_RUNNING:
            released = _state._AUTO_GATE_CONDITION.wait_for(
                lambda: token not in _state._AUTO_GATE_RUNNING,
                timeout=_state._AUTO_GATE_WAIT_S,
            )
            if not released:
                if fsdp_proof_required:
                    _deny_automatic_fsdp("claim")
                log.error(
                    "auto-gate claim wait timed out; keeping pipeline sequential")
                return True
        # INCONCLUSIVE, ERROR or no entry never grants prefetch. A cached FAIL is
        # terminal for this token; a cached PASS only while the ledger reads unknown
        # or stale, since unscoped damage or a non-PASS row needs a fresh ceremony.
        cached = _process_gate_verdict_locked(token)
        if (
            fsdp_proof_required
            and cached in ("FAIL", "INCONCLUSIVE", "ERROR")
        ):
            _deny_automatic_fsdp("completed", cached)
        if cached == "PASS" and state not in ("unknown", "stale"):
            return True
        return cached not in ("PASS", "FAIL")


def maybe_auto_gate(
    model: Any,
    request: dict,
    latent: dict,
    cfg_value: float | None,
    steps_hint: int,
) -> str | None:
    """Run one bounded first-use ceremony before the user's full render."""
    from .common import (
        _auto_gate_context,
        _process_gate_verdict,
        _process_gate_verdict_locked,
        _record_process_gate_verdicts,
        _release_auto_gate_claim,
        _restore_auto_gate_active,
        gate_inconclusive,
        log,
    )

    del steps_hint  # request steps are the canonical source below
    if getattr(_state._AUTO_GATE_ACTIVE, "on", False):
        return None
    if request.get("uncond_model") is not None:
        return "DUAL_MODEL_STOCK"
    claimed_token: tuple[str, ...] | None = None
    fsdp_proof_required = fsdp_resolved = False
    try:
        context = _auto_gate_context(
            model, request.get("kind", ""), latent, cfg_value)
        if context is None:
            return None
        state, token = context
        fsdp_proof_required, fsdp_resolved = token_requires_fsdp_proof(token), token_resolves_fsdp(token)
        process_verdict = _process_gate_verdict(token)
        if process_verdict in ("FAIL", "INCONCLUSIVE", "ERROR"):
            if fsdp_proof_required:
                _deny_automatic_fsdp("process-local", process_verdict)
            return gate_inconclusive.cached_verdict(token, process_verdict)
        if state == "fail" and fsdp_proof_required:
            _deny_automatic_fsdp("durable", "FAIL")
        if state in ("pass", "fail", gate_inconclusive.GRANTED_STATE):
            return gate_inconclusive.granted_verdict(state)
        with _state._AUTO_GATE_CONDITION:
            if token in _state._AUTO_GATE_SESSION:
                cached = _process_gate_verdict_locked(token)
                if cached != "PASS" or state in ("unknown", "stale"):
                    if fsdp_proof_required and cached != "PASS":
                        _deny_automatic_fsdp("cached", str(cached or "ERROR"))
                    return gate_inconclusive.cached_verdict(token, cached)
            if token in _state._AUTO_GATE_RUNNING:
                released = _state._AUTO_GATE_CONDITION.wait_for(
                    lambda: token not in _state._AUTO_GATE_RUNNING,
                    timeout=_state._AUTO_GATE_WAIT_S,
                )
                if not released:
                    if fsdp_proof_required:
                        _deny_automatic_fsdp("claim")
                    log.error(
                        "auto-gate claim wait timed out; forcing stock residency")
                    return "ERROR"
                completed = _process_gate_verdict_locked(token) or "ERROR"
                if fsdp_proof_required and completed != "PASS":
                    _deny_automatic_fsdp("completed", completed)
                return gate_inconclusive.cached_verdict(token, completed)
            claimed_token = token
            _state._AUTO_GATE_RUNNING.add(token)
        from ..first_render import gate_finished, gate_started
        from .gate import run_identity_ceremony

        light = dict(request)
        light["steps"] = min(2, int(request.get("steps") or 2))
        light["advanced"] = {
            "add_noise": True,
            "start_at_step": 0,
            "end_at_step": None,
            "return_with_leftover_noise": False,
        }
        light["kind"] = "ksampler_advanced"
        gate_was_active = getattr(_state._AUTO_GATE_ACTIVE, "on", False)
        ceremony_result: dict | None = None
        fsdp_return_guard: dict[str, Any] = {}
        try:
            gate_started(state)  # inside the try whose finally closes it
            _state._AUTO_GATE_ACTIVE.on = True
            result = run_identity_ceremony(
                model,
                light,
                dict(latent),
                float(1.0 if cfg_value is None else cfg_value),
                light["steps"],
                origin="auto_first_use",
                _fsdp_return_guard=fsdp_return_guard,
            )
            ceremony_result = result
            result_is_fsdp = arm_fsdp_return_guard(
                fsdp_return_guard, result, proof_required=fsdp_proof_required)
            if result["verdict"] == "PASS":
                actual_token = tuple(result.get("_gate_token") or ())
                current: Any = _auto_gate_context(
                    model, light["kind"], latent, cfg_value)
                token_matches = (
                    actual_token == token
                    and current is not None
                    and current[1] == actual_token
                )
                if not token_matches:
                    log.error(
                        "auto-gate artifact/context changed across the ceremony; "
                        "discarding PASS")
                    result["verdict"] = "ERROR"
                elif current[0] not in ("pass", "unknown", "stale"):
                    log.error(
                        "auto-gate PASS cannot be used in the current %s ledger state; caching "
                        "ERROR, so the render gets no Gate grant from it",
                        current[0],
                    )
                    result["verdict"] = "ERROR"
            if fsdp_proof_required != result_is_fsdp:
                log.error(
                    "auto-gate FSDP proof scope/result mismatch; denying the "
                    "automatic render"
                )
                result["verdict"] = "ERROR"
            result_tokens = [
                tuple(item) for item in result.get("_gate_tokens", ())]
            process_publish_error: BaseException | None = None
            process_publish_cancel: BaseException | None = None
            for _attempt in range(2):
                try:
                    _record_process_gate_verdicts(
                        result_tokens or [token], str(result["verdict"]), result)
                except BaseException as exc:
                    if process_publish_error is None:
                        process_publish_error = exc
                    if (not isinstance(exc, Exception)
                            and process_publish_cancel is None):
                        process_publish_cancel = exc
                else:
                    break
            else:
                if process_publish_cancel is not None:
                    raise process_publish_cancel
                raise process_publish_error  # type: ignore[misc]
            if process_publish_cancel is not None:
                # Retry made the verdict terminal, but caller cancellation
                # remains authoritative after state is safe.
                raise process_publish_cancel
            verdict = str(result["verdict"])
            if (fsdp_proof_required or result_is_fsdp) and verdict != "PASS":
                raise FsdpGateProofError(
                    "automatic FSDP execution requires an exact clean-reload "
                    f"Gate PASS (got {verdict})",
                    verdict=verdict,
                )
            if fsdp_return_guard.get("cleanup_required", False):
                fsdp_return_guard["accepted"] = True
            return gate_inconclusive.dispatch_verdict(result)
        finally:
            active_primary = sys.exception()
            safe_call(gate_finished, ceremony_result)
            if (
                fsdp_return_guard.get("cleanup_required", False)
                and not fsdp_return_guard.get("accepted", False)
            ):
                cleanup_primary = (
                    active_primary if active_primary is not None
                    else FsdpGateProofError(
                        "automatic FSDP provisional PASS was rejected"))
                cleanup_rejected_fsdp_pass(
                    fsdp_return_guard.get("handle"),
                    cleanup_primary,
                    logger=log,
                    state=fsdp_return_guard.get("state", {}),
                )
            active_cleanup_error = _restore_auto_gate_active(gate_was_active)
            if active_cleanup_error is not None:
                _raise_cleanup_cancellation(
                    active_primary, active_cleanup_error,
                    "auto-gate active-context restoration was interrupted")
    except FsdpGateProofError:
        raise
    except PACKED_TOPOLOGY_REFUSALS:
        raise
    except Exception as exc:
        from .consent_waiver import ceremony_abort_reason, settle_aborted_ceremony

        # A refusal that already answered keeps its own class and remedy. The
        # ceremony has settled it and removed its provisional denial, so wrapping
        # it here would replace a typed answer with an untyped abort.
        if (fsdp_proof_required or fsdp_resolved) and settled_refusal_tag(exc) is not None:
            raise  # FSDP has no stock fallback, LoRA or not: surface the typed cause
        if fsdp_proof_required or fsdp_resolved:
            raise FsdpGateProofError(ceremony_abort_reason(exc)) from exc
        settle_aborted_ceremony(model, exc)
        safe_call(log.error,
            "auto-gate failed (%r); forcing stock residency before the full render", exc)
        return gate_inconclusive.settle_known_wrong_abort(claimed_token, exc) or "ERROR"
    finally:
        if claimed_token is not None:
            claim_primary = sys.exception()
            claim_cleanup_error = _release_auto_gate_claim(claimed_token)
            if claim_cleanup_error is not None:
                _raise_cleanup_cancellation(
                    claim_primary, claim_cleanup_error,
                    "auto-gate claim release was interrupted")
