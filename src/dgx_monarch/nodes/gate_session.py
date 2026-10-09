"""Render-session ownership for one identity-gate ceremony."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..transfer_utils import raise_with_distinct_cause, reconcile_error, safe_call
from .gate_fsdp import (
    FSDP_PROOF_KIND,
    FsdpGateProofError,
    cleanup_aborted_fsdp_proof,
    model_declares_fsdp_proof,
    settled_refusal_tag,
)
from .gate_inconclusive import is_no_material
from .gate_provenance import ProofCohortAttestor


def run_identity_ceremony(
    model: Any,
    request: dict,
    latent: dict,
    cfg_value: float,
    steps_hint: int,
    origin: str,
    run_id: str,
    *,
    runtime: Mapping[str, Any],
    provenance_attestor=None,
    fsdp_return_guard: dict[str, Any] | None = None,
) -> dict:
    """Run every ceremony leg under one setup/residency capability.

    ``runtime`` is the public :mod:`nodes.gate` namespace, read at call time so
    a monkeypatch of that module reaches every seam. The optional observer is
    wrapped by, and cannot replace, cohort attestation.
    """
    if request.get("uncond_model") is not None:
        raise ValueError(
            "the Identity Gate proves one model only; a dual-model request renders "
            "with slab_weights and lora_low_rss forced off, or refuses under automatic FSDP")
    from .common import _bind_packed_render_model

    model, bound_handle = _bind_packed_render_model(
        model,
        latent,
        cfg_value,
        ensure_live_fn=runtime["ensure_live"],
    )
    handle = (bound_handle if bound_handle is not None
              else runtime["ensure_live"](model.mesh.handle))
    provenance_attestor = ProofCohortAttestor(provenance_attestor)
    fsdp_scope_state = {
        "required": model_declares_fsdp_proof(model, handle),
    }
    session = runtime["RenderSession"]()
    primary: BaseException | None = None
    claimed = False
    try:
        try:
            # The session object is caller-owned before bind, so an interruption
            # at the bind return boundary still has cleanup authority.
            session.bind(handle)
            claimed = True
            with session.activate():
                result = runtime["_run_identity_ceremony_bound"](
                    model, request, latent, cfg_value, steps_hint, origin, run_id,
                    handle=handle,
                    provenance_attestor=provenance_attestor,
                    fsdp_scope_state=fsdp_scope_state,
                )
                result["_gate_handle"] = handle
                result["_fsdp_cleanup_attempted"] = fsdp_scope_state.get(
                    "cleanup_attempted", False)
                result["_fsdp_cleanup_confirmed"] = fsdp_scope_state.get(
                    "cleanup_confirmed", False)
                if (
                    fsdp_return_guard is not None
                    and origin == "auto_first_use"
                    and result.get("verdict") == "PASS"
                    and (
                        fsdp_scope_state["required"]
                        or result.get("proof_kind") == FSDP_PROOF_KIND
                    )
                ):
                    # Arm caller-owned cleanup before this function can cross its
                    # return/session-close boundary. One dict.update is the
                    # publication point: an interruption before it is still
                    # caught by this session, while one after it leaves the auto
                    # caller with the exact handle and shared exactly-once state.
                    fsdp_return_guard.update({
                        "cleanup_required": True,
                        "accepted": False,
                        "handle": handle,
                        "state": fsdp_scope_state,
                    })
                if (
                    origin == "auto_first_use"
                    and result.get("verdict") != "PASS"
                    and result.get("proof_kind") != FSDP_PROOF_KIND
                    and not is_no_material(result)
                ):
                    # Keep quarantine inside this ceremony capability: another
                    # queue must not dispatch under the rejected risky policy.
                    # A ceremony that had nothing to prove rejected nothing, so
                    # it takes no lever from the queues behind it.
                    runtime["_force_stock_quarantine"](
                        model, handle, timeout_s=600)
                return result
        except BaseException as exc:
            # A settled refusal keeps its own class here too: the ceremony
            # below has already recorded and retracted for it, so rewriting it
            # into an untyped abort would lose the guard the operator acts on.
            surfaced = (
                FsdpGateProofError(
                    "FSDP clean-reload proof aborted before a terminal verdict"
                )
                if fsdp_scope_state["required"]
                and isinstance(exc, Exception)
                and not isinstance(exc, FsdpGateProofError)
                and settled_refusal_tag(exc) is None
                else exc
            )
            primary = surfaced
            if claimed and fsdp_scope_state["required"]:
                cleanup_aborted_fsdp_proof(
                    handle,
                    primary,
                    logger=runtime["log"],
                    state=fsdp_scope_state,
                )
            elif (
                claimed
                and not isinstance(exc, FsdpGateProofError)
            ):
                try:
                    with session.activate():
                        runtime["_force_stock_quarantine"](
                            model, handle, timeout_s=600, cause=exc)
                except BaseException as quarantine_exc:
                    previous = primary
                    primary, cause = reconcile_error(
                        previous, quarantine_exc,
                        "identity-gate stock quarantine retry failed")
                    safe_call(
                        runtime["log"].error,
                        "identity-gate stock quarantine retry failed while "
                        "preserving %r: %r", primary, quarantine_exc)
                    if primary is not previous:
                        raise_with_distinct_cause(primary, cause)
            if surfaced is not exc:
                raise surfaced from exc
            raise
        finally:
            try:
                runtime["_close_gate_session"](session, primary)
            except BaseException as cleanup_exc:
                if primary is None:
                    primary = cleanup_exc
                    raise
                previous = primary
                primary, cause = reconcile_error(
                    previous, cleanup_exc, "identity-gate session cleanup failed")
                safe_call(
                    runtime["log"].error,
                    "identity-gate session cleanup failed while preserving %r: %r",
                    primary, cleanup_exc)
                if primary is not previous:
                    raise_with_distinct_cause(primary, cause)
    finally:
        try:
            runtime["_close_gate_session"](session, primary)
        except BaseException as cleanup_exc:
            if primary is None:
                raise
            previous = primary
            primary, cause = reconcile_error(
                previous, cleanup_exc, "identity-gate session cleanup retry failed")
            safe_call(
                runtime["log"].error,
                "identity-gate session cleanup retry failed while preserving %r: %r",
                primary, cleanup_exc)
            if primary is not previous:
                raise_with_distinct_cause(primary, cause)
