"""One-queue-press proof that the risky residency and swap path matches a fresh stock load.

The ceremony renders a clean stock lineage, forces the risky residency/swap
transitions, renders again, and compares the output tensors for exact equality. Both renders run in one
always-dirty node execution, so Comfy's cache cannot serve a stale half.
"""
from __future__ import annotations

import json  # noqa: F401
import os  # noqa: F401
import time  # noqa: F401

from .. import (
    fsdp_reload_price,  # noqa: F401  # ceremony runtime seam
    mesh_setup,  # noqa: F401
)
from .. import mesh_safety as _mesh_safety
from ..gate_ledger import (
    ArtifactSetSignature,
    GateLedger,  # noqa: F401
    artifact_set_signature,
    artifact_signature,
    combo_key,
)
from ..log import get_logger
from ..mesh import ensure_live as ensure_live
from ..transfer_utils import raise_with_distinct_cause, reconcile_error, safe_call  # noqa: F401
from . import gate_identity as _gate_identity
from . import gate_inconclusive as _gate_inconclusive  # noqa: F401
from .common import ModelSpec, run_render  # noqa: F401
from .common import conditioning_for_wire as conditioning_for_wire
from .gate_cross_mode import record_ceremony_certificate, run_cross_residency_reference  # noqa: F401
from .gate_fsdp import (
    FSDP_PROOF_KIND,  # noqa: F401
    FsdpGateProofError,  # noqa: F401
    aborted_proof_error,  # noqa: F401
    ceremony_waiver_boundary_error,  # noqa: F401
    cleanup_aborted_fsdp_proof,  # noqa: F401
    close_aborted_proof,  # noqa: F401
    log_aborted_proof,  # noqa: F401
    record_proof_stop,  # noqa: F401
    require_confirmed_fsdp_unload,  # noqa: F401
    settled_refusal_tag,  # noqa: F401
)
from .gate_identity import (
    copy_transaction,  # noqa: F401
    equivalent_slab_mode_contexts,  # noqa: F401
    force_stock_quarantine,
    freeze_transaction,  # noqa: F401
    gate_verdict_token,  # noqa: F401
    model_with_worker_overrides,
    require_transaction_unchanged,  # noqa: F401
    resolve_effective_worker_args,
    temporary_worker_policy,
    transaction_tensor_versions,  # noqa: F401
)
from .gate_node import DGXMonarchIdentityGate as DGXMonarchIdentityGate
from .latent_identity import compare_latents  # noqa: F401
from .render_session import RenderSession
from .setup_binding import render_setup_token  # noqa: F401

log = get_logger(__name__)

FLEET_RESIDENCY_CAPABILITY = _mesh_safety.FLEET_RESIDENCY_CAPABILITY
is_artifact_binding_error = _mesh_safety.is_artifact_binding_error
is_memory_exhaustion = _mesh_safety.is_memory_exhaustion
is_stock_load_capacity_error = _mesh_safety.is_stock_load_capacity_error
physical_capability_context = _mesh_safety.physical_capability_context
_apply_worker_policy = _gate_identity.apply_worker_policy
_fleet_residency_capability_context = _gate_identity.fleet_residency_capability_context
_gate_capability_context = _gate_identity.gate_capability_context
_force_stock_quarantine = force_stock_quarantine
_model_with_worker_overrides = model_with_worker_overrides
_effective_worker_args = resolve_effective_worker_args
_temporary_worker_policy = temporary_worker_policy


def gate_capability_context(model, handle=None, *, effective_worker_args=None,
                            resolved_topology=None, resolved_attention=None):
    effective = (_effective_worker_args(model, handle) if effective_worker_args is None else effective_worker_args)
    return _gate_capability_context(model, handle, effective_worker_args=effective,
                                    resolved_topology=resolved_topology,
                                    resolved_attention=resolved_attention)


def fleet_residency_capability_context(model, handle=None, *, effective_worker_args=None):
    effective = (_effective_worker_args(model, handle) if effective_worker_args is None else effective_worker_args)
    return _fleet_residency_capability_context(model, handle, effective_worker_args=effective)


def _ledger_dir() -> str:
    import folder_paths

    return folder_paths.get_output_directory()


def _combo_of(model: ModelSpec) -> tuple[str, ArtifactSetSignature]:
    """(combo key, artifact signature) for this model + stack, resolved
    against the driver's model folders (workers resolve the same names).

    A sentinel identity is refused before it can reach the digest: an
    unidentified artifact must not compose a key that reads as "no verdict".
    """
    import folder_paths

    from ..gate_artifacts import refuse_unresolved

    def signed(kind: str, name: str) -> tuple[str, str]:
        path = folder_paths.get_full_path(kind, name)
        return f"{kind}/{name}", (artifact_signature(path) if path else "missing")

    pairs = [signed("diffusion_models", model.unet_name)]
    pairs.extend(signed("loras", entry["name"]) for entry in model.loras)
    refuse_unresolved(pairs)
    key = combo_key(model.unet_name, model.options, [e["name"] for e in model.loras])
    return key, artifact_set_signature(signature for _name, signature in pairs)


def run_identity_ceremony(model: ModelSpec, request: dict, latent: dict,
                          cfg_value: float, steps_hint: int, origin: str,
                          run_id: str = "", provenance_attestor=None,
                          _fsdp_return_guard: dict | None = None) -> dict:
    """Run every ceremony leg under one setup/residency capability."""
    from .gate_session import run_identity_ceremony as run_with_session

    return run_with_session(
        model, request, latent, cfg_value, steps_hint, origin, run_id,
        runtime=globals(), provenance_attestor=provenance_attestor,
        fsdp_return_guard=_fsdp_return_guard)


def _close_gate_session(session: RenderSession, primary: BaseException | None) -> None:
    try:
        session.close()
    except BaseException as cleanup_exc:
        if primary is None:
            raise
        winner, cause = reconcile_error(primary, cleanup_exc, "gate session cleanup failed")
        if winner is not primary:
            raise_with_distinct_cause(winner, cause)


def _publish_process_gate_verdicts(
    tokens: list[tuple[str, ...]] | tuple[tuple[str, ...], ...], verdict: str,
    ceremony: dict | None = None) -> None:
    """Publish atomically, retrying once while preserving cancellation."""
    from .common import _record_process_gate_verdicts

    first_error: BaseException | None = None
    cancellation: BaseException | None = None
    for _attempt in range(2):
        try:
            _record_process_gate_verdicts(tokens, verdict, ceremony)
        except BaseException as exc:
            first_error = exc if first_error is None else first_error
            if not isinstance(exc, Exception) and cancellation is None:
                cancellation = exc
        else:
            break
    else:
        raise cancellation if cancellation is not None else first_error  # type: ignore[misc]
    if cancellation is not None:
        raise cancellation


def _retract_process_gate_verdicts(
    tokens: list[tuple[str, ...]] | tuple[tuple[str, ...], ...]) -> None:
    """Withdraw atomically, retrying once while preserving cancellation."""
    from .common import _retract_process_gate_verdicts as _retract

    first_error: BaseException | None = None
    cancellation: BaseException | None = None
    for _attempt in range(2):
        try:
            _retract(tokens)
        except BaseException as exc:
            first_error = exc if first_error is None else first_error
            if not isinstance(exc, Exception) and cancellation is None:
                cancellation = exc
        else:
            break
    else:
        raise cancellation if cancellation is not None else first_error  # type: ignore[misc]
    if cancellation is not None:
        raise cancellation


def _run_identity_ceremony_bound(model: ModelSpec, request: dict, latent: dict,
                                 cfg_value: float, steps_hint: int, origin: str,
                                 run_id: str = "", *, handle,
                                 provenance_attestor,
                                 fsdp_scope_state: dict[str, bool] | None = None) -> dict:
    """The gate engine: stock-lineage render from a clean load, swap or FSDP
    reload cycle, swap-lineage render, bit-compare.
    Records the verdict in the gate report jsonl and the ledger. A residency or
    swap FAIL quarantines the session by pushing the implicated levers off to
    every worker: slab_weights for a cross-residency divergence, lora_low_rss
    for an in-mode one, both when both hold; an FSDP FAIL takes no lever.
    ["latent"] holds the stock-lineage latent, or the stock-residency leg on a
    cross-residency FAIL."""
    if request.get("uncond_model") is not None:
        raise ValueError(
            "the Identity Gate proves one model only; a dual-model request renders "
            "with slab_weights and lora_low_rss forced off, or refuses under automatic FSDP")
    # Both halves read this namespace at call time, which is why the imports
    # above stay bound here even where nothing in this file calls them.
    from .gate_ceremony import gather_ceremony_evidence
    from .gate_verdict import publish_ceremony_verdict

    evidence = gather_ceremony_evidence(
        runtime=globals(),
        model=model,
        request=request,
        latent=latent,
        cfg_value=cfg_value,
        steps_hint=steps_hint,
        origin=origin,
        run_id=run_id,
        handle=handle,
        provenance_attestor=provenance_attestor,
        fsdp_scope_state=fsdp_scope_state,
    )
    return publish_ceremony_verdict(
        runtime=globals(),
        evidence=evidence,
        model=model,
        handle=handle,
        origin=origin,
        run_id=run_id,
    )
