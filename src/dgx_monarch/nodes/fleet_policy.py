"""Fleet worker-policy and residency-grant authorization."""
from __future__ import annotations

import copy
from dataclasses import dataclass

from .. import __version__, residency_mode
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from ..residency_mode import ComfyManagedResidencyError
from .common import ModelSpec, _process_gate_verdict

log = get_logger(f"{__package__}.fleet")


@dataclass(frozen=True)
class _FleetAuthorization:
    """One internally consistent fleet policy/model/grant snapshot."""

    worker_args: dict
    model_request: dict
    residency_grant: dict | None


def _fleet_worker_policy(spec: ModelSpec, handle) -> _FleetAuthorization:
    """Snapshot the model/policy and authorize risky fleet residency.

    Fleet dispatch bypasses ``run_render`` and therefore cannot run the
    first-use identity ceremony itself. A PASS from the ledger or from this
    process's own ceremony is trusted only for the exact artifact set and
    capability context that produced it. Any other state, a lookup failure
    included, gets stock cudaMalloc and baked LoRAs for this call, or a refusal
    when comfy-managed residency is requested. The graph's requested policy
    stays untouched, so a later normal render can still prove and enable it.
    """
    # Snapshot every mutable graph payload exactly once.  Context lookup,
    # setup, and all job requests below use this one view even if another
    # Comfy execution mutates the graph-owned dictionaries concurrently.
    requested = dict(spec.mesh.worker_args)
    model_request = copy.deepcopy(spec.request_dict())
    # The fail-closed arm below reads the merged policy, never the graph's own:
    # cluster.toml [worker_args] can turn comfy-managed residency on, and a refusal
    # that read only the graph would take the stock stamp and never fire. The
    # initial value of policy is merged too, so a raise before the reassignment still fails closed.
    policy = {**dict(getattr(getattr(handle, "config", None), "worker_args", None) or {}), **requested}
    state, entry, process_verdict = "unknown", None, None
    session_pass_safe = False
    try:
        from ..actor.model_store import request_artifact_identity
        from ..gate_ledger import (
            GATE_PROTOCOL_VERSION,
            GateLedger,
            artifact_set_signature,
        )
        from ..mesh_safety import fleet_policy_is_risky, request_combo_key
        from .gate import (
            _effective_worker_args,
            _ledger_dir,
            fleet_residency_capability_context,
        )
        from .gate_identity import gate_verdict_token

        effective = _effective_worker_args(
            spec, handle, requested_worker_args=requested)
        policy = effective
        if not fleet_policy_is_risky(model_request, effective):
            return _FleetAuthorization(requested, model_request, None)

        # Build the detailed request identity once, then derive the ledger's
        # aggregate from those same signatures. A replacement between
        # two independent identity reads could otherwise authorize bytes B
        # using the PASS that belonged to bytes A.
        identity = request_artifact_identity(
            model_request["unet_name"], model_request.get("loras"))
        artifacts = artifact_set_signature(
            item["signature"] for item in identity["artifacts"])
        if artifacts.current != identity["digest"]:
            raise RuntimeError("fleet artifact identity aggregate is inconsistent")
        key = request_combo_key(model_request)
        context = fleet_residency_capability_context(
            spec, handle, effective_worker_args=effective)
        lookup = GateLedger(_ledger_dir()).lookup_with_integrity(
            key, artifacts, identity["comfy"], context)
        state, entry = lookup.state, lookup.entry
        session_pass_safe = lookup.session_pass_safe
        token = gate_verdict_token(
            key, artifacts.current, identity["comfy"], context)
        process_verdict = _process_gate_verdict(token)
    except Exception as exc:
        state = "unknown"
        log.error(
            "fleet: identity-gate state lookup failed (%r); treating the ledger state as unknown",
            exc,
        )

    process_denied = process_verdict in ("FAIL", "INCONCLUSIVE", "ERROR")
    process_pass = (
        process_verdict == "PASS"
        and session_pass_safe
        and state in ("unknown", "stale")
        and (entry is None or entry.get("verdict") == "PASS")
    )
    if not process_denied and (state == "pass" or process_pass):
        return _FleetAuthorization(
            requested,
            model_request,
            {
                "gate_protocol": GATE_PROTOCOL_VERSION,
                "dgx_monarch": __version__,
                "comfy": identity["comfy"],
                "artifact_digest": artifacts.current,
                "gate_token": list(token),
                "combo_key": key,
                "model_request": copy.deepcopy(model_request),
                "uncond_model_request": None,
                "artifact_sets": [copy.deepcopy(identity)],
                "capability_context": context,
            },
        )

    if residency_mode.requested(policy):
        # Fleet cannot run the ceremony and cannot turn this rung off for one
        # call, so the fail-closed branch has no residency to fall back to.
        # Class P, not class U: a Fleet call has no bypass, and the text names
        # the working alternative, as class P requires.
        raise ComfyManagedResidencyError(refusal(
            RefusalClass.PHYSICS,
            "Fleet cannot run the identity ceremony itself, and this model and LoRA combination "
            f"has no usable PASS for the comfy-managed residency capability context (ledger state: {state}). "
            "Fleet can turn slab_weights and lora_low_rss off for one call, but comfy-managed residency "
            "is a bootstrap policy that one call cannot turn off, so no safe residency is left to fall "
            "back to. Nothing was loaded and nothing was quarantined. What works instead: render this "
            "combination once through a normal DGX Monarch KSampler so the first-use ceremony can prove "
            "it, then Fleet takes that exact PASS; or turn the Init node's comfy_managed widget off (or remove "
            "comfy_managed from cluster.toml), reset the attached mesh (the panel's Reset attached mesh button), "
            "and restart the Worker service with `dgxm restart` only if the reset cannot confirm actor teardown.",
            troubleshooting=residency_mode.TROUBLESHOOTING))

    log.warning(
        "fleet: this model+LoRA combination has no usable identity-gate PASS for the "
        "current capability context (ledger state: %s); forcing slab_weights=off and "
        "lora_low_rss=off until an exact PASS exists",
        state,
    )
    return _FleetAuthorization(
        {**requested, "slab_weights": False, "lora_low_rss": False},
        model_request,
        None,
    )
