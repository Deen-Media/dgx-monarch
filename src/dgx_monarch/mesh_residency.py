"""Pure identity-gated residency request and grant validation."""
from __future__ import annotations

from typing import cast

from . import residency_mode
from .mesh_capacity_consent import (  # noqa: F401  # Public mesh_residency compatibility API.
    CAPACITY_CONSENT_CONTRACT,
    CAPACITY_RESCUE_CONSENT_CAPABILITY,
    RESIDENCY_MODE_CAPACITY_CONSENT,
    _compat_runtime,
    assert_capacity_consent_grant,
)
from .refusal import RefusalClass, refusal
from .residency_mode import ComfyManagedResidencyError

FLEET_RESIDENCY_CAPABILITY = "fleet_per_rank_residency"
NORMAL_RENDER_RESIDENCY_CAPABILITY = "normal_render_residency"


def physical_capability_context(handle) -> dict:
    """Physical/config identity shared by capability grants and preflight."""
    config = getattr(handle, "config", None)
    return {
        "mesh_mode": "cluster" if getattr(config, "hosts", ()) else "local",
        "config_source": str(getattr(config, "source", "") or "local"),
        "config_fingerprint": str(getattr(handle, "config_fingerprint", "local")),
        "physical_world": int(getattr(handle, "world", 0) or 0),
        "hosts": int(getattr(handle, "n_hosts", 0) or 0),
        "gpus_per_host": int(getattr(handle, "gpus_per_host", 0) or 0),
    }


def fleet_policy_is_risky(model_spec: dict, effective_worker_args: dict) -> bool:
    """Whether a fleet request can exercise either identity-gated path."""
    lora_risk = bool(model_spec.get("loras")) and (
        effective_worker_args.get("lora_low_rss") is not False)
    # Omitted means worker-side auto on UMA and is therefore still risky.
    slab_risk = effective_worker_args.get("slab_weights") is not False
    # comfy-managed is a third identity-gated residency, and unlike the other
    # two it cannot be turned off for one call (docs/TROUBLESHOOTING.md #62),
    # so it is risky whenever it is on and there is no safe downgrade.
    managed_risk = residency_mode.requested(effective_worker_args)
    return lora_risk or slab_risk or managed_risk


def request_combo_key(model_spec: dict) -> str:
    """Ledger combination key represented by a wire model specification."""
    from .gate_ledger import combo_key

    return combo_key(
        str(model_spec["unet_name"]),
        dict(model_spec.get("options") or {}),
        [str(entry["name"]) for entry in model_spec.get("loras") or []],
    )


def assert_request_artifact_binding(request: dict, artifact_sets: list[dict]) -> bool:
    """Bind a multi-phase operation to one model combo and artifact snapshot."""
    runtime = _compat_runtime()
    ArtifactBindingError = runtime.ArtifactBindingError

    binding = request.get("_dgxm_artifact_binding")
    if binding is None:
        return False
    if not isinstance(binding, dict):
        raise ArtifactBindingError("request artifact binding is malformed")
    if binding.get("combo_key") != runtime.request_combo_key(request["model"]):
        raise ArtifactBindingError(
            "model/options/LoRA combination changed after the ceremony snapshot"
        )
    if binding.get("model_request") != request["model"]:
        raise ArtifactBindingError(
            "full model request changed after the ceremony artifact snapshot"
        )
    if binding.get("uncond_model_request") != request.get("uncond_model"):
        raise ArtifactBindingError(
            "full unconditional model request changed after the ceremony artifact snapshot"
        )
    if binding.get("artifact_sets") != artifact_sets:
        raise ArtifactBindingError(
            "model/LoRA/Comfy identity changed after the ceremony snapshot"
        )
    return True


def assert_normal_render_residency_mode(
    request: dict, effective_worker_args: dict
) -> str | None:
    """Require an explicit safe authority mode for every risky normal sample."""
    runtime = _compat_runtime()
    if request.get("_dgxm_fleet_job"):
        return None
    mode = request.get("_dgxm_normal_residency_mode")
    model_specs = [request["model"]]
    if request.get("uncond_model"):
        model_specs.append(request["uncond_model"])
    risky = any(
        runtime.fleet_policy_is_risky(spec, effective_worker_args)
        for spec in model_specs
    )
    if mode == "required":
        if request.get("_dgxm_normal_residency_grant") is None:
            raise RuntimeError(
                "risky first-use render lost its required identity-gate grant")
    elif mode == RESIDENCY_MODE_CAPACITY_CONSENT:
        # Class U, availability only. The policy is risky by design: stock
        # residency cannot load this checkpoint here, so the ceremony that would
        # prove the slab path has no reference and can never PASS. The consent
        # envelope is the authority and is mandatory: a mode with no envelope is
        # a stripped request, not a consent.
        if request.get("_dgxm_normal_residency_grant") is None:
            raise RuntimeError(
                "capacity-consent render carries no rescue consent envelope")
    elif mode == "stock":
        if residency_mode.requested(effective_worker_args):
            # The driver stamps "stock" whenever it could not vouch the
            # combination, and for every dual-model request. Under this rung
            # that stamp is unreachable, not merely unproven: the levers it
            # names are already off and the one that matters cannot be turned
            # off in a live process. Typed, so the operator gets the cause and
            # the remedy instead of a bare "risky policy" RuntimeError naming
            # two levers they never set.
            raise ComfyManagedResidencyError(refusal(
                RefusalClass.PHYSICS,
                "comfy-managed residency is on for this worker and the driver could "
                "not authorize this render's residency: either this model and LoRA "
                "combination has no exact identity-gate PASS for the comfy-managed "
                "capability context, or the render dispatches a second (unconditional) "
                "model, which is always stamped stock because a ceremony proves one model. "
                "Unlike slab_weights and lora_low_rss there is no safe residency to fall back "
                "to: ComfyUI's DynamicVRAM is a bootstrap policy and cannot be turned off "
                "for one render. Nothing was loaded and nothing was quarantined. What works "
                "instead: for a single-model render, render this combination once with "
                "auto_gate=first_use (the default) so the first-use ceremony can prove it. For any "
                "render, turn the Init node's comfy_managed widget off and reset the Attached mesh. If "
                "that reset cannot confirm teardown, restart the Worker service (`dgxm restart`) and then ComfyUI.",
                troubleshooting=residency_mode.TROUBLESHOOTING))
        if risky:
            raise RuntimeError(
                "stock normal-render mode reached a risky worker residency policy")
    elif mode == "gate_internal":
        if request.get("_dgxm_artifact_binding") is None:
            raise RuntimeError(
                "identity-gate internal render has no exact artifact binding")
    elif mode == "operator_off":
        pass  # documented explicit opt-out
    elif risky:
        raise RuntimeError(
            "risky normal render has no driver-stamped residency authority mode")
    return mode


def assert_fleet_residency_grant(
    request: dict,
    artifact_sets: list[dict],
    *,
    effective_worker_args: dict,
    expected_context: dict | None = None,
    rank_world: int = 1,
) -> bool:
    """Validate a risky fleet request's exact artifact/context grant.

    The driver passes ``expected_context`` to bind physical/config identity as
    well as policy.  Workers cannot reconstruct driver config paths, but still
    bind the same grant to their current raw effective policy, world-1 setup,
    request combo, and freshly sampled local artifact identity.
    """
    runtime = _compat_runtime()
    grant = request.get("_dgxm_fleet_residency_grant")
    if grant is None:
        model_specs = [request["model"]]
        if request.get("uncond_model"):
            model_specs.append(request["uncond_model"])
        if request.get("_dgxm_fleet_job") and any(
            runtime.fleet_policy_is_risky(spec, effective_worker_args)
            for spec in model_specs
        ):
            raise RuntimeError(
                "risky fleet residency requested without an identity-gate grant"
            )
        return False
    if not isinstance(grant, dict):
        raise RuntimeError("fleet residency grant is malformed")
    context = grant.get("capability_context")
    if not isinstance(context, dict):
        raise RuntimeError("fleet residency grant has no capability context")
    if expected_context is not None and context != expected_context:
        raise RuntimeError(
            "fleet residency grant context no longer matches the current mesh/policy"
        )
    if (context.get("capability") != runtime.FLEET_RESIDENCY_CAPABILITY
            or context.get("rank_world") != 1
            or int(rank_world) != 1
            or context.get("worker_args") != effective_worker_args):
        raise RuntimeError(
            "fleet residency grant does not match the executing world-1 worker policy"
        )
    combo = runtime.request_combo_key(request["model"])
    if grant.get("combo_key") != combo:
        raise RuntimeError(
            "fleet model/options/LoRA combination changed after residency authorization"
        )
    if grant.get("model_request") != request["model"]:
        raise RuntimeError(
            "full fleet model request changed after residency authorization"
        )
    if grant.get("uncond_model_request") != request.get("uncond_model"):
        raise RuntimeError(
            "full fleet unconditional model request changed after residency authorization"
        )
    if grant.get("artifact_sets") != artifact_sets:
        raise RuntimeError(
            "model/LoRA/Comfy identity changed after fleet residency authorization"
        )
    if not artifact_sets:
        raise RuntimeError("fleet residency grant has no artifact identity")
    identity = artifact_sets[0]
    from . import __version__
    from .gate_ledger import GATE_PROTOCOL_VERSION, gate_verdict_token

    if (grant.get("gate_protocol") != GATE_PROTOCOL_VERSION
            or grant.get("dgx_monarch") != __version__
            or grant.get("comfy") != identity.get("comfy")
            or grant.get("artifact_digest") != identity.get("digest")):
        raise RuntimeError(
            "fleet residency grant protocol/package/identity is stale or malformed")
    expected_token = gate_verdict_token(
        combo, cast(str, identity.get("digest")),
        cast(str, identity.get("comfy")), context)
    if tuple(grant.get("gate_token") or ()) != expected_token:
        raise RuntimeError(
            "fleet residency grant token does not match its artifact/context payload")
    return True


def assert_normal_render_residency_grant(
    request: dict,
    artifact_sets: list[dict],
    *,
    effective_worker_args: dict,
    expected_context: dict | None = None,
    rank_world: int | None = None,
    setup_generation: int | None = None,
    setup_key: tuple | None = None,
    worker_args_key: tuple | None = None,
    worker_topology: dict | None = None,
) -> bool:
    """Validate an exact first-use Gate grant for a normal risky render."""
    runtime = _compat_runtime()
    grant = request.get("_dgxm_normal_residency_grant")
    if grant is None:
        return False
    if not isinstance(grant, dict):
        raise RuntimeError("normal-render residency grant is malformed")
    if grant.get("capability") == CAPACITY_RESCUE_CONSENT_CAPABILITY:
        # A capacity-rescue consent rides the same request field and the same
        # dispatch binding, so every caller that already validates a Gate grant
        # validates a consent too, with no new call site to forget.
        return assert_capacity_consent_grant(
            request,
            artifact_sets,
            grant=grant,
            effective_worker_args=effective_worker_args,
            expected_context=expected_context,
            rank_world=rank_world,
            setup_generation=setup_generation,
            setup_key=setup_key,
            worker_args_key=worker_args_key,
            worker_topology=worker_topology,
        )
    if request.get("_dgxm_normal_residency_mode") != "required":
        raise RuntimeError(
            "normal-render residency grant is not attached to required mode")
    from . import __version__
    from .gate_ledger import GATE_PROTOCOL_VERSION

    if (grant.get("capability") != runtime.NORMAL_RENDER_RESIDENCY_CAPABILITY
            or grant.get("gate_protocol") != GATE_PROTOCOL_VERSION
            or grant.get("dgx_monarch") != __version__):
        raise RuntimeError(
            "normal-render residency grant protocol/package is stale or malformed")
    context = grant.get("capability_context")
    if not isinstance(context, dict):
        raise RuntimeError("normal-render residency grant has no capability context")
    if expected_context is not None and context != expected_context:
        raise RuntimeError(
            "normal-render residency grant context no longer matches the current mesh/policy"
        )
    if context.get("worker_args") != effective_worker_args:
        raise RuntimeError(
            "normal-render residency grant does not match the executing worker policy"
        )
    if rank_world is not None and int(context.get("world", -1)) != int(rank_world):
        raise RuntimeError(
            "normal-render residency grant does not match the executing worker world"
        )
    if (setup_generation is not None
            and grant.get("setup_generation") != int(setup_generation)):
        raise RuntimeError(
            "normal-render residency grant does not match the READY setup generation"
        )
    if setup_key is not None and tuple(grant.get("setup_key") or ()) != tuple(setup_key):
        raise RuntimeError(
            "normal-render residency grant does not match the READY setup key"
        )
    if (worker_args_key is not None
            and tuple(grant.get("worker_args_key") or ()) != tuple(worker_args_key)):
        raise RuntimeError(
            "normal-render residency grant does not match the READY worker policy key"
        )
    if (worker_topology is not None
            and grant.get("worker_topology") != worker_topology):
        raise RuntimeError(
            "normal-render residency grant does not match the executing topology"
        )
    if worker_topology is not None:
        if context.get("resolved_topology") != worker_topology:
            raise RuntimeError(
                "normal-render Gate context does not match the executing topology"
            )
        if context.get("resolved_attention") != request.get("sage_kernel"):
            raise RuntimeError(
                "normal-render Gate context does not match the sample attention kernel"
            )
        if bool(context.get("sync_ulysses")) != bool(
                request.get("sync_ulysses", True)):
            raise RuntimeError(
                "normal-render Gate context does not match the sample sync policy"
            )
    combo = runtime.request_combo_key(request["model"])
    if grant.get("combo_key") != combo:
        raise RuntimeError(
            "normal-render model/options/LoRA combination changed after authorization"
        )
    if grant.get("model_request") != request["model"]:
        raise RuntimeError(
            "full normal-render model request changed after authorization"
        )
    if grant.get("uncond_model_request") != request.get("uncond_model"):
        raise RuntimeError(
            "normal-render unconditional model request changed after authorization"
        )
    if grant.get("artifact_sets") != artifact_sets:
        raise RuntimeError(
            "model/LoRA/Comfy identity changed after normal-render authorization"
        )
    if len(artifact_sets) != 1:
        raise RuntimeError("normal-render Gate grants authorize exactly one model")
    identity = artifact_sets[0]
    if (grant.get("comfy") != identity.get("comfy")
            or grant.get("artifact_digest") != identity.get("digest")):
        raise RuntimeError(
            "normal-render residency grant identity summary does not match its artifacts"
        )

    from .gate_ledger import gate_verdict_token

    expected_token = gate_verdict_token(
        combo,
        cast(str, identity.get("digest")),
        cast(str, identity.get("comfy")),
        context,
    )
    if tuple(grant.get("gate_token") or ()) != expected_token:
        raise RuntimeError(
            "normal-render residency grant token does not match its artifact/context payload"
        )
    return True
