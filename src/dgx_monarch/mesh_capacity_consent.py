"""Capacity-rescue consent constants and the dispatch binding they carry."""
from __future__ import annotations

# The third authority a sample can carry, and the weakest of the three: an
# operator's capacity-rescue consent. It authorizes residency for one dispatch
# and nothing else: never an identity verdict, vouching, or a
# quarantine override (docs/DESIGN.md section 5.9, invariants 2 and 4).
CAPACITY_RESCUE_CONSENT_CAPABILITY = "capacity_rescue_consent"
RESIDENCY_MODE_CAPACITY_CONSENT = "capacity_consent"
# Bumped whenever the envelope's shape or its rules change. The worker compares
# it, so a driver and a worker loop that disagree about what a consent
# authorizes refuse the render instead of each acting on its own reading.
CAPACITY_CONSENT_CONTRACT = 1


def _compat_runtime():
    """Return ``mesh_safety`` at call time, so a monkeypatch on its public names applies."""
    from . import mesh_safety

    return mesh_safety


def assert_capacity_consent_grant(
    request: dict,
    artifact_sets: list[dict],
    *,
    grant: dict,
    effective_worker_args: dict,
    expected_context: dict | None = None,
    rank_world: int | None = None,
    setup_generation: int | None = None,
    setup_key: tuple | None = None,
    worker_args_key: tuple | None = None,
    worker_topology: dict | None = None,
) -> bool:
    """Validate capacity-rescue consent for one exact dispatch.

    Keep this separate from ``assert_normal_render_residency_grant``: consent
    permits a residency choice but proves no identity verdict and carries no
    ledger token. Shared validation must not let missing fields turn consent into
    an identity-gate PASS.

    Require consent mode, no claimed identity verdict, matching contract versions,
    a valid rescue grant, and no explicit residency opt-out. Bind it to one model
    without LoRAs or a second slot, the current setup and policy, the selected
    topology, and the exact artifact bytes.
    """
    runtime = _compat_runtime()
    from . import __version__, consent_pending
    from .consent_descriptor import KIND_RESCUE_SLAB

    if request.get("_dgxm_normal_residency_mode") != RESIDENCY_MODE_CAPACITY_CONSENT:
        raise RuntimeError(
            "capacity-rescue consent is not attached to capacity-consent mode")
    if grant.get("gate_token") is not None or grant.get("gate_protocol") is not None:
        raise RuntimeError(
            "a capacity-rescue consent must not carry identity-gate authority")
    if (grant.get("dgx_monarch") != __version__
            or grant.get("consent_contract") != CAPACITY_CONSENT_CONTRACT):
        # The one skew that could otherwise end in a silent stock load: a
        # driver that believes it authorized slab residency talking to a loop
        # that never read the field. Name the skew and its fix.
        raise RuntimeError(
            "capacity-rescue consent was written by dgx-monarch "
            f"{grant.get('dgx_monarch')!r} under consent contract "
            f"{grant.get('consent_contract')!r}, and this worker runs "
            f"{__version__} under contract {CAPACITY_CONSENT_CONTRACT}: the "
            "driver and the Worker service disagree about what a consent "
            "authorizes, so nothing was loaded. Restart the Worker service "
            "(`dgxm restart`), then restart ComfyUI and queue the render again")
    wire = consent_pending.grant_from_wire(grant.get("consent"))
    if wire is None or wire.kind != KIND_RESCUE_SLAB:
        raise RuntimeError(
            "capacity-rescue consent carries no readable rescue grant")
    if effective_worker_args.get("slab_weights") is False:
        raise RuntimeError(
            "a capacity-rescue consent never overrides an explicit slab_weights=off")
    if request["model"].get("loras"):
        raise RuntimeError(
            "a capacity-rescue consent covers the checkpoint only, and a LoRA "
            "stack is a second unproven path it says nothing about")
    if (request.get("uncond_model") is not None
            or grant.get("uncond_model_request") is not None
            or len(artifact_sets) != 1):
        raise RuntimeError(
            "a capacity-rescue consent authorizes exactly one model slot")
    context = grant.get("capability_context")
    if not isinstance(context, dict):
        raise RuntimeError("capacity-rescue consent has no capability context")
    if expected_context is not None and context != expected_context:
        raise RuntimeError(
            "capacity-rescue consent context no longer matches the current mesh/policy")
    if context.get("worker_args") != effective_worker_args:
        raise RuntimeError(
            "capacity-rescue consent does not match the executing worker policy")
    if rank_world is not None and int(context.get("world", -1)) != int(rank_world):
        raise RuntimeError(
            "capacity-rescue consent does not match the executing worker world")
    if (setup_generation is not None
            and grant.get("setup_generation") != int(setup_generation)):
        raise RuntimeError(
            "capacity-rescue consent does not match the READY setup generation")
    if setup_key is not None and tuple(grant.get("setup_key") or ()) != tuple(setup_key):
        raise RuntimeError(
            "capacity-rescue consent does not match the READY setup key")
    if (worker_args_key is not None
            and tuple(grant.get("worker_args_key") or ()) != tuple(worker_args_key)):
        raise RuntimeError(
            "capacity-rescue consent does not match the READY worker policy key")
    if worker_topology is not None:
        if grant.get("worker_topology") != worker_topology:
            raise RuntimeError(
                "capacity-rescue consent does not match the executing topology")
        if context.get("resolved_topology") != worker_topology:
            raise RuntimeError(
                "capacity-rescue consent context does not match the executing topology")
        if context.get("resolved_attention") != request.get("sage_kernel"):
            raise RuntimeError(
                "capacity-rescue consent context does not match the sample attention kernel")
        if bool(context.get("sync_ulysses")) != bool(request.get("sync_ulysses", True)):
            raise RuntimeError(
                "capacity-rescue consent context does not match the sample sync policy")
    if grant.get("combo_key") != runtime.request_combo_key(request["model"]):
        raise RuntimeError(
            "model/options/LoRA combination changed after the capacity consent")
    if grant.get("model_request") != request["model"]:
        raise RuntimeError("full model request changed after the capacity consent")
    if grant.get("artifact_sets") != artifact_sets:
        raise RuntimeError("model/Comfy identity changed after the capacity consent")
    identity = artifact_sets[0]
    if (grant.get("comfy") != identity.get("comfy")
            or grant.get("artifact_digest") != identity.get("digest")):
        raise RuntimeError(
            "capacity-rescue consent identity summary does not match its artifacts")
    return True
