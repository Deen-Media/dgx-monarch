"""Worker artifact and residency checks, separate from group lifecycle."""
from __future__ import annotations


def verify_sample_artifact_authorization(
    worker, request: dict, identity_fn
) -> list[dict]:
    """Recheck the driver's preflight identity and any residency grant on this rank."""
    requested_specs = [request["model"]]
    if request.get("uncond_model"):
        requested_specs.append(request["uncond_model"])
    local_artifacts = [
        identity_fn(spec["unet_name"], spec.get("loras"))
        for spec in requested_specs
    ]
    from ..mesh_safety import (
        assert_normal_render_residency_mode,
        assert_request_artifact_binding,
    )

    assert_request_artifact_binding(request, local_artifacts)
    assert_normal_render_residency_mode(
        request, dict(getattr(worker, "_active_worker_args", {})))
    if (request.get("_dgxm_fleet_job")
            or request.get("_dgxm_fleet_residency_grant") is not None):
        from ..mesh_safety import assert_fleet_residency_grant

        assert_fleet_residency_grant(
            request,
            local_artifacts,
            effective_worker_args=dict(getattr(worker, "_active_worker_args", {})),
            rank_world=int(worker.world or 0),
        )
    if request.get("_dgxm_normal_residency_grant") is not None:
        from ..mesh_safety import assert_normal_render_residency_grant

        assert_normal_render_residency_grant(
            request,
            local_artifacts,
            effective_worker_args=dict(getattr(worker, "_active_worker_args", {})),
            rank_world=int(worker.world or 0),
            setup_generation=getattr(worker, "_setup_generation", None),
            worker_topology=dict(getattr(worker, "topology", {}) or {}),
        )
    expected = request.get("_dgxm_artifact_sets")
    if expected is None or local_artifacts != expected:
        raise RuntimeError(
            "model/LoRA/Comfy identity changed after the driver's parity "
            f"preflight on rank {worker.rank}: expected={expected!r}, "
            f"local={local_artifacts!r}"
        )
    return expected


def sample_rescue_consent(request: dict) -> dict | None:
    """Return the rescue consent in the worker-validated request envelope.

    Only the driver reads the consent store. Worker authorization has already
    bound the stamp to worker policy, setup generation, topology and artifact
    identity. Keeping the stamp in the request also preserves checkpoint-scoped
    consent after a projected worker policy returns to ``auto``.
    """
    from ..mesh_residency import CAPACITY_RESCUE_CONSENT_CAPABILITY

    grant = request.get("_dgxm_normal_residency_grant")
    if not isinstance(grant, dict):
        return None
    if grant.get("capability") != CAPACITY_RESCUE_CONSENT_CAPABILITY:
        return None
    consent = grant.get("consent")
    return dict(consent) if isinstance(consent, dict) else None


def activate_accuracy_waivers(request: dict) -> int:
    """Install this dispatch's class-K waivers before any sample work.

    Activation first clears the prior dispatch's thread state, so an authorization
    cannot outlive its dispatch. The worker reads only id, kind and consent_source
    of the driver's stamp; an invalid stamp installs nothing and the guard refuses.
    """
    from .. import accuracy_waiver

    return accuracy_waiver.activate(request)


def accuracy_waiver_stamps() -> list[dict]:
    """Return waivers spent by this dispatch for driver-owned audit records.

    Empty unless a granted class-K waiver was used; use is never automatic.
    """
    from .. import accuracy_waiver

    return accuracy_waiver.stamps()


def assert_resident_artifact_identity(worker, slot: str, expected: dict) -> None:
    """Bind the model adopted by ModelStore to the preflight/grant identity."""
    resident = worker.store.uncond if slot == "uncond" else worker.store.current
    if resident is None or resident.artifact_identity != expected:
        from ..mesh_safety import ArtifactBindingError

        label = "unconditional model" if slot == "uncond" else "model"
        raise ArtifactBindingError(
            f"resident {label} identity changed after the artifact snapshot "
            f"and load on rank {worker.rank}"
        )
