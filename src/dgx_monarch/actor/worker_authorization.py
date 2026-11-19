"""Worker artifact and residency checks, separate from group lifecycle."""
from __future__ import annotations

from ..log import get_logger

log = get_logger(__name__)


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
    from .. import gate_runtime

    local_runtime = gate_runtime.runtime_build_identity()  # a grant must name this rank's own build
    if (request.get("_dgxm_fleet_job")
            or request.get("_dgxm_fleet_residency_grant") is not None):
        from ..mesh_safety import assert_fleet_residency_grant

        assert_fleet_residency_grant(
            request,
            local_artifacts,
            effective_worker_args=dict(getattr(worker, "_active_worker_args", {})),
            rank_world=int(worker.world or 0),
            local_runtime=local_runtime,
        )
    if request.get("_dgxm_normal_residency_grant") is not None:
        from ..mesh_safety import assert_normal_render_residency_grant

        granted = assert_normal_render_residency_grant(
            request,
            local_artifacts,
            effective_worker_args=dict(getattr(worker, "_active_worker_args", {})),
            rank_world=int(worker.world or 0),
            setup_generation=getattr(worker, "_setup_generation", None),
            worker_topology=dict(getattr(worker, "topology", {}) or {}),
            local_runtime=local_runtime,
        )
    else:
        granted = False
    expected = request.get("_dgxm_artifact_sets")
    if expected is None or local_artifacts != expected:
        raise RuntimeError(
            "model/LoRA/Comfy identity changed after the driver's parity "
            f"preflight on rank {worker.rank}: expected={expected!r}, "
            f"local={local_artifacts!r}"
        )
    if granted:  # logged once every authorization check on this rank has passed
        log_gate_grant(worker, request)
    return expected


def gate_grant_line(rank: object, request: dict) -> str | None:
    """The journal line for an identity gate grant this rank accepted, or None.

    It names the model slots the grant covers, so a hardware leg can read from
    a Worker journal that a dual-model render (Ideogram 4) ran on a grant over
    both slots. It names the checkpoints as the model store's lines do and
    counts the LoRAs without naming them. A capacity-rescue consent rides the
    same request field and logs nothing here.
    """
    from ..mesh_residency import NORMAL_RENDER_RESIDENCY_CAPABILITY

    grant = request.get("_dgxm_normal_residency_grant")
    if not isinstance(grant, dict) or grant.get("capability") != NORMAL_RENDER_RESIDENCY_CAPABILITY:
        return None
    slots = [("cond", request["model"])]
    if request.get("uncond_model"):
        slots.append(("uncond", request["uncond_model"]))

    def named(slot: str, spec: dict) -> str:
        count = len(spec.get("loras") or [])
        loras = "no LoRA" if count == 0 else f"{count} LoRA" + ("s" if count > 1 else "")
        return f"[{slot}] {spec.get('unet_name')} with {loras}"

    cover = ("both model slots" if len(slots) == 2 else "its one model slot")
    return (f"identity gate grant accepted on rank {rank}: the gate PASS covers {cover}, "
            + " and ".join(named(slot, spec) for slot, spec in slots)
            + "; this render keeps the residency the gate proved")


def log_gate_grant(worker, request: dict) -> None:
    """Log gate_grant_line for this rank, when the grant is an identity gate grant."""
    line = gate_grant_line(getattr(worker, "rank", None), request)
    if line is not None:
        log.info(line)


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
