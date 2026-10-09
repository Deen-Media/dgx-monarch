"""Plan-first setup for explicit hosts, with all mutations behind ``SetupOps``."""

from __future__ import annotations

import difflib
import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from ..operator_profiles import ProfileResolution, compile_profile
from . import operator_receipt
from .setup_config_io import (
    decode_snapshot,
    publish_receipt,
    recommend_fabric,
    render_setup_config,
    setup_readiness,
)
from .setup_models import CandidateHost as CandidateHost
from .setup_models import (
    SetupOps,
    SetupOutcome,
    SetupPlan,
    SetupRequest,
)
from .setup_probe import compare_artifacts, probe_hosts, validate_artifact_paths


def build_setup_plan(request: SetupRequest, *, ops: SetupOps | None = None) -> SetupPlan:
    ops = ops or SetupOps()
    receipt_start = ops.begin_receipt()
    source = ops.source_snapshot()
    artifacts = validate_artifact_paths(request.artifacts)
    provisional_profile = compile_profile("balanced", observations=())
    provisional_fabric = request.fabric_profile or ("single-node" if len(request.hosts) == 1 else "generic-roce")
    provisional_text = _render_candidate(request, provisional_fabric, provisional_profile)
    provisional = ops.roundtrip(provisional_text)
    probes = probe_hosts(provisional, artifacts=artifacts, runner=ops.run_host)
    observations = tuple(probe.hardware_observation() for probe in probes)
    resolution = compile_profile(request.profile, observations=observations)
    recommended_fabric = recommend_fabric(probes)
    fabric_profile = request.fabric_profile or recommended_fabric
    if fabric_profile is None:
        raise ValueError("fabric profile could not be inferred from the candidate hosts; pass --fabric-profile")
    text = _render_candidate(request, fabric_profile, resolution)
    config = ops.roundtrip(text)
    snapshot = ops.snapshot(request.config_path)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    diff = _config_diff(snapshot, text)
    changed = snapshot.digest != digest
    warning = _gate_warning(snapshot, changed)
    comparisons = compare_artifacts(probes, len(artifacts))
    blockers, warnings = setup_readiness(
        probes=probes,
        expected_gpus=[host.gpus for host in request.hosts],
        artifacts=comparisons,
        fabric_profile=fabric_profile,
        install_service=request.install_service,
        start_workers=request.start_workers,
        verify=request.verify,
        transport_security=request.transport_security,
    )
    if snapshot.existed and changed:
        warnings.append("existing_config_replacement_planned")
    elif snapshot.existed and snapshot.mode != 0o600:
        warnings.append("config_mode_hardening_planned")
    return SetupPlan(
        config_path=Path(request.config_path).expanduser().absolute(),
        config_text=text,
        config=config,
        snapshot=snapshot,
        receipt_start=receipt_start,
        source=source,
        request_binding=request.binding(),
        probes=probes,
        artifacts=comparisons,
        resolution=resolution,
        recommended_profile="balanced",
        fabric_profile=fabric_profile,
        topology="auto",
        config_digest=digest,
        current_digest=snapshot.digest,
        diff=diff,
        gate_context_changes=changed,
        gate_warning=warning,
        privileged_process_inspection=request.privileged_process_inspection,
        blockers=tuple(blockers),
        warnings=tuple(sorted(set(warnings))),
    )


def run_setup(request: SetupRequest, *, ops: SetupOps | None = None, plan: SetupPlan | None = None) -> SetupOutcome:
    ops = ops or SetupOps()
    plan = build_setup_plan(request, ops=ops) if plan is None else _validated_plan(request, plan, ops)
    builder = plan.receipt_start.materialize(
        ops.receipt_builder, plan.resolution.name, plan.config_digest, plan.source.manifest
    )
    reachable = sum(probe.reachable for probe in plan.probes)
    builder.add_step(
        "host_probe",
        "succeeded" if reachable == len(plan.probes) else "failed",
        counts={"hosts": len(plan.probes), "reachable": reachable},
    )
    builder.add_step("profile_compile", "succeeded", counts={"hosts": len(plan.probes)})
    builder.add_step(
        "config_plan",
        "succeeded" if not plan.blockers else "failed",
        counts={"blockers": len(plan.blockers), "warnings": len(plan.warnings)},
        notes=[
            plan.gate_warning,
            "Privileged process inspection: "
            + (
                "requested; an administrator must start the root process inspector on each host."
                if request.privileged_process_inspection
                else "not requested."
            ),
        ],
    )
    if not request.apply:
        receipt = builder.finish("planned", notes=["No config or service change was requested."])
        written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=False)
        return SetupOutcome("planned", plan, False, False, False, False, None, receipt, written)
    if plan.blockers:
        builder.add_step("readiness_blocked", "failed")
        receipt = builder.finish("failed", notes=["Setup refused before any change: readiness checks found blockers."])
        written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
        blocked_status: operator_receipt.ReceiptStatus = "failed" if written else "partial"
        failure = "readiness_blocked" if written else "readiness_blocked_receipt_write_failed"
        return SetupOutcome(blocked_status, plan, False, False, False, False, failure, receipt, written)
    if not request.assume_yes and ops.confirm("Apply the reviewed setup plan? [y/N] ") is not True:
        receipt = builder.finish("planned", notes=["The operator did not confirm the plan."])
        written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=False)
        return SetupOutcome("planned", plan, False, False, False, False, "confirmation_declined", receipt, written)
    from .setup_apply import run_confirmed_setup

    entered = False
    try:
        with ops.config_lock(plan.config_path):
            entered = True
            return run_confirmed_setup(request, ops, plan, builder)
    except BaseException as error:
        if entered:
            raise
        return _lock_failure(request, ops, plan, builder, error)


def _lock_failure(
    request: SetupRequest,
    ops: SetupOps,
    plan: SetupPlan,
    builder: Any,
    primary: BaseException,
) -> SetupOutcome:
    """Record a config-lock failure in a receipt; re-raise the first cancellation, if any, instead of returning."""
    interruption = primary if not isinstance(primary, Exception) else None
    receipt: dict[str, object] = {}
    try:
        builder.add_step("config_lock_unavailable", "failed")
        receipt = cast(
            dict[str, object],
            builder.finish("failed", notes=["The config lock was unavailable, so no config or service changed."]),
        )
    except BaseException as receipt_error:
        if interruption is None and not isinstance(receipt_error, Exception):
            interruption = receipt_error
    try:
        written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
    except BaseException as publish_error:
        written = False
        if interruption is None and not isinstance(publish_error, Exception):
            interruption = publish_error
    if interruption is not None:
        if interruption is primary:
            raise primary
        raise interruption from primary
    status: operator_receipt.ReceiptStatus = "failed" if written else "partial"
    failure = "config_lock_unavailable" if written else "config_lock_unavailable_receipt_write_failed"
    return SetupOutcome(status, plan, False, False, False, False, failure, receipt, written)


def _render_candidate(request: SetupRequest, fabric_profile: str, resolution: ProfileResolution) -> str:
    return render_setup_config(
        [(host.name, host.fabric_ip, host.gpus, host.ssh_user, host.comfy_dir) for host in request.hosts],
        client_ip=request.client_ip,
        fabric_profile=fabric_profile,
        resolution=resolution,
        ssh_key=request.ssh_key,
        python_bin=request.python_bin,
        comfy_dir=request.comfy_dir,
        transport_security=request.transport_security,
    )


def _validated_plan(request: SetupRequest, plan: SetupPlan, ops: SetupOps) -> SetupPlan:
    path = request.config_path.expanduser().absolute()
    if len(plan.probes) != len(request.hosts):
        raise ValueError("supplied setup plan does not match the current request")
    observations = tuple(probe.hardware_observation() for probe in plan.probes)
    resolution = compile_profile(request.profile, observations=observations)
    fabric = request.fabric_profile or recommend_fabric(plan.probes)
    rendered = _render_candidate(request, plan.fabric_profile, resolution)
    config = ops.roundtrip(rendered)
    comparisons = compare_artifacts(plan.probes, len(request.artifacts))
    blockers, warnings = setup_readiness(
        probes=plan.probes,
        expected_gpus=[host.gpus for host in request.hosts],
        artifacts=comparisons,
        fabric_profile=plan.fabric_profile,
        install_service=request.install_service,
        start_workers=request.start_workers,
        verify=request.verify,
        transport_security=request.transport_security,
    )
    if plan.snapshot.existed and plan.gate_context_changes:
        warnings.append("existing_config_replacement_planned")
    elif plan.snapshot.existed and plan.snapshot.mode != 0o600:
        warnings.append("config_mode_hardening_planned")
    mismatch = (
        plan.request_binding != request.binding()
        or plan.config_path != path
        or plan.snapshot.path != path
        or plan.current_digest != plan.snapshot.digest
        or plan.resolution != resolution
        or fabric != plan.fabric_profile
        or replace(config, source=plan.config.source) != plan.config
        or rendered != plan.config_text
        or hashlib.sha256(rendered.encode("utf-8")).hexdigest() != plan.config_digest
        or plan.diff != _config_diff(plan.snapshot, rendered)
        or plan.gate_context_changes != (plan.snapshot.digest != plan.config_digest)
        or plan.gate_warning != _gate_warning(plan.snapshot, plan.gate_context_changes)
        or plan.privileged_process_inspection != request.privileged_process_inspection
        or comparisons != plan.artifacts
        or tuple(blockers) != plan.blockers
        or tuple(sorted(set(warnings))) != plan.warnings
        or plan.topology != "auto"
        or plan.recommended_profile != "balanced"
    )
    if mismatch:
        raise ValueError("supplied setup plan does not match the current request")
    return plan


def _config_diff(snapshot: Any, text: str) -> str:
    return "".join(
        difflib.unified_diff(
            decode_snapshot(snapshot).splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile="current/cluster.toml",
            tofile="planned/cluster.toml",
            lineterm="\n",
        )
    )


def _gate_warning(snapshot: Any, changed: bool) -> str:
    if not changed:
        return "The config digest is unchanged, so the config does not change the Gate context."
    if not snapshot.existed:
        return "The new config starts a Gate context bound to its digest, and that context must validate before use."
    message = "The existing config will be replaced after confirmation; "
    return message + "its digest changes, so contextual Gate results must revalidate."
