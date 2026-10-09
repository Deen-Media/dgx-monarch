"""Confirmed, lock-held mutation phase for guided setup."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from . import operator_receipt
from .setup_config_io import ConfigMutation, ConfigSnapshot, decode_snapshot, publish_receipt, verify_setup
from .setup_config_transaction import ConfigTransaction
from .setup_models import SetupOps, SetupOutcome, SetupPlan, SetupRequest
from .setup_recheck import recheck_setup
from .setup_services import ServiceCertainty, SetupServiceRequest, SetupServiceResult
from .setup_verification import SetupVerification, SetupVerificationProgress


def run_confirmed_setup(
    request: SetupRequest,
    ops: SetupOps,
    plan: SetupPlan,
    builder: Any,
) -> SetupOutcome:
    """Apply one confirmed plan while the caller holds its config lock."""
    transaction = ConfigTransaction(plan.config_path, plan.config_text, plan.snapshot, ops.apply_config)
    mutation: ConfigMutation | None = None
    published: ConfigSnapshot | None = None
    service_request: SetupServiceRequest | None = None
    service_result: SetupServiceResult | None = None
    services_changed = False
    settlement_authorized = True
    try:
        try:
            recheck = recheck_setup(
                plan.config,
                reviewed_probes=plan.probes,
                reviewed_artifacts=plan.artifacts,
                reviewed_source=plan.source,
                expected_gpus=[host.gpus for host in request.hosts],
                artifact_paths=request.artifacts,
                fabric_profile=plan.fabric_profile,
                install_service=request.install_service,
                start_workers=request.start_workers,
                verify=request.verify,
                transport_security=request.transport_security,
                runner=ops.run_host,
                source_snapshot=ops.source_snapshot,
            )
        except BaseException:
            try:
                builder.add_step("live_recheck", "unknown")
            except BaseException:
                pass
            raise
        builder.add_step(
            "live_recheck",
            "succeeded" if recheck.matched else "failed",
            counts=recheck.receipt_counts(),
        )
        if not recheck.matched:
            builder.add_step("live_readiness_changed", "failed")
            receipt = builder.finish("failed", notes=["The reviewed setup evidence changed; setup made no changes."])
            written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
            status: operator_receipt.ReceiptStatus = "failed" if written else "partial"
            failure = "live_readiness_changed" if written else "live_readiness_changed_receipt_write_failed"
            return SetupOutcome(status, plan, False, False, False, False, failure, receipt, written)
        candidate_config = replace(plan.config, source=str(plan.config_path))
        if request.install_service:
            service_request = SetupServiceRequest(
                candidate_config,
                plan.source.root,
                plan.source.manifest,
                install_service=True,
                start_service=request.start_workers,
                privileged_process_inspection=request.privileged_process_inspection,
            )
            service_result = ops.execute_services(service_request)
            services_changed = service_result.changed
            _record_service_step(builder, "service_transaction", service_result)
            if service_result.interruption is not None:
                raise service_result.interruption
            if service_result.certainty is not ServiceCertainty.SUCCEEDED:
                return _pre_config_service_outcome(
                    request,
                    ops,
                    plan,
                    builder,
                    transaction,
                    services_changed=services_changed,
                    service_compensated=service_result.compensated,
                    failure=f"service_{service_result.code}",
                )
        mutation = transaction.publish()
        if not mutation.durable:
            raise OSError("cluster config publication was not durable")
        published = _published_snapshot(ops, plan, mutation)
        loaded = ops.roundtrip(decode_snapshot(published))
        if replace(loaded, source=plan.config.source) != plan.config:
            raise OSError("published cluster config did not round-trip to the reviewed candidate")
        config = replace(loaded, source=str(plan.config_path))
        builder.add_step(
            "config_apply",
            "succeeded",
            counts={
                "backup_created": int(mutation.backup_path is not None),
                "changed": int(mutation.changed),
                "durable": int(mutation.durable),
                "rollback_available": int(mutation.rollback_available),
            },
        )
        if request.verify:
            progress = SetupVerificationProgress()
            try:
                verification = verify_setup(
                    config,
                    plan.config_path,
                    request.comfy_dir,
                    plan.source.manifest,
                    doctor=ops.run_doctor,
                    smoke=ops.run_smoke,
                    progress=progress,
                )
            except BaseException:
                if progress.smoke_attempted:
                    # Once smoke is dispatched, only a cancellation reaches here; verify_setup returns
                    # smoke errors as unknown. The smoke mesh's teardown is unproven: keep all state.
                    settlement_authorized = False
                try:
                    _record_verification(builder, progress, None)
                except BaseException:
                    try:
                        builder.add_step("verification_evidence_unavailable", "unknown")
                    except BaseException:
                        pass
                raise
            if verification.unknown:
                # Latch before receipt work: unknown smoke may have left a mesh
                # active, so later receipt failure cannot authorize teardown.
                settlement_authorized = False
            _record_verification(builder, progress, verification)
            if verification.unknown:
                return _uncertain_verification_outcome(
                    request,
                    ops,
                    plan,
                    builder,
                    mutation,
                    services_changed=services_changed,
                )
            if not verification.smoke_ok or not verification.doctor_ok:
                if service_request is None or service_result is None:
                    raise RuntimeError("verification lacks an owned service transaction")
                service_result = ops.compensate_services(service_request, service_result)
                _record_service_step(builder, "service_compensation", service_result)
                if service_result.interruption is not None:
                    raise service_result.interruption
                return _failed_outcome(
                    request,
                    ops,
                    plan,
                    builder,
                    mutation,
                    services_changed=services_changed,
                    service_compensated=service_result.compensated,
                    failure="verification_failed",
                )
        _require_same_publication(ops, published)
        receipt = builder.finish(
            "succeeded", notes=["Setup completed; contextual Gate results are bound to the config digest."]
        )
        written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
        if not written:
            return SetupOutcome(
                "partial",
                plan,
                True,
                mutation.changed,
                services_changed,
                request.verify,
                "receipt_write_failed",
                receipt,
                False,
            )
        return SetupOutcome(
            "succeeded", plan, True, mutation.changed, services_changed, request.verify, None, receipt, written
        )
    except BaseException as primary:
        interruption = primary if not isinstance(primary, Exception) else None
        if mutation is None:
            try:
                mutation = transaction.recover_mutation()
            except BaseException as recovery_error:
                if interruption is None and not isinstance(recovery_error, Exception):
                    interruption = recovery_error
        service_compensated = service_request is None and settlement_authorized
        if settlement_authorized and service_request is not None and service_result is not None:
            if (
                interruption is None
                and service_result.interruption is not None
                and not isinstance(service_result.interruption, Exception)
            ):
                interruption = service_result.interruption
            service_compensated = service_result.compensated
            if not service_compensated:
                try:
                    service_result = ops.compensate_services(service_request, service_result)
                    service_compensated = service_result.compensated
                    if (
                        interruption is None
                        and service_result.interruption is not None
                        and not isinstance(service_result.interruption, Exception)
                    ):
                        interruption = service_result.interruption
                    try:
                        _record_service_step(builder, "service_compensation", service_result)
                    except BaseException as receipt_error:
                        if interruption is None and not isinstance(receipt_error, Exception):
                            interruption = receipt_error
                except BaseException as cleanup_error:
                    if interruption is None and not isinstance(cleanup_error, Exception):
                        interruption = cleanup_error
        config_rolled_back = False
        if mutation is None:
            try:
                config_rolled_back = transaction.unchanged()
            except BaseException as cleanup_error:
                if interruption is None and not isinstance(cleanup_error, Exception):
                    interruption = cleanup_error
        if mutation is not None and service_compensated and settlement_authorized:
            try:
                config_rolled_back = ops.rollback_config(mutation) is True
            except BaseException as cleanup_error:
                if interruption is None and not isinstance(cleanup_error, Exception):
                    interruption = cleanup_error
        fully_settled = service_compensated and config_rolled_back
        failure = "interrupted" if interruption is not None else "apply_failed"
        status = "failed" if fully_settled else "partial"
        try:
            receipt = _settlement_receipt(
                builder,
                plan,
                ops,
                status,
                failure,
                config_rolled_back=config_rolled_back,
                service_compensated=service_compensated,
            )
        except BaseException as receipt_error:
            if interruption is None and not isinstance(receipt_error, Exception):
                interruption = receipt_error
            try:
                receipt = _recovery_receipt(
                    plan,
                    status,
                    failure,
                    config_rolled_back=config_rolled_back,
                    service_compensated=service_compensated,
                )
            except BaseException as recovery_error:
                receipt = {}
                if interruption is None and not isinstance(recovery_error, Exception):
                    interruption = recovery_error
        try:
            written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
        except BaseException as publish_error:
            written = False
            if interruption is None and not isinstance(publish_error, Exception):
                interruption = publish_error
        if not written:
            status = "partial"
            failure = f"{failure}_receipt_write_failed"
        if interruption is not None:
            if interruption is primary:
                raise
            raise interruption from primary
        return SetupOutcome(
            status,
            plan,
            mutation is not None,
            bool(mutation and mutation.changed),
            services_changed,
            False,
            failure,
            receipt,
            written,
        )


def _published_snapshot(ops: SetupOps, plan: SetupPlan, mutation: ConfigMutation) -> ConfigSnapshot:
    published = ops.snapshot(plan.config_path)
    if (
        not published.existed
        or published.content != plan.config_text.encode("utf-8")
        or published.digest != plan.config_digest
        or published.mode != 0o600
        or published.identity != mutation.installed_identity
    ):
        raise OSError("published cluster config readback was not exact")
    return published


def _require_same_publication(ops: SetupOps, published: ConfigSnapshot | None) -> None:
    if published is None or ops.snapshot(published.path) != published:
        raise OSError("cluster config changed during setup")


def _record_verification(
    builder: Any,
    progress: SetupVerificationProgress,
    result: SetupVerification | None,
) -> None:
    doctor_status: operator_receipt.ReceiptStepStatus = (
        "succeeded"
        if progress.doctor_completed and progress.doctor_ok is True
        else "failed"
        if progress.doctor_completed and progress.doctor_ok is False
        else "unknown"
    )
    builder.add_step(
        "doctor_verify",
        doctor_status,
        counts={"attempted": int(progress.doctor_attempted), "passed": int(progress.doctor_ok is True)},
    )
    if not progress.smoke_attempted:
        smoke_status: operator_receipt.ReceiptStepStatus = "planned"
    elif not progress.smoke_completed or (result is not None and result.smoke_ok is None):
        smoke_status = "unknown"
    elif result is not None and result.smoke_ok:
        smoke_status = "succeeded"
    else:
        smoke_status = "failed"
    counts = (
        result.smoke_counts()
        if result is not None
        else {
            "attempted": int(progress.smoke_attempted),
            "passed": 0,
            "source_matched": 0,
            "source_rank_coverage": 0,
            "status_rank_coverage": 0,
            "teardown_confirmed": 0,
            "world": 0,
        }
    )
    builder.add_step("cluster_smoke", smoke_status, counts=counts)


def _pre_config_service_outcome(
    request: SetupRequest,
    ops: SetupOps,
    plan: SetupPlan,
    builder: Any,
    transaction: ConfigTransaction,
    *,
    services_changed: bool,
    service_compensated: bool,
    failure: str,
) -> SetupOutcome:
    config_unchanged = transaction.unchanged()
    fully_settled = service_compensated and config_unchanged
    status: operator_receipt.ReceiptStatus = "failed" if fully_settled else "partial"
    builder.add_step(
        "setup_compensation",
        "succeeded" if fully_settled else "unknown",
        counts={"config_unchanged": int(config_unchanged), "service_compensated": int(service_compensated)},
    )
    builder.add_step(failure, "unknown" if failure.endswith("_unknown") else "failed")
    note = (
        "No service change remains, and the config was not published; private coordination state may remain."
        if fully_settled
        else "Service state owned by this setup may remain; setup did not try to publish the config."
    )
    receipt = builder.finish(status, notes=[note])
    written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
    if not written:
        status = "partial"
        failure = f"{failure}_receipt_write_failed"
    return SetupOutcome(status, plan, False, False, services_changed, False, failure, receipt, written)


def _uncertain_verification_outcome(
    request: SetupRequest,
    ops: SetupOps,
    plan: SetupPlan,
    builder: Any,
    mutation: ConfigMutation,
    *,
    services_changed: bool,
) -> SetupOutcome:
    failure = "verification_unknown"
    builder.add_step("setup_compensation", "unknown", counts={"config_rolled_back": 0, "service_compensated": 0})
    builder.add_step(failure, "unknown")
    receipt = builder.finish(
        "partial",
        notes=["Verification outcome is unknown; config and Worker service state were retained for inspection."],
    )
    written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
    if not written:
        failure = f"{failure}_receipt_write_failed"
    return SetupOutcome("partial", plan, True, mutation.changed, services_changed, False, failure, receipt, written)


def _failed_outcome(
    request: SetupRequest,
    ops: SetupOps,
    plan: SetupPlan,
    builder: Any,
    mutation: ConfigMutation,
    *,
    services_changed: bool,
    service_compensated: bool,
    failure: str,
) -> SetupOutcome:
    config_rolled_back = service_compensated and ops.rollback_config(mutation) is True
    fully_compensated = service_compensated and config_rolled_back
    status: operator_receipt.ReceiptStatus = "failed" if fully_compensated else "partial"
    builder.add_step(
        "setup_compensation",
        "succeeded" if fully_compensated else "unknown",
        counts={"config_rolled_back": int(config_rolled_back), "service_compensated": int(service_compensated)},
    )
    builder.add_step(failure, "failed")
    note = (
        "Published config and service changes were rolled back; private backup and coordination state may remain."
        if fully_compensated
        else "Config or service state owned by this setup may remain; inspect it by hand."
    )
    receipt = builder.finish(status, notes=[note])
    written = publish_receipt(ops.write_receipt, receipt, request.receipt_path, required=True)
    if not written:
        status = "partial"
        failure = f"{failure}_receipt_write_failed"
    return SetupOutcome(status, plan, True, mutation.changed, services_changed, False, failure, receipt, written)


def _record_service_step(builder: Any, name: str, result: SetupServiceResult) -> None:
    status: operator_receipt.ReceiptStepStatus
    if result.certainty is ServiceCertainty.SUCCEEDED:
        status = "succeeded"
    elif result.certainty is ServiceCertainty.FAILED:
        status = "failed"
    else:
        status = "unknown"
    builder.add_step(name, status, counts=result.receipt_counts())


def _settlement_receipt(
    builder: Any,
    plan: SetupPlan,
    ops: SetupOps,
    status: operator_receipt.ReceiptStatus,
    failure: str,
    *,
    config_rolled_back: bool,
    service_compensated: bool,
) -> dict[str, object]:
    counts = {"config_rolled_back": int(config_rolled_back), "service_compensated": int(service_compensated)}
    try:
        builder.add_step("setup_compensation", "succeeded" if status == "failed" else "unknown", counts=counts)
        builder.add_step(failure, "failed" if status == "failed" else "unknown")
        return cast(
            dict[str, object],
            builder.finish(
                status,
                notes=["Setup stopped; see setup_compensation. Private backup and coordination state may remain."],
            ),
        )
    except Exception:
        return _recovery_receipt(
            plan,
            "partial",
            failure,
            config_rolled_back=config_rolled_back,
            service_compensated=service_compensated,
        )


def _recovery_receipt(
    plan: SetupPlan,
    status: operator_receipt.ReceiptStatus,
    failure: str,
    *,
    config_rolled_back: bool,
    service_compensated: bool,
) -> dict[str, object]:
    """Rebuild the final receipt from the plan if the original receipt builder failed."""
    recovery = plan.receipt_start.materialize(
        operator_receipt.OperatorReceiptBuilder,
        plan.resolution.name,
        plan.config_digest,
        plan.source.manifest,
    )
    counts = {"config_rolled_back": int(config_rolled_back), "service_compensated": int(service_compensated)}
    recovery.add_step("receipt_recovery", "partial")
    recovery.add_step("setup_compensation", "succeeded" if status == "failed" else "unknown", counts=counts)
    recovery.add_step(failure, "failed" if status == "failed" else "unknown")
    return cast(
        dict[str, object],
        recovery.finish(status, notes=["Setup stopped; this receipt was rebuilt after the first one failed."]),
    )
