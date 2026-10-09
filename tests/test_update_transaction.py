from __future__ import annotations

from dataclasses import replace

import pytest

from dgx_monarch.cli.update_receipt import interrupted_receipt
from dgx_monarch.cli.update_stage_failure import attach_stage_cleanup
from dgx_monarch.cli.update_transaction import (
    ActivitySnapshot,
    Certainty,
    ExactReadback,
    OperationResult,
    RepoSnapshot,
    StagedRelease,
    UpdatePreconditionError,
    UpdateRequest,
    execute_verified_update,
)

OLD = "1" * 40
TARGET = "2" * 40
SOURCE = "3" * 64
DEPS = "4" * 64
OK = OperationResult(Certainty.SUCCEEDED)
FAIL = OperationResult(Certainty.FAILED)
UNKNOWN = OperationResult(Certainty.UNKNOWN)


def _stage(**changes: object) -> StagedRelease:
    stage = StagedRelease(
        target_sha=TARGET,
        version="0.5.0",
        torchmonarch_pin="0.7.0",
        source_manifest=SOURCE,
        dependency_manifest=DEPS,
        detached_head=True,
        pin_staged_no_deps=True,
        dependencies_validated=True,
        driver_release_ready=True,
        worker_releases_expected=2,
        worker_releases_staged=2,
        ranks_expected=2,
    )
    return replace(stage, **changes)


class FakeOps:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.snapshots = [RepoSnapshot(OLD, True, True), RepoSnapshot(OLD, True, True)]
        self.activity = [ActivitySnapshot(False, False, 0, 0)] * 2
        self.staged = _stage()
        self.confirmed = True
        self.stop_results = [OK]
        self.activation = OK
        self.compensation = OK
        self.start_results = [OK]
        self.doctor_result = OK
        self.smoke_result = OK
        self.readback = ExactReadback(
            Certainty.SUCCEEDED, 2, 2, TARGET, "0.5.0", "0.7.0", SOURCE
        )
        self.commit_result = OK
        self.prior_driver = OK
        self.finalize_result = OK
        self.cleanup_result = OK
        self.raise_at: str | None = None

    def _event(self, name: str) -> None:
        self.events.append(name)
        if self.raise_at == name:
            raise RuntimeError("host=worker-01 path=/private secret=do-not-record")

    def resolve_target(self, target_ref: str) -> str:
        self._event(f"resolve:{target_ref}")
        return TARGET

    def repo_snapshot(self, target_sha: str) -> RepoSnapshot:
        self._event(f"repo:{target_sha}")
        return self.snapshots.pop(0)

    def activity_snapshot(self) -> ActivitySnapshot:
        self._event("activity")
        return self.activity.pop(0)

    def stage(self, target_sha: str) -> StagedRelease:
        self._event(f"stage:{target_sha}")
        return self.staged

    def confirm(self, staged: StagedRelease) -> bool:
        self._event("confirm")
        return self.confirmed

    def stop_workers(self) -> OperationResult:
        self._event("stop")
        return self.stop_results.pop(0) if self.stop_results else OK

    def stop_started_workers(self, staged: StagedRelease) -> OperationResult:
        assert staged is self.staged
        return self.stop_workers()

    def activate(self, staged: StagedRelease) -> OperationResult:
        self._event("activate")
        return self.activation

    def compensate(self, staged: StagedRelease) -> OperationResult:
        self._event("compensate")
        return self.compensation

    def start_workers_without_sync(self) -> OperationResult:
        self._event("start_no_sync")
        return self.start_results.pop(0) if self.start_results else OK

    def start_target_workers(self, staged: StagedRelease) -> OperationResult:
        assert staged is self.staged
        return self.start_workers_without_sync()

    def start_prior_workers(self, staged: StagedRelease) -> OperationResult:
        assert staged is self.staged
        return self.start_workers_without_sync()

    def doctor(self) -> OperationResult:
        self._event("doctor")
        return self.doctor_result

    def cluster_smoke(self, staged: StagedRelease) -> OperationResult:
        self._event("cluster_smoke")
        return self.smoke_result

    def exact_readback(self, staged: StagedRelease) -> ExactReadback:
        self._event("readback")
        return self.readback

    def commit_driver(self, staged: StagedRelease) -> OperationResult:
        self._event("commit_driver")
        return self.commit_result

    def prior_driver_readback(self, staged: StagedRelease) -> OperationResult:
        self._event("driver_prior_readback")
        return self.prior_driver

    def finalize(self, staged: StagedRelease) -> OperationResult:
        self._event("finalize")
        return self.finalize_result

    def cleanup_stage(self, staged: StagedRelease) -> OperationResult:
        self._event("cleanup")
        return self.cleanup_result


def _step_names(result) -> list[str]:
    return [step["name"] for step in result.receipt["steps"]]


def test_success_obeys_exact_order_and_rechecks_immediately_before_stop():
    ops = FakeOps()

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.succeeded
    assert result.code == "updated"
    assert ops.events == [
        "resolve:origin/HEAD",
        f"repo:{TARGET}",
        "activity",
        f"stage:{TARGET}",
        f"repo:{TARGET}",
        "activity",
        "stop",
        "activate",
        "start_no_sync",
        "doctor",
        "cluster_smoke",
        "readback",
        "commit_driver",
        "finalize",
        "cleanup",
    ]
    assert result.receipt["status"] == "succeeded"
    assert result.receipt["selection"] == {
        "profile": None,
        "target": TARGET,
        "source_hashes": {"dgx_monarch": SOURCE},
    }
    assert "restart" not in ops.events


@pytest.mark.parametrize(
    ("snapshot", "code"),
    [
        (RepoSnapshot(OLD, False, True), "repo_dirty"),
        (RepoSnapshot(OLD, True, False), "not_fast_forward"),
        (RepoSnapshot("short", True, True), "target_ambiguous"),
    ],
)
def test_repository_preconditions_block_before_staging(snapshot: RepoSnapshot, code: str):
    ops = FakeOps()
    ops.snapshots = [snapshot]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == code
    assert not any(event.startswith("stage:") for event in ops.events)
    assert "stop" not in ops.events


@pytest.mark.parametrize(
    ("activity", "code"),
    [
        (ActivitySnapshot(True, False, 0, 0), "comfy_running"),
        (ActivitySnapshot(False, True, 0, 0), "render_active"),
        (ActivitySnapshot(False, False, 1, 0), "active_leases"),
        (ActivitySnapshot(False, False, 0, 1), "marked_actors"),
        (ActivitySnapshot(False, None, 0, 0), "activity_unknown"),
        (ActivitySnapshot(False, False, -1, 0), "activity_unknown"),
    ],
)
def test_activity_evidence_must_be_clear_and_unambiguous(
    activity: ActivitySnapshot, code: str
):
    ops = FakeOps()
    ops.activity = [activity]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == code
    assert not any(event.startswith("stage:") for event in ops.events)
    assert "stop" not in ops.events


def test_positive_activity_evidence_outranks_unknown_secondary_telemetry():
    snapshot = ActivitySnapshot(True, None, None, None)

    assert snapshot.blockers() == ("comfy_running",)


@pytest.mark.parametrize(
    "changes",
    [
        {"target_sha": OLD},
        {"detached_head": False},
        {"pin_staged_no_deps": False},
        {"dependencies_validated": False},
        {"driver_release_ready": False},
        {"worker_releases_staged": 1},
        {"worker_releases_expected": True, "worker_releases_staged": True},
        {"ranks_expected": 0},
        {"source_manifest": "bad"},
    ],
)
def test_invalid_stage_is_cleaned_and_live_state_is_untouched(changes: dict[str, object]):
    ops = FakeOps()
    ops.staged = _stage(**changes)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "stage_invalid"
    assert ops.events[-1] == "cleanup"
    assert "stop" not in ops.events
    assert "activate" not in ops.events


def test_declined_confirmation_cleans_stage_without_mutation():
    ops = FakeOps()
    ops.confirmed = False

    result = execute_verified_update(UpdateRequest(), ops)

    assert result.code == "cancelled"
    assert result.exit_code == 2
    assert result.receipt["status"] == "planned"
    assert "stop" not in ops.events
    assert ops.events[-1] == "cleanup"


def test_typed_precondition_refusal_is_operator_actionable_and_pre_mutation():
    ops = FakeOps()
    def refuse(_target: str) -> StagedRelease:
        raise UpdatePreconditionError(
            "release_layout_unsupported",
            "verified update requires remote-only worker hosts",
        )
    ops.stage = refuse  # type: ignore[method-assign]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "release_layout_unsupported"
    assert result.exit_code == 2
    assert "stop" not in ops.events


def test_assume_yes_skips_prompt_but_not_second_guard():
    ops = FakeOps()
    ops.activity[1] = ActivitySnapshot(False, False, 1, 0)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert "confirm" not in ops.events
    assert result.code == "active_leases"
    assert "stop" not in ops.events


def test_checkout_change_during_staging_blocks_before_stop():
    ops = FakeOps()
    ops.snapshots[1] = RepoSnapshot("5" * 40, True, True)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "repo_changed"
    assert "stop" not in ops.events


@pytest.mark.parametrize(
    ("stop", "status", "cleanup"),
    [(FAIL, "failed", True), (UNKNOWN, "partial", False)],
)
def test_unconfirmed_stop_never_activates_or_starts(
    stop: OperationResult, status: str, cleanup: bool
):
    ops = FakeOps()
    ops.stop_results = [stop]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code in {"stop_failed", "stop_unknown"}
    assert result.receipt["status"] == status
    assert "activate" not in ops.events
    assert "start_no_sync" not in ops.events
    assert ("cleanup" in ops.events) is cleanup
    worker_stop = next(
        step for step in result.receipt["steps"] if step["name"] == "worker_stop"
    )
    assert worker_stop["status"] == stop.certainty.value


def test_definite_activation_failure_is_compensated_before_prior_restart():
    ops = FakeOps()
    ops.activation = FAIL

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "activation_failed"
    assert result.receipt["status"] == "failed"
    assert ops.events.index("compensate") < ops.events.index("start_no_sync")
    assert "doctor" not in ops.events


def test_failed_prior_restart_records_final_fail_closed_stop():
    ops = FakeOps()
    ops.activation = FAIL
    ops.start_results = [FAIL]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "activation_failed"
    assert result.receipt["status"] == "partial"
    assert ops.events.count("stop") == 2
    assert "fail_closed_stop" in _step_names(result)


@pytest.mark.parametrize("mode", [UNKNOWN, "exception"])
def test_ambiguous_activation_never_compensates_or_starts(mode: object):
    ops = FakeOps()
    if mode == "exception":
        ops.raise_at = "activate"
    else:
        ops.activation = UNKNOWN

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code in {"activation_unknown", "operation_exception"}
    assert "compensate" not in ops.events
    assert "start_no_sync" not in ops.events


def test_failed_start_restores_prior_and_never_verifies_target():
    ops = FakeOps()
    ops.start_results = [FAIL]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "start_failed"
    assert ops.events.count("stop") == 2
    assert "compensate" in ops.events
    assert ops.events.count("start_no_sync") == 2
    assert "doctor" not in ops.events


def test_partial_target_start_with_replaced_generation_never_stops_new_occupant():
    ops = FakeOps()
    ops.start_results = [FAIL]

    def generation_refused(_staged: StagedRelease) -> OperationResult:
        ops.events.append("generation_refused")
        return UNKNOWN

    ops.stop_started_workers = generation_refused  # type: ignore[method-assign]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "start_failed"
    assert ops.events.count("stop") == 1
    assert ops.events.count("start_no_sync") == 1
    assert "generation_refused" in ops.events
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    assert "stage_retention" in _step_names(result)


def test_unknown_start_preserves_recovery_state_without_further_mutation():
    ops = FakeOps()
    ops.start_results = [UNKNOWN]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "start_unknown"
    assert result.receipt["status"] == "partial"
    assert ops.events.count("stop") == 1
    assert ops.events.count("start_no_sync") == 1
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    assert "doctor" not in ops.events
    assert "stage_retention" in _step_names(result)
    assert result.receipt["summary"]["step_counts"]["unknown"] == 1


def test_exception_during_target_start_grants_no_further_mutation():
    ops = FakeOps()
    ops.raise_at = "start_no_sync"

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "operation_exception"
    assert ops.events.count("stop") == 1
    assert ops.events.count("start_no_sync") == 1
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    assert "stage_retention" in _step_names(result)


def test_cancellation_during_target_start_preserves_identity_without_mutation():
    ops = FakeOps()
    interrupt = KeyboardInterrupt()

    def interrupted_start() -> OperationResult:
        ops.events.append("start_no_sync")
        raise interrupt

    ops.start_target_workers = lambda _staged: interrupted_start()  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert ops.events.count("stop") == 1
    assert ops.events.count("start_no_sync") == 1
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    attached = interrupted_receipt(interrupt)
    assert attached is not None
    assert "stage_retention" in [step["name"] for step in attached["steps"]]


def test_failed_doctor_quarantines_without_running_smoke():
    ops = FakeOps()
    ops.doctor_result = FAIL

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "doctor_failed"
    assert result.receipt["status"] == "failed"
    assert ops.events.count("stop") == 2
    assert "cluster_smoke" not in ops.events
    assert "readback" not in ops.events


def test_partial_prior_restart_with_replaced_generation_refuses_stale_stop():
    ops = FakeOps()
    ops.doctor_result = FAIL
    ops.start_results = [OK, FAIL]
    generation_stops = 0

    def generation_stop(_staged: StagedRelease) -> OperationResult:
        nonlocal generation_stops
        generation_stops += 1
        ops.events.append("generation_stop")
        return OK if generation_stops == 1 else UNKNOWN

    ops.stop_started_workers = generation_stop  # type: ignore[method-assign]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "doctor_failed"
    assert generation_stops == 2
    assert ops.events.count("stop") == 1
    assert ops.events.count("start_no_sync") == 2
    assert "cleanup" not in ops.events
    assert "stage_retention" in _step_names(result)


def test_failed_smoke_quarantines_without_readback():
    ops = FakeOps()
    ops.smoke_result = FAIL

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "cluster_smoke_failed"
    assert result.receipt["status"] == "failed"
    assert ops.events.count("stop") == 2
    assert "readback" not in ops.events


@pytest.mark.parametrize("check", ["doctor", "cluster_smoke"])
def test_unknown_post_activation_check_preserves_state_without_rollback(check: str):
    ops = FakeOps()
    if check == "doctor":
        ops.doctor_result = UNKNOWN
    else:
        ops.smoke_result = UNKNOWN

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == f"{check}_unknown"
    assert result.receipt["status"] == "partial"
    assert ops.events.count("stop") == 2
    assert ops.events.count("start_no_sync") == 1
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    step = next(item for item in result.receipt["steps"] if item["name"] == check)
    assert step["status"] == "unknown"
    assert "stage_retention" in _step_names(result)


@pytest.mark.parametrize("doctor_result", [UNKNOWN, FAIL])
def test_interrupted_target_settlement_is_attempted_exactly_once(
    doctor_result: OperationResult,
):
    ops = FakeOps()
    ops.doctor_result = doctor_result
    interrupt = KeyboardInterrupt()
    calls = 0

    def interrupted_second_stop() -> OperationResult:
        nonlocal calls
        calls += 1
        ops.events.append("stop")
        if calls == 2:
            raise interrupt
        return OK

    ops.stop_workers = interrupted_second_stop  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert calls == 2
    assert ops.events.count("stop") == 2
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events


@pytest.mark.parametrize(
    "readback",
    [
        ExactReadback(Certainty.SUCCEEDED, 2, 1, TARGET, "0.5.0", "0.7.0", SOURCE),
        ExactReadback(Certainty.SUCCEEDED, 2, 2, OLD, "0.5.0", "0.7.0", SOURCE),
        ExactReadback(Certainty.SUCCEEDED, 2, 2, TARGET, "0.5.1", "0.7.0", SOURCE),
        ExactReadback(Certainty.SUCCEEDED, 2, 2, TARGET, "0.5.0", "0.6.0", SOURCE),
        ExactReadback(Certainty.SUCCEEDED, 2, 2, TARGET, "0.5.0", "0.7.0", DEPS),
    ],
)
def test_any_exact_readback_mismatch_quarantines(readback: ExactReadback):
    ops = FakeOps()
    ops.readback = readback

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "readback_mismatch"
    assert result.receipt["status"] == "failed"
    assert ops.events.count("stop") == 2
    assert "finalize" not in ops.events


def test_unknown_exact_readback_preserves_state_without_rollback():
    ops = FakeOps()
    ops.readback = ExactReadback(Certainty.UNKNOWN, 2, 0)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "readback_unknown"
    assert result.receipt["status"] == "partial"
    assert ops.events.count("stop") == 2
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    step = next(
        item for item in result.receipt["steps"] if item["name"] == "exact_readback"
    )
    assert step["status"] == "unknown"
    assert step["counts"] == {"ranks": 0}


def test_definite_driver_commit_failure_restores_the_prior_release():
    ops = FakeOps()
    ops.commit_result = FAIL

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "driver_commit_failed"
    assert result.receipt["status"] == "failed"
    assert ops.events.count("stop") == 2
    assert ops.events.index("commit_driver") < ops.events.index("compensate")
    assert ops.events.index("compensate") < ops.events.index("start_no_sync", 9)
    assert "finalize" not in ops.events


@pytest.mark.parametrize("mode", [UNKNOWN, "exception"])
def test_ambiguous_driver_commit_preserves_state_without_mutation(mode: object):
    ops = FakeOps()
    if mode == "exception":
        ops.raise_at = "commit_driver"
    else:
        ops.commit_result = UNKNOWN

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "driver_commit_unknown"
    assert ops.events.count("stop") == 2
    assert "compensate" not in ops.events
    assert ops.events.count("start_no_sync") == 1
    assert "finalize" not in ops.events
    assert "cleanup" not in ops.events
    commit = next(
        item for item in result.receipt["steps"] if item["name"] == "driver_commit"
    )
    assert commit["status"] == "unknown"


def test_commit_interruption_with_ambiguous_driver_state_never_restarts_prior():
    ops = FakeOps()
    interrupt = KeyboardInterrupt()
    ops.prior_driver = UNKNOWN

    def interrupted_commit(_staged: StagedRelease) -> OperationResult:
        ops.events.append("commit_driver")
        raise interrupt

    ops.commit_driver = interrupted_commit  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert ops.events.count("stop") == 2
    assert "driver_prior_readback" not in ops.events
    assert "compensate" not in ops.events
    assert ops.events.count("start_no_sync") == 1
    assert "cleanup" not in ops.events


def test_commit_interruption_never_uses_prior_readback_as_mutation_authority():
    ops = FakeOps()
    interrupt = KeyboardInterrupt()

    def interrupted_commit(_staged: StagedRelease) -> OperationResult:
        ops.events.append("commit_driver")
        raise interrupt

    ops.commit_driver = interrupted_commit  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert ops.events.count("stop") == 2
    assert "driver_prior_readback" not in ops.events
    assert "compensate" not in ops.events
    assert ops.events.count("start_no_sync") == 1
    assert "cleanup" not in ops.events


def test_interruption_after_driver_commit_preserves_target_and_recovery_state():
    ops = FakeOps()
    interrupt = KeyboardInterrupt()
    ops.prior_driver = UNKNOWN

    def interrupted_finalize(_staged: StagedRelease) -> OperationResult:
        ops.events.append("finalize")
        raise interrupt

    ops.finalize = interrupted_finalize  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert ops.events.count("stop") == 1
    assert "driver_prior_readback" not in ops.events
    assert "compensate" not in ops.events
    assert ops.events.count("start_no_sync") == 1
    assert "cleanup" not in ops.events


@pytest.mark.parametrize("finalize", [FAIL, UNKNOWN])
def test_finalize_failure_after_commit_never_rolls_back_verified_target(
    finalize: OperationResult,
):
    ops = FakeOps()
    ops.finalize_result = finalize

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code in {"finalize_failed", "finalize_unknown"}
    assert ops.events.count("stop") == 1
    assert "compensate" not in ops.events
    assert ops.events.count("start_no_sync") == 1


def test_cleanup_failure_turns_verified_activation_into_partial_result():
    ops = FakeOps()
    ops.cleanup_result = UNKNOWN

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "stage_cleanup_unconfirmed"
    assert result.exit_code == 1
    assert result.receipt["status"] == "partial"
    assert _step_names(result)[-1] == "stage_cleanup"


def test_cleanup_uncertainty_turns_a_cancelled_plan_into_partial_result():
    ops = FakeOps()
    ops.confirmed = False
    ops.cleanup_result = UNKNOWN

    result = execute_verified_update(UpdateRequest(), ops)

    assert result.code == "cancelled"
    assert result.exit_code == 1
    assert result.receipt["status"] == "partial"


def test_unexpected_error_text_is_not_copied_to_receipt():
    ops = FakeOps()
    ops.raise_at = "stage:" + TARGET

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)
    payload = str(result.receipt)

    assert result.code == "operation_exception"
    assert "worker-01" not in payload
    assert "/private" not in payload
    assert "do-not-record" not in payload


def test_unexpected_error_after_start_stops_target_without_rollback():
    ops = FakeOps()
    ops.raise_at = "doctor"

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "operation_exception"
    assert ops.events.count("stop") == 2
    assert "fail_closed_stop" in _step_names(result)
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    assert "stage_retention" in _step_names(result)


def test_cancellation_after_start_retains_recovery_state_and_attaches_receipt():
    ops = FakeOps()
    interrupt = KeyboardInterrupt()

    def interrupted_doctor() -> OperationResult:
        ops.events.append("doctor")
        raise interrupt

    ops.doctor = interrupted_doctor  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert "cleanup" not in ops.events
    assert ops.events.count("stop") == 2
    assert "compensate" not in ops.events
    receipt = interrupted_receipt(interrupt)
    assert receipt is not None and receipt["status"] == "partial"
    assert "stage_retention" in [step["name"] for step in receipt["steps"]]


@pytest.mark.parametrize("phase", ["doctor", "cluster_smoke", "exact_readback"])
@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit])
def test_post_start_interruption_preserves_primary_and_settles_once(
    phase: str, interrupt_type: type[BaseException],
):
    ops = FakeOps()
    interrupt = interrupt_type(130) if interrupt_type is SystemExit else interrupt_type()

    def interrupted(*_args: object) -> OperationResult:
        ops.events.append(phase if phase != "exact_readback" else "readback")
        raise interrupt

    setattr(ops, phase, interrupted)

    with pytest.raises(interrupt_type) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert ops.events.count("stop") == 2
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events
    attached = interrupted_receipt(interrupt)
    assert attached is not None
    assert "stage_retention" in [step["name"] for step in attached["steps"]]


def test_exception_while_restoring_compensated_release_retains_recovery_state():
    ops = FakeOps()
    ops.activation = FAIL
    ops.raise_at = "start_no_sync"

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "operation_exception"
    assert ops.events.count("stop") == 1
    assert "cleanup" not in ops.events
    assert "stage_retention" in _step_names(result)


def test_invalid_resolver_output_is_not_used_as_receipt_target():
    ops = FakeOps()
    ops.resolve_target = lambda _ref: "origin/main"  # type: ignore[method-assign]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "target_ambiguous"
    assert result.receipt["selection"]["target"] is None


def test_keyboard_interrupt_after_start_stops_target_without_rollback():
    ops = FakeOps()
    interrupt = KeyboardInterrupt()
    def interrupted_doctor() -> OperationResult:
        ops.events.append("doctor")
        raise interrupt
    ops.doctor = interrupted_doctor  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    assert ops.events.count("stop") == 2
    assert "compensate" not in ops.events
    assert "cleanup" not in ops.events


def test_system_exit_before_mutation_keeps_identity_and_cleans_staged_release():
    ops = FakeOps()
    interrupt = SystemExit(130)
    def interrupted_confirmation(_staged: StagedRelease) -> bool:
        ops.events.append("confirm")
        raise interrupt
    ops.confirm = interrupted_confirmation  # type: ignore[method-assign]

    with pytest.raises(SystemExit) as raised:
        execute_verified_update(UpdateRequest(), ops)

    assert raised.value is interrupt
    assert "stop" not in ops.events
    assert ops.events[-1] == "cleanup"


def test_interrupted_stage_with_uncertain_cleanup_has_partial_receipt():
    ops = FakeOps()
    interrupt = KeyboardInterrupt()
    attach_stage_cleanup(interrupt, "unknown")
    def interrupted_stage(_target: str) -> StagedRelease:
        raise interrupt
    ops.stage = interrupted_stage  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    receipt = interrupted_receipt(interrupt)
    assert receipt is not None and receipt["status"] == "partial"
