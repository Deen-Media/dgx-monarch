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

    def recheck_dependencies(self) -> None:
        pass

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

    def retire_releases(self, staged: StagedRelease) -> OperationResult:
        self._event("retire")
        return OK

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
        "retire",
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


def test_an_unknown_activation_records_each_hosts_count_and_names_the_repair():
    ops = FakeOps()
    ops.activation = OperationResult(
        Certainty.UNKNOWN, counts={"hosts_switched": 0, "hosts_unknown": 1, "hosts_not_reached": 1})

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "activation_unknown"
    assert "different releases" in result.message and "TROUBLESHOOTING entry 118" in result.message
    step = next(item for item in result.receipt["steps"] if item["name"] == "activation")
    assert step["status"] == "unknown"
    assert step["counts"] == {"hosts_not_reached": 1, "hosts_switched": 0, "hosts_unknown": 1}
    assert "compensate" not in ops.events and ops.events.count("start_no_sync") == 0


def test_an_unknown_activation_after_a_switched_host_names_only_the_forward_repair():
    """A host already on the new release rules out `dgxm up` as the repair."""
    ops = FakeOps()
    ops.activation = OperationResult(
        Certainty.UNKNOWN, counts={"hosts_switched": 1, "hosts_unknown": 1, "hosts_not_reached": 0})

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    end = _end(result)
    assert result.code == "activation_unknown"
    assert end.startswith("At least one host runs the new release") and "TROUBLESHOOTING entry 118" in end
    assert "`dgxm up`" not in end


def _end(result) -> str:
    return result.message.split(". ", 1)[1]


def test_a_failed_activation_whose_compensation_is_unconfirmed_says_the_pair_may_be_split():
    ops = FakeOps()
    ops.activation = OperationResult(
        Certainty.FAILED, counts={"hosts_switched": 1, "hosts_failed": 1, "hosts_not_reached": 0})
    ops.compensation = UNKNOWN

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "activation_failed" and result.receipt["status"] == "partial"
    end = _end(result)
    assert "may run different releases" in end and "every Worker stopped" in end
    assert "`dgxm up`" in end and "TROUBLESHOOTING entry 118" in end


def test_a_restored_release_whose_workers_did_not_all_start_names_dgxm_up():
    ops = FakeOps()
    ops.activation = FAIL
    ops.start_results = [UNKNOWN]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "activation_failed"
    end = _end(result)
    assert "back on the old release" in end and "not every Worker runs" in end and "`dgxm up`" in end
    assert "compensated" in result.message.split(". ", 1)[0]


@pytest.mark.parametrize(("configure", "code", "words"), [
    (lambda ops: (setattr(ops, "start_results", [FAIL]), setattr(ops, "compensation", UNKNOWN)),
     "start_failed", "may run different releases"),
    (lambda ops: (setattr(ops, "smoke_result", OperationResult(Certainty.FAILED, "killed")),
                  setattr(ops, "start_results", [OK, UNKNOWN])),
     "cluster_smoke_killed", "not every Worker runs"),
    (lambda ops: (setattr(ops, "doctor_result", FAIL), setattr(ops, "stop_results", [OK, UNKNOWN])),
     "doctor_failed", "its Worker may still run"),
    (lambda ops: setattr(ops, "smoke_result", FAIL), "cluster_smoke_failed", "runs the old release again"),
    (lambda ops: setattr(ops, "stop_results", [UNKNOWN]), "stop_unknown", "Nothing was activated"),
    (lambda ops: setattr(ops, "activation", UNKNOWN), "activation_unknown", "may run different releases"),
    (lambda ops: setattr(ops, "start_results", [UNKNOWN]), "start_unknown", "its Worker may still run"),
    (lambda ops: setattr(ops, "doctor_result", UNKNOWN), "doctor_unknown", "its Worker stopped"),
    (lambda ops: setattr(ops, "finalize_result", UNKNOWN), "finalize_unknown", "every Worker run the new release"),
])
def test_every_end_state_says_where_the_hosts_are_and_the_repair(configure, code, words):
    """The last sentence names each host's release, whether its Worker runs, and the repair."""
    ops = FakeOps()
    configure(ops)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == code
    end = _end(result)
    assert words in end, end
    # The sentence is its own receipt note and survives the 240-character note limit whole.
    assert result.receipt["notes"][1] == end and len(end) <= 240
    assert ("`dgxm up`" in end or "TROUBLESHOOTING entry 118" in end or "rerun the update" in end
            or "new release" in end)


def test_the_longest_end_sentence_fits_one_receipt_note():
    from dgx_monarch.cli.receipt_sanitize import sanitize_note
    from dgx_monarch.cli.update_end_state import END_STATES

    for text in END_STATES.values():
        sentence = "The restore was not confirmed. " + text
        assert sanitize_note(sentence) == sentence, sentence


def test_a_restore_that_fell_short_says_so_before_where_the_hosts_are():
    unconfirmed = FakeOps()
    unconfirmed.start_results = [FAIL]
    unconfirmed.compensation = UNKNOWN
    smoke = FakeOps()
    smoke.smoke_result = OperationResult(Certainty.FAILED, "killed")
    smoke.start_results = [OK, UNKNOWN]
    restored = FakeOps()
    restored.smoke_result = FAIL

    for ops, code in ((unconfirmed, "start_failed"), (smoke, "cluster_smoke_killed")):
        result = execute_verified_update(UpdateRequest(assume_yes=True), ops)
        assert result.code == code
        assert _end(result).startswith("The restore was not confirmed. ")
    result = execute_verified_update(UpdateRequest(assume_yes=True), restored)
    assert "was not confirmed" not in result.message and "runs the old release again" in result.message


@pytest.mark.parametrize(("answer", "status"), [
    (OperationResult(Certainty.SUCCEEDED, counts={"releases_retired": 2, "releases_not_retired": 0}), "succeeded"),
    (OperationResult(Certainty.SUCCEEDED, counts={"releases_retired": 1, "releases_not_retired": 1}), "partial"),
    (OperationResult(Certainty.UNKNOWN, counts={"releases_retired": 0, "hosts_retire_unconfirmed": 1}), "unknown"),
    (KeyboardInterrupt(), "unknown"),
    (RuntimeError("host=worker-01"), "unknown"),
])
def test_retirement_is_its_own_step_after_finalize_and_never_changes_the_outcome(answer, status):
    ops = FakeOps()
    events = []

    def retire_releases(staged):
        events.append("retire")
        if isinstance(answer, BaseException):
            raise answer
        return answer

    ops.retire_releases = retire_releases  # type: ignore[attr-defined]

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "updated" and result.exit_code == 0 and result.receipt["status"] == "succeeded"
    names = _step_names(result)
    assert names.index("release_retirement") == names.index("finalize") + 1
    step = next(item for item in result.receipt["steps"] if item["name"] == "release_retirement")
    assert step["status"] == status and events == ["retire"]
    assert "cleanup" in ops.events


def test_activation_counts_that_are_not_plain_counts_are_left_out():
    ops = FakeOps()
    ops.activation = OperationResult(Certainty.SUCCEEDED, counts={"host worker-01": 1})

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    step = next(item for item in result.receipt["steps"] if item["name"] == "activation")
    assert result.code == "updated" and step["counts"] == {}


@pytest.mark.parametrize(("check", "cause", "words"), [
    ("cluster_smoke", "cleanup_unconfirmed", "could not confirm the stop of its own ProcMesh"),
    ("cluster_smoke", "killed", "killed by a signal"),
    ("cluster_smoke", "timed_out", "did not report within its time limit"),
    ("doctor", "killed", "killed by a signal"),
    ("doctor", "timed_out", "did not report within its time limit"),
])
def test_a_named_check_failure_restores_the_prior_release_and_says_why(check: str, cause: str, words: str):
    ops = FakeOps()
    setattr(ops, "smoke_result" if check == "cluster_smoke" else "doctor_result",
            OperationResult(Certainty.FAILED, cause))

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == f"{check}_{cause}"
    assert words in result.message and words in result.receipt["notes"][0]
    assert "runs the old release again" in result.message
    assert result.receipt["status"] == "failed"
    assert ops.events.count("stop") == 2
    assert ops.events.count("start_no_sync") == 2
    assert "compensate" in ops.events and "readback" not in ops.events
    step = next(item for item in result.receipt["steps"] if item["name"] == check)
    assert step["status"] == "failed" and step["notes"] == [cause]
    restored = next(item for item in result.receipt["steps"] if item["name"] == "prior_release_start")
    assert restored["status"] == "succeeded"


def test_an_unrecognized_failure_cause_reads_as_a_plain_failure():
    ops = FakeOps()
    ops.smoke_result = OperationResult(Certainty.FAILED, "host worker-01 at /private")

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "cluster_smoke_failed"
    step = next(item for item in result.receipt["steps"] if item["name"] == "cluster_smoke")
    assert step["notes"] == []


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


def test_dependency_drift_after_confirmation_refuses_without_service_mutation():
    ops = FakeOps()

    def changed():
        assert "confirm" in ops.events
        raise UpdatePreconditionError("target_dependencies_unsatisfied", "candidate prerequisite changed")

    ops.recheck_dependencies = changed
    result = execute_verified_update(UpdateRequest(), ops)
    assert result.code == "target_dependencies_unsatisfied"
    assert result.receipt["status"] == "failed"
    assert "stop" not in ops.events and "activate" not in ops.events
    assert "cleanup" in ops.events


@pytest.mark.parametrize("inspection", [False, True])
@pytest.mark.parametrize("clear", [False, True])
def test_privileged_inspection_receipt_records_selection_without_claiming_success(inspection, clear):
    ops = FakeOps()
    ops.privileged_process_inspection = inspection
    if not clear:
        ops.activity[0] = ActivitySnapshot(False, False, 0, None)
    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)
    note = "Privileged process inspection was required for activity guards."
    assert (note in result.receipt["notes"]) is inspection
    assert result.succeeded is clear
    if not clear:
        assert "stop" not in ops.events


# The receipts TROUBLESHOOTING entry 118 reads for operation_exception and
# operation_interrupted: the last step before the ones the update adds as it
# settles, and those closing steps, for each place a failure can land after the
# Worker stop began.
_CLOSING = {"fail_closed_stop", "stage_cleanup", "stage_retention"}


def _tail(receipt: dict) -> tuple[tuple[str, str], list[str]]:
    steps = [(step["name"], step["status"]) for step in receipt["steps"]]
    end = len(steps)
    while end and steps[end - 1][0] in _CLOSING:
        end -= 1
    return steps[end - 1], [name for name, _status in steps[end:]]


@pytest.mark.parametrize(("setup", "raise_at", "last", "closing"), [
    ({}, "stop", ("pre_stop_recheck", "succeeded"), ["stage_retention"]),
    ({}, "activate", ("worker_stop", "succeeded"), ["stage_retention"]),
    ({"activation": FAIL}, "compensate", ("activation", "failed"), ["stage_retention"]),
    ({"activation": FAIL}, "start_no_sync", ("compensation", "succeeded"), ["stage_retention"]),
    ({}, "start_no_sync", ("activation", "succeeded"), ["stage_retention"]),
    ({}, "doctor", ("worker_start", "succeeded"), ["fail_closed_stop", "stage_retention"]),
    ({}, "cluster_smoke", ("doctor", "succeeded"), ["fail_closed_stop", "stage_retention"]),
    ({}, "readback", ("cluster_smoke", "succeeded"), ["fail_closed_stop", "stage_retention"]),
])
def test_an_exception_after_the_stop_began_leaves_the_steps_the_partial_section_reads(
        setup, raise_at, last, closing):
    """An exception inside the stop leaves pre_stop_recheck last and no
    worker_stop; inside the compensation after a failed activation it leaves
    the failed activation last. The settling steps always close the list."""
    ops = FakeOps()
    for name, value in setup.items():
        setattr(ops, name, value)
    ops.raise_at = raise_at

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert (result.code, result.receipt["status"]) == ("operation_exception", "partial")
    assert _tail(result.receipt) == (last, closing)
    if raise_at == "stop":
        assert "worker_stop" not in _step_names(result) and "activate" not in ops.events


def _interrupt_at(ops: FakeOps, name: str, occurrence: int = 1) -> KeyboardInterrupt:
    interrupt = KeyboardInterrupt()
    record = ops._event
    seen = []

    def event(event_name: str) -> None:
        record(event_name)
        if event_name == name:
            seen.append(event_name)
            if len(seen) == occurrence:
                raise interrupt

    ops._event = event  # type: ignore[method-assign]
    return interrupt


@pytest.mark.parametrize(("setup", "at", "occurrence", "last", "closing"), [
    ({}, "stop", 1, ("pre_stop_recheck", "succeeded"), ["stage_retention"]),
    ({"activation": FAIL}, "compensate", 1, ("activation", "failed"), ["stage_retention"]),
    ({"doctor_result": UNKNOWN}, "stop", 2, ("doctor", "unknown"), ["stage_retention"]),
    ({}, "commit_driver", 1, ("exact_readback", "succeeded"), ["fail_closed_stop", "stage_retention"]),
    ({}, "finalize", 1, ("driver_commit", "succeeded"), ["stage_retention"]),
])
def test_an_interruption_after_the_stop_began_leaves_the_steps_the_partial_section_reads(
        setup, at, occurrence, last, closing):
    ops = FakeOps()
    for name, value in setup.items():
        setattr(ops, name, value)
    interrupt = _interrupt_at(ops, at, occurrence)

    with pytest.raises(KeyboardInterrupt) as raised:
        execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert raised.value is interrupt
    receipt = interrupted_receipt(interrupt)
    assert receipt is not None and receipt["status"] == "partial"
    assert _tail(receipt) == (last, closing)
    if (at, occurrence) == ("stop", 1):
        assert "worker_stop" not in [step["name"] for step in receipt["steps"]]
