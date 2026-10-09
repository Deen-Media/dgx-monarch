"""Coordinate ``dgxm update --verify`` and stop on uncertain changes.

Platform operations are injected through UpdateOps. A timeout during stop,
activation or driver-pin promotion leaves the outcome unknown; do not roll
back, restart or clean up after it. One containment stop is allowed only if
target startup was confirmed and the recorded generation still matches.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

from .update_receipt import UpdateReceipt, attach_interrupted_receipt
from .update_stage_failure import StageFailure, interrupted_stage_cleanup
from .update_types import (
    DEFAULT_TARGET as DEFAULT_TARGET,
)
from .update_types import (
    ActivitySnapshot,
    Certainty,
    ExactReadback,
    OperationResult,
    RepoSnapshot,
    StagedRelease,
    UpdateOps,
    UpdatePreconditionError,
    UpdateRequest,
    UpdateResult,
)

_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
GATE_NOTE = (
    "Source-bound Gate grants are re-evaluated after source activation; "
    "sticky FAIL evidence remains authoritative."
)


@dataclass(frozen=True)
class _Step:
    name: str
    status: Literal["planned", "succeeded", "failed", "partial", "unknown"]
    counts: Mapping[str, int] | None = None
    notes: tuple[str, ...] = ()


class _Abort(UpdatePreconditionError):
    def __init__(self, code: str, message: str, exit_code: int = 1) -> None:
        super().__init__(code, message, exit_code)


def _validate_target(value: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise _Abort("target_ambiguous", "target did not resolve to one full commit")
    return value


def _repo_guard(snapshot: RepoSnapshot, *, target: str, current: str | None = None) -> str:
    current_sha = _validate_target(snapshot.current_sha)
    if current is not None and current_sha != current:
        raise _Abort("repo_changed", "driver checkout changed during update staging", 2)
    if not snapshot.clean:
        raise _Abort("repo_dirty", "driver checkout is not clean: git status --untracked-files=all shows changes, "
                     "or a file is marked assume-unchanged or skip-worktree", 2)
    if not snapshot.fast_forward and current_sha != target:
        raise _Abort("not_fast_forward", "target is not a fast-forward of the driver checkout", 2)
    return current_sha


def _validate_stage(staged: StagedRelease, target: str) -> None:
    counts_valid = bool(
        type(staged.worker_releases_expected) is int
        and type(staged.worker_releases_staged) is int
        and type(staged.ranks_expected) is int
        and staged.worker_releases_expected > 0
        and staged.worker_releases_staged == staged.worker_releases_expected
        and staged.ranks_expected > 0
    )
    valid = (
        isinstance(staged.target_sha, str)
        and staged.target_sha == target
        and _SHA.fullmatch(staged.target_sha) is not None
        and isinstance(staged.version, str)
        and _VERSION.fullmatch(staged.version) is not None
        and isinstance(staged.torchmonarch_pin, str)
        and _VERSION.fullmatch(staged.torchmonarch_pin) is not None
        and isinstance(staged.source_manifest, str)
        and _DIGEST.fullmatch(staged.source_manifest) is not None
        and isinstance(staged.dependency_manifest, str)
        and _DIGEST.fullmatch(staged.dependency_manifest) is not None
        and staged.detached_head is True
        and staged.pin_staged_no_deps is True
        and staged.dependencies_validated is True
        and staged.driver_release_ready is True
        and counts_valid
    )
    if not valid:
        raise _Abort("stage_invalid", "staged release did not satisfy the exact release contract")


def _require_clear(snapshot: ActivitySnapshot) -> None:
    blockers = snapshot.blockers()
    if blockers:
        raise _Abort(
            blockers[0],
            "update blocked: ComfyUI, a Render session, a lease or an actor is active, or activity is unknown",
            2,
        )


def _readback_matches(readback: ExactReadback, staged: StagedRelease) -> bool:
    return bool(
        readback.certainty == Certainty.SUCCEEDED
        and readback.ranks_expected == staged.ranks_expected
        and readback.ranks_matched == staged.ranks_expected
        and readback.target_sha == staged.target_sha
        and readback.version == staged.version
        and readback.torchmonarch_pin == staged.torchmonarch_pin
        and readback.source_manifest == staged.source_manifest
    )


def _attempt(operation: Callable[[], OperationResult]) -> OperationResult:
    try:
        result = operation()
    except Exception:
        return OperationResult(Certainty.UNKNOWN)
    return result if isinstance(result, OperationResult) else OperationResult(Certainty.UNKNOWN)


def _operation_status(result: OperationResult) -> Literal[
    "succeeded", "failed", "unknown"
]:
    if result.certainty == Certainty.SUCCEEDED:
        return "succeeded"
    if result.certainty == Certainty.FAILED:
        return "failed"
    return "unknown"


def _fail_closed_stop(
    ops: UpdateOps, staged: StagedRelease, steps: list[_Step]
) -> bool:
    """Attempt a Worker stop only; do not roll back or restart the release."""
    stopped = _attempt(lambda: ops.stop_started_workers(staged))
    steps.append(_Step("fail_closed_stop", _operation_status(stopped)))
    return stopped.certainty == Certainty.SUCCEEDED


def _restore_prior_release(
    ops: UpdateOps, staged: StagedRelease, steps: list[_Step],
    *, require_driver_prior: bool = False,
) -> bool:
    """Stop the new release, then restore the prior release only after confirming shutdown."""
    stopped = _attempt(lambda: ops.stop_started_workers(staged))
    steps.append(_Step("fail_closed_stop", _operation_status(stopped)))
    if stopped.certainty != Certainty.SUCCEEDED:
        return False
    if require_driver_prior:
        prior = _attempt(lambda: ops.prior_driver_readback(staged))
        steps.append(_Step("driver_prior_readback", _operation_status(prior)))
        if prior.certainty != Certainty.SUCCEEDED:
            return False
    compensated = _attempt(lambda: ops.compensate(staged))
    steps.append(_Step("post_activation_compensation", _operation_status(compensated)))
    if compensated.certainty != Certainty.SUCCEEDED:
        return False
    restored = _attempt(lambda: ops.start_prior_workers(staged))
    steps.append(_Step("prior_release_start", _operation_status(restored)))
    if restored.certainty == Certainty.FAILED:
        stopped_again = _attempt(lambda: ops.stop_started_workers(staged))
        steps.append(_Step("fail_closed_stop", _operation_status(stopped_again)))
    return restored.certainty == Certainty.SUCCEEDED


def execute_verified_update(request: UpdateRequest, ops: UpdateOps) -> UpdateResult:
    """Execute one verified update, never continuing past uncertain mutation."""
    receipt_builder = UpdateReceipt()
    steps: list[_Step] = []
    target: str | None = None
    staged: StagedRelease | None = None
    current: str | None = None
    mutation_started = False
    target_started = False
    driver_committed = False
    target_settlement_attempted = False
    cleanup_allowed = False
    settled = False
    code = "unexpected_error"
    message = "verified update failed unexpectedly"
    exit_code = 1
    terminal: Literal["planned", "succeeded", "failed", "partial"] = "failed"
    pending_base: BaseException | None = None

    try:
        target = _validate_target(ops.resolve_target(request.target_ref))
        steps.append(_Step("target_resolution", "succeeded"))
        current = _repo_guard(ops.repo_snapshot(target), target=target)
        steps.append(_Step("repository_guard", "succeeded"))
        _require_clear(ops.activity_snapshot())
        steps.append(_Step("activity_guard", "succeeded"))

        staged = ops.stage(target)
        cleanup_allowed = True
        _validate_stage(staged, target)
        steps.append(_Step(
            "release_staging",
            "succeeded",
            counts={
                "worker_releases": staged.worker_releases_staged,
                "ranks": staged.ranks_expected,
            },
        ))
        if not request.assume_yes and not ops.confirm(staged):
            steps.append(_Step("confirmation", "planned", notes=("operator declined activation",)))
            raise _Abort("cancelled", "update cancelled before mutation", 2)
        steps.append(_Step("confirmation", "succeeded"))

        _repo_guard(ops.repo_snapshot(target), target=target, current=current)
        _require_clear(ops.activity_snapshot())
        steps.append(_Step("pre_stop_recheck", "succeeded"))

        mutation_started = True
        cleanup_allowed = False
        stop = ops.stop_workers()
        if stop.certainty != Certainty.SUCCEEDED:
            steps.append(_Step("worker_stop", _operation_status(stop)))
            if stop.certainty == Certainty.FAILED:
                settled = True
                cleanup_allowed = True
            suffix = "unknown" if stop.certainty == Certainty.UNKNOWN else "failed"
            raise _Abort(f"stop_{suffix}", "worker stop was not confirmed; nothing was started")
        steps.append(_Step("worker_stop", "succeeded"))

        activation = ops.activate(staged)
        if activation.certainty == Certainty.UNKNOWN:
            steps.append(_Step("activation", "unknown"))
            raise _Abort(
                "activation_unknown",
                "activation outcome is unknown; the update did not roll back or start workers",
            )
        if activation.certainty == Certainty.FAILED:
            steps.append(_Step("activation", "failed"))
            compensation = ops.compensate(staged)
            steps.append(_Step("compensation", _operation_status(compensation)))
            if compensation.certainty == Certainty.SUCCEEDED:
                restored = ops.start_prior_workers(staged)
                steps.append(_Step("prior_release_start", _operation_status(restored)))
                settled = restored.certainty == Certainty.SUCCEEDED
                cleanup_allowed = settled
                if restored.certainty == Certainty.FAILED:
                    stopped_again = _attempt(lambda: ops.stop_started_workers(staged))
                    steps.append(_Step("fail_closed_stop", _operation_status(stopped_again)))
            compensation_note = (
                "activation failed; definite changes were compensated"
                if compensation.certainty == Certainty.SUCCEEDED
                else "activation failed and compensation was not confirmed"
            )
            raise _Abort("activation_failed", compensation_note)
        steps.append(_Step("activation", "succeeded"))

        started = ops.start_target_workers(staged)
        if started.certainty == Certainty.UNKNOWN:
            steps.append(_Step("worker_start", "unknown"))
            raise _Abort(
                "start_unknown",
                "new worker service start outcome is unknown; the update did not roll back",
            )
        if started.certainty == Certainty.FAILED:
            steps.append(_Step("worker_start", "failed"))
            settled = _restore_prior_release(ops, staged, steps)
            cleanup_allowed = settled
            raise _Abort("start_failed", "new worker service start failed")
        steps.append(_Step("worker_start", "succeeded"))
        target_started = True

        for name, check_call in (
            ("doctor", ops.doctor),
            ("cluster_smoke", lambda: ops.cluster_smoke(staged)),
        ):
            check = check_call()
            if check.certainty == Certainty.UNKNOWN:
                steps.append(_Step(name, "unknown"))
                target_settlement_attempted = True
                _fail_closed_stop(ops, staged, steps)
                raise _Abort(
                    f"{name}_unknown",
                    f"{name} outcome is unknown; the update did not roll back",
                )
            if check.certainty == Certainty.FAILED:
                steps.append(_Step(name, "failed"))
                target_settlement_attempted = True
                settled = _restore_prior_release(ops, staged, steps)
                cleanup_allowed = settled
                raise _Abort(f"{name}_failed", f"{name} did not verify the activated release")
            steps.append(_Step(name, "succeeded"))

        readback = ops.exact_readback(staged)
        if readback.certainty == Certainty.UNKNOWN:
            steps.append(_Step("exact_readback", "unknown", counts={
                "ranks": max(readback.ranks_matched, 0),
            }))
            target_settlement_attempted = True
            _fail_closed_stop(ops, staged, steps)
            raise _Abort(
                "readback_unknown",
                "all-rank exact source readback is unknown; the update did not roll back",
            )
        if not _readback_matches(readback, staged):
            steps.append(_Step("exact_readback", "failed", counts={
                "ranks": max(readback.ranks_matched, 0),
            }))
            target_settlement_attempted = True
            settled = _restore_prior_release(ops, staged, steps)
            cleanup_allowed = settled
            raise _Abort("readback_mismatch", "all-rank exact source readback did not match")
        steps.append(_Step("exact_readback", "succeeded", counts={"ranks": readback.ranks_matched}))

        committed = _attempt(lambda: ops.commit_driver(staged))
        if committed.certainty != Certainty.SUCCEEDED:
            steps.append(_Step("driver_commit", _operation_status(committed)))
            target_settlement_attempted = True
            if committed.certainty == Certainty.FAILED:
                settled = _restore_prior_release(
                    ops, staged, steps, require_driver_prior=True
                )
                cleanup_allowed = settled
            else:
                _fail_closed_stop(ops, staged, steps)
            suffix = "unknown" if committed.certainty == Certainty.UNKNOWN else "failed"
            raise _Abort(f"driver_commit_{suffix}", "live driver commit was not confirmed")
        steps.append(_Step("driver_commit", "succeeded"))
        driver_committed = True

        finalized = _attempt(lambda: ops.finalize(staged))
        if finalized.certainty != Certainty.SUCCEEDED:
            steps.append(_Step("finalize", _operation_status(finalized)))
            suffix = "unknown" if finalized.certainty == Certainty.UNKNOWN else "failed"
            raise _Abort(f"finalize_{suffix}", "prior release cleanup was not confirmed")
        steps.append(_Step("finalize", "succeeded"))
        cleanup_allowed = True
        code, message, exit_code, terminal = "updated", "verified update completed", 0, "succeeded"
    except _Abort as exc:
        code, message, exit_code = exc.code, exc.message, exc.exit_code
        terminal = "planned" if exc.code == "cancelled" else (
            "failed" if settled or not mutation_started else "partial"
        )
    except StageFailure as exc:
        code, message, exit_code = exc.code, exc.message, exc.exit_code
        terminal = "failed" if exc.cleanup_status == "succeeded" else "partial"
        steps.append(_Step(
            "stage_cleanup",
            "succeeded" if exc.cleanup_status == "succeeded" else exc.cleanup_status,
        ))
    except UpdatePreconditionError as exc:
        code, message, exit_code = exc.code, exc.message, exc.exit_code
        terminal = "partial" if mutation_started else "failed"
    except Exception:
        # An exception after mutation started is an unknown outcome: do not
        # compensate. Compensation needs a definite FAILED result first.
        terminal = "partial" if mutation_started else "failed"
        code = "operation_exception"
        message = (
            "an update operation raised after mutation; services remain fail-closed"
            if mutation_started
            else "an update operation failed before mutation"
        )
        if (
            staged is not None
            and target_started
            and not driver_committed
            and not target_settlement_attempted
        ):
            target_settlement_attempted = True
            try:
                _fail_closed_stop(ops, staged, steps)
            except BaseException as recovery_error:
                steps.append(_Step("fail_closed_stop", "unknown"))
                if not isinstance(recovery_error, Exception):
                    pending_base = recovery_error
    except BaseException as exc:
        pending_base = exc
        terminal = "partial" if mutation_started else "failed"
        code = "operation_interrupted"
        message = "verified update was interrupted; fail-closed settlement was attempted"
        stage_status = interrupted_stage_cleanup(exc)
        if stage_status is not None:
            steps.append(_Step(
                "stage_cleanup",
                "succeeded" if stage_status == "succeeded" else stage_status,
            ))
            if stage_status != "succeeded":
                terminal = "partial"
        if (
            staged is not None
            and target_started
            and not driver_committed
            and not target_settlement_attempted
        ):
            target_settlement_attempted = True
            try:
                _fail_closed_stop(ops, staged, steps)
            except BaseException:
                steps.append(_Step("fail_closed_stop", "unknown"))

    if staged is not None and cleanup_allowed:
        try:
            cleaned = ops.cleanup_stage(staged)
        except BaseException as exc:
            cleaned = OperationResult(Certainty.UNKNOWN)
            if pending_base is None and not isinstance(exc, Exception):
                pending_base = exc
        if cleaned.certainty == Certainty.SUCCEEDED:
            steps.append(_Step("stage_cleanup", "succeeded"))
        else:
            steps.append(_Step("stage_cleanup", _operation_status(cleaned)))
            was_success = terminal == "succeeded"
            terminal, exit_code = "partial", 1
            if was_success:
                code = "stage_cleanup_unconfirmed"
                message = "release activated and verified, but staging cleanup was not confirmed"
    elif staged is not None:
        steps.append(_Step(
            "stage_retention",
            "succeeded",
            notes=("recovery artifacts retained for inspection",),
        ))

    source = (
        staged.source_manifest
        if (
            staged is not None
            and isinstance(staged.source_manifest, str)
            and _DIGEST.fullmatch(staged.source_manifest) is not None
        )
        else None
    )
    try:
        receipt = receipt_builder.finish(
            steps, terminal, target=target, source=source, notes=(message, GATE_NOTE)
        )
    except BaseException:
        if pending_base is not None:
            raise pending_base from None
        raise
    if pending_base is not None:
        attach_interrupted_receipt(pending_base, receipt)
        raise pending_base
    return UpdateResult(exit_code=exit_code, code=code, message=message, receipt=receipt)
