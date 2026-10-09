"""Run cleanup after a failed stage and choose the exception to raise."""
from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn

from .update_stage_failure import CleanupStatus, StageFailure, attach_stage_cleanup
from .update_transaction import Certainty, OperationResult, UpdatePreconditionError


def _attempt(cleanup: Callable[[], OperationResult]) -> tuple[Certainty, BaseException | None]:
    try:
        result = cleanup()
    except BaseException as error:
        return Certainty.UNKNOWN, error
    if not isinstance(result, OperationResult):
        return Certainty.UNKNOWN, None
    return result.certainty, None


def settle_stage_failure(
    primary: BaseException,
    *,
    release_started: bool,
    cleanup_release: Callable[[], OperationResult],
    cleanup_worktree: Callable[[], OperationResult],
) -> NoReturn:
    """Run the cleanups, then raise a cancelling primary, else a cancelling cleanup error, else a StageFailure."""
    release = _attempt(cleanup_release) if release_started else (Certainty.SUCCEEDED, None)
    worktree = _attempt(cleanup_worktree)
    certainties = (release[0], worktree[0])
    status: CleanupStatus = (
        "succeeded" if all(value == Certainty.SUCCEEDED for value in certainties)
        else "failed" if Certainty.FAILED in certainties and Certainty.UNKNOWN not in certainties
        else "unknown"
    )
    cleanup_cancellation = next(
        (error for error in (release[1], worktree[1]) if error is not None and not isinstance(error, Exception)),
        None,
    )
    if not isinstance(primary, Exception):
        attach_stage_cleanup(primary, status)
        raise primary
    if cleanup_cancellation is not None:
        attach_stage_cleanup(cleanup_cancellation, status)
        raise cleanup_cancellation
    if isinstance(primary, UpdatePreconditionError):
        raise StageFailure(primary.code, primary.message, primary.exit_code, status) from None
    message = (
        "release staging failed and cleanup was not confirmed"
        if status != "succeeded" else "release staging failed before any live change; cleanup succeeded"
    )
    raise StageFailure("stage_failed", message, 1, status) from None
