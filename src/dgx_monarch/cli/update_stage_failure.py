"""Errors for a failed stage that carry the status of the cleanup that followed."""
from __future__ import annotations

from typing import Literal

CleanupStatus = Literal["succeeded", "failed", "unknown"]
_CLEANUP_ATTRIBUTE = "_dgxm_stage_cleanup_status"


class StageFailure(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        exit_code: int,
        cleanup_status: CleanupStatus,
    ) -> None:
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.cleanup_status = cleanup_status
        super().__init__(message)


def attach_stage_cleanup(error: BaseException, status: CleanupStatus) -> None:
    try:
        object.__setattr__(error, _CLEANUP_ATTRIBUTE, status)
    except BaseException:
        pass


def interrupted_stage_cleanup(error: BaseException) -> CleanupStatus | None:
    try:
        value = object.__getattribute__(error, _CLEANUP_ATTRIBUTE)
    except BaseException:
        return None
    return value if value in ("succeeded", "failed", "unknown") else None
