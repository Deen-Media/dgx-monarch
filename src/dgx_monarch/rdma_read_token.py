"""Read-token publication, proof and retirement for one scratch read job.

Every function takes the job as its first argument and reaches the job's own
helpers back through it, so an overridden or rebound attribute still decides
what runs.
"""
from __future__ import annotations

from typing import Any

from .rdma_job_registry import prefer


def token_capable(job: Any) -> bool:
    return (
        hasattr(job.guard, "begin_read")
        and hasattr(job.guard, "ensure_read_token")
        and hasattr(job.guard, "has_read_token")
        and hasattr(job.guard, "end_read")
    )


def token_is_confirmed(job: Any) -> bool:
    return bool(
        job.token_owner
        and job.guard.has_read_token(job.token_owner[0])
    )


def reconcile_token(
    job: Any, primary: BaseException | None
) -> tuple[bool, BaseException | None]:
    error = primary
    for _attempt in range(2):
        try:
            if job.token_owner:
                job.guard.ensure_read_token(job.token_owner[0])
            else:
                job.guard.begin_read(job.token_owner)
        except BaseException as publication_error:
            error = prefer(
                error, publication_error, "read-token publication retry failed")
        try:
            confirmed = job._token_is_confirmed()
        except BaseException as proof_error:
            error = prefer(
                error, proof_error, "read-token publication proof failed")
            confirmed = False
        if confirmed:
            job.confirmation_error = None
            return True, error
    job.confirmation_error = (
        error
        if error is not None
        else RuntimeError("sample read token publication was not confirmed")
    )
    return False, job.confirmation_error


def reconcile_prepare(
    job: Any, primary: BaseException,
) -> tuple[bool, BaseException | None]:
    if not job.token_capable:
        job.token_retired = True
        return True, primary
    return job._reconcile_token(primary)


def before_failure_ack(job: Any) -> None:
    if job.confirmation_error is not None:
        raise RuntimeError(
            "pre-read job-token publication was not confirmed"
        ) from job.confirmation_error
    if job.token_capable and not job._token_is_confirmed():
        raise RuntimeError("pre-read job token retired before ACK")


def mark_token_retired(job: Any) -> bool:
    if not job.token_owner:
        job.token_retired = True
        return True
    try:
        retired = not job.guard.has_read_token(job.token_owner[0])
    except BaseException:
        raise
    if retired:
        try:
            job.token_retired = True
        except BaseException as boundary:
            if job.guard.has_read_token(job.token_owner[0]):
                raise
            job.token_retired = True
            raise boundary
    return retired
