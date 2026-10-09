"""Bounded process roots for unresolved driver RDMA read jobs."""
from __future__ import annotations

import threading
from typing import Any

from .transfer_utils import prefer_error as prefer

MAX_PENDING_RDMA_READ_JOBS = 64
_LOCK = threading.Lock()
_JOBS: list[Any] = []


def retire_token(guard: Any, token: object) -> BaseException | None:
    """Retire one exact key and prove absence after every attempted call."""
    primary: BaseException | None = None
    for _attempt in range(2):
        try:
            guard.end_read(token)
        except BaseException as error:
            primary = prefer(primary, error, "later read-token retirement failure")
        try:
            present = guard.has_read_token(token)
        except BaseException as proof_error:
            primary = prefer(primary, proof_error, "read-token retirement proof failed")
            continue
        if not present:
            return primary
    if primary is None:
        primary = RuntimeError("sample read token remained active after retirement")
    raise primary


def registered(job: Any) -> bool:
    with _LOCK:
        return any(existing is job for existing in _JOBS)


def register(job: Any) -> BaseException | None:
    """Publish one strong root, adopting an interrupted append."""
    first: BaseException | None = None
    for _attempt in range(2):
        try:
            with _LOCK:
                if any(existing is job for existing in _JOBS):
                    return first
                if len(_JOBS) >= MAX_PENDING_RDMA_READ_JOBS:
                    raise RuntimeError(
                        "RDMA scratch job capacity is exhausted by unresolved reads")
                _JOBS.append(job)
        except BaseException as error:
            first = prefer(
                first, error, "later RDMA scratch job registration failure")
            if registered(job):
                return first
            continue
        return first
    if first is None:
        raise RuntimeError("RDMA scratch job registration made no attempt")
    raise first


def unregister(job: Any) -> BaseException | None:
    """Remove only this exact root, proving an interrupted pop applied."""
    first: BaseException | None = None
    for _attempt in range(2):
        try:
            with _LOCK:
                for index, existing in enumerate(_JOBS):
                    if existing is job:
                        _JOBS.pop(index)
                        break
        except BaseException as error:
            first = prefer(
                first, error, "later RDMA scratch root retirement failure")
            if not registered(job):
                return first
            continue
        if not registered(job):
            return first
    if first is None:
        raise RuntimeError("RDMA scratch job root could not be retired")
    raise first


def recover_final_delivery(job: Any, boundary: BaseException) -> None:
    """Publish a final delivery boundary and prove detached-root retirement."""
    result, primary = job.outcome_owner[0]
    outcome = (result, prefer(primary, boundary, "final outcome delivery failed"))
    job.outcome_owner[0] = outcome
    if job.holder:
        job.holder[0] = outcome
    if not registered(job):
        return
    for _attempt in range(2):
        try:
            job._deliver_outcome()
        except BaseException as delivery_error:
            result, primary = job.outcome_owner[0]
            outcome = (result, prefer(
                primary, delivery_error, "final outcome delivery retry failed"))
            job.outcome_owner[0] = outcome
            if job.holder:
                job.holder[0] = outcome
            continue
        if not registered(job):
            return
        result, primary = job.outcome_owner[0]
        outcome = (result, prefer(
            primary,
            RuntimeError("final outcome copied before scratch root retirement"),
            "final scratch root remains registered"))
        job.outcome_owner[0] = outcome
        if job.holder:
            job.holder[0] = outcome
    if registered(job):
        unretired_error = job.outcome_owner[0][1]
        if unretired_error is None:
            unretired_error = RuntimeError(
                "final scratch root retirement could not be confirmed")
        raise unretired_error


def pending_job_count() -> int:
    with _LOCK:
        return len(_JOBS)


def pending_job_states() -> tuple[str, ...]:
    with _LOCK:
        return tuple(job.state for job in _JOBS)


def pending_job_count_for_handle(handle: Any) -> int:
    with _LOCK:
        return sum(getattr(job.guard, "handle", None) is handle for job in _JOBS)
