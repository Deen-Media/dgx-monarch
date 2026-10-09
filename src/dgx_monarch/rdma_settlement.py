"""Durable publication for RDMA drop and settlement outcomes."""
from __future__ import annotations

from typing import Any

from .transfer_utils import prefer_error, reconcile_error, safe_note


def drop_cancellation(failures: Any) -> BaseException | None:
    """Return the first exact cancellation observed while releasing buffers."""
    cancellation: BaseException | None = None
    for failure in failures:
        error = failure.error
        if not isinstance(error, Exception):
            cancellation = prefer_error(
                cancellation, error,
                "another RDMA buffer release cancellation also failed")
    return cancellation


def reconcile_drop_error(
    failures: Any, error: BaseException,
) -> tuple[BaseException | None, BaseException | None]:
    """Compose a drop boundary error with durable native failures."""
    cancellation = drop_cancellation(failures)
    if cancellation is not None:
        reconcile_error(
            cancellation, error, "RDMA drop outcome publication also failed")
        return None, None
    cause = None
    for failure in failures:
        native_error = failure.error
        if native_error is error:
            continue
        if cause is None:
            cause = native_error
        else:
            safe_note(error, "another RDMA buffer release also failed", native_error)
    return error, cause


def retry_poison_publication(
    first_error: BaseException, failures: Any, keepalive: Any, phase: str,
    unowned: Any, publish: Any,
) -> BaseException:
    """Retry one poison publication while preserving exact cancellation."""
    try:
        missing = unowned(failures)
        if missing:
            publish(missing, keepalive=keepalive, phase=phase)
    except BaseException as retry_error:
        return prefer_error(
            first_error, retry_error,
            "RDMA poison publication retry also failed")
    return first_error


def publish_drop_result(
    owner: dict | None, failures: Any, error: BaseException | None,
) -> None:
    if owner is None:
        return
    outcome = (failures, error)
    for _attempt in range(2):
        try:
            owner["drop_outcome"] = outcome
        except BaseException as publication_error:
            error = prefer_error(
                error, publication_error, "RDMA drop outcome publication failed")
            outcome = (failures, error)
            continue
        return
    if "drop_outcome" not in owner:
        owner["drop_outcome"] = outcome


def publish_settlement(
    owner: dict | None,
    failures: Any,
    poison_error: BaseException | None,
    cleanup_error: BaseException | None,
) -> tuple[Any, BaseException | None, BaseException | None]:
    outcome = (failures, poison_error, cleanup_error)
    if owner is None:
        return outcome
    for _attempt in range(2):
        try:
            owner["outcome"] = outcome
        except BaseException as publication_error:
            cleanup_error = prefer_error(
                cleanup_error, publication_error,
                "RDMA settlement outcome publication failed")
            outcome = (failures, poison_error, cleanup_error)
            continue
        return outcome
    if "outcome" not in owner:
        owner["outcome"] = outcome
    return owner["outcome"]
