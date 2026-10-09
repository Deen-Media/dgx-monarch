"""Durable per-runner attempt state for an exact-once RDMA read job."""
from __future__ import annotations

from typing import Any

from .rdma_job_registry import prefer

NOT_ENTERED = "NOT_ENTERED"
ENTERED = "ENTERED"
COMPLETED = "COMPLETED"


class Claim:
    def __init__(self, primary: BaseException | None = None) -> None:
        self.key = object()
        self.primary_owner: list[BaseException] = []
        self.operation_phase: tuple[str, object | None] = (NOT_ENTERED, None)
        self.operation_outcome: list[tuple[Any, BaseException | None]] = []
        if primary is not None:
            self.primary_owner.append(primary)

    @property
    def primary(self) -> BaseException | None:
        return self.primary_owner[0] if self.primary_owner else None

    def remember(self, error: BaseException, label: str) -> None:
        primary = prefer(self.primary, error, label)
        if self.primary_owner:
            self.primary_owner[0] = primary
            return
        try:
            self.primary_owner.append(primary)
        except BaseException as boundary:
            if not self.primary_owner or self.primary_owner[0] is not primary:
                raise
            self.primary_owner[0] = prefer(
                primary, boundary, "primary failure publication was interrupted")

    def enter_operation(self) -> None:
        try:
            if self.operation_phase[0] == NOT_ENTERED:
                self.operation_phase = (ENTERED, None)
        except BaseException:
            if self.operation_phase[0] == NOT_ENTERED:
                self.operation_phase = (ENTERED, None)
            raise

    def publish_operation(self, result: Any, error: BaseException | None) -> None:
        if self.operation_phase[0] == NOT_ENTERED:
            return
        outcome = (result, error)
        try:
            if not self.operation_outcome:
                self.operation_outcome.append(outcome)
        except BaseException as boundary:
            error = prefer(error, boundary, "operation outcome publication failed")
            outcome = (result, error)
            if self.operation_outcome:
                self.operation_outcome[0] = outcome
            else:
                self.operation_outcome.append(outcome)
        if self.operation_outcome:
            prior_result, prior_error = self.operation_outcome[0]
            if error is not None and error is not prior_error:
                prior_error = prefer(
                    prior_error, error, "later operation boundary failed")
                self.operation_outcome[0] = (prior_result, prior_error)
        self.operation_phase = (COMPLETED, self.operation_outcome[0])
