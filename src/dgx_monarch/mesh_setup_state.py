"""Typed lifecycle state for same-handle process-group transitions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from typing import Any


class SetupCleanupOutcome(StrEnum):
    """Non-success outcomes for an all-rank setup cleanup."""

    IN_PROGRESS = "in_progress"
    TIMEOUT_UNKNOWN = "timeout_unknown"
    REPORTED_FAILURE = "reported_failure"
    INTERRUPTED = "interrupted"
    INVALID_RESULT = "invalid_result"


@dataclass(frozen=True)
class SetupCleanupState:
    generation: int
    outcome: SetupCleanupOutcome
    phase: str
    budget_s: float


@dataclass(frozen=True)
class SetupToken:
    """Driver authorization for one exact READY setup generation."""

    generation: int
    key: tuple
    worker_args_key: tuple


class TopologyTransitionError(RuntimeError):
    """A setup generation is dirty and cannot authorize more GPU work."""

    def __init__(self, state: SetupCleanupState, operation: str = "continue") -> None:
        self.state = state
        timing = (
            f"is in progress under a {state.budget_s:g}s budget"
            if state.outcome is SetupCleanupOutcome.IN_PROGRESS
            else (
                f"timed out after its {state.budget_s:g}s budget"
                if state.outcome is SetupCleanupOutcome.TIMEOUT_UNKNOWN
                else f"ended with {state.outcome.value} under a "
                f"{state.budget_s:g}s budget"
            )
        )
        super().__init__(
            f"{state.phase} for setup generation {state.generation} {timing}; "
            f"cannot {operation} on this Attached mesh. Reset the Attached mesh "
            "before retrying."
        )


class StaleSetupGenerationError(RuntimeError):
    """A caller tried to dispatch against a replaced READY generation."""

    def __init__(self, expected: SetupToken, actual_generation: int) -> None:
        self.expected = expected
        self.actual_generation = actual_generation
        super().__init__(
            f"setup generation/policy {expected.generation} is no longer current "
            f"(actual generation {actual_generation}) before dispatch; rerun setup "
            "for this request"
        )


class SetupVerdictError(RuntimeError):
    """Driver-observed unusable setup state with no call in flight.

    Unlike a dispatch failure with an unknown outcome, this verdict comes from
    local handle state and can be acted on as definite negative evidence.
    """


class InvalidCleanupResult(ValueError):
    """The all-rank cleanup acknowledgement is missing or malformed."""


class LifecycleBusyError(RuntimeError):
    """A destructive lifecycle request could not linearize on the handle."""


class SampleResultBusyError(LifecycleBusyError):
    """Live sample/result ownership blocks destructive lifecycle work."""


@dataclass(frozen=True)
class SetupTransitionBudget:
    """Independent phase caps; no caller-controlled teardown authority."""

    # Large-resident measurement and final-head transition evidence:
    # docs/VALIDATION.md, "Client lifecycle on worker loops".
    group_cleanup_s: float = 600.0
    rank_setup_s: float = 600.0
    rollback_cleanup_s: float = 600.0

    def __post_init__(self) -> None:
        for name in ("group_cleanup_s", "rank_setup_s", "rollback_cleanup_s"):
            object.__setattr__(
                self, name, _positive_timeout(getattr(self, name), name)
            )

    @classmethod
    def for_setup_timeout(cls, timeout_s: object) -> SetupTransitionBudget:
        return cls(rank_setup_s=_positive_timeout(timeout_s, "setup timeout"))


def _positive_timeout(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be a finite positive number")
    return result


def cleanup_in_progress(
    generation: int, phase: str, budget_s: float
) -> SetupCleanupState:
    return SetupCleanupState(
        generation, SetupCleanupOutcome.IN_PROGRESS, phase, budget_s
    )


def cleanup_failure(
    generation: int, phase: str, budget_s: float, exc: BaseException
) -> SetupCleanupState:
    if isinstance(exc, TimeoutError):
        outcome = SetupCleanupOutcome.TIMEOUT_UNKNOWN
    elif isinstance(exc, InvalidCleanupResult):
        outcome = SetupCleanupOutcome.INVALID_RESULT
    elif isinstance(exc, Exception):
        outcome = SetupCleanupOutcome.REPORTED_FAILURE
    else:
        outcome = SetupCleanupOutcome.INTERRUPTED
    return SetupCleanupState(generation, outcome, phase, budget_s)


def validate_cleanup_result(value_mesh: Any, world: int) -> list[dict]:
    """Require one explicit UNSETUP acknowledgement from every actor."""
    try:
        values = [value for _point, value in value_mesh.items()]
    except Exception as exc:
        raise InvalidCleanupResult(
            "group cleanup returned an invalid value mesh"
        ) from exc
    if len(values) != world:
        raise InvalidCleanupResult(
            f"group cleanup returned {len(values)} acknowledgements for world {world}"
        )
    if any(
        not isinstance(value, dict)
        or value.get("cleanup_state") != "UNSETUP"
        or not isinstance(value.get("torn_down"), bool)
        for value in values
    ):
        raise InvalidCleanupResult(
            "group cleanup returned an invalid actor acknowledgement"
        )
    return values
