"""Bounded retry outcomes with cancellation-first error composition."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

from .error_utils import reconcile_error, safe_call


@dataclass(frozen=True)
class RetryOutcome:
    """The strongest error observed across one initial try and one retry."""

    error: BaseException | None
    cause: BaseException | None
    completed: bool

    @property
    def effective_error(self) -> BaseException | None:
        """Drop an ordinary attempt error once a retry confirms completion."""
        if self.completed and isinstance(self.error, Exception):
            return None
        return self.error

    @overload
    def against(
        self, primary: BaseException, terminal_label: str,
    ) -> tuple[BaseException, BaseException | None]: ...

    @overload
    def against(
        self, primary: None, terminal_label: str,
    ) -> tuple[BaseException | None, BaseException | None]: ...

    def against(
        self, primary: BaseException | None, terminal_label: str,
    ) -> tuple[BaseException | None, BaseException | None]:
        """Compose this retry with a primary without stale failure notes."""
        candidate = self.effective_error
        if candidate is None:
            return primary, None
        if self.completed:
            cause = primary if candidate is not primary else self.cause
            return candidate, cause
        winner, cause = reconcile_error(primary, candidate, terminal_label)
        if primary is None and winner is candidate:
            cause = self.cause
        return winner, cause


def retry_twice(
    callback: Callable[[], BaseException | None],
    retry_label: str,
    *,
    on_failure: Callable[[BaseException], Any] | None = None,
) -> RetryOutcome:
    """Attempt at most twice; a returned or raised exception is a failure."""
    error: BaseException | None = None
    cause: BaseException | None = None
    for _attempt in range(2):
        try:
            candidate = callback()
        except BaseException as exc:
            candidate = exc
        if candidate is None:
            return RetryOutcome(error, cause, True)
        if on_failure is not None:
            safe_call(on_failure, candidate)
        previous = error
        error, next_cause = reconcile_error(error, candidate, retry_label)
        if error is not previous:
            cause = next_cause
    return RetryOutcome(error, cause, False)
