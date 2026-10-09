"""Exception diagnostics that never raise, and cancellation-first error composition."""
from __future__ import annotations

from typing import Any, NoReturn

_NO_NOTE_DETAIL = object()


def failure_summary(exc: BaseException) -> str:
    try:
        return repr(exc)
    except BaseException:
        return f"<{type(exc).__name__}>"


def failure_text(exc: BaseException) -> str:
    """Return exception text without letting diagnostics execute control flow."""
    try:
        return str(exc)
    except BaseException:
        return f"<{type(exc).__name__}>"


def safe_call(method: Any, *args: Any) -> None:
    try:
        method(*args)
    except BaseException:
        pass


def safe_note(
    primary: BaseException,
    label: str,
    detail: Any = _NO_NOTE_DETAIL,
) -> None:
    """Attach an exact note or labelled detail without raising."""
    try:
        note = label if detail is _NO_NOTE_DETAIL else f"{label}: {detail!r}"
        primary.add_note(note)
    except BaseException:
        pass


def prefer_error(
    current: BaseException | None, candidate: BaseException, label: str,
) -> BaseException:
    """Prefer cancellation while preserving every lesser failure as a note."""
    if current is None:
        return candidate
    if isinstance(current, Exception) and not isinstance(candidate, Exception):
        safe_note(candidate, label, current)
        return candidate
    if current is not candidate:
        safe_note(current, label, candidate)
    return current


def reconcile_error(
    current: BaseException | None, candidate: BaseException, label: str,
) -> tuple[BaseException, BaseException | None]:
    strongest = prefer_error(current, candidate, label)
    cause = current if strongest is candidate else candidate
    return strongest, cause


def raise_with_distinct_cause(
    strongest: BaseException, cause: BaseException | None,
) -> NoReturn:
    """Raise an exact winner without ever self-linking reused exceptions."""
    if strongest is cause:
        raise strongest
    raise strongest from cause
