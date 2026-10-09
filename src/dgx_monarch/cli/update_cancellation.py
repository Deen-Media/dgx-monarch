"""Preserve cancellation when cleanup and the primary operation both fail.

A cancellation derives from BaseException but not Exception.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn, TypeVar, cast

_T = TypeVar("_T")
_MISSING = object()


def stronger_cancellation(
    current: BaseException | None, candidate: BaseException
) -> BaseException:
    """Prefer a cancellation over an ordinary error; otherwise keep the first seen."""
    if current is None or (isinstance(current, Exception) and not isinstance(candidate, Exception)):
        return candidate
    return current


def run_finalizer_pair(
    first: Callable[[], _T], second: Callable[[], _T]
) -> tuple[_T, _T]:
    """Run both finalizers, even when the first raises; then raise the stronger failure, if any."""
    values: list[object] = []
    pending: BaseException | None = None
    for finalizer in (first, second):
        try:
            values.append(finalizer())
        except BaseException as error:
            values.append(_MISSING)
            pending = stronger_cancellation(pending, error)
    if pending is not None:
        raise pending
    return cast(_T, values[0]), cast(_T, values[1])


def reraise_after_cleanup(primary: BaseException, cleanup: Callable[[], object]) -> NoReturn:
    """Run cleanup, then raise primary, unless cleanup raised a cancellation and primary is not one."""
    try:
        cleanup()
    except BaseException as error:
        raise stronger_cancellation(primary, error) from None
    raise primary
