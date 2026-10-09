"""Process-local state for the first-use gate, plus its two cleanup restores.

``nodes/auto_gate`` reaches every binding here through this module object, so
the denial map's copy-on-write publish and any test rebind stay visible to
every later reader. Nothing here imports ``nodes/common``.
"""
from __future__ import annotations

import os
import threading

_AUTO_GATE_ACTIVE = threading.local()  # ceremony lifetime, per thread: re-entry guard, restored in a finally
_AUTO_GATE_SESSION: dict[tuple[str, ...], str] = {}  # session lifetime, FIFO-capped; nodes/gate_token_lifetimes
_PROCESS_GATE_DENIALS: dict[tuple[str, ...], str] = {}  # session lifetime, never evicted; same home
_AUTO_GATE_SESSION_LIMIT = 128
_AUTO_GATE_WAIT_S = float(os.environ.get("DGXM_AUTO_GATE_WAIT", "1800"))
_AUTO_GATE_RUNNING: set[tuple[str, ...]] = set()  # ceremony lifetime: one in-flight claim; same home
_AUTO_GATE_LOCK = threading.Lock()
_AUTO_GATE_CONDITION = threading.Condition(_AUTO_GATE_LOCK)


def restore_auto_gate_active(previous: bool) -> BaseException | None:
    """Restore thread-local recursion state across a one-shot interruption."""
    first_error: BaseException | None = None
    first_cancel: BaseException | None = None
    for _attempt in range(2):
        try:
            _AUTO_GATE_ACTIVE.on = previous
        except BaseException as exc:
            first_error = exc if first_error is None else first_error
            if not isinstance(exc, Exception) and first_cancel is None:
                first_cancel = exc
        else:
            return first_cancel
    return first_cancel if first_cancel is not None else first_error


def release_auto_gate_claim(token: tuple[str, ...]) -> BaseException | None:
    """Atomically release a ceremony claim and wake every blocked waiter."""
    first_error: BaseException | None = None
    first_cancel: BaseException | None = None
    for _attempt in range(2):
        try:
            with _AUTO_GATE_CONDITION:
                _AUTO_GATE_RUNNING.discard(token)
                _AUTO_GATE_CONDITION.notify_all()
        except BaseException as exc:
            first_error = exc if first_error is None else first_error
            if not isinstance(exc, Exception) and first_cancel is None:
                first_cancel = exc
        else:
            return first_cancel
    return first_cancel if first_cancel is not None else first_error
