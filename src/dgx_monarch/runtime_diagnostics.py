"""Bounded, advisory phase measurements shared by driver and actor processes.

This standard-library leaf never polls workers or owns runtime resources. Its
fixed key space and recent ring live for the process lifetime. Counters are
measurements, not evidence that a lease, native read, or teardown has settled.
"""
from __future__ import annotations

import math
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

PHASES = frozenset({
    "attach", "setup", "queue_wait", "compute", "collect", "rdma_register",
    "rdma_read", "rdma_drop", "ack", "recycle", "spawn", "readiness", "cancel",
})
OUTCOMES = frozenset({"ok", "error", "timeout", "cancelled", "unknown"})
RETURN_MODES = frozenset({"none", "message", "rdma"})
BUCKET_BOUNDS_S = (0.001, 0.01, 0.1, 1.0, 10.0, 60.0, 300.0, 1800.0)
MAX_RECENT = 128
MAX_SERIES = len(PHASES) * len(OUTCOMES) * len(RETURN_MODES)
_MAX_NUMBER = (1 << 53) - 1


def valid_labels(phase_name: object, outcome: object, return_mode: object) -> bool:
    """Accept only bounded public dimensions; never stringify caller objects."""
    return (isinstance(phase_name, str) and phase_name in PHASES
            and isinstance(outcome, str) and outcome in OUTCOMES
            and isinstance(return_mode, str) and return_mode in RETURN_MODES)


def _duration(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) and 0 <= number <= _MAX_NUMBER else None


class PhaseMeasurements:
    """Fixed-size aggregates and a self-evicting ring, with no blocking writer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._series: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT)
        self._sequence = 0
        self._process_nonce = uuid.uuid4().hex

    def record(self, phase_name: str, duration_s: float, *, outcome: str = "ok",
               return_mode: str = "none") -> dict[str, Any] | None:
        if not valid_labels(phase_name, outcome, return_mode):
            return None
        duration = _duration(duration_s)
        if duration is None or not self._lock.acquire(blocking=False):
            return None
        try:
            key = (phase_name, outcome, return_mode)
            previous = self._series.get(key)
            buckets = list(previous["buckets"]) if previous else [0] * (len(BUCKET_BOUNDS_S) + 1)
            index = next((i for i, bound in enumerate(BUCKET_BOUNDS_S) if duration <= bound),
                         len(BUCKET_BOUNDS_S))
            buckets[index] = min(_MAX_NUMBER, buckets[index] + 1)
            row = {
                "phase": phase_name, "outcome": outcome, "return_mode": return_mode,
                "count": min(_MAX_NUMBER, previous["count"] + 1) if previous else 1,
                "sum_s": min(_MAX_NUMBER, previous["sum_s"] + duration) if previous else duration,
                "max_s": max(previous["max_s"], duration) if previous else duration,
                "buckets": buckets,
            }
            # Publish complete aggregates; a diagnostic interruption cannot
            # expose a half-updated count/bucket pair to the next snapshot.
            self._series[key] = row
            self._sequence += 1
            event = {"sequence": self._sequence, "phase": phase_name,
                     "outcome": outcome, "return_mode": return_mode,
                     "duration_s": duration, "finished_at": time.time()}
            self._recent.append(event)
            return dict(event)
        finally:
            self._lock.release()

    def snapshot(self) -> dict[str, Any]:
        stamp = time.time()
        if not self._lock.acquire(blocking=False):
            return {"schema": 1, "scope": "process", "observed_at": stamp,
                    "available": False, "phases": [], "recent": []}
        try:
            return {
                "schema": 1, "scope": "process", "observed_at": stamp,
                "available": True, "sequence": self._sequence,
                "process_nonce": self._process_nonce,
                "bucket_bounds_s": list(BUCKET_BOUNDS_S),
                "phases": [{**row, "buckets": list(row["buckets"])}
                           for _, row in sorted(self._series.items())],
                "recent": [dict(row) for row in self._recent],
            }
        finally:
            self._lock.release()


_STORE = PhaseMeasurements()
# Optional sinks are installed explicitly by monarch_diagnostics.configure.
# They must not retain application values or perform synchronous network IO.
_OBSERVER: Callable[[dict[str, Any]], None] | None = None
_SPAN_FACTORY: Callable[[str], Any] | None = None


def configure_export(observer: Callable[[dict[str, Any]], None] | None = None,
                     span_factory: Callable[[str], Any] | None = None) -> None:
    """Install optional process-local diagnostic consumers, never a backend."""
    global _OBSERVER, _SPAN_FACTORY
    _OBSERVER, _SPAN_FACTORY = observer, span_factory


def record_phase(phase_name: str, duration_s: float, *, outcome: str = "ok",
                 return_mode: str = "none") -> None:
    """Record advisory timing without changing the operation's outcome."""
    try:
        event = _STORE.record(phase_name, duration_s, outcome=outcome, return_mode=return_mode)
        observer = _OBSERVER
        if event is not None and observer is not None:
            observer(event)
    except BaseException:
        # Includes interruptions in diagnostic code. The caller's operation
        # and any exception already escaping it retain their original result.
        pass


def snapshot() -> dict[str, Any]:
    """Return only plain JSON data; unavailable observations are never idle."""
    try:
        return _STORE.snapshot()
    except BaseException:
        return {"schema": 1, "scope": "process", "available": False,
                "phases": [], "recent": []}


def events_since(sequence: int = 0, *, limit: int = MAX_RECENT) -> dict[str, Any]:
    """Poll bounded local spans without callbacks on the application's thread.

    ``missed`` counts evicted accepted records, not operations skipped on lock
    contention. A capture consumer must preserve this explicit coverage limit.
    """
    current = snapshot()
    if not current.get("available"):
        return {"available": False, "events": [], "missed": None}
    cursor = max(0, sequence) if isinstance(sequence, int) and not isinstance(sequence, bool) else 0
    bound = max(0, min(limit, MAX_RECENT)) if isinstance(limit, int) else MAX_RECENT
    recent = current["recent"]
    oldest = recent[0]["sequence"] if recent else current["sequence"] + 1
    events = [row for row in recent if row["sequence"] > cursor][:bound]
    return {"available": True, "sequence": current["sequence"],
            "next_sequence": events[-1]["sequence"] if events else cursor,
            "events": events,
            "missed": max(0, oldest - cursor - 1),
            "coverage": "best_effort_process_local"}


@contextmanager
def phase(phase_name: str, *, return_mode: str = "none") -> Iterator[None]:
    """Measure a scope, preserving success, failure and cancellation exactly."""
    started: float | None = None
    span = None
    outcome = "ok"
    try:
        if valid_labels(phase_name, outcome, return_mode):
            started = time.monotonic()
            factory = _SPAN_FACTORY
            if factory is not None:
                candidate = factory(phase_name)
                candidate.__enter__()
                span = candidate
    except BaseException:
        pass
    try:
        yield
    except BaseException as error:
        if isinstance(error, TimeoutError):
            outcome = "timeout"
        elif isinstance(error, (KeyboardInterrupt, SystemExit)) or type(error).__name__ == "CancelledError":
            outcome = "cancelled"
        else:
            outcome = "error"
        raise
    finally:
        try:
            if span is not None:
                # Traces do not receive exception values or suppress errors.
                span.__exit__(None, None, None)
        except BaseException:
            pass
        try:
            if started is not None:
                record_phase(phase_name, max(0.0, time.monotonic() - started),
                             outcome=outcome, return_mode=return_mode)
        except BaseException:
            pass


phase_scope = phase
record = record_phase
