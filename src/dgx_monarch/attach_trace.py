"""Structured timing and listener-generation evidence for cluster attaches.

Every line carries a fixed prefix and attempt sequence. Values are bounded,
space-free tokens, allowing the runbook to parse and join phases without
quoting rules (docs/TROUBLESHOOTING.md #101).
"""
from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

PREFIX = "attach_trace"
UNKNOWN = "unknown"
NONE = "none"
_MAX_VALUE = 160
_SEQ_LOCK = threading.Lock()
_SEQ = 0
_ACTIVE = threading.local()
# Last successful listener generation per loop, retained across attaches
# to detect a restart since this driver's previous attach.
_SEEN_LOCK = threading.Lock()
_LAST_SEEN: dict[str, str] = {}


def next_sequence() -> int:
    """Process-lifetime attach counter; the join key for one attempt's lines."""
    global _SEQ
    with _SEQ_LOCK:
        _SEQ += 1
        return _SEQ


def token(value: Any) -> str:
    """Render one field value as a single bounded, space-free token."""
    if value is None:
        return UNKNOWN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [token(item) for item in value]
        return ",".join(items) if items else NONE
    text = "_".join(str(value).split()).replace(",", ";")
    return (text[:_MAX_VALUE] or NONE)


def line(event: str, **fields: Any) -> str:
    """Build one trace line: the prefix, the event, then the fields in order."""
    parts = [PREFIX, f"event={token(event)}"]
    parts.extend(f"{key}={token(value)}" for key, value in fields.items())
    return " ".join(parts)


def parse(text: str) -> dict[str, str] | None:
    """Read one trace line back into fields, or None when it is not one.

    The runbook counts incidents from these lines, so the writer and the
    reader live in this module and a test pins the round trip.
    """
    if not isinstance(text, str):
        return None
    start = text.find(PREFIX + " ")
    if start < 0:
        return None
    fields: dict[str, str] = {}
    for part in text[start + len(PREFIX):].split():
        key, separator, value = part.partition("=")
        if separator and key:
            fields.setdefault(key, value)
    return fields if "event" in fields else None


class AttachTrace:
    """The lines one cluster attach writes, all sharing its sequence number."""

    def __init__(self, log: Any, addresses: list[str],
                 clock: Any = time.perf_counter) -> None:
        self.log = log
        self.addresses = list(addresses)
        self.seq = next_sequence()
        self._clock = clock
        self._opened = clock()
        self._closed = False

    def since_open(self) -> float:
        return self._clock() - self._opened

    def emit(self, event: str, **fields: Any) -> str:
        """Write one line at info level and return it for the caller's tests."""
        text = line(event, seq=self.seq, **fields)
        self.log.info("%s", text)
        return text

    def opened(self, *, client_bind: str, attach_config_timeout: str,
               transport_bound: bool, released: bool, auto_heal: bool,
               init_wait_s: float) -> str:
        """Record the state inherited by this attach attempt."""
        return self.emit(
            "open", hosts=len(self.addresses), addresses=self.addresses,
            client_bind=client_bind, attach_config_timeout=attach_config_timeout,
            transport_bound=transport_bound, released_predecessor=released,
            auto_heal=auto_heal, init_wait_s=init_wait_s)

    def generation(self, address: str, fields: Mapping[str, Any],
                   *, phase: str) -> str:
        """Record one loop's listener generation and whether it changed.

        Compare against this process's last successful reading, including prior
        attaches. An unreadable generation reports both fields as unknown and
        preserves the previous reading.
        """
        current = str(fields.get("gen", UNKNOWN))
        observed = current != UNKNOWN
        with _SEEN_LOCK:
            previous = _LAST_SEEN.get(address)
            if observed:
                _LAST_SEEN[address] = current
        payload = {
            "phase": phase, "address": address,
            "changed": (None if previous is None or not observed
                        else current != previous),
            "previous": previous, **dict(fields)}
        return self.emit("loop_generation", **payload)

    def phase(self, name: str, **fields: Any) -> Any:
        """Log a phase's elapsed time on both success and failure."""
        return timed(self.emit, "phase", phase=name, **fields)

    def closed(self, outcome: str, **fields: Any) -> str:
        """The one terminal line: a second would name a stage never reached."""
        if self._closed:
            return ""
        self._closed = True
        return self.emit("close", outcome=outcome, wall_s=self.since_open(),
                         **fields)


@contextmanager
def timed(emit: Any, event: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Run a block and log its elapsed time and outcome.

    ``outcome`` is ``ok`` or the exception class name. The yielded dictionary
    lets the caller add fields known only after the block finishes.
    """
    extra: dict[str, Any] = {}
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield extra
    except BaseException as exc:
        outcome = type(exc).__name__
        raise
    finally:
        emit(event, outcome=outcome, wall_s=time.perf_counter() - started,
             **{**fields, **extra})


def half(log: Any, name: str, **fields: Any) -> Any:
    """Time the config push and initialized wait separately.

    Only one handshake phase is bounded by this repository; separate timings
    identify which phase consumed the attach budget.
    """
    def emit(event: str, **line_fields: Any) -> str:
        return note(log, event, **line_fields)

    return timed(emit, "attach_call", half=name, **fields)


@contextmanager
def terminal(trace: AttachTrace) -> Iterator[AttachTrace]:
    """Close an unfinished trace with the exception class and terminal stage.

    Exception messages are excluded from trace evidence.
    """
    try:
        yield trace
    except BaseException as exc:
        trace.closed("failed", stage="unexpected", evidence=type(exc).__name__)
        raise


@contextmanager
def active(trace: AttachTrace) -> Iterator[AttachTrace]:
    """Publish *trace* to this thread so the heal path can join its lines."""
    previous = getattr(_ACTIVE, "trace", None)
    _ACTIVE.trace = trace
    try:
        yield trace
    finally:
        _ACTIVE.trace = previous


def note(log: Any, event: str, **fields: Any) -> str:
    """Write a line from code that has no trace object, on the active sequence.

    The pre-attach heal and the post-failure restart run inside an attach but
    are reached through a seam that carries a logger and nothing else. They
    still belong to the attempt, so they read the sequence off the thread.
    """
    trace = getattr(_ACTIVE, "trace", None)
    text = line(event, seq=(0 if trace is None else trace.seq), **fields)
    log.info("%s", text)
    return text
