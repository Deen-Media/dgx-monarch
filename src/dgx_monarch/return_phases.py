"""The phase split of one native latent return; observation only.

A native return passes through phases each process already times in
runtime_diagnostics. On the Worker actor: the child-process preflight (the
first native pack in an actor process only) and the registration. On the
driver: the read, the drops and the ACK. The QP handshake runs inside the
first read, so it is part of ``rdma_read`` and cannot be split from it.

This leaf reads the bounded phase ring between a mark taken before one pack or
read and the end of it. It reports each phase's duration, and the time no
phase covered as ``unattributed``. When a record reads ``complete: true``, its
phases add up to its total within half a millisecond for each rounded value,
the total included. The parts of one read run at once, so a phase counts the
time any of its parts ran. The ring is process-wide: another native pack or
read in the same process during the window adds its phases here, so the split
is exact when one native return runs at a time.

Three cases make a record partial. A window the ring evicted, or phases that
cover more than the total, read ``complete: false``. A ring copy that finds the
store busy reads no phases at all, so the record reads ``complete: false`` and
the whole window lands in ``unattributed``. A single record the ring skipped on
a contended lock also lands in ``unattributed``, and nothing marks it. Nothing
here changes ownership, transport or the return choice, and no call raises.
"""
from __future__ import annotations

import math
import time
from typing import Any

from . import runtime_diagnostics

SCHEMA = 1
RESOLUTION_S = 0.001
RESULT_KEY = "return_phases"
WORKER_PHASES = ("rdma_preflight", "rdma_register", "rdma_drop")
DRIVER_PHASES = ("rdma_read", "rdma_drop", "ack")
_MAX_PHASES = 8

Mark = tuple[int | None, float]


def mark() -> Mark:
    """The ring position, then the clock, before a pack or read.

    It takes no lock, so marking cannot crowd out a phase record.
    """
    try:
        sequence = runtime_diagnostics.sequence()
    except BaseException:
        sequence = None
    valid = isinstance(sequence, int) and not isinstance(sequence, bool)
    return (sequence if valid else None), time.monotonic()


def split(started: Mark, phases: tuple[str, ...], *, always: bool = False) -> dict[str, Any] | None:
    """Durations of ``phases`` recorded since ``started``; None when none ran.

    ``always`` returns a record even when no phase was recorded, as for a
    native descriptor whose phases the ring did not keep.
    """
    cursor, began = started
    total = max(0.0, time.monotonic() - began)
    if not always and cursor is not None and runtime_diagnostics.sequence() == cursor:
        return None  # nothing recorded since the mark; skip the ring copy
    spans: dict[str, list[tuple[float, float]]] = {}
    complete = cursor is not None
    if cursor is not None:
        window = runtime_diagnostics.events_since(cursor)
        complete = window.get("available") is True and window.get("missed") == 0
        for event in window.get("events", ()):
            name = event.get("phase")
            if event.get("return_mode") == "rdma" and name in phases:
                end, length = float(event["finished_at"]), float(event["duration_s"])
                spans.setdefault(name, []).append((end - length, end))
    if not spans and not always:
        return None
    # Parts of one read run at once, so a phase is the union of its spans.
    sums = {name: _covered(intervals) for name, intervals in spans.items()}
    covered = sum(sums.values())
    if covered > total:
        complete = False
    phases_s = {name: round(sums[name], 3) for name in phases if name in sums}
    phases_s["unattributed"] = round(max(0.0, total - covered), 3)
    return {"schema": SCHEMA, "total_s": round(total, 3), "resolution_s": RESOLUTION_S,
            "phases_s": phases_s, "complete": complete}


def _covered(intervals: list[tuple[float, float]]) -> float:
    covered, reach = 0.0, None
    for start, end in sorted(intervals):
        if reach is None or start > reach:
            covered += end - start
            reach = end
        elif end > reach:
            covered += end - reach
            reach = end
    return covered


def attach(result: dict, started: Mark, descriptor: Any) -> None:
    """Add the Worker pack split to a render result row; never raises."""
    try:
        native = isinstance(descriptor, dict) and descriptor.get("kind") == "rdma"
        record = split(started, WORKER_PHASES, always=native)
        if record is not None:
            result[RESULT_KEY] = record
    except BaseException:
        pass


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and 0 <= number <= 86_400 else None


def describe(record: object) -> str:
    """``2.051s (rdma_preflight 1.912s, ..., unattributed 0.058s)``, or not recorded.

    A Worker on older source sends no record, and a row is checked before use.
    """
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return "not recorded"
    total, phases = _seconds(record.get("total_s")), record.get("phases_s")
    if total is None or not isinstance(phases, dict) or len(phases) > _MAX_PHASES:
        return "not recorded"
    parts = []
    for name, value in phases.items():
        seconds = _seconds(value)
        if not isinstance(name, str) or not name.isidentifier() or seconds is None:
            return "not recorded"
        parts.append(f"{name} {seconds:.3f}s")
    partial = "" if record.get("complete") is True else ", partial"
    return f"{total:.3f}s ({', '.join(parts)}{partial})"


def log_read(log: Any, descriptor: Any, rows: list, started: Mark) -> None:
    """One INFO line per native return: the Worker's pack and the driver's read."""
    try:
        if not isinstance(descriptor, dict) or descriptor.get("kind") != "rdma":
            return
        driver = split(started, DRIVER_PHASES, always=True)
        row = next((r for r in rows if isinstance(r, dict) and r.get("latent") is descriptor), {})
        log.info("rank %s/%s native return: worker pack %s; driver read %s",
                 row.get("rank"), row.get("host"), describe(row.get(RESULT_KEY)),
                 describe(driver))
    except BaseException:
        pass
