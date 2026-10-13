"""Finite, sanitized captures from existing telemetry, with private exports.

Collectors read snapshots only. They never attach, create an admin listener,
stop a Worker, or infer settlement from a trace. Missing observations remain
coverage gaps. Artifacts use the Chrome/Perfetto complete-event JSON format.
"""
from __future__ import annotations

import errno
import json
import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .runtime_diagnostics import MAX_RECENT, valid_labels

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_NONCE = re.compile(r"[0-9a-f]{32}\Z")
_VERSION = re.compile(r"[A-Za-z0-9.+_-]{1,64}\Z")
MAX_WORKERS = 256
_ADMIN_REASON = "torchmonarch_0_7_admin_handle_has_no_stop_or_partial_start_recovery_api"
_FETCH_WINDOW_S, _POLL_INTERVAL_S = 2.0, 0.5
# Fixed refusal sentences carry no caller data, so the CLI may print them verbatim.
_DURATION = "capture duration must be between 0.1 and 300 seconds"
_EVENTS = "capture event limit must be between 1 and 20000"
_BYTES = "capture byte limit must be between 65536 and 8388608"
_DESTINATION = "capture output requires an existing directory without symbolic links"
_METADATA = "capture metadata exceeds the byte limit; no artifact written"
REFUSALS = frozenset({_DURATION, _EVENTS, _BYTES, _DESTINATION, _METADATA})


def capabilities() -> dict[str, Any]:
    """Exact-version support boundary; this function starts nothing."""
    return {"schema": 1, "torchmonarch": "0.7.0",
            "phase_capture": {"available": True, "scope": "existing_telemetry"},
            "topology_summary": {"available": True, "scope": "reported_worker_ranks"},
            **{name: {"available": False, "reason": _ADMIN_REASON}
               for name in ("attached_admin", "distributed_sql", "owned_process_stacks")},
            "source": "https://github.com/meta-pytorch/monarch/blob/v0.7.0/python/monarch/_rust_bindings/monarch_hyperactor/host_mesh.pyi#L179"}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and 0 <= number <= (1 << 53) - 1 else None


def _identity(block: dict) -> dict[str, str]:
    result = {}
    for key in ("source_manifest_sha256", "transport_sha256"):
        value = block.get(key)
        if isinstance(value, str) and _HASH.fullmatch(value):
            result[key] = value
    for key in ("torchmonarch", "torch", "dgx_monarch"):
        value = block.get(key)
        if isinstance(value, str) and _VERSION.fullmatch(value):
            result[key] = value
    return result


class Capture:
    """One bounded capture; ingestion requires no callbacks or background jobs."""

    def __init__(self, *, duration_s: float = 30, max_events: int = 4096,
                 max_bytes: int = 1_048_576) -> None:
        duration = _number(duration_s)
        if duration is None or not 0.1 <= duration <= 300:
            raise ValueError(_DURATION)
        if type(max_events) is not int or not 1 <= max_events <= 20_000:
            raise ValueError(_EVENTS)
        if type(max_bytes) is not int or not 65_536 <= max_bytes <= 8_388_608:
            raise ValueError(_BYTES)
        self.started_at = time.time()
        self.deadline = time.monotonic() + duration
        self.duration_s, self.max_events, self.max_bytes = duration, max_events, max_bytes
        self._events: list[dict] = []
        self._event_bytes = 0
        self._cursors: dict[str, int] = {}
        self._identities: dict[str, dict] = {}
        self._topology: list[dict] = []
        self._expected: set[str] = set()
        self._gaps: dict[str, int] = {}
        self._samples = 0
        self._stopped: str | None = None

    @property
    def remaining_s(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def done(self) -> bool:
        return self._stopped is not None or self.remaining_s <= 0

    def gap(self, reason: str) -> None:
        if reason in {"fetch_failed", "stale_snapshot", "collector_missing", "ring_evicted",
                      "invalid_event", "identity_changed", "interrupted", "invalid_snapshot",
                      "stale_workers", "stale_worker"}:
            self._gaps[reason] = self._gaps.get(reason, 0) + 1

    def _stream(self, name: str, block: object, identity: dict) -> None:
        if not isinstance(block, dict) or block.get("available") is not True or block.get("schema") != 1:
            self.gap("collector_missing")
            return
        process_nonce = block.get("process_nonce")
        if isinstance(process_nonce, str) and _NONCE.fullmatch(process_nonce):
            identity = {**identity, "process_nonce": process_nonce}
        previous_identity = self._identities.get(name)
        if previous_identity is not None and previous_identity != identity:
            self.gap("identity_changed")
            self._cursors.pop(name, None)
        self._identities[name] = identity
        rows = block.get("recent")
        if not isinstance(rows, list):
            self.gap("invalid_snapshot")
            return
        cursor = self._cursors.get(name)
        sequence = block.get("sequence")
        if type(sequence) is not int or not 0 <= sequence <= (1 << 53) - 1:
            self.gap("invalid_snapshot")
            return
        if cursor is not None and sequence < cursor:
            self.gap("identity_changed")
            cursor = None
        for row in rows[-MAX_RECENT:]:
            if not isinstance(row, dict):
                self.gap("invalid_event")
                continue
            seq, finished = row.get("sequence"), _number(row.get("finished_at"))
            duration = _number(row.get("duration_s"))
            if (type(seq) is not int or not 0 < seq <= sequence or finished is None or duration is None
                    or not valid_labels(row.get("phase"), row.get("outcome"), row.get("return_mode"))):
                self.gap("invalid_event")
                continue
            if cursor is not None and seq <= cursor:
                continue
            if cursor is not None and seq > cursor + 1:
                self.gap("ring_evicted")
            cursor = seq
            # First observation must not relabel old timings as a new capture.
            if finished < self.started_at:
                continue
            event = {"name": row["phase"], "cat": "dgxm", "ph": "X",
                     "ts": (finished - duration) * 1_000_000, "dur": duration * 1_000_000,
                     "pid": name, "tid": "phases", "args": {
                         "outcome": row["outcome"], "return_mode": row["return_mode"]}}
            cost = len(json.dumps(event, separators=(",", ":")).encode()) + 1
            # Leave room for fixed metadata and at most 257 participant identities.
            if len(self._events) >= self.max_events:
                self._stopped = "event_limit"
                break
            if self._event_bytes + cost > self.max_bytes // 2:
                self._stopped = "byte_limit"
                break
            self._events.append(event)
            self._event_bytes += cost
        self._cursors[name] = sequence

    def ingest(self, payload: object) -> None:
        """Accept one already-fetched snapshot and retain only fixed fields."""
        if self.done:
            return
        if not isinstance(payload, dict):
            self.gap("invalid_snapshot")
            return
        self._samples += 1
        if payload.get("telemetry_error"):
            self.gap("stale_snapshot")
            return
        observed = {"driver"}
        identity = payload.get("diagnostic_identity", {})
        self._stream("driver", payload.get("diagnostics"), _identity(identity if isinstance(identity, dict) else {}))
        workers = payload.get("workers", [])
        poll = payload.get("status_poll")
        poll_stale = "status_poll" in payload and (
            not isinstance(poll, dict) or poll.get("fresh") is not True)
        if poll_stale:
            self.gap("stale_workers")
        topology = []
        for index, worker in enumerate(workers[:MAX_WORKERS] if isinstance(workers, list) else []):
            name = f"worker-{index}"
            observed.add(name)
            if not isinstance(worker, dict):
                self.gap("collector_missing")
                continue
            worker_poll = worker.get("status_poll")
            if (poll_stale or "status_error" in worker or worker.get("telemetry_error")
                    or ("status_poll" in worker and (not isinstance(worker_poll, dict)
                        or worker_poll.get("fresh") is not True))):
                self.gap("stale_worker")
                continue
            identity = worker.get("diagnostic_identity", worker)
            self._stream(name, worker.get("diagnostics"), _identity(identity if isinstance(identity, dict) else {}))
            row: dict[str, Any] = {"participant": name}
            for key in ("rank", "world"):
                value = worker.get(key)
                if type(value) is int and 0 <= value <= 4096:
                    row[key] = value
            for key in ("nccl", "setup_cleanup_failed"):
                if type(worker.get(key)) is bool:
                    row[key] = worker[key]
            topology.append(row)
        if self._expected - observed:
            self.gap("collector_missing")
        self._expected |= observed
        self._topology = topology

    def report(self) -> dict[str, Any]:
        missing_identity = any(not all(key in identity for key in (
            "source_manifest_sha256", "transport_sha256", "torchmonarch", "dgx_monarch", "process_nonce"))
            for identity in self._identities.values()) or not self._identities
        return {"schema": 1, "traceEvents": [{**event, "args": dict(event["args"])} for event in self._events],
                "displayTimeUnit": "ms", "capture": {
                    "started_at": self.started_at, "requested_duration_s": self.duration_s,
                    "stop_reason": self._stopped or ("deadline" if self.done else "snapshot"),
                    "samples": self._samples, "max_events": self.max_events,
                    "max_bytes": self.max_bytes, "coverage": "best_effort_telemetry_snapshots",
                    "clock_scope": "per-process durations; cross-host ordering depends on clock synchronization",
                    "gaps": dict(self._gaps), "identity_coverage": "partial" if missing_identity else "reported",
                    "partial": bool(self._gaps) or self._stopped is not None or missing_identity},
                "identities": {name: dict(value) for name, value in self._identities.items()},
                "topology": [dict(row) for row in self._topology],
                "capabilities": capabilities()}

    def export(self, destination: str | Path) -> Path:
        """Create one private artifact; refuse overwrites and symbolic links."""
        output = validate_destination(destination)
        report = self.report()
        encoded = json.dumps(report, allow_nan=False, separators=(",", ":")).encode()
        while len(encoded) > self.max_bytes:
            self._stopped = "byte_limit"
            report["capture"].update(stop_reason="byte_limit", partial=True)
            if report["traceEvents"]:
                report["traceEvents"].pop()
            elif report["topology"]:
                report["topology"].pop()
            elif report["identities"]:
                report["identities"].popitem()
            else:
                raise ValueError(_METADATA)
            encoded = json.dumps(report, allow_nan=False, separators=(",", ":")).encode()
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            fd = os.open(output.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=directory)
            identity = os.fstat(fd)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
            except BaseException:
                try:
                    current = os.stat(output.name, dir_fd=directory, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                        os.unlink(output.name, dir_fd=directory)
                except OSError:
                    pass
                raise
        finally:
            os.close(directory)
        return output


def validate_destination(destination: str | Path) -> Path:
    """Refuse an unusable artifact path before any capture time is spent.

    ``export`` repeats this check and still creates the file with O_EXCL and
    O_NOFOLLOW, so a path that appears during the capture is refused too.
    """
    output = Path(destination).expanduser().absolute()
    if output.parent.resolve() != output.parent or not output.parent.is_dir():
        raise ValueError(_DESTINATION)
    if os.path.lexists(output):
        raise FileExistsError(errno.EEXIST, "capture output already exists")
    return output


def collect(fetch: Callable[[float], object], *, duration_s: float = 30,
            max_events: int = 4096, max_bytes: int = 1_048_576) -> Capture:
    """Poll an existing endpoint; ``fetch`` must honor its supplied deadline.

    No asynchronous collectors are created, so partial startup or fetch loss
    leaves no background process, listener, or actor to clean up. A final
    window shorter than the slowest fetch so far is slept, not polled: that
    fetch could only lose its race with the deadline. The unpolled tail is then
    the same whether the loop ends on a fetch or on a sleep, so ``partial``
    reports coverage rather than where the deadline happened to fall. The
    slowest fetch, not the last, covers the route's alternating cached reads.
    """
    capture = Capture(duration_s=duration_s, max_events=max_events, max_bytes=max_bytes)
    slowest = 0.0
    try:
        while not capture.done:
            window = min(_FETCH_WINDOW_S, capture.remaining_s)
            if window < min(slowest, _FETCH_WINDOW_S):
                time.sleep(window)
                continue
            started = time.monotonic()
            try:
                capture.ingest(fetch(window))
            except Exception:
                capture.gap("fetch_failed")
            slowest = max(slowest, time.monotonic() - started)
            if not capture.done:
                time.sleep(min(_POLL_INTERVAL_S, capture.remaining_s))
    except KeyboardInterrupt:
        capture.gap("interrupted")
        capture._stopped = "interrupted"
    return capture
