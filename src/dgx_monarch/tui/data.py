"""Poll, retain and replay telemetry samples for ``dgxm top``.

``Poller.tick`` reads driver telemetry and its reported Worker-loop state. If
the driver does not answer, read local host statistics without attaching a
mesh. A live RingStore retains one hour of samples, with a minimum of 60;
replay retains the last 3,600 samples.

Each sample contains events added since the previous one. Recordings store one
JSON sample per line: ``--record`` appends, ``s`` saves the retained history,
and ``--replay`` reads either.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import urllib.request
from collections import OrderedDict, deque
from typing import TextIO, cast

from ..config import (
    ClusterConfigError,
    find_config_path,
    format_tcp_address,
    load_cluster_config,
    tcp_endpoint,
)

RAIL_ALIASES = {"rocep1s0f0": "f0", "roceP2p1s0f0": "P2p"}
_SEEN_EVENT_LIMIT = 4096
MIN_POLL_INTERVAL_S = 0.1
MAX_POLL_INTERVAL_S = 3600.0


def _private_record_stream(path: str, *, append: bool) -> TextIO:
    """Open one owner-held 0600 recording without following a final symlink.

    An existing file is checked before truncation or append, and the opened
    descriptor's device and inode must match that check, so a file swapped in
    between them cannot receive the recording.
    """
    write_flags = os.O_WRONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    ) | getattr(os, "O_NONBLOCK", 0)
    write_flags |= os.O_APPEND if append else 0
    descriptor = -1
    before: os.stat_result | None = None
    try:
        try:
            descriptor = os.open(
                path, write_flags | os.O_CREAT | os.O_EXCL, 0o600
            )
            os.fchmod(descriptor, 0o600)
        except FileExistsError:
            before = os.lstat(path)
            _require_private_record(before)
            descriptor = os.open(path, write_flags)

        opened = os.fstat(descriptor)
        _require_private_record(opened)
        if before is not None and (before.st_dev, before.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise PermissionError("recording target changed while it was opened")
        if not append:
            os.ftruncate(descriptor, 0)
            os.lseek(descriptor, 0, os.SEEK_SET)
        mode = "a" if append else "w"
        stream = os.fdopen(descriptor, mode, encoding="utf-8", newline="\n")
        descriptor = -1
        return cast(TextIO, stream)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        # A file this call created is already owner-only. Keep it: another file
        # may now sit at this path, and unlinking the path would remove that
        # other file.
        raise


def _require_private_record(metadata: os.stat_result) -> None:
    owner = os.geteuid() if hasattr(os, "geteuid") else os.getuid()
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != owner
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise PermissionError(
            "recording target must be one owner-held regular file with mode 0600"
        )


def event_time(event: object) -> float:
    """An event's finite `t`, or 0.0 for any malformed event."""
    if not isinstance(event, dict):
        return 0.0
    value = event.get("t", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _event_fingerprint(event: dict) -> bytes:
    """A fixed-size dedupe key; an event that will not serialize hashes its type name."""
    try:
        encoded = json.dumps(
            event, sort_keys=True, separators=(",", ":"), default=str,
        ).encode("utf-8", "replace")
    except Exception:
        encoded = f"<{type(event).__name__}>".encode("ascii")
    return hashlib.sha256(encoded).digest()


def poll_interval_is_valid(interval: object) -> bool:
    """The 0.1 s floor caps the one-hour ring at 36,000 ticks.

    At 1e-320 s, `3600 / interval` is infinite, so the ring's maxlen raises.
    Textual's timer (`textual.timer.Timer._run`) fails too: its deadline
    `start + (count + 1) * interval` stays at `start`, and its catch-up count
    `int((now - start) / interval + 1)` overflows.
    """
    return (not isinstance(interval, bool) and isinstance(interval, (int, float))
            and math.isfinite(interval)
            and MIN_POLL_INTERVAL_S <= interval <= MAX_POLL_INTERVAL_S)


class RingStore:
    """Fixed-length tick history; each tick holds the events since the one before."""

    def __init__(self, seconds: int = 3600, interval: float = 1.0):
        if not poll_interval_is_valid(interval):
            raise ValueError(
                f"poll interval must be between {MIN_POLL_INTERVAL_S:g} and "
                f"{MAX_POLL_INTERVAL_S:g} seconds")
        self.interval = interval
        self.ticks: deque = deque(maxlen=max(60, int(seconds / interval)))
        self._seen_events: OrderedDict[tuple, None] = OrderedDict()

    def push(self, tick: dict) -> None:
        self.ticks.append(tick)

    def dedupe_events(self, events: list[dict]) -> list[dict]:
        fresh = []
        for e in events or []:
            if not isinstance(e, dict):
                continue
            key = (round(event_time(e), 3), _event_fingerprint(e))
            if key in self._seen_events:
                # The telemetry route resends its tail. Refresh recency so an
                # unchanged current tail cannot be evicted behind newer events
                # and then appear fresh on the following poll.
                self._seen_events.move_to_end(key)
                continue
            self._seen_events[key] = None
            fresh.append(e)
            while len(self._seen_events) > _SEEN_EVENT_LIMIT:
                self._seen_events.popitem(last=False)
        return fresh

    def at(self, index: int) -> dict | None:
        if not self.ticks:
            return None
        index = max(-len(self.ticks), min(-1, index))
        return self.ticks[index]

    def save(self, path: str) -> int:
        with _private_record_stream(path, append=False) as f:
            for tick in self.ticks:
                f.write(json.dumps(tick, default=str) + "\n")
        return len(self.ticks)

    @classmethod
    def load(cls, path: str) -> RingStore:
        ring = cls()
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    try:
                        tick = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no}: invalid recording JSON: {exc}") from exc
                    if not isinstance(tick, dict) or "t" not in tick:
                        raise ValueError(f"{path}:{line_no}: recording row must be an object with `t`")
                    ring.push(tick)
        return ring


def _configured_loops(config_path: str | None = None) -> tuple[str, ...]:
    """Worker endpoints from the same config search path used by the mesh."""
    try:
        path = find_config_path(config_path)
        if path is None:
            return ()
        return tuple(host.address for host in load_cluster_config(path).hosts)
    except (ClusterConfigError, OSError):
        if config_path:
            raise
        return ()


def _normalize_loop(value) -> str | None:
    try:
        if isinstance(value, str):
            if value.startswith("tcp://"):
                host, port = tcp_endpoint(value)
            else:
                host, raw_port = value.rsplit(":", 1)
                port = int(raw_port)
            return format_tcp_address(host.strip("[]"), port)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return format_tcp_address(str(value[0]), int(value[1]))
        if isinstance(value, dict):
            return _normalize_loop(
                value.get("address") or value.get("addr") or value.get("worker_address"))
    except (TypeError, ValueError):
        pass
    return None


class Poller:
    """One normalized tick per `tick()` call, which never raises."""

    def __init__(self, driver: str = "127.0.0.1:8191", loops=None,
                 config_path: str | None = None):
        self.driver = driver
        candidates = _configured_loops(config_path) if loops is None else loops
        self.loops = tuple(
            endpoint for value in candidates if (endpoint := _normalize_loop(value)) is not None)
        self._prev_rails: dict = {}

    def _loop_addresses(self, tele: dict | None) -> tuple[str, ...]:
        """Merge configured endpoints with any addresses exposed by telemetry."""
        values: list[object] = list(self.loops)
        if tele:
            raw_loops = tele.get("loops")
            values.extend(raw_loops if isinstance(raw_loops, list) else ())
            raw_workers = tele.get("workers")
            values.extend(
                w.get("address") or w.get("worker_address")
                for w in (raw_workers if isinstance(raw_workers, list) else ())
                if isinstance(w, dict)
            )
        out: list[str] = []
        for value in values:
            endpoint = _normalize_loop(value)
            if endpoint is not None and endpoint not in out:
                out.append(endpoint)
        return tuple(out)

    @staticmethod
    def _loop_state(address: str, tele: dict | None) -> bool | None:
        """Use Worker state reported by the driver; missing state remains unknown."""
        if not isinstance(tele, dict):
            return None
        candidates: list[object] = []
        raw_loops = tele.get("loops")
        candidates.extend(raw_loops if isinstance(raw_loops, list) else ())
        raw_workers = tele.get("workers")
        candidates.extend(raw_workers if isinstance(raw_workers, list) else ())
        for item in candidates:
            if _normalize_loop(item) != address or not isinstance(item, dict):
                continue
            for key in ("up", "healthy"):
                if isinstance(item.get(key), bool):
                    return item[key]
        return None

    def _driver_telemetry(self) -> dict | None:
        try:
            with urllib.request.urlopen(
                    f"http://{self.driver}/dgxm/telemetry", timeout=2) as r:
                return json.load(r)
        except Exception:
            return None

    def _rail_rates(self, host: str, rails: object, t: float) -> dict:
        """GB/s from cumulative byte counters and the previous sample; a first sample reads 0."""
        rates: dict[str, dict[str, float]] = {}
        if not isinstance(rails, dict):
            return rates
        for dev, c in rails.items():
            if not isinstance(dev, str) or not isinstance(c, dict):
                continue
            rx = c.get("rx_bytes")
            tx = c.get("tx_bytes")
            if (isinstance(rx, bool) or not isinstance(rx, (int, float)) or
                    not math.isfinite(rx) or isinstance(tx, bool) or
                    not isinstance(tx, (int, float)) or not math.isfinite(tx)):
                continue
            name = RAIL_ALIASES.get(dev, dev)
            prev = self._prev_rails.get((host, dev))
            if prev:
                dt = max(1e-3, t - prev["t"])
                rx_gbs = (rx - prev["rx"]) / dt / 1e9
                tx_gbs = (tx - prev["tx"]) / dt / 1e9
                rates[name] = {
                    "rx_gbs": max(0.0, rx_gbs) if math.isfinite(rx_gbs) else 0.0,
                    "tx_gbs": max(0.0, tx_gbs) if math.isfinite(tx_gbs) else 0.0,
                }
            else:
                rates[name] = {"rx_gbs": 0.0, "tx_gbs": 0.0}
            self._prev_rails[(host, dev)] = {"t": t, "rx": rx, "tx": tx}
        return rates

    @staticmethod
    def unknown_tick(marker: str = "TelemetryMalformed") -> dict:
        """Return a complete UNKNOWN sample after an unusable refresh."""
        return {
            "t": time.time(),
            "driver_up": False,
            "loops": [],
            "comfy": None,
            "readiness": None,
            "render": {"active": False},
            "ledger": {},
            "hosts": {},
            "workers": {},
            "events": [],
            "telemetry_error": marker,
        }

    def tick(self) -> dict:
        try:
            return self._tick()
        except Exception as exc:
            # Telemetry is advisory. An unusable nested shape must publish a
            # fresh UNKNOWN tick instead of leaving an earlier READY tick live.
            return self.unknown_tick(type(exc).__name__)

    def _tick(self) -> dict:
        now = time.time()
        tele = self._driver_telemetry()
        hosts: dict = {}
        events: list = []
        render: dict = {"active": False}
        ledger: dict = {}
        comfy = None
        readiness = None
        workers_meta: dict = {}

        if tele:
            comfy = tele.get("comfy")
            readiness = tele.get("readiness")
            raw_render = tele.get("render")
            render = raw_render if isinstance(raw_render, dict) else render
            raw_ledger = tele.get("ledger")
            ledger = raw_ledger if isinstance(raw_ledger, dict) else {}
            raw_events = tele.get("events")
            events += [event for event in raw_events
                       if isinstance(event, dict)] if isinstance(raw_events, list) else []
            dh = tele.get("driver_host") or {}
            if not isinstance(dh, dict):
                dh = {}
            if dh.get("host"):
                hosts[dh["host"]] = {"stats": dh, "source": "driver"}
            raw_workers = tele.get("workers")
            for w in raw_workers if isinstance(raw_workers, list) else []:
                if not isinstance(w, dict):
                    continue
                hstats = w.get("host")
                if isinstance(hstats, dict) and hstats.get("host"):
                    hosts[hstats["host"]] = {"stats": hstats, "source": "worker"}
                    workers_meta[hstats["host"]] = {
                        "vram": w.get("vram"), "memory": w.get("memory"),
                        "nccl": w.get("nccl"), "models": w.get("models"),
                    }
                w_events = w.get("events")
                events += [event for event in w_events
                           if isinstance(event, dict)] if isinstance(w_events, list) else []
        else:
            # No telemetry: read this host's stats directly.
            from ..telemetry import host_stats

            local = host_stats()
            hosts[local["host"]] = {"stats": local, "source": "local"}

        for name, entry in hosts.items():
            entry["rails"] = self._rail_rates(name, entry["stats"].get("rails"), now)

        return {
            "t": now,
            "driver_up": tele is not None,
            "loops": [
                {"addr": address, "up": self._loop_state(address, tele)}
                for address in self._loop_addresses(tele)
            ],
            "comfy": comfy,
            "readiness": readiness,
            "render": render,
            "ledger": ledger,
            "hosts": hosts,
            "workers": workers_meta,
            "events": sorted(events, key=event_time),
        }
