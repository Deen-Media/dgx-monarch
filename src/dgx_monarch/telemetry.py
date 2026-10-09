"""Telemetry spine for `dgxm top` and the /dgxm/telemetry route.

Three small pieces, importable on the driver and on every worker:

* an event ring for structured, timestamped events (swaps, gates, verifies,
  quarantines) emitted where they happen. Workers ship their ring tail in
  status(); the telemetry route carries the driver's tail and each worker's
  side by side, and `dgxm top` merges them.
* a render-progress tracker fed per step by the driver's ProgressReceiver:
  live step, total and rate for the TUI without touching the mesh, plus the
  last finished render's summary for an idle reader.
* host_stats(), the readings of this host: the unified-pool breakdown from
  /proc/meminfo, GPU util/clock/power/temp (pynvml when present, nvidia-smi
  fallback, neither required), and the RDMA rail byte counters from sysfs.

The event ring and the progress tracker only take a lock and copy in memory.
host_stats() costs more: every call runs nvidia-smi (3 s timeout; twice when
pynvml is missing or fails), and at most every 10 s the pool reading runs it
once more and walks /proc smaps for up to 0.5 s. Through the telemetry route,
which caches its payload for 1 s, host_stats() runs at most about once a second;
with the driver down, `dgxm top` calls it directly every 1 s by default.
"""
from __future__ import annotations

import glob
import os
import subprocess
import threading
import time
from collections import deque
from typing import Any, cast

from .capacity_memory import read_meminfo, shmem_credited
from .telemetry_events import public_event_tail

# Process lifetime, unkeyed, self-evicting: the ring drops its oldest event at
# 256 and the monotonic sequence beside it never resets, so a reader can tell a
# dropped event from a quiet one.
_EVENTS: deque = deque(maxlen=256)
_EVENTS_LOCK = threading.Lock()
_EVENT_SEQ = 0  # process lifetime, monotonic: the ring's drop watermark
_LEGACY_PROGRESS_TOKEN = object()
# The render a ceremony gates starts inside the same call, right after the
# window closes. A gate that refuses that render instead (the FSDP proof path)
# leaves a claim nothing will spend, so the claim expires rather than label an
# unrelated later render a first render.
_CEREMONY_CLAIM_S = 300.0
def emit(kind: str, **fields) -> None:
    """Append one structured event to the ring; this does not log."""
    global _EVENT_SEQ
    with _EVENTS_LOCK:
        _EVENT_SEQ += 1
        _EVENTS.append({**fields, "t": time.time(), "seq": _EVENT_SEQ, "kind": kind})


def events_tail(n: int = 64) -> list[dict]:
    with _EVENTS_LOCK:
        events = list(_EVENTS)[-n:]
    return public_event_tail(events, n)


def event_snapshot(n: int = 64) -> tuple[int, list[dict]]:
    """Return one coherent event watermark/tail snapshot.

    Consumers that publish both values must not read them under separate lock
    acquisitions: an intervening emit could produce a pair that never existed
    together. ``events_tail`` and ``event_sequence`` stay for callers that need
    one value.
    """
    with _EVENTS_LOCK:
        sequence, events = _EVENT_SEQ, list(_EVENTS)[-n:]
    return sequence, public_event_tail(events, n)


def event_sequence() -> int:
    """Process-local watermark for race-free event-window capture."""
    with _EVENTS_LOCK:
        return _EVENT_SEQ


class RenderProgress:
    """Last-known render state, updated by the driver's progress channel."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Tokens and presentation fields commit in this one attribute store, so
        # async interruption can expose the whole old or new snapshot, never a
        # torn token/count/state combination.
        #
        # Two key spaces run through it. Render tokens, with render lifetime:
        # `_tokens` holds every active render and `_ceremony_tokens` the ceremony
        # members among them, a subset at every step; both are empty at the last
        # finish. Thread idents, with ceremony lifetime: `_ceremony_windows`
        # counts open gate windows per thread, opened and closed by
        # first_render's `gate_started` and `gate_finished`, and
        # `_ceremony_claim` holds the one-shot claim a closed window leaves,
        # which expires after _CEREMONY_CLAIM_S because the render meant to
        # spend it may be refused.
        self._state: dict = {
            "active": False,
            "active_renders": 0,
            "_tokens": frozenset(),
            "_legacy_count": 0,
        }

    def start(
        self, steps: int, meta: dict | None = None, *, token: object | None = None
    ) -> object:
        legacy = token is None or token is _LEGACY_PROGRESS_TOKEN
        token = _LEGACY_PROGRESS_TOKEN if legacy else token
        with self._lock:
            previous = self._state
            active_tokens = set(previous.get("_tokens", ()))
            legacy_count = int(previous.get("_legacy_count", 0))
            if legacy:
                legacy_count += 1
            else:
                active_tokens.add(token)
            now = time.time()
            # Pipeline submissions can overlap: keep the oldest start for elapsed
            # time and stay active until every paired finish, or the first
            # completion marks a still-running pipeline inactive.
            started = previous.get("started", now) if previous.get("active") else now
            ident = threading.get_ident()
            windows = previous.get("_ceremony_windows", {})
            claims = dict(previous.get("_ceremony_claim", {}))
            if previous.get("active"):
                ceremony_tokens = set(previous.get("_ceremony_tokens", ()))
                ceremony_legacy = int(previous.get("_ceremony_legacy", 0))
                bundle_ceremony = bool(previous.get("_bundle_ceremony"))
            else:
                ceremony_tokens, ceremony_legacy, bundle_ceremony = set(), 0, False
            # Membership, never inheritance: only the render whose own thread
            # holds an open gate window (a proof leg) or consumes the one-shot
            # claim (the operator's render) is a ceremony render. A concurrent
            # or pipelined render sharing the tracker must not inherit the flag.
            mine = bool(windows.get(ident))
            if not mine:
                # Proof renders run inside the open window; the operator's
                # render starts on the same thread after it closes, so exactly
                # one later start may claim it.
                claimed_at = claims.pop(ident, None)
                if claimed_at is not None and now - claimed_at < _CEREMONY_CLAIM_S:
                    mine = True
            if mine:
                bundle_ceremony = True
                if legacy:
                    ceremony_legacy += 1
                else:
                    ceremony_tokens.add(token)
            state = {
                "active": True,
                "active_renders": len(active_tokens) + legacy_count,
                "step": 0,
                "steps": int(steps),
                "started": started,
                "last_step_t": now,
                **(meta or {}),
                "ceremony": bool(ceremony_tokens) or ceremony_legacy > 0,
                "_tokens": frozenset(active_tokens),
                "_legacy_count": legacy_count,
                "_ceremony_windows": windows,
                "_ceremony_claim": claims,
                "_ceremony_tokens": frozenset(ceremony_tokens),
                "_ceremony_legacy": ceremony_legacy,
                "_bundle_ceremony": bundle_ceremony,
            }
            last = previous.get("last")
            if last is not None:
                state["last"] = last
            self._state = state
        return token

    def step(self, step: int, total: int | None = None) -> None:
        with self._lock:
            state = dict(self._state)
            tokens = set(state.get("_tokens", ()))
            legacy_count = int(state.get("_legacy_count", 0))
            if not tokens and not legacy_count:
                legacy_count = 1
            if not state.get("active"):
                state.update(
                    active=True,
                    active_renders=len(tokens) + legacy_count,
                    started=time.time(),
                )
            now = time.time()
            prev = state.get("last_step_t", now)
            state.update({
                "step": int(step),
                "steps": int(total) if total else state.get("steps"),
                "sec_per_step": round(now - prev, 3),
                "last_step_t": now,
                "_tokens": frozenset(tokens),
                "_legacy_count": legacy_count,
            })
            self._state = state

    def finish(self, token: object | None = None) -> None:
        with self._lock:
            active_tokens = set(self._state.get("_tokens", ()))
            legacy_count = int(self._state.get("_legacy_count", 0))
            ceremony_tokens = set(self._state.get("_ceremony_tokens", ()))
            ceremony_legacy = int(self._state.get("_ceremony_legacy", 0))
            started = self._state.get("started")
            if token is None or token is _LEGACY_PROGRESS_TOKEN:
                if legacy_count <= 0:
                    return
                legacy_count -= 1
                # Legacy renders are indistinguishable; keep the flag while any
                # of them could still be the ceremony one.
                ceremony_legacy = min(ceremony_legacy, legacy_count)
            else:
                if token not in active_tokens:
                    return
                active_tokens.discard(token)
                ceremony_tokens.discard(token)
            active_renders = len(active_tokens) + legacy_count
            if active_renders:
                state = dict(self._state)
                state.update(
                    active=True, active_renders=active_renders,
                    ceremony=bool(ceremony_tokens) or ceremony_legacy > 0,
                    _tokens=frozenset(active_tokens),
                    _legacy_count=legacy_count,
                    _ceremony_tokens=frozenset(ceremony_tokens),
                    _ceremony_legacy=ceremony_legacy,
                )
            else:
                now = time.time()
                last: dict | None
                if started:
                    wall_s = round(now - started, 1)
                    # The summary of the render that just ended, for an idle
                    # panel. A start with no stamp cannot describe one, so the
                    # previous summary stays.
                    last = {
                        "model": self._state.get("model"),
                        "steps": self._state.get("steps"),
                        "wall_s": wall_s,
                        "ended_at": now,
                        # Overlapping renders drain into one record, which
                        # says whether any of them was a ceremony render, not
                        # only the last to finish.
                        "ceremony": bool(self._state.get("_bundle_ceremony")),
                    }
                else:
                    wall_s = self._state.get("last_wall_s")
                    last = self._state.get("last")
                state = {
                    "active": False,
                    "active_renders": 0,
                    "last_wall_s": wall_s,
                    "_tokens": frozenset(),
                    "_legacy_count": 0,
                    "_ceremony_windows": self._state.get("_ceremony_windows", {}),
                    "_ceremony_claim": self._state.get("_ceremony_claim", {}),
                }
                if last is not None:
                    state["last"] = last
            self._state = state

    def note_ceremony(self) -> None:
        """Open an identity-gate ceremony window for the calling thread.

        Windows are counted per thread, as first_render keeps its ceremony
        thread-local: a Fleet or pipeline render on another thread is no proof
        leg. The count is a depth, so an inner close never ends an outer window.
        """
        ident = threading.get_ident()
        with self._lock:
            state = dict(self._state)
            windows = dict(state.get("_ceremony_windows", {}))
            windows[ident] = windows.get(ident, 0) + 1
            state["_ceremony_windows"] = windows
            self._state = state

    def close_ceremony(self) -> None:
        """Close one window; the outermost close leaves the claim behind."""
        ident = threading.get_ident()
        now = time.time()
        with self._lock:
            state = dict(self._state)
            windows = dict(state.get("_ceremony_windows", {}))
            depth = int(windows.get(ident, 0))
            if depth > 1:
                windows[ident] = depth - 1
            else:
                windows.pop(ident, None)
                if depth == 1:
                    claims = {
                        thread: at
                        for thread, at in state.get("_ceremony_claim", {}).items()
                        if now - at < _CEREMONY_CLAIM_S
                    }
                    claims[ident] = now
                    state["_ceremony_claim"] = claims
            state["_ceremony_windows"] = windows
            self._state = state

    def forget_ceremony(self) -> None:
        """Drop every window and claim on every thread (tests and reset)."""
        with self._lock:
            state = dict(self._state)
            state["_ceremony_windows"] = {}
            state["_ceremony_claim"] = {}
            self._state = state

    def snapshot(self) -> dict:
        with self._lock:
            return {
                key: value for key, value in self._state.items()
                if not key.startswith("_")
            }


render_progress = RenderProgress()


_RAIL_LANE_BYTES = 4  # infiniband port_{rcv,xmit}_data counters tick in 4-byte lanes


def rail_counters() -> dict[str, dict[str, int]]:
    """Cumulative RX/TX bytes per RDMA device (sysfs; cheap)."""
    out: dict[str, dict[str, int]] = {}
    for port in glob.glob("/sys/class/infiniband/*/ports/*/counters"):
        dev = port.split("/")[4]
        try:
            rx = int(open(os.path.join(port, "port_rcv_data")).read()) * _RAIL_LANE_BYTES
            tx = int(open(os.path.join(port, "port_xmit_data")).read()) * _RAIL_LANE_BYTES
        except (OSError, ValueError):
            continue
        if rx or tx:  # skip uncabled ports: both counters stay at zero on an unused rail.
            out[dev] = {"rx_bytes": rx, "tx_bytes": tx}
    return out


def _meminfo() -> dict[str, float]:
    # Append-only fields, read by name. LRU counters distinguish slab-backed
    # Shmem from staging and support the MemAvailable accounting check.
    fields = ("MemTotal", "MemFree", "MemAvailable", "Cached", "AnonPages", "Shmem", "Buffers",
              "Dirty", "Writeback", "Active(anon)", "Inactive(anon)", "Active(file)", "Inactive(file)")
    out = {}
    try:
        for line in open("/proc/meminfo"):
            key, _, rest = line.partition(":")
            if key in fields:
                out[key] = round(int(rest.split()[0]) / 2**20, 2)  # GiB
    except OSError:
        pass
    return out


_NVML = None  # process lifetime, one slot: the imported handle, bound once on first use


def _visible_gpu_selector() -> str:
    """Physical NVML/nvidia-smi selector for this CUDA-isolated actor."""
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
    return first or "0"


def _gpu_stats() -> dict:
    """Read GPU metrics through pynvml, with an nvidia-smi fallback."""
    global _NVML
    if _NVML is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            _NVML = pynvml
        except Exception:
            _NVML = False
    if _NVML:
        try:
            nvml = cast(Any, _NVML)
            selector = _visible_gpu_selector()
            if selector.startswith("GPU-"):
                h = nvml.nvmlDeviceGetHandleByUUID(selector.encode())
            else:
                h = nvml.nvmlDeviceGetHandleByIndex(int(selector))
            return {
                "util_pct": nvml.nvmlDeviceGetUtilizationRates(h).gpu,
                "clock_mhz": nvml.nvmlDeviceGetClockInfo(h, nvml.NVML_CLOCK_SM),
                "power_w": round(nvml.nvmlDeviceGetPowerUsage(h) / 1000, 1),
                "temp_c": nvml.nvmlDeviceGetTemperature(h, nvml.NVML_TEMPERATURE_GPU),
            }
        except Exception:
            pass
    try:
        row = subprocess.run(
            ["nvidia-smi", "-i", _visible_gpu_selector(),
             "--query-gpu=utilization.gpu,clocks.sm,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip().splitlines()
        util, clock, power, temp = (v.strip() for v in row[0].split(","))
        return {"util_pct": int(float(util)), "clock_mhz": int(float(clock)),
                "power_w": float(power), "temp_c": int(float(temp))}
    except Exception:
        return {}


# Process lifetime, unkeyed, one slot: a stamped reading refreshed in place
# every 10 s, a cache and not a record.
_POOL_CACHE: dict = {"t": 0.0, "gib": 0.0}
_POOL_SCAN_BUDGET_S = 0.5


def _retained_pool_gib() -> float:
    """Sum anonymous mappings of at least 4 GiB across CUDA processes.

    The CUDA/aimdo allocator can retain a model-sized staging arena after unload
    until process exit. Report it separately from other anonymous memory in the
    TUI. Cache for 10 seconds; load, unload, and restart can change the value.
    """
    now = time.time()
    if now - _POOL_CACHE["t"] < 10.0:
        return _POOL_CACHE["gib"]
    total_kb = 0
    scan_complete = True
    deadline = time.monotonic() + _POOL_SCAN_BUDGET_S
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3).stdout
        # One row per (GPU, process): a multi-GPU process appears once per
        # GPU, and scanning its smaps twice would count the arena twice.
        for pid in sorted({p.strip().rstrip(",") for p in out.split()}):
            if time.monotonic() >= deadline:
                scan_complete = False
                break
            if not pid.isdigit():
                continue
            try:
                anon = False
                for line in open(f"/proc/{pid}/smaps"):
                    if time.monotonic() >= deadline:
                        scan_complete = False
                        break
                    if line[0].isdigit() or line[0] in "abcdef":
                        anon = line.rstrip("\n").endswith(" 0") or line.split()[-1] == "0"
                    elif line.startswith("Rss:") and anon:
                        kb = int(line.split()[1])
                        if kb >= 4 * 1024 * 1024:
                            total_kb += kb
            except OSError:
                continue
            if not scan_complete:
                break
    except Exception:
        scan_complete = False
    if scan_complete:
        _POOL_CACHE.update(t=now, gib=round(total_kb / 2**20, 2))
    else:
        # A partial walk is not a measurement. Keep the last complete value and
        # back off for one cache window, so a driver-down TUI poll cannot wedge
        # behind repeated /proc/*/smaps traversals.
        _POOL_CACHE["t"] = now
    return _POOL_CACHE["gib"]


def _gpu_proc_gib() -> float:
    """Sum of per-process GPU memory (nvidia-smi). On GB10 these are
    driver-owned pages that appear in no /proc/meminfo category. Without
    this number, models render as anonymous 'other' memory in the bars."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout
        return round(sum(float(v) for v in out.split()) / 1024, 2)
    except Exception:
        return 0.0


def host_stats() -> dict:
    import socket

    stats = {
        "host": socket.gethostname(),
        "t": time.time(),
        "mem_gib": _meminfo(),
        "gpu": _gpu_stats(),
        "gpu_proc_gib": _gpu_proc_gib(),
        "pool_gib": _retained_pool_gib(),
        "rails": rail_counters(),
    }
    # False is what every price assumes, None is a reading with no LRU counters,
    # and a missing key means no reading. A raise here would lose every other
    # reading in `stats`.
    try:
        stats["shmem_credited"] = shmem_credited(read_meminfo()).credited
    except Exception:
        pass
    return stats
