"""Ownership occupancy as Monarch up-down counters; opt-in metrics only.

When ``[monarch.diagnostics] metrics = true``, the metrics thread that
``monarch_diagnostics`` already runs reads the occupancy this process reports
(on the driver through /dgxm/telemetry, on an actor through ``status()``), at
most once a second, and adds the change since its last export to one up-down
counter per value, so a collector reads the level. The first reading exports
the whole level, so a counter starts at the true value.

Readers use the sources of the /dgxm/metrics series that share each name and
never import a runtime. A process that has not loaded a source exports nothing
for it, and a reading that is unavailable this time (a busy return-ownership
lock, an unknown mesh state) leaves the level where it was. The mesh reading
does not skip a busy lock: like the telemetry route, it waits for the mesh
registry lock. An ``add`` that raises leaves the exported level unchanged, so
the next sample sends the change again.

This module starts no thread, timer or service. A meter without up-down
counters, or a counter that raises, costs one DEBUG line per process; the phase
metrics and every render go on.
"""
from __future__ import annotations

import sys
import threading
import weakref
from typing import Any

from . import return_diagnostics

SAMPLE_INTERVAL_S = 1.0
# (OpenTelemetry name, the /dgxm/metrics series that reads the same source)
COUNTERS = (
    ("dgxm.return.live_handoffs", "dgxm_return_live_handoffs"),
    ("dgxm.return.pending_read_jobs", "dgxm_return_pending_read_jobs"),
    ("dgxm.return.poison_owners", "dgxm_return_poison_owners"),
    ("dgxm.mesh.active_leases", "dgxm_mesh_active_leases"),
    ("dgxm.mesh.abandoned_samples", "dgxm_mesh_abandoned_samples"),
    ("dgxm.native_return.latched", "dgxm_native_return_latched"),
)
_KNOWN_MESH_STATES = frozenset({"ok", "busy", "dirty", "poisoned", "idle"})
_MAX_WORKERS = 16
_PACKAGE = __package__ or "dgx_monarch"
_LOCK = threading.Lock()
# Process lifetime: the level each counter last exported. A replacement
# exporter continues from it, so turning metrics off and on never adds a level
# twice to the same upstream counter.
_EXPORTED: dict[str, int] = {}
_WORKERS: list[weakref.ref] = []
_FAILURE_LOGGED = threading.Event()


def _note_failure(what: str, error: BaseException) -> None:
    """One DEBUG line per process; the caller always continues."""
    if _FAILURE_LOGGED.is_set():
        return
    _FAILURE_LOGGED.set()
    try:
        from .log import get_logger

        get_logger(__name__).debug(
            "Monarch occupancy counters: %s (%s); renders and phase metrics continue",
            what, type(error).__name__)
    except BaseException:
        pass


def track_worker(worker: Any) -> None:
    """Let the sampler read this actor's handoff registry; never raises."""
    try:
        with _LOCK:
            _WORKERS[:] = [ref for ref in _WORKERS if ref() is not None]
            if (len(_WORKERS) < _MAX_WORKERS
                    and not any(ref() is worker for ref in _WORKERS)):
                _WORKERS.append(weakref.ref(worker))
    except BaseException:
        pass


def _module(name: str) -> Any:
    return sys.modules.get(f"{_PACKAGE}.{name}")


def _count(value: object) -> int | None:
    return (value if isinstance(value, int) and not isinstance(value, bool)
            and 0 <= value <= (1 << 53) - 1 else None)


def _return_levels() -> dict[str, int]:
    """Pending read jobs (read leases) and poison owners, skipped when busy."""
    snapshot = return_diagnostics.driver_snapshot()
    levels = {}
    for field, name in (("pending_read_jobs", "dgxm.return.pending_read_jobs"),
                        ("native_poison_owners", "dgxm.return.poison_owners")):
        count = _count(snapshot.get(field))
        if count is not None:
            levels[name] = count
    return levels


def _handoff_levels() -> dict[str, int]:
    with _LOCK:
        workers = [worker for worker in (ref() for ref in _WORKERS) if worker is not None]
    total = None
    for worker in workers:
        count = _count(return_diagnostics.actor_snapshot(worker).get("live_handoffs"))
        if count is not None:
            total = (total or 0) + count
    return {} if total is None else {"dgxm.return.live_handoffs": total}


def _mesh_levels() -> dict[str, int]:
    """The driver's sample leases, as the /dgxm/metrics mesh series read them."""
    health = _module("mesh_health")
    if _module("mesh") is None or health is None:
        return {}
    block = health.mesh_health_snapshot()
    if not isinstance(block, dict) or block.get("state") not in _KNOWN_MESH_STATES:
        return {}
    levels = {}
    for field, name in (("active_leases", "dgxm.mesh.active_leases"),
                        ("abandoned_samples", "dgxm.mesh.abandoned_samples")):
        count = _count(block.get(field))
        if count is not None:
            levels[name] = count
    return levels


def _latch_levels() -> dict[str, int]:
    latch = _module("native_return_latch")
    return {} if latch is None else {"dgxm.native_return.latched": int(bool(latch.latched()))}


def read_levels() -> dict[str, int]:
    """Each source is read on its own; one failing source drops only its values."""
    levels: dict[str, int] = {}
    for reader in (_return_levels, _handoff_levels, _mesh_levels, _latch_levels):
        try:
            levels.update(reader())
        except Exception as error:
            _note_failure(f"{reader.__name__} unreadable", error)
    return levels


class LevelCounters:
    """One up-down counter per occupancy value; adds only level changes."""

    def __init__(self, meter: Any) -> None:
        self._counters = {name: meter.create_up_down_counter(name) for name, _series in COUNTERS}

    def sample(self, levels: dict[str, int]) -> None:
        for name, counter in self._counters.items():
            level = _count(levels.get(name))
            if level is None:
                continue
            with _LOCK:
                previous = _EXPORTED.get(name)
            if previous == level:
                continue
            try:
                # The first export sends the whole level, zero included, so the
                # series exists from the start.
                counter.add(level - (previous or 0), {})
            except Exception as error:
                _note_failure(f"{name} add failed", error)
                continue
            with _LOCK:
                _EXPORTED[name] = level

    def sample_safely(self) -> None:
        try:
            self.sample(read_levels())
        except BaseException as error:
            _note_failure("sample failed", error)


def level_counters(meter: Any) -> LevelCounters | None:
    """The counters, or None when this meter cannot make them; never raises."""
    try:
        return LevelCounters(meter)
    except BaseException as error:
        _note_failure("up-down counters unavailable", error)
        return None
