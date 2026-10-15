"""Ownership occupancy up-down counters: levels, sources and the off switch."""
from __future__ import annotations

import gc
import sys
import threading
import time
import types

import pytest

from dgx_monarch import monarch_diagnostics as adapter
from dgx_monarch import occupancy_metrics as occupancy
from dgx_monarch import runtime_diagnostics as diagnostics

NAMES = [name for name, _series in occupancy.COUNTERS]


class _Counter:
    def __init__(self, name, adds, fail):
        self.name, self.adds, self.fail = name, adds, fail

    def add(self, value, attributes):
        if self.name in self.fail:
            raise RuntimeError("backend down")
        self.adds.append((self.name, value, attributes))


class _Meter:
    """A fake Monarch meter: phase instruments plus up-down counters."""

    def __init__(self):
        self.adds, self.fail, self.created = [], set(), []

    def create_counter(self, name):
        return types.SimpleNamespace(add=lambda *_: None)

    def create_histogram(self, name):
        return types.SimpleNamespace(record=lambda *_: None)

    def create_up_down_counter(self, name):
        self.created.append(name)
        return _Counter(name, self.adds, self.fail)


@pytest.fixture(autouse=True)
def fresh_levels(monkeypatch):
    monkeypatch.setattr(occupancy, "_EXPORTED", {})
    monkeypatch.setattr(occupancy, "_WORKERS", [])
    monkeypatch.setattr(occupancy, "_FAILURE_LOGGED", threading.Event())
    monkeypatch.setattr(adapter, "_POLICY_STATE", {})
    adapter.configure()
    yield
    adapter.configure()


def test_counter_names_pair_with_the_prometheus_series():
    from dgx_monarch.nodes.metrics_route import _metrics_text

    assert [series for _name, series in occupancy.COUNTERS] == [
        "dgxm_" + name.removeprefix("dgxm.").replace(".", "_") for name in NAMES]
    assert NAMES == ["dgxm.return.live_handoffs", "dgxm.return.pending_read_jobs",
                     "dgxm.return.poison_owners", "dgxm.mesh.active_leases",
                     "dgxm.mesh.abandoned_samples", "dgxm.native_return.latched"]
    # The route really emits each paired series when its source has a value.
    text = _metrics_text({
        "return_ownership": {"schema": 1, "available": True, "live_handoffs": 3,
                             "tombstones": 0, "native_poison_owners": 2,
                             "pending_read_jobs": 1},
        "mesh": {"state": "ok", "active_leases": 2, "abandoned_samples": 1},
        "native_return_latch": {"schema": 1, "latched": True, "fault_count": 1}})
    emitted = {line.split("{", 1)[0].split(" ", 1)[0] for line in text.splitlines()}
    for _name, series in occupancy.COUNTERS:
        assert series in emitted, series


@pytest.mark.parametrize("name", NAMES)
def test_each_counter_adds_increments_and_decrements_as_level_changes(name):
    meter = _Meter()
    counters = occupancy.LevelCounters(meter)
    assert meter.created == NAMES
    for level in (0, 2, 3, 3, 1, 0):
        counters.sample({name: level})
    # The first export sends the whole level, zero included, so the series exists.
    assert meter.adds == [(name, 0, {}), (name, 2, {}), (name, 1, {}),
                          (name, -2, {}), (name, -1, {})]


def test_the_latch_rises_once_and_never_falls_within_a_process():
    meter = _Meter()
    counters = occupancy.LevelCounters(meter)
    for latched in (0, 1, 1, 1):
        counters.sample({"dgxm.native_return.latched": latched})
    assert [value for _name, value, _ in meter.adds] == [0, 1]


def test_an_unavailable_reading_keeps_the_level_and_a_failed_add_is_resent():
    meter = _Meter()
    counters = occupancy.LevelCounters(meter)
    counters.sample({"dgxm.return.poison_owners": 1})
    counters.sample({})  # a busy lock or an unknown state: no reading this tick
    meter.fail.add("dgxm.return.poison_owners")
    counters.sample({"dgxm.return.poison_owners": 3})
    meter.fail.clear()
    counters.sample({"dgxm.return.poison_owners": 3})
    assert meter.adds == [("dgxm.return.poison_owners", 1, {}),
                          ("dgxm.return.poison_owners", 2, {})]


def test_a_replacement_exporter_continues_from_the_exported_level():
    meter = _Meter()
    occupancy.LevelCounters(meter).sample({"dgxm.mesh.active_leases": 2})
    occupancy.LevelCounters(meter).sample({"dgxm.mesh.active_leases": 2})
    occupancy.LevelCounters(meter).sample({"dgxm.mesh.active_leases": 1})
    assert [value for _name, value, _ in meter.adds] == [2, -1]


def test_invalid_levels_are_ignored():
    meter = _Meter()
    counters = occupancy.LevelCounters(meter)
    counters.sample({"dgxm.mesh.active_leases": True, "dgxm.return.poison_owners": -1,
                     "dgxm.return.live_handoffs": 1.5, "not.a.counter": 4})
    assert meter.adds == []


class _Worker:
    """An actor stand-in with the production handoff registry."""

    def __init__(self, live):
        from dgx_monarch.actor.rdma_handoff import RDMAHandoffMixin

        RDMAHandoffMixin._init_rdma_handoffs(self)
        for token in range(live):
            handoff = {}
            self._latent_handoffs.publish(handoff, 1, capacity=8)
            assert self._latent_handoffs.owns(handoff), token


def _fake_sources(monkeypatch, *, state="busy", active=2, abandoned=1, latched=True):
    import dgx_monarch.rdma_job_registry as jobs
    import dgx_monarch.transfer as transfer

    monkeypatch.setattr(jobs, "_JOBS", [object()])
    monkeypatch.setattr(transfer, "_RDMA_POISONED_OWNERS", [object(), object()])
    monkeypatch.setitem(sys.modules, "dgx_monarch.mesh", types.ModuleType("dgx_monarch.mesh"))
    monkeypatch.setitem(sys.modules, "dgx_monarch.mesh_health", types.SimpleNamespace(
        mesh_health_snapshot=lambda: {"state": state, "active_leases": active,
                                      "abandoned_samples": abandoned}))
    monkeypatch.setitem(sys.modules, "dgx_monarch.native_return_latch",
                        types.SimpleNamespace(latched=lambda: latched))


def test_levels_come_from_the_same_sources_as_the_metrics_series(monkeypatch):
    _fake_sources(monkeypatch)
    worker = _Worker(live=3)
    assert occupancy.read_levels() == {
        "dgxm.return.pending_read_jobs": 1, "dgxm.return.poison_owners": 2,
        "dgxm.return.live_handoffs": 3, "dgxm.mesh.active_leases": 2,
        "dgxm.mesh.abandoned_samples": 1, "dgxm.native_return.latched": 1}
    del worker


def test_an_unknown_mesh_state_or_an_unloaded_source_gives_no_reading(monkeypatch):
    _fake_sources(monkeypatch, state="unknown", latched=False)
    levels = occupancy.read_levels()
    assert "dgxm.mesh.active_leases" not in levels
    assert "dgxm.return.live_handoffs" not in levels  # no actor registered here
    assert levels["dgxm.native_return.latched"] == 0
    monkeypatch.delitem(sys.modules, "dgx_monarch.mesh")
    monkeypatch.delitem(sys.modules, "dgx_monarch.native_return_latch")
    levels = occupancy.read_levels()
    assert "dgxm.mesh.abandoned_samples" not in levels
    assert "dgxm.native_return.latched" not in levels


def test_a_busy_registry_lock_skips_that_reading(monkeypatch):
    import dgx_monarch.rdma_job_registry as jobs

    lock = threading.Lock()
    monkeypatch.setattr(jobs, "_LOCK", lock)
    with lock:
        assert "dgxm.return.pending_read_jobs" not in occupancy.read_levels()


def test_a_failing_source_drops_only_its_values(monkeypatch):
    _fake_sources(monkeypatch)

    def broken():
        raise RuntimeError("unreadable")

    monkeypatch.setitem(sys.modules, "dgx_monarch.mesh_health",
                        types.SimpleNamespace(mesh_health_snapshot=broken))
    levels = occupancy.read_levels()
    assert "dgxm.mesh.active_leases" not in levels
    assert levels["dgxm.return.poison_owners"] == 2
    assert occupancy._FAILURE_LOGGED.is_set()


def test_tracked_workers_are_weak_and_bounded():
    worker = _Worker(live=1)
    occupancy.track_worker(worker)  # idempotent
    assert len(occupancy._WORKERS) == 1
    assert occupancy.read_levels()["dgxm.return.live_handoffs"] == 1
    del worker
    gc.collect()
    assert "dgxm.return.live_handoffs" not in occupancy.read_levels()
    keep = [_Worker(live=0) for _ in range(occupancy._MAX_WORKERS + 2)]
    assert len(occupancy._WORKERS) == occupancy._MAX_WORKERS
    del keep


def test_disabled_metrics_never_build_counters_or_read_sources(monkeypatch):
    meter = _Meter()
    monkeypatch.setitem(sys.modules, "monarch.actor", types.SimpleNamespace(
        get_meter=lambda: meter))
    reads = []
    monkeypatch.setattr(occupancy, "read_levels", lambda: reads.append(1) or {})
    # A state that differs from the request makes configure() run its body,
    # not the early return for unchanged settings.
    monkeypatch.setattr(adapter, "_STATE", {"metrics": False, "available": False})
    assert adapter.configure() == {"metrics": False, "available": True}
    _Worker(live=2)
    diagnostics.record_phase("collect", 0.1)
    time.sleep(0.05)
    assert meter.created == [] and meter.adds == [] and reads == []


def test_turning_metrics_off_stops_the_sampling_and_the_adds(monkeypatch):
    meter = _Meter()
    monkeypatch.setitem(sys.modules, "monarch.actor", types.SimpleNamespace(
        get_meter=lambda: meter))
    monkeypatch.setattr(occupancy, "SAMPLE_INTERVAL_S", 0.01)
    reads, sampled = [], threading.Event()

    def read():
        reads.append(1)
        sampled.set()
        return {"dgxm.return.poison_owners": len(reads)}  # a new level each sample

    monkeypatch.setattr(occupancy, "read_levels", read)
    assert adapter.configure(metrics=True)["metrics"] is True
    exporter = adapter._EXPORTER
    assert sampled.wait(5)
    assert adapter.configure()["metrics"] is False
    exporter._thread.join(5)
    assert not exporter._thread.is_alive()
    reads_at_stop, adds_at_stop = len(reads), len(meter.adds)
    time.sleep(0.05)
    assert (len(reads), len(meter.adds)) == (reads_at_stop, adds_at_stop)
    assert meter.created == NAMES  # made once, when metrics turned on


def test_enabled_metrics_sample_levels_on_the_existing_thread(monkeypatch):
    meter = _Meter()
    monkeypatch.setitem(sys.modules, "monarch.actor", types.SimpleNamespace(
        get_meter=lambda: meter))
    monkeypatch.setattr(occupancy, "SAMPLE_INTERVAL_S", 0.01)
    seen = threading.Event()
    levels = iter([{"dgxm.return.poison_owners": 1}, {"dgxm.return.poison_owners": 0}])

    def read():
        try:
            return next(levels)
        except StopIteration:
            seen.set()
            return {}

    monkeypatch.setattr(occupancy, "read_levels", read)
    assert adapter.configure(metrics=True)["available"] is True
    assert seen.wait(5)
    assert meter.adds == [("dgxm.return.poison_owners", 1, {}),
                          ("dgxm.return.poison_owners", -1, {})]
    # The existing metrics thread does the sampling; nothing else starts.
    assert [t.name for t in threading.enumerate()].count("dgxm-metrics") == 1


def test_a_meter_without_up_down_counters_keeps_phase_metrics(monkeypatch):
    records, delivered = [], threading.Event()

    class PhaseOnlyMeter:
        def create_counter(self, name):
            return types.SimpleNamespace(add=lambda value, labels: records.append(value))

        def create_histogram(self, name):
            return types.SimpleNamespace(record=lambda value, labels: delivered.set())

    debug = []
    monkeypatch.setattr("dgx_monarch.log.get_logger", lambda _name: types.SimpleNamespace(
        debug=lambda *args: debug.append(args)))
    monkeypatch.setitem(sys.modules, "monarch.actor", types.SimpleNamespace(
        get_meter=lambda: PhaseOnlyMeter()))
    monkeypatch.setattr(diagnostics, "_STORE", diagnostics.PhaseMeasurements())
    assert adapter.configure(metrics=True)["available"] is True
    diagnostics.record_phase("collect", 0.25, return_mode="message")
    assert delivered.wait(2) and records == [1]
    occupancy.level_counters(PhaseOnlyMeter())
    assert len(debug) == 1  # logged once per process, at DEBUG
