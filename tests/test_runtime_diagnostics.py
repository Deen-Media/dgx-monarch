"""Bounded timing evidence must never change application outcomes."""
from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from dgx_monarch import runtime_diagnostics as diagnostics


@pytest.fixture
def measurements(monkeypatch):
    store = diagnostics.PhaseMeasurements()
    monkeypatch.setattr(diagnostics, "_STORE", store)
    monkeypatch.setattr(diagnostics, "_OBSERVER", None)
    return store


def test_histograms_have_correct_disjoint_buckets_and_detached_snapshots(measurements):
    for duration in (0.001, 0.01, 0.011, 2000):
        diagnostics.record_phase("collect", duration, return_mode="message")
    result = diagnostics.snapshot()
    row = result["phases"][0]
    assert row["count"] == 4
    assert row["sum_s"] == pytest.approx(2000.022)
    assert row["buckets"] == [1, 1, 1, 0, 0, 0, 0, 0, 1]
    assert row["max_s"] == 2000
    row["buckets"][0] = 800
    result["recent"][0]["phase"] = "changed"
    assert diagnostics.snapshot()["phases"][0]["buckets"][0] == 1
    assert diagnostics.snapshot()["recent"][0]["phase"] == "collect"
    json.dumps(diagnostics.snapshot(), allow_nan=False)


@pytest.mark.parametrize("duration", [float("inf"), float("nan"), -1, True, 10**1000, "2"])
def test_invalid_durations_are_ignored(measurements, duration):
    diagnostics.record_phase("compute", duration)
    assert diagnostics.snapshot()["phases"] == []


def test_fixed_cardinality_and_recent_ring_bound(measurements):
    for index in range(1000):
        diagnostics.record_phase(f"prompt-{index}", 1)
        diagnostics.record_phase("compute", 1, outcome=f"secret-{index}")
        diagnostics.record_phase("compute", 1, return_mode=f"address-{index}")
        diagnostics.record_phase("compute", 1)
    result = diagnostics.snapshot()
    assert len(result["phases"]) == 1
    assert result["phases"][0]["count"] == 1000
    assert len(result["recent"]) == diagnostics.MAX_RECENT
    assert result["recent"][0]["sequence"] == 1000 - diagnostics.MAX_RECENT + 1
    assert "secret" not in json.dumps(result)


def test_contention_drops_observation_without_waiting(measurements):
    with measurements._lock:
        diagnostics.record_phase("ack", 1)
        assert diagnostics.snapshot()["available"] is False
    assert diagnostics.snapshot()["phases"] == []


def test_concurrent_records_remain_coherent(measurements):
    barrier = threading.Barrier(4)

    def writer():
        barrier.wait()
        for _ in range(500):
            diagnostics.record_phase("compute", 0.1)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: writer(), range(4)))
    row = diagnostics.snapshot()["phases"][0]
    # Observations may be dropped on contention, but never invent completion.
    assert 1 <= row["count"] <= 2000
    assert sum(row["buckets"]) == row["count"]
    assert row["sum_s"] == pytest.approx(row["count"] * 0.1)


@pytest.mark.parametrize(("error", "outcome"), [
    (RuntimeError("private payload"), "error"),
    (TimeoutError("private timeout"), "timeout"),
    (asyncio.CancelledError("private cancellation"), "cancelled"),
    (KeyboardInterrupt(), "cancelled"),
])
def test_phase_preserves_original_exception_and_hides_payload(measurements, error, outcome):
    with pytest.raises(type(error)) as raised:
        with diagnostics.phase("rdma_read", return_mode="rdma"):
            raise error
    assert raised.value is error
    result = diagnostics.snapshot()
    assert result["phases"][0]["outcome"] == outcome
    assert "private" not in json.dumps(result)


def test_measurement_failure_does_not_replace_operation(monkeypatch, measurements):
    def broken_record(*_args, **_kwargs):
        raise SystemExit("diagnostic failure")

    monkeypatch.setattr(diagnostics, "record_phase", broken_record)
    original = RuntimeError("application failure")
    with pytest.raises(RuntimeError) as raised:
        with diagnostics.phase("ack"):
            raise original
    assert raised.value is original
    with diagnostics.phase("ack"):
        completed = True
    assert completed


def test_duration_uses_monotonic_clock(monkeypatch, measurements):
    ticks = iter((100.0, 102.5))
    monkeypatch.setattr(diagnostics.time, "monotonic", lambda: next(ticks))
    with diagnostics.phase("attach"):
        pass
    assert diagnostics.snapshot()["phases"][0]["sum_s"] == 2.5


def test_broken_optional_observer_does_not_discard_local_record(monkeypatch, measurements):
    def fail(_event):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(diagnostics, "_OBSERVER", fail)
    diagnostics.record_phase("recycle", 2)
    assert diagnostics.snapshot()["phases"][0]["count"] == 1


def test_trace_poll_reports_eviction_and_never_calls_a_subscriber(measurements):
    for _ in range(diagnostics.MAX_RECENT + 2):
        diagnostics.record_phase("readiness", 0.01)
    result = diagnostics.events_since(0, limit=2)
    assert result["missed"] == 2
    assert [event["sequence"] for event in result["events"]] == [3, 4]
    assert result["coverage"] == "best_effort_process_local"
    tail = diagnostics.events_since(result["events"][-1]["sequence"])
    assert tail["missed"] == 0
    assert tail["events"][0]["sequence"] == 5
