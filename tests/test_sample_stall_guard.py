"""The sample stall guard: a fleet that dies mid-collective must surface a
typed error at the driver instead of holding the prompt queue. A worker whose
CUDA context died on a sticky fault refuses its GPU endpoints until a recycle.
"""
from __future__ import annotations

import time

import pytest

from dgx_monarch import mesh_lease, mesh_runtime, progress
from dgx_monarch.actor import failure


class _FakeFuture:
    pass


def test_stall_budget_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv(mesh_lease.SAMPLE_STALL_ENV, raising=False)
    assert mesh_lease.stall_budget_s() == mesh_lease.SAMPLE_STALL_DEFAULT_S
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "120.5")
    assert mesh_lease.stall_budget_s() == 120.5
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "0")
    assert mesh_lease.stall_budget_s() == 0.0
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "not-a-number")
    assert mesh_lease.stall_budget_s() == mesh_lease.SAMPLE_STALL_DEFAULT_S


def test_no_activity_fn_is_a_plain_off_loop_get(monkeypatch):
    calls = []

    def fake_get(future, timeout_s, thread_name="x"):
        calls.append(timeout_s)
        return "value-mesh"

    monkeypatch.setattr(mesh_runtime, "get_off_loop", fake_get)
    out = mesh_lease.collect_with_liveness(None, _FakeFuture(), 900.0, None)
    assert out == "value-mesh"
    assert calls == [900.0]


def test_disabled_budget_is_a_plain_off_loop_get(monkeypatch):
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "0")
    calls = []

    def fake_get(future, timeout_s, thread_name="x"):
        calls.append(timeout_s)
        return "value-mesh"

    monkeypatch.setattr(mesh_runtime, "get_off_loop", fake_get)
    out = mesh_lease.collect_with_liveness(
        None, _FakeFuture(), 900.0, lambda: time.monotonic())
    assert out == "value-mesh"
    assert calls == [900.0]


def test_live_fleet_keeps_polling_until_the_result_lands(monkeypatch):
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "600")
    attempts = []

    def fake_get(future, timeout_s, thread_name="x"):
        attempts.append(timeout_s)
        if len(attempts) < 3:
            raise TimeoutError("slice elapsed")
        return "value-mesh"

    monkeypatch.setattr(mesh_runtime, "get_off_loop", fake_get)
    out = mesh_lease.collect_with_liveness(
        None, _FakeFuture(), 900.0, lambda: time.monotonic())
    assert out == "value-mesh"
    assert len(attempts) == 3
    assert all(t <= mesh_lease._STALL_SLICE_S for t in attempts)


def test_stalled_fleet_raises_the_typed_error(monkeypatch):
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "0.05")
    stale = time.monotonic() - 1000.0

    def fake_get(future, timeout_s, thread_name="x"):
        raise TimeoutError("slice elapsed")

    monkeypatch.setattr(mesh_runtime, "get_off_loop", fake_get)
    with pytest.raises(mesh_lease.SampleStallError) as excinfo:
        mesh_lease.collect_with_liveness(
            None, _FakeFuture(), 900.0, lambda: stale)
    message = str(excinfo.value)
    assert "no progress" in message
    assert mesh_lease.SAMPLE_STALL_ENV in message
    assert "TROUBLESHOOTING.md #86" in message


def test_idle_anchors_at_wait_start_not_receiver_birth(monkeypatch):
    """A deferred collect against a long-stale receiver stamp must not stall
    at once: idle time never counts from before this wait began."""
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "600")
    stale = time.monotonic() - 10_000.0
    attempts = []

    def fake_get(future, timeout_s, thread_name="x"):
        attempts.append(timeout_s)
        if len(attempts) < 3:
            raise TimeoutError("slice elapsed")
        return "value-mesh"

    monkeypatch.setattr(mesh_runtime, "get_off_loop", fake_get)
    out = mesh_lease.collect_with_liveness(
        None, _FakeFuture(), 900.0, lambda: stale)
    assert out == "value-mesh"
    assert len(attempts) == 3


def test_stall_evicts_deliberately_and_other_errors_stay_classified(monkeypatch):
    from dgx_monarch import mesh_helpers

    calls = []
    monkeypatch.setattr(
        mesh_helpers, "mark_defunct_deliberate",
        lambda handle, exc, holding_lock=False: calls.append(("deliberate", handle)))
    monkeypatch.setattr(
        mesh_helpers, "mark_defunct_preserving_primary",
        lambda handle, exc, holding_lock=False: calls.append(("classified", handle)))

    def raise_stall(handle, future, timeout_s, activity_fn):
        raise mesh_lease.SampleStallError("stalled")

    monkeypatch.setattr(mesh_lease, "collect_with_liveness", raise_stall)
    handle = object()
    with pytest.raises(mesh_lease.SampleStallError):
        mesh_lease.collect_sample(handle, _FakeFuture(), 900.0, lambda: 0.0)
    assert calls == [("deliberate", handle)]

    def raise_other(handle, future, timeout_s, activity_fn):
        raise TimeoutError("hard timeout")

    monkeypatch.setattr(mesh_lease, "collect_with_liveness", raise_other)
    with pytest.raises(TimeoutError):
        mesh_lease.collect_sample(handle, _FakeFuture(), 900.0, lambda: 0.0)
    assert calls == [("deliberate", handle), ("classified", handle)]


def test_hard_timeout_still_surfaces_the_canonical_error(monkeypatch):
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "600")

    def fake_get(future, timeout_s, thread_name="x"):
        raise TimeoutError(f"canonical timeout at {timeout_s}")

    monkeypatch.setattr(mesh_runtime, "get_off_loop", fake_get)
    # A liveness stamp that always reads fresh: only the hard deadline ends it.
    with pytest.raises(TimeoutError):
        mesh_lease.collect_with_liveness(
            None, _FakeFuture(), 0.05, lambda: time.monotonic())


def test_progress_receiver_exposes_a_monotonic_activity_stamp():
    receiver = progress.ProgressReceiver(10)
    stamp = receiver.activity()
    assert isinstance(stamp, float)
    assert stamp <= time.monotonic()


class _Worker:
    """Bare stand-in for GPUWorker: the latch is a plain attribute."""


def test_poisoned_worker_refuses_before_the_endpoint_runs():
    ran = []

    @failure.cleanup_on_failure
    def endpoint(self):
        ran.append(True)
        return "ok"

    worker = _Worker()
    worker._cuda_context_poisoned = True
    with pytest.raises(failure.CudaContextPoisonedError) as excinfo:
        endpoint(worker)
    assert not ran
    assert "Recycle" in str(excinfo.value)
    assert "TROUBLESHOOTING.md #86" in str(excinfo.value)


def test_sticky_fault_latches_poison_and_reraises_the_primary(monkeypatch):
    monkeypatch.setattr(failure, "_cuda_context_dead", lambda: True)

    @failure.cleanup_on_failure
    def endpoint(self):
        raise RuntimeError("primary boom")

    worker = _Worker()
    with pytest.raises(RuntimeError, match="primary boom"):
        endpoint(worker)
    assert worker._cuda_context_poisoned is True


def test_benign_failure_leaves_the_context_trusted(monkeypatch):
    monkeypatch.setattr(failure, "_cuda_context_dead", lambda: False)

    @failure.cleanup_on_failure
    def endpoint(self):
        raise ValueError("typed refusal, context fine")

    worker = _Worker()
    with pytest.raises(ValueError):
        endpoint(worker)
    assert getattr(worker, "_cuda_context_poisoned", False) is False


def test_healthy_call_never_probes(monkeypatch):
    probes = []
    monkeypatch.setattr(
        failure, "_cuda_context_dead", lambda: probes.append(1) or False)

    @failure.cleanup_on_failure
    def endpoint(self):
        return "ok"

    assert endpoint(_Worker()) == "ok"
    assert probes == []
