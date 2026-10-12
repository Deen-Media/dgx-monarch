"""Return telemetry observes ownership and successful descriptors only."""
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.rdma_job_registry as jobs
import dgx_monarch.transfer as transfer
from dgx_monarch.actor.rdma_handoff import RDMAHandoffMixin
from dgx_monarch.return_diagnostics import ReturnCounts, actor_snapshot, driver_snapshot
from transfer_helpers import _isolate_process_lifetime_rdma_poison  # noqa: F401


class Worker(RDMAHandoffMixin):
    def __init__(self, mode="rdma", threshold=1):
        self.rank = 0
        self._setup_generation = 7
        self._latent_return = transfer.LatentReturn(mode, min_bytes=threshold)
        self._init_rdma_handoffs()

    def pack(self, tensor=None):
        return self._pack_latent_result(
            {}, {"samples": torch.ones(4) if tensor is None else tensor}, {})["latent"]


def test_absent_state_is_unavailable_not_zero(monkeypatch):
    monkeypatch.delitem(sys.modules, "dgx_monarch.rdma_job_registry")
    monkeypatch.delitem(sys.modules, "dgx_monarch.transfer")
    assert driver_snapshot() == {"schema": 1, "available": False}
    fake = SimpleNamespace(_latent_handoffs=SimpleNamespace(live_count=0, tombstone_count=0))
    assert actor_snapshot(fake) == {"schema": 1, "available": False}


def test_driver_snapshot_does_not_remove_or_wait_for_owners(monkeypatch):
    pending, poison = object(), object()
    lock = threading.Lock()
    monkeypatch.setattr(jobs, "_JOBS", [pending])
    monkeypatch.setattr(jobs, "_LOCK", lock)
    monkeypatch.setattr(transfer, "_RDMA_POISONED_OWNERS", [poison])
    assert driver_snapshot()["pending_read_jobs"] == 1
    assert driver_snapshot()["native_poison_owners"] == 1
    with lock:
        assert "pending_read_jobs" not in driver_snapshot()
    assert jobs._JOBS == [pending]
    assert transfer._RDMA_POISONED_OWNERS == [poison]


def test_counts_have_fixed_labels_and_detached_snapshots():
    counts = ReturnCounts()
    counts.record("message", "secret-or-exception-text")
    counts.record("rdma", "native")
    counts.record("rdma", "native")
    snapshot = counts.snapshot()
    assert snapshot == [{"kind": "rdma", "reason": "native", "count": 2}]
    snapshot[0]["count"] = 0
    assert counts.snapshot()[0]["count"] == 2


@pytest.mark.parametrize("mode,threshold,usable,reason", [
    ("message", 1, True, "configured_message"),
    ("rdma", 1000, True, "below_threshold"),
    ("rdma", 1, False, "pack_fallback"),
])
def test_message_descriptors_count_actual_gate(monkeypatch, mode, threshold, usable, reason):
    monkeypatch.setattr(transfer, "_rdma_usable", lambda: usable)
    worker = Worker(mode, threshold)
    assert worker.pack()["kind"] == "message"
    snapshot = actor_snapshot(worker)
    assert snapshot["return_counts"] == [{"kind": "message", "reason": reason, "count": 1}]
    assert snapshot["live_handoffs"] == 0


def test_native_then_capacity_fallback_and_snapshot_cannot_ack(monkeypatch):
    monkeypatch.setattr(transfer, "_rdma_usable", lambda: True)
    monkeypatch.setattr(transfer, "_qp_split", lambda: 1)
    worker = Worker()
    descriptor = worker.pack()
    assert descriptor["kind"] == "rdma"
    buffer = descriptor["parts"][0]["buffer"]
    assert worker.pack()["kind"] == "message"
    for _ in range(2):
        snapshot = actor_snapshot(worker)
        assert snapshot["live_handoffs"] == 1
        assert snapshot["tombstones"] == 0
        assert snapshot["return_counts"] == [
            {"kind": "message", "reason": "registry_capacity", "count": 1},
            {"kind": "rdma", "reason": "native", "count": 1},
        ]
    snapshot["live_handoffs"] = 0
    assert worker._latent_handoffs.live_count == 1
    assert buffer.dropped == 0
    # Only the existing generation-bound ACK retires ownership.
    assert worker._ack_latent_handoff_impl(7, descriptor["owner_token"])["status"] == "released"
    assert actor_snapshot(worker)["live_handoffs"] == 0
    assert actor_snapshot(worker)["tombstones"] == 1


def test_ambiguous_pack_does_not_count_success_or_release_owner():
    worker = Worker()
    retained = object()

    def pack(_key, _tensor, *, handoff, **_kwargs):
        handoff.update(parts=[{"buffer": retained}], state="registered")
        return {"kind": "message"}

    worker._latent_return = SimpleNamespace(mode="rdma", pack=pack)
    with pytest.raises(RuntimeError, match="ownership is ambiguous"):
        worker.pack()
    snapshot = actor_snapshot(worker)
    assert snapshot["live_handoffs"] == 1
    assert snapshot["return_counts"] == []


def test_counter_failure_cannot_change_native_handoff(monkeypatch):
    monkeypatch.setattr(transfer, "_rdma_usable", lambda: True)
    monkeypatch.setattr(transfer, "_qp_split", lambda: 1)
    worker = Worker()

    def fail(*_args):
        raise RuntimeError("diagnostic failure")

    monkeypatch.setattr(worker._return_counts, "record", fail)
    assert worker.pack()["kind"] == "rdma"
    assert worker._latent_handoffs.live_count == 1


def test_nested_tensor_records_observed_gate():
    class NestedTensor:
        def __init__(self, parts):
            self.parts = parts

        def unbind(self):
            return self.parts

    worker = Worker()
    assert worker.pack(NestedTensor((torch.ones(2), torch.ones(3))))["kind"] == "message"
    assert actor_snapshot(worker)["return_counts"] == [
        {"kind": "message", "reason": "nested_tensor", "count": 1}]
