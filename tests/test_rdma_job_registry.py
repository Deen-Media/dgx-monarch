"""rdma_job_registry token registration and holder lifetime."""
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.rdma_read_job as rdma_read_job
from dgx_monarch.mesh_lease import SetupBoundFuture
from dgx_monarch.rdma_ownership import HandoffRegistry
from dgx_monarch.transfer import read_latent_result
from transfer_helpers import (  # noqa: F401  # autouse fixture import.
    _call_instruction_offset,
    _FakeRDMABuffer,
    _isolate_process_lifetime_rdma_poison,
    _real_pending_rdma,
    _run_at_instruction,
)


def test_durable_job_outcome_precedes_holder_copy_and_unregister(monkeypatch):
    class CopyInterrupted(BaseException):
        pass

    boundary = CopyInterrupted("holder append applied then interrupted")
    events = []
    baseline = rdma_read_job.pending_job_count()
    real_register = rdma_read_job._registry.register
    real_unregister = rdma_read_job._unregister

    class InterruptingHolder(list):
        interrupted = False

        def __init__(self, job):
            super().__init__()
            self.job = job

        def append(self, outcome):
            assert self.job.outcome_owner[0] is outcome
            assert self.job.state == rdma_read_job.RESULT_READY
            assert rdma_read_job._registry.registered(self.job)
            events.append("holder-copy")
            super().append(outcome)
            if not self.interrupted:
                self.interrupted = True
                raise boundary

    def capture_job(job):
        job.holder = InterruptingHolder(job)
        return real_register(job)

    def checked_unregister(job):
        assert job.state == rdma_read_job.CONSUMED
        assert job.holder[0] is job.outcome_owner[0]
        events.append("unregister")
        return real_unregister(job)

    monkeypatch.setattr(rdma_read_job._registry, "register", capture_job)
    monkeypatch.setattr(rdma_read_job, "_unregister", checked_unregister)

    class Buffer(_FakeRDMABuffer):
        reads = 0

        def read_into(self, dst, timeout=None):
            self.reads += 1
            return super().read_into(dst, timeout=timeout)

    buffer = Buffer(torch.arange(2, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 2}],
    }
    with pytest.raises(CopyInterrupted) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is boundary
    assert buffer.reads == buffer.dropped == 1
    assert events == ["holder-copy", "unregister"]
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("successor", [False, True], ids=["call", "successor"])
def test_job_registry_append_boundary_returns_exact_error_and_root(
    monkeypatch, successor,
):
    class StopNow(BaseException):
        pass

    registry = rdma_read_job._registry
    monkeypatch.setattr(registry, "_JOBS", [])
    job = SimpleNamespace(state="PENDING", guard=None)
    code = registry.register.__code__
    target = _call_instruction_offset(code, "append", successor=successor)
    boundary = StopNow(f"registry append {successor=}")

    result = _run_at_instruction(
        code, target, boundary, lambda: registry.register(job))

    assert result is boundary
    assert registry.registered(job)
    assert registry._JOBS == [job]
    assert registry.pending_job_count() == 1


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("successor", [False, True], ids=["call", "successor"])
def test_job_registry_pop_boundary_returns_exact_error_and_removes_root(
    monkeypatch, successor,
):
    class StopNow(BaseException):
        pass

    registry = rdma_read_job._registry
    job = SimpleNamespace(state="CONSUMED", guard=None)
    monkeypatch.setattr(registry, "_JOBS", [job])
    code = registry.unregister.__code__
    target = _call_instruction_offset(code, "pop", successor=successor)
    boundary = StopNow(f"registry pop {successor=}")

    result = _run_at_instruction(
        code, target, boundary, lambda: registry.unregister(job))

    assert result is boundary
    assert not registry.registered(job)
    assert registry._JOBS == []
    assert registry.pending_job_count() == 0


def test_full_read_job_registry_rejects_before_token_or_native_effect(monkeypatch):
    from dgx_monarch import mesh_setup

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def __init__(self):
            self.reads = self.drops = 0

        def read_into(self, _dst, timeout=None):
            self.reads += 1
            return ImmediateFuture()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    roots = [SimpleNamespace(state="PENDING", guard=None) for _ in range(64)]
    monkeypatch.setattr(rdma_read_job._registry, "_JOBS", roots)
    generation = 21
    buffer = Buffer()
    context = _real_pending_rdma(monkeypatch, buffer, generation)
    with pytest.raises(RuntimeError, match="capacity"):
        context.pending.result()

    assert context.guard.begins == 0 and context.guard.readers == 0
    assert context.ack_calls == []
    assert buffer.reads == buffer.drops == 0
    assert rdma_read_job.pending_job_count() == 64
    retained = context.registry.get(generation, context.token)
    assert retained["_registry_state"] == "READY"
    assert retained["parts"][0] is context.actor_part
    assert retained["keepalive"] is context.keepalive
    assert context.registry.live_count == 1
    assert context.registry.tombstone_count == 0

    assert context.pending._state == "closed"
    assert context.closed == [True]
    assert context.guard.state == "abandoned"
    assert context.handle.sample_leases == {}
    assert context.handle.abandoned_sample_leases == {generation: 1}
    mesh_setup.require_no_sample_leases(
        context.handle, "recycle the worker fleet", allow_abandoned=True)
    with pytest.raises(mesh_setup.LifecycleBusyError, match="abandoned sample"):
        mesh_setup.require_no_abandoned_samples(
            context.handle, "dispatch another sample")


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("registry_full", [False, True])
def test_constructor_owner_boundary_preserves_exact_failure(monkeypatch, registry_full):
    class StopNow(BaseException):
        pass

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def __init__(self):
            self.reads = self.drops = 0

        def read_into(self, _dst, timeout=None):
            self.reads += 1
            return ImmediateFuture()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    roots = (
        [SimpleNamespace(state="PENDING", guard=None) for _ in range(64)]
        if registry_full else [])
    monkeypatch.setattr(rdma_read_job._registry, "_JOBS", roots)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_job.LatentReadJob.__init__.__code__
    target = _call_instruction_offset(code, "append", successor=True)
    boundary = StopNow("constructor owner append interrupted")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary, lambda: read_latent_result(desc))

    assert exc_info.value is boundary
    assert buffer.reads == 0
    assert buffer.drops == int(not registry_full)
    assert rdma_read_job.pending_job_count() == baseline


def test_unconfirmed_job_token_never_dispatches_and_gates_lifecycle(monkeypatch):
    from dgx_monarch import mesh_setup

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def __init__(self):
            self.reads = self.drops = 0

        def read_into(self, _dst, timeout=None):
            self.reads += 1
            return ImmediateFuture()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    monkeypatch.setattr(rdma_read_job._registry, "_JOBS", [])
    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={1: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)
    guard = SetupBoundFuture(None, handle, 1)
    guard.abandon()
    generation = 1
    sender_registry = HandoffRegistry()
    buffer = Buffer()
    actor_part = {"buffer": buffer, "offset": 0, "nbytes": 1}
    keepalive = object()
    handoff = {}
    sender_registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[actor_part], keepalive=keepalive, state="registered")
    sender_registry.mark_ready(handoff)
    token = handoff["token"]
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [dict(actor_part)], "owner_token": token,
        "setup_generation": generation,
    }
    with pytest.raises(RuntimeError, match="already abandoned"):
        read_latent_result(desc, guard)

    assert buffer.reads == buffer.drops == 0 and guard.readers == 0
    assert rdma_read_job.pending_job_count() == 1
    assert rdma_read_job.pending_job_count_for_handle(handle) == 1
    retained = sender_registry.get(generation, token)
    assert retained["_registry_state"] == "READY"
    assert retained["parts"][0] is actor_part
    assert retained["keepalive"] is keepalive
    assert sender_registry.live_count == 1
    assert sender_registry.tombstone_count == 0
    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        mesh_setup.require_no_sample_leases(
            handle, "recycle the worker fleet", allow_abandoned=True)
