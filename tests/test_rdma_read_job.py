"""rdma_read_job claim, settlement and token boundaries."""
import dis
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.rdma_read_driver as rdma_read_driver
import dgx_monarch.rdma_read_job as rdma_read_job
import dgx_monarch.transfer as transfer
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


def test_job_token_end_read_baseexception_retires_exact_token():
    class EndInterrupted(BaseException):
        pass

    boundary = EndInterrupted("job-token end_read return interrupted")
    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={1: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)

    class Guard(SetupBoundFuture):
        interrupted = False

        def end_read(self, token=None):
            result = super().end_read(token)
            if (getattr(token, "lease", None) is self
                    and not self.interrupted):
                self.interrupted = True
                raise boundary
            return result

    class Buffer(_FakeRDMABuffer):
        reads = 0

        def read_into(self, dst, timeout=None):
            self.reads += 1
            return super().read_into(dst, timeout=timeout)

    guard = Guard(None, handle, 1)
    buffer = Buffer(torch.arange(2, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 2}],
    }
    with pytest.raises(EndInterrupted) as exc_info:
        read_latent_result(desc, guard)
    assert exc_info.value is boundary
    assert guard.interrupted and guard.readers == 0
    assert buffer.reads == buffer.dropped == 1


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_retire_token_call_before_apply_retries_exact_key():
    class StopNow(BaseException):
        pass

    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={1: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)
    guard = SetupBoundFuture(None, handle, 1)

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
    code = rdma_read_job.LatentReadJob._finish_winner.__code__
    target = _call_instruction_offset(code, "_retire_token")
    boundary = StopNow("token retirement call interrupted before apply")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary,
            lambda: read_latent_result(desc, guard))

    assert exc_info.value is boundary
    assert buffer.reads == buffer.dropped == 1
    assert guard.readers == 0
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_claim_primary_append_boundary_preserves_cancellation():
    class StopNow(BaseException):
        pass

    ordinary = RuntimeError("token publication returned with an error")
    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={1: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)

    class Guard(SetupBoundFuture):
        def begin_read(self, owner=None):
            super().begin_read(owner)
            raise ordinary

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

    guard = Guard(None, handle, 1)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_job._Claim.remember.__code__
    target = _call_instruction_offset(code, "append", successor=True)
    boundary = StopNow("claim primary append interrupted")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary,
            lambda: read_latent_result(desc, guard))

    assert exc_info.value is boundary
    assert buffer.reads == 0 and buffer.drops == 1
    assert guard.readers == 0
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_off_loop_sync_fallback_claim_return_boundary_still_settles(monkeypatch):
    class StopNow(BaseException):
        pass

    preparation = RuntimeError("read-token preparation recovered")
    first = RuntimeError("first start failed")
    second = RuntimeError("second start failed")

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

    starts = 0
    real_start = threading.Thread.start

    def fail_both(self):
        nonlocal starts
        if self.name != "dgxm-latent-read":
            return real_start(self)
        starts += 1
        raise first if starts == 1 else second

    monkeypatch.setattr(threading.Thread, "start", fail_both)
    def prepare_with_recovered_error(job):
        # Set what production prepare sets for a tokenless job: the job owns
        # settlement although it holds no read token.
        job.token_retired = True
        return True, preparation

    monkeypatch.setattr(
        rdma_read_job.LatentReadJob, "prepare", prepare_with_recovered_error)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_driver.run_owned_off_loop.__code__
    target = _call_instruction_offset(
        code, "remember", after=-1, successor=True)
    boundary = StopNow("sync fallback primary return interrupted")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary,
            lambda: read_latent_result(desc))

    assert exc_info.value is boundary
    assert starts == 2 and buffer.reads == 0 and buffer.drops == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.parametrize("reuse_error", [False, True])
def test_runner_construction_retry_never_self_chains(
    monkeypatch, reuse_error,
):
    from dgx_monarch import mesh_setup

    cancellation = KeyboardInterrupt("runner retry cancelled")
    ordinary = (cancellation if reuse_error else
                RuntimeError("first runner construction failed"))
    calls = 0

    def fail_twice(self, thread_name, primary=None):
        nonlocal calls
        calls += 1
        raise ordinary if calls == 1 else cancellation

    monkeypatch.setattr(
        rdma_read_job.LatentReadJob, "new_runner", fail_twice)
    class Buffer(_FakeRDMABuffer):
        reads = 0

        def read_into(self, dst, timeout=None):
            self.reads += 1
            return super().read_into(dst, timeout=timeout)

    generation = 22
    buffer = Buffer(torch.arange(1, dtype=torch.uint8))
    context = _real_pending_rdma(monkeypatch, buffer, generation)
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(KeyboardInterrupt) as exc_info:
        context.pending.result()

    assert exc_info.value is cancellation
    if reuse_error:
        assert exc_info.value.__cause__ is not exc_info.value
        assert exc_info.value.__context__ is not exc_info.value
    else:
        assert exc_info.value.__cause__ is ordinary
    assert calls == 2 and buffer.reads == buffer.dropped == 0
    assert context.guard.begins == context.guard.readers == 0
    assert context.ack_calls == []
    assert rdma_read_job.pending_job_count() == baseline
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
def test_runner_error_append_boundary_replaces_adopted_entry():
    class StopNow(BaseException):
        pass

    ordinary = RuntimeError("contender observed the durable failure")
    holder = []
    construction_owner = []
    job = rdma_read_job.LatentReadJob(
        None, lambda *_args: None, holder, construction_owner)
    claim = rdma_read_job._Claim()
    job.claims.append(claim)
    outcome = (None, ordinary)
    job.outcome_owner.append(outcome)
    job._phase = (rdma_read_job.RESULT_READY, outcome)
    rdma_read_job._registry.register(job)
    code = rdma_read_job.LatentReadJob.run_thread.__code__
    target = _call_instruction_offset(code, "append", successor=True)
    boundary = StopNow("runner error append interrupted after apply")

    _run_at_instruction(
        code, target, boundary, lambda: job.run_thread(claim))

    assert job.runner_errors == [boundary]
    driver_error = RuntimeError("driver also failed")
    assert rdma_read_driver._merge_job_error(job, driver_error) is boundary
    assert rdma_read_job.pending_job_count_for_handle(None) == 0


def test_authoritative_outcome_cancellation_outranks_join_timeout(monkeypatch):
    import asyncio

    cancellation = KeyboardInterrupt("holder copy cancelled before apply")
    entered = threading.Event()
    release = threading.Event()
    baseline = rdma_read_job.pending_job_count()
    real_register = rdma_read_job._registry.register

    class BlockingHolder(list):
        calls = 0

        def append(self, outcome):
            self.calls += 1
            if self.calls == 1:
                raise cancellation
            entered.set()
            assert release.wait(2.0)
            super().append(outcome)

    def capture_job(job):
        job.holder = BlockingHolder()
        return real_register(job)

    def short_join(runner, timeout_s, thread_name):
        runner.join(timeout=0.01)
        if runner.is_alive():
            raise TimeoutError(f"{thread_name} timed out in the test")

    monkeypatch.setattr(rdma_read_job._registry, "register", capture_job)
    monkeypatch.setattr(rdma_read_driver, "_join_or_timeout", short_join)
    buffer = _FakeRDMABuffer(torch.arange(2, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 2}],
    }

    async def on_loop():
        with pytest.raises(KeyboardInterrupt) as exc_info:
            read_latent_result(desc)
        return exc_info.value

    try:
        assert asyncio.run(on_loop()) is cancellation
        assert entered.wait(1.0)
    finally:
        release.set()
    deadline = time.monotonic() + 2.0
    while (rdma_read_job.pending_job_count() != baseline
           and time.monotonic() < deadline):
        time.sleep(0.005)
    assert buffer.dropped == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("successor", [False, True])
def test_final_delivery_boundary_is_published_to_caller(monkeypatch, successor):
    import asyncio

    class StopNow(BaseException):
        pass

    ordinary = RuntimeError("protected delivery attempt failed")
    boundary = StopNow("final delivery fallback interrupted")
    calls = 0
    real_deliver = rdma_read_job.LatentReadJob._deliver_outcome

    def fail_first_two(self):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise ordinary
        return real_deliver(self)

    monkeypatch.setattr(
        rdma_read_job.LatentReadJob, "_deliver_outcome", fail_first_two)
    buffer = _FakeRDMABuffer(torch.arange(2, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 2}],
    }
    code = rdma_read_job.LatentReadJob.run_thread.__code__
    target = _call_instruction_offset(
        code, "_deliver_outcome", after=1, successor=successor)
    baseline = rdma_read_job.pending_job_count()

    async def on_loop():
        with pytest.raises(StopNow) as exc_info:
            _run_at_instruction(
                code, target, boundary,
                lambda: read_latent_result(desc))
        return exc_info.value

    assert asyncio.run(on_loop()) is boundary
    assert calls == 3
    assert buffer.dropped == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("successor", [False, True])
def test_detached_final_recovery_boundary_retires_root(monkeypatch, successor):
    import asyncio

    class StopNow(BaseException):
        pass

    entered = threading.Event()
    release = threading.Event()

    class BlockingFuture:
        def get(self, timeout=None):
            entered.set()
            assert release.wait(2.0)

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def __init__(self):
            self.reads = self.drops = 0

        def read_into(self, dst, timeout=None):
            self.reads += 1
            dst.fill_(8)
            return BlockingFuture()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    ordinary = RuntimeError("all protected delivery attempts failed")
    boundary = StopNow("final recovery helper interrupted")
    calls = 0
    captured = []
    hooks = []
    real_deliver = rdma_read_job.LatentReadJob._deliver_outcome
    real_register = rdma_read_job._registry.register

    def fail_first_three(self):
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise ordinary
        return real_deliver(self)

    def capture_job(job):
        captured.append(job)
        return real_register(job)

    def short_join(runner, timeout_s, thread_name):
        runner.join(timeout=0.01)
        if runner.is_alive():
            raise TimeoutError(f"{thread_name} timed out in the test")

    monkeypatch.setattr(
        rdma_read_job.LatentReadJob, "_deliver_outcome", fail_first_three)
    monkeypatch.setattr(rdma_read_job._registry, "register", capture_job)
    monkeypatch.setattr(rdma_read_driver, "_join_or_timeout", short_join)
    monkeypatch.setattr(threading, "excepthook", hooks.append)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_job.LatentReadJob.run_thread.__code__
    target = _call_instruction_offset(
        code, "recover_final_delivery", successor=successor)
    baseline = rdma_read_job.pending_job_count()

    async def on_loop():
        with pytest.raises(TimeoutError) as exc_info:
            read_latent_result(desc)
        return exc_info.value

    def exercise():
        timeout_error = asyncio.run(on_loop())
        assert entered.is_set()
        release.set()
        deadline = time.monotonic() + 2.0
        while (rdma_read_job.pending_job_count() != baseline
               and time.monotonic() < deadline):
            time.sleep(0.005)
        return timeout_error

    _run_at_instruction(code, target, boundary, exercise)

    assert calls == 4 and buffer.reads == buffer.drops == 1
    assert captured[0].outcome_owner[0][1] is boundary
    assert hooks == []
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("protected_failures", [1, 2])
@pytest.mark.parametrize("successor", [False, True])
def test_holder_publication_never_substitutes_for_root_retirement(
    monkeypatch, protected_failures, successor,
):
    import asyncio

    class StopNow(BaseException):
        pass

    entered = threading.Event()
    release = threading.Event()

    class BlockingFuture:
        def get(self, timeout=None):
            entered.set()
            assert release.wait(2.0)

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def __init__(self):
            self.reads = self.drops = 0

        def read_into(self, dst, timeout=None):
            self.reads += 1
            dst.fill_(9)
            return BlockingFuture()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    ordinary = RuntimeError("protected delivery failed before caller copy")
    boundary = StopNow("holder published before root retirement")
    calls = 0
    captured = []
    hooks = []
    real_deliver = rdma_read_job.LatentReadJob._deliver_outcome
    real_register = rdma_read_job._registry.register

    def fail_protected(self):
        nonlocal calls
        calls += 1
        if calls <= protected_failures:
            raise ordinary
        return real_deliver(self)

    def capture_job(job):
        captured.append(job)
        return real_register(job)

    def short_join(runner, timeout_s, thread_name):
        runner.join(timeout=0.01)
        if runner.is_alive():
            raise TimeoutError(f"{thread_name} timed out in the test")

    monkeypatch.setattr(
        rdma_read_job.LatentReadJob, "_deliver_outcome", fail_protected)
    monkeypatch.setattr(rdma_read_job._registry, "register", capture_job)
    monkeypatch.setattr(rdma_read_driver, "_join_or_timeout", short_join)
    monkeypatch.setattr(threading, "excepthook", hooks.append)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_job.LatentReadJob._deliver_outcome_locked.__code__
    target = _call_instruction_offset(
        code, "_mark_token_retired", successor=successor)
    baseline = rdma_read_job.pending_job_count()

    async def on_loop():
        with pytest.raises(TimeoutError) as exc_info:
            read_latent_result(desc)
        return exc_info.value

    def exercise():
        timeout_error = asyncio.run(on_loop())
        assert entered.is_set()
        release.set()
        deadline = time.monotonic() + 2.0
        while (rdma_read_job.pending_job_count() != baseline
               and time.monotonic() < deadline):
            time.sleep(0.005)
        return timeout_error

    _run_at_instruction(code, target, boundary, exercise)

    job = captured[0]
    assert calls == protected_failures + 2
    assert buffer.reads == buffer.drops == 1
    assert job.outcome_owner[0][1] is boundary
    assert job.holder and job.holder[0] is job.outcome_owner[0]
    assert job.state == rdma_read_job.CONSUMED
    assert not rdma_read_job._registry.registered(job)
    assert hooks == []
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("seam", ["begin-read", "prepare-return"])
def test_prepare_call_boundaries_use_unread_settlement(seam):
    """Interrupted just after begin_read or prepare returns, the job reuses its token and drops unread."""
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

    generation = 2
    registry = HandoffRegistry()
    buffer = Buffer()
    actor_part = {"buffer": buffer, "offset": 0, "nbytes": 1}
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(parts=[actor_part], keepalive=object(), state="registered")
    registry.mark_ready(handoff)
    token = handoff["token"]

    class Handle:
        world = 1

        def __init__(self):
            self.lock = threading.RLock()
            self.sample_leases = {generation: 1}
            self.abandoned_sample_leases = {}
            self.deferred_supervision_error = None
            self.ack_readers = []
            self.guard = None

        def call_all(self, _endpoint, setup_generation, owner_token, timeout_s):
            self.ack_readers.append(self.guard.readers)
            return [{
                "setup_generation": setup_generation,
                "token": owner_token,
                "status": registry.acknowledge(setup_generation, owner_token),
                "rank": 0,
            }]

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    handle.guard = guard
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [dict(actor_part)], "owner_token": token,
        "setup_generation": generation,
    }
    code = (rdma_read_job.LatentReadJob.prepare.__code__
            if seam == "begin-read" else rdma_read_driver.run_owned_off_loop.__code__)
    instructions = list(dis.get_instructions(code))
    needle = "begin_read" if seam == "begin-read" else "prepare"
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == needle)
    call = next(
        index for index in range(load + 1, len(instructions))
        if instructions[index].opname == "CALL")
    if seam == "begin-read":
        assert instructions[call + 1].opname == "POP_TOP"
        target = instructions[call + 2].offset
    else:
        target = instructions[call + 1].offset
    boundary = StopNow(f"interrupted at {seam}")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(
                tool_id, code, 0)
            raise boundary

    baseline = rdma_read_job.pending_job_count()
    monitoring.use_tool_id(tool_id, "dgxm-rdma-begin-read-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, code, monitoring.events.INSTRUCTION)
    with pytest.raises(StopNow) as exc_info:
        try:
            read_latent_result(desc, guard)
        finally:
            monitoring.set_local_events(
                tool_id, code, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is boundary
    assert buffer.reads == 0 and buffer.drops == 1
    assert handle.ack_readers == [1] and guard.readers == 0
    assert registry.get(generation, token)["_registry_state"] == "RELEASED"
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize(
    ("attribute", "successor"), [("register", 2), ("prepare", 1)])
def test_tokenless_root_and_prepare_boundaries_are_contained(
    monkeypatch, attribute, successor,
):
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

    monkeypatch.setattr(rdma_read_job._registry, "_JOBS", [])
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_driver.run_owned_off_loop.__code__
    target = _call_instruction_offset(
        code, attribute, successor=successor)
    boundary = StopNow(f"{attribute} continuation interrupted")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary, lambda: read_latent_result(desc))

    assert exc_info.value is boundary
    assert buffer.reads == 0
    if attribute == "register":
        assert buffer.drops == 0
    else:
        assert buffer.drops == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("successor", [1, 2])
def test_normal_run_thread_finally_boundaries_deliver_and_unroot(successor):
    class StopNow(BaseException):
        pass

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
    code = rdma_read_job.LatentReadJob.run_thread.__code__
    instructions = list(dis.get_instructions(code))
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "run_once")
    call = next(
        index for index in range(load + 1, len(instructions))
        if instructions[index].opname == "CALL")
    assert instructions[call + 1].opname == "POP_TOP"
    target = instructions[call + successor].offset
    boundary = StopNow(f"run-thread continuation {successor} interrupted")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary, lambda: read_latent_result(desc))

    assert exc_info.value is boundary
    assert buffer.reads == buffer.dropped == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_run_thread_first_protected_opcode_claims_unread_recovery():
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

    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_job.LatentReadJob.run_thread.__code__
    instructions = list(dis.get_instructions(code))
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "run_once")
    assert instructions[load - 1].opname == "LOAD_FAST"
    target = instructions[load - 1].offset
    boundary = StopNow("run-thread entry interrupted")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary, lambda: read_latent_result(desc))

    assert exc_info.value is boundary
    assert buffer.reads == 0 and buffer.drops == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_claim_call_return_interruption_has_one_unread_winner():
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

    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_job.LatentReadJob.run_once.__code__
    instructions = list(dis.get_instructions(code))
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "_resolve_claim")
    call = next(
        index for index in range(load + 1, len(instructions))
        if instructions[index].opname == "CALL")
    target = instructions[call + 1].offset
    boundary = StopNow("claim return interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise boundary

    baseline = rdma_read_job.pending_job_count()
    monitoring.use_tool_id(tool_id, "dgxm-rdma-claim-test")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    with pytest.raises(StopNow) as exc_info:
        try:
            read_latent_result(desc)
        finally:
            monitoring.set_local_events(tool_id, code, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is boundary
    assert buffer.reads == 0 and buffer.drops == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("position", ["before-store", "after-store"])
def test_operation_entry_store_interruption_never_redrops(position):
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

    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    code = rdma_read_job._Claim.enter_operation.__code__
    instructions = list(dis.get_instructions(code))
    store = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "operation_phase")
    target = instructions[store if position == "before-store" else store + 1].offset
    boundary = StopNow(f"operation entry {position} interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise boundary

    baseline = rdma_read_job.pending_job_count()
    monitoring.use_tool_id(tool_id, "dgxm-rdma-operation-entry-test")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    with pytest.raises(StopNow) as exc_info:
        try:
            read_latent_result(desc)
        finally:
            monitoring.set_local_events(tool_id, code, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is boundary
    assert buffer.reads == 0 and buffer.drops == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_finish_call_return_interruption_adopts_durable_outcome_once():
    class StopNow(BaseException):
        pass

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
    code = rdma_read_job.LatentReadJob._settle_claim.__code__
    instructions = list(dis.get_instructions(code))
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "_finish_winner")
    call = next(
        index for index in range(load + 1, len(instructions))
        if instructions[index].opname == "CALL")
    target = instructions[call + 1].offset
    boundary = StopNow("finish return interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise boundary

    baseline = rdma_read_job.pending_job_count()
    monitoring.use_tool_id(tool_id, "dgxm-rdma-finish-test")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    with pytest.raises(StopNow) as exc_info:
        try:
            read_latent_result(desc)
        finally:
            monitoring.set_local_events(tool_id, code, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is boundary
    assert buffer.reads == buffer.dropped == 1
    assert rdma_read_job.pending_job_count() == baseline


def test_read_authority_publication_interrupt_reuses_exact_token():
    """After begin_read publishes a token and raises, recovery reuses it, drops unread and frees the reader."""
    from dgx_monarch import mesh_setup

    class StopNow(BaseException):
        pass

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def __init__(self):
            self.reads = 0
            self.drops = 0

        def read_into(self, _dst, timeout=None):
            self.reads += 1
            return ImmediateFuture()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    handle = SimpleNamespace(
        lock=threading.RLock(),
        sample_leases={1: 1},
        abandoned_sample_leases={},
        deferred_supervision_error=None,
    )
    boundary = StopNow("reader publication interrupted")

    class Guard(mesh_setup.SetupBoundFuture):
        def __init__(self, *args):
            super().__init__(*args)
            self.published = None
            self.ensure_tokens = []

        def begin_read(self, owner=None):
            token = super().begin_read(owner)
            assert owner is not None and owner[0] is token
            self.published = token
            raise boundary

        def ensure_read_token(self, token):
            self.ensure_tokens.append(token)
            return super().ensure_read_token(token)

    guard = Guard(object(), handle, 1)
    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    with pytest.raises(StopNow) as exc_info:
        read_latent_result(desc, guard)

    assert exc_info.value is boundary
    assert guard.published is not None
    assert guard.ensure_tokens
    assert all(token is guard.published for token in guard.ensure_tokens)
    assert guard.readers == 0
    assert buffer.reads == 0
    assert buffer.drops == 1
    assert transfer._RDMA_POISONED_OWNERS == []
    guard.abandon()
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
