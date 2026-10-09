"""rdma_read_driver on-loop and off-loop read thread lifetime."""
import gc
import threading
import time
import weakref
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
    _FakeRDMABuffer,
    _isolate_process_lifetime_rdma_poison,
    _LoopRefusingBuffer,
    _real_pending_rdma,
)


def test_read_latent_result_rdma_works_from_a_running_event_loop():
    """The read and release must run off ComfyUI's event loop. On 2026-07-13,
    under torchmonarch 0.5.0, Future.get() refused on that loop, and pixel-space
    PixelDiT/PiD latents, the first to cross the RDMA threshold from the render
    path, hit the refusal (docs/TROUBLESHOOTING.md #29)."""
    import asyncio

    import torch

    from dgx_monarch.transfer import read_latent_result

    src = torch.arange(64, dtype=torch.uint8)
    buffer = _LoopRefusingBuffer(src)
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [64],
        "parts": [{"buffer": buffer, "offset": 0, "length": 64}],
    }

    async def _on_loop():
        return read_latent_result(desc)

    out = asyncio.run(_on_loop())
    assert torch.equal(out, src)
    assert buffer.dropped == 1


def test_on_loop_rdma_full_failure_envelope_preserves_read_error(monkeypatch):
    """A failed read keeps its error; the outer join covers the read and drop budgets plus both margins."""
    import asyncio

    class ReadFailure(RuntimeError):
        pass

    class DropFailure(RuntimeError):
        pass

    calls: list[tuple[str, float | None]] = []

    class _BudgetFailureFuture:
        def __init__(self, phase: str, failure: BaseException):
            self._phase = phase
            self._failure = failure

        def get(self, timeout=None):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                # Fail at once, as a future that spent its whole budget would;
                # the outer-timeout assert below checks the budget without a real wait.
                calls.append((self._phase, timeout))
                raise self._failure from None
            raise AssertionError("RDMA futures must be driven off the event loop")

    class _ReadAndDropFailingBuffer:
        def __init__(self):
            self.dropped = 0

        def read_into(self, _dst, timeout=None):
            return _BudgetFailureFuture("read", ReadFailure("read timed out"))

        def drop(self):
            self.dropped += 1
            return _BudgetFailureFuture("drop", DropFailure("drop timed out"))

    per_operation_s = 0.125
    monkeypatch.setattr(transfer, "RDMA_READ_TIMEOUT_S", per_operation_s)
    real_off_loop = transfer.run_blocking_off_loop
    outer: dict[str, float] = {}

    def _capture_outer(guard, operation, timeout_s, thread_name):
        outer["timeout_s"] = timeout_s()
        return real_off_loop(
            guard, operation, timeout_s=timeout_s, thread_name=thread_name)

    monkeypatch.setattr(transfer, "run_blocking_off_loop", _capture_outer)
    buffer = _ReadAndDropFailingBuffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    async def _on_loop():
        with pytest.raises(ReadFailure, match="read timed out"):
            read_latent_result(desc)

    asyncio.run(_on_loop())
    assert buffer.dropped == 0
    # The read's get() waits the margin past the native read deadline, so that
    # deadline resolves the future first; an unfinished read is never dropped.
    assert calls == [("read", per_operation_s + transfer.RDMA_GET_MARGIN_S)]
    assert outer["timeout_s"] == pytest.approx(
        2 * per_operation_s + transfer.RDMA_GET_MARGIN_S
        + transfer.RDMA_READ_OUTER_MARGIN_S
    )
    assert len(transfer._RDMA_POISONED_OWNERS) == 1
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.phase == "failed latent read cleanup"
    assert owner.parts[0]["buffer"] is buffer
    assert owner.keepalive.numel() == 1


def test_on_loop_tokenized_outer_budget_covers_both_ack_attempts(monkeypatch):
    import asyncio

    generation = 11
    token = "a" * 32
    primary = RuntimeError("first ACK timed out")
    retry = RuntimeError("second ACK timed out")
    per_operation_s = 0.125
    ack_timeout_s = 0.25
    monkeypatch.setattr(transfer, "RDMA_READ_TIMEOUT_S", per_operation_s)
    monkeypatch.setattr(
        transfer.rdma_ownership, "RDMA_ACK_TIMEOUT_S", ack_timeout_s)
    real_off_loop = transfer.run_blocking_off_loop
    outer = {}

    def capture_outer(guard, operation, timeout_s, thread_name):
        outer.update(timeout_s=timeout_s(), thread_name=thread_name)
        return real_off_loop(
            guard, operation, timeout_s=timeout_s, thread_name=thread_name)

    monkeypatch.setattr(transfer, "run_blocking_off_loop", capture_outer)

    class Handle:
        world = 1

        def __init__(self):
            self.lock = threading.RLock()
            self.sample_leases = {generation: 1}
            self.abandoned_sample_leases = {}
            self.deferred_supervision_error = None
            self.calls = []

        def call_all(self, endpoint, setup_generation, owner_token, timeout_s):
            self.calls.append(
                (endpoint, setup_generation, owner_token, timeout_s))
            raise primary if len(self.calls) == 1 else retry

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    buffer = _LoopRefusingBuffer(torch.arange(4, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [4],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 4}],
        "owner_token": token, "setup_generation": generation,
    }

    async def on_loop():
        with pytest.raises(RuntimeError) as exc_info:
            read_latent_result(desc, guard)
        return exc_info.value

    assert asyncio.run(on_loop()) is primary
    assert handle.calls == [
        ("ack_latent_handoff", generation, token, ack_timeout_s),
        ("ack_latent_handoff", generation, token, ack_timeout_s),
    ]
    assert outer == {
        "timeout_s": pytest.approx(
            2 * per_operation_s + transfer.RDMA_GET_MARGIN_S
            + ack_timeout_s * transfer.rdma_ownership.RDMA_ACK_ATTEMPTS
            + transfer.RDMA_READ_OUTER_MARGIN_S),
        "thread_name": "dgxm-latent-read",
    }
    assert buffer.dropped == 1 and guard.readers == 0


def test_on_loop_rdma_release_failure_follows_successful_read():
    """A successful off-loop read still reports a failed registration drop."""
    import asyncio

    class DropFailure(RuntimeError):
        pass

    class _LoopRefusingDropFailureFuture:
        def get(self, timeout=None):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                raise DropFailure("drop failed") from None
            raise AssertionError("RDMA futures must be driven off the event loop")

    class _ReleaseFailingBuffer(_LoopRefusingBuffer):
        destination_ref = None

        def read_into(self, dst, timeout=None):
            root = dst
            while isinstance(getattr(root, "_base", None), torch.Tensor):
                root = root._base
            self.destination_ref = weakref.ref(root)
            return super().read_into(dst, timeout=timeout)

        def drop(self):
            self.dropped += 1
            return _LoopRefusingDropFailureFuture()

    src = torch.arange(8, dtype=torch.uint8)
    buffer = _ReleaseFailingBuffer(src)
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [8],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 8}],
    }

    async def _on_loop():
        with pytest.raises(RuntimeError, match="read completed but 1 buffer") as exc_info:
            read_latent_result(desc)
        assert isinstance(exc_info.value.__cause__, DropFailure)
        assert buffer.destination_ref is not None
        gc.collect()
        assert buffer.destination_ref() is None

    asyncio.run(_on_loop())
    assert buffer.dropped == 1
    assert len(transfer._RDMA_POISONED_OWNERS) == 1
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.phase == "latent read release"
    assert owner.parts[0]["buffer"] is buffer
    assert owner.keepalive is None


def test_late_on_loop_rdma_thread_keeps_read_lease_until_exit(monkeypatch):
    """An outer join timeout cannot authorize teardown over a live reader."""
    import asyncio

    from dgx_monarch import mesh_setup

    entered = threading.Event()
    release = threading.Event()

    class BlockingFuture:
        def get(self, timeout=None):
            entered.set()
            assert release.wait(timeout=2.0)
            return 1

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def read_into(self, dst, timeout=None):
            dst.fill_(7)
            return BlockingFuture()

        def drop(self):
            return ImmediateFuture()

    handle = SimpleNamespace(
        lock=threading.RLock(),
        sample_leases={1: 1},
        abandoned_sample_leases={},
        deferred_supervision_error=None,
    )
    guard = mesh_setup.SetupBoundFuture(object(), handle, 1)
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": Buffer(), "offset": 0, "nbytes": 1}],
    }
    monkeypatch.setattr(transfer, "RDMA_READ_TIMEOUT_S", 0.01)
    monkeypatch.setattr(transfer, "RDMA_GET_MARGIN_S", 0.01)
    monkeypatch.setattr(transfer, "RDMA_READ_OUTER_MARGIN_S", 0.01)
    baseline = rdma_read_job.pending_job_count()

    async def _on_loop():
        with pytest.raises(TimeoutError, match="dgxm-latent-read"):
            read_latent_result(desc, guard)

    asyncio.run(_on_loop())
    assert entered.is_set() and guard.readers == 2
    guard.abandon()
    assert guard.retire_to == "abandoned"
    assert handle.sample_leases == {1: 1}
    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        mesh_setup.require_no_sample_leases(
            handle, "recycle the worker fleet", allow_abandoned=True)

    release.set()
    deadline = time.monotonic() + 2.0
    while (guard.state == "active"
           or rdma_read_job.pending_job_count() != baseline):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.005)

    assert guard.state == "abandoned" and guard.readers == 0
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert rdma_read_job.pending_job_count() == baseline
    mesh_setup.require_no_sample_leases(
        handle, "recycle the worker fleet", allow_abandoned=True)


def test_late_drop_keeps_read_lease_until_native_release_finishes(monkeypatch):
    """A successful read cannot authorize recycle while drop is still live."""
    import asyncio

    from dgx_monarch import mesh_setup

    drop_entered = threading.Event()
    release_drop = threading.Event()

    class ImmediateFuture:
        def get(self, timeout=None):
            return 1

    class BlockingDropFuture:
        def get(self, timeout=None):
            drop_entered.set()
            assert release_drop.wait(timeout=2.0)

    class Buffer:
        def read_into(self, dst, timeout=None):
            dst.fill_(9)
            return ImmediateFuture()

        def drop(self):
            return BlockingDropFuture()

    handle = SimpleNamespace(
        lock=threading.RLock(),
        sample_leases={1: 1},
        abandoned_sample_leases={},
        deferred_supervision_error=None,
    )
    guard = mesh_setup.SetupBoundFuture(object(), handle, 1)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": Buffer(), "offset": 0, "nbytes": 1}],
    }
    monkeypatch.setattr(transfer, "RDMA_READ_TIMEOUT_S", 0.01)
    monkeypatch.setattr(transfer, "RDMA_GET_MARGIN_S", 0.01)
    monkeypatch.setattr(transfer, "RDMA_READ_OUTER_MARGIN_S", 0.01)
    baseline = rdma_read_job.pending_job_count()

    async def on_loop():
        with pytest.raises(TimeoutError, match="dgxm-latent-read"):
            read_latent_result(desc, guard)

    asyncio.run(on_loop())
    assert drop_entered.is_set() and guard.readers == 2
    guard.abandon()
    assert guard.retire_to == "abandoned"
    assert handle.sample_leases == {1: 1}
    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        mesh_setup.require_no_sample_leases(
            handle, "recycle the worker fleet", allow_abandoned=True)

    release_drop.set()
    deadline = time.monotonic() + 2.0
    while (guard.state == "active"
           or rdma_read_job.pending_job_count() != baseline):
        if time.monotonic() >= deadline:
            break
        time.sleep(0.005)
    assert guard.state == "abandoned" and guard.readers == 0
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert rdma_read_job.pending_job_count() == baseline
    mesh_setup.require_no_sample_leases(
        handle, "recycle the worker fleet", allow_abandoned=True)


def test_thread_start_before_apply_recovers_one_unread_settlement(monkeypatch):
    """A start that fails before its thread runs gets one recovery winner and no read."""
    import asyncio

    class StartInterrupted(BaseException):
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

    generation = 1
    registry = HandoffRegistry()
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(parts=[], keepalive=object(), state="registered")
    registry.mark_ready(handoff)
    token = handoff["token"]

    class Handle:
        world = 1

        def __init__(self):
            self.lock = threading.RLock()
            self.sample_leases = {generation: 1}
            self.abandoned_sample_leases = {}
            self.deferred_supervision_error = None
            self.calls = []
            self.guard = None

        def call_all(self, _endpoint, setup_generation, token, timeout_s):
            assert self.guard.readers >= 1
            self.calls.append((setup_generation, token, timeout_s))
            return [{
                "setup_generation": setup_generation,
                "token": token,
                "status": registry.acknowledge(setup_generation, token),
                "rank": 0,
            }]

    handle = Handle()
    guard = SetupBoundFuture(None, handle, generation)
    handle.guard = guard
    buffer = Buffer()
    actor_part = {"buffer": buffer, "offset": 0, "nbytes": 1}
    handoff["parts"].append(actor_part)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [dict(actor_part)], "owner_token": token,
        "setup_generation": generation,
    }
    boundary = StartInterrupted("interrupted before thread start")
    baseline = rdma_read_job.pending_job_count()
    real_start = threading.Thread.start
    starts = 0

    def fail_before_apply(self):
        nonlocal starts
        if self.name != "dgxm-latent-read":
            return real_start(self)
        starts += 1
        if starts == 1:
            raise boundary
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", fail_before_apply)

    async def on_loop():
        with pytest.raises(StartInterrupted) as exc_info:
            read_latent_result(desc, guard)
        return exc_info.value

    assert asyncio.run(on_loop()) is boundary
    assert starts == 2 and buffer.reads == 0 and buffer.drops == 1
    assert len(handle.calls) == 1 and guard.readers == 0
    assert registry.get(generation, token)["_registry_state"] == "RELEASED"
    assert registry.live_count == 0 and registry.tombstone_count == 1
    assert registry.acknowledge(generation, token) == "already_released"
    assert buffer.drops == 1
    assert rdma_read_job.pending_job_count() == baseline


def test_thread_start_applied_then_raises_keeps_one_normal_winner(monkeypatch):
    """If the first start runs its thread and then raises, the two runners together read and drop once."""
    import asyncio

    class StartInterrupted(BaseException):
        pass

    entered = threading.Event()
    release = threading.Event()

    class Future:
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
            dst.fill_(3)
            return Future()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    generation = 3
    registry = HandoffRegistry()
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(parts=[], keepalive=object(), state="registered")
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
    buffer = Buffer()
    actor_part = {"buffer": buffer, "offset": 0, "nbytes": 1}
    handoff["parts"].append(actor_part)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [dict(actor_part)], "owner_token": token,
        "setup_generation": generation,
    }
    boundary = StartInterrupted("interrupted after thread start applied")
    baseline = rdma_read_job.pending_job_count()
    real_start = threading.Thread.start
    starts = 0

    def start_then_raise(self):
        nonlocal starts
        if self.name != "dgxm-latent-read":
            return real_start(self)
        starts += 1
        result = real_start(self)
        if starts == 1:
            assert entered.wait(2.0)
            raise boundary
        release.set()
        return result

    monkeypatch.setattr(threading.Thread, "start", start_then_raise)

    async def on_loop():
        with pytest.raises(StartInterrupted) as exc_info:
            read_latent_result(desc, guard)
        return exc_info.value

    try:
        assert asyncio.run(on_loop()) is boundary
    finally:
        release.set()
    assert starts == 2 and buffer.reads == buffer.drops == 1
    assert handle.ack_readers == [1] and guard.readers == 0
    assert registry.get(generation, token)["_registry_state"] == "RELEASED"
    assert registry.live_count == 0 and registry.tombstone_count == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.parametrize("reuse_error", [False, True])
def test_join_error_chaining_handles_distinct_and_reused_objects(
    monkeypatch, reuse_error,
):
    import asyncio

    cancellation = KeyboardInterrupt("scratch join cancelled")
    ordinary = (cancellation if reuse_error else
                RuntimeError("first thread start returned an ordinary error"))
    starts = 0
    real_start = threading.Thread.start

    def start_then_raise(self):
        nonlocal starts
        if self.name != "dgxm-latent-read":
            return real_start(self)
        starts += 1
        result = real_start(self)
        if starts == 1:
            self.join(2.0)
            assert not self.is_alive()
            raise ordinary
        return result

    def cancel_join(runner, _timeout_s, _thread_name):
        runner.join(2.0)
        assert not runner.is_alive()
        raise cancellation

    monkeypatch.setattr(threading.Thread, "start", start_then_raise)
    monkeypatch.setattr(rdma_read_driver, "_join_or_timeout", cancel_join)
    buffer = _FakeRDMABuffer(torch.arange(1, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    baseline = rdma_read_job.pending_job_count()

    async def on_loop():
        with pytest.raises(KeyboardInterrupt) as exc_info:
            read_latent_result(desc)
        return exc_info.value

    error = asyncio.run(on_loop())
    assert error is cancellation
    if reuse_error:
        assert error.__cause__ is not error
        assert error.__context__ is not error
    else:
        assert error.__cause__ is ordinary
    assert error.__cause__ is not error
    assert starts == 2 and buffer.dropped == 1
    assert rdma_read_job.pending_job_count() == baseline


@pytest.mark.parametrize("reuse_error", [False, True])
def test_both_starts_retain_root_without_caller_native_or_self_chain(
    monkeypatch, reuse_error,
):
    import asyncio

    from dgx_monarch import mesh_setup

    second = KeyboardInterrupt("recovery start failed before apply")
    first = (second if reuse_error else
             RuntimeError("original start failed before apply"))
    events = []

    class LoopRefusingFuture:
        def get(self, timeout=None):
            events.append(("get", threading.get_ident()))
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return None
            raise AssertionError("native future get ran on the event-loop caller")

    class Buffer:
        def __init__(self):
            self.reads = self.drops = 0

        def read_into(self, _dst, timeout=None):
            self.reads += 1
            events.append(("read", threading.get_ident()))
            return LoopRefusingFuture()

        def drop(self):
            self.drops += 1
            events.append(("drop", threading.get_ident()))
            return LoopRefusingFuture()

    monkeypatch.setattr(rdma_read_job._registry, "_JOBS", [])

    starts = 0
    real_start = threading.Thread.start

    def fail_both(self):
        nonlocal starts
        if self.name != "dgxm-latent-read":
            return real_start(self)
        starts += 1
        raise first if starts == 1 else second

    monkeypatch.setattr(threading.Thread, "start", fail_both)
    generation = 23
    buffer = Buffer()
    context = _real_pending_rdma(monkeypatch, buffer, generation)
    caller_thread = None

    async def on_loop():
        nonlocal caller_thread
        caller_thread = threading.get_ident()
        with pytest.raises(KeyboardInterrupt) as exc_info:
            context.pending.result()
        return exc_info.value

    error = asyncio.run(on_loop())
    assert error is second
    if reuse_error:
        assert error.__cause__ is not error
        assert error.__context__ is not error
    else:
        assert error.__cause__ is first
    assert error.__cause__ is not error
    assert starts == 2 and events == []
    assert buffer.reads == buffer.drops == 0
    assert caller_thread is not None
    assert context.ack_calls == []
    assert context.pending._state == "closed" and context.closed == [True]
    assert context.guard.begins == context.guard.readers == 1
    assert context.guard.state == "active"
    assert context.guard.retire_to == "abandoned"
    assert context.handle.sample_leases == {generation: 1}
    assert context.handle.abandoned_sample_leases == {}
    retained = context.registry.get(generation, context.token)
    assert retained["_registry_state"] == "READY"
    assert retained["parts"][0] is context.actor_part
    assert retained["keepalive"] is context.keepalive
    assert context.registry.live_count == 1
    assert context.registry.tombstone_count == 0
    assert rdma_read_job.pending_job_states() == ("PENDING",)
    with pytest.raises(mesh_setup.LifecycleBusyError, match="unresolved sample"):
        mesh_setup.require_no_sample_leases(
            context.handle, "recycle the worker fleet", allow_abandoned=True)


def test_settle_recovery_cancellation_is_returned_exactly(monkeypatch):
    ordinary = RuntimeError("ordinary settlement failure")
    cancellation = KeyboardInterrupt("fail-closed cancelled after entry")
    calls = 0

    def fail_settle(parts, keepalive, phase, outcome_owner=None):
        raise ordinary

    def recover(parts, keepalive, phase, primary):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise cancellation
        return [], None

    monkeypatch.setattr(transfer, "_settle_parts", fail_settle)
    monkeypatch.setattr(transfer, "_fail_closed", recover)

    failures, poison_error, cleanup_error = transfer._settle_or_fail_closed(
        [], None, "test settlement")

    assert failures == [] and poison_error is None
    assert cleanup_error is cancellation and calls == 2
