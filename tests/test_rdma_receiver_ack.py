"""Receiver acknowledgement and release: rdma_receiver._ack_owner and _read_and_release_owned."""
import dis
import gc
import inspect
import sys
import threading
import weakref
from types import CodeType, SimpleNamespace

import pytest
import torch

import dgx_monarch.rdma_read_job as rdma_read_job
import dgx_monarch.rdma_receiver as rdma_receiver
import dgx_monarch.transfer as transfer
from dgx_monarch.mesh_lease import SetupBoundFuture
from dgx_monarch.rdma_ownership import HandoffRegistry
from dgx_monarch.transfer import read_latent_result
from transfer_helpers import (  # noqa: F401  # autouse fixture import.
    _call_instruction_offset,
    _FakeFuture,
    _FakeRDMABuffer,
    _isolate_process_lifetime_rdma_poison,
    _record_unlocked_poison_errors,
    _run_at_instruction,
)


def test_owner_ack_runs_after_read_drop_and_authority_exit():
    token = "a" * 32
    events = []

    class Buffer(_FakeRDMABuffer):
        def read_into(self, dst, timeout=None):
            events.append("read")
            return super().read_into(dst, timeout=timeout)

        def drop(self):
            events.append("drop")
            return super().drop()

    class Authority:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_args):
            events.append("exit")

    class Guard:
        def read_authority(self):
            return Authority()

        def ack_latent(self, generation, owner_token):
            assert events == ["enter", "read", "drop", "exit"]
            events.append(("ack", generation, owner_token))

    source = torch.arange(8, dtype=torch.uint8)
    buffer = Buffer(source)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [8],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 8}],
        "owner_token": token, "setup_generation": 7,
    }

    assert torch.equal(read_latent_result(desc, Guard()), source)
    assert events == ["enter", "read", "drop", "exit", ("ack", 7, token)]
    assert buffer.dropped == 1


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_authority_exit_failure_never_acknowledges_sender(error_type):
    generation = 2
    authority_error = error_type("authority exit failed")
    registry = HandoffRegistry()
    backing = torch.arange(4, dtype=torch.uint8)
    buffer = _FakeRDMABuffer(backing)
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[{"buffer": buffer, "offset": 0, "nbytes": 4}],
        keepalive=backing, state="registered")
    registry.mark_ready(handoff)
    token = handoff["token"]
    ack_readers = []
    handle = SimpleNamespace(
        world=1, lock=threading.RLock(), sample_leases={generation: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)

    class Guard(SetupBoundFuture):
        def __init__(self, *args):
            super().__init__(*args)
            self.inner_token = None
            self.inner_end_attempts = 0

        def _begin_read_token(self, read_token):
            result = super()._begin_read_token(read_token)
            if self.inner_token is None and self.readers == 2:
                self.inner_token = read_token
            return result

        def _end_read_token(self, read_token):
            if read_token is self.inner_token and self.inner_end_attempts < 4:
                self.inner_end_attempts += 1
                raise authority_error
            return super()._end_read_token(read_token)

    guard = Guard(None, handle, generation)

    def call_all(_endpoint, setup_generation, owner_token, timeout_s):
        ack_readers.append(guard.readers)
        return [{
            "setup_generation": setup_generation,
            "token": owner_token,
            "status": registry.acknowledge(setup_generation, owner_token),
            "rank": 0,
        }]

    handle.call_all = call_all
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [4],
        "parts": [dict(handoff["parts"][0])],
        "owner_token": token, "setup_generation": generation,
    }

    with pytest.raises(error_type) as exc_info:
        read_latent_result(desc, guard)
    assert exc_info.value is authority_error
    assert authority_error.__cause__ is not authority_error
    assert authority_error.__context__ is not authority_error
    assert buffer.dropped == 1
    assert guard.inner_end_attempts == 4
    assert guard.readers == 1 and guard.inner_token in guard._reader_tokens
    assert ack_readers == []
    retained = registry.get(generation, token)
    assert retained["_registry_state"] == "READY"
    assert retained["parts"] and retained["keepalive"] is backing


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize(
    "ordering", ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"])
def test_body_and_authority_exit_failures_never_false_confirm_reader_retirement(
    ordering,
):
    ordinary = RuntimeError("ordinary read-authority failure")
    cancellation = KeyboardInterrupt("read body or authority cancelled")
    body_error, cleanup_error = {
        "ordinary-cancellation": (ordinary, cancellation),
        "cancellation-ordinary": (cancellation, ordinary),
        "same-cancellation": (cancellation, cancellation),
    }[ordering]
    generation = 3
    registry = HandoffRegistry()
    backing = torch.arange(2, dtype=torch.uint8)
    buffer = _FakeRDMABuffer(backing)
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[{"buffer": buffer, "offset": 0, "nbytes": 2}],
        keepalive=backing, state="registered")
    registry.mark_ready(handoff)
    token = handoff["token"]
    ack_readers = []
    handle = SimpleNamespace(
        world=1, lock=threading.RLock(), sample_leases={generation: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)

    class Guard(SetupBoundFuture):
        def __init__(self, *args):
            super().__init__(*args)
            self.inner_token = None
            self.inner_end_attempts = 0

        def _begin_read_token(self, read_token):
            result = super()._begin_read_token(read_token)
            if self.inner_token is None and self.readers == 2:
                self.inner_token = read_token
            return result

        def _end_read_token(self, read_token):
            if read_token is self.inner_token and self.inner_end_attempts < 4:
                self.inner_end_attempts += 1
                raise cleanup_error
            return super()._end_read_token(read_token)

    guard = Guard(None, handle, generation)

    def call_all(_endpoint, setup_generation, owner_token, timeout_s):
        ack_readers.append(guard.readers)
        return [{
            "setup_generation": setup_generation,
            "token": owner_token,
            "status": registry.acknowledge(setup_generation, owner_token),
            "rank": 0,
        }]

    handle.call_all = call_all
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [dict(handoff["parts"][0])],
        "owner_token": token, "setup_generation": generation,
    }
    drive_code = next(
        constant for constant in rdma_receiver._read_and_release_owned.__code__.co_consts
        if isinstance(constant, CodeType) and constant.co_name == "drive")
    target = _call_instruction_offset(drive_code, "finish", successor=True)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        _run_at_instruction(
            drive_code, target, body_error,
            lambda: read_latent_result(desc, guard))

    assert exc_info.value is cancellation
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation
    assert buffer.dropped == 1
    assert guard.inner_end_attempts == 4
    assert guard.readers == 1 and guard.inner_token in guard._reader_tokens
    assert ack_readers == []
    retained = registry.get(generation, token)
    assert retained["_registry_state"] == "READY"
    assert retained["parts"] and retained["keepalive"] is backing


def test_owner_ack_retry_never_repeats_native_drop():
    token = "c" * 32
    calls = []

    class Guard:
        def ack_latent(self, generation, owner_token):
            calls.append((generation, owner_token))
            if len(calls) == 1:
                raise RuntimeError("reply lost")

    source = torch.arange(4, dtype=torch.uint8)
    buffer = _FakeRDMABuffer(source)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [4],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 4}],
        "owner_token": token, "setup_generation": 3,
    }

    assert torch.equal(read_latent_result(desc, Guard()), source)
    assert calls == [(3, token), (3, token)]
    assert buffer.dropped == 1


def test_drop_failure_never_acknowledges_sender(monkeypatch):
    poison_errors = _record_unlocked_poison_errors(monkeypatch)

    class DropFuture:
        def get(self, timeout=None):
            raise RuntimeError("drop outcome unknown")

    class Buffer(_FakeRDMABuffer):
        def drop(self):
            self.dropped += 1
            return DropFuture()

    class Guard:
        def __init__(self):
            self.acks = []

        def ack_latent(self, *identity):
            self.acks.append(identity)

    guard = Guard()
    buffer = Buffer(torch.arange(4, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [4],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 4}],
        "owner_token": "d" * 32, "setup_generation": 4,
    }

    with pytest.raises(RuntimeError, match="read completed but 1 buffer"):
        read_latent_result(desc, guard)
    assert buffer.dropped == 1
    assert guard.acks == []
    assert len(poison_errors) == 1
    assert poison_errors[0][0] is False
    assert "ownership POISONED" in poison_errors[0][1]


def test_late_read_settlement_reports_poison_after_outer_timeout(monkeypatch):
    poison_errors = _record_unlocked_poison_errors(monkeypatch)
    read_started = threading.Event()
    release_read = threading.Event()
    read_finished = threading.Event()
    runner_errors: list[BaseException] = []
    runners: list[threading.Thread] = []
    part = {"buffer": object(), "offset": 0, "nbytes": 1}

    def late_read(*_args, **_kwargs):
        read_started.set()
        if not release_read.wait(timeout=5):
            raise RuntimeError("late read test timed out")
        failure = transfer._drop_failure(
            part, RuntimeError("late settlement drop failed"))
        transfer._poison_failed_drops(
            [failure], keepalive=object(), phase="late read settlement")
        raise RuntimeError("late read settlement failed")

    def timeout_before_settlement(_guard, operation, **_kwargs):
        def run():
            try:
                operation(None, None, lambda: None, lambda *_a, **_k: None)
            except BaseException as exc:
                runner_errors.append(exc)
            finally:
                read_finished.set()

        runner = threading.Thread(target=run)
        runners.append(runner)
        runner.start()
        assert read_started.wait(timeout=2)
        raise TimeoutError("outer read timed out")

    monkeypatch.setattr(transfer, "_read_and_release", late_read)
    monkeypatch.setattr(
        transfer, "run_blocking_off_loop", timeout_before_settlement)
    descriptor = {
        "kind": "rdma", "dtype": "uint8", "shape": [1], "parts": [part],
    }

    try:
        with pytest.raises(TimeoutError, match="outer read timed out"):
            read_latent_result(descriptor)
        assert poison_errors == []
    finally:
        release_read.set()

    assert read_finished.wait(timeout=2)
    runners[0].join(timeout=2)
    assert not runners[0].is_alive()
    assert len(runner_errors) == 1
    assert len(poison_errors) == 1
    assert poison_errors[0][0] is False
    assert "ownership POISONED" in poison_errors[0][1]


@pytest.mark.parametrize(
    "ordering", ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"])
def test_drop_cleanup_promotes_exact_cancellation_after_poison_ownership(
    monkeypatch, ordering,
):
    class CleanupCancellation(KeyboardInterrupt):
        def __bool__(self):
            raise AssertionError("exception truthiness must not be evaluated")

    ordinary = RuntimeError("ordinary receiver failure")
    cancellation = CleanupCancellation("drop cleanup cancelled")
    primary, drop_error = {
        "ordinary-cancellation": (ordinary, cancellation),
        "cancellation-ordinary": (cancellation, ordinary),
        "same-cancellation": (cancellation, cancellation),
    }[ordering]

    class DropFuture:
        def get(self, timeout=None):
            raise drop_error

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return DropFuture()

    def fail_empty(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(rdma_receiver.torch, "empty", fail_empty)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(CleanupCancellation) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is cancellation
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation
    if ordering != "same-cancellation":
        assert cancellation.__cause__ is ordinary
    assert buffer.dropped == 1
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.parts[0]["buffer"] is buffer


def test_completed_read_drop_cancellation_outranks_synthesized_failure():
    cancellation = KeyboardInterrupt("completed-read drop cancelled")

    class DropFuture:
        def get(self, timeout=None):
            raise cancellation

    class Buffer(_FakeRDMABuffer):
        def drop(self):
            self.dropped += 1
            return DropFuture()

    backing = torch.arange(2, dtype=torch.uint8)
    buffer = Buffer(backing)
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 2}],
    }

    with pytest.raises(KeyboardInterrupt) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is cancellation
    assert isinstance(cancellation.__cause__, RuntimeError)
    assert "read completed" in str(cancellation.__cause__)
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation
    assert buffer.dropped == 1
    assert transfer._RDMA_POISONED_OWNERS[0].parts[0]["buffer"] is buffer


@pytest.mark.parametrize(
    "ordering", ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"])
def test_poison_cleanup_promotes_exact_cancellation_after_durable_publication(
    monkeypatch, ordering,
):
    ordinary = RuntimeError("ordinary poison cleanup failure")
    cancellation = KeyboardInterrupt("poison cleanup cancelled")
    primary, poison_error = {
        "ordinary-cancellation": (ordinary, cancellation),
        "cancellation-ordinary": (cancellation, ordinary),
        "same-cancellation": (cancellation, cancellation),
    }[ordering]
    drop_error = RuntimeError("drop outcome unknown")

    class DropFuture:
        def get(self, timeout=None):
            raise drop_error

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return DropFuture()

    def fail_empty(*_args, **_kwargs):
        raise primary

    real_poison = transfer._poison_failed_drops
    publications = 0

    def publish_then_fail(*args, **kwargs):
        nonlocal publications
        publications += 1
        real_poison(*args, **kwargs)
        raise poison_error

    monkeypatch.setattr(rdma_receiver.torch, "empty", fail_empty)
    monkeypatch.setattr(transfer, "_poison_failed_drops", publish_then_fail)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(KeyboardInterrupt) as exc_info:
        read_latent_result(desc)

    assert exc_info.value is cancellation
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation
    assert publications == 1 and buffer.dropped == 1
    assert transfer._RDMA_POISONED_OWNERS[0].parts[0]["buffer"] is buffer
    if poison_error is ordinary:
        assert any("ordinary poison cleanup failure" in note
                   for note in cancellation.__notes__)


@pytest.mark.parametrize(
    ("token", "generation"),
    [("e" * 32, None), (None, 1), ("short", 1), ("E" * 32, 1),
     ("e" * 32, True), ("e" * 32, 0)],
)
def test_malformed_owner_metadata_is_never_acknowledged(token, generation):
    class Guard:
        def ack_latent(self, *identity):
            raise AssertionError(f"malformed metadata was ACKed: {identity!r}")

    buffer = _FakeRDMABuffer(torch.arange(2, dtype=torch.uint8))
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 2}],
        "owner_token": token, "setup_generation": generation,
    }
    with pytest.raises(RuntimeError, match="invalid owner metadata"):
        read_latent_result(desc, Guard())
    assert buffer.dropped == 1


@pytest.mark.parametrize("stage", ["preparation", "authority factory", "authority enter"])
def test_never_started_read_only_acks_before_authority_acquisition(
    monkeypatch, stage,
):
    token = "f" * 32
    primary = RuntimeError(f"{stage} failed")
    calls = []

    class Authority:
        def __enter__(self):
            if stage == "authority enter":
                raise primary

        def __exit__(self, *_args):
            return None

    class Guard:
        def read_authority(self):
            if stage == "authority factory":
                raise primary
            return Authority()

        def ack_latent(self, generation, owner_token):
            calls.append((generation, owner_token))

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    buffer = Buffer()
    if stage == "preparation":
        monkeypatch.setattr(
            rdma_receiver.torch, "empty",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(primary))
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
        "owner_token": token, "setup_generation": 5,
    }

    with pytest.raises(RuntimeError) as exc_info:
        read_latent_result(desc, Guard())
    assert exc_info.value is primary
    assert buffer.dropped == 1
    assert calls == ([(5, token)] if stage == "preparation" else [])


def test_pre_read_ack_failure_preserves_primary_and_does_not_redrop():
    calls = []

    class Guard:
        def ack_latent(self, *identity):
            calls.append(identity)
            raise RuntimeError("ACK unavailable")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "missing_dtype", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
        "owner_token": "1" * 32, "setup_generation": 6,
    }
    with pytest.raises(AttributeError) as exc_info:
        read_latent_result(desc, Guard())
    assert buffer.dropped == 1
    assert calls == [(6, "1" * 32), (6, "1" * 32)]
    assert any(
        "ACK during failure cleanup" in note for note in exc_info.value.__notes__)


def test_receiver_cleanup_entry_interruption_preserves_read_and_owns_parts():
    class ReadFailure(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = ReadFailure("read failed first")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def read_into(self, _dst, timeout=None):
            raise primary

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    source, start = inspect.getsourcelines(rdma_receiver._read_and_release_owned)
    target = next(
        start + offset for offset, line in enumerate(source)
        if "failures, poison_error = fail_closed(parts, keepalive, phase, primary)" in line
    )
    cleanup_stop = CleanupStop("interrupted before cleanup helper entry")

    def stop_cleanup_entry(frame, event, _arg):
        if (event == "line" and frame.f_code.co_name == "finish"
                and frame.f_lineno == target):
            raise cleanup_stop
        return stop_cleanup_entry

    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    sys.settrace(stop_cleanup_entry)
    with pytest.raises(ReadFailure) as exc_info:
        try:
            read_latent_result(desc)
        finally:
            sys.settrace(None)

    assert exc_info.value is primary
    assert buffer.dropped == 0
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.parts[0]["buffer"] is buffer
    assert any("cleanup helper entry" in note for note in primary.__notes__)


def test_read_failure_stays_primary_when_authority_exit_also_fails():
    class ReadFailure(RuntimeError):
        pass

    class AuthorityFailure(RuntimeError):
        pass

    primary = ReadFailure("read failed")
    authority_failure = AuthorityFailure("authority exit failed")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def read_into(self, _dst, timeout=None):
            raise primary

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    class Authority:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            raise authority_failure

    class Guard:
        def read_authority(self):
            return Authority()

    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(ReadFailure) as exc_info:
        read_latent_result(desc, Guard())

    assert exc_info.value is primary
    assert buffer.dropped == 0
    assert transfer._RDMA_POISONED_OWNERS[0].keepalive is not None
    assert any("authority exit failed" in note for note in primary.__notes__)


def test_read_failure_cannot_be_suppressed_by_foreign_authority():
    primary = RuntimeError("read failed")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def read_into(self, _dst, timeout=None):
            raise primary

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    class SuppressingAuthority:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return True

    class Guard:
        def read_authority(self):
            return SuppressingAuthority()

    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(RuntimeError) as exc_info:
        read_latent_result(desc, Guard())

    assert exc_info.value is primary
    assert buffer.dropped == 0
    assert transfer._RDMA_POISONED_OWNERS[0].keepalive is not None


def test_pre_read_authority_failure_does_not_retain_destination(monkeypatch):
    class DropFailureFuture:
        def get(self, timeout=None):
            raise RuntimeError("drop failed")

    primary = RuntimeError("authority refused")

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def drop(self):
            self.dropped += 1
            return DropFailureFuture()

    class Guard:
        def read_authority(self):
            raise primary

    real_empty = torch.empty
    destinations = []

    def capture_empty(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        destinations.append(weakref.ref(out))
        return out

    monkeypatch.setattr(transfer.torch, "empty", capture_empty)
    buffer = Buffer()
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }

    with pytest.raises(RuntimeError) as exc_info:
        read_latent_result(desc, Guard())

    assert exc_info.value is primary
    assert buffer.dropped == 1
    assert transfer._RDMA_POISONED_OWNERS[0].keepalive is None
    gc.collect()
    assert destinations[0]() is None


def test_poison_owner_does_not_retain_successfully_dropped_sibling():
    class DropFailureFuture:
        def get(self, timeout=None):
            raise RuntimeError("first drop failed")

    class DropFailingBuffer(_FakeRDMABuffer):
        def drop(self):
            self.dropped += 1
            return DropFailureFuture()

    failed = DropFailingBuffer(torch.arange(4, dtype=torch.uint8))
    successful = _FakeRDMABuffer(torch.arange(4, 8, dtype=torch.uint8))
    successful_ref = weakref.ref(successful)
    desc = {
        "kind": "rdma",
        "dtype": "uint8",
        "shape": [8],
        "parts": [
            {"buffer": failed, "offset": 0, "nbytes": 4},
            {"buffer": successful, "offset": 4, "nbytes": 4},
        ],
    }

    with pytest.raises(RuntimeError, match="read completed but 1 buffer"):
        read_latent_result(desc)

    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert [part["buffer"] for part in owner.parts] == [failed]
    assert owner.keepalive is None
    assert owner.failures == ("RuntimeError('first drop failed')",)
    assert successful.dropped == 1
    del desc, successful
    gc.collect()
    assert successful_ref() is None


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize(
    "seam", [
        "entry-call", "entry-return", "settle-return",
        "drop-confirm-store", "drop-confirm-store-after",
        "lower-settle-return", "guard-return",
        "release-return", "drop-publish-return", "settlement-publish-before",
        "settlement-publish-after", "drop-owner-before", "drop-owner-after",
        "ack-entry", "ack-owner-call", "ack-owner-return",
        "ack-complete-before", "ack-complete-after"])
def test_receiver_handoff_entry_boundaries_settle_and_ack_once(seam):
    class StopNow(BaseException):
        pass

    class ImmediateFuture:
        def get(self, timeout=None):
            return None

    class Buffer:
        def __init__(self):
            self.reads = self.drops = 0

        def read_into(self, dst, timeout=None):
            self.reads += 1
            dst.fill_(5)
            return ImmediateFuture()

        def drop(self):
            self.drops += 1
            return ImmediateFuture()

    generation = 14
    registry = HandoffRegistry()
    buffer = Buffer()
    actor_part = {"buffer": buffer, "offset": 0, "nbytes": 1}
    keepalive = object()
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[actor_part], keepalive=keepalive, state="registered")
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
    if seam == "lower-settle-return":
        code = transfer._settle_or_fail_closed.__code__
        target = _call_instruction_offset(
            code, "_settle_parts", successor=True)
    elif seam == "guard-return":
        code = transfer._settle_parts.__code__
        target = _call_instruction_offset(
            code, "_release_rdma_parts_guarded", successor=True)
    elif seam == "release-return":
        code = transfer._release_rdma_parts_guarded.__code__
        target = _call_instruction_offset(
            code, "_release_rdma_parts", successor=True)
    elif seam == "drop-publish-return":
        code = transfer._release_rdma_parts.__code__
        target = _call_instruction_offset(
            code, "_publish_drop_result", successor=True)
    elif seam.startswith("settlement-publish"):
        code = transfer._publish_settlement.__code__
        instructions = list(dis.get_instructions(code))
        store = next(
            index for index, instruction in enumerate(instructions)
            if instruction.opname == "STORE_SUBSCR")
        target = instructions[
            store + int(seam.endswith("after"))].offset
    elif seam.startswith("drop-owner"):
        code = transfer._publish_drop_result.__code__
        instructions = list(dis.get_instructions(code))
        store = next(
            index for index, instruction in enumerate(instructions)
            if instruction.opname == "STORE_SUBSCR")
        target = instructions[
            store + int(seam.endswith("after"))].offset
    elif seam.startswith("ack-owner"):
        code = rdma_receiver._ack_owner.__code__
        target = _call_instruction_offset(
            code, "append", successor=seam.endswith("return"))
    else:
        owner_code = rdma_receiver._read_and_release_owned.__code__
        nested_name = (
            "enter_operation" if seam.startswith("entry")
            else "finish" if seam in {
                "settle-return", "drop-confirm-store",
                "drop-confirm-store-after"}
            else "acknowledge_preserving")
        code = next(
            const for const in owner_code.co_consts
            if isinstance(const, CodeType) and const.co_name == nested_name)
    if seam.startswith("entry"):
        target = _call_instruction_offset(
            code, "mark_operation_entry", after=1,
            successor=seam == "entry-return")
    elif seam == "settle-return":
        target = _call_instruction_offset(code, "settle", successor=True)
    elif seam.startswith("drop-confirm-store"):
        instructions = list(dis.get_instructions(code))
        key = next(
            index for index, instruction in enumerate(instructions)
            if instruction.opname == "LOAD_CONST"
            and instruction.argval == "drops_confirmed")
        store = next(
            index for index in range(key + 1, len(instructions))
            if instructions[index].opname == "STORE_SUBSCR")
        successor = int(seam.endswith("after"))
        assert instructions[key].argval == "drops_confirmed"
        assert instructions[store].opname == "STORE_SUBSCR"
        assert store + successor < len(instructions)
        target = instructions[store + successor].offset
    elif seam.startswith("ack-complete"):
        instructions = list(dis.get_instructions(code))
        key = next(
            index for index, instruction in enumerate(instructions)
            if instruction.opname == "LOAD_CONST"
            and instruction.argval == "ack_complete")
        store = next(
            index for index in range(key + 1, len(instructions))
            if instructions[index].opname == "STORE_SUBSCR")
        target = instructions[
            store + int(seam.endswith("after"))].offset
    elif seam in {
            "lower-settle-return", "guard-return", "release-return",
            "drop-publish-return", "settlement-publish-before",
            "settlement-publish-after", "drop-owner-before",
            "drop-owner-after", "ack-owner-call", "ack-owner-return"}:
        pass
    else:
        target = next(
            instruction.offset for instruction in dis.get_instructions(code)
            if instruction.opname not in {"COPY_FREE_VARS", "RESUME"})
    boundary = StopNow(f"receiver {seam} interrupted")
    baseline = rdma_read_job.pending_job_count()

    with pytest.raises(StopNow) as exc_info:
        _run_at_instruction(
            code, target, boundary,
            lambda: read_latent_result(desc, guard))

    assert exc_info.value is boundary
    assert buffer.reads == int(
        seam not in {"entry-call", "entry-return"})
    assert buffer.drops == 1
    assert handle.ack_readers == [1] and guard.readers == 0
    assert registry.get(generation, token)["_registry_state"] == "RELEASED"
    assert registry.live_count == 0 and registry.tombstone_count == 1
    assert rdma_read_job.pending_job_count() == baseline
