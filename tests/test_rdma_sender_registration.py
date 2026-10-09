"""Sender registration ownership in dgx_monarch.transfer.

The tests cover the rollback around _register_parts and descriptor publication, and the tokenless
keepalive window. When read_latent_result fails or is interrupted, a poison owner keeps every part,
and nothing drops or ACKs them.
"""
import dis
import inspect
import sys

import pytest
import torch

import dgx_monarch.transfer as transfer
import dgx_monarch.transfer_utils as transfer_utils
from dgx_monarch.transfer import LatentReturn, read_latent_result
from transfer_helpers import (  # noqa: F401  # autouse fixture import.
    _FakeFuture,
    _FakeRDMABuffer,
    _isolate_process_lifetime_rdma_poison,
    _record_unlocked_poison_errors,
)


def test_rdma_read_failure_retains_every_registration_without_dropping(monkeypatch):
    class FailingReadBuffer(_FakeRDMABuffer):
        def read_into(self, dst, timeout=None):
            raise RuntimeError("read failed")

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", FailingReadBuffer)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "3")
    desc = LatentReturn("rdma", min_bytes=1).pack(
        "samples", torch.arange(24, dtype=torch.uint8)
    )

    with pytest.raises(RuntimeError, match="read failed"):
        read_latent_result(desc)
    assert all(part["buffer"].dropped == 0 for part in desc["parts"])
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.phase == "failed latent read cleanup"
    assert list(owner.parts) == desc["parts"]
    assert owner.keepalive.numel() == 24


def test_partial_rdma_registration_is_rolled_back(monkeypatch):
    created: list[_FakeRDMABuffer] = []

    def second_registration_fails(data):
        if created:
            raise RuntimeError("registration failed")
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", second_registration_fails)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    tensor = torch.arange(24, dtype=torch.uint8)
    desc = LatentReturn("rdma", min_bytes=1).pack("samples", tensor)

    assert desc["kind"] == "message"
    assert len(created) == 1 and created[0].dropped == 1
    assert len(transfer._RDMA_POISONED_OWNERS) == 1
    intent = transfer._RDMA_POISONED_OWNERS[0]
    assert intent.phase == "RDMA buffer construction"
    assert intent.parts[0]["_buffer_owner"] == []
    assert intent.keepalive.data_ptr() == tensor.data_ptr()
    monkeypatch.setattr(
        transfer, "RDMABuffer",
        lambda _data: pytest.fail("poisoned process retried native registration"))
    assert LatentReturn("rdma", min_bytes=1).pack(
        "samples", tensor + 1)["kind"] == "message"


def test_post_constructor_drop_poison_logs_after_registration_lock_release(
    monkeypatch,
):
    poison_errors = _record_unlocked_poison_errors(monkeypatch)

    class Handoff(dict):
        def __setitem__(self, key, value):
            if key == "state" and value == "registered":
                raise RuntimeError("registration state publication failed")
            super().__setitem__(key, value)

    class DropFuture:
        def get(self, timeout=None):
            raise RuntimeError("registration rollback drop failed")

    class Buffer(_FakeRDMABuffer):
        def drop(self):
            self.dropped += 1
            return DropFuture()

    created: list[Buffer] = []

    def register(data):
        buffer = Buffer(data)
        created.append(buffer)
        return buffer

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    tensor = torch.arange(8, dtype=torch.uint8)

    result = LatentReturn("rdma", min_bytes=1).pack(
        "samples", tensor, handoff=Handoff())

    assert result["kind"] == "message"
    assert len(created) == 1 and created[0].dropped == 1
    assert len(transfer._RDMA_POISONED_OWNERS) == 1
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.phase == "registration rollback"
    assert owner.parts[0]["buffer"] is created[0]
    assert len(poison_errors) == 1
    assert poison_errors[0][0] is False
    assert "ownership POISONED" in poison_errors[0][1]


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_python_constructor_native_call_store_gap_latches_process_poison(
    monkeypatch, caplog,
):
    class ConstructorStop(BaseException):
        pass

    native_handles = []

    def create_native(_data):
        handle = object()
        native_handles.append(handle)
        return handle

    class PythonBuffer:
        def __init__(self, data):
            self._buffer = create_native(data)

    instructions = list(dis.get_instructions(PythonBuffer.__init__))
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_DEREF" and instruction.argval == "create_native")
    call = next(
        index for index in range(load, len(instructions))
        if instructions[index].opname == "CALL")
    target = instructions[call + 1].offset
    primary = ConstructorStop("native handle created before wrapper publication")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, PythonBuffer.__init__.__code__, 0)
            raise primary

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", PythonBuffer)
    tensor = torch.arange(8, dtype=torch.uint8)
    monitoring.use_tool_id(tool_id, "dgxm-constructor-intent-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, PythonBuffer.__init__.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(ConstructorStop) as exc_info:
        try:
            LatentReturn("rdma", min_bytes=1).pack("samples", tensor)
        finally:
            monitoring.set_local_events(tool_id, PythonBuffer.__init__.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is primary and len(native_handles) == 1
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.phase == "RDMA buffer construction"
    assert owner.parts[0]["_buffer_owner"] == []
    assert owner.keepalive.data_ptr() == tensor.data_ptr()
    poison_records = [
        record for record in caplog.records
        if "RDMA registration ownership POISONED" in record.getMessage()
    ]
    assert len(poison_records) == 1
    assert poison_records[0].levelname == "ERROR"
    assert "native registration outcome is unknown" in poison_records[0].getMessage()
    assert "Attached mesh reset required" in poison_records[0].getMessage()
    caplog.clear()
    assert LatentReturn("rdma", min_bytes=1).pack(
        "samples", tensor + 1)["kind"] == "message"
    assert len(native_handles) == 1
    assert caplog.records == []


def test_explosive_exception_truthiness_is_never_consulted(monkeypatch):
    class Primary(BaseException):
        bool_calls = 0

        def __bool__(self):
            self.bool_calls += 1
            raise AssertionError("exception truthiness was consulted")

    class CleanupBoundary(BaseException):
        pass

    primary = Primary("read failed")
    boundary = CleanupBoundary("cleanup entry interrupted")
    real_fail_closed = transfer._fail_closed
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise boundary
        return real_fail_closed(*args, **kwargs)

    class Buffer:
        dropped = 0

        def read_into(self, _dst, timeout=None):
            raise primary

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    monkeypatch.setattr(transfer, "_fail_closed", fail_once)
    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
    }
    with pytest.raises(Primary) as exc_info:
        read_latent_result(desc)
    assert exc_info.value is primary and primary.bool_calls == 0
    assert buffer.dropped == 0
    assert transfer._RDMA_POISONED_OWNERS[0].keepalive is not None


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_single_read_future_internal_call_store_gap_never_drops_or_acks():
    class ReadStop(BaseException):
        pass

    spawned = []

    class Future:
        def spawn_handle(self):
            spawned.append(object())
            return object()

        def get(self, timeout=None):
            status = self.spawn_handle()
            return status

    class Buffer:
        def __init__(self):
            self.dropped = 0

        def read_into(self, _dst, timeout=None):
            return Future()

        def drop(self):
            self.dropped += 1
            return _FakeFuture(None)

    class Guard:
        def ack_latent(self, *_identity):
            raise AssertionError("incomplete read was ACKed")

    instructions = list(dis.get_instructions(Future.get))
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_ATTR" and instruction.argval == "spawn_handle")
    call = next(index for index in range(load, len(instructions))
                if instructions[index].opname == "CALL")
    target = instructions[call + 1].offset
    primary = ReadStop("future status publication interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, Future.get.__code__, 0)
            raise primary

    buffer = Buffer()
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [{"buffer": buffer, "offset": 0, "nbytes": 1}],
        "owner_token": "6" * 32, "setup_generation": 1,
    }
    monitoring.use_tool_id(tool_id, "dgxm-read-future-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, Future.get.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(ReadStop) as exc_info:
        try:
            read_latent_result(desc, Guard())
        finally:
            monitoring.set_local_events(tool_id, Future.get.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is primary and len(spawned) == 1
    assert buffer.dropped == 0
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.parts[0]["buffer"] is buffer and owner.keepalive is not None


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_concurrent_executor_exit_gap_never_drops_or_acks(monkeypatch):
    class ExitStop(BaseException):
        pass

    class Complete:
        def result(self):
            return None

    class Executor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def submit(self, fn, part):
            fn(part)
            return Complete()

        def shutdown(self, wait=True):
            assert wait is True

        def __exit__(self, *_args):
            self.shutdown(wait=True)
            return False

    class Buffer(_FakeRDMABuffer):
        def __init__(self, data):
            super().__init__(data)
            self.reads = 0

        def read_into(self, dst, timeout=None):
            self.reads += 1
            return super().read_into(dst, timeout=timeout)

    class Guard:
        def ack_latent(self, *_identity):
            raise AssertionError("ambiguous executor shutdown was ACKed")

    instructions = list(dis.get_instructions(Executor.__exit__))
    load = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_ATTR" and instruction.argval == "shutdown")
    call = next(index for index in range(load, len(instructions))
                if instructions[index].opname == "CALL")
    target = instructions[call + 1].offset
    primary = ExitStop("executor shutdown publication interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, Executor.__exit__.__code__, 0)
            raise primary

    monkeypatch.setattr(
        transfer_utils.concurrent.futures, "ThreadPoolExecutor", Executor)
    buffers = [Buffer(torch.tensor([1], dtype=torch.uint8)),
               Buffer(torch.tensor([2], dtype=torch.uint8))]
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [2],
        "parts": [
            {"buffer": buffer, "offset": index, "nbytes": 1}
            for index, buffer in enumerate(buffers)
        ],
        "owner_token": "7" * 32, "setup_generation": 1,
    }
    monitoring.use_tool_id(tool_id, "dgxm-executor-exit-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, Executor.__exit__.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(ExitStop) as exc_info:
        try:
            read_latent_result(desc, Guard())
        finally:
            monitoring.set_local_events(tool_id, Executor.__exit__.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is primary
    assert [buffer.reads for buffer in buffers] == [1, 1]
    assert [buffer.dropped for buffer in buffers] == [0, 0]
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert [part["buffer"] for part in owner.parts] == buffers
    assert owner.keepalive is not None


def test_partial_registration_drop_failure_keeps_durable_poison_owner(
    monkeypatch, caplog
):
    """Direct message fallback must retain a registration whose drop failed."""
    poison_errors = _record_unlocked_poison_errors(monkeypatch)
    created: list[_FakeRDMABuffer] = []

    class DropFailureFuture:
        def get(self, timeout=None):
            raise RuntimeError("drop failed")

    class DropFailingBuffer(_FakeRDMABuffer):
        def drop(self):
            self.dropped += 1
            return DropFailureFuture()

    registration_calls = 0

    def second_registration_fails(data):
        nonlocal registration_calls
        registration_calls += 1
        if created:
            raise RuntimeError("registration failed")
        buffer = DropFailingBuffer(data)
        created.append(buffer)
        return buffer

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", second_registration_fails)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    tensor = torch.arange(24, dtype=torch.uint8)

    desc = LatentReturn("rdma", min_bytes=1).pack("samples", tensor)

    assert desc["kind"] == "message"
    assert torch.equal(desc["tensor"], tensor)
    assert len(created) == 1 and created[0].dropped == 1
    owner = next(
        item for item in transfer._RDMA_POISONED_OWNERS
        if item.phase == "registration rollback")
    assert owner.phase == "registration rollback"
    assert owner.parts[0]["buffer"] is created[0]
    assert owner.keepalive.data_ptr() == desc["tensor"].data_ptr()
    assert owner.failures == ("RuntimeError('drop failed')",)
    assert "registration failed" in caplog.text
    assert len(poison_errors) == 1
    assert poison_errors[0][0] is False
    assert "ownership POISONED" in poison_errors[0][1]

    # Poison is local to this sender process and lasts until the process exits:
    # the next send takes the message path and makes no RDMABuffer call.
    again = LatentReturn("rdma", min_bytes=1).pack("samples", tensor + 1)
    assert again["kind"] == "message"
    assert registration_calls == 2
    assert {item.phase for item in transfer._RDMA_POISONED_OWNERS} == {
        "RDMA buffer construction", "registration rollback"}
    assert len(poison_errors) == 1


@pytest.mark.parametrize("failed_drop_indices", [(), (0,), (0, 1)])
def test_post_registration_publication_failure_releases_or_poison_owns(
    monkeypatch, caplog, failed_drop_indices
):
    """Full registration still belongs locally until descriptor handoff."""

    class PublicationFailure(RuntimeError):
        pass

    class DropFuture:
        def __init__(self, index):
            self.index = index

        def get(self, timeout=None):
            if self.index in failed_drop_indices:
                raise RuntimeError("publication rollback drop failed")
            return None

    class Buffer(_FakeRDMABuffer):
        def __init__(self, data, index):
            super().__init__(data)
            self.index = index

        def drop(self):
            self.dropped += 1
            return DropFuture(self.index)

    created: list[Buffer] = []

    def register(data):
        buffer = Buffer(data, len(created))
        created.append(buffer)
        return buffer

    def fail_publication(self, key, seq, tensor, depth):
        raise PublicationFailure("descriptor publication failed")

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setattr(LatentReturn, "_retain", fail_publication)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    tensor = torch.arange(24, dtype=torch.uint8)

    desc = LatentReturn("rdma", min_bytes=1).pack("samples", tensor)

    assert desc["kind"] == "message"
    assert torch.equal(desc["tensor"], tensor)
    assert len(created) == 2 and all(buffer.dropped == 1 for buffer in created)
    assert "descriptor publication failed" in caplog.text
    if failed_drop_indices:
        assert len(transfer._RDMA_POISONED_OWNERS) == 1
        owner = transfer._RDMA_POISONED_OWNERS[0]
        assert owner.phase == "descriptor publication rollback"
        assert [part["buffer"] for part in owner.parts] == [
            created[index] for index in failed_drop_indices
        ]
        assert owner.keepalive.data_ptr() == desc["tensor"].data_ptr()
    else:
        assert transfer._RDMA_POISONED_OWNERS == []


def test_baseexception_registration_failure_stays_primary_when_drop_poisoned(
    monkeypatch,
):
    class StopNow(BaseException):
        pass

    class DropFailureFuture:
        def get(self, timeout=None):
            raise RuntimeError("drop failed")

    class Buffer(_FakeRDMABuffer):
        def drop(self):
            self.dropped += 1
            return DropFailureFuture()

    created: list[Buffer] = []

    def register(data):
        if created:
            raise StopNow("registration interrupted")
        buffer = Buffer(data)
        created.append(buffer)
        return buffer

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")

    with pytest.raises(StopNow, match="registration interrupted"):
        LatentReturn("rdma", min_bytes=1).pack(
            "samples", torch.arange(24, dtype=torch.uint8)
        )

    assert created[0].dropped == 1
    assert len(transfer._RDMA_POISONED_OWNERS) == 2
    owner = next(
        item for item in transfer._RDMA_POISONED_OWNERS
        if item.phase == "registration rollback")
    assert owner.parts[0]["buffer"] is created[0]


def test_registered_buffer_is_owned_before_the_next_line_can_interrupt(monkeypatch):
    class StopNow(BaseException):
        pass

    created: list[_FakeRDMABuffer] = []

    def register(data):
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    source, start = inspect.getsourcelines(transfer._register_parts)
    target = next(
        start + offset for offset, line in enumerate(source)
        if "if _part_buffer(part) is None" in line
    )
    primary = StopNow("interrupted after successful registration")

    def stop_before_publish(frame, event, _arg):
        if (event == "line" and frame.f_code is transfer._register_parts.__code__
                and frame.f_lineno == target):
            raise primary
        return stop_before_publish

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    sys.settrace(stop_before_publish)
    with pytest.raises(StopNow) as exc_info:
        try:
            LatentReturn("rdma", min_bytes=1).pack(
                "samples", torch.arange(24, dtype=torch.uint8)
            )
        finally:
            sys.settrace(None)

    assert exc_info.value is primary
    assert len(created) == 1
    assert created[0].dropped == 1


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("handoff", ["constructor", "register return"])
def test_instruction_boundary_never_orphans_a_successful_registration(
    monkeypatch, handoff,
):
    class StopNow(BaseException):
        pass

    created: list[_FakeRDMABuffer] = []

    def register(data):
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    code = (transfer._register_parts.__code__ if handoff == "constructor"
            else LatentReturn.pack.__code__)
    instructions = list(dis.get_instructions(code))
    needle = "extend" if handoff == "constructor" else "_register_parts"
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_GLOBAL"}
        and instruction.argval == needle
    )
    calls = [
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    ]
    target = instructions[calls[1] + 1].offset
    primary = StopNow(f"{handoff} instruction interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise primary

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "1")
    monitoring.use_tool_id(tool_id, "dgxm-transfer-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, code, monitoring.events.INSTRUCTION)
    with pytest.raises(StopNow) as exc_info:
        try:
            LatentReturn("rdma", min_bytes=1).pack(
                "samples", torch.arange(24, dtype=torch.uint8)
            )
        finally:
            monitoring.set_local_events(tool_id, code, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is primary
    assert len(created) == 1 and created[0].dropped == 1
    if handoff == "constructor":
        owner = transfer._RDMA_POISONED_OWNERS[0]
        assert owner.phase == "RDMA buffer construction"
        assert owner.keepalive.numel() == 24
    else:
        assert transfer._RDMA_POISONED_OWNERS == []


def test_registration_return_handoff_is_inside_rollback_envelope(monkeypatch):
    class StopNow(BaseException):
        pass

    created: list[_FakeRDMABuffer] = []

    def register(data):
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    source, start = inspect.getsourcelines(transfer._register_parts)
    target = next(
        start + offset for offset, line in enumerate(source)
        if line.strip() == "return parts"
    )
    primary = StopNow("registration return interrupted")

    def stop_return(frame, event, _arg):
        if (event == "line" and frame.f_code is transfer._register_parts.__code__
                and frame.f_lineno == target):
            raise primary
        return stop_return

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    sys.settrace(stop_return)
    with pytest.raises(StopNow) as exc_info:
        try:
            LatentReturn("rdma", min_bytes=1).pack(
                "samples", torch.arange(24, dtype=torch.uint8)
            )
        finally:
            sys.settrace(None)

    assert exc_info.value is primary
    assert len(created) == 2
    assert [buffer.dropped for buffer in created] == [1, 1]


def test_registration_cleanup_entry_interruption_preserves_primary_and_ownership(
    monkeypatch,
):
    class RegistrationStop(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = RegistrationStop("second registration interrupted")
    cleanup_stop = CleanupStop("registration cleanup helper entry interrupted")
    created: list[_FakeRDMABuffer] = []

    def register(data):
        if created:
            raise primary
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    source, start = inspect.getsourcelines(transfer._register_parts)
    target = next(
        start + offset for offset, line in enumerate(source)
        if "_, poison_error, cleanup_error = _settle_or_fail_closed(" in line
    )

    def stop_cleanup_entry(frame, event, _arg):
        if (event == "line" and frame.f_code is transfer._register_parts.__code__
                and frame.f_lineno == target):
            raise cleanup_stop
        return stop_cleanup_entry

    monkeypatch.setattr(transfer, "RDMABuffer", register)
    sys.settrace(stop_cleanup_entry)
    with pytest.raises(RegistrationStop) as exc_info:
        try:
            transfer._register_parts(torch.arange(24, dtype=torch.uint8), 2)
        finally:
            sys.settrace(None)

    assert exc_info.value is primary
    assert len(created) == 1 and created[0].dropped == 0
    owner = next(
        item for item in transfer._RDMA_POISONED_OWNERS
        if item.phase == "registration rollback")
    assert owner.phase == "registration rollback"
    assert owner.parts[0]["buffer"] is created[0]
    assert any("cleanup helper entry" in note for note in primary.__notes__)


def test_descriptor_cleanup_entry_interruption_preserves_primary_and_ownership(
    monkeypatch,
):
    class PublicationStop(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = PublicationStop("descriptor publication interrupted")
    cleanup_stop = CleanupStop("descriptor cleanup helper entry interrupted")
    created: list[_FakeRDMABuffer] = []

    def register(data):
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    def fail_publication(self, key, seq, tensor, depth):
        raise primary

    source, start = inspect.getsourcelines(LatentReturn.pack)
    target = next(
        start + offset for offset, line in enumerate(source)
        if "_, poison_error, cleanup_error = _settle_or_fail_closed(" in line
    )

    def stop_cleanup_entry(frame, event, _arg):
        if (event == "line" and frame.f_code is LatentReturn.pack.__code__
                and frame.f_lineno == target):
            raise cleanup_stop
        return stop_cleanup_entry

    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setattr(LatentReturn, "_retain", fail_publication)
    monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    sys.settrace(stop_cleanup_entry)
    with pytest.raises(PublicationStop) as exc_info:
        try:
            LatentReturn("rdma", min_bytes=1).pack(
                "samples", torch.arange(24, dtype=torch.uint8)
            )
        finally:
            sys.settrace(None)

    assert exc_info.value is primary
    assert len(created) == 2 and [buffer.dropped for buffer in created] == [0, 0]
    owner = transfer._RDMA_POISONED_OWNERS[0]
    assert owner.phase == "descriptor publication rollback"
    assert [part["buffer"] for part in owner.parts] == created
    assert any("cleanup helper entry" in note for note in primary.__notes__)


@pytest.mark.parametrize("path", ["registration", "descriptor"])
def test_sender_poison_entry_interruption_retries_publication_without_redrop(
    monkeypatch, path,
):
    class PrimaryStop(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    class PoisonStop(BaseException):
        pass

    primary = PrimaryStop(f"{path} primary")
    cleanup_stop = CleanupStop("force fail-closed fallback")
    poison_stop = PoisonStop("poison helper entry interrupted")
    created: list[_FakeRDMABuffer] = []

    def register(data):
        if path == "registration" and created:
            raise primary
        buffer = _FakeRDMABuffer(data)
        created.append(buffer)
        return buffer

    def fail_publication(self, key, seq, tensor, depth):
        raise primary

    def fail_settle(*_args, **_kwargs):
        raise cleanup_stop

    source, start = inspect.getsourcelines(transfer._poison_with_retry)
    target = next(
        start + offset for offset, line in enumerate(source)
        if line.strip().startswith("_poison_failed_drops(failures,")
    )

    def stop_poison_entry(frame, event, _arg):
        if (event == "line" and frame.f_code is transfer._poison_with_retry.__code__
                and frame.f_lineno == target):
            raise poison_stop
        return stop_poison_entry

    monkeypatch.setattr(transfer, "RDMABuffer", register)
    monkeypatch.setattr(transfer, "_settle_parts", fail_settle)
    if path == "descriptor":
        monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
        monkeypatch.setattr(LatentReturn, "_retain", fail_publication)
        monkeypatch.setenv("DGXM_RDMA_QP_SPLIT", "2")
    sys.settrace(stop_poison_entry)
    with pytest.raises(PrimaryStop) as exc_info:
        try:
            if path == "registration":
                transfer._register_parts(torch.arange(24, dtype=torch.uint8), 2)
            else:
                LatentReturn("rdma", min_bytes=1).pack(
                    "samples", torch.arange(24, dtype=torch.uint8)
                )
        finally:
            sys.settrace(None)

    assert exc_info.value is primary
    assert [buffer.dropped for buffer in created] == [0] * len(created)
    expected_phase = ("registration rollback" if path == "registration"
                      else "descriptor publication rollback")
    owner = next(
        item for item in transfer._RDMA_POISONED_OWNERS
        if item.phase == expected_phase)
    assert [part["buffer"] for part in owner.parts] == created
    assert any("poison helper entry" in note for note in primary.__notes__)


def test_legacy_no_handoff_keepalive_depth_window_bounds_backings(monkeypatch):
    """Only tokenless packs enter the depth-bounded keepalive window.

    A production handoff that reaches the RDMA path carries a token and stays registry-owned until the
    matching ACK or actor recycle.
    """
    import dgx_monarch.transfer as transfer
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", _FakeRDMABuffer)
    lr = LatentReturn("rdma", min_bytes=1, keepalive_depth=2)
    t = torch.arange(1, 33, dtype=torch.uint8)
    for seq in range(5):
        lr.pack("samples", t.clone(), seq=seq, depth=2)
        assert len([k for k in lr._keepalive if k[0] == "samples"]) <= 2
    assert set(lr._keepalive) == {("samples", 3), ("samples", 4)}
    lr.drop("samples")
    assert lr._keepalive == {}


def test_legacy_no_handoff_window_handles_sequence_restart(monkeypatch):
    """A sequence restart evicts the old high-sequence packs from the tokenless window.

    The window keeps the newest packs by insertion order, not the highest sequence numbers.
    """
    monkeypatch.setattr(transfer, "is_ibverbs_available", lambda: True)
    monkeypatch.setattr(transfer, "RDMABuffer", _FakeRDMABuffer)
    lr = LatentReturn("rdma", min_bytes=1, keepalive_depth=2)
    t = torch.arange(1, 33, dtype=torch.uint8)

    for seq in range(8):
        lr.pack("samples", t.clone(), seq=seq, depth=2)
    assert set(lr._keepalive) == {("samples", 6), ("samples", 7)}

    lr.pack("samples", t.clone(), seq=0, depth=2)
    lr.pack("samples", t.clone(), seq=1, depth=2)
    assert set(lr._keepalive) == {("samples", 0), ("samples", 1)}
