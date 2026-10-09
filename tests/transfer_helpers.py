"""Fixtures and CPU doubles shared by the RDMA transfer test modules."""
import dis
import sys
import threading
from types import SimpleNamespace

import pytest

import dgx_monarch.transfer as transfer
from dgx_monarch.mesh_lease import SetupBoundFuture
from dgx_monarch.rdma_ownership import HandoffRegistry
from dgx_monarch.transfer import read_latent_result


@pytest.fixture(autouse=True)
def _isolate_process_lifetime_rdma_poison(monkeypatch):
    """Each test gets the fresh process state production gets at actor start."""
    monkeypatch.setattr(transfer, "_RDMA_POISONED_OWNERS", [])
    monkeypatch.setattr(
        transfer, "_RDMA_POISON_REPORTER", transfer.rdma_poison.PoisonReporter())


def _record_unlocked_poison_errors(monkeypatch):
    is_owned = getattr(transfer._RDMA_POISON_LOCK, "_is_owned", None)
    if not callable(is_owned):
        pytest.skip("RLock ownership probe unavailable")
    records = []

    def record(message, *args):
        records.append((is_owned(), message % args))

    monkeypatch.setattr(transfer.log, "error", record)
    return records


def _call_instruction_offset(code, attribute, *, successor=False, after=0):
    """Locate a semantic method CALL without pinning CPython byte offsets."""
    instructions = list(dis.get_instructions(code))
    loads = [
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {
            "LOAD_ATTR", "LOAD_DEREF", "LOAD_GLOBAL", "LOAD_METHOD"}
        and instruction.argval == attribute
    ]
    load = loads[after]
    call = next(
        index for index in range(load + 1, len(instructions))
        if instructions[index].opname == "CALL")
    return instructions[call + int(successor)].offset


def _run_at_instruction(code, target, boundary, callback):
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, code, 0)
            raise boundary

    monitoring.use_tool_id(tool_id, "dgxm-rdma-opcode-test")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(tool_id, code, monitoring.events.INSTRUCTION)
    try:
        return callback()
    finally:
        monitoring.set_local_events(tool_id, code, 0)
        monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)


def _real_pending_rdma(monkeypatch, buffer, generation):
    from dgx_monarch import telemetry
    from dgx_monarch.nodes.pending import PendingRender

    monkeypatch.setattr(telemetry.render_progress, "finish", lambda *_args: None)
    registry = HandoffRegistry()
    actor_part = {"buffer": buffer, "offset": 0, "nbytes": 1}
    keepalive = object()
    handoff = {}
    registry.publish(handoff, generation, 1)
    handoff.update(
        parts=[actor_part], keepalive=keepalive, state="registered")
    registry.mark_ready(handoff)
    token = handoff["token"]
    ack_calls = []

    handle = SimpleNamespace(
        world=1, lock=threading.RLock(), sample_leases={generation: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None,
        cancel_sample=lambda *_args, **_kwargs: None)

    def call_all(_endpoint, setup_generation, owner_token, timeout_s):
        ack_calls.append((setup_generation, owner_token, timeout_s))
        return [{
            "setup_generation": setup_generation,
            "token": owner_token,
            "status": registry.acknowledge(setup_generation, owner_token),
            "rank": 0,
        }]

    handle.call_all = call_all
    handle.collect_sample = lambda *_args, **_kwargs: []

    class Guard(SetupBoundFuture):
        def __init__(self, *args):
            super().__init__(*args)
            self.begins = 0

        def begin_read(self, owner=None):
            self.begins += 1
            return super().begin_read(owner)

    guard = Guard(None, handle, generation)
    closed = []
    desc = {
        "kind": "rdma", "dtype": "uint8", "shape": [1],
        "parts": [dict(actor_part)], "owner_token": token,
        "setup_generation": generation,
    }
    pending = PendingRender(
        handle, guard,
        SimpleNamespace(__exit__=lambda *_args: closed.append(True)),
        None, {}, 1.0, f"render-{generation}",
        lambda *_args: read_latent_result(desc, guard))
    return SimpleNamespace(
        registry=registry, actor_part=actor_part, keepalive=keepalive,
        token=token, ack_calls=ack_calls, handle=handle, guard=guard,
        pending=pending, closed=closed, desc=desc)


class _FakeFuture:
    def __init__(self, n: int):
        self._n = n

    def get(self, timeout=None):
        return self._n


class _FakeRDMABuffer:
    """CPU stand-in for RDMABuffer: snapshots the source bytes at construction
    and copies them into the destination on read_into, so the split packing and
    concurrent reconstruction run without native RDMA."""

    def __init__(self, data):
        self._src = data.clone()  # 1-D uint8 slice
        self.dropped = 0

    def read_into(self, dst, timeout=None):
        dst.copy_(self._src)
        return _FakeFuture(self._src.numel())

    def drop(self):
        self.dropped += 1
        return _FakeFuture(None)


class _LoopRefusingFuture(_FakeFuture):
    """get() raises PythonTask's on-loop error where monarch 0.6.0's Future.get only warns
    (tests/fixtures/torchmonarch_pin_0_6_0.json), so an on-loop read fails the test."""

    def get(self, timeout=None):
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return super().get(timeout=timeout)
        raise RuntimeError(
            "Attempting to __await__ a PythonTask when the asyncio event loop "
            "is active."
        )


class _LoopRefusingBuffer(_FakeRDMABuffer):
    def read_into(self, dst, timeout=None):
        dst.copy_(self._src)
        return _LoopRefusingFuture(self._src.numel())

    def drop(self):
        self.dropped += 1
        return _LoopRefusingFuture(None)
