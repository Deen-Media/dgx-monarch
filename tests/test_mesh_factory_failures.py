"""Failure-path regressions for the production worker-fleet factory."""

from __future__ import annotations

import asyncio
import dis
import inspect
import sys
import threading
import types

import pytest

import dgx_monarch
from dgx_monarch.mesh_factory import (
    FleetHandoff,
    WorkerSpawnError,
    _unregister_callback_once,
    spawn_worker_fleet,
)


class SpawnAbort(BaseException):
    """Stand-in for non-Exception native/runtime interruption."""


class RollbackAbort(BaseException):
    """Stand-in for non-Exception failure while stopping partial procs."""


class HandoffAbort(BaseException):
    """Stand-in for a non-Exception interruption at one step of fleet bring-up."""


class AccessorAbort(BaseException):
    """Stand-in for a non-Exception failure inside a ProcMesh ownership accessor."""


class _StopFuture:
    def __init__(self, failure: BaseException | None = None):
        self.failure = failure
        self.timeouts: list[float | None] = []
        self.thread_names: list[str] = []

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        self.thread_names.append(threading.current_thread().name)
        if self.failure is not None:
            raise self.failure
        return None


def _install_actor_stub(monkeypatch) -> None:
    """Stub dgx_monarch.actor: the factory imports GPUWorker from it, and the real package imports torch."""
    actor = types.ModuleType("dgx_monarch.actor")
    actor.GPUWorker = object
    monkeypatch.setitem(sys.modules, "dgx_monarch.actor", actor)
    monkeypatch.setattr(dgx_monarch, "actor", actor, raising=False)


def _hosts_for(procs):
    shutdown_calls: list[object] = []

    def shutdown(*args, **kwargs):
        shutdown_calls.append((args, kwargs))
        raise AssertionError("attached worker loops must never be shut down")

    hosts = types.SimpleNamespace(
        spawn_procs=lambda **_kwargs: procs,
        shutdown=shutdown,
    )
    return hosts, shutdown_calls


def _successful_handoff_line() -> int:
    lines, first_line = inspect.getsourcelines(spawn_worker_fleet)
    for offset, line in enumerate(lines):
        if line.strip() == "return hosts, procs, workers, owns_hosts":
            return first_line + offset
    raise AssertionError("spawn_worker_fleet handoff line not found")


def test_mesh_factory_safe_note_is_shared_without_changing_either_call_form():
    from dgx_monarch import error_utils, mesh_factory

    primary = RuntimeError("primary")
    detail = ValueError("detail")

    assert mesh_factory.safe_note is error_utils.safe_note
    mesh_factory.safe_note(primary, "exact note")
    error_utils.safe_note(primary, "secondary failure", detail)

    assert primary.__notes__ == [
        "exact note",
        "secondary failure: ValueError('detail')",
    ]


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("boundary", ["return", "spawn", "attach"])
def test_cluster_attach_transition_keeps_the_transport_handoff_pending(boundary):
    from dgx_monarch import mesh_factory

    handoff = FleetHandoff()
    instructions = list(dis.get_instructions(spawn_worker_fleet))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_FAST"
        and instruction.argval == "attach_cluster"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    if boundary == "return":
        target = instructions[call_index + 1]
        assert target.opname == "STORE_FAST" and target.argval == "hosts"
    else:
        target = next(
            instruction for instruction in instructions[call_index + 1:]
            if instruction.opname == "STORE_ATTR"
            and instruction.argval == f"{boundary}_in_progress"
        )
    primary = HandoffAbort("cluster attach return instruction interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(
                tool_id, spawn_worker_fleet.__code__, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-cluster-attach-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, spawn_worker_fleet.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(HandoffAbort) as failure:
        try:
            spawn_worker_fleet(
                None, True, 1, lambda _config, _handoff: object(), None, handoff)
        finally:
            monitoring.set_local_events(
                tool_id, spawn_worker_fleet.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value is primary
    assert handoff.attach_in_progress is True
    assert handoff.spawn_in_progress is (boundary == "attach")
    assert handoff.proc_meshes == [] and handoff.fleet is None
    assert mesh_factory._cleanup_pending(handoff) is True


def test_spawn_baseexception_rolls_back_partial_proc_mesh(monkeypatch):
    _install_actor_stub(monkeypatch)
    primary = SpawnAbort("spawn interrupted")
    stop_future = _StopFuture()

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def spawn(self, *_args, **_kwargs):
            raise primary

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()
    hosts, shutdown_calls = _hosts_for(procs)

    async def bring_up():
        with pytest.raises(WorkerSpawnError) as caught:
            spawn_worker_fleet(None, True, 1, lambda _config, _handoff: hosts, None)
        return caught.value

    error = asyncio.run(bring_up())
    assert error.original is primary
    assert error.cleanup is None
    assert error.__cause__ is primary
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]
    assert stop_future.thread_names == ["dgxm-bringup-rollback"]
    assert shutdown_calls == []


def test_rollback_baseexception_preserves_spawn_and_cleanup_evidence(monkeypatch):
    _install_actor_stub(monkeypatch)
    primary = RuntimeError("worker spawn failed")
    cleanup = RollbackAbort("partial rollback interrupted")
    stop_future = _StopFuture(cleanup)

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def spawn(self, *_args, **_kwargs):
            raise primary

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()
    hosts, shutdown_calls = _hosts_for(procs)

    async def bring_up():
        with pytest.raises(WorkerSpawnError) as caught:
            spawn_worker_fleet(None, True, 1, lambda _config, _handoff: hosts, None)
        return caught.value

    error = asyncio.run(bring_up())
    assert error.original is primary
    assert error.cleanup is cleanup
    assert error.__cause__ is primary
    assert str(error) == str(primary)
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]
    assert stop_future.thread_names == ["dgxm-bringup-rollback"]
    assert shutdown_calls == []


def test_worker_spawn_error_survives_broken_original_str_and_repr(monkeypatch):
    _install_actor_stub(monkeypatch)
    stop_future = _StopFuture()

    class BrokenOriginal(BaseException):
        def __str__(self):
            raise RuntimeError("original str unavailable")

        def __repr__(self):
            raise RuntimeError("original repr unavailable")

    primary = BrokenOriginal()

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def spawn(self, *_args, **_kwargs):
            raise primary

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()
    hosts, shutdown_calls = _hosts_for(procs)
    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(None, True, 1, lambda _config, _handoff: hosts, None)

    assert failure.value.original is primary
    assert str(failure.value) == "<BrokenOriginal>"
    assert failure.value.cleanup is None
    assert failure.value.__cause__ is primary
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]
    assert shutdown_calls == []


@pytest.mark.parametrize("probe", ["host", "active", "candidate"])
def test_hostile_ownership_probe_preserves_exact_spawn_failure(monkeypatch, probe):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    _install_actor_stub(monkeypatch)
    primary = SpawnAbort("spawn failed before returning its ProcMesh")
    diagnostic = AccessorAbort(f"{probe} ownership accessor failed")
    callbacks: list[object] = []
    handoff = FleetHandoff()

    class Candidate:
        def __getattribute__(self, name):
            if name == "_host_mesh":
                raise diagnostic
            return super().__getattribute__(name)

    candidate = Candidate()

    class Hosts:
        def __getattribute__(self, name):
            if name == "_proc_meshes" and probe == "host":
                raise diagnostic
            return super().__getattribute__(name)

        def spawn_procs(self, **_kwargs):
            if probe == "candidate":
                callbacks[0](candidate)
            raise primary

    def active_proc_meshes():
        if probe == "active":
            raise diagnostic
        return []

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", callbacks.remove)
    monkeypatch.setattr(
        proc_mesh_api, "get_active_proc_meshes", active_proc_meshes)

    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(
            None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)

    assert failure.value.original is primary
    assert failure.value.__cause__ is primary
    assert failure.value.rollback_confirmed is False
    assert failure.value.cleanup is not diagnostic
    assert failure.value.cleanup is not None
    assert "ownership is unconfirmed" in str(failure.value.cleanup)
    assert handoff.proc_ownership_unconfirmed is True
    assert callbacks == []


def test_normal_ownership_probe_cancellation_remains_exact(monkeypatch):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    _install_actor_stub(monkeypatch)
    cancellation = KeyboardInterrupt("normal ownership resolution interrupted")
    callbacks: list[object] = []
    handoff = FleetHandoff()

    class Hosts:
        def __getattribute__(self, name):
            if name == "_proc_meshes":
                raise cancellation
            return super().__getattribute__(name)

        def spawn_procs(self, **_kwargs):
            return None

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", callbacks.remove)

    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(
            None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)

    error = failure.value
    assert error.original is cancellation
    assert error.__cause__ is cancellation
    assert handoff.proc_ownership_unconfirmed is True
    assert callbacks == []
    with pytest.raises(KeyboardInterrupt) as restored:
        error.raise_cancellation()
    assert restored.value is cancellation


def test_hostile_ownership_recovery_call_preserves_exact_spawn_failure(monkeypatch):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    from dgx_monarch import mesh_factory

    _install_actor_stub(monkeypatch)
    primary = SpawnAbort("spawn failed before ownership recovery")
    diagnostic = AccessorAbort("ownership recovery helper failed")
    callbacks: list[object] = []
    handoff = FleetHandoff()

    class Hosts:
        def spawn_procs(self, **_kwargs):
            raise primary

    def fail_ownership_recovery(*_args, **_kwargs):
        raise diagnostic

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", callbacks.remove)
    monkeypatch.setattr(mesh_factory, "owned_spawned_proc", fail_ownership_recovery)

    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(
            None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)

    assert failure.value.original is primary
    assert failure.value.__cause__ is primary
    assert failure.value.rollback_confirmed is False
    assert failure.value.cleanup is not diagnostic
    assert failure.value.cleanup is not None
    assert "ownership is unconfirmed" in str(failure.value.cleanup)
    assert primary.__notes__ == [
        "spawned ProcMesh ownership recovery also failed: "
        "AccessorAbort('ownership recovery helper failed')"
    ]
    assert handoff.proc_ownership_unconfirmed is True
    assert callbacks == []


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_successful_spawn_handoff_baseexception_rolls_back_owned_proc_mesh(
    monkeypatch, rollback_fails,
):
    _install_actor_stub(monkeypatch)
    primary = HandoffAbort("fleet ownership handoff interrupted")
    cleanup = (RollbackAbort("handoff rollback interrupted")
               if rollback_fails else None)
    stop_future = _StopFuture(cleanup)
    workers = object()

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def spawn(self, *_args, **_kwargs):
            return workers

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()
    hosts, shutdown_calls = _hosts_for(procs)
    handoff = FleetHandoff()
    handoff_line = _successful_handoff_line()

    def interrupt_handoff(frame, event, _arg):
        if (frame.f_code is spawn_worker_fleet.__code__
                and event == "line" and frame.f_lineno == handoff_line):
            assert frame.f_locals["procs"] is procs
            assert frame.f_locals["workers"] is workers
            raise primary
        return interrupt_handoff

    async def bring_up():
        previous_trace = sys.gettrace()
        sys.settrace(interrupt_handoff)
        try:
            with pytest.raises(WorkerSpawnError) as caught:
                spawn_worker_fleet(
                    None, True, 1, lambda _config, _handoff: hosts, None, handoff)
        finally:
            sys.settrace(previous_trace)
        return caught.value

    error = asyncio.run(bring_up())
    assert error.original is primary
    assert error.cleanup is cleanup
    assert error.__cause__ is primary
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]
    assert stop_future.thread_names == ["dgxm-bringup-rollback"]
    assert shutdown_calls == []
    assert handoff.fleet == (hosts, procs, workers, True)
    assert handoff.cleanup_started is True
    assert handoff.cleanup_confirmed is (not rollback_fails)


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("boundary", ["callback", "factory"])
def test_spawn_procs_instruction_handoff_never_orphans_proc_mesh(
    monkeypatch, boundary,
):
    """Monarch's spawn callback records the ProcMesh before the factory stores it."""
    from monarch._src.actor import proc_mesh as proc_mesh_api

    _install_actor_stub(monkeypatch)
    primary = HandoffAbort("ProcMesh return instruction interrupted")
    stop_future = _StopFuture()
    callbacks: list[object] = []
    handoff = FleetHandoff()

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()

    class Hosts:
        def spawn_procs(self, **_kwargs):
            assert len(callbacks) == 1
            callback = callbacks[0]
            assert getattr(callback, "__self__", None) is handoff.proc_meshes
            assert getattr(callback, "__name__", None) == "append"
            assert not hasattr(callback, "__code__")
            procs._host_mesh = self
            callback(procs)
            return procs

        def shutdown(self):
            pytest.fail("attached worker loops are not client-owned")

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", callbacks.remove)
    code = (Hosts.spawn_procs.__code__ if boundary == "callback"
            else spawn_worker_fleet.__code__)
    instructions = list(dis.get_instructions(code))
    if boundary == "callback":
        lines, first_line = inspect.getsourcelines(Hosts.spawn_procs)
        callback_line = first_line + next(
            index for index, line in enumerate(lines)
            if line.strip() == "callback(procs)"
        )
        call_index = next(
            index for index, instruction in enumerate(instructions)
            if instruction.opname == "CALL"
            and instruction.positions.lineno == callback_line
        )
    else:
        load_index = next(
            index for index, instruction in enumerate(instructions)
            if instruction.opname == "LOAD_ATTR"
            and instruction.argval == "spawn_procs"
        )
        call_index = next(
            index for index in range(load_index + 1, len(instructions))
            if instructions[index].opname == "CALL"
        )
    target = instructions[call_index + 1]
    expected = ("POP_TOP", None) if boundary == "callback" else ("STORE_FAST", "procs")
    assert (target.opname, target.argval) == expected
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, code, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-proc-mesh-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, code, monitoring.events.INSTRUCTION)
    with pytest.raises(WorkerSpawnError) as failure:
        try:
            spawn_worker_fleet(
                None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)
        finally:
            monitoring.set_local_events(tool_id, code, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value.original is primary
    assert failure.value.cleanup is None
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]
    assert callbacks == []
    assert handoff.proc_meshes and set(map(id, handoff.proc_meshes)) == {id(procs)}
    assert handoff.cleanup_started is True
    assert handoff.cleanup_confirmed is True


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_spawn_callback_registration_instruction_is_reconciled(monkeypatch):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    _install_actor_stub(monkeypatch)
    primary = HandoffAbort("callback registration instruction interrupted")
    callbacks: list[object] = []
    handoff = FleetHandoff()

    class Hosts:
        def spawn_procs(self, **_kwargs):
            pytest.fail("an interrupted callback registration must abort spawning")

    def register(callback):
        callbacks.append(callback)

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", register)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", callbacks.remove)
    instructions = list(dis.get_instructions(register))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "append"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = instructions[call_index + 1]
    assert target.opname == "POP_TOP"
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, register.__code__, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-spawn-callback-register-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, register.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(WorkerSpawnError) as failure:
        try:
            spawn_worker_fleet(
                None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)
        finally:
            monitoring.set_local_events(tool_id, register.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value.original is primary
    assert callbacks == []
    assert handoff.callback_cleanup_unconfirmed is False
    assert handoff.proc_meshes == []
    assert handoff.cleanup_started is False


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_spawn_callback_unregister_instruction_is_reconciled(monkeypatch):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    _install_actor_stub(monkeypatch)
    primary = HandoffAbort("callback unregister instruction interrupted")
    stop_future = _StopFuture()
    callbacks: list[object] = []
    handoff = FleetHandoff()

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()

    class Hosts:
        def spawn_procs(self, **_kwargs):
            callbacks[0](procs)
            return procs

    def unregister(callback):
        callbacks.remove(callback)

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", unregister)
    instructions = list(dis.get_instructions(unregister))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "remove"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = instructions[call_index + 1]
    assert target.opname == "POP_TOP"
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, unregister.__code__, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-spawn-callback-unregister-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, unregister.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(WorkerSpawnError) as failure:
        try:
            spawn_worker_fleet(
                None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)
        finally:
            monitoring.set_local_events(tool_id, unregister.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value.original is primary
    assert callbacks == []
    assert handoff.callback_cleanup_unconfirmed is False
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_spawn_callback_cleanup_flag_instruction_is_reconciled(monkeypatch):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    _install_actor_stub(monkeypatch)
    primary = HandoffAbort("callback cleanup publication interrupted")
    stop_future = _StopFuture()
    callbacks: list[object] = []
    handoff = FleetHandoff()

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()

    class Hosts:
        def spawn_procs(self, **_kwargs):
            callbacks[0](procs)
            return procs

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", callbacks.remove)
    instructions = list(dis.get_instructions(_unregister_callback_once))
    call_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "CALL"
    )
    target = next(
        instruction for instruction in instructions[call_index + 1:]
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "callback_cleanup_unconfirmed"
    )
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(
                tool_id, _unregister_callback_once.__code__, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-spawn-callback-cleanup-flag-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, _unregister_callback_once.__code__,
        monitoring.events.INSTRUCTION)
    with pytest.raises(WorkerSpawnError) as failure:
        try:
            spawn_worker_fleet(
                None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)
        finally:
            monitoring.set_local_events(
                tool_id, _unregister_callback_once.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value.original is primary
    assert callbacks == []
    assert handoff.callback_cleanup_unconfirmed is False
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]


def test_spawn_callback_cleanup_that_cannot_be_confirmed_stays_latched(monkeypatch):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    _install_actor_stub(monkeypatch)
    cleanup = HandoffAbort("callback removal unavailable")
    stop_future = _StopFuture()
    callbacks: list[object] = []
    unregister_calls = 0
    handoff = FleetHandoff()

    class Procs:
        def __init__(self):
            self.stop_reasons: list[str] = []

        def stop(self, reason):
            self.stop_reasons.append(reason)
            return stop_future

    procs = Procs()

    class Hosts:
        def spawn_procs(self, **_kwargs):
            callbacks[0](procs)
            return procs

    def unregister(_callback):
        nonlocal unregister_calls
        unregister_calls += 1
        raise cleanup

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", unregister)
    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(
            None, True, 1, lambda _config, _handoff: Hosts(), None, handoff)

    assert failure.value.original is cleanup
    assert unregister_calls == 2 and len(callbacks) == 1
    assert handoff.callback_cleanup_unconfirmed is True
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert stop_future.timeouts == [60]
