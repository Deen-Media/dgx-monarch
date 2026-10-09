"""Mesh creation claims, handle publication and partial-failure ownership.

The ProcMesh, worker fleet and lease thread are fakes; no Monarch runtime runs.
"""
from __future__ import annotations

import dis
import sys
from types import SimpleNamespace

import pytest

from dgx_monarch import client_lease


class _PublishedProcMesh:
    def __init__(self, stop_error: BaseException | None = None) -> None:
        self.stop_error = stop_error
        self.stop_reasons: list[str] = []
        self.stop_timeouts: list[float | None] = []

    def stop(self, reason: str):
        self.stop_reasons.append(reason)
        owner = self

        class _StopResult:
            def get(self, timeout=None):
                owner.stop_timeouts.append(timeout)
                if owner.stop_error is not None:
                    raise owner.stop_error

        return _StopResult()


def _mesh_publication_fakes(monkeypatch, spawn, start):
    from dgx_monarch import mesh, mesh_factory, mesh_helpers

    monkeypatch.setattr(mesh, "install_fault_hook", lambda: None)
    monkeypatch.setattr(mesh, "find_config_path", lambda _path: None)
    monkeypatch.setattr(mesh, "_detect_comfy_dir", lambda _path: "/comfy")
    monkeypatch.setattr(mesh, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(mesh, "_src_pythonpath", lambda: "/repo/src")
    monkeypatch.setattr(mesh.client_lease, "plant_driver_marker", lambda: "")
    monkeypatch.setattr(mesh, "spawn_worker_fleet", spawn)
    monkeypatch.setattr(mesh_helpers, "start_client_lease", start)
    monkeypatch.setattr(mesh, "_MESHES", {})
    monkeypatch.setattr(mesh, "_MESH_PENDING", {})
    monkeypatch.setattr(mesh, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh, "_TRANSPORT_MODE", None)
    monkeypatch.setattr(mesh, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh, "_TRANSPORT_BIND", None)
    monkeypatch.setattr(mesh, "_TRANSPORT_CLAIM_GENERATION", 0)
    monkeypatch.setattr(mesh, "_TRANSPORT_COMMITTED_GENERATION", None)
    monkeypatch.setattr(mesh, "_TRANSPORT_CLAIMS", set())
    monkeypatch.setattr(mesh_factory, "_CREATION_CLEANUP_CLAIMS", {})
    monkeypatch.setenv("DGXM_PYTHONPATH", "before-test")
    monkeypatch.setenv("PYTHONPATH", "before-test")
    return mesh


@pytest.mark.parametrize("startup_error", [
    RuntimeError("client lease thread did not start"),
    KeyboardInterrupt("client lease thread start interrupted"),
])
def test_mesh_publication_waiters_never_reuse_a_handle_whose_lease_start_failed(
    monkeypatch, startup_error,
):
    import threading

    published = threading.Event()
    release_failure = threading.Event()
    second_entered = threading.Event()
    second_done = threading.Event()
    proc_meshes: list[_PublishedProcMesh] = []
    handles_at_start: list[object] = []
    starts = 0

    def spawn(*_args):
        procs = _PublishedProcMesh()
        proc_meshes.append(procs)
        hosts = SimpleNamespace(
            shutdown=lambda: pytest.fail("attached worker loops are not client-owned"))
        return hosts, procs, SimpleNamespace(), False

    def start():
        nonlocal starts
        starts += 1
        assert mesh._MESHES == {}
        assert len(mesh._MESH_PENDING) == 1
        handles_at_start.append(next(iter(mesh._MESH_PENDING.values())))
        if starts == 1:
            published.set()
            assert release_failure.wait(timeout=2)
            raise startup_error
        return True

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    errors: list[BaseException] = []
    results: list[object] = []

    def first_creator():
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1)
        except BaseException as exc:
            errors.append(exc)

    def waiting_reuser():
        try:
            second_entered.set()
            results.append(mesh.get_mesh(mode="local", gpus_per_host=1))
        finally:
            second_done.set()

    first = threading.Thread(target=first_creator)
    second = threading.Thread(target=waiting_reuser)
    first.start()
    assert published.wait(timeout=2)
    second.start()
    assert second_entered.wait(timeout=2)
    assert not second_done.wait(timeout=0.05)
    release_failure.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive() and not second.is_alive()
    assert errors == [startup_error]
    assert len(proc_meshes) == 2 and starts == 2
    assert proc_meshes[0].stop_reasons == ["dgx-monarch client detach"]
    assert proc_meshes[0].stop_timeouts == [30]
    assert proc_meshes[1].stop_reasons == []
    assert results == [handles_at_start[1]]
    assert results[0] is not handles_at_start[0]
    assert list(mesh._MESHES.values()) == results
    assert mesh._MESH_PENDING == {}


def test_publication_boundary_baseexception_retires_the_unleased_handle(monkeypatch):
    proc_mesh = _PublishedProcMesh()
    starts = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        hosts = SimpleNamespace(
            shutdown=lambda: pytest.fail("attached worker loops are not client-owned"))
        handoff.fleet = hosts, proc_mesh, SimpleNamespace(), False
        return handoff.fleet

    def start():
        nonlocal starts
        starts += 1
        return True

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    interrupted = KeyboardInterrupt("publication boundary interrupted")

    def inject_after_publication(frame, event, _arg):
        if frame.f_globals is not mesh.__dict__:
            return None
        if (event == "line" and mesh._MESH_PENDING and starts == 0
                and not mesh._MESH_LOCK.locked()):
            raise interrupted
        return inject_after_publication

    previous_trace = sys.gettrace()
    sys.settrace(inject_after_publication)
    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            mesh.get_mesh(mode="local", gpus_per_host=1)
    finally:
        sys.settrace(previous_trace)

    assert failure.value is interrupted
    assert starts == 0
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert mesh._MESHES == {}
    assert mesh._MESH_PENDING == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()


def test_post_ready_boundary_baseexception_finishes_publication(monkeypatch):
    proc_mesh = _PublishedProcMesh()
    starts = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        hosts = SimpleNamespace(
            shutdown=lambda: pytest.fail("attached worker loops are not client-owned"))
        handoff.fleet = hosts, proc_mesh, SimpleNamespace(), False
        return handoff.fleet

    def start():
        nonlocal starts
        starts += 1
        return True

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    interrupted = KeyboardInterrupt("lease-return boundary interrupted")
    fired = False

    def inject_after_lease(frame, event, _arg):
        nonlocal fired
        if (frame.f_code is mesh.get_mesh.__code__ and event == "line"
                and mesh._MESHES and starts == 1 and not fired):
            fired = True
            raise interrupted
        return inject_after_lease

    previous_trace = sys.gettrace()
    sys.settrace(inject_after_lease)
    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            mesh.get_mesh(mode="local", gpus_per_host=1)
    finally:
        sys.settrace(previous_trace)

    assert failure.value is interrupted
    assert starts == 1
    assert proc_mesh.stop_reasons == []
    assert proc_mesh.stop_timeouts == []
    assert len(mesh._MESHES) == 1
    assert mesh._MESH_PENDING == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()


def test_first_line_after_creation_marker_always_settles_the_latch(monkeypatch):
    spawn_calls = 0

    def spawn(*_args):
        nonlocal spawn_calls
        spawn_calls += 1
        return SimpleNamespace(), _PublishedProcMesh(), SimpleNamespace(), False

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    interrupted = KeyboardInterrupt("creation marker boundary interrupted")

    def inject_after_marker(frame, event, _arg):
        if (frame.f_code is mesh.get_mesh.__code__ and event == "line"
                and mesh._MESH_CREATING and frame.f_locals.get("fleet") is None
                and frame.f_locals.get("transport_claim") is not None):
            raise interrupted
        return inject_after_marker

    previous_trace = sys.gettrace()
    sys.settrace(inject_after_marker)
    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            mesh.get_mesh(mode="local", gpus_per_host=1)
    finally:
        sys.settrace(previous_trace)

    assert failure.value is interrupted
    assert spawn_calls == 0
    assert mesh._MESHES == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()


@pytest.mark.parametrize("boundary", ["fleet returned", "handle constructed"])
def test_prepublication_baseexception_stops_the_owned_proc_mesh(monkeypatch, boundary):
    proc_mesh = _PublishedProcMesh()
    starts = 0

    def spawn(*_args):
        hosts = SimpleNamespace(
            shutdown=lambda: pytest.fail("attached worker loops are not client-owned"))
        return hosts, proc_mesh, SimpleNamespace(), False

    def start():
        nonlocal starts
        starts += 1
        return True

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    interrupted = KeyboardInterrupt(f"{boundary} boundary interrupted")

    def inject_before_publication(frame, event, _arg):
        if frame.f_code is not mesh.get_mesh.__code__ or event != "line":
            return inject_before_publication
        fleet_returned = frame.f_locals.get("fleet") is not None
        handle_constructed = frame.f_locals.get("handle") is not None
        reached = (fleet_returned and not handle_constructed
                   if boundary == "fleet returned" else handle_constructed)
        if reached and not mesh._MESHES:
            raise interrupted
        return inject_before_publication

    previous_trace = sys.gettrace()
    sys.settrace(inject_before_publication)
    try:
        with pytest.raises(KeyboardInterrupt) as failure:
            mesh.get_mesh(mode="local", gpus_per_host=1)
    finally:
        sys.settrace(previous_trace)

    assert failure.value is interrupted
    assert starts == 0
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert mesh._MESHES == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh._TRANSPORT_POISON is None


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("boundary", ["fleet", "handle", "lease"])
def test_creation_instruction_handoffs_stop_the_owned_proc_mesh(monkeypatch, boundary):
    proc_mesh = _PublishedProcMesh()
    interrupted = KeyboardInterrupt(f"{boundary} instruction interrupted")
    starts = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        fleet = (
            SimpleNamespace(
                shutdown=lambda: pytest.fail(
                    "attached worker loops are not client-owned")),
            proc_mesh,
            SimpleNamespace(),
            False,
        )
        handoff.fleet = fleet
        return fleet

    def start():
        nonlocal starts
        starts += 1
        return True

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    instructions = list(dis.get_instructions(mesh.get_mesh))
    needle = {
        "fleet": ("LOAD_GLOBAL", "spawn_worker_fleet", "STORE_FAST", "fleet"),
        "handle": ("LOAD_GLOBAL", "MeshHandle", "STORE_FAST", "handle"),
        "lease": ("LOAD_ATTR", "start_client_lease", "POP_TOP", None),
    }[boundary]
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == needle[0] and instruction.argval == needle[1]
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = instructions[call_index + 1]
    assert target.opname == needle[2] and target.argval == needle[3]
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-fleet-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, mesh.get_mesh.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(KeyboardInterrupt) as failure:
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1)
        finally:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value is interrupted
    assert starts == (1 if boundary == "lease" else 0)
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert mesh._MESHES == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh._TRANSPORT_POISON is None


@pytest.mark.parametrize(
    "cleanup", ["not_started", "unconfirmed", "confirmed", "callback_unconfirmed"])
def test_proc_only_factory_handoff_is_reconciled_once(monkeypatch, cleanup):
    proc_mesh = _PublishedProcMesh()
    spawn_host = object()
    primary = KeyboardInterrupt(f"factory handler interrupted: {cleanup}")

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        handoff.spawn_host = spawn_host
        handoff.spawn_host_isolated = True
        handoff.proc_spawn_called = True
        proc_mesh._host_mesh = spawn_host
        handoff.proc_meshes.append(proc_mesh)
        if cleanup == "callback_unconfirmed":
            handoff.callback_cleanup_unconfirmed = True
        elif cleanup != "not_started":
            handoff.cleanup_started = True
            future = proc_mesh.stop("dgx-monarch partial bring-up rollback")
            if cleanup == "confirmed":
                future.get(timeout=60)
                handoff.cleanup_confirmed = True
        raise primary

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    with pytest.raises(KeyboardInterrupt) as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is primary
    expected_reason = ("dgx-monarch client detach"
                       if cleanup in {"not_started", "callback_unconfirmed"}
                       else "dgx-monarch partial bring-up rollback")
    assert proc_mesh.stop_reasons == [expected_reason]
    assert proc_mesh.stop_timeouts == ([30]
                                       if cleanup in {"not_started", "callback_unconfirmed"} else
                                       [60] if cleanup == "confirmed" else [])
    assert mesh._MESHES == {} and mesh._MESH_CREATING == {}
    if cleanup in {"unconfirmed", "callback_unconfirmed"}:
        expected = ("rollback outcome is unconfirmed" if cleanup == "unconfirmed"
                    else "spawn callback cleanup is unconfirmed")
        assert expected in mesh._TRANSPORT_POISON
        assert any("owned ProcMesh cleanup" in note
                   for note in getattr(primary, "__notes__", ()))
    else:
        assert mesh._TRANSPORT_POISON is None


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_outer_proc_stop_instruction_publishes_ambiguity_before_the_call_returns(
    monkeypatch,
):
    from dgx_monarch import mesh_factory

    proc_mesh = _PublishedProcMesh()
    primary = RuntimeError("handle construction failed")
    handoffs = []
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        handoffs.append(handoff)
        return fleet

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)

    def fail_handle(**_kwargs):
        raise primary

    monkeypatch.setattr(mesh, "MeshHandle", fail_handle)
    instructions = list(dis.get_instructions(
        mesh_factory.reconcile_creation_cleanup))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_ATTR" and instruction.argval == "stop"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = instructions[call_index + 1]
    interrupted = KeyboardInterrupt("ProcMesh stop return instruction interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(
                tool_id, mesh_factory.reconcile_creation_cleanup.__code__, 0)
            raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-outer-proc-stop-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, mesh_factory.reconcile_creation_cleanup.__code__,
        monitoring.events.INSTRUCTION)
    with pytest.raises(RuntimeError, match="handle construction failed") as failure:
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1)
        finally:
            monitoring.set_local_events(
                tool_id, mesh_factory.reconcile_creation_cleanup.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value is primary
    assert len(handoffs) == 1
    assert handoffs[0].cleanup_started is True
    assert handoffs[0].cleanup_confirmed is False
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == []
    assert "owned ProcMesh cleanup unconfirmed" in mesh._TRANSPORT_POISON
    with pytest.raises(mesh.MeshAttachError, match="not reusable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)
    assert spawn_calls == 1
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]


@pytest.mark.parametrize("broken", ["cleanup_repr", "primary_add_note"])
def test_outer_proc_cleanup_diagnostics_are_best_effort(monkeypatch, broken):
    class BrokenCleanup(RuntimeError):
        def __repr__(self):
            raise RuntimeError("cleanup repr unavailable")

    class BrokenNotePrimary(RuntimeError):
        def add_note(self, _note):
            raise RuntimeError("primary note unavailable")

    cleanup = (BrokenCleanup("proc stop failed") if broken == "cleanup_repr"
               else RuntimeError("proc stop failed"))
    primary = (BrokenNotePrimary("handle construction failed")
               if broken == "primary_add_note" else
               RuntimeError("handle construction failed"))
    proc_mesh = _PublishedProcMesh(cleanup)
    handoffs = []
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        handoffs.append(handoff)
        return fleet

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)

    def fail_handle(**_kwargs):
        raise primary

    monkeypatch.setattr(mesh, "MeshHandle", fail_handle)
    with pytest.raises(RuntimeError, match="handle construction failed") as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is primary
    assert len(handoffs) == 1
    assert handoffs[0].cleanup_started is True
    assert handoffs[0].cleanup_confirmed is False
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert "owned ProcMesh cleanup unconfirmed" in mesh._TRANSPORT_POISON
    if broken == "cleanup_repr":
        assert "<BrokenCleanup>" in mesh._TRANSPORT_POISON
    with pytest.raises(mesh.MeshAttachError, match="not reusable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)
    assert spawn_calls == 1
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]


def test_worker_spawn_error_with_broken_text_still_poisons_before_diagnostics(
    monkeypatch,
):
    from dgx_monarch.mesh_factory import WorkerSpawnError

    class BrokenOriginal(BaseException):
        def __str__(self):
            raise RuntimeError("original str unavailable")

        def __repr__(self):
            raise RuntimeError("original repr unavailable")

    primary = BrokenOriginal()
    spawn_error = WorkerSpawnError(
        primary, None, local_transport_initialized=True)
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        handoff.cleanup_started = True
        handoff.cleanup_confirmed = True
        raise spawn_error

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    with pytest.raises(BrokenOriginal) as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is primary
    assert failure.value.__cause__ is None
    assert "actor spawn failed" in mesh._TRANSPORT_POISON
    assert "<BrokenOriginal>" in mesh._TRANSPORT_POISON
    with pytest.raises(mesh.MeshAttachError, match="not reusable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)
    assert spawn_calls == 1


@pytest.mark.parametrize("cleanup_wins", [False, True])
def test_worker_spawn_cancellation_survives_mesh_composition(
    monkeypatch, cleanup_wins,
):
    from dgx_monarch.mesh_factory import WorkerSpawnError

    original = (RuntimeError("ordinary worker spawn failure") if cleanup_wins
                else KeyboardInterrupt("worker spawn cancelled"))
    cleanup = (KeyboardInterrupt("spawn rollback cancelled") if cleanup_wins
               else RuntimeError("ordinary rollback failure"))
    winner = cleanup if cleanup_wins else original
    cause = original if cleanup_wins else cleanup
    spawn_error = WorkerSpawnError(
        original, cleanup, local_transport_initialized=True)
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        handoff.cleanup_started = True
        handoff.cleanup_confirmed = True
        raise spawn_error

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    with pytest.raises(type(winner)) as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is winner
    assert failure.value.__cause__ is cause
    assert failure.value.__cause__ is not failure.value
    assert "actor spawn failed" in mesh._TRANSPORT_POISON
    assert spawn_calls == 1


def test_active_creation_claim_serializes_an_unrelated_key(
    monkeypatch,
):
    import threading

    from dgx_monarch import mesh_factory

    acquired = threading.Event()
    release_failure = threading.Event()
    proc_meshes: list[_PublishedProcMesh] = []
    primary = RuntimeError("first handle construction failed")
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        proc_mesh = _PublishedProcMesh()
        proc_meshes.append(proc_mesh)
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        return fleet

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    monkeypatch.setattr(mesh, "_detect_comfy_dir", lambda path: path)
    monkeypatch.setattr(
        mesh, "_mesh_cache_key",
        lambda _source, comfy, _cluster: ("test", comfy))

    def fail_handle(**_kwargs):
        acquired.set()
        assert release_failure.wait(timeout=3)
        raise primary

    monkeypatch.setattr(mesh, "MeshHandle", fail_handle)
    errors = []

    def create_first():
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1, comfy_dir="/first")
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=create_first, name="first-creator")
    first.start()
    assert acquired.wait(timeout=3)
    assert mesh_factory.creation_cleanup_pending() is False
    with pytest.raises(
        mesh.MeshAttachError,
        match="another mesh creation is still in progress",
    ):
        mesh.get_mesh(mode="local", gpus_per_host=1, comfy_dir="/second")
    assert spawn_calls == 1
    release_failure.set()
    first.join(timeout=3)
    assert not first.is_alive()

    assert errors == [primary]
    assert all(
        proc_mesh.stop_reasons == ["dgx-monarch client detach"]
        for proc_mesh in proc_meshes)
    assert all(proc_mesh.stop_timeouts == [30] for proc_mesh in proc_meshes)
    assert mesh_factory.creation_cleanup_pending() is False
    assert mesh._TRANSPORT_POISON is None


def test_confirmed_factory_cleanup_still_blocks_until_transport_verdict(
    monkeypatch,
):
    import threading

    from dgx_monarch import mesh_factory
    from dgx_monarch.mesh_factory import WorkerSpawnError

    cleanup_confirmed = threading.Event()
    release_failure = threading.Event()
    proc_mesh = _PublishedProcMesh()
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        if spawn_calls != 1:
            pytest.fail("a second key crossed an unresolved transport verdict")
        handoff.proc_meshes.append(proc_mesh)
        handoff.spawn_in_progress = True
        handoff.cleanup_started = True
        handoff.cleanup_confirmed = True
        cleanup_confirmed.set()
        assert release_failure.wait(timeout=3)
        raise WorkerSpawnError(
            RuntimeError("actor spawn failed after rollback"), None,
            local_transport_initialized=True)

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    monkeypatch.setattr(mesh, "_detect_comfy_dir", lambda path: path)
    monkeypatch.setattr(
        mesh, "_mesh_cache_key",
        lambda _source, comfy, _cluster: ("test", comfy))
    errors: list[BaseException] = []

    def create_first():
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1, comfy_dir="/first")
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=create_first, name="first-creator")
    first.start()
    try:
        assert cleanup_confirmed.wait(timeout=3)
        assert mesh_factory.creation_cleanup_pending() is True
        with pytest.raises(mesh.MeshAttachError, match="not reusable"):
            mesh.get_mesh(
                mode="local", gpus_per_host=1, comfy_dir="/second")
        assert spawn_calls == 1
    finally:
        release_failure.set()
        first.join(timeout=3)

    assert not first.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], mesh.MeshAttachError)
    assert "bring-up failed" in str(errors[0])
    assert proc_mesh.stop_reasons == []
    assert "actor spawn failed" in mesh._TRANSPORT_POISON
    assert mesh_factory.creation_cleanup_pending() is False


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_preacquisition_claim_registration_boundary_never_reaches_spawn(
    monkeypatch,
):
    from dgx_monarch import mesh_factory

    proc_mesh = _PublishedProcMesh()
    handoffs = []
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        handoffs.append(handoff)
        return fleet

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    instructions = list(dis.get_instructions(mesh.get_mesh))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_ATTR"
        and instruction.argval == "begin_creation_cleanup"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = instructions[call_index + 1]
    assert target.opname == "POP_TOP"
    interrupted = KeyboardInterrupt("cleanup claim return interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-cleanup-claim-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, mesh.get_mesh.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(KeyboardInterrupt) as failure:
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1)
        finally:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value is interrupted
    assert mesh._TRANSPORT_POISON is None
    assert mesh._TRANSPORT_MODE is None
    assert mesh_factory.creation_cleanup_pending() is False
    assert handoffs == []
    assert proc_mesh.stop_reasons == []
    assert spawn_calls == 0

    monkeypatch.setattr(mesh, "find_config_path", lambda path: path or None)
    monkeypatch.setattr(
        mesh, "load_cluster_config",
        lambda path: SimpleNamespace(
            source=path, comfy_dir=None,
            hosts=[SimpleNamespace(gpus=1)]))
    monkeypatch.setattr(mesh, "config_fingerprint", lambda _path: "cluster")
    handle = mesh.get_mesh(
        mode="cluster", config_path="/fake/cluster.toml")
    assert isinstance(handle, mesh.MeshHandle)
    assert mesh._TRANSPORT_MODE == "cluster"
    assert spawn_calls == 1
    assert proc_mesh.stop_reasons == []


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_preacquisition_claim_object_boundary_has_no_owned_proc_mesh(monkeypatch):
    proc_mesh = _PublishedProcMesh()
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        return fleet

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    instructions = list(dis.get_instructions(mesh.get_mesh))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname == "LOAD_GLOBAL"
        and instruction.argval == "object"
        and any(
            later.opname == "STORE_FAST"
            and later.argval == "cleanup_claim"
            for later in instructions[index + 1:index + 5]
        )
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    target = next(
        instruction for instruction in instructions[call_index + 1:]
        if instruction.opname == "STORE_FAST"
        and instruction.argval == "cleanup_claim"
    )
    assert target.opname == "STORE_FAST" and target.argval == "cleanup_claim"
    interrupted = KeyboardInterrupt("cleanup claim object return interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-cleanup-claim-object-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, mesh.get_mesh.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(KeyboardInterrupt) as failure:
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1)
        finally:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    from dgx_monarch import mesh_factory

    assert failure.value is interrupted
    assert spawn_calls == 0
    assert proc_mesh.stop_reasons == []
    assert mesh_factory.creation_cleanup_pending() is False
    assert mesh._TRANSPORT_POISON is None
    assert mesh._MESHES == {} and mesh._MESH_CREATING == {}


def test_success_claim_confirmation_retry_opens_other_keys(monkeypatch):
    from dgx_monarch import mesh_factory

    proc_meshes: list[_PublishedProcMesh] = []
    starts = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        proc_mesh = _PublishedProcMesh()
        proc_meshes.append(proc_mesh)
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        return fleet

    def start():
        nonlocal starts
        starts += 1
        return True

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    monkeypatch.setattr(mesh, "_detect_comfy_dir", lambda path: path)
    monkeypatch.setattr(
        mesh, "_mesh_cache_key",
        lambda _source, comfy, _cluster: ("test", comfy))
    interrupted = KeyboardInterrupt("success claim return interrupted")
    real_confirm = mesh_factory.confirm_creation_cleanup
    confirmations = 0

    def interrupt_after_confirm(claim):
        nonlocal confirmations
        result = real_confirm(claim)
        confirmations += 1
        if confirmations == 1:
            raise interrupted
        return result

    monkeypatch.setattr(
        mesh_factory, "confirm_creation_cleanup", interrupt_after_confirm)
    with pytest.raises(KeyboardInterrupt) as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1, comfy_dir="/first")

    assert failure.value is interrupted
    result = mesh.get_mesh(
        mode="local", gpus_per_host=1, comfy_dir="/second")
    assert result is not None and confirmations == 3
    assert starts == 2 and len(proc_meshes) == 2
    assert all(proc_mesh.stop_reasons == [] for proc_mesh in proc_meshes)
    assert len(mesh._MESHES) == 2
    assert mesh_factory.creation_cleanup_pending() is False
    assert mesh._MESH_CREATING == {} and mesh._TRANSPORT_CLAIMS == set()


def test_new_cleanup_claim_between_confirm_and_poison_clear_still_blocks(
    monkeypatch,
):
    from dgx_monarch import mesh_factory

    proc_mesh = _PublishedProcMesh()
    primary = RuntimeError("handle construction failed")
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        return fleet

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)

    def fail_handle(**_kwargs):
        raise primary

    monkeypatch.setattr(mesh, "MeshHandle", fail_handle)
    real_confirm = mesh_factory.confirm_creation_cleanup
    raced_claims = []

    def confirm_then_race(claim):
        empty = real_confirm(claim)
        assert empty is True
        raced_claims.append(mesh_factory.begin_creation_cleanup())
        return empty

    monkeypatch.setattr(
        mesh_factory, "confirm_creation_cleanup", confirm_then_race)
    with pytest.raises(RuntimeError, match="handle construction failed") as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is primary
    assert raced_claims and mesh_factory.creation_cleanup_pending() is True
    assert mesh._TRANSPORT_POISON is None
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    with pytest.raises(mesh.MeshAttachError, match="not reusable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)
    assert spawn_calls == 1


def test_broken_handle_cleanup_attribute_cannot_open_the_creation_gate(monkeypatch):
    proc_mesh = _PublishedProcMesh()
    primary = RuntimeError("client lease start failed")
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        fleet = SimpleNamespace(), proc_mesh, SimpleNamespace(), False
        handoff.fleet = fleet
        return fleet

    class BrokenHandle:
        teardown_complete = False

        @property
        def replacement_blocked(self):
            raise RuntimeError("replacement state unreadable")

    mesh = _mesh_publication_fakes(
        monkeypatch, spawn, lambda: (_ for _ in ()).throw(primary))
    monkeypatch.setattr(mesh, "MeshHandle", lambda **_kwargs: BrokenHandle())
    with pytest.raises(RuntimeError, match="replacement state unreadable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)

    from dgx_monarch import mesh_factory

    assert mesh_factory.creation_cleanup_pending() is True
    assert proc_mesh.stop_reasons == []
    with pytest.raises(mesh.MeshAttachError, match="not reusable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)
    assert spawn_calls == 1


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_spawn_failure_poison_precedes_transport_evidence_read(monkeypatch):
    from dgx_monarch.mesh_factory import WorkerSpawnError

    spawn_error = WorkerSpawnError(
        RuntimeError("actor spawn failed"), None,
        local_transport_initialized=True)
    spawn_calls = 0

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        nonlocal spawn_calls
        spawn_calls += 1
        handoff.cleanup_started = True
        handoff.cleanup_confirmed = True
        raise spawn_error

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    instructions = list(dis.get_instructions(mesh.get_mesh))
    target = next(
        instruction for instruction in instructions
        if instruction.opname == "LOAD_ATTR"
        and instruction.argval == "local_transport_initialized"
    )
    interrupted = KeyboardInterrupt("spawn evidence read interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target.offset:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-spawn-poison-publication-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, mesh.get_mesh.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(KeyboardInterrupt) as failure:
        try:
            mesh.get_mesh(mode="local", gpus_per_host=1)
        finally:
            monitoring.set_local_events(tool_id, mesh.get_mesh.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert failure.value is interrupted
    assert "actor spawn failed" in mesh._TRANSPORT_POISON
    with pytest.raises(mesh.MeshAttachError, match="not reusable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)
    assert spawn_calls == 1


def test_lease_start_failure_with_unconfirmed_proc_cleanup_blocks_every_retry(monkeypatch):
    from dgx_monarch import mesh_factory

    proc_mesh = _PublishedProcMesh(RuntimeError("proc stop was not confirmed"))
    spawn_calls = 0
    start_calls = 0
    pending: list[object] = []

    def spawn(*_args):
        nonlocal spawn_calls
        spawn_calls += 1
        hosts = SimpleNamespace(
            shutdown=lambda: pytest.fail("attached worker loops are not client-owned"))
        return hosts, proc_mesh, SimpleNamespace(), False

    def start():
        nonlocal start_calls
        start_calls += 1
        assert mesh._MESHES == {} and len(mesh._MESH_PENDING) == 1
        pending.extend(mesh._MESH_PENDING.values())
        raise RuntimeError("client lease thread did not start")

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    with pytest.raises(RuntimeError, match="client lease thread did not start") as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert mesh._MESHES == {}
    assert mesh._MESH_PENDING == {}
    assert client_lease.registry_snapshot() == []
    assert len(pending) == 1
    retained = pending[0]
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert retained.teardown_complete is False
    assert "proc stop was not confirmed" in retained.replacement_blocked
    assert mesh_factory.creation_cleanup_pending() is True
    assert any("owned ProcMesh cleanup" in note
               for note in getattr(failure.value, "__notes__", ()))
    assert "owned ProcMesh cleanup unconfirmed" in mesh._TRANSPORT_POISON

    with pytest.raises(mesh.MeshAttachError, match="not reusable"):
        mesh.get_mesh(mode="local", gpus_per_host=1)
    assert spawn_calls == 1 and start_calls == 1
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]


def test_confirmed_stop_outranks_an_interrupted_cache_eviction(monkeypatch):
    proc_mesh = _PublishedProcMesh()
    startup_error = RuntimeError("client lease thread did not start")
    published: list[object] = []

    def spawn(*_args):
        hosts = SimpleNamespace(
            shutdown=lambda: pytest.fail("attached worker loops are not client-owned"))
        return hosts, proc_mesh, SimpleNamespace(), False

    def start():
        assert mesh._MESHES == {}
        published.extend(mesh._MESH_PENDING.values())
        raise startup_error

    mesh = _mesh_publication_fakes(monkeypatch, spawn, start)
    original_retire = mesh.MeshHandle._retire_completed
    retire_calls = 0

    def interrupt_first_retirement(handle):
        nonlocal retire_calls
        retire_calls += 1
        if retire_calls == 1:
            raise KeyboardInterrupt("cache eviction interrupted after confirmed stop")
        return original_retire(handle)

    monkeypatch.setattr(mesh.MeshHandle, "_retire_completed", interrupt_first_retirement)
    with pytest.raises(RuntimeError) as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is startup_error
    assert retire_calls == 2
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert published[0].teardown_complete is True
    assert published[0].replacement_blocked is None
    assert mesh._MESHES == {}
    assert mesh._TRANSPORT_POISON is None
    assert getattr(failure.value, "__notes__", ()) == ()
