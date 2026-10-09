"""Recovery regressions for owner-tagged mesh creation attempts."""

from __future__ import annotations

import dis
import linecache
import sys
import threading
import types

import pytest

from dgx_monarch import client_lease, mesh, mesh_creation, mesh_factory


class _Interrupt(BaseException):
    """Identity-bearing stand-in for asynchronous cancellation."""


class _StopResult:
    def __init__(self, owner: _ProcMesh) -> None:
        self._owner = owner

    def get(self, timeout=None):
        self._owner.stop_timeouts.append(timeout)


class _ProcMesh:
    def __init__(self) -> None:
        self.stop_reasons: list[str] = []
        self.stop_timeouts: list[float | None] = []

    def stop(self, reason: str) -> _StopResult:
        self.stop_reasons.append(reason)
        return _StopResult(self)


class _Workers:
    def __init__(self) -> None:
        self.lease_casts: list[float] = []
        self.renew_client_lease = types.SimpleNamespace(
            broadcast=self.lease_casts.append)


@pytest.fixture
def isolated_mesh_runtime(monkeypatch):
    """Give get_mesh fresh process-global registries without replacing locks."""
    monkeypatch.setattr(mesh, "install_fault_hook", lambda: None)
    monkeypatch.setattr(mesh, "find_config_path", lambda _path: None)
    monkeypatch.setattr(mesh, "_detect_comfy_dir", lambda _path: "/fake/comfy")
    monkeypatch.setattr(mesh, "_visible_gpu_count", lambda: 1)
    monkeypatch.setattr(mesh, "_src_pythonpath", lambda: "/fake/repo/src")
    monkeypatch.setattr(mesh.client_lease, "plant_driver_marker", lambda: "")
    monkeypatch.setattr(
        mesh,
        "local_config",
        lambda _path=None: types.SimpleNamespace(
            source="local", comfy_dir=None, worker_args={}),
    )
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
    monkeypatch.setenv("DGXM_PYTHONPATH", "/before-test")
    monkeypatch.setenv("PYTHONPATH", "/before-test")
    return mesh


def _fleet(proc_mesh: _ProcMesh, workers: _Workers | None = None):
    hosts = types.SimpleNamespace(
        shutdown=lambda: pytest.fail(
            "attached worker loops are not owned by the client"))
    return hosts, proc_mesh, workers or _Workers(), False


def _spawn_returning(proc_mesh: _ProcMesh, workers: _Workers | None = None):
    fleet = _fleet(proc_mesh, workers)

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        handoff.fleet = fleet
        return fleet

    return spawn


def _next_instruction(instructions, target):
    index = instructions.index(target)
    return instructions[index + 1]


def _instruction_after_ready_boundary(boundary: str) -> int:
    instructions = list(dis.get_instructions(mesh_creation.publish_ready_handle))
    if boundary == "marker":
        store = [
            instruction
            for instruction in instructions
            if instruction.opname == "STORE_ATTR"
            and instruction.argval == "ready_for_publication"
        ][-1]
        return _next_instruction(instructions, store).offset
    if boundary == "public":
        store = [
            instruction
            for instruction in instructions
            if instruction.opname == "STORE_SUBSCR"
        ][-1]
        return _next_instruction(instructions, store).offset
    if boundary == "pending-pop":
        pop_load = [
            index
            for index, instruction in enumerate(instructions)
            if instruction.opname == "LOAD_ATTR" and instruction.argval == "pop"
        ][-1]
        call = next(
            instruction
            for instruction in instructions[pop_load + 1 :]
            if instruction.opname == "CALL"
        )
        return _next_instruction(instructions, call).offset
    raise AssertionError(f"unknown READY boundary: {boundary}")


def _run_with_instruction_interrupt(function, offset: int, interrupted, call):
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == offset:
            monitoring.set_local_events(tool_id, function.__code__, 0)
            raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-mesh-creation-recovery-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, function.__code__, monitoring.events.INSTRUCTION)
    try:
        return call()
    finally:
        monitoring.set_local_events(tool_id, function.__code__, 0)
        monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)


def test_mesh_selection_claim_reports_only_the_fresh_creator(
    isolated_mesh_runtime, monkeypatch,
):
    proc_mesh = _ProcMesh()
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(proc_mesh))
    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", lambda: True)

    creator_claim = object()
    first = mesh.get_mesh(
        mode="local", gpus_per_host=1, _selection_claim=creator_claim)
    cached_claim = object()
    second = mesh.get_mesh(
        mode="local", gpus_per_host=1, _selection_claim=cached_claim)

    assert first is second
    assert first._selection_claim is creator_claim
    assert second._selection_claim is not cached_claim
    assert proc_mesh.stop_reasons == []


def test_local_mesh_is_replaced_when_worker_args_change(
    isolated_mesh_runtime, monkeypatch,
):
    """A local mesh consumes `[worker_args]`, so a change to them replaces it.

    Without a digest on the local path the first attach's arguments outlive
    every later edit: the cached handle comes back untouched and the workers
    keep running the arguments they started with.
    """
    proc_mesh = _ProcMesh()
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(proc_mesh))
    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", lambda: True)
    monkeypatch.setattr(mesh, "find_config_path", lambda _path=None: "/cluster.toml")
    live = {"swap_verify": 2}
    monkeypatch.setattr(
        mesh,
        "local_config",
        lambda _path=None: types.SimpleNamespace(
            source="local", comfy_dir=None, worker_args=dict(live)),
    )

    first = mesh.get_mesh(mode="local", gpus_per_host=1)
    assert mesh.get_mesh(mode="local", gpus_per_host=1) is first

    live["swap_verify"] = 7
    assert mesh.get_mesh(mode="local", gpus_per_host=1) is not first


def test_local_worker_args_fingerprint_tracks_values_not_order():
    fingerprint = mesh.mesh_helpers.local_worker_args_fingerprint
    assert fingerprint({"a": True, "b": 2}) == fingerprint({"b": 2, "a": True})
    assert fingerprint({"a": True}) != fingerprint({"a": False})
    assert fingerprint({}) != fingerprint({"a": True})


def test_an_empty_worker_args_table_matches_having_no_config_at_all():
    """Both hand the workers nothing, so neither may recycle a live fleet.

    `get_mesh` stores the plain "local" digest when it finds no config, so a
    file that turns up later carrying an empty table has to agree with it.
    """
    assert mesh.mesh_helpers.local_worker_args_fingerprint({}) == "local"


def test_local_mesh_survives_a_config_that_adds_no_worker_args(
    isolated_mesh_runtime, monkeypatch,
):
    proc_mesh = _ProcMesh()
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(proc_mesh))
    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", lambda: True)
    found: list[str | None] = [None]
    monkeypatch.setattr(mesh, "find_config_path", lambda _path=None: found[0])

    first = mesh.get_mesh(mode="local", gpus_per_host=1)
    found[0] = "/cluster.toml"          # a default config appears with no [worker_args]
    assert mesh.get_mesh(mode="local", gpus_per_host=1) is first
    assert proc_mesh.stop_reasons == []


def test_mesh_selection_claim_does_not_transfer_to_a_waiting_reuser(
    isolated_mesh_runtime, monkeypatch,
):
    proc_mesh = _ProcMesh()
    creator_ready = threading.Event()
    release_creator = threading.Event()
    waiter_started = threading.Event()
    waiter_done = threading.Event()
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(proc_mesh))

    def start_client_lease() -> bool:
        creator_ready.set()
        assert release_creator.wait(timeout=2)
        return True

    monkeypatch.setattr(
        mesh.mesh_helpers, "start_client_lease", start_client_lease)
    claims = [object(), object()]
    selected: list[tuple[object, bool]] = []

    def select(index: int) -> None:
        if index == 1:
            waiter_started.set()
        handle = mesh.get_mesh(
            mode="local", gpus_per_host=1, _selection_claim=claims[index])
        selected.append((handle, handle._selection_claim is claims[index]))
        if index == 1:
            waiter_done.set()

    creator = threading.Thread(target=select, args=(0,))
    waiter = threading.Thread(target=select, args=(1,))
    creator.start()
    assert creator_ready.wait(timeout=2)
    waiter.start()
    assert waiter_started.wait(timeout=2)
    assert not waiter_done.wait(timeout=0.05)
    release_creator.set()
    creator.join(timeout=3)
    waiter.join(timeout=3)

    assert not creator.is_alive() and not waiter.is_alive()
    assert len(selected) == 2
    assert selected[0][0] is selected[1][0]
    assert sorted(created_here for _handle, created_here in selected) == [False, True]
    assert proc_mesh.stop_reasons == []


def test_late_transport_poison_refuses_ready_and_cleans_private_handle(
    isolated_mesh_runtime, monkeypatch,
):
    proc_mesh = _ProcMesh()
    pending: list[object] = []
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(proc_mesh))

    def poison_before_ready():
        pending.extend(mesh._MESH_PENDING.values())
        mesh._TRANSPORT_POISON = "late transport poison"
        return True

    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", poison_before_ready)

    with pytest.raises(mesh.MeshAttachError, match="transport verdict changed"):
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert len(pending) == 1
    assert pending[0].teardown_complete is True
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert mesh._MESHES == {}
    assert mesh._MESH_PENDING == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh._TRANSPORT_POISON == "late transport poison"


def test_failed_pending_creation_is_evicted_after_confirmed_stop_and_retries(
    isolated_mesh_runtime, monkeypatch,
):
    first_proc = _ProcMesh()
    second_proc = _ProcMesh()
    fleets = iter((_fleet(first_proc), _fleet(second_proc)))
    primary = RuntimeError("client lease startup failed")
    retirement_failure = RuntimeError("completed-handle eviction interrupted")
    pending: list[object] = []

    def spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        fleet = next(fleets)
        handoff.fleet = fleet
        return fleet

    lease_calls = 0

    def start_lease():
        nonlocal lease_calls
        lease_calls += 1
        if lease_calls == 1:
            pending.extend(mesh._MESH_PENDING.values())
            raise primary
        return True

    retire_calls = 0
    real_retire = mesh.MeshHandle._retire_completed

    def fail_retire(_handle):
        nonlocal retire_calls
        retire_calls += 1
        raise retirement_failure

    monkeypatch.setattr(mesh, "spawn_worker_fleet", spawn)
    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", start_lease)
    monkeypatch.setattr(mesh.MeshHandle, "_retire_completed", fail_retire)

    with pytest.raises(RuntimeError) as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is primary
    assert len(pending) == 1
    assert pending[0].teardown_complete is True
    assert retire_calls == 2
    assert first_proc.stop_reasons == ["dgx-monarch client detach"]
    assert mesh._MESHES == {}
    assert mesh._MESH_PENDING == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh_factory.creation_cleanup_pending() is False

    monkeypatch.setattr(mesh.MeshHandle, "_retire_completed", real_retire)
    handle = mesh.get_mesh(mode="local", gpus_per_host=1)

    assert handle.procs is second_proc
    assert first_proc.stop_reasons == ["dgx-monarch client detach"]
    assert second_proc.stop_reasons == []
    assert mesh._MESH_PENDING == {}
    assert list(mesh._MESHES.values()) == [handle]


def test_hostile_ownership_recovery_cannot_replace_primary_failure(
    isolated_mesh_runtime, monkeypatch,
):
    primary = RuntimeError("actor spawn failed first")
    ownership_interrupt = _Interrupt("ownership accessor interrupted")

    def fail_spawn(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        handoff.proc_spawn_called = True
        handoff.spawn_in_progress = True
        raise primary

    ownership_calls = 0

    def fail_ownership(_handoff):
        nonlocal ownership_calls
        ownership_calls += 1
        raise ownership_interrupt

    monkeypatch.setattr(mesh, "spawn_worker_fleet", fail_spawn)
    monkeypatch.setattr(mesh_factory, "owned_spawned_proc", fail_ownership)

    with pytest.raises(RuntimeError) as failure:
        mesh.get_mesh(mode="local", gpus_per_host=1)

    assert failure.value is primary
    assert ownership_calls == 1
    assert any(
        "ownership recovery also failed" in note
        and "ownership accessor interrupted" in note
        for note in getattr(primary, "__notes__", ())
    )
    assert mesh._MESHES == {}
    assert mesh._MESH_PENDING == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh._TRANSPORT_POISON is not None


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_failure_latch_store_interruption_cleans_returned_fleet_and_wins(
    isolated_mesh_runtime, monkeypatch,
):
    proc_mesh = _ProcMesh()
    ordinary = RuntimeError("handle construction failed after fleet return")
    interrupted = _Interrupt("failure-latch store interrupted")
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(proc_mesh))

    def refuse_handle(*_args, **_kwargs):
        raise ordinary

    monkeypatch.setattr(mesh, "MeshHandle", refuse_handle)
    target = next(
        instruction.offset
        for instruction in dis.get_instructions(mesh.get_mesh)
        if instruction.opname == "STORE_ATTR"
        and instruction.argval == "failure_latched"
    )

    with pytest.raises(_Interrupt) as failure:
        _run_with_instruction_interrupt(
            mesh.get_mesh,
            target,
            interrupted,
            lambda: mesh.get_mesh(mode="local", gpus_per_host=1),
        )

    assert failure.value is interrupted
    assert failure.value is not ordinary
    assert proc_mesh.stop_reasons == ["dgx-monarch client detach"]
    assert proc_mesh.stop_timeouts == [30]
    assert mesh._MESHES == {}
    assert mesh._MESH_PENDING == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh_factory.creation_cleanup_pending() is False


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
def test_bind_evidence_interruption_cannot_release_cluster_transport(
    isolated_mesh_runtime, monkeypatch,
):
    config = types.SimpleNamespace(
        source="/fake/cluster.toml",
        comfy_dir=None,
        worker_args={},
        client_bind="tcp://10.0.0.1:0",
        hosts=[types.SimpleNamespace(gpus=1)],
    )
    monkeypatch.setattr(mesh, "find_config_path", lambda _path: config.source)
    monkeypatch.setattr(mesh, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh, "config_fingerprint", lambda _path: "config-digest")

    def fail_after_bind(_config, _cluster, _gph, _attach, _bootstrap, handoff):
        mesh._TRANSPORT_BIND = config.client_bind
        handoff.attach_in_progress = True
        mesh_factory.safe_transport_failure(handoff)
        raise mesh.MeshAttachError("pre-attach failure after transport enable")

    monkeypatch.setattr(mesh, "spawn_worker_fleet", fail_after_bind)
    target = max(
        instruction.offset
        for instruction in dis.get_instructions(mesh.get_mesh)
        if instruction.opname == "STORE_FAST"
        and instruction.argval == "transport_releasable"
        and instruction.positions.lineno is not None
        and "transport_releasable = False"
        in linecache.getline(mesh.__file__, instruction.positions.lineno)
    )
    interrupted = _Interrupt("transport evidence publication interrupted")

    with pytest.raises(_Interrupt) as failure:
        _run_with_instruction_interrupt(
            mesh.get_mesh,
            target,
            interrupted,
            lambda: mesh.get_mesh(
                mode="cluster", config_path=config.source),
        )

    assert failure.value is interrupted
    assert mesh._TRANSPORT_MODE == "cluster"
    assert mesh._TRANSPORT_BIND == config.client_bind
    assert mesh._TRANSPORT_POISON is None
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh._MESH_CREATING == {}
    assert mesh_factory.creation_cleanup_pending() is False


def test_finalizer_retry_cannot_aba_remove_a_newer_attempt_or_pending_handle(
    isolated_mesh_runtime, monkeypatch,
):
    key = ("same-cache-key",)
    old_claim = (1, "local", object())
    new_claim = (1, "local", object())
    old = mesh_creation.CreationAttempt(
        key,
        sys._getframe(),
        1,
        mesh_factory.FleetHandoff(),
        object(),
        old_claim,
        None,
    )
    newer = mesh_creation.CreationAttempt(
        key,
        sys._getframe(),
        2,
        mesh_factory.FleetHandoff(),
        object(),
        new_claim,
        None,
    )
    old_handle = types.SimpleNamespace(
        _creation_phase="building",
        teardown_complete=True,
        replacement_blocked=None,
    )
    newer_handle = types.SimpleNamespace(_creation_phase="building")
    old.handle = old_handle
    newer.handle = newer_handle
    mesh._MESH_CREATING[key] = old
    mesh._MESH_PENDING[key] = old_handle
    mesh._TRANSPORT_MODE = "local"
    mesh._TRANSPORT_CLAIM_GENERATION = 1
    mesh._TRANSPORT_CLAIMS.update((old_claim, new_claim))
    interrupted = _Interrupt("first finalizer return interrupted")
    real_finalize_once = mesh_creation.finalize_creation_once
    calls = 0

    def interrupt_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_finalize_once(*args, **kwargs)
        if calls == 1:
            assert key not in mesh._MESH_CREATING
            assert key not in mesh._MESH_PENDING
            mesh._MESH_CREATING[key] = newer
            mesh._MESH_PENDING[key] = newer_handle
            raise interrupted
        return result

    monkeypatch.setattr(
        mesh_creation, "finalize_creation_once", interrupt_after_first)

    with pytest.raises(_Interrupt) as failure:
        mesh_creation.finalize_creation(
            mesh.__dict__, old, old_handle, False, False, None)

    assert failure.value is interrupted
    assert calls == 2
    assert mesh._MESH_CREATING == {key: newer}
    assert mesh._MESH_PENDING == {key: newer_handle}
    assert old_claim not in mesh._TRANSPORT_CLAIMS
    assert new_claim in mesh._TRANSPORT_CLAIMS


def _exited_frame_attempt(key, handoff, cleanup_claim, transport_claim):
    return mesh_creation.CreationAttempt(
        key=key,
        owner_frame=sys._getframe(),
        owner_thread_id=__import__("threading").get_ident(),
        handoff=handoff,
        cleanup_claim=cleanup_claim,
        transport_claim=transport_claim,
        expected_bind=None,
    )


def test_abandoned_reaper_propagates_exact_ownership_cancellation(
    isolated_mesh_runtime, monkeypatch,
):
    key = ("abandoned-ownership-cancellation",)
    handoff = mesh_factory.FleetHandoff()
    handoff.proc_spawn_called = True
    handoff.spawn_in_progress = True
    cleanup_claim = object()
    transport_claim = (1, "local", object())
    attempt = _exited_frame_attempt(
        key, handoff, cleanup_claim, transport_claim)
    interrupted = KeyboardInterrupt("ownership recovery interrupted")
    ownership_calls = 0

    def fail_ownership(_handoff):
        nonlocal ownership_calls
        ownership_calls += 1
        raise interrupted

    assert mesh_creation.attempt_is_active(attempt) is False
    mesh._MESH_CREATING[key] = attempt
    mesh._TRANSPORT_MODE = "local"
    mesh._TRANSPORT_CLAIM_GENERATION = 1
    mesh._TRANSPORT_CLAIMS.add(transport_claim)
    mesh_factory.begin_creation_cleanup(
        cleanup_claim, handoff, reserve_transport=True)
    monkeypatch.setattr(mesh_factory, "owned_spawned_proc", fail_ownership)

    with pytest.raises(KeyboardInterrupt) as failure:
        mesh_creation.reap_abandoned_attempts(mesh.__dict__)

    assert failure.value is interrupted
    assert getattr(interrupted, "__notes__", ()) == ()
    assert ownership_calls == 1
    assert handoff.proc_ownership_unconfirmed is True
    assert mesh._MESH_CREATING == {}
    assert mesh._MESH_PENDING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert mesh._TRANSPORT_POISON is not None
    assert mesh_factory.creation_cleanup_registered(cleanup_claim) is True


def test_next_get_mesh_terminally_reconciles_an_exited_creator_frame(
    isolated_mesh_runtime, monkeypatch,
):
    stale_proc = _ProcMesh()
    fresh_proc = _ProcMesh()
    stale_handoff = mesh_factory.FleetHandoff()
    stale_handoff.fleet = _fleet(stale_proc)
    cleanup_claim = object()
    transport_claim = (1, "local", object())
    key = ("abandoned-key",)
    stale = _exited_frame_attempt(
        key, stale_handoff, cleanup_claim, transport_claim)
    assert mesh_creation.attempt_is_active(stale) is False
    mesh._MESH_CREATING[key] = stale
    mesh._TRANSPORT_MODE = "local"
    mesh._TRANSPORT_CLAIM_GENERATION = 1
    mesh._TRANSPORT_CLAIMS.add(transport_claim)
    mesh_factory.begin_creation_cleanup(
        cleanup_claim, stale_handoff, reserve_transport=True)
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(fresh_proc))
    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", lambda: True)

    handle = mesh.get_mesh(mode="local", gpus_per_host=1)

    assert stale_proc.stop_reasons == ["dgx-monarch client detach"]
    assert stale_proc.stop_timeouts == [30]
    assert fresh_proc.stop_reasons == []
    assert mesh_factory.creation_cleanup_registered(cleanup_claim) is False
    assert mesh._MESH_CREATING == {}
    assert mesh._MESH_PENDING == {}
    assert list(mesh._MESHES.values()) == [handle]
    assert mesh._TRANSPORT_CLAIMS == set()


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("boundary", ["marker", "public", "pending-pop"])
def test_ready_publication_interruptions_preserve_the_exact_public_handle(
    isolated_mesh_runtime, monkeypatch, boundary,
):
    proc_mesh = _ProcMesh()
    pending: list[object] = []
    interrupted = _Interrupt(f"READY {boundary} interrupted")
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(proc_mesh))

    def start_lease():
        pending.extend(mesh._MESH_PENDING.values())
        return True

    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", start_lease)
    target = _instruction_after_ready_boundary(boundary)

    with pytest.raises(_Interrupt) as failure:
        _run_with_instruction_interrupt(
            mesh_creation.publish_ready_handle,
            target,
            interrupted,
            lambda: mesh.get_mesh(mode="local", gpus_per_host=1),
        )

    assert failure.value is interrupted
    assert len(pending) == 1
    exact = pending[0]
    assert list(mesh._MESHES.values()) == [exact]
    assert mesh._MESH_PENDING == {}
    assert mesh._MESH_CREATING == {}
    assert mesh._TRANSPORT_CLAIMS == set()
    assert exact._creation_phase == "ready"
    assert proc_mesh.stop_reasons == []
    assert mesh.get_mesh(mode="local", gpus_per_host=1) is exact


def test_pending_handle_in_rollback_phase_is_not_renewed():
    workers = _Workers()
    handle = types.SimpleNamespace(
        workers=workers,
        defunct=False,
        teardown_complete=False,
        replacement_blocked=None,
        setup_cleanup_state=None,
        sample_leases={},
        abandoned_sample_leases={},
        _creation_phase="rollback",
    )

    assert client_lease.skip_reason(handle) == "mesh creation rollback in progress"
    assert client_lease.renew_pass([handle], 300.0, failures={}) == 0
    assert workers.lease_casts == []


def _run_with_repeated_interrupt(function, offset: int, interrupted, call, fires: int):
    """`_run_with_instruction_interrupt`, armed for more than one fire.

    The harness above disarms itself inside its first callback, so it reaches
    the recovery pass but never a recovery pass that is interrupted again. One
    creation attempt can enter `publish_ready_handle`, the function the test
    below watches, five times over two `get_mesh` calls: the creator's own
    call, then `finalize_creation`'s pass and retry, once from that `get_mesh`
    and once from the next call's abandoned-attempt reaper, which runs again
    on each later call while the attempt stays unresolved.
    """
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6) if monitoring.get_tool(index) is None)
    fired = 0

    def interrupt(_code, instruction_offset):
        nonlocal fired
        if instruction_offset != offset:
            return
        fired += 1
        if fired >= fires:
            monitoring.set_local_events(tool_id, function.__code__, 0)
        raise interrupted

    monitoring.use_tool_id(tool_id, "dgxm-mesh-creation-repeated-interrupt")
    monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, function.__code__, monitoring.events.INSTRUCTION)
    try:
        return call()
    finally:
        monitoring.set_local_events(tool_id, function.__code__, 0)
        monitoring.register_callback(tool_id, monitoring.events.INSTRUCTION, None)
        monitoring.free_tool_id(tool_id)


@pytest.mark.skipif(not hasattr(sys, "monitoring"), reason="requires Python 3.12")
@pytest.mark.parametrize("fires", [2, 4])
def test_repeated_ready_interruptions_never_admit_a_second_worker_fleet(
    isolated_mesh_runtime, monkeypatch, fires,
):
    """A creator that cannot finish publishing must keep its creation claim.

    `publish_ready_handle` stores the READY marker before `_MESHES[key]`, so an
    interrupt between the two leaves a live fleet only in `_MESH_PENDING`. The
    creator keeps its creation claim in `_MESH_CREATING`, the transport claim and
    the cleanup claim until that fleet is public or no longer live, and admission
    refuses over a live pending entry, so the next call never spawns a second
    fleet on its ports.
    `fires=2` also interrupts the finalize pass; `fires=4` also interrupts its retry.
    """
    first_proc = _ProcMesh()
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(first_proc))
    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", lambda: True)
    interrupted = _Interrupt("READY marker interrupted")

    with pytest.raises(BaseException):  # noqa: B017
        _run_with_repeated_interrupt(
            mesh_creation.publish_ready_handle,
            _instruction_after_ready_boundary("marker"),
            interrupted,
            lambda: mesh.get_mesh(mode="local", gpus_per_host=1),
            fires,
        )

    # The fleet is public, stopped, or still claimed by its creator: never orphaned.
    assert (
        list(mesh._MESHES.values())
        or first_proc.stop_reasons
        or mesh._MESH_CREATING
    )

    rival = _ProcMesh()
    rival_spawn = _spawn_returning(rival)
    spawned: list[object] = []

    def spawn_rival(*args):
        spawned.append(rival)
        return rival_spawn(*args)

    monkeypatch.setattr(mesh, "spawn_worker_fleet", spawn_rival)
    try:
        again = mesh.get_mesh(mode="local", gpus_per_host=1)
    except BaseException:  # refusing is a valid fail-closed leg
        again = None

    # A live unpublished fleet must never be joined by a second one on the
    # same transport, whether the retry recovered it or not.
    assert spawned == []
    assert again is None or again.procs is first_proc
    # By now the fleet is settled: public, stopped or refused.
    assert list(mesh._MESHES.values()) or first_proc.stop_reasons or again is None
    assert mesh._MESH_PENDING == {} or again is None
