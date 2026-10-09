"""Pinned-torchmonarch regressions for exact ProcMesh spawn ownership."""

from __future__ import annotations

from typing import Any

import pytest
from monarch._src.actor.host_mesh import HostMesh
from monarch._src.actor.proc_mesh import ProcMesh

from dgx_monarch.mesh_factory import (
    FleetHandoff,
    WorkerSpawnError,
    _isolated_spawn_host,
    owned_spawned_proc,
    spawn_worker_fleet,
)


def _host() -> HostMesh:
    """Construct the pinned Python wrapper without starting native work."""
    return HostMesh(object(), object(), False, False, None)


def _proc(host: HostMesh | None, stop_calls: list[str]) -> ProcMesh:
    proc = object.__new__(ProcMesh)
    if host is not None:
        proc._host_mesh = host

    def stop(reason: str) -> Any:
        stop_calls.append(reason)
        pytest.fail("a candidate without exact ownership must never be stopped")

    proc.stop = stop
    return proc


def _install_registry(monkeypatch, callbacks: list[Any], active: list[ProcMesh]) -> None:
    from monarch._src.actor import proc_mesh as proc_mesh_api

    monkeypatch.setattr(
        proc_mesh_api, "register_proc_mesh_spawn_callback", callbacks.append)
    monkeypatch.setattr(
        proc_mesh_api, "unregister_proc_mesh_spawn_callback", callbacks.remove)
    monkeypatch.setattr(proc_mesh_api, "get_active_proc_meshes", lambda: list(active))


def test_pinned_host_wrapper_isolates_spawn_registry_identity():
    attached = _host()

    isolated, exact = _isolated_spawn_host(attached)

    assert exact is True
    assert type(isolated) is HostMesh
    assert isolated is not attached
    assert isolated._inner_host_mesh is attached._inner_host_mesh
    assert isolated._proc_meshes == []


def test_returned_proc_is_authoritative_despite_foreign_and_ambiguous_observers(
    monkeypatch,
):
    callbacks: list[Any] = []
    active: list[ProcMesh] = []
    stop_calls: list[str] = []
    workers = object()
    attached = _host()
    foreign_host = _host()
    observed: dict[str, ProcMesh] = {}
    handoff = FleetHandoff()
    _install_registry(monkeypatch, callbacks, active)

    def spawn_procs(spawn_host: HostMesh, **_kwargs: Any) -> ProcMesh:
        returned = _proc(spawn_host, stop_calls)
        collision = _proc(spawn_host, stop_calls)
        foreign = _proc(foreign_host, stop_calls)
        returned.spawn = lambda *_args, **_kwargs: workers
        spawn_host._proc_meshes.extend((returned, collision))
        active.extend((collision, foreign))
        callbacks[0](foreign)
        observed.update(returned=returned, collision=collision, foreign=foreign)
        return returned

    monkeypatch.setattr(HostMesh, "spawn_procs", spawn_procs)
    fleet = spawn_worker_fleet(
        None, True, 1, lambda _config, _handoff: attached, None, handoff)

    assert fleet[1] is observed["returned"]
    assert handoff.fleet is not None
    assert handoff.fleet[0] is fleet[0]
    assert handoff.fleet[1] is fleet[1]
    assert handoff.fleet[2] is fleet[2]
    assert handoff.fleet[3] is fleet[3]
    assert handoff.proc_ownership_unconfirmed is False
    assert fleet[1] is not observed["collision"]
    assert fleet[1] is not observed["foreign"]
    assert stop_calls == []


def test_foreign_proc_is_never_selected_or_stopped_after_spawn_failure(monkeypatch):
    callbacks: list[Any] = []
    active: list[ProcMesh] = []
    stop_calls: list[str] = []
    attached = _host()
    foreign_host = _host()
    foreign = _proc(foreign_host, stop_calls)
    active.append(foreign)
    handoff = FleetHandoff()
    primary = RuntimeError("spawn did not return")
    _install_registry(monkeypatch, callbacks, active)

    def spawn_procs(_spawn_host: HostMesh, **_kwargs: Any) -> ProcMesh:
        callbacks[0](foreign)
        raise primary

    monkeypatch.setattr(HostMesh, "spawn_procs", spawn_procs)
    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(
            None, True, 1, lambda _config, _handoff: attached, None, handoff)

    assert failure.value.original is primary
    assert failure.value.rollback_confirmed is False
    assert failure.value.cleanup is not None
    assert "ownership is unconfirmed" in str(failure.value.cleanup)
    assert handoff.proc_ownership_unconfirmed is True
    assert stop_calls == []


def test_multiple_exact_host_candidates_are_ambiguous_and_none_are_stopped(
    monkeypatch,
):
    callbacks: list[Any] = []
    active: list[ProcMesh] = []
    stop_calls: list[str] = []
    attached = _host()
    handoff = FleetHandoff()
    primary = RuntimeError("spawn interrupted after publication")
    _install_registry(monkeypatch, callbacks, active)

    def spawn_procs(spawn_host: HostMesh, **_kwargs: Any) -> ProcMesh:
        first = _proc(spawn_host, stop_calls)
        second = _proc(spawn_host, stop_calls)
        callbacks[0](first)
        spawn_host._proc_meshes.append(second)
        active.extend((first, second))
        raise primary

    monkeypatch.setattr(HostMesh, "spawn_procs", spawn_procs)
    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(
            None, True, 1, lambda _config, _handoff: attached, None, handoff)

    assert failure.value.original is primary
    assert failure.value.rollback_confirmed is False
    assert failure.value.cleanup is not None
    assert handoff.proc_ownership_unconfirmed is True
    assert stop_calls == []


def test_missing_proc_host_identity_fails_closed(monkeypatch):
    from monarch._src.actor import proc_mesh as proc_mesh_api

    spawn_host = _host()
    candidate = _proc(None, [])
    handoff = FleetHandoff()
    handoff.spawn_host = spawn_host
    handoff.spawn_host_isolated = True
    handoff.proc_spawn_called = True
    handoff.proc_meshes.append(candidate)
    monkeypatch.setattr(proc_mesh_api, "get_active_proc_meshes", lambda: [candidate])

    assert owned_spawned_proc(handoff) is None
    assert handoff.proc_ownership_unconfirmed is True
