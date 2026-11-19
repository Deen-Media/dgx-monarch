"""A per-rank bootstrap refusal leaves nothing spawned, so the rollback is confirmed.

torchmonarch 0.8.0 runs the per-rank bootstrap callable once per rank on the
calling thread before spawning begins, and its errors raise from
``HostMesh.spawn_procs`` itself (upstream ``host_mesh.py``, ``spawn_procs``
docstring; tests/spawn_contract_probe.py pins it on the real runtime).
0.7.0 called it during the native spawn, returned a ProcMesh, and the factory
stopped that ProcMesh and reported a confirmed rollback. These doubles follow
the 0.8.0 order.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import dgx_monarch
from dgx_monarch import monarch_launch
from dgx_monarch.mesh_factory import FleetHandoff, WorkerSpawnError, spawn_worker_fleet

_CONFIG = SimpleNamespace(monarch=SimpleNamespace(startup_timeout_s=180))


@pytest.fixture(autouse=True)
def _doubles(monkeypatch):
    actor = types.ModuleType("dgx_monarch.actor")
    actor.GPUWorker = object
    monkeypatch.setitem(sys.modules, "dgx_monarch.actor", actor)
    monkeypatch.setattr(dgx_monarch, "actor", actor, raising=False)
    monkeypatch.setattr(monarch_launch, "wait_initialized", lambda *_args: None)
    monkeypatch.setattr(monarch_launch, "wait_worker_ready", lambda *_args: None)


class _EagerHosts:
    """spawn_procs in 0.8.0's order: every rank's command first, then the native spawn."""

    def __init__(self, ranks: int):
        self.ranks = ranks
        self.native_spawns = 0

    def spawn_procs(self, *, per_host, bootstrap, bootstrap_command, name):
        del per_host, bootstrap, name
        commands = [bootstrap_command({"hosts": rank, "gpus": 0}) for rank in range(self.ranks)]
        self.native_spawns += 1
        raise AssertionError(f"the refusal must stop the spawn before it starts: {commands}")

    def shutdown(self, *_args, **_kwargs):
        pytest.fail("attached worker loops are never shut down")


def test_bootstrap_refusal_reports_a_confirmed_rollback_without_stopping_anything(monkeypatch):
    refusal = ValueError("rank 1 has no launch command")

    def command_for(_config, _use_cluster):
        def command(point):
            if point["hosts"] == 1:
                raise refusal
            return object()
        return command

    monkeypatch.setattr(monarch_launch, "actor_bootstrap_command", command_for)
    hosts = _EagerHosts(ranks=2)
    handoff = FleetHandoff()
    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(_CONFIG, True, 1, lambda _config, _handoff: hosts, None, handoff)
    assert failure.value.original is refusal
    assert failure.value.cleanup is None
    assert failure.value.rollback_confirmed is True
    assert hosts.native_spawns == 0
    assert handoff.proc_spawn_called is False
    assert handoff.proc_ownership_unconfirmed is False
    assert handoff.cleanup_started is False
    assert handoff.callback_cleanup_unconfirmed is False


def test_a_spawn_error_that_is_not_the_refusal_keeps_ownership_unconfirmed(monkeypatch):
    """Only the callable's own exception proves nothing spawned."""
    native = RuntimeError("native spawn failed after the commands were built")

    class FailingHosts(_EagerHosts):
        def spawn_procs(self, *, per_host, bootstrap, bootstrap_command, name):
            del per_host, bootstrap, name
            for rank in range(self.ranks):
                bootstrap_command({"hosts": rank, "gpus": 0})
            self.native_spawns += 1
            raise native

    monkeypatch.setattr(monarch_launch, "actor_bootstrap_command",
                        lambda _config, _use_cluster: lambda _point: object())
    hosts = FailingHosts(ranks=2)
    handoff = FleetHandoff()
    with pytest.raises(WorkerSpawnError) as failure:
        spawn_worker_fleet(_CONFIG, True, 1, lambda _config, _handoff: hosts, None, handoff)
    assert failure.value.original is native
    assert failure.value.rollback_confirmed is False
    assert handoff.proc_spawn_called is True
    assert handoff.proc_ownership_unconfirmed is True
