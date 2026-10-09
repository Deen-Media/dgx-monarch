"""Attach self-heal with a faked attach, probe and Worker restart; no worker runs."""
import sys
import threading
import types

import pytest


def _stub(monkeypatch, probe_results, attach_results, restart_calls):
    import monarch.actor as monarch_actor

    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch import mesh_attach
    from dgx_monarch.cli import worker_health

    fake_lifecycle = types.SimpleNamespace(
        restart=lambda config, sync=True: restart_calls.append(sync) or True,
        run_on_host=lambda *_args, **_kwargs: None,
    )
    import dgx_monarch.cli as cli_pkg

    monkeypatch.setattr(
        worker_health,
        "passive_worker_health",
        lambda _config, host, **_kwargs: {
            "healthy": probe_results.get(host.address, True)
        },
    )

    def fake_attach_once(addresses):
        result = attach_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(mesh_mod, "_attach_once", fake_attach_once)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_BIND", None)
    monkeypatch.setattr(cli_pkg, "lifecycle", fake_lifecycle, raising=False)
    monkeypatch.setitem(sys.modules, "dgx_monarch.cli.lifecycle", fake_lifecycle)
    monkeypatch.setattr(mesh_attach.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(monarch_actor, "enable_transport", lambda *_args, **_kwargs: None)
    return mesh_mod


def _config(auto_heal=True):
    from dgx_monarch.config import ClusterConfig, HostConfig

    return ClusterConfig(hosts=(HostConfig(name="h1", address="tcp://10.0.0.1:26600"),),
                         client_bind="tcp://10.0.0.1:0", auto_heal=auto_heal,
                         transport_security="trusted_fabric")


def test_healthy_loops_attach_without_restart(monkeypatch):
    calls = []
    mesh_mod = _stub(monkeypatch, {}, ["HOSTS"], calls)
    assert mesh_mod._attach_cluster(_config()) == "HOSTS"
    assert calls == []


def test_dead_loop_restarts_before_first_attach(monkeypatch):
    calls = []
    seen = []
    mesh_mod = _stub(monkeypatch, {}, ["HOSTS"], calls)
    from dgx_monarch.cli import worker_health

    def probe(_config, host, **_kwargs):
        seen.append(host.address)
        return {"healthy": len(seen) > 1}

    monkeypatch.setattr(worker_health, "passive_worker_health", probe)
    assert mesh_mod._attach_cluster(_config()) == "HOSTS"
    assert calls == [False]  # restart(sync=False) runs once, before the attach


def test_attach_failure_heals_fleet_then_raises(monkeypatch):
    from dgx_monarch.mesh import MeshAttachError

    calls = []
    # The first attach and its one in-process retry both fail. The post-failure
    # restart runs after both, so the next ComfyUI session starts clean.
    mesh_mod = _stub(monkeypatch, {}, [RuntimeError("wedge"), RuntimeError("wedge")], calls)
    with pytest.raises(MeshAttachError, match="wedge"):
        mesh_mod._attach_cluster(_config())
    assert calls == [False]


def _two_host_config():
    from dgx_monarch.config import ClusterConfig, HostConfig

    return ClusterConfig(
        hosts=(HostConfig(name="h1", address="tcp://10.0.0.1:26600"),
               HostConfig(name="h2", address="tcp://10.0.0.2:26600")),
        client_bind="tcp://10.0.0.1:0", auto_heal=True,
        transport_security="trusted_fabric")


def test_a_blind_probe_restarts_nothing_and_refuses_the_attach(monkeypatch):
    """Unknown worker health is not restart authority.

    auto_heal defaults on and every setup profile requires it, so this probe
    runs before every attach on a default or profiled cluster;
    mesh_attach.heal_dead_loops says why a blind one must not restart anything.
    A blind pass raises the pre-attach heal error, and no attach runs.
    """
    from dgx_monarch.cli import worker_health
    from dgx_monarch.mesh import MeshAttachError

    calls = []
    mesh_mod = _stub(monkeypatch, {}, ["HOSTS"], calls)
    monkeypatch.setattr(
        worker_health,
        "passive_worker_health",
        lambda _config, _host, **_kwargs: {
            "healthy": None, "error": "TimeoutExpired"},
    )

    with pytest.raises(MeshAttachError, match="auto-heal failed"):
        mesh_mod._attach_cluster(_config())
    assert calls == []


def test_one_blind_host_blocks_healing_a_definitely_dead_one(monkeypatch):
    """A restart is fleet-wide, so one unobserved host withholds all of it."""
    from dgx_monarch.cli import worker_health
    from dgx_monarch.mesh import MeshAttachError

    calls = []
    mesh_mod = _stub(monkeypatch, {}, ["HOSTS"], calls)
    verdicts = {
        "tcp://10.0.0.1:26600": {"healthy": False},
        "tcp://10.0.0.2:26600": {"healthy": None, "error": "TimeoutExpired"},
    }
    monkeypatch.setattr(
        worker_health,
        "passive_worker_health",
        lambda _config, host, **_kwargs: verdicts[host.address],
    )

    with pytest.raises(MeshAttachError, match="auto-heal failed"):
        mesh_mod._attach_cluster(_two_host_config())
    assert calls == []


def test_auto_heal_off_touches_nothing(monkeypatch):
    from dgx_monarch.mesh import MeshAttachError

    calls = []
    mesh_mod = _stub(monkeypatch, {"tcp://10.0.0.1:26600": False},
                     [RuntimeError("down"), RuntimeError("down")], calls)
    with pytest.raises(MeshAttachError, match="down"):
        mesh_mod._attach_cluster(_config(auto_heal=False))
    assert calls == []


def test_transient_attach_failure_retries_in_process_without_restart(monkeypatch):
    """First attach fails, the one in-process retry succeeds: no loop restart,
    no poison latch. torchmonarch 0.6.0 evicts the stale attach session
    (meta-pytorch/monarch#4067); this repository pinned 0.6.0 on 2026-07-29."""
    calls = []
    sentinel = object()
    mesh_mod = _stub(monkeypatch, {}, [RuntimeError("blip"), sentinel], calls)
    hosts = mesh_mod._attach_cluster(_config())
    assert hosts is sentinel
    assert calls == []


def test_deliberate_recycle_race_absorbs_without_blocking(monkeypatch):
    """An in-flight sample dying with 'stopped: dgx-monarch recycle' is absorbed
    before any shutdown attempt, because the recycle already stopped the procs.
    On 2026-07-10 a second stop on the recycled mesh failed, marked
    replacement_blocked and wedged the session until ComfyUI restarted."""
    from types import SimpleNamespace

    from dgx_monarch import mesh as mesh_mod

    calls = []
    handle = SimpleNamespace(
        defunct=False, teardown_complete=True, replacement_blocked=None,
        shutdown=lambda timeout_s=60.0: calls.append("shutdown"),
    )

    class FakeSupervision(Exception):
        pass

    FakeSupervision.__name__ = "SupervisionError"
    exc = FakeSupervision(
        "Endpoint call dgxm_worker.sample() failed, Supervision event: "
        "stopped: dgx-monarch recycle")
    mesh_mod.mark_defunct_on_supervision_failure(handle, exc)
    assert handle.defunct is True
    assert handle.replacement_blocked is None
    assert calls == []


def test_shutdown_is_noop_after_completed_teardown():
    from types import SimpleNamespace

    from dgx_monarch.mesh import MeshHandle

    handle = MeshHandle.__new__(MeshHandle)
    handle.teardown_complete = True
    handle.replacement_blocked = None
    handle.procs = SimpleNamespace(stop=lambda reason: (_ for _ in ()).throw(
        AssertionError("must not stop procs twice")))
    handle.shutdown()


def test_concurrent_supervision_failures_share_one_reconcile_stop():
    from types import SimpleNamespace

    from dgx_monarch.mesh import mark_defunct_on_supervision_failure

    entered = threading.Event()
    release = threading.Event()
    calls = []
    handle = SimpleNamespace(
        lock=threading.RLock(),
        defunct=False,
        teardown_complete=False,
        replacement_blocked=None,
        sample_leases={},
        _supervision_reconcile_token=None,
        _supervision_reconcile_started=False,
    )

    def shutdown(*, timeout_s, _reconcile_token):
        calls.append(_reconcile_token)
        assert _reconcile_token is handle._supervision_reconcile_token
        entered.set()
        assert release.wait(timeout=2)
        handle.teardown_complete = True

    handle.shutdown = shutdown

    class SupervisionError(RuntimeError):
        pass

    error = SupervisionError("peer closed")
    first = threading.Thread(
        target=mark_defunct_on_supervision_failure, args=(handle, error))
    first.start()
    assert entered.wait(timeout=1)

    mark_defunct_on_supervision_failure(handle, error)
    release.set()
    first.join(timeout=2)

    assert not first.is_alive()
    assert len(calls) == 1
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
