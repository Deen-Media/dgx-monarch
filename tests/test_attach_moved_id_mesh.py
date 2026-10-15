"""The re-attach after a Worker restart, driven from ``ensure_live`` down.

docs/TROUBLESHOOTING.md #2. A Worker loop restarts under an attached driver.
The render that meets the old actors records a supervision death, the stop of
the old fleet times out, the liveness proof releases the latch, and the next
``ensure_live`` builds a replacement through ``get_mesh``. That replacement
attach must read the loop's new id, send the restarted loop its throwaway
message first, and dial the new id, all in the same process.

Monarch, ssh and the fleet spawn are stubbed; the attach, the release and the
mesh bookkeeping between them are the real code.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    attach_trace,
    mesh_factory,
    mesh_helpers,
    mesh_liveness,
    mesh_safety,
    mesh_teardown,
    worker_proc_id,
)
from dgx_monarch import mesh as mesh_mod
from dgx_monarch.cli import listener_generation

ADDRESS = "tcp://10.0.0.2:26600"
OLD_ID, NEW_ID = "aaaaaaaaaaaa0001", "bbbbbbbbbbbb0002"
SOURCE = "/fake/cluster.toml"


class _StopTimesOut:
    """The old fleet's procs: their stop never acknowledges."""

    def stop(self, _reason):
        return SimpleNamespace(get=lambda timeout: (_ for _ in ()).throw(
            TimeoutError("detach acknowledgement timed out")))


@pytest.fixture
def driver(isolated_environ, monkeypatch):
    """A driver process with one configured Worker loop and no Monarch under it."""
    import monarch.actor as monarch_actor

    from dgx_monarch import monarch_launch
    from dgx_monarch.config import ClusterConfig, HostConfig

    for name in ("_TRANSPORT_MODE", "_TRANSPORT_POISON", "_TRANSPORT_BIND",
                 "_TRANSPORT_COMMITTED_GENERATION"):
        monkeypatch.setattr(mesh_mod, name, None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_CLAIM_GENERATION", 0)
    for name in ("_TRANSPORT_CLAIMS", "_MESHES", "_MESH_CREATING"):
        monkeypatch.setattr(mesh_mod, name, type(getattr(mesh_mod, name))())
    monkeypatch.setattr(mesh_factory, "_CREATION_CLEANUP_CLAIMS",
                        type(mesh_factory._CREATION_CLEANUP_CLAIMS)())
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    monkeypatch.setattr(mesh_teardown, "_RELEASED_PREDECESSOR", None)
    monkeypatch.setattr(mesh_teardown, "_RELEASED_ADDRESSES", None)
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})

    config = ClusterConfig(
        hosts=(HostConfig(name="h1", address=ADDRESS),), client_bind="tcp://10.0.0.1:0",
        auto_heal=False, transport_security="trusted_fabric", source=SOURCE)
    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _value: "/fake/comfy")
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda path: path or None)
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh_helpers, "start_client_lease", lambda: True)
    monkeypatch.setattr(monarch_actor, "enable_transport", lambda *_a, **_k: None)

    state = SimpleNamespace(proc_id=OLD_ID, dials=[], primed=[], config=config)

    def spawn(config, _use_cluster, _gph, attach, _bootstrap, handoff):
        return attach(config, handoff), _StopTimesOut(), object(), True

    def fleet(config, **_kwargs):
        return {host.address: {"gen": state.proc_id[:12], "proc_id": state.proc_id}
                for host in config.hosts}

    def attach_once(dial):
        state.dials.append(list(dial))
        return SimpleNamespace(name=f"hosts_{len(state.dials)}")

    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn)
    monkeypatch.setattr(listener_generation, "fleet", fleet)
    monkeypatch.setattr(mesh_mod, "_attach_once", attach_once)
    monkeypatch.setattr(worker_proc_id, "prime",
                        lambda addresses: state.primed.append(list(addresses)) or [])
    monkeypatch.setattr(
        mesh_liveness, "probe_fleet",
        lambda handle, **_kwargs: mesh_liveness.LivenessEvidence(
            mesh_liveness.GONE, ((ADDRESS, mesh_liveness.GONE),), time.monotonic()))
    return state


def test_ensure_live_after_a_worker_restart_dials_the_new_id_in_the_same_process(
        driver, monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(mesh_mod.log, "info",
                        lambda template, *args, **_k: lines.append(template % args if args else template))
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    assert driver.dials == [[worker_proc_id.proc_addr(ADDRESS, OLD_ID)]]
    assert driver.primed == []

    # The loop restarts with a new id. The render that meets the old actors
    # records their death, and the stop of the old fleet times out.
    driver.proc_id = NEW_ID
    handle._supervision_death_report = "ProcessExited: rank 0 is gone"
    with pytest.raises(RuntimeError, match="ProcMesh stop timed out"):
        handle._shutdown_impl()

    replacement = mesh_mod.ensure_live(handle)

    assert replacement is not handle
    assert handle.teardown_complete is True
    assert mesh_mod._TRANSPORT_POISON is None
    # The replacement read the new id, primed the restarted loop, and dialed it.
    assert driver.dials[1:] == [[worker_proc_id.proc_addr(ADDRESS, NEW_ID)]]
    assert driver.primed == [[ADDRESS]]
    opened = [fields for fields in map(attach_trace.parse, lines)
              if fields and fields.get("event") == "open"]
    assert [fields["released_predecessor"] for fields in opened] == ["false", "true"]
    closed = [fields for fields in map(attach_trace.parse, lines)
              if fields and fields.get("event") == "close"]
    assert [fields["outcome"] for fields in closed] == ["attached", "attached"]
