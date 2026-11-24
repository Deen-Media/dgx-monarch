"""A failed cluster bring-up's poison lifts once its spawned processes are proven gone.

A Worker lost while the driver brings a fleet up leaves the bring-up's rollback
stop unconfirmed, and the session refuses every render. The next render and
Reset attached mesh ask every host the liveness question; only a fleet-wide
gone lifts the poison. A bring-up whose rollback Monarch confirmed, or that
started nothing, records no poison and asks no host. No host is reached: the
host runner is a recording fake.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from dgx_monarch import failed_bring_up, mesh_factory, mesh_liveness

ADDRESS_A = "box-a"
ADDRESS_B = "box-b"
POISON = ("mesh creation failed (MeshAttachError('worker actor bring-up failed (...); spawned "
          "processes were not safely stopped')); owned ProcMesh cleanup unconfirmed (worker "
          "spawn rollback outcome is unconfirmed)")


def _line(**overrides) -> str:
    payload = {"schema": 1, "actors": 0, "zombies": 0, "unreadable": 0,
               "ledger_rows_alive": 0, "loop_alive": True}
    payload.update(overrides)
    return json.dumps(payload)


class Runner:
    """Answers each host's liveness script; records which hosts were asked."""

    def __init__(self, **answers):
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, config, host, script, timeout=None):
        self.asked.append(host.address)
        return SimpleNamespace(returncode=0, stdout=self.answers.get(host.address, _line()),
                               stderr="")


def _config(*addresses):
    return SimpleNamespace(
        hosts=tuple(SimpleNamespace(address=address, name=address) for address in addresses),
        python_bin="~/venv/bin/python", worker_args={}, source="")


def _failed_spawn(config=None, **state):
    """The handoff a bring-up leaves when its rollback stop could not confirm."""
    handoff = mesh_factory.FleetHandoff()
    handoff.config = _config(ADDRESS_A, ADDRESS_B) if config is None else config
    handoff.failure_latched = handoff.spawn_in_progress = True
    handoff.proc_spawn_called = handoff.cleanup_started = True
    for name, value in state.items():
        setattr(handoff, name, value)
    claim = mesh_factory.begin_creation_cleanup(object(), handoff)
    return claim, handoff


def _runtime(poison=POISON, creating=None):
    return {"_TRANSPORT_POISON": poison, "_MESH_LOCK": threading.Lock(),
            "_MESH_CREATING": creating or {}}


@pytest.fixture(autouse=True)
def _fresh_claims_and_proof(monkeypatch):
    monkeypatch.setattr(mesh_factory, "_CREATION_CLEANUP_CLAIMS", {})
    monkeypatch.setattr(failed_bring_up, "_LAST_PROOF", [])


def test_a_bring_up_whose_processes_are_gone_on_every_host_lifts_the_poison():
    _failed_spawn()
    runtime, runner = _runtime(), Runner()

    assert failed_bring_up.release(runtime, runner=runner) is True

    assert runtime["_TRANSPORT_POISON"] is None
    assert sorted(runner.asked) == [ADDRESS_A, ADDRESS_B]
    assert mesh_factory.creation_cleanup_pending() is False
    from dgx_monarch import mesh_teardown

    # Monarch never confirmed the rollback's stop: the next attach takes a released fleet's rules.
    assert mesh_teardown.released_predecessor_recently(addresses=[ADDRESS_A, ADDRESS_B]) is True


def test_a_later_failed_bring_up_is_proven_afresh_never_by_the_earlier_proof():
    """Its processes are newer than any earlier answer, however recent that answer is."""
    _failed_spawn()
    runner = Runner()
    assert failed_bring_up.release(_runtime(), runner=runner) is True
    _failed_spawn()  # the next bring-up fails too, seconds later
    runtime = _runtime()

    assert failed_bring_up.release(runtime, runner=runner) is True
    assert sorted(runner.asked) == sorted([ADDRESS_A, ADDRESS_B] * 2)


def test_claims_on_different_hosts_are_not_settled_by_one_probe():
    _failed_spawn()
    _failed_spawn(config=_config(ADDRESS_A))
    runtime, runner = _runtime(), Runner()

    assert failed_bring_up.release(runtime, runner=runner) is False
    assert runner.asked == [] and runtime["_TRANSPORT_POISON"] == POISON


@pytest.mark.parametrize("answer", [_line(actors=1), _line(unreadable=1)], ids=["alive", "unknown"])
def test_a_host_that_still_shows_them_keeps_the_poison_and_the_refusal_says_so(answer):
    _failed_spawn()
    runtime = _runtime()

    assert failed_bring_up.release(runtime, runner=Runner(**{ADDRESS_B: answer})) is False

    assert runtime["_TRANSPORT_POISON"] == POISON
    assert mesh_factory.creation_cleanup_pending() is True
    text = failed_bring_up.refusal(POISON)
    assert "cannot yet prove gone" in text and f"{ADDRESS_B}=" in text
    assert "Reset attached mesh" in text and "restart ComfyUI" in text


@pytest.mark.parametrize("state", [
    {"attach_in_progress": True},           # the attach's result handoff was interrupted
    {"callback_cleanup_unconfirmed": True},  # Monarch's spawn callback may still be registered
    {"config": SimpleNamespace(hosts=())},   # a local fleet
], ids=["unsafe_attach", "callback", "local"])
def test_states_no_proof_settles_keep_the_poison_without_asking_any_host(state):
    _failed_spawn(**state)
    runtime, runner = _runtime(), Runner()

    assert failed_bring_up.release(runtime, runner=runner) is False

    assert runner.asked == [] and runtime["_TRANSPORT_POISON"] == POISON
    assert "not reusable" in failed_bring_up.refusal(POISON)


def test_an_attach_poison_is_not_a_bring_up_poison():
    _failed_spawn()
    poison = "cluster attach failed: TimeoutError('config push')"
    runtime, runner = _runtime(poison), Runner()

    assert failed_bring_up.release(runtime, runner=runner) is False
    assert runner.asked == [] and runtime["_TRANSPORT_POISON"] == poison
    assert "not reusable" in failed_bring_up.refusal(poison)


def test_no_release_while_another_creation_runs():
    _failed_spawn()
    runtime, runner = _runtime(creating={("key",): object()}), Runner()

    assert failed_bring_up.release(runtime, runner=runner) is False
    assert runner.asked == [] and runtime["_TRANSPORT_POISON"] == POISON


def _gone_probe(monkeypatch, answer=mesh_liveness.GONE):
    import time

    monkeypatch.setattr(mesh_liveness, "probe_fleet", lambda _handle, **_kwargs: (
        mesh_liveness.LivenessEvidence(
            answer, ((ADDRESS_A, answer), (ADDRESS_B, answer)), time.monotonic())))


def test_the_reset_lifts_a_proven_poison_and_says_so(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.nodes import recycle_drain

    monkeypatch.setattr(mesh_mod, "_MESHES", {})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", POISON)
    _failed_spawn()
    _gone_probe(monkeypatch)

    payload, status = recycle_drain.recycle_attached_mesh()

    assert (status, payload["ok"], payload["status"]) == (200, True, "poison_released")
    assert mesh_mod._TRANSPORT_POISON is None


def test_the_reset_over_unproven_processes_answers_with_the_cause_not_no_mesh(monkeypatch):
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.nodes import recycle_drain

    monkeypatch.setattr(mesh_mod, "_MESHES", {})
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {})
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", POISON)
    _failed_spawn()
    _gone_probe(monkeypatch, mesh_liveness.ALIVE)

    payload, status = recycle_drain.recycle_attached_mesh()

    assert (status, payload["ok"], payload["status"]) == (200, False, "poisoned")
    assert "cannot yet prove gone" in payload["detail"]
    assert mesh_mod._TRANSPORT_POISON == POISON


class _AttachedHosts:
    """Two attached Worker loops; spawn_procs builds every rank's command first, as Monarch does."""

    def __init__(self, procs=None):
        self.procs, self._proc_meshes = procs, []

    def spawn_procs(self, *, per_host, bootstrap, bootstrap_command, name):
        del per_host, bootstrap, name
        [bootstrap_command({"hosts": rank, "gpus": 0}) for rank in range(2)]
        self.procs._host_mesh = self
        self._proc_meshes.append(self.procs)
        return self.procs

    def shutdown(self, *_args, **_kwargs):
        pytest.fail("attached worker loops are never shut down")


def _cluster_session(monkeypatch, hosts, refuse_rank=None):
    """get_mesh in cluster mode on two hosts. Its first bring-up runs the real factory on ``hosts``.

    The second hands back a fresh fleet. Returns the mesh module, that fresh
    ProcMesh, the spawn calls and the liveness probes asked.
    """
    import sys
    import types

    import dgx_monarch
    from dgx_monarch import mesh_teardown, monarch_launch, proc_stops
    from test_mesh_publication_ownership import _mesh_publication_fakes, _PublishedProcMesh

    fresh, calls, probes = _PublishedProcMesh(), [], []

    def spawn(config, use_cluster, gph, _attach, bootstrap, handoff):
        calls.append(handoff)
        if len(calls) == 1:
            return mesh_factory.spawn_worker_fleet(
                config, use_cluster, gph, lambda _config, _handoff: hosts, bootstrap, handoff)
        handoff.fleet = SimpleNamespace(), fresh, SimpleNamespace(), True
        return handoff.fleet

    def command_for(_config, _use_cluster):
        def command(point):
            if point["hosts"] == refuse_rank:
                raise ValueError(f"rank {refuse_rank} has no launch command")
            return object()
        return command

    mesh = _mesh_publication_fakes(monkeypatch, spawn, lambda: True)
    config = SimpleNamespace(
        hosts=tuple(SimpleNamespace(address=address, name=address, gpus=1)
                    for address in (ADDRESS_A, ADDRESS_B)),
        source="/cluster.toml", comfy_dir="", worker_args={}, client_bind=None,
        monarch=SimpleNamespace(startup_timeout_s=180))
    monkeypatch.setattr(mesh, "find_config_path", lambda _path: "/cluster.toml")
    monkeypatch.setattr(mesh, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh, "config_fingerprint", lambda _path: "digest")
    monkeypatch.setattr(mesh.os.path, "getmtime", lambda _path: 0.0)
    actor = types.ModuleType("dgx_monarch.actor")
    actor.GPUWorker = object
    monkeypatch.setitem(sys.modules, "dgx_monarch.actor", actor)
    monkeypatch.setattr(dgx_monarch, "actor", actor, raising=False)
    monkeypatch.setattr(monarch_launch, "actor_bootstrap_command", command_for)
    monkeypatch.setattr(monarch_launch, "wait_initialized", lambda *_args: None)
    monkeypatch.setattr(monarch_launch, "wait_worker_ready", lambda *_args: None)
    monkeypatch.setattr(proc_stops, "_STOPS", [])
    monkeypatch.setattr(mesh_teardown, "_RELEASED_ADDRESSES", None)
    _gone_probe(monkeypatch)
    probe = mesh_liveness.probe_fleet
    monkeypatch.setattr(mesh_liveness, "probe_fleet",
                        lambda handle, **kwargs: probes.append(handle) or probe(handle, **kwargs))
    return mesh, fresh, calls, probes


class _InitFails:
    """The fleet's ProcMesh: the Worker's constructor raises, and its stop answers as given."""

    def __init__(self, stop_error=None):
        self.stop_error, self.stop_reasons = stop_error, []

    def spawn(self, *_args, **_kwargs):
        raise RuntimeError("worker init raised")

    def stop(self, reason):
        self.stop_reasons.append(reason)
        return SimpleNamespace(get=self._get)

    def _get(self, timeout=None):
        if self.stop_error is not None:
            raise self.stop_error


@pytest.mark.parametrize("refuse_rank", [None, 1], ids=["rollback_confirmed", "nothing_started"])
def test_a_cluster_bring_up_that_left_nothing_running_needs_no_restart(monkeypatch, refuse_rank):
    """Monarch confirmed the stop of every process it spawned, or none started.

    Nothing is unknown, so nothing is poisoned and no host is asked: the next
    render attaches a fresh fleet in the same session.
    """
    procs = _InitFails()
    mesh, fresh, calls, probes = _cluster_session(monkeypatch, _AttachedHosts(procs), refuse_rank)

    with pytest.raises(mesh.MeshAttachError, match="spawned processes were rolled back"):
        mesh.get_mesh(mode="cluster")
    assert procs.stop_reasons == ([] if refuse_rank else ["dgx-monarch partial bring-up rollback"])
    assert mesh._TRANSPORT_POISON is None
    assert mesh_factory.creation_cleanup_pending() is False

    assert mesh.get_mesh(mode="cluster").procs is fresh
    assert len(calls) == 2 and probes == []


def test_a_cluster_rollback_stop_that_did_not_confirm_is_proven_by_asking_every_host(monkeypatch):
    """A stop Monarch never confirmed leaves the processes unknown: the next render asks first."""
    procs = _InitFails(TimeoutError("proc stop did not finish"))
    mesh, fresh, calls, probes = _cluster_session(monkeypatch, _AttachedHosts(procs))

    with pytest.raises(mesh.MeshAttachError, match="not safely stopped"):
        mesh.get_mesh(mode="cluster")
    assert failed_bring_up.liftable(mesh._TRANSPORT_POISON)

    assert mesh.get_mesh(mode="cluster").procs is fresh
    assert len(calls) == 2 and len(probes) == 1


def test_a_confirmed_rollback_whose_spawn_callback_stays_registered_keeps_the_restart(monkeypatch):
    """The rollback stopped every process, but Monarch may still call this bring-up's callback."""
    from monarch._src.actor import proc_mesh as monarch_proc_mesh

    def unregister(_callback):
        raise RuntimeError("callback registry busy")

    procs = _InitFails()
    mesh, _fresh, calls, probes = _cluster_session(monkeypatch, _AttachedHosts(procs))
    monkeypatch.setattr(monarch_proc_mesh, "register_proc_mesh_spawn_callback", lambda _callback: None)
    monkeypatch.setattr(monarch_proc_mesh, "unregister_proc_mesh_spawn_callback", unregister)

    with pytest.raises(mesh.MeshAttachError, match="spawned processes were rolled back"):
        mesh.get_mesh(mode="cluster")
    with pytest.raises(mesh.MeshAttachError, match="restart ComfyUI"):
        mesh.get_mesh(mode="cluster")
    assert procs.stop_reasons == ["dgx-monarch partial bring-up rollback"]
    assert len(calls) == 1 and probes == []


def test_the_mesh_row_names_the_remedy_that_lifts_a_provable_poison():
    from dgx_monarch import mesh_health

    _failed_spawn()
    health = mesh_health.health_of([], poison=POISON)
    assert health.state == "poisoned"
    assert "cannot yet prove gone" in health.reason
    assert health.remedy.startswith("start the next render or press Reset attached mesh")
    unprovable = mesh_health.health_of([], poison="cluster attach failed: x")
    assert unprovable.remedy.startswith("restart ComfyUI")
