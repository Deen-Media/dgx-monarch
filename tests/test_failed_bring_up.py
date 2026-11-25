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

from dgx_monarch import failed_bring_up, mesh_factory, mesh_liveness, worker_proc_id

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
    """The handoff a bring-up leaves when its rollback stop could not confirm.

    By default every rank's proc had started (a Worker lost while the Workers
    came up, the hardware case), so no spawn can still be queued.
    """
    handoff = mesh_factory.FleetHandoff()
    handoff.config = _config(ADDRESS_A, ADDRESS_B) if config is None else config
    handoff.failure_latched = handoff.spawn_in_progress = True
    handoff.proc_spawn_called = handoff.cleanup_started = handoff.spawn_settled = True
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
    monkeypatch.setattr(failed_bring_up, "_LAST_PROOF", None)


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


def test_a_proof_another_check_stores_or_clears_mid_decision_neither_raises_nor_lifts(monkeypatch):
    """A render and a Reset at once: one read of the stored proof, so its keys and evidence agree.

    Between this check's read and its use, another check stores a gone taken
    for other claims and then clears the store. This check must probe for its
    own claims, keep the poison on what its probe says, and raise nothing.
    """
    import time

    _failed_spawn()
    keys = frozenset(claim for claim, _handoff in failed_bring_up._claims_to_prove())
    now = time.monotonic()
    alive = mesh_liveness.LivenessEvidence(mesh_liveness.ALIVE, ((ADDRESS_B, "alive"),), now)
    gone = mesh_liveness.LivenessEvidence(mesh_liveness.GONE, ((ADDRESS_A, "gone"), (ADDRESS_B, "gone")), now)
    monkeypatch.setattr(failed_bring_up, "_LAST_PROOF", (keys, alive, ()))

    def another_check_lands(_evidence):
        failed_bring_up._store((frozenset({object()}), gone, ()))
        failed_bring_up._store(None)
        return None  # this check's own stored answer is too old to reuse

    monkeypatch.setattr(mesh_liveness, "_still_answers", another_check_lands)
    runtime, runner = _runtime(), Runner(**{ADDRESS_B: _line(actors=1)})

    assert failed_bring_up.release(runtime, runner=runner) is False
    assert runtime["_TRANSPORT_POISON"] == POISON and sorted(runner.asked) == [ADDRESS_A, ADDRESS_B]
    assert "cannot yet prove gone" in failed_bring_up.refusal(POISON)


def test_a_check_whose_probe_outlives_another_checks_lift_returns_quietly(monkeypatch):
    """The slower check finds the poison already lifted: no lift of its own, no exception."""
    import time

    from dgx_monarch.cli import lifecycle

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    _failed_spawn()
    entered, let_go, results = threading.Event(), threading.Event(), []
    real_probe = mesh_liveness.probe_fleet

    def probe(handle, **_kwargs):
        if not entered.is_set():
            entered.set()
            assert let_go.wait(10)
            return mesh_liveness.LivenessEvidence(mesh_liveness.ALIVE, ((ADDRESS_B, "alive"),), time.monotonic())
        return real_probe(handle, runner=Runner())

    monkeypatch.setattr(mesh_liveness, "probe_fleet", probe)
    runtime = _runtime()
    slow = threading.Thread(target=lambda: results.append(failed_bring_up.release(runtime)))
    slow.start()
    assert entered.wait(10)

    assert failed_bring_up.release(runtime) is True
    let_go.set()
    slow.join(10)
    assert results == [False] and runtime["_TRANSPORT_POISON"] is None
    assert failed_bring_up.held_hosts() == ()


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


def _cluster_session(monkeypatch, hosts, refuse_rank=None, *, dialed=None, scans=None,
                     init_error=None):
    """get_mesh in cluster mode on two hosts. Its first bring-up runs the real factory on ``hosts``.

    The second hands back a fresh fleet. Returns the mesh module, that fresh
    ProcMesh, the spawn calls and the liveness probes asked. ``dialed`` is what
    the attach records of each loop it dialed (mesh_attach.read_dial does it in
    production); ``scans`` answers each host with the real scan (_HostScans) in
    place of a fleet-wide gone; ``init_error`` is what waiting for every rank's
    proc raises.
    """
    import sys
    import types

    import dgx_monarch
    from dgx_monarch import mesh_teardown, monarch_launch, proc_stops
    from dgx_monarch.cli import lifecycle
    from test_mesh_publication_ownership import _mesh_publication_fakes, _PublishedProcMesh

    fresh, calls, probes = _PublishedProcMesh(), [], []

    def attach(_config, handoff):
        handoff.loop_gens = dict(dialed or {})
        return hosts

    def spawn(config, use_cluster, gph, _attach, bootstrap, handoff):
        calls.append(handoff)
        if len(calls) == 1:
            return mesh_factory.spawn_worker_fleet(config, use_cluster, gph, attach, bootstrap, handoff)
        handoff.fleet = SimpleNamespace(), fresh, SimpleNamespace(), True
        return handoff.fleet

    def wait_initialized(*_args):
        if init_error is not None:
            raise init_error

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
        python_bin="~/venv/bin/python", monarch=SimpleNamespace(startup_timeout_s=180))
    monkeypatch.setattr(mesh, "find_config_path", lambda _path: "/cluster.toml")
    monkeypatch.setattr(mesh, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh, "config_fingerprint", lambda _path: "digest")
    monkeypatch.setattr(mesh.os.path, "getmtime", lambda _path: 0.0)
    actor = types.ModuleType("dgx_monarch.actor")
    actor.GPUWorker = object
    monkeypatch.setitem(sys.modules, "dgx_monarch.actor", actor)
    monkeypatch.setattr(dgx_monarch, "actor", actor, raising=False)
    monkeypatch.setattr(monarch_launch, "actor_bootstrap_command", command_for)
    monkeypatch.setattr(monarch_launch, "wait_initialized", wait_initialized)
    monkeypatch.setattr(monarch_launch, "wait_worker_ready", lambda *_args: None)
    monkeypatch.setattr(proc_stops, "_STOPS", [])
    monkeypatch.setattr(mesh_teardown, "_RELEASED_ADDRESSES", None)
    if scans is not None:
        monkeypatch.setattr(lifecycle, "run_on_host", scans)
        monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
        return mesh, fresh, calls, scans.asked
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


class _HostScans:
    """Each host's real liveness scan over a synthetic /proc, run when the probe asks.

    Stands in for cli.lifecycle.run_on_host: the line is what actor_reaper.liveness
    prints for that host's processes at that moment, so a test can restart a
    host's loop between two checks. ``loops`` holds (pid, starttime, state) per
    host; ``zombie_leaders`` puts a leader that exited, with its Threads count.
    """

    def __init__(self, tmp_path):
        self.tmp_path, self.asked = tmp_path, []
        self.loops: dict[str, list[tuple[int, int, str]]] = {}
        self.zombie_leaders: dict[str, str] = {}

    def __call__(self, _config, host, _script, timeout=None):
        from dgx_monarch.cli import actor_reaper
        from test_actor_reaper import (
            LOOP_ARGV,
            SELF_PID,
            SPAWNER_ENV,
            UID,
            _zombie_leader,
            make_proc,
            make_proc_root,
            write_ledger,
        )

        self.asked.append(host.address)
        base, boot = self.tmp_path / f"{host.address}-{len(self.asked)}", f"boot-{host.address}"
        root = make_proc_root(base, boot=boot)
        for pid, starttime, state in self.loops.get(host.address, ()):
            make_proc(root, pid, starttime=starttime, state=state, argv=LOOP_ARGV, env=SPAWNER_ENV)
        rows = []
        if host.address in self.zombie_leaders:
            _zombie_leader(root, 5100, threads=self.zombie_leaders[host.address], argv=(), env=())
            rows = [{"pid": 5100, "starttime": 1000}]
        line = actor_reaper.liveness(proc_root=str(root), uid=UID, self_pid=SELF_PID,
                                     ledger_file=write_ledger(base, rows, boot=boot))
        return SimpleNamespace(returncode=0, stdout=json.dumps(line), stderr="")


def _generation(address: str, pid: int, starttime: int) -> str:
    """What the attach's probe reads for the loop with this pid and start time on that host."""
    return worker_proc_id.derive(f"boot-{address}", pid, starttime)[:12]


DIALED = {ADDRESS_A: _generation(ADDRESS_A, 6001, 4000), ADDRESS_B: _generation(ADDRESS_B, 6002, 4000)}


def _queued_spawn_session(monkeypatch, tmp_path, *, settled=False):
    """A bring-up whose wait for every rank's proc failed (unless ``settled``) and whose stop timed out."""
    scans = _HostScans(tmp_path)
    procs = _InitFails(TimeoutError("proc stop did not finish"))
    session = _cluster_session(monkeypatch, _AttachedHosts(procs), dialed=DIALED, scans=scans,
                               init_error=None if settled else TimeoutError("procs not ready"))
    mesh, _fresh, calls, _asked = session
    with pytest.raises(mesh.MeshAttachError, match="not safely stopped"):
        mesh.get_mesh(mode="cluster")
    assert calls[0].spawn_settled is settled and failed_bring_up.liftable(mesh._TRANSPORT_POISON)
    return scans, session


def test_a_queued_spawn_whose_loops_were_replaced_lifts(monkeypatch, tmp_path):
    """The spawn died with the loops that held it and no actor runs: the next render builds afresh."""
    scans, (mesh, fresh, calls, asked) = _queued_spawn_session(monkeypatch, tmp_path)
    scans.loops = {ADDRESS_A: [(7001, 5000, "S")]}  # box-a's loop restarted, box-b's is down

    assert mesh.get_mesh(mode="cluster").procs is fresh
    assert len(calls) == 2 and sorted(asked) == [ADDRESS_A, ADDRESS_B]


def test_a_queued_spawn_whose_loop_still_runs_holds_and_names_the_host(monkeypatch, tmp_path):
    """A frozen loop may still hold the spawn: no actor yet, and it lands when the loop resumes."""
    from dgx_monarch import mesh_health
    from dgx_monarch.nodes import recycle_drain

    scans, (mesh, fresh, calls, _asked) = _queued_spawn_session(monkeypatch, tmp_path)
    scans.loops = {ADDRESS_A: [(6001, 4000, "T")], ADDRESS_B: [(7002, 5000, "S")]}

    with pytest.raises(mesh.MeshAttachError, match="the Worker loop it attached to still runs on box-a,") as held:
        mesh.get_mesh(mode="cluster")
    assert "restart the Worker service on box-a (dgxm restart)" in str(held.value)
    assert failed_bring_up.held_hosts() == ((ADDRESS_A, "running"),)
    row = mesh_health.health_of([], poison=mesh._TRANSPORT_POISON)
    assert "still runs on box-a" in row.reason and row.remedy.startswith("restart the Worker service on box-a")
    payload, _status = recycle_drain.recycle_attached_mesh()
    assert payload["status"] == "poisoned" and "still runs on box-a" in payload["detail"]
    assert len(calls) == 1

    # The remedy the refusal names: a Worker restart on box-a ends the spawn with its loop.
    scans.loops[ADDRESS_A] = [(7001, 5000, "S")]
    monkeypatch.setattr(mesh_liveness, "PROBE_MIN_INTERVAL_S", 0.0)
    assert mesh.get_mesh(mode="cluster").procs is fresh


def test_a_spawn_that_settled_lifts_on_no_actor_while_the_same_loops_run(monkeypatch, tmp_path):
    """Every rank's proc had started before the failure, so nothing can still be queued."""
    scans, (mesh, fresh, calls, _asked) = _queued_spawn_session(monkeypatch, tmp_path, settled=True)
    scans.loops = {ADDRESS_A: [(6001, 4000, "S")], ADDRESS_B: [(6002, 4000, "S")]}

    assert mesh.get_mesh(mode="cluster").procs is fresh
    assert len(calls) == 2


def test_a_queued_spawn_the_attach_kept_no_loop_for_holds_and_asks_for_a_restart(monkeypatch):
    """Nothing on the hosts can show that spawn gone: only a new ComfyUI process clears it."""
    from dgx_monarch.cli import lifecycle

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    _failed_spawn(spawn_settled=False, loop_gens={})
    runtime, line = _runtime(), _line(loop_gens=[])

    assert failed_bring_up.release(runtime, runner=Runner(**{ADDRESS_A: line, ADDRESS_B: line})) is False
    assert failed_bring_up.held_hosts() == ((ADDRESS_A, "unrecorded"), (ADDRESS_B, "unrecorded"))
    text = failed_bring_up.refusal(POISON)
    assert "recorded no loop for box-a, box-b, so restart ComfyUI" in text and "dgxm restart" not in text


def test_a_host_on_an_older_package_holds_until_it_can_list_its_loops(monkeypatch):
    """A Worker restart cannot help a host that prints no loop list; deploying the package does."""
    from dgx_monarch.cli import lifecycle

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(mesh_liveness, "PROBE_MIN_INTERVAL_S", 0.0)
    _failed_spawn(spawn_settled=False, loop_gens=dict(DIALED))
    runtime = _runtime()

    assert failed_bring_up.release(runtime, runner=Runner(**{ADDRESS_B: _line(loop_gens=[])})) is False
    assert failed_bring_up.held_hosts() == ((ADDRESS_A, "unlisted"),)
    text = failed_bring_up.refusal(POISON)
    assert "deploy the current package on box-a (dgxm up)" in text and "dgxm restart" not in text

    listed = Runner(**{ADDRESS_A: _line(loop_gens=["0123456789ab"]), ADDRESS_B: _line(loop_gens=[])})
    assert failed_bring_up.release(runtime, runner=listed) is True  # the package now lists its loops
    assert runtime["_TRANSPORT_POISON"] is None


@pytest.mark.parametrize(("threads", "lifts"), [("2", False), ("1", True)], ids=["threads_run", "lone_zombie"])
def test_a_zombie_leader_whose_threads_run_keeps_the_bring_up_poison(monkeypatch, tmp_path, threads, lifts):
    """The decision, not only the scan: the real scan's line through the fold and the lift."""
    from dgx_monarch.cli import lifecycle

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    _failed_spawn()
    scans = _HostScans(tmp_path)
    scans.zombie_leaders = {ADDRESS_B: threads}
    runtime = _runtime()

    assert failed_bring_up.release(runtime, runner=scans) is lifts
    assert (runtime["_TRANSPORT_POISON"] is None) is lifts
