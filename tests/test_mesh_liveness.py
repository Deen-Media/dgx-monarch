"""The dead-worker latch release: the probe, the fold, the five preconditions,
and the attach that follows a release.

Everything here is CPU only. No host is contacted: the runner is injected and
records the script it was handed, and the local path takes an injected scanner
so no test ever walks this machine's own /proc.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    mesh_health,
    mesh_helpers,
    mesh_liveness,
    mesh_recycle,
    mesh_safety,
    mesh_setup,
    mesh_teardown,
)
from dgx_monarch.cli import actor_reaper, lifecycle

ADDRESS_A = "tcp://10.0.0.1:26600"
ADDRESS_B = "tcp://10.0.0.2:26600"


@pytest.fixture(autouse=True)
def _hosts_are_remote(monkeypatch):
    """Every host here is remote unless one test says otherwise.

    The real locality answer runs `ip -o addr` and can resolve a name, and
    neither belongs in a unit test. One test below covers both branches.
    """
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)


@pytest.fixture(autouse=True)
def _idle_scanner():
    """Hand the next test an idle shared scanner.

    One worker serves every local scan in the process, so a test that wedges
    it must let it go before the next test offers a scan to it.
    """
    yield
    for _attempt in range(500):
        with mesh_liveness._scan_lock:
            if not mesh_liveness._scan_busy:
                return
        time.sleep(0.01)
    raise AssertionError("the shared liveness scanner never went idle")


def _line(**overrides) -> str:
    payload = {
        "schema": 1, "actors": 0, "actors_marked": 0, "zombies": 0,
        "unreadable": 0, "ledger_rows_alive": 0, "ledger_boot_ok": True,
        "ledger_age_s": 1.0, "loop_alive": True, "pids": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def _host(address: str):
    return SimpleNamespace(address=address, name=address)


def _config(*addresses: str):
    return SimpleNamespace(
        hosts=tuple(_host(address) for address in addresses),
        python_bin="~/venv/bin/python", worker_args={}, source="")


class Runner:
    """A recording stand-in for cli.lifecycle.run_on_host."""

    def __init__(self, answers=None, *, raises=None, default=None):
        self.answers = answers or {}
        self.default = default if default is not None else (_line(), "", 0)
        self.raises = raises
        self.calls: list[tuple[str, str, int]] = []
        self.threads: list[str] = []
        self.lock_owned: list[bool] = []
        self.handle = None

    def __call__(self, config, host, script, timeout=None):
        self.calls.append((host.address, script, timeout))
        self.threads.append(threading.current_thread().name)
        if self.handle is not None:
            owned = getattr(self.handle.lock, "_is_owned", lambda: False)
            self.lock_owned.append(owned())
        if self.raises is not None:
            raise self.raises
        stdout, stderr, code = self.answers.get(host.address, self.default)
        return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=code)


class TimeoutProcs:
    def __init__(self):
        self.calls: list[str] = []

    def stop(self, reason):
        self.calls.append(reason)
        return SimpleNamespace(
            get=lambda timeout: (_ for _ in ()).throw(
                TimeoutError("detach acknowledgement timed out")))


class FailingProcs:
    def stop(self, reason):
        return SimpleNamespace(
            get=lambda timeout: (_ for _ in ()).throw(
                RuntimeError("stop reported a failure")))


def _handle(*addresses: str, procs=None):
    from dgx_monarch.mesh import MeshHandle

    return MeshHandle(
        config=_config(*addresses), hosts=None, procs=procs, workers=None,
        world=2, gpus_per_host=1, n_hosts=max(len(addresses), 1),
        comfy_dir="", owns_hosts=True)


def _latched(monkeypatch, *addresses: str, death="ProcessExited: rank 1 is gone",
             deliberate=False, timeout=True):
    """A handle blocked by a real stop failure, through the real publish path."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _handle(*addresses, procs=TimeoutProcs() if timeout else FailingProcs())
    if death:
        handle._supervision_death_report = death
    if deliberate:
        handle._deferred_eviction_deliberate = True
    match = "ProcMesh stop timed out" if timeout else "ProcMesh stop failed"
    with pytest.raises(RuntimeError, match=match):
        handle._shutdown_impl()
    return handle


def _gone_now(*addresses: str):
    return mesh_liveness.LivenessEvidence(
        mesh_liveness.GONE,
        tuple((address, mesh_liveness.GONE) for address in addresses),
        time.monotonic())


def test_the_schema_number_matches_the_answering_module():
    assert mesh_liveness._SCHEMA == actor_reaper.LIVENESS_SCHEMA


@pytest.mark.parametrize(("payload", "expected"), [
    ({}, mesh_liveness.GONE),
    ({"zombies": 1}, mesh_liveness.GONE),
    ({"actors": 1}, mesh_liveness.ALIVE),
    ({"ledger_rows_alive": 1}, mesh_liveness.ALIVE),
    ({"unreadable": 1}, mesh_liveness.UNKNOWN),
    ({"actors": 1, "unreadable": 1}, mesh_liveness.ALIVE),
])
def test_one_hosts_line_decides_one_answer(payload, expected):
    config = _config(ADDRESS_A)
    runner = Runner(default=(_line(**payload), "", 0))
    assert mesh_liveness.probe_host(
        config, config.hosts[0], runner) == expected


@pytest.mark.parametrize("runner", [
    Runner(raises=subprocess.TimeoutExpired(cmd="ssh", timeout=10)),
    Runner(raises=OSError("no route to host")),
    Runner(default=("", "boom", 1)),
    Runner(default=("", "", 0)),
    Runner(default=(_line() + "\n" + _line(), "", 0)),
    Runner(default=(_line(schema=2), "", 0)),
    Runner(default=('{"schema": 1, "actors": "many"}', "", 0)),
    Runner(default=("not json at all", "", 0)),
])
def test_every_failure_shape_answers_unknown(runner):
    config = _config(ADDRESS_A)
    assert mesh_liveness.probe_host(
        config, config.hosts[0], runner) == mesh_liveness.UNKNOWN


def test_a_host_without_the_flag_answers_unknown_and_names_the_remedy(caplog):
    config = _config(ADDRESS_A)
    runner = Runner(default=(
        "", "usage: actor_reaper\nerror: unrecognized arguments: --liveness", 2))
    with caplog.at_level("WARNING"):
        answer = mesh_liveness.probe_host(config, config.hosts[0], runner)
    assert answer == mesh_liveness.UNKNOWN
    assert "run dgxm up" in caplog.text
    assert "restart ComfyUI if that host runs the driver" in caplog.text
    assert "unrecognized arguments" in caplog.text


def test_the_host_script_asks_for_every_configured_address():
    config = _config(ADDRESS_A, ADDRESS_B)
    runner = Runner()
    mesh_liveness.probe_host(config, config.hosts[0], runner)
    _address, script, timeout = runner.calls[0]
    assert "--liveness" in script
    assert script.count("--loop-address") == 2
    assert ADDRESS_B in script
    assert timeout == int(mesh_liveness.PROBE_HOST_BUDGET_S)


def test_the_probe_line_carries_every_count_for_the_operator(caplog):
    config = _config(ADDRESS_A)
    runner = Runner(default=(
        _line(actors=2, zombies=1, unreadable=0, ledger_rows_alive=3,
              loop_alive=False), "", 0))
    with caplog.at_level("WARNING"):
        mesh_liveness.probe_host(config, config.hosts[0], runner)
    assert (f"liveness probe host={ADDRESS_A} answer=alive actors=2 zombies=1 "
            "unreadable=0 ledger_alive=3 loop_alive=false") in caplog.text


def test_the_host_script_reads_the_checkout_on_the_box_running_the_driver(
        monkeypatch):
    # A package sync never writes the managed tree on a colocated host, so the
    # box that runs the driver must import the checkout the driver runs. The
    # deployed pair puts rank 0 there, so this branch is on every probe.
    config = _config(ADDRESS_A)
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: True)
    local = mesh_liveness.host_script(config, config.hosts[0], (ADDRESS_A,))
    assert str(lifecycle._pkg_src_dir()) in local
    assert lifecycle._MANAGED_SRC_REL not in local
    # Reading the checkout must not leave bytecode in it.
    assert "PYTHONDONTWRITEBYTECODE=1" in local
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    remote = mesh_liveness.host_script(config, config.hosts[0], (ADDRESS_A,))
    assert lifecycle._MANAGED_SRC_REL in remote
    assert str(lifecycle._pkg_src_dir()) not in remote


@pytest.mark.parametrize(("first", "second", "expected"), [
    ({}, {}, mesh_liveness.GONE),
    ({"actors": 1}, {}, mesh_liveness.ALIVE),
    ({}, {"actors": 1}, mesh_liveness.ALIVE),
    ({"unreadable": 1}, {}, mesh_liveness.UNKNOWN),
    ({"actors": 1}, {"unreadable": 1}, mesh_liveness.ALIVE),
])
def test_gone_needs_every_host_and_one_alive_is_enough(first, second, expected):
    handle = _handle(ADDRESS_A, ADDRESS_B)
    runner = Runner({ADDRESS_A: (_line(**first), "", 0),
                     ADDRESS_B: (_line(**second), "", 0)})
    evidence = mesh_liveness.probe_fleet(handle, runner=runner)
    assert evidence.answer == expected
    assert [address for address, _answer in evidence.per_host] == [
        ADDRESS_A, ADDRESS_B]


def test_a_fleet_that_runs_out_of_budget_leaves_the_rest_unknown():
    handle = _handle(ADDRESS_A, ADDRESS_B)
    runner = Runner()
    ticks = iter([0.0, 0.0, mesh_liveness.PROBE_FLEET_BUDGET_S + 1.0])
    evidence = mesh_liveness.probe_fleet(
        handle, now=lambda: next(ticks), runner=runner)
    assert evidence.per_host == (
        (ADDRESS_A, mesh_liveness.GONE), (ADDRESS_B, mesh_liveness.UNKNOWN))
    assert evidence.answer == mesh_liveness.UNKNOWN
    assert len(runner.calls) == 1


def test_a_local_mesh_scans_in_process_and_builds_no_ssh_command():
    handle = _handle()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a local mesh has no host to reach")

    evidence = mesh_liveness.probe_fleet(
        handle, runner=forbidden, scanner=lambda: json.loads(_line(actors=1)))
    assert evidence.answer == mesh_liveness.ALIVE
    assert evidence.per_host == (("local", mesh_liveness.ALIVE),)


def test_a_local_scan_that_overruns_its_join_answers_unknown(monkeypatch):
    monkeypatch.setattr(mesh_liveness, "PROBE_HOST_BUDGET_S", 0.05)
    release = threading.Event()

    def slow():
        release.wait(timeout=5)
        return json.loads(_line())

    try:
        evidence = mesh_liveness.probe_fleet(_handle(), scanner=slow)
    finally:
        release.set()
    assert evidence.answer == mesh_liveness.UNKNOWN
    assert "did not finish" in evidence.detail


def test_a_local_scan_that_raises_answers_unknown():
    def broken():
        raise OSError("procfs is not readable")

    evidence = mesh_liveness.probe_fleet(_handle(), scanner=broken)
    assert evidence.answer == mesh_liveness.UNKNOWN
    assert "OSError" in evidence.detail


def test_a_wedged_local_scan_costs_one_thread_and_names_itself(monkeypatch):
    """A thread per probe would leak one live thread per hung procfs read.

    One shared worker cannot leak: the second probe is told the first scan
    still holds it, and says so in the answer.
    """
    monkeypatch.setattr(mesh_liveness, "PROBE_HOST_BUDGET_S", 0.05)
    release = threading.Event()

    def wedged():
        release.wait(timeout=10)
        return json.loads(_line())

    try:
        first = mesh_liveness.probe_fleet(_handle(), scanner=wedged)
        threads = threading.active_count()
        second = mesh_liveness.probe_fleet(_handle(), scanner=wedged)
        assert threading.active_count() <= threads
    finally:
        release.set()
    assert first.answer == second.answer == mesh_liveness.UNKNOWN
    assert "did not finish" in first.detail
    assert "stuck on an earlier scan" in second.detail


def test_a_gone_probe_retires_the_latch_once_and_issues_no_stop(monkeypatch, caplog):
    handle = _latched(monkeypatch, ADDRESS_A)
    handle.abandoned_sample_leases = {0: 1}
    runner = Runner()
    with caplog.at_level("WARNING"):
        assert mesh_liveness.try_release(handle, runner=runner) is True
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert handle._replacement_retryable is False
    assert mesh_teardown.block_cause(handle) is None
    assert not mesh_safety.token_present(id(handle))
    assert mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError) == "completed"
    assert handle.procs.calls == ["dgx-monarch client detach"]
    assert ("replacement latch released on liveness proof "
            "cause=stop_timeout_after_death") in caplog.text
    assert f"hosts={ADDRESS_A}=gone abandoned=1" in caplog.text
    # A second call finds a completed handle and changes nothing.
    assert mesh_liveness.try_release(handle, runner=runner) is False
    assert handle.procs.calls == ["dgx-monarch client detach"]
    assert len(runner.calls) == 1


def test_the_release_emits_one_telemetry_event(monkeypatch):
    from dgx_monarch import telemetry

    handle = _latched(monkeypatch, ADDRESS_A)
    monkeypatch.setattr(telemetry, "_EVENTS", telemetry._EVENTS.__class__(maxlen=8))
    assert mesh_liveness.try_release(handle, runner=Runner()) is True
    events = [event for event in telemetry.events_tail(8)
              if event["kind"] == "latch_release"]
    assert len(events) == 1
    assert events[0]["cause"] == mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH
    assert events[0]["hosts"] == f"{ADDRESS_A}=gone"


@pytest.mark.parametrize("payload", [{"actors": 1}, {"unreadable": 1}])
def test_anything_short_of_gone_keeps_the_latch(monkeypatch, payload):
    handle = _latched(monkeypatch, ADDRESS_A)
    runner = Runner(default=(_line(**payload), "", 0))
    assert mesh_liveness.try_release(handle, runner=runner) is False
    assert handle.teardown_complete is False
    assert handle.replacement_blocked


def test_a_reported_stop_failure_is_never_released_by_a_probe(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A, timeout=False)
    assert mesh_teardown.block_cause(handle) == mesh_teardown.TAG_STOP_REPORTED_FAILURE
    runner = Runner()
    assert mesh_liveness.try_release(handle, runner=runner) is False
    assert runner.calls == []


def test_a_timeout_with_no_death_reported_is_never_released_by_a_probe(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A, death="")
    assert mesh_teardown.block_cause(handle) == mesh_teardown.TAG_STOP_TIMEOUT_DELIBERATE
    assert mesh_liveness.try_release(handle, runner=Runner()) is False
    assert handle.teardown_complete is False


def test_a_deliberate_eviction_route_is_never_released_by_a_probe(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A, deliberate=True)
    assert mesh_teardown.block_cause(handle) == mesh_teardown.TAG_STOP_TIMEOUT_DELIBERATE
    assert mesh_liveness.try_release(handle, runner=Runner()) is False


def test_a_live_sample_lease_refuses_the_release(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    handle.sample_leases = {0: 1}
    assert mesh_teardown.retire_after_liveness_proof(
        handle, _gone_now(ADDRESS_A)) is False
    assert handle.teardown_complete is False


def test_a_pending_read_refuses_the_release(monkeypatch):
    from dgx_monarch import rdma_read_job

    handle = _latched(monkeypatch, ADDRESS_A)
    monkeypatch.setattr(
        rdma_read_job, "pending_job_count_for_handle", lambda _handle: 1)
    assert mesh_teardown.retire_after_liveness_proof(
        handle, _gone_now(ADDRESS_A)) is False
    assert handle.teardown_complete is False


def test_an_abandoned_lease_is_accepted_and_counted_in_the_evidence(monkeypatch, caplog):
    handle = _latched(monkeypatch, ADDRESS_A)
    handle.abandoned_sample_leases = {0: 1, 1: 2}
    with caplog.at_level("WARNING"):
        assert mesh_teardown.retire_after_liveness_proof(
            handle, _gone_now(ADDRESS_A)) is True
    assert "abandoned=3" in caplog.text


def test_a_live_reconcile_thread_refuses_the_release(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    release = threading.Event()
    thread = threading.Thread(target=release.wait, kwargs={"timeout": 5})
    thread.start()
    try:
        handle._supervision_reconcile_thread = thread
        assert mesh_teardown.retire_after_liveness_proof(
            handle, _gone_now(ADDRESS_A)) is False
    finally:
        release.set()
        thread.join(timeout=5)
    assert handle.teardown_complete is False
    # The same thread, now finished, no longer blocks the release.
    assert mesh_teardown.retire_after_liveness_proof(
        handle, _gone_now(ADDRESS_A)) is True
    assert handle._supervision_reconcile_thread is None


def test_a_partly_gone_fleet_never_releases(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A, ADDRESS_B)
    evidence = mesh_liveness.LivenessEvidence(
        mesh_liveness.GONE,
        ((ADDRESS_A, mesh_liveness.GONE), (ADDRESS_B, mesh_liveness.UNKNOWN)),
        time.monotonic())
    assert mesh_teardown.retire_after_liveness_proof(handle, evidence) is False


def test_several_blocked_attempts_inside_the_floor_cost_one_probe(monkeypatch):
    # A queue of blocked renders calls the door in a burst. For an alive answer
    # the floor, not the evidence window, keeps that burst to one ssh probe.
    handle = _latched(monkeypatch, ADDRESS_A)
    runner = Runner(default=(_line(actors=1), "", 0))
    for _attempt in range(3):
        assert mesh_liveness.try_release(handle, runner=runner) is False
    assert len(runner.calls) == 1


def test_an_alive_answer_past_the_floor_is_probed_again(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    handle._liveness_evidence = mesh_liveness.LivenessEvidence(
        mesh_liveness.ALIVE, ((ADDRESS_A, mesh_liveness.ALIVE),),
        time.monotonic() - (mesh_liveness.PROBE_MIN_INTERVAL_S + 1.0))
    runner = Runner(default=(_line(actors=1), "", 0))
    assert mesh_liveness.try_release(handle, runner=runner) is False
    assert len(runner.calls) == 1


def test_an_unknown_answer_past_the_floor_is_probed_again(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    handle._liveness_evidence = mesh_liveness.LivenessEvidence(
        mesh_liveness.UNKNOWN, ((ADDRESS_A, mesh_liveness.UNKNOWN),),
        time.monotonic() - (mesh_liveness.PROBE_MIN_INTERVAL_S + 1.0))
    runner = Runner(default=(_line(actors=1), "", 0))
    assert mesh_liveness.try_release(handle, runner=runner) is False
    assert len(runner.calls) == 1


def test_a_rank_that_dies_after_an_alive_answer_releases_on_the_next_call(
        monkeypatch):
    # An alive answer stands only for the floor. Kept for the 30 s evidence
    # window, it would hold the latch on a rank that exits five seconds later
    # for twenty five seconds more.
    handle = _latched(monkeypatch, ADDRESS_A)
    handle._liveness_evidence = mesh_liveness.LivenessEvidence(
        mesh_liveness.ALIVE, ((ADDRESS_A, mesh_liveness.ALIVE),),
        time.monotonic() - (mesh_liveness.PROBE_MIN_INTERVAL_S + 1.0))
    assert mesh_liveness.try_release(handle, runner=Runner()) is True
    assert handle.teardown_complete is True


def test_a_fresh_gone_inside_the_window_is_not_probed_again(monkeypatch):
    # Gone is the one answer worth keeping: it is the only one that releases,
    # and the retire re-reads its age under the handle lock before acting.
    handle = _latched(monkeypatch, ADDRESS_A)
    handle._liveness_evidence = mesh_liveness.LivenessEvidence(
        mesh_liveness.GONE, ((ADDRESS_A, mesh_liveness.GONE),),
        time.monotonic() - (mesh_liveness.PROBE_MIN_INTERVAL_S + 1.0))
    runner = Runner(default=(_line(actors=1), "", 0))
    assert mesh_liveness.try_release(handle, runner=runner) is True
    assert runner.calls == []


def test_a_sub_second_budget_reaches_the_runner_whole(monkeypatch):
    # Rounded up to a whole second, the remaining budget would let the last
    # host of a fleet pass overrun PROBE_FLEET_BUDGET_S.
    runner = Runner()
    ticks = iter([0.0, mesh_liveness.PROBE_FLEET_BUDGET_S - 0.3])
    mesh_liveness.probe_fleet(
        _handle(ADDRESS_A), now=lambda: next(ticks), runner=runner)
    assert runner.calls[0][2] == pytest.approx(0.3)


def test_the_host_budget_has_a_floor_well_under_a_second():
    runner = Runner()
    config = _config(ADDRESS_A)
    mesh_liveness.probe_host(config, config.hosts[0], runner, 0.0)
    assert runner.calls[0][2] == pytest.approx(0.05)


def test_a_stale_gone_is_re_probed_before_any_release(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    handle._liveness_evidence = mesh_liveness.LivenessEvidence(
        mesh_liveness.GONE, ((ADDRESS_A, mesh_liveness.GONE),),
        time.monotonic() - (mesh_teardown.EVIDENCE_MAX_AGE_S + 5.0))
    runner = Runner(default=(_line(actors=1), "", 0))
    assert mesh_liveness.try_release(handle, runner=runner) is False
    assert len(runner.calls) == 1
    assert handle.teardown_complete is False


def test_evidence_older_than_the_window_never_retires(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    stale = mesh_liveness.LivenessEvidence(
        mesh_liveness.GONE, ((ADDRESS_A, mesh_liveness.GONE),),
        time.monotonic() - (mesh_teardown.EVIDENCE_MAX_AGE_S + 1.0))
    assert mesh_teardown.retire_after_liveness_proof(handle, stale) is False


def test_the_probe_never_runs_on_the_reconcile_thread(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    runner = Runner()
    seen: list[bool] = []

    def on_reconcile():
        seen.append(mesh_liveness.try_release(handle, runner=runner))

    thread = threading.Thread(target=on_reconcile, name="dgxm-reconcile-stop")
    thread.start()
    thread.join(timeout=5)
    assert seen == [False]
    assert runner.calls == []
    # The same handle on any other thread does reach the probe.
    assert mesh_liveness.try_release(handle, runner=runner) is True
    assert runner.threads == ["MainThread"]


def _bind_runner(monkeypatch, runner, order=None):
    """Send every try_release, the recycle door's included, to this runner."""
    real = mesh_liveness.try_release

    def bound(handle, **_kwargs):
        if order is not None:
            order.append("probe")
        return real(handle, runner=runner)

    monkeypatch.setattr(mesh_liveness, "try_release", bound)


def test_the_probe_runs_before_the_lifecycle_lock_is_taken(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    runner = Runner()
    runner.handle = handle
    _bind_runner(monkeypatch, runner)
    retired: list[str] = []
    outcome = mesh_recycle.recycle_detailed_impl(
        handle, lambda: retired.append("retire"), lambda: retired.append("evict"))
    assert outcome.status is mesh_recycle.RecycleStatus.RECYCLED
    assert outcome.already_recycled is True
    assert retired == ["retire"]
    assert runner.lock_owned == [False]


def test_the_recycle_door_still_refuses_when_the_probe_says_alive(monkeypatch):
    handle = _latched(monkeypatch, ADDRESS_A)
    _bind_runner(monkeypatch, Runner(default=(_line(actors=1), "", 0)))
    outcome = mesh_recycle.recycle_detailed_impl(handle, lambda: None, lambda: None)
    assert outcome.status is mesh_recycle.RecycleStatus.PRIOR_TEARDOWN_UNKNOWN
    assert mesh_teardown.sentence_for_cause(
        mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH) in outcome.detail


def test_a_failed_bring_up_after_a_release_names_it_in_the_poison_text(monkeypatch):
    monkeypatch.setattr(mesh_teardown, "_RELEASED_PREDECESSOR", None)
    released = _latched(monkeypatch, ADDRESS_A)
    assert mesh_liveness.try_release(released, runner=Runner()) is True
    fresh = _handle(ADDRESS_A)
    mesh_teardown.publish_creation_block(fresh, "owned ProcMesh cleanup unconfirmed")
    assert fresh._replacement_retryable is False
    assert "owned ProcMesh cleanup unconfirmed" in fresh.replacement_blocked
    assert "proved its worker processes were gone" in fresh.replacement_blocked
    assert (mesh_teardown.block_cause(fresh)
            == mesh_teardown.TAG_CREATION_CLEANUP_UNCONFIRMED)
    # One shot: the next failure names only itself.
    later = _handle(ADDRESS_A)
    mesh_teardown.publish_creation_block(later, "spawn rollback unconfirmed")
    assert "proved its worker processes were gone" not in later.replacement_blocked


def test_a_release_older_than_one_bring_up_no_longer_claims_the_failure(monkeypatch):
    # The ordinary path is a release whose replacement succeeds and consumes
    # nothing. An unrelated failure later must not claim that release.
    monkeypatch.setattr(
        mesh_teardown, "_RELEASED_PREDECESSOR",
        time.monotonic() - (mesh_teardown._release_note_window_s() + 1.0))
    later = _handle(ADDRESS_A)
    mesh_teardown.publish_creation_block(later, "spawn rollback unconfirmed")
    assert "proved its worker processes were gone" not in later.replacement_blocked


def test_a_creation_block_never_poisons_a_handle_whose_stop_confirmed():
    handle = _handle(ADDRESS_A)
    handle.teardown_complete = True
    mesh_teardown.publish_creation_block(handle, "cleanup unconfirmed")
    assert handle.replacement_blocked is None


def _publish(handle, text, **kwargs):
    mesh_helpers._publish_supervision_failure_state(
        handle, RuntimeError(text), text, **kwargs)


def test_a_classified_supervision_death_is_recorded_and_truncated(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _handle(ADDRESS_A)
    text = "{'SupervisionError'} ProcessExited: rank 1 " + "e" * 400
    _publish(handle, text)
    assert handle._supervision_death_report == text[:200]
    assert (mesh_teardown.timeout_cause(handle)
            == mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH)


def test_a_deliberate_route_records_no_death(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _handle(ADDRESS_A)
    _publish(handle, "{'SupervisionError'} the stall guard evicted this fleet",
             deliberate_route=True)
    assert getattr(handle, "_supervision_death_report", None) is None
    assert (mesh_teardown.timeout_cause(handle)
            == mesh_teardown.TAG_STOP_TIMEOUT_DELIBERATE)


def test_a_fault_carrying_a_reset_stop_reason_records_no_death(monkeypatch):
    # A supervision error raised at an actor call during a Reset carries that
    # Reset's own stop reason and is absorbed as deliberate. Nothing died, so
    # a stop timeout on this handle must stay one no proof can release.
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _handle(ADDRESS_A)
    _publish(handle, "{'SupervisionError'} dgx-monarch recycle: channel closed")
    assert getattr(handle, "_supervision_death_report", None) is None
    assert (mesh_teardown.timeout_cause(handle)
            == mesh_teardown.TAG_STOP_TIMEOUT_DELIBERATE)


ALL_TAGS = (
    None,
    mesh_teardown.TAG_NO_STOP_ISSUED,
    mesh_teardown.TAG_STOP_REPORTED_FAILURE,
    mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH,
    mesh_teardown.TAG_STOP_TIMEOUT_DELIBERATE,
    mesh_teardown.TAG_SETUP_ROLLBACK_INTERRUPTED,
    mesh_teardown.TAG_STOP_INTERRUPTED,
    mesh_teardown.TAG_CREATION_CLEANUP_UNCONFIRMED,
)


def test_every_cause_names_its_own_clearing_action():
    sentences = [mesh_teardown.sentence_for_cause(tag) for tag in ALL_TAGS]
    assert len(set(sentences)) == len(sentences)
    assert all(sentence.strip() for sentence in sentences)
    # Only the after-death timeout promises a self-heal, because it is the only
    # cause the driver can prove its way out of.
    healing = [tag for tag in ALL_TAGS
               if "the next attach" in mesh_teardown.sentence_for_cause(tag)]
    assert healing == [mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH]


@pytest.mark.parametrize("tag", ALL_TAGS[1:])
def test_each_cause_reaches_every_operator_surface(monkeypatch, tag):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    # No host is reachable from a unit test, and this handle carries no proof.
    monkeypatch.setattr(mesh_liveness, "try_release", lambda _h, **_k: False)
    sentence = mesh_teardown.sentence_for_cause(tag)
    handle = _handle(ADDRESS_A)
    handle.replacement_blocked = "an earlier stop"
    handle._block_cause = tag
    assert sentence in mesh_teardown.blocked_refusal(handle, "get_mesh")
    assert "could not be stopped safely" in mesh_teardown.blocked_refusal(
        handle, "get_mesh")
    assert sentence in mesh_teardown.blocked_refusal(handle, "ensure_live")
    assert sentence in mesh_health.health_of([handle]).remedy
    outcome = mesh_recycle.recycle_detailed_impl(handle, lambda: None, lambda: None)
    assert outcome.status is mesh_recycle.RecycleStatus.PRIOR_TEARDOWN_UNKNOWN
    assert sentence in outcome.detail


def test_the_panel_reason_names_the_cause_without_repeating_the_remedy():
    handle = _handle(ADDRESS_A)
    handle.replacement_blocked = "an earlier stop"
    handle._block_cause = mesh_teardown.TAG_STOP_TIMEOUT_DELIBERATE
    health = mesh_health.health_of([handle])
    assert "an earlier attached-mesh process stop failed" in health.reason
    assert "the stop timed out with no worker death reported" in health.reason
    assert health.state == "dirty"


def test_an_unrecorded_cause_still_answers_every_surface():
    handle = _handle(ADDRESS_A)
    handle.replacement_blocked = "an earlier stop"
    assert mesh_teardown.block_cause(handle) is None
    assert "restart ComfyUI" in mesh_teardown.blocked_refusal(handle, "ensure_live")
    assert mesh_health.health_of([handle]).remedy


def test_a_lifecycle_lock_is_never_reversed_by_the_probe(monkeypatch):
    # mesh_setup.lifecycle_lock is the recycle door's lock; the probe must have
    # finished before it is taken, on every path, including the refusing one.
    handle = _latched(monkeypatch, ADDRESS_A)
    order: list[str] = []
    real_lock = mesh_setup.lifecycle_lock

    def recording(*args, **kwargs):
        order.append("lock")
        return real_lock(*args, **kwargs)

    _bind_runner(monkeypatch, Runner(default=(_line(actors=1), "", 0)), order)
    monkeypatch.setattr(mesh_setup, "lifecycle_lock", recording)
    mesh_recycle.recycle_detailed_impl(handle, lambda: None, lambda: None)
    assert order == ["probe", "lock"]


# Replacement attach after a liveness release gets one attempt and no listener
# wait. In torchmonarch 0.6.0, hardware checks on 2026-10-01 found that retries
# could not recover a process whose fleet changed underneath it: actor death,
# one- or two-loop restart, and HostMesh.stop() all required both Worker loops
# to restart together, followed by a fresh driver. The dead-actor failure first
# appeared on 2026-09-03. See docs/TROUBLESHOOTING.md #2 for the evidence table.
def _cluster_config(*addresses: str, auto_heal=False):
    from dgx_monarch.config import ClusterConfig, HostConfig

    return ClusterConfig(
        hosts=tuple(HostConfig(name=f"h{index}", address=address)
                    for index, address in enumerate(addresses, 1)),
        client_bind="tcp://10.0.0.1:0", auto_heal=auto_heal,
        transport_security="trusted_fabric")


class _Listeners:
    """A per-poll answer for every host, and a record of what was asked."""

    def __init__(self, polls):
        self.polls = list(polls)
        self.asked: list[str] = []

    def __call__(self, _config, host, **_kwargs):
        self.asked.append(host.address)
        healthy = self.polls[0] if len(self.polls) == 1 else self.polls.pop(0)
        return {"healthy": healthy, "running": healthy, "listening": healthy,
                "mode": "systemd", "error": None}


@pytest.fixture
def attach_rig(monkeypatch):
    """The cluster attach with monarch's transport, ssh and sleep stubbed out."""
    import monarch.actor as monarch_actor

    from dgx_monarch import attach_trace, mesh_attach
    from dgx_monarch import mesh as mesh_mod
    from dgx_monarch.cli import listener_generation, worker_health

    # The attach reads each loop's listener generation over the lifecycle
    # runner. Canned here so the rig stays free of ssh, and the process-lifetime
    # readings start empty so no earlier test's reading is compared against.
    monkeypatch.setattr(attach_trace, "_LAST_SEEN", {})
    monkeypatch.setattr(
        listener_generation, "fleet",
        lambda config, **_kwargs: {
            host.address: {"gen": "aaaaaaaaaaaa", "loop_pid": "111",
                           "age_s": "9", "listening": "true", "marker": "match"}
            for host in config.hosts})
    monkeypatch.setattr(mesh_teardown, "_RELEASED_PREDECESSOR", None)
    monkeypatch.setattr(mesh_teardown, "_RELEASED_ADDRESSES", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_POISON", None)
    monkeypatch.setattr(mesh_mod, "_TRANSPORT_BIND", None)
    monkeypatch.setattr(monarch_actor, "enable_transport", lambda *_a, **_k: None)
    slept: list[float] = []
    monkeypatch.setattr(mesh_attach.time, "sleep", slept.append)
    restarts: list[bool] = []
    monkeypatch.setattr(mesh_attach, "restart_after_failed_attach",
                        lambda *_a, **_k: restarts.append(True) or True)
    attempts: list[object] = []

    def rig(attach_results, listeners=None, *, auto_heal=False):
        def attach_once(addresses):
            result = attach_results.pop(0)
            attempts.append(result)
            if isinstance(result, BaseException):
                raise result
            return result

        monkeypatch.setattr(mesh_mod, "_attach_once", attach_once)
        if listeners is not None:
            monkeypatch.setattr(worker_health, "passive_worker_health", listeners)
        monkeypatch.setattr(mesh_mod, "_heal_dead_loops", lambda *_a, **_k: None)
        return mesh_mod

    return SimpleNamespace(rig=rig, slept=slept, restarts=restarts,
                           attempts=attempts, config=_cluster_config)


def _release_the_latch(monkeypatch, *addresses: str):
    """Retire a real stop-timeout latch on a real fleet-wide gone proof."""
    handle = _latched(monkeypatch, *addresses)
    runner = Runner()
    assert mesh_liveness.try_release(handle, runner=runner) is True
    assert handle.teardown_complete is True
    assert mesh_teardown.released_predecessor_recently() is True
    return handle


def test_a_released_replacement_fails_fast_after_exactly_one_attempt(
        monkeypatch, attach_rig):
    """A released fleet gets exactly one try, since a second cannot succeed
    (the comment above `_cluster_config`), and the error says why no retry
    ran."""
    _release_the_latch(monkeypatch, ADDRESS_A)
    mesh_mod = attach_rig.rig([RuntimeError("worker gone")], _Listeners([True]))
    with pytest.raises(mesh_mod.MeshAttachError) as raised:
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A))
    assert len(attach_rig.attempts) == 1
    assert raised.value.__cause__ is attach_rig.attempts[0]
    text = str(raised.value)
    assert "reproduced on hardware, 2026-10-01" in text
    assert "docs/TROUBLESHOOTING.md #101" in text
    assert mesh_mod._TRANSPORT_POISON == "cluster attach failed: RuntimeError('worker gone')"
    assert attach_rig.slept == []          # no wait ran either


def test_a_released_fail_fast_restarts_the_loops_together_when_auto_heal_is_on(
        monkeypatch, attach_rig):
    _release_the_latch(monkeypatch, ADDRESS_A)
    mesh_mod = attach_rig.rig([RuntimeError("worker gone")], _Listeners([True]))
    with pytest.raises(mesh_mod.MeshAttachError) as raised:
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A, auto_heal=True))
    assert attach_rig.restarts == [True]
    assert "worker loops were restarted together" in str(raised.value)


def test_a_released_fail_fast_leaves_the_restart_to_the_operator_when_auto_heal_is_off(
        monkeypatch, attach_rig):
    _release_the_latch(monkeypatch, ADDRESS_A)
    mesh_mod = attach_rig.rig([RuntimeError("worker gone")], _Listeners([True]))
    with pytest.raises(mesh_mod.MeshAttachError) as raised:
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A, auto_heal=False))
    assert attach_rig.restarts == []
    assert "Run `dgxm restart` first." in str(raised.value)


def test_a_released_fail_fast_says_so_when_the_restart_did_not_confirm(
        monkeypatch, attach_rig):
    """The restart is best effort. A message that claimed it ran would send
    the operator to restart ComfyUI against loops that never restarted."""
    from dgx_monarch import mesh_attach

    _release_the_latch(monkeypatch, ADDRESS_A)
    mesh_mod = attach_rig.rig([RuntimeError("worker gone")], _Listeners([True]))
    monkeypatch.setattr(mesh_attach, "restart_after_failed_attach", lambda *_a, **_k: False)
    with pytest.raises(mesh_mod.MeshAttachError) as raised:
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A, auto_heal=True))
    text = str(raised.value)
    assert "worker loops were restarted together" not in text
    assert "did not confirm" in text and "run `dgxm restart` first" in text


def test_a_release_of_another_fleet_leaves_this_attach_its_retry(
        monkeypatch, attach_rig):
    """The release note names the fleet the proof released; an attach to a
    different host set inside the window keeps its two attempts."""
    _release_the_latch(monkeypatch, ADDRESS_B)
    mesh_mod = attach_rig.rig([RuntimeError("blip"), RuntimeError("blip again")],
                              _Listeners([True]))
    with pytest.raises(mesh_mod.MeshAttachError):
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A))
    assert len(attach_rig.attempts) == 2


def test_an_attach_with_no_third_attempt_still_raises_the_second(
        monkeypatch, attach_rig):
    """With no release there is no fail-fast: the attach makes its two
    attempts and raises the second failure."""
    listeners = _Listeners([True])
    mesh_mod = attach_rig.rig(
        [RuntimeError("first timeout"), RuntimeError("second timeout")],
        listeners)
    assert mesh_teardown.released_predecessor_recently() is False
    with pytest.raises(mesh_mod.MeshAttachError) as raised:
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A))
    assert "second timeout" in str(raised.value)
    assert raised.value.__cause__ is attach_rig.attempts[1]
    assert listeners.asked == []


def test_an_attach_with_no_liveness_release_poisons_exactly_as_before(
        monkeypatch, attach_rig):
    """Only a replacement the driver authorized itself fails fast. Every other
    attach keeps its two attempts and the transport poison, and asks no host
    for its listener state."""
    listeners = _Listeners([True])
    mesh_mod = attach_rig.rig(
        [RuntimeError("wedge"), RuntimeError("wedge"), "HOSTS"], listeners)
    assert mesh_teardown.released_predecessor_recently() is False
    with pytest.raises(mesh_mod.MeshAttachError, match="in-process retry"):
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A))
    assert len(attach_rig.attempts) == 2
    assert listeners.asked == []
    assert mesh_mod._TRANSPORT_POISON == "cluster attach failed: RuntimeError('wedge')"


def test_the_release_is_read_before_the_attach_spends_its_window(
        monkeypatch, attach_rig):
    """The note lives for one creation timeout, and the released fail-fast
    spends its one attempt before it can act on the answer. Asked after that
    attempt, the note could already be stale and lose the fail-fast in the
    slowest incident."""
    from dgx_monarch import mesh_attach

    _release_the_latch(monkeypatch, ADDRESS_A)
    mesh_mod = attach_rig.rig([RuntimeError("blip")], _Listeners([True]))
    order: list[str] = []
    real_gate = mesh_teardown.released_predecessor_recently
    real_attach = mesh_mod._attach_once

    def gate(*args, **kwargs):
        order.append("gate")
        return real_gate(*args, **kwargs)

    def attach(addresses):
        order.append("attach")
        return real_attach(addresses)

    monkeypatch.setattr(mesh_attach.mesh_teardown,
                        "released_predecessor_recently", gate)
    monkeypatch.setattr(mesh_mod, "_attach_once", attach)
    with pytest.raises(mesh_mod.MeshAttachError):
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A))
    assert order == ["gate", "attach"]


def test_a_stale_release_does_not_authorize_a_fail_fast(monkeypatch, attach_rig):
    """The note is time bounded as well as one shot: a release from hours ago
    is not the reason this attach failed."""
    listeners = _Listeners([True])
    mesh_mod = attach_rig.rig(
        [RuntimeError("wedge"), RuntimeError("wedge"), "HOSTS"], listeners)
    mesh_teardown.note_liveness_release(now=time.monotonic() - 10_000.0)
    assert mesh_teardown.released_predecessor_recently() is False
    with pytest.raises(mesh_mod.MeshAttachError, match="in-process retry"):
        mesh_mod._attach_cluster(attach_rig.config(ADDRESS_A))
    assert listeners.asked == []


def test_reading_the_release_never_spends_the_note_the_poison_text_needs(
        monkeypatch):
    """``publish_creation_block`` owns the one shot. If the attach's own read
    consumed it, the bring-up that then failed would lose the sentence saying
    which fleet it was replacing."""
    monkeypatch.setattr(mesh_teardown, "_RELEASED_PREDECESSOR", None)
    mesh_teardown.note_liveness_release()
    assert mesh_teardown.released_predecessor_recently() is True
    assert mesh_teardown.released_predecessor_recently() is True
    handle = _handle(ADDRESS_A)
    mesh_teardown.publish_creation_block(handle, "cleanup unconfirmed")
    assert "worker processes were gone" in handle.replacement_blocked
    assert mesh_teardown.released_predecessor_recently() is False
