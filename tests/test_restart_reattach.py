"""The first render after a Worker restart: a fast price-wait end, then a re-attach.

Issue #47, docs/TROUBLESHOOTING.md #2. After the Worker loops restart under an
attached mesh, a message to the old actors comes back to the fault hook as
undeliverable and the price request's future never resolves. The capacity
wait now ends within one poll of such a failure, when it names the mesh the
wait prices on (``delivery_faults``), and a render that owns its session reads
the loops' ids, retires the stale mesh and attaches by the new ids before it
dispatches again (``restart_reattach``).

CPU only. Monarch, ssh, the liveness scan and the fleet spawn are stubbed; the
fault record, the wait, the move check, the teardown, the latch release, the
attach and the render's retry are the real code.
"""
from __future__ import annotations

import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import (
    attach_trace,
    capacity_agreement,
    delivery_faults,
    mesh_attach,
    mesh_factory,
    mesh_helpers,
    mesh_liveness,
    mesh_safety,
    mesh_teardown,
    restart_reattach,
    telemetry,
    worker_proc_id,
)
from dgx_monarch import mesh as mesh_mod
from dgx_monarch.actor.capacity_quote import RankPrice
from dgx_monarch.capacity_agreement import FleetQuoteUnavailableError, decide
from dgx_monarch.cli import listener_generation
from dgx_monarch.delivery_faults import WorkerDeliveryError
from dgx_monarch.refusal import parse_leading_refusal_tag

SRC = Path(__file__).resolve().parents[1] / "src" / "dgx_monarch"
A, B = "tcp://10.0.0.1:26600", "tcp://10.0.0.2:26600"
OLD = {A: "aaaaaaaaaaaa0001", B: "bbbbbbbbbbbb0001"}
NEW = {A: "aaaaaaaaaaaa0002", B: "bbbbbbbbbbbb0002"}
SOURCE = "/fake/cluster.toml"
# The uid of the fleet mesh the price waits on, as ActorMesh._name reads it.
LIVE = "OldMesh01"


def _fault(address: str, mesh: str = f"dgxm_worker<{LIVE}>") -> str:
    """The shape the 2026-10-15 driver log shows for a call to a restarted Worker."""
    return (
        "MeshFailure(mesh_name=<none>, crashed_ranks=[], event=Supervision event: actor "
        "<RootClientActor client> failed:\n  undeliverable message for "
        "dgxm_worker.capacity_quote() (call_one):\n\terror: delivery failure: ttl expired "
        f"for {mesh}.anon-0<Proc01>:<Port01>@anon-0<Proc01>.{address}\n)")


# ---------------------------------------------------------------- the record


def test_a_failure_names_its_own_address_and_never_a_longer_port():
    assert delivery_faults.record(_fault(A))
    assert delivery_faults.failed_since(0.0, [A, B], LIVE) == [A]
    assert delivery_faults.failed_since(0.0, ["tcp://10.0.0.1:2660"], LIVE) == []


def test_a_failure_for_another_actor_or_with_no_marker_is_not_kept():
    controller = ("undeliverable message to host_agent.service<Agent01>@tcp://10.0.0.1:26600: "
                  "delivery failure: ttl expired")
    assert not delivery_faults.record(controller)
    assert not delivery_faults.record("dgxm_worker<Mesh01> answered tcp://10.0.0.1:26600")
    assert delivery_faults.failed_since(0.0, [A], LIVE) == []


def test_a_failure_recorded_before_the_wait_started_does_not_count():
    delivery_faults.record(_fault(A), now=10.0)
    assert delivery_faults.failed_since(10.5, [A], LIVE) == []
    assert delivery_faults.failed_since(10.0, [A], LIVE) == [A]


def test_only_a_failure_naming_the_watched_mesh_counts():
    """A mesh that Recycle, Reset, the supervision reconcile or a re-attach
    retired served the same address under another uid."""
    delivery_faults.record(_fault(A, "dgxm_worker<RetiredByReset>"))
    assert delivery_faults.failed_since(0.0, [A], LIVE) == []
    delivery_faults.record(_fault(A))
    assert delivery_faults.failed_since(0.0, [A], LIVE) == [A]
    assert delivery_faults.failed_since(0.0, [A], "RetiredByReset") == [A]
    # A mesh whose name could not be read matches nothing.
    assert delivery_faults.failed_since(0.0, [A], None) == []


def test_the_mesh_uid_reads_the_name_monarch_gives_the_mesh():
    assert delivery_faults.mesh_uid("dgxm_worker-zHy4wqhm79S") == "zHy4wqhm79S"
    for name in ("logger-nGBCqCCdMCb", "dgxm_worker-", "dgxm_worker", "", None, 7,
                 "dgxm_worker-a<b>"):
        assert delivery_faults.mesh_uid(name) is None, name


def test_the_record_reads_the_markers_the_fault_marker_site_matches():
    assert delivery_faults.MARKERS == mesh_safety._TEARDOWN_FAULT_MARKERS


def test_the_fault_marker_site_records_and_a_stop_reason_does_not():
    mesh_safety.is_deliberate_teardown_fault(_fault(A), live_mesh_exists=True)
    assert delivery_faults.failed_since(0.0, [A], LIVE) == [A]
    delivery_faults.clear()
    mesh_safety.is_deliberate_teardown_fault(
        _fault(A) + " dgx-monarch recycle", live_mesh_exists=True)
    assert delivery_faults.failed_since(0.0, [A], LIVE) == []


def test_the_watched_mesh_name_is_the_one_the_fleet_spawns():
    source = (SRC / "mesh_factory.py").read_text()
    assert f'procs.spawn("{delivery_faults.WORKER_MESH}", GPUWorker)' in source


# ------------------------------------------------------------- the price wait


def _row(rank: int) -> dict:
    return {"host": f"box{rank}", "rank": rank, "quotes": [
        RankPrice(verdict="fits", rung="stock_fits", reason="fits").to_row()]}


class _Name:
    """``ActorMesh._name``: a future of the mesh's name."""

    def __init__(self, name: str | None) -> None:
        self.name = name

    def get(self, timeout=None):
        if self.name is None:
            raise TimeoutError("no name")
        return self.name


class _Fleet:
    """A two-host fleet whose futures answer, or block for the step they are given.

    A rank whose answer is a number answers after that many seconds.
    """

    def __init__(self, answers: dict[int, bool | float], *, addresses=(A, B),
                 name: str | None = f"dgxm_worker-{LIVE}") -> None:
        self.world, self.gpus_per_host = 2, 1
        self.config = SimpleNamespace(hosts=tuple(
            SimpleNamespace(name=f"h{index}", address=address)
            for index, address in enumerate(addresses)))
        self.defunct, self.setup_cleanup_state, self.setup_key = False, None, ("ready",)
        self.answers = answers
        self.sent: list[int] = []
        self.steps: list[float] = []
        self.started = time.monotonic()
        self.workers = types.SimpleNamespace(
            extent=types.SimpleNamespace(labels=["hosts", "gpus"]), slice=self._slice,
            _name=_Name(name))

    def _slice(self, **coords):
        rank = coords["hosts"]
        return types.SimpleNamespace(capacity_quote=types.SimpleNamespace(
            call_one=lambda _request: self.sent.append(rank) or rank))

    def _await_or_evict(self, future, timeout_s):
        self.steps.append(timeout_s)
        answer = self.answers[future]
        if answer is True:
            return _row(future)
        if answer is not False:
            due = self.started + float(answer) - time.monotonic()
            if due <= timeout_s:
                time.sleep(max(due, 0.0))
                return _row(future)
        time.sleep(timeout_s)
        raise TimeoutError("no answer")


def test_a_failure_recorded_during_the_wait_ends_it_within_one_poll():
    """The mutation that drops the short circuit makes this wait run to its
    deadline and refuse as a silent rank instead."""
    fleet = _Fleet({0: True, 1: False})
    recorded_at: list[float] = []

    def restart() -> None:
        time.sleep(0.3)
        recorded_at.append(time.monotonic())
        delivery_faults.record(_fault(B))

    threading.Thread(target=restart, daemon=True).start()
    started = time.monotonic()
    with pytest.raises(WorkerDeliveryError) as raised:
        capacity_agreement.agree_or_refuse(fleet, {}, deadline_s=20.0)
    ended = time.monotonic()
    assert ended - recorded_at[0] <= delivery_faults.POLL_S + 0.5
    assert ended - started < 2.0
    assert all(step <= delivery_faults.POLL_S for step in fleet.steps)
    text = str(raised.value)
    assert "h1 (tcp://10.0.0.2:26600)" in text and "undeliverable" in text
    assert "KSampler (DGX Monarch)" in text and "TROUBLESHOOTING.md #2" in text
    assert "run `dgxm restart`" not in text  # restarting the loops again clears nothing
    tag = parse_leading_refusal_tag(text)
    assert (tag.refusal_class.value, tag.guard) == ("C", capacity_agreement.GUARD)
    # A liveness fault, never a capacity verdict, and never a supervision death.
    assert not mesh_safety.is_stock_load_capacity_error(raised.value)
    assert "Supervision" not in text and "peer closed" not in text


def test_without_a_failure_the_wait_keeps_its_deadline_and_asks_once_more(monkeypatch):
    monkeypatch.setattr(delivery_faults, "REASK_S", 0.4)
    monkeypatch.setattr(delivery_faults, "POLL_S", 0.2)
    fleet = _Fleet({0: True, 1: False})
    started = time.monotonic()
    answers = capacity_agreement.collect(fleet, {}, 2, deadline_s=1.2)
    assert 1.1 <= time.monotonic() - started < 2.0
    assert answers[1].silent == "did not answer in time"
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(answers, 2)
    assert not isinstance(raised.value, WorkerDeliveryError)
    # Both ranks were sent to once, and the silent one was asked once more.
    assert fleet.sent == [0, 1, 1]


def test_a_failure_from_another_address_or_before_the_wait_does_not_end_it():
    delivery_faults.record(_fault(B), now=time.monotonic() - 5.0)
    delivery_faults.record(_fault("tcp://10.0.0.9:26600"))
    fleet = _Fleet({0: True, 1: False})
    answers = capacity_agreement.collect(fleet, {}, 2, deadline_s=1.0)
    assert answers[1].silent == "did not answer in time"


def test_a_late_failure_from_a_retired_mesh_does_not_end_a_healthy_wait():
    """Review finding on 7bcb372: a straggler from a mesh Reset retired names
    the address the live mesh shares. Rank 1 answers at 1.2 s, after a poll
    has timed out. The mutation that drops the mesh filter refuses here."""
    fleet = _Fleet({0: True, 1: 1.2})

    def straggler() -> None:
        time.sleep(0.2)
        delivery_faults.record(_fault(B, "dgxm_worker<RetiredByReset>"))

    threading.Thread(target=straggler, daemon=True).start()
    capacity_agreement.agree_or_refuse(fleet, {}, deadline_s=10.0)
    assert len(fleet.steps) >= 3  # it waited through polls that timed out


def test_a_mesh_whose_name_cannot_be_read_keeps_the_whole_deadline(monkeypatch):
    monkeypatch.setattr(delivery_faults, "REASK_S", 0.3)
    monkeypatch.setattr(delivery_faults, "POLL_S", 0.2)
    delivery_faults.record(_fault(B))
    fleet = _Fleet({0: True, 1: False}, name=None)
    started = time.monotonic()
    answers = capacity_agreement.collect(fleet, {}, 2, deadline_s=1.0)
    assert time.monotonic() - started >= 0.9
    assert answers[1].silent == "did not answer in time"
    # One wait for the whole deadline, as before #47: no polls, no second request.
    assert fleet.sent == [0, 1] and len(fleet.steps) == 2


def test_a_failure_that_beats_the_last_send_still_counts():
    """The wait counts from before the first send, so a failure that comes back
    while a later rank's request is still going out is not dropped."""
    fleet = _Fleet({0: True, 1: False})
    send = fleet._slice

    def slow_send(**coords):
        if coords["hosts"] == 1:
            delivery_faults.record(_fault(A))  # rank 0's report, mid-send
        return send(**coords)

    fleet.workers.slice = slow_send
    fleet.answers[0] = False
    with pytest.raises(WorkerDeliveryError, match="h0"):
        capacity_agreement.collect(fleet, {}, 2, deadline_s=5.0)


def test_a_local_fleet_waits_once_for_its_whole_deadline():
    fleet = _Fleet({0: True, 1: True}, addresses=())
    capacity_agreement.collect(fleet, {}, 2, deadline_s=7.0)
    assert fleet.steps[0] == pytest.approx(7.0, abs=0.1)
    assert len(fleet.steps) == 2


# ------------------------------------------------------------ the move check


def _readings(ids: dict[str, str]) -> dict[str, dict[str, str]]:
    return {address: {"gen": value[:12], "proc_id": value} for address, value in ids.items()}


def _config():
    from dgx_monarch.config import ClusterConfig, HostConfig

    return ClusterConfig(
        hosts=(HostConfig(name="h1", address=A), HostConfig(name="h2", address=B)),
        client_bind="tcp://10.0.0.9:0", auto_heal=False,
        transport_security="trusted_fabric", source=SOURCE)


@pytest.mark.parametrize(("now", "verdict", "moved", "kept"), [
    (NEW, restart_reattach.ALL, (A, B), ()),
    ({A: NEW[A], B: OLD[B]}, restart_reattach.PARTIAL, (A,), (B,)),
    (OLD, restart_reattach.NONE, (), (A, B)),
], ids=["all_moved", "one_moved", "none_moved"])
def test_the_move_check_compares_each_loop_with_the_id_this_driver_dialed(
        monkeypatch, now, verdict, moved, kept):
    monkeypatch.setattr(mesh_attach, "_LAST_DIALED", dict(OLD))
    monkeypatch.setattr(listener_generation, "fleet", lambda _config, **_k: _readings(now))
    moves = restart_reattach.read_moves(_config())
    assert (moves.verdict, moves.moved, moves.kept) == (verdict, moved, kept)


def test_an_unreadable_id_or_a_missing_dial_record_is_unknown(monkeypatch):
    monkeypatch.setattr(mesh_attach, "_LAST_DIALED", {A: OLD[A]})
    readings = _readings(NEW)
    readings[A] = {"gen": "unknown", "error": "ssh failed"}
    monkeypatch.setattr(listener_generation, "fleet", lambda _config, **_k: readings)
    moves = restart_reattach.read_moves(_config())
    assert moves.verdict == restart_reattach.UNKNOWN
    assert "ssh failed" in moves.unknown[0] and "no dialed id" in moves.unknown[1]

    def broken(_config, **_kwargs):
        raise OSError("no route")

    monkeypatch.setattr(listener_generation, "fleet", broken)
    assert restart_reattach.read_moves(_config()).verdict == restart_reattach.UNKNOWN


# ----------------------------------------------- the teardown and re-attach


class _Procs:
    """An old fleet's procs: their stop is never acknowledged."""

    def __init__(self) -> None:
        self.stops: list[float] = []

    def stop(self, _reason):
        def get(timeout):
            self.stops.append(timeout)
            raise TimeoutError("detach acknowledgement timed out")

        return SimpleNamespace(get=get)


@pytest.fixture
def driver(isolated_environ, monkeypatch):
    """A driver process with two configured Worker loops and no Monarch under it."""
    import monarch.actor as monarch_actor

    from dgx_monarch import monarch_launch

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

    config = _config()
    monkeypatch.setattr(monarch_launch, "prepare_driver", lambda _config: None)
    monkeypatch.setattr(mesh_mod, "_detect_comfy_dir", lambda _value: "/fake/comfy")
    monkeypatch.setattr(mesh_mod, "find_config_path", lambda path: path or None)
    monkeypatch.setattr(mesh_mod, "load_cluster_config", lambda _path: config)
    monkeypatch.setattr(mesh_helpers, "start_client_lease", lambda: True)
    monkeypatch.setattr(monarch_actor, "enable_transport", lambda *_a, **_k: None)

    state = SimpleNamespace(ids=dict(OLD), dials=[], primed=[], procs=[],
                            liveness=mesh_liveness.GONE, lines=[], events=[])

    def spawn(config, _use_cluster, _gph, attach, _bootstrap, handoff):
        procs = _Procs()
        state.procs.append(procs)
        return attach(config, handoff), procs, object(), True

    def attach_once(dial):
        state.dials.append(list(dial))
        return SimpleNamespace(name=f"hosts_{len(state.dials)}")

    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", spawn)
    monkeypatch.setattr(listener_generation, "fleet",
                        lambda _config, **_k: _readings(state.ids))
    monkeypatch.setattr(mesh_mod, "_attach_once", attach_once)
    monkeypatch.setattr(worker_proc_id, "prime",
                        lambda addresses: state.primed.append(list(addresses)) or [])
    monkeypatch.setattr(
        mesh_liveness, "probe_fleet",
        lambda handle, **_k: mesh_liveness.LivenessEvidence(
            state.liveness, ((A, state.liveness), (B, state.liveness)), time.monotonic()))
    for module in (mesh_mod, restart_reattach):
        for level in ("info", "warning"):
            monkeypatch.setattr(module.log, level, lambda template, *args, **_k: state.lines.append(
                template % args if args else template))
    monkeypatch.setattr(telemetry, "emit", lambda kind, **fields: state.events.append(
        {"kind": kind, **fields}))
    return state


def _refused() -> WorkerDeliveryError:
    return WorkerDeliveryError("[dgxm:C guard=cross_rank_capacity waivable=0] undeliverable")


def _trace(state, event: str) -> list[dict[str, str]]:
    return [fields for fields in map(attach_trace.parse, state.lines)
            if fields and fields.get("event") == event]


def test_every_loop_moved_retires_the_stale_mesh_and_attaches_by_the_new_ids(driver):
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = dict(NEW)
    outcome = restart_reattach.heal(handle, _refused())

    assert outcome.handle is not handle and outcome.teardown == "liveness_release"
    assert handle.teardown_complete is True and mesh_mod._TRANSPORT_POISON is None
    # The stale fleet's stop was tried with the short bound, never the 60 s one.
    assert driver.procs[0].stops == [restart_reattach.STOP_BOUND_S]
    assert driver.dials[1:] == [[worker_proc_id.proc_addr(A, NEW[A]),
                                 worker_proc_id.proc_addr(B, NEW[B])]]
    assert driver.primed == [[A, B]]
    line = _trace(driver, "restart_reattach")[-1]
    assert (line["outcome"], line["verdict"], line["teardown"]) == (
        "attached", "all", "liveness_release")
    assert line["seq"] == _trace(driver, "open")[-1]["seq"]
    assert float(line["wall_s"]) == pytest.approx(outcome.wall_s, abs=0.01)
    assert any(text.startswith("restart_reattach outcome=attached") for text in driver.lines)
    assert driver.events[-1]["kind"] == "restart_reattach"
    assert driver.events[-1]["outcome"] == "attached"
    # The replacement is the cached mesh the render's retry finds.
    assert mesh_mod.ensure_live(handle) is outcome.handle


def test_one_loop_moved_refuses_fast_and_leaves_the_mesh_alone(driver):
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = {A: NEW[A], B: OLD[B]}
    started = time.monotonic()
    with pytest.raises(WorkerDeliveryError) as raised:
        restart_reattach.heal(handle, _refused())
    assert time.monotonic() - started < 1.0
    assert "dgxm restart" in str(raised.value) and "h2 (tcp://10.0.0.2:26600)" in str(raised.value)
    assert handle.defunct is False and driver.procs[0].stops == []
    assert len(driver.dials) == 1
    assert _trace(driver, "restart_reattach")[-1]["reason"] == "partial"


def test_no_loop_moved_refuses_with_the_reset_remedy(driver):
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    with pytest.raises(WorkerDeliveryError, match="Reset attached mesh"):
        restart_reattach.heal(handle, _refused())
    assert handle.defunct is False and len(driver.dials) == 1


def test_old_processes_not_proven_gone_refuse_and_attach_nothing(driver):
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = dict(NEW)
    driver.liveness = mesh_liveness.ALIVE
    with pytest.raises(WorkerDeliveryError, match="liveness check answered alive"):
        restart_reattach.heal(handle, _refused())
    assert handle.teardown_complete is False and handle.replacement_blocked
    assert mesh_teardown.block_cause(handle) == mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH
    assert len(driver.dials) == 1
    assert _trace(driver, "restart_reattach")[-1]["reason"] == "not_retired"


@pytest.mark.parametrize("owner", ["deferred", "deliberate"])
def test_a_death_another_path_owns_refuses_without_a_stop(driver, monkeypatch, owner):
    """A held sample lease defers the death; a teardown already in flight owns
    it. Either way the detour stops nothing and attaches nothing."""
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = dict(NEW)
    if owner == "deferred":
        handle.sample_leases = {1: 1}
    else:
        real = mesh_helpers.publish_supervision_failure_state
        monkeypatch.setattr(
            mesh_helpers, "publish_supervision_failure_state",
            lambda live, exc, text: (real(live, exc, text)[0], True, False, False))
    with pytest.raises(WorkerDeliveryError, match="could not retire the stale mesh"):
        restart_reattach.heal(handle, _refused())
    assert driver.procs[0].stops == [] and len(driver.dials) == 1
    assert _trace(driver, "restart_reattach")[-1]["reason"] == "not_retired"


def test_a_failed_replacement_attach_propagates_its_own_error(driver, monkeypatch):
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = dict(NEW)

    def stalls(_dial):
        raise TimeoutError("config push timed out")

    monkeypatch.setattr(mesh_mod, "_attach_once", stalls)
    with pytest.raises(mesh_mod.MeshAttachError, match="released"):
        restart_reattach.heal(handle, _refused())
    line = _trace(driver, "restart_reattach")[-1]
    assert (line["outcome"], line["reason"]) == ("failed", "MeshAttachError")
    # The long-outage fallback stays the one documented: this session is poisoned.
    assert mesh_mod._TRANSPORT_POISON is not None


# ---------------------------------------------------------- the render path


def _render_rig(monkeypatch, *, heal=None, inherited=False):
    from dgx_monarch.nodes import recycle_drain, render_submit

    state = SimpleNamespace(calls=0, facts=[], healed=[], drops=0)
    refused_handle, fresh = object(), object()

    def guarded(model, request, latent, cfg, steps, seq, depth, progress, handoff, guard,
                *, adoption_context=None):
        state.calls += 1
        state.facts.append(render_submit._REATTACH_FACTS.get())
        if state.calls == 1:
            guard.handle = refused_handle
            raise _refused()
        return "pending"

    def fake_heal(handle, exc):
        state.healed.append((handle, exc))
        return restart_reattach.Reattached(fresh, 4.25, (A, B), "liveness_release")

    monkeypatch.setattr(render_submit, "_submit_render_guarded", guarded)
    monkeypatch.setattr(restart_reattach, "heal", heal or fake_heal)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos",
                        lambda: setattr(state, "drops", state.drops + 1))
    monkeypatch.setattr(render_submit, "active_render_session",
                        lambda: object() if inherited else None)
    state.refused_handle = refused_handle
    return render_submit, state


def test_a_render_that_meets_restarted_workers_attaches_again_and_dispatches_once_more(
        monkeypatch):
    """The mutation that drops the in-render re-attach makes this render raise."""
    render_submit, state = _render_rig(monkeypatch)
    assert render_submit.submit_render(None, {}, {}, None, 1) == "pending"
    assert state.calls == 2
    assert [handle for handle, _exc in state.healed] == [state.refused_handle]
    assert state.drops == 1
    # Only the retry carries the record of the re-attach, and only during it.
    assert state.facts == [None, {"reattach": "restart_reattach", "reattach_s": 4.25}]
    assert render_submit._REATTACH_FACTS.get() is None


def test_a_pipelined_or_gate_render_keeps_the_fast_refusal(monkeypatch):
    render_submit, state = _render_rig(monkeypatch, inherited=True)
    with pytest.raises(WorkerDeliveryError):
        render_submit.submit_render(None, {}, {}, None, 1)
    assert state.calls == 1 and state.healed == []


def test_a_detour_that_cannot_heal_raises_its_refusal_without_a_retry(monkeypatch):
    def refuses(_handle, exc):
        raise WorkerDeliveryError("[dgxm:C guard=cross_rank_capacity waivable=0] partial") from exc

    render_submit, state = _render_rig(monkeypatch, heal=refuses)
    with pytest.raises(WorkerDeliveryError, match="partial"):
        render_submit.submit_render(None, {}, {}, None, 1)
    assert state.calls == 1 and state.drops == 0


@pytest.fixture
def loader_rig(monkeypatch):
    """The UNET Loader's eager load on a fleet whose loops all restarted."""
    import contextlib

    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import (
        abandoned_heal,
        loader_preflight,
        loaders,
        recycle_drain,
        render_preflight,
        render_session,
    )
    from dgx_monarch.nodes.common import MeshSpec

    events: list[tuple] = []
    rig = SimpleNamespace(events=events, healed=False, heal=None)

    def handle(name):
        return SimpleNamespace(name=name, world=2, call_all=lambda endpoint, *_a, **_k: (
            events.append((endpoint, name)) or [{"host": name, "transition": "load"}]))

    rig.stale, rig.fresh = handle("stale"), handle("fresh")

    def agree_load(live, *_args):
        events.append(("agree", live.name))
        if live is rig.stale:
            raise _refused()

    def heal(live, exc, *, respawn=None):
        events.append(("heal", live.name))
        if rig.heal is not None:
            rig.heal(live, exc)
        rig.healed = True
        return restart_reattach.Reattached(respawn(live), 3.5, (A, B), "liveness_release")

    monkeypatch.setattr(loaders, "ensure_live", lambda live, **_k: (
        events.append(("ensure_live",)) or (rig.fresh if rig.healed else rig.stale)))
    monkeypatch.setattr(abandoned_heal, "heal_before_eager_setup", lambda live, **_k: live)
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda live, *_a, **_k: (
        events.append(("setup", live.name))))
    monkeypatch.setattr(render_preflight, "preflight_sol_sequence_parallel", lambda *_a: None)
    monkeypatch.setattr(render_session, "mutation_render_session",
                        lambda _live: contextlib.nullcontext())
    monkeypatch.setattr(recycle_drain, "drop_residency_memos",
                        lambda: events.append(("drop_memos",)))
    monkeypatch.setattr(loader_preflight, "preflight_loader_footprint",
                        lambda *_a, **_k: events.append(("price",)) or object())
    monkeypatch.setattr(loader_preflight, "record_rescue_row", lambda *_a, handle, **_k: (
        events.append(("record", handle.name))))
    monkeypatch.setattr(capacity_agreement, "agree_load", agree_load)
    monkeypatch.setattr(restart_reattach, "heal", heal)
    rig.load = lambda: loaders.DGXMonarchUNETLoader().load(
        MeshSpec(handle=rig.stale, topology_preset="uly2", attention="TORCH_FLASH",
                 sync_ulysses=True), "x.safetensors")
    return rig


def test_an_explicit_loader_attaches_again_and_loads_once_more(loader_rig):
    """Review finding on 7bcb372: the loader runs before the sampler, so a
    loader that only refused would refuse on every queue. The mutation that
    drops the loader's re-attach raises here."""
    loader_rig.load()
    assert loader_rig.events == [
        ("price",), ("ensure_live",), ("setup", "stale"), ("agree", "stale"),
        ("heal", "stale"), ("drop_memos",), ("price",), ("ensure_live",),
        ("setup", "fresh"), ("agree", "fresh"), ("load_model", "fresh"),
        ("record", "fresh")]


def test_a_loader_whose_detour_refuses_loads_nothing(loader_rig):
    def partial(_live, exc):
        raise WorkerDeliveryError("[dgxm:C guard=cross_rank_capacity waivable=0] partial") from exc

    loader_rig.heal = partial
    with pytest.raises(WorkerDeliveryError, match="partial"):
        loader_rig.load()
    assert loader_rig.events[-1] == ("heal", "stale")
    assert ("load_model", "stale") not in loader_rig.events


# ------------------------------------------------------------------ records


def test_the_finish_record_names_the_re_attach_and_its_wall():
    from dgx_monarch.nodes import render_submit

    progress = telemetry.RenderProgress()
    token = progress.start(1, {"model": "m", "reattach": restart_reattach.EVENT,
                               "reattach_s": 4.25}, token=object())
    progress.finish(token)
    last = progress.snapshot()["last"]
    assert (last["reattach"], last["reattach_s"]) == ("restart_reattach", 4.25)
    # An ordinary render carries both keys as None.
    assert set(render_submit._NO_REATTACH) <= set(telemetry._RENDER_FACTS)


def test_the_event_ring_publishes_the_re_attach_fields(monkeypatch):
    monkeypatch.setattr(telemetry, "_EVENTS", type(telemetry._EVENTS)(maxlen=256))
    telemetry.emit(restart_reattach.EVENT, outcome="attached", verdict="all",
                   moved=f"{A},{B}", teardown="liveness_release", wall_s=4.25)
    event = telemetry.events_tail()[-1]
    assert {key: event[key] for key in ("kind", "outcome", "moved", "wall_s")} == {
        "kind": "restart_reattach", "outcome": "attached", "moved": f"{A},{B}",
        "wall_s": 4.25}


def test_the_guard_literal_is_the_agreement_guard():
    assert restart_reattach.GUARD == capacity_agreement.GUARD
