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
    driver_footprint,
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


class _Future:
    """One rank's quote future: answers, raises, or blocks for the wait it is given."""

    def __init__(self, fleet: _Fleet, rank: int) -> None:
        self.fleet, self.rank = fleet, rank

    def get(self, timeout=None):
        fleet = self.fleet
        fleet.steps.append(timeout)
        answer = fleet.answers[self.rank]
        if isinstance(answer, tuple):  # (seconds, exception): fails that late
            due = fleet.started + answer[0] - time.monotonic()
            if due > timeout:
                time.sleep(timeout)
                raise TimeoutError("no answer yet")
            time.sleep(max(due, 0.0))
            answer = answer[1]
        if isinstance(answer, BaseException):
            raise answer
        if answer is True:
            return _row(self.rank)
        if answer is not False:
            due = fleet.started + float(answer) - time.monotonic()
            if due <= timeout:
                time.sleep(max(due, 0.0))
                return _row(self.rank)
        time.sleep(timeout)
        raise TimeoutError("no answer")


class _Fleet:
    """A two-host fleet whose futures answer, or block for the step they are given.

    A rank whose answer is a number answers after that many seconds. Its
    evicting wait records each call and reads the future as the real one does.
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
        self.evicting: list[tuple[int, float]] = []
        self.join_s = 0.0
        self.started = time.monotonic()
        self.workers = types.SimpleNamespace(
            extent=types.SimpleNamespace(labels=["hosts", "gpus"]), slice=self._slice,
            _name=_Name(name))

    def _slice(self, **coords):
        rank = coords["hosts"]
        return types.SimpleNamespace(capacity_quote=types.SimpleNamespace(
            call_one=lambda _request: self.sent.append(rank) or _Future(self, rank)))

    def _await_or_evict(self, future, timeout_s):
        self.evicting.append((future.rank, timeout_s))
        try:
            return future.get(timeout_s)
        except TimeoutError:
            raise
        except Exception:
            time.sleep(self.join_s)  # the post-death stop the real eviction joins
            raise


def _head_named(monkeypatch, name: str) -> None:
    """The driver runs on the configured host ``name``, read without DNS."""
    monkeypatch.setattr(driver_footprint, "host_is_local", lambda host: host == name)


def test_a_failure_recorded_during_the_wait_ends_it_within_one_poll(monkeypatch):
    """The mutation that drops the short circuit makes this wait run to its
    deadline and refuse as a silent rank instead."""
    _head_named(monkeypatch, "h0")
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
    assert "Worker actor on the sibling undeliverable" in text
    assert "10.0.0." not in text and "h1" not in text
    assert "KSampler (DGX Monarch)" in text and "TROUBLESHOOTING.md #2" in text
    assert "run `dgxm restart`" not in text  # restarting the loops again clears nothing
    tag = parse_leading_refusal_tag(text)
    assert (tag.refusal_class.value, tag.guard) == ("C", capacity_agreement.GUARD)
    # A liveness fault, never a capacity verdict, and never a supervision death.
    assert not mesh_safety.is_stock_load_capacity_error(raised.value)
    assert "Supervision" not in text and "peer closed" not in text


def test_without_a_failure_the_wait_keeps_its_deadline_and_asks_each_rank_once(monkeypatch):
    """A rank that never answers is asked once and waited on for the whole
    deadline. The wait this replaces sent a silent rank a second request 2 s
    in, so with a 2.6 s deadline that request shows here as a third send."""
    monkeypatch.setattr(delivery_faults, "POLL_S", 0.2)
    fleet = _Fleet({0: True, 1: False})
    started = time.monotonic()
    answers = capacity_agreement.collect(fleet, {}, 2, deadline_s=2.6)
    assert 2.5 <= time.monotonic() - started < 3.5
    assert answers[1].silent == "did not answer in time"
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(answers, 2)
    assert not isinstance(raised.value, WorkerDeliveryError)
    assert fleet.sent == [0, 1]


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
    monkeypatch.setattr(delivery_faults, "POLL_S", 0.2)
    delivery_faults.record(_fault(B))
    fleet = _Fleet({0: True, 1: False}, name=None)
    started = time.monotonic()
    answers = capacity_agreement.collect(fleet, {}, 2, deadline_s=1.0)
    assert time.monotonic() - started >= 0.9
    assert answers[1].silent == "did not answer in time"
    # One wait for the whole deadline, as before #47: no polls, no second request.
    assert fleet.sent == [0, 1] and len(fleet.steps) == 2


def test_a_failure_that_beats_the_last_send_still_counts(monkeypatch):
    """The wait counts from before the first send, so a failure that comes back
    while a later rank's request is still going out is not dropped."""
    _head_named(monkeypatch, "h0")
    fleet = _Fleet({0: True, 1: False})
    send = fleet._slice

    def slow_send(**coords):
        if coords["hosts"] == 1:
            delivery_faults.record(_fault(A))  # rank 0's report, mid-send
        return send(**coords)

    fleet.workers.slice = slow_send
    fleet.answers[0] = False
    with pytest.raises(WorkerDeliveryError, match="Worker actor on the head undeliverable"):
        capacity_agreement.collect(fleet, {}, 2, deadline_s=5.0)


def test_a_local_fleet_waits_once_for_its_whole_deadline():
    fleet = _Fleet({0: True, 1: True}, addresses=())
    capacity_agreement.collect(fleet, {}, 2, deadline_s=7.0)
    assert fleet.steps[0] == pytest.approx(7.0, abs=0.1)
    assert len(fleet.steps) == 2


def _supervision_error():
    from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError

    return SupervisionError(
        "monitor for actor dgxm_worker<x>.anon-0<y> stopped before report")


def _reads(monkeypatch, moves) -> list[tuple[object, float | None]]:
    """Stand in for the loops' id reading; record each call and its bound."""
    reads: list[tuple[object, float | None]] = []

    def read(config, *, timeout=None):
        reads.append((config, timeout))
        return moves

    monkeypatch.setattr(restart_reattach, "read_moves", read)
    return reads


def test_a_supervision_error_after_a_restart_heals_before_any_eviction(monkeypatch):
    """Monarch's supervision can give up on the old actors before a delivery
    failure is recorded; the restart runs traced it about 2 s into the price.
    The handle's own wait would then evict the fleet and join its post-death
    stop, about 24 s on loopback, before anything else ran. The price reads the
    loops' ids first, with the short bound, and ends with the error the
    render's re-attach catches, carrying that reading. On the commit before
    this check, the fleet was evicted first and the price refused."""
    _head_named(monkeypatch, "h0")
    moves = restart_reattach.Moves(moved=(A, B))
    reads = _reads(monkeypatch, moves)
    fleet = _Fleet({0: _supervision_error(), 1: _supervision_error()})
    fleet.join_s = 5.0
    started = time.monotonic()
    with pytest.raises(WorkerDeliveryError) as raised:
        capacity_agreement.agree_or_refuse(fleet, {}, deadline_s=20.0)
    assert time.monotonic() - started < 1.0
    assert reads == [(fleet.config, restart_reattach.PRECHECK_PROBE_S)]
    assert fleet.evicting == []  # no eviction and no join: the heal's own stops follow
    assert raised.value.moves is moves
    text = str(raised.value)
    assert "failed (SupervisionError)" in text and "every Worker loop serves a new proc id" in text
    assert "on the head and the sibling" in text and "10.0.0." not in text
    assert "KSampler (DGX Monarch)" in text and "TROUBLESHOOTING.md #2" in text
    tag = parse_leading_refusal_tag(text)
    assert (tag.refusal_class.value, tag.guard) == ("C", capacity_agreement.GUARD)
    assert not mesh_safety.is_stock_load_capacity_error(raised.value)


def test_the_heal_is_reached_when_the_deadline_is_nearly_spent(monkeypatch):
    """A SupervisionError that comes 0.9 s into a 1 s price, where the
    eviction's join would last longer than the deadline. Checked before the
    eviction, it still ends the price for the heal; evicted first, the later
    rank was called silent unread and the heal was never reached."""
    moves = restart_reattach.Moves(moved=(A, B))
    reads = _reads(monkeypatch, moves)
    fleet = _Fleet({0: (0.9, _supervision_error()), 1: _supervision_error()})
    fleet.join_s = 1.5
    with pytest.raises(WorkerDeliveryError) as raised:
        capacity_agreement.collect(fleet, {}, 2, deadline_s=1.0)
    assert raised.value.moves is moves and len(reads) == 1 and fleet.evicting == []


@pytest.mark.parametrize("verdict", ["partial", "none", "unknown"])
def test_a_supervision_error_without_a_full_restart_evicts_and_refuses_as_before(
        monkeypatch, verdict):
    """One reading for the whole price; then each failure goes through the
    handle's own evicting wait, read again at once, and the price refuses as a
    fleet that did not answer."""
    moves = {"partial": restart_reattach.Moves(moved=(A,), kept=(B,)),
             "none": restart_reattach.Moves(kept=(A, B)),
             "unknown": restart_reattach.Moves(unknown=("the head (ssh failed)",))}[verdict]
    reads = _reads(monkeypatch, moves)
    fleet = _Fleet({0: _supervision_error(), 1: _supervision_error()})
    answers = capacity_agreement.collect(fleet, {}, 2, deadline_s=5.0)
    assert [answer.silent for answer in answers] == ["failed to answer (SupervisionError)"] * 2
    assert len(reads) == 1 and fleet.evicting == [(0, 0.0), (1, 0.0)]
    with pytest.raises(FleetQuoteUnavailableError) as raised:
        decide(answers, 2)
    assert not isinstance(raised.value, WorkerDeliveryError)


def test_other_failures_and_timeouts_read_no_ids_and_evict_as_before(monkeypatch):
    def unexpected(_config, **_kwargs):
        raise AssertionError("the move check ran")

    monkeypatch.setattr(restart_reattach, "read_moves", unexpected)
    fleet = _Fleet({0: RuntimeError("endpoint raised"), 1: False})
    answers = capacity_agreement.collect(fleet, {}, 2, deadline_s=1.0)
    assert [answer.silent for answer in answers] == [
        "failed to answer (RuntimeError)", "did not answer in time"]
    # The endpoint failure went through the evicting wait; the timeouts did not.
    assert fleet.evicting == [(0, 0.0)]
    # A local fleet has no loops to read and waits through the handle as before.
    local = _Fleet({0: _supervision_error(), 1: _supervision_error()}, addresses=())
    capacity_agreement.collect(local, {}, 2, deadline_s=1.0)
    assert [rank for rank, _step in local.evicting] == [0, 1]


# ------------------------------------------------------------ the move check


def _readings(ids: dict[str, str]) -> dict[str, dict[str, str]]:
    import monarch.actor  # noqa: F401  # the driver imports Monarch before any attach

    from dgx_monarch.monarch_capabilities import loaded_version

    return {address: {"gen": value[:12], "proc_id": value, "monarch": loaded_version()}
            for address, value in ids.items()}


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
    _head_named(monkeypatch, "h1")
    monkeypatch.setattr(mesh_attach, "_LAST_DIALED", {A: OLD[A]})
    readings = _readings(NEW)
    readings[A] = {"gen": "unknown", "error": "ssh failed"}
    monkeypatch.setattr(listener_generation, "fleet", lambda _config, **_k: readings)
    moves = restart_reattach.read_moves(_config())
    assert moves.verdict == restart_reattach.UNKNOWN
    assert "ssh failed" in moves.unknown[0] and "no dialed id" in moves.unknown[1]
    # The refusal reads these: each box by its plain name, never its address.
    assert moves.unknown[0].startswith("the head (") and "10.0.0." not in moves.unknown[0]
    assert moves.unknown[1] == "the sibling (this driver holds no dialed id for it)"

    def broken(_config, **_kwargs):
        raise OSError("no route")

    monkeypatch.setattr(listener_generation, "fleet", broken)
    probed = restart_reattach.read_moves(_config())
    assert probed.verdict == restart_reattach.UNKNOWN
    assert probed.unknown == ("the head (the probe raised OSError)",
                              "the sibling (the probe raised OSError)")


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


def test_the_detour_reuses_the_reading_the_supervision_check_took(driver, monkeypatch):
    """refuse_if_restarted read every loop's id moments before; the detour does
    not run the probe a second time on every host."""
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = dict(NEW)
    refused = _refused()
    refused.moves = restart_reattach.read_moves(_config())

    def read_again(_config, **_kwargs):
        raise AssertionError("the detour read the ids again")

    monkeypatch.setattr(restart_reattach, "read_moves", read_again)
    outcome = restart_reattach.heal(handle, refused)
    assert outcome.handle is not handle and outcome.moved == (A, B)
    assert _trace(driver, "restart_reattach")[-1]["outcome"] == "attached"


class _HeldProcs:
    """Procs whose stop answers once ``release`` is set; it is set from the start
    for a stop that is done, and later for a stop another thread is still in."""

    def __init__(self, release: threading.Event) -> None:
        self.release, self.stops, self.entered = release, [], threading.Event()

    def stop(self, _reason):
        def get(timeout):
            self.stops.append(timeout)
            self.entered.set()
            if not self.release.wait(timeout):
                raise TimeoutError("detach acknowledgement timed out")
            return None

        return SimpleNamespace(get=get)


def _restarted(driver) -> WorkerDeliveryError:
    """The error the price's restart check raises once every loop restarted."""
    driver.ids = dict(NEW)
    refused = _refused()
    refused.moves = restart_reattach.read_moves(_config())
    assert refused.moves.verdict == restart_reattach.ALL
    return refused


@pytest.mark.parametrize("when", ["before_the_dial", "during_attempt_1"])
def test_an_attach_the_exit_stands_down_leaves_the_transport_unpoisoned(driver, monkeypatch, when):
    """Through the real fleet spawn and get_mesh: an attach the driver's exit
    reaches ends in the exit error and leaves the transport unpoisoned, as a
    refusal before any dial does. Before, the spawn's attach mark made get_mesh
    poison it ("cluster attach result handoff interrupted"). With the flag
    cleared, the next get_mesh attaches."""
    from dgx_monarch import driver_exit, mesh_factory

    monkeypatch.setattr(driver_exit, "_EXITING", threading.Event())
    stand_in, real_once = mesh_mod.spawn_worker_fleet, mesh_mod._attach_once
    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", mesh_factory.spawn_worker_fleet)
    if when == "before_the_dial":
        driver_exit._EXITING.set()
    else:
        def attach_once(dial):
            driver.dials.append(list(dial))
            driver_exit._EXITING.set()  # the exit hook ran while this attempt waited
            raise RuntimeError("attach failed: config push failed during attach")

        monkeypatch.setattr(mesh_mod, "_attach_once", attach_once)
    with pytest.raises(mesh_mod.MeshAttachError, match="the driver is exiting"):
        mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    assert mesh_mod._TRANSPORT_POISON is None
    assert len(driver.dials) == (0 if when == "before_the_dial" else 1)
    driver_exit._EXITING.clear()
    monkeypatch.setattr(mesh_mod, "spawn_worker_fleet", stand_in)
    monkeypatch.setattr(mesh_mod, "_attach_once", real_once)
    assert mesh_mod.get_mesh(mode="cluster", config_path=SOURCE) is not None
    assert mesh_mod._TRANSPORT_POISON is None


def test_a_heal_whose_stop_is_confirmed_after_an_earlier_release_keeps_its_retry(
        driver, monkeypatch):
    """A first restart ends in a liveness release, which arms the release
    note, and its replacement attaches. A second restart in the same window
    has its stop confirmed, which arms nothing. Its replacement attach fails
    once and must get its in-process retry: the note was spent when the first
    replacement attached. Before that, the stale note made it fail fast."""
    newer = {A: "aaaaaaaaaaaa0003", B: "bbbbbbbbbbbb0003"}
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    first = restart_reattach.heal(handle, _restarted(driver))
    assert first.teardown == "liveness_release"
    done = threading.Event()
    done.set()
    first.handle.procs = _HeldProcs(done)
    driver.ids = dict(newer)
    refused = _refused()
    refused.moves = restart_reattach.read_moves(_config())
    real_attach, failed = mesh_mod._attach_once, []

    def flaky(dial):
        if not failed:
            failed.append(list(dial))
            raise RuntimeError("blip")
        return real_attach(dial)

    monkeypatch.setattr(mesh_mod, "_attach_once", flaky)
    second = restart_reattach.heal(first.handle, refused)
    assert second.teardown == "stop_confirmed" and second.handle is not first.handle
    retried = [worker_proc_id.proc_addr(A, newer[A]), worker_proc_id.proc_addr(B, newer[B])]
    assert failed == [retried] and driver.dials[-1] == retried


# Every logger the detour and its attach write through, named here rather than
# read from driver_exit, so a logger dropped from its quiet list shows.
_EXIT_PATH_LOGGERS = ("dgx_monarch.mesh", "dgx_monarch.mesh_liveness", "dgx_monarch.restart_reattach",
                      "dgx_monarch.mesh_runtime", "dgx_monarch.capacity_agreement")


class _ClientStoppedProcs:
    """Procs whose stop fails as Monarch's exit step leaves it; ``exit_first``
    sets the driver's exit flag as the stop runs, as the exit hook does then."""

    def __init__(self, exit_first: bool) -> None:
        self.exit_first, self.stops = exit_first, []

    def stop(self, _reason):
        from dgx_monarch import driver_exit

        def get(timeout):
            self.stops.append(timeout)
            if self.exit_first:
                driver_exit._EXITING.set()
            raise ValueError("error stopping mesh: client.local<C1>@tcp://10.0.0.9:1: owner "
                             "terminated: stopped: client stopped")

        return SimpleNamespace(get=get)


@pytest.mark.parametrize("path", ["heal", "pinned"])
@pytest.mark.parametrize("exiting", [True, False], ids=["exiting", "not_exiting"])
def test_an_exit_inside_the_retires_stop_leaves_the_renders_thread_quiet(
        driver, monkeypatch, path, exiting):
    """The driver begins to exit while a render's heal, or a pinned render's
    retire, is inside its own stop, and Monarch's exit step ends that stop with
    "client stopped". From then on the render's thread logs no WARNING or
    ERROR, probes no host, attaches nothing and sends no telemetry: it raises
    the exit refusal. Not exiting, the same failed stop logs as it always did,
    and the detour refuses as one whose stop has no known outcome."""
    import logging

    from dgx_monarch import driver_exit

    monkeypatch.setattr(driver_exit, "_EXITING", threading.Event())  # a fresh flag, undone after
    for logger in (mesh_mod.log, restart_reattach.log):  # the real methods, through the filters
        for level in ("info", "warning"):
            monkeypatch.delattr(logger, level, raising=False)
    kept: list[logging.LogRecord] = []
    keep = logging.Handler()
    keep.emit = kept.append  # type: ignore[method-assign]
    loggers = [logging.getLogger(name) for name in _EXIT_PATH_LOGGERS]
    probes: list[int] = []
    real_probe = mesh_liveness.probe_fleet
    monkeypatch.setattr(mesh_liveness, "probe_fleet",
                        lambda handle, **kw: probes.append(1) or real_probe(handle, **kw))
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    handle.procs = _ClientStoppedProcs(exit_first=exiting)
    refused = _restarted(driver)
    run = restart_reattach.heal if path == "heal" else restart_reattach.refuse_pinned
    for logger in loggers:
        logger.addHandler(keep)
    try:
        with pytest.raises(WorkerDeliveryError) as raised:
            run(handle, refused)
    finally:
        for logger in loggers:
            logger.removeHandler(keep)
    mine = [record for record in kept if record.thread == threading.get_ident()]
    loud = [record.getMessage() for record in mine if record.levelno >= logging.WARNING]
    assert len(driver.dials) == 1 and driver_exit._QUIET_THREADS == set()  # the mark ends with it
    if exiting:
        assert "the driver is exiting" in str(raised.value) and loud == [], loud
        assert probes == [] and driver.events == []
    else:
        assert "could not retire the stale mesh" in str(raised.value)
        assert any("proc stop on detach failed" in text for text in loud), loud
        assert any("restart_reattach outcome=refused" in text for text in loud), loud
        assert driver.events


@pytest.mark.parametrize(("path", "stage"), [
    ("heal", "liveness_probe"), ("pinned", "liveness_probe"), ("heal", "liveness_alive"),
    ("pinned", "liveness_alive"), ("heal", "attach")])
def test_an_exit_after_the_stop_ends_the_detour_with_the_exit_refusal(
        driver, monkeypatch, path, stage):
    """The driver begins to exit during the liveness probe (which answers gone,
    or alive as a host whose processes Monarch's exit step is still stopping)
    or during the replacement attach (which the exit then cuts short). The
    render's thread logs no WARNING or ERROR, dials nothing new, and its error
    is always the exit refusal: not the failed attach's own advice, not a
    pinned render's "queue again", and not the retire's "could not retire",
    which a probe the exit overtook raised before."""
    import logging

    from dgx_monarch import driver_exit

    monkeypatch.setattr(driver_exit, "_EXITING", threading.Event())
    for logger in (mesh_mod.log, restart_reattach.log):
        for level in ("info", "warning"):
            monkeypatch.delattr(logger, level, raising=False)
    kept: list[logging.LogRecord] = []
    keep = logging.Handler()
    keep.emit = kept.append  # type: ignore[method-assign]
    loggers = [logging.getLogger(name) for name in _EXIT_PATH_LOGGERS]
    real_probe, real_live, attaches = mesh_liveness.probe_fleet, mesh_mod.ensure_live, []
    if stage == "liveness_alive":
        driver.liveness = mesh_liveness.ALIVE
    if stage.startswith("liveness"):
        def probe(handle, **kwargs):
            kept.clear()  # what the stop logged came before the exit
            driver_exit._EXITING.set()  # the exit hook ran while the probe read the hosts
            return real_probe(handle, **kwargs)

        monkeypatch.setattr(mesh_liveness, "probe_fleet", probe)
        monkeypatch.setattr(mesh_mod, "ensure_live",
                            lambda *a, **k: attaches.append(1) or real_live(*a, **k))
    else:
        def ensure_live(_handle):
            kept.clear()
            driver_exit._EXITING.set()
            raise RuntimeError("Handle producer ended without publishing a result")

        monkeypatch.setattr(mesh_mod, "ensure_live", ensure_live)
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    refused = _restarted(driver)
    run = restart_reattach.heal if path == "heal" else restart_reattach.refuse_pinned
    for logger in loggers:
        logger.addHandler(keep)
    try:
        with pytest.raises(WorkerDeliveryError) as raised:
            run(handle, refused)
    finally:
        for logger in loggers:
            logger.removeHandler(keep)
    loud = [record.getMessage() for record in kept
            if record.thread == threading.get_ident() and record.levelno >= logging.WARNING]
    text = str(raised.value)
    assert "the driver is exiting, so this render stops here" in text
    assert "queue again" not in text and "could not retire" not in text
    assert loud == [], loud  # nothing above INFO after the exit began
    assert len(driver.dials) == 1 and attaches == []  # no attach was even begun


def _exit_rig(driver, monkeypatch):
    """A fresh exit flag, the real log methods through their filters, and a count of
    the id readings. Each reading is one ``listener_generation.fleet`` call, which on
    the pair asks every host over ssh in turn. ``exit_on_read`` sets the flag inside
    the reading, as the exit hook would while it asks the hosts; ``readings`` stands
    in for what the hosts answer. ``listen`` starts the count and keeps every record
    on the detour's loggers from then on; ``loud`` is this thread's WARNING and ERROR
    records."""
    import logging

    from dgx_monarch import driver_exit

    monkeypatch.setattr(driver_exit, "_EXITING", threading.Event())  # a fresh flag, undone after
    for logger in (mesh_mod.log, restart_reattach.log):  # the real methods, through the filters
        for level in ("info", "warning"):
            monkeypatch.delattr(logger, level, raising=False)
    rig = SimpleNamespace(kept=[], reads=0, exit_on_read=False, readings=None,
                          flag=driver_exit._EXITING)

    def reading(_config, **_kwargs):
        rig.reads += 1
        if rig.exit_on_read:
            rig.flag.set()
        return rig.readings if rig.readings is not None else _readings(driver.ids)

    def listen() -> None:
        rig.reads = 0  # the first attach's own reading came before
        keep = logging.Handler()
        keep.emit = rig.kept.append  # type: ignore[method-assign]
        for name in _EXIT_PATH_LOGGERS:
            logger = logging.getLogger(name)
            monkeypatch.setattr(logger, "handlers", [*logger.handlers, keep])

    def loud() -> list[str]:
        return [record.getMessage() for record in rig.kept
                if record.thread == threading.get_ident() and record.levelno >= logging.WARNING]

    monkeypatch.setattr(listener_generation, "fleet", reading)
    rig.listen, rig.loud = listen, loud
    return rig


def _first_reading_site(site: str, handle, error, fleet):
    """Run the price's restart check, the heal or a pinned retire, each of which reads
    the loops' ids first: the check after a SupervisionError, the other two on a
    delivery failure, which carries no reading."""
    if site == "price_check":
        return capacity_agreement.collect(fleet, {}, 2, deadline_s=5.0)
    if site == "heal":
        return restart_reattach.heal(handle, error)
    return restart_reattach.refuse_pinned(handle, error)


_EXIT_REFUSAL = "the driver is exiting, so this render stops here"


@pytest.mark.parametrize("site", ["price_check", "heal", "pinned"])
@pytest.mark.parametrize("exiting", [True, False], ids=["exiting", "not_exiting"])
def test_at_exit_the_first_id_reading_never_starts(driver, monkeypatch, site, exiting):
    """ComfyUI begins to exit after the price failed and before the restart check,
    the heal or a pinned render's retire reads the loops' ids. None of them starts
    that reading, which on the pair asks each host over ssh with up to 3 s per host:
    each ends at once in the exit refusal, with no WARNING or ERROR on the render's
    thread, no eviction, no stop, no attach and no telemetry. On the commit before,
    each read every host first. Not exiting, each reads once and goes on as before:
    the check raises the heal's error with its reading, the heal attaches by the
    new ids, and a pinned render retires the stale mesh and asks for a re-queue."""
    rig = _exit_rig(driver, monkeypatch)
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    error = _delivery_error(driver)  # every loop restarted; the error carries no reading
    fleet = _Fleet({0: _supervision_error(), 1: _supervision_error()})
    rig.listen()
    if exiting:
        rig.flag.set()  # the exit hook ran after the price failed
        with pytest.raises(WorkerDeliveryError) as raised:
            _first_reading_site(site, handle, error, fleet)
        assert _EXIT_REFUSAL in str(raised.value)
        assert getattr(raised.value, "moves", None) is None  # nothing for the heal to trust
        assert rig.reads == 0 and rig.loud() == [], rig.loud()
        assert fleet.evicting == [] and handle.defunct is False
        assert driver.procs[0].stops == [] and len(driver.dials) == 1 and driver.events == []
        return
    if site == "heal":
        outcome = _first_reading_site(site, handle, error, fleet)
        assert outcome.handle is not handle and len(driver.dials) == 2
        assert any(text.startswith("restart_reattach outcome=attached") for text in rig.loud())
    else:
        with pytest.raises(WorkerDeliveryError) as raised:
            _first_reading_site(site, handle, error, fleet)
        assert _EXIT_REFUSAL not in str(raised.value) and len(driver.dials) == 1
        if site == "price_check":
            assert raised.value.moves.verdict == restart_reattach.ALL and fleet.evicting == []
        else:
            assert "The Worker loops restarted; queue again" in str(raised.value)
            assert any("outcome=retired" in text for text in rig.loud()), rig.loud()
    assert rig.reads == (2 if site == "heal" else 1)  # the heal's replacement attach reads its own


@pytest.mark.parametrize("site", ["price_check", "heal", "pinned"])
@pytest.mark.parametrize("found", ["partial", "unknown", "none"])
def test_a_reading_the_exit_overtakes_ends_in_the_exit_refusal(driver, monkeypatch, site, found):
    """The driver begins to exit while the loops' ids are read, and the reading
    finds that not every loop restarted, or an id it could not read. The checks
    that follow a full restart never run then. On the commit before, the user got
    that verdict's own words while ComfyUI closed: the price went on to its
    evicting wait and refused as a fleet that did not answer, the heal named the
    verdict's remedy (restart the loops together, queue again once each publishes
    its id, or press Reset), and a pinned render raised the price's refusal. Now
    each ends in the exit refusal after that one reading, with no eviction, no stop,
    no attach and no WARNING or ERROR on the render's thread."""
    rig = _exit_rig(driver, monkeypatch)
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    error = _delivery_error(driver, {"partial": {A: NEW[A], B: OLD[B]}, "none": OLD}.get(found))
    if found == "unknown":
        rig.readings = _readings(NEW)
        rig.readings[A] = {"gen": "unknown", "error": "ssh failed"}
    fleet = _Fleet({0: _supervision_error(), 1: _supervision_error()})
    rig.exit_on_read = True
    rig.listen()
    with pytest.raises(WorkerDeliveryError) as raised:
        _first_reading_site(site, handle, error, fleet)
    text = str(raised.value)
    assert _EXIT_REFUSAL in text
    for today in ("dgxm restart", "Reset attached mesh", "publishes its id", "queue again"):
        assert today not in text
    assert rig.reads == 1 and rig.loud() == [], rig.loud()
    assert fleet.evicting == [] and handle.defunct is False
    assert driver.procs[0].stops == [] and len(driver.dials) == 1 and driver.events == []


def _owned_submit(monkeypatch, handle, error, during=None, *, raises=True):
    """The real submit_render for a sampler render that owns its session, with the
    real heal: its dispatch binds ``handle``, runs ``during``, meets ``error``, and
    any later dispatch answers. Returns the render's error, or with ``raises``
    false its result, and one entry per dispatch."""
    from dgx_monarch.nodes import recycle_drain, render_submit

    dispatches: list[int] = []

    def guarded(model, request, latent, cfg, steps, seq, depth, progress, handoff, guard,
                *, adoption_context=None):
        dispatches.append(1)
        guard.handle = handle
        guard.dispatch_started = True
        if len(dispatches) > 1:
            return "pending"
        if during is not None:
            during()
        raise error

    monkeypatch.setattr(render_submit, "_submit_render_guarded", guarded)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    if not raises:
        return render_submit.submit_render(None, {}, {}, None, 1), dispatches
    with pytest.raises(WorkerDeliveryError) as raised:
        render_submit.submit_render(None, {}, {}, None, 1)
    return raised.value, dispatches


@pytest.mark.parametrize(("render", "when"), [
    ("owned", "price_check"), ("pinned", "price_check"), ("owned", "after_the_price"),
    ("pinned", "after_the_price"), ("owned", "during_the_replacement_attach")])
def test_through_the_submit_path_an_exit_starts_no_reading_and_no_second_dispatch(
        driver, monkeypatch, render, when):
    """Through the real submit path and the real detours, for a sampler render that
    owns its session and for one in a claimed Pipeline or Gate session. The exit
    begins before the price's restart check, or after the price raised a delivery
    failure and before the detour runs: the price's exit refusal goes to the
    guard's eviction, which does nothing at exit, and on to the detour, which
    stands down again; or the delivery failure reaches a detour that reads nothing.
    Either way no id is read, nothing is stopped or attached, and the render ends
    in the exit refusal. The exit can also begin while the replacement attach
    dials, which then succeeds: the render does not dispatch again on the new
    mesh, where on the commit before it did."""
    rig = _exit_rig(driver, monkeypatch)
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    during = None
    if when == "price_check":
        rig.flag.set()
        try:
            try:
                raise _supervision_error()
            except Exception as failure:
                restart_reattach.refuse_if_restarted(handle, failure, 2.0, 30.0)
        except WorkerDeliveryError as exc:
            error = exc
    else:
        error = _delivery_error(driver)
        if when == "after_the_price":
            during = rig.flag.set
        else:
            real_live = mesh_mod.ensure_live

            def ensure_live(*args, **kwargs):
                fresh = real_live(*args, **kwargs)
                rig.kept.clear()  # the stop's lines came before the exit
                rig.flag.set()  # the exit hook ran while the attach dialed
                return fresh

            monkeypatch.setattr(mesh_mod, "ensure_live", ensure_live)
    rig.listen()
    if render == "owned":
        refusal, dispatches = _owned_submit(monkeypatch, handle, error, during)
    else:
        refusal, events = _pinned_submit(monkeypatch, handle, error, during)
        dispatches = [event for event in events if event == "dispatch"]
    assert _EXIT_REFUSAL in str(refusal) and "queue again" not in str(refusal)
    assert len(dispatches) == 1  # never a second dispatch
    assert rig.loud() == [], rig.loud()
    if when == "during_the_replacement_attach":
        assert rig.reads == 2 and len(driver.dials) == 2  # the heal's and the attach's; nothing after
        return
    assert rig.reads == 0 and handle.defunct is False
    assert driver.procs[0].stops == [] and len(driver.dials) == 1 and driver.events == []


def _exit_as_heal_returns(monkeypatch, flag, kept=None) -> None:
    """The driver begins to exit just after ``restart_reattach.heal`` returns, before
    its caller's one retry; what was logged before that came before the exit."""
    real_heal = restart_reattach.heal

    def heal(*args, **kwargs):
        outcome = real_heal(*args, **kwargs)
        if kept is not None:
            kept.clear()
        flag.set()
        return outcome

    monkeypatch.setattr(restart_reattach, "heal", heal)


@pytest.mark.parametrize("exiting", [True, False], ids=["exiting", "not_exiting"])
def test_an_exit_after_the_heal_returns_dispatches_nothing_on_the_new_mesh(
        driver, monkeypatch, exiting):
    """Through the real submit path and the real heal, for a sampler render that owns
    its session. The exit begins after the heal returned the new mesh and before the
    render dispatches again: the render ends in the exit refusal, with no second
    dispatch, so no price and no call reach the new mesh, and nothing above INFO
    on its thread. On the commit before, it dispatched again and rendered while
    ComfyUI closed. Not exiting, it dispatches once more and renders, as before."""
    rig = _exit_rig(driver, monkeypatch)
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    error = _delivery_error(driver)
    if exiting:
        _exit_as_heal_returns(monkeypatch, rig.flag, rig.kept)
    rig.listen()
    result, dispatches = _owned_submit(monkeypatch, handle, error, raises=exiting)
    assert rig.reads == 2 and len(driver.dials) == 2  # the heal's reading and its attach
    if not exiting:
        assert result == "pending" and len(dispatches) == 2
        return
    assert _EXIT_REFUSAL in str(result) and result.__cause__ is error
    assert len(dispatches) == 1  # no second dispatch: no price and no call on the new mesh
    assert rig.loud() == [], rig.loud()


@pytest.mark.parametrize("exiting", [True, False], ids=["exiting", "not_exiting"])
def test_an_exit_after_the_loaders_heal_returns_loads_nothing_on_the_new_mesh(
        loader_rig, monkeypatch, exiting):
    """The UNET Loader's eager load: the exit begins after the heal returned the
    new mesh and before the second load. The load ends in the exit refusal with no
    setup, no price and no load on the new mesh, and nothing above INFO. On the
    commit before, it priced and loaded on the new mesh while ComfyUI closed. Not
    exiting, it loads once more on the new mesh, as before."""
    import logging

    from dgx_monarch import driver_exit

    monkeypatch.setattr(driver_exit, "_EXITING", threading.Event())
    kept: list[logging.LogRecord] = []
    keep = logging.Handler()
    keep.emit = kept.append  # type: ignore[method-assign]
    for name in (*_EXIT_PATH_LOGGERS, "dgx_monarch.nodes.loaders"):
        logger = logging.getLogger(name)
        monkeypatch.setattr(logger, "handlers", [*logger.handlers, keep])
    if exiting:
        _exit_as_heal_returns(monkeypatch, driver_exit._EXITING, kept)
    before_heal = [("price",), ("ensure_live",), ("setup", "stale"), ("agree", "stale"),
                   ("heal", "stale"), ("drop_memos",), ("price",), ("ensure_live",)]
    if not exiting:
        loader_rig.load()
        assert loader_rig.events == [*before_heal, ("setup", "fresh"), ("agree", "fresh"),
                                     ("load_model", "fresh"), ("record", "fresh")]
        return
    with pytest.raises(WorkerDeliveryError, match=_EXIT_REFUSAL):
        loader_rig.load()
    assert loader_rig.events == before_heal  # nothing on the new mesh after the heal
    loud = [record.getMessage() for record in kept if record.levelno >= logging.WARNING]
    assert loud == [], loud


def test_a_real_handle_evicted_with_its_stop_done_heals_without_a_second_stop(driver):
    """Another path (the panel's status poll, a second render) evicted the
    fleet through the supervision publication and its stop finished. The heal
    reads that outcome, stops nothing more, and attaches by the new ids."""
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    done = threading.Event()
    done.set()
    handle.procs = _HeldProcs(done)
    mesh_helpers.mark_defunct_on_supervision_failure(handle, _supervision_error())
    assert handle.defunct and handle.teardown_complete
    stops = list(handle.procs.stops)
    outcome = restart_reattach.heal(handle, _restarted(driver))
    assert outcome.handle is not handle and outcome.teardown == "stop_confirmed"
    assert handle.procs.stops == stops  # the heal sent no stop of its own
    assert driver.dials[-1] == [worker_proc_id.proc_addr(A, NEW[A]),
                                worker_proc_id.proc_addr(B, NEW[B])]


def _reconcile_running(driver, release: threading.Event):
    """A real handle whose supervision reconcile another thread runs, inside its stop."""
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    handle.procs = _HeldProcs(release)
    mesh_helpers.mark_defunct_on_supervision_failure(
        handle, _supervision_error(), holding_lock=True)  # the reconcile runs on its own thread
    assert handle.procs.entered.wait(5.0)
    return handle


def test_a_real_handle_whose_reconcile_is_running_heals_after_that_stop_ends(driver):
    """A reconcile another thread started is still in its stop when the price's
    check finds every loop restarted. The heal waits for that stop under the
    handle's lock, sends none of its own, and attaches by the new ids once it
    has ended: the wait the price's own eviction would have joined, with no
    second stop beside it."""
    release = threading.Event()
    handle = _reconcile_running(driver, release)
    refused = _restarted(driver)
    timer = threading.Timer(0.5, release.set)
    timer.start()
    started = time.monotonic()
    try:
        outcome = restart_reattach.heal(handle, refused)
    finally:
        timer.cancel()
        release.set()
    assert time.monotonic() - started >= 0.4  # it waited for the running stop
    assert outcome.handle is not handle and outcome.teardown == "stop_confirmed"
    assert len(handle.procs.stops) == 1 and len(driver.dials) == 2


def test_a_real_handle_whose_running_reconcile_fails_refuses_and_attaches_nothing(
        driver, monkeypatch):
    """The same, when the other thread's stop runs out its bound: the latch
    stays blocked, and the heal refuses without a second stop or an attach."""
    monkeypatch.setattr(mesh_teardown, "proc_stop_bound_s", lambda: 0.5)
    release = threading.Event()
    handle = _reconcile_running(driver, release)
    try:
        with pytest.raises(WorkerDeliveryError, match="could not retire the stale mesh"):
            restart_reattach.heal(handle, _restarted(driver))
    finally:
        release.set()
    assert len(handle.procs.stops) == 1 and len(driver.dials) == 1
    assert handle.replacement_blocked


def test_one_loop_moved_refuses_fast_and_leaves_the_mesh_alone(driver, monkeypatch):
    _head_named(monkeypatch, "h1")
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = {A: NEW[A], B: OLD[B]}
    started = time.monotonic()
    with pytest.raises(WorkerDeliveryError) as raised:
        restart_reattach.heal(handle, _refused())
    assert time.monotonic() - started < 1.0
    assert "dgxm restart" in str(raised.value)
    assert ("the Worker loops on the head serve new proc ids while those on the sibling "
            "still serve the id this mesh dialed") in str(raised.value)
    assert "10.0.0." not in str(raised.value) and "h2" not in str(raised.value)
    assert handle.defunct is False and driver.procs[0].stops == []
    assert len(driver.dials) == 1
    assert _trace(driver, "restart_reattach")[-1]["reason"] == "partial"


def test_no_loop_moved_refuses_with_the_reset_remedy(driver):
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    with pytest.raises(WorkerDeliveryError, match="Reset attached mesh"):
        restart_reattach.heal(handle, _refused())
    assert handle.defunct is False and len(driver.dials) == 1


def test_old_processes_not_proven_gone_refuse_and_attach_nothing(driver, monkeypatch):
    _head_named(monkeypatch, "h1")
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = dict(NEW)
    driver.liveness = mesh_liveness.ALIVE
    with pytest.raises(WorkerDeliveryError, match="liveness check answered alive") as raised:
        restart_reattach.heal(handle, _refused())
    assert ("(the head: alive, the sibling: alive)" in str(raised.value)
            and "(on the head and the sibling)" in str(raised.value))
    assert "10.0.0." not in str(raised.value)
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


def _restart_check_error(monkeypatch, moves) -> WorkerDeliveryError:
    """The error the price's restart check raises, built as production builds it."""
    monkeypatch.setattr(restart_reattach, "read_moves", lambda config, timeout=None: moves)
    handle = SimpleNamespace(config=_config())
    try:
        try:
            raise _supervision_error()
        except Exception as failure:
            restart_reattach.refuse_if_restarted(handle, failure, 2.0, 30.0)
    except WorkerDeliveryError as exc:
        return exc
    raise AssertionError("the restart check did not raise")


def _guard_route(monkeypatch, route: str) -> BaseException:
    if route == "restart_check":
        return _restart_check_error(monkeypatch, restart_reattach.Moves(moved=(A, B)))
    if route == "delivery":
        try:
            restart_reattach.refuse_undelivered(SimpleNamespace(config=_config()), [A], 0.5, 30.0)
        except WorkerDeliveryError as exc:
            return exc
    if route == "check_not_all":
        # Not a reading the check raises on; the hook must not trust its type alone.
        error = WorkerDeliveryError("[dgxm:C guard=cross_rank_capacity waivable=0] failed "
                                    "(SupervisionError)",
                                    moves=restart_reattach.Moves(moved=(A,), kept=(B,)))
        error.__cause__ = _supervision_error()
        return error
    if route == "supervision":
        return _supervision_error()
    if route == "quote_unavailable":
        return FleetQuoteUnavailableError(
            "[dgxm:C guard=cross_rank_capacity waivable=0] the sibling at a failed to "
            "answer (SupervisionError)")
    raise AssertionError(route)


@pytest.mark.parametrize(("route", "inherited", "expected"), [
    ("restart_check", False, ["heal"]),
    ("delivery", False, ["heal"]),
    ("check_not_all", False, ["evicted", "heal"]),
    ("supervision", False, ["evicted"]),
    ("quote_unavailable", False, ["evicted"]),
    ("restart_check", True, ["pinned"]),
    ("delivery", True, ["pinned"]),
    ("supervision", True, ["evicted"]),
    ("busy_then_restart", False, ["evicted"]),
    ("busy_then_restart", True, ["evicted"]),
], ids=["restart_check", "delivery", "check_not_all", "supervision", "quote_unavailable",
        "restart_check_pipeline", "delivery_pipeline", "supervision_pipeline",
        "retry_restart_check", "retry_restart_check_pipeline"])
def test_the_submit_guard_leaves_only_a_full_restart_to_the_heal_unevicted(
        monkeypatch, route, inherited, expected):
    """Through the real submit_render, with the dispatch marked started before
    the price raises, as a sampler render marks it. The restart check's error
    with every loop restarted reaches the heal with the fleet unevicted, as a
    delivery failure does; on the commit before this rule the guard's failure
    hook evicted it first and joined the post-death stop. Every other failure
    of a started dispatch is evicted as before. A Pipeline or Identity Gate
    session does not heal: both errors reach refuse_pinned unevicted (its
    behaviour is tested below under a real session), and any other failure is
    evicted there as before. A retry that a busy fleet's heal or an old
    model's clear runs is caught by neither, so a full restart on it is
    evicted as before."""
    from dgx_monarch import mesh_setup
    from dgx_monarch.nodes import recycle_drain, render_submit

    events: list[str] = []
    retry = route == "busy_then_restart"
    error = _guard_route(monkeypatch, "restart_check" if retry else route)
    errors = [mesh_setup.LifecycleBusyError("busy"), error] if retry else [error]
    monkeypatch.setattr(mesh_helpers, "_publish_defunct",
                        lambda handle, exc, text, **_kwargs: events.append("evicted"))
    monkeypatch.setattr(render_submit, "_heal_abandoned_wedge_once", lambda *_a: True)
    monkeypatch.setattr(render_submit.render_preflight, "_model_with_bound_handle",
                        lambda model, _handle: model)
    calls = {"n": 0}

    def guarded(model, request, latent, cfg, steps, seq, depth, progress, handoff, guard,
                *, adoption_context=None):
        calls["n"] += 1
        if calls["n"] <= len(errors):
            guard.handle = object()
            guard.dispatch_started = True  # set before submit_sample runs the price
            raise errors[calls["n"] - 1]
        return "pending"

    def heal(_handle, _exc):
        events.append("heal")
        return restart_reattach.Reattached(object(), 1.0, (A, B), "liveness_release")

    def pinned(_handle, exc):
        events.append("pinned")
        raise exc

    monkeypatch.setattr(render_submit, "_submit_render_guarded", guarded)
    monkeypatch.setattr(restart_reattach, "heal", heal)
    monkeypatch.setattr(restart_reattach, "refuse_pinned", pinned)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    monkeypatch.setattr(render_submit.render_preflight,
                        "activation_footprint_preflight_for_request", lambda *a: None)
    monkeypatch.setattr(render_submit, "active_render_session",
                        lambda: object() if inherited else None)
    raises = route in ("supervision", "quote_unavailable") or inherited or retry
    if raises:
        with pytest.raises(type(error)):
            render_submit.submit_render(None, {}, {}, None, 1)
    else:
        assert render_submit.submit_render(None, {}, {}, None, 1) == "pending"
    assert events == expected and calls["n"] == len(errors) + (not raises)


# ------------------------------------------------- a pinned session's restart


def _check_error(driver) -> WorkerDeliveryError:
    """Every loop restarted; the price's restart check raises, as production builds it."""
    driver.ids = dict(NEW)
    try:
        try:
            raise _supervision_error()
        except Exception as failure:
            restart_reattach.refuse_if_restarted(
                SimpleNamespace(config=_config()), failure, 2.0, 30.0)
    except WorkerDeliveryError as exc:
        return exc
    raise AssertionError("the restart check did not raise")


def _delivery_error(driver, ids=None) -> WorkerDeliveryError:
    """A message to the old actors came back undeliverable; ``ids`` the loops serve now."""
    driver.ids = dict(ids or NEW)
    try:
        restart_reattach.refuse_undelivered(SimpleNamespace(config=_config()), [A], 0.5, 30.0)
    except WorkerDeliveryError as exc:
        return exc
    raise AssertionError("no delivery refusal")


def _lease(handle):
    """A sample lease an earlier pipelined render still holds on ``handle``."""
    from dgx_monarch.mesh_lease import SetupBoundFuture

    lease = SetupBoundFuture.prepared(handle, 0)
    with handle.lock:
        lease._register_locked()
        lease._mark_enqueue_started_locked()
        lease._bind_future_locked(SimpleNamespace(get=lambda timeout=None: None))
    return lease


def _pinned_rig(monkeypatch, handle, error, during=None):
    """The real submit_render in a Pipeline's or Gate's place: its body binds
    ``handle``, marks the dispatch started, runs ``during`` and meets ``error``."""
    from dgx_monarch.nodes import recycle_drain, render_submit

    events: list[str] = []
    real_publish = mesh_helpers._publish_defunct

    def publish(target, exc, text, **kwargs):
        events.append("evicted")
        return real_publish(target, exc, text, **kwargs)

    def guarded(model, request, latent, cfg, steps, seq, depth, progress, handoff, guard,
                *, adoption_context=None):
        events.append("dispatch")
        guard.handle = handle
        guard.dispatch_started = True
        if during is not None:
            during()
        raise error

    monkeypatch.setattr(mesh_helpers, "_publish_defunct", publish)
    monkeypatch.setattr(render_submit, "_submit_render_guarded", guarded)
    monkeypatch.setattr(restart_reattach, "heal",
                        lambda *_a, **_k: pytest.fail("a pinned session must not heal"))
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    return render_submit, events


def _pinned_submit(monkeypatch, handle, error, during=None, before_close=None):
    """One render under a real claimed RenderSession, as a Pipeline or Gate run makes it;
    ``before_close`` sees that session after the refusal, before the run closes it."""
    from dgx_monarch.mesh_session import RenderSession

    render_submit, events = _pinned_rig(monkeypatch, handle, error, during)
    session = RenderSession()
    session.bind(handle)
    try:
        with session.activate(), pytest.raises(WorkerDeliveryError) as raised:
            render_submit.submit_render(None, {}, {}, None, 1)
        if before_close is not None:
            before_close(session)
    finally:
        session.close()
    return raised.value, events


@pytest.mark.parametrize("route", ["restart_check", "delivery"])
@pytest.mark.parametrize("teardown", ["stop_confirmed", "liveness_release", "release_refused"])
def test_a_pinned_session_retires_a_full_restart_and_refuses_queue_again(
        driver, monkeypatch, route, teardown):
    """A Pipeline or Identity Gate render cannot attach again inside its run.
    With every loop restarted and nothing else holding the fleet, it retires
    the stale fleet with the heal's own stop and proof, attaches nothing, and
    refuses so that the next render attaches a fresh one. Before this rule the
    restart check's error was evicted behind the post-death stop's join, and a
    delivery failure left the fleet as it was, so every re-queue refused until
    a KSampler render healed. A retire that cannot finish refuses as the heal
    does, with no eviction and nothing attached."""
    from dgx_monarch.mesh_session import (
        ConcurrentRenderSessionError,
        RenderSession,
        RetiredRenderHandleError,
    )

    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    if teardown == "stop_confirmed":
        done = threading.Event()
        done.set()
        handle.procs = _HeldProcs(done)
    if teardown == "release_refused":
        driver.liveness = mesh_liveness.ALIVE
    error = _check_error(driver) if route == "restart_check" else _delivery_error(driver)
    other = SimpleNamespace(fresh=None)

    def before_close(session) -> None:
        # A later step of the same run can neither dispatch onto the retired
        # handle nor cross to another mesh inside the old session.
        other.dials = len(driver.dials)  # what the refused render attached: nothing
        with pytest.raises(RetiredRenderHandleError):
            session.bind(handle)
        if teardown != "release_refused":
            other.fresh = mesh_mod.ensure_live(handle)
            with pytest.raises(ConcurrentRenderSessionError, match="multiple mesh handles"):
                session.bind(other.fresh)

    refusal, events = _pinned_submit(monkeypatch, handle, error, before_close=before_close)
    assert events == ["dispatch"]  # no eviction and no second dispatch
    assert other.dials == 1 and handle.defunct  # nothing attached inside the session
    if teardown == "release_refused":
        assert "could not retire the stale mesh" in str(refusal)
        assert handle.replacement_blocked and len(driver.dials) == 1
        assert _trace(driver, "restart_reattach")[-1]["reason"] == "pinned_not_retired"
        return
    assert refusal is not error and refusal.__cause__ is error
    assert "The Worker loops restarted; queue again" in str(refusal)
    assert parse_leading_refusal_tag(str(refusal)) is not None
    traced = _trace(driver, "restart_reattach")[-1]
    assert (traced["outcome"], traced["teardown"]) == ("retired", teardown)
    with pytest.raises(RetiredRenderHandleError):
        RenderSession().bind(handle)  # no later run can dispatch onto it either
    fresh = mesh_mod.ensure_live(handle)
    assert fresh is other.fresh and fresh is not handle and len(driver.dials) == 2
    RenderSession().bind(fresh)  # the next run's session takes the fresh fleet
    assert driver.dials[-1] == [worker_proc_id.proc_addr(A, NEW[A]),
                                worker_proc_id.proc_addr(B, NEW[B])]


@pytest.mark.parametrize(("route", "case"), [
    ("restart_check", "leases_held"), ("delivery", "leases_held"),
    ("restart_check", "teardown_in_progress"), ("delivery", "teardown_in_progress"),
    ("restart_check", "prior_stop_unknown"), ("delivery", "prior_stop_unknown"),
    ("delivery", "partial"), ("delivery", "none"), ("delivery", "unknown"),
])
def test_a_pinned_session_leaves_every_other_case_as_before(driver, monkeypatch, route, case):
    """Every case outside a full restart with nothing holding the fleet keeps
    the behaviour before this rule: the restart check's error is evicted as
    any supervision failure (deferred while a lease is held, absorbed by a
    teardown in progress) and a delivery failure is raised unchanged, with no
    stop of this path's own and nothing attached."""
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    ids = {"partial": {A: NEW[A], B: OLD[B]}, "none": dict(OLD)}.get(case)
    error = _check_error(driver) if route == "restart_check" else _delivery_error(driver, ids)
    if case == "unknown":
        monkeypatch.setattr(listener_generation, "fleet", lambda _config, **_k: {})
    held: list = []

    def during():
        if case == "leases_held":
            held.append(_lease(handle))
        elif case == "teardown_in_progress":
            mesh_safety.begin_deliberate_teardown(id(handle))  # a Reset's stop in flight
        elif case == "prior_stop_unknown":
            handle.replacement_blocked = "an earlier ProcMesh stop has no known outcome"

    refusal, events = _pinned_submit(monkeypatch, handle, error, during)
    assert refusal is error and len(driver.dials) == 1
    assert _trace(driver, "restart_reattach")[-1]["outcome"] == "refused"
    assert driver.procs[0].stops == []  # no stop of this path's own
    if route == "delivery":
        assert events == ["dispatch"]
        return
    assert events[-1] == "evicted" and handle.defunct
    if case == "leases_held":
        assert handle.deferred_supervision_error is not None  # deferred to the lease
    text = str(refusal)
    assert text.index("queue this workflow again") < text.index(
        "run one render through KSampler")
    # It says what the render did, which holds in every case here, and never
    # that the fleet is gone.
    assert "handed this fleet to the driver's failure cleanup" in text
    assert "evicted this fleet" not in text
    if case == "prior_stop_unknown":
        # That cleanup cannot stop the fleet, so the next render names what blocks it.
        assert handle.replacement_blocked and not handle.teardown_complete
        with pytest.raises(Exception, match=r"cannot be used or replaced: .*teardown outcome"):
            mesh_mod.ensure_live(handle)
        assert len(driver.dials) == 1


@pytest.mark.parametrize("route", ["restart_check", "delivery"])
@pytest.mark.parametrize("reconcile", ["running", "stuck", "ended_blocked"])
def test_a_pinned_render_during_a_running_cleanup_says_to_try_again_when_it_ends(
        driver, monkeypatch, route, reconcile):
    """The driver's own cleanup is still stopping this fleet when a Pipeline or
    Gate render meets every loop restarted. That is a teardown in progress, so
    nothing is retired, and while the reconcile's thread is alive and inside
    its own wait the refusal says what the driver's other doors say then: try
    again when that stop ends. A reconcile alive past its wait (stuck), or one
    whose stop ended without an outcome, keeps the refusal it had."""
    from dgx_monarch import mesh_teardown as teardown
    from dgx_monarch import reconcile_running

    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    error = _check_error(driver) if route == "restart_check" else _delivery_error(driver)
    release = threading.Event()
    if reconcile == "stuck":
        monkeypatch.setattr(reconcile_running, "reconcile_join_s", lambda: 0.0)
    if reconcile == "ended_blocked":
        monkeypatch.setattr(mesh_teardown, "proc_stop_bound_s", lambda: 0.3)

    def during():
        handle.procs = _HeldProcs(release)
        mesh_helpers.mark_defunct_on_supervision_failure(
            handle, _supervision_error(), holding_lock=True)  # the reconcile's own thread
        assert handle.procs.entered.wait(5.0)
        if reconcile == "ended_blocked":
            handle._supervision_reconcile_thread.join(10.0)
            assert not handle._supervision_reconcile_thread.is_alive()
            assert handle.replacement_blocked
        else:
            # The session's close waits on the lock that stop holds, as it always
            # has; the pinned path decides before that.
            timer = threading.Timer(1.0, release.set)
            timer.daemon = True
            timer.start()

    try:
        refusal, _events = _pinned_submit(monkeypatch, handle, error, during)
    finally:
        release.set()
    sentence = teardown.cleanup_running_sentence()
    assert len(driver.dials) == 1 and len(handle.procs.stops) == 1  # no stop of its own
    if reconcile == "running":
        assert refusal is not error and refusal.__cause__ is error
        assert sentence in str(refusal) and "try again when it ends" in str(refusal)
        assert parse_leading_refusal_tag(str(refusal)) is not None
        assert _trace(driver, "restart_reattach")[-1]["reason"] == "pinned_teardown_in_progress"
    else:
        assert refusal is error and sentence not in str(refusal)


def test_loops_restarted_on_another_torchmonarch_retire_then_refuse_before_dialing(
        driver, monkeypatch):
    """TROUBLESHOOTING entry 2, Upgrading: every loop restarted on another
    torchmonarch. The ids moved, so the heal retires the stale mesh, and its
    attach then refuses before it dials, naming both versions."""
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    driver.ids = dict(NEW)
    older = {address: dict(fields, monarch="0.7.0")
             for address, fields in _readings(NEW).items()}
    monkeypatch.setattr(listener_generation, "fleet", lambda _config, **_k: older)
    refused = _refused()
    refused.moves = restart_reattach.read_moves(_config())
    assert refused.moves.verdict == restart_reattach.ALL
    with pytest.raises(Exception, match=r"runs torchmonarch 0\.7\.0 and this driver runs"):
        restart_reattach.heal(handle, refused)
    assert handle.defunct and len(driver.dials) == 1  # retired, and nothing dialed after
    assert _trace(driver, "restart_reattach")[-1]["outcome"] == "failed"


def test_a_pipeline_at_depth_two_stops_nothing_before_its_abort_retires_the_lease(
        driver, monkeypatch):
    """A real RenderPipeline at depth 2: the second render meets every loop
    restarted while the first one's result is still in flight on the old
    fleet. The pinned path retires nothing; the restart check's error is
    evicted as before and that eviction waits on the lease, so no stop runs
    until the pipeline's abort abandons the first render, and the deferred
    stop runs then. Nothing is attached inside the pipeline's session."""
    from dgx_monarch import mesh_session
    from dgx_monarch.nodes import common
    from dgx_monarch.nodes.pipeline import RenderPipeline

    monkeypatch.setattr(mesh_teardown, "proc_stop_bound_s", lambda: 0.5)
    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    error = _check_error(driver)
    render_submit, events = _pinned_rig(monkeypatch, handle, error)
    order: list[tuple[str, int]] = []

    class InFlight:
        def __init__(self) -> None:
            self.lease = _lease(handle)

        def result(self):
            raise RuntimeError("the old actors are gone")

        def abandon(self) -> None:
            order.append(("abandon", len(driver.procs[0].stops)))
            self.lease.abandon()

    real_submit = render_submit.submit_render

    def submit(model, request, latent, cfg_value, steps_hint, seq=0, depth=1,
               progress_on_step=None, handoff=None):
        if seq:
            return real_submit(model, request, latent, cfg_value, steps_hint, seq=seq,
                               depth=depth, progress_on_step=progress_on_step, handoff=handoff)
        mesh_session.active_render_session().bind(handle)  # as the real submit claims it
        return InFlight()

    monkeypatch.setattr(common, "submit_render", submit)
    pipe = RenderPipeline(depth=2)
    pipe.push(None, {}, {}, 4.5, 10)
    with pytest.raises(WorkerDeliveryError) as raised:
        pipe.push(None, {}, {}, 4.5, 10)
    # The eviction published at the refusal and deferred, then resumed by the lease.
    assert raised.value is error and events == ["dispatch", "evicted", "evicted"]
    assert order == [("abandon", 0)]  # no stop before the abort retired the lease
    assert driver.procs[0].stops and not handle.sample_leases.get(0)  # the deferred stop ran
    assert len(driver.dials) == 1
    assert _trace(driver, "restart_reattach")[-1]["reason"] == "pinned_leases_held"


def test_a_gate_run_quarantines_pushes_nothing_and_the_next_attach_is_fresh(
        driver, monkeypatch):
    """An Identity Gate ceremony under its own real session: its leg meets
    every loop restarted, the stale fleet is retired, the abort writes the
    quarantine as for any abort and pushes nothing to the gone actors, the
    session is released, and the next attach is a fresh fleet."""
    from dgx_monarch.mesh_session import RenderSession
    from dgx_monarch.nodes import common, gate, gate_identity, gate_session

    handle = mesh_mod.get_mesh(mode="cluster", config_path=SOURCE)
    error = _check_error(driver)
    render_submit, events = _pinned_rig(monkeypatch, handle, error)
    quarantined: list = []
    monkeypatch.setattr(gate_identity, "quarantine_unproven_paths",
                        lambda model, cause=None, uncond=None: quarantined.append(cause))
    handle.call_all = lambda *_a, **_k: pytest.fail("nothing may be pushed to gone actors")
    model = SimpleNamespace(mesh=SimpleNamespace(handle=handle, topology_preset="auto"),
                            loras=())
    monkeypatch.setattr(common, "_bind_packed_render_model",
                        lambda model, latent, cfg, **_k: (model, handle))

    def bound(*_args, **_kwargs):
        return render_submit.submit_render(None, {}, {}, None, 1)

    runtime = {"ensure_live": lambda target: target, "RenderSession": RenderSession,
               "_run_identity_ceremony_bound": bound,
               "_force_stock_quarantine": gate_identity.force_stock_quarantine,
               "_close_gate_session": gate._close_gate_session, "log": restart_reattach.log}
    with pytest.raises(WorkerDeliveryError) as raised:
        gate_session.run_identity_ceremony(model, {}, {}, 4.5, 10, "auto_first_use", "run1",
                                           runtime=runtime)
    assert "The Worker loops restarted; queue again" in str(raised.value)
    assert quarantined == [raised.value] and events == ["dispatch"]
    RenderSession().bind(fresh := mesh_mod.ensure_live(handle))  # released; a fresh fleet
    assert fresh is not handle and len(driver.dials) == 2


def test_a_detour_that_cannot_heal_raises_its_refusal_without_a_retry(monkeypatch):
    def refuses(_handle, exc):
        raise WorkerDeliveryError("[dgxm:C guard=cross_rank_capacity waivable=0] partial") from exc

    render_submit, state = _render_rig(monkeypatch, heal=refuses)
    with pytest.raises(WorkerDeliveryError, match="partial"):
        render_submit.submit_render(None, {}, {}, None, 1)
    assert state.calls == 1 and state.drops == 0


# A minimax_h3 checkpoint the size of the vendor's bf16 DiT, byte exact
# (docs/VALIDATION.md), and the driver host's MemAvailable once the old
# fleet's processes are gone with its weights (the auto posture in
# test_h3_activation_preflight.py).
H3_UNET = "minimax_h3_ref2va_bf16.safetensors"
H3_DIT_BF16 = 66_280_487_368
H3_MEM = int(92.2 * 2 ** 30)
# 1344x768 at 37 latent frames, whose activations price at about 39 GiB on one
# rank, and 512x320 at 2, about 0.4 GiB.
# The 15 second clip, 362 frames: at the 2026-10-31 activation price the
# 124-frame shape fits beside the bf16 weights in this posture, and this one
# does not, while its weights and its activations each fit alone.
H3_LONG_CLIP = (1, 24, 107, 48, 84)
H3_SHORT_CLIP = (1, 24, 2, 20, 32)


def _h3_retry_rig(monkeypatch, tmp_path, video):
    """An automatic minimax_h3 render on one rank that shares the driver's memory.

    The previous render's success memo credits its weights as resident. The
    checkpoint, its size and MemAvailable are stubbed and the driver footprint
    guard stands down, so only the H3 activation guard can refuse. The memo
    drain and the price are the shipped code; the dispatch and the re-attach
    are the render rig's stand-ins, and the re-attach returns a new mesh.
    """
    import sys

    from dgx_monarch import driver_footprint, h3_activation
    from dgx_monarch.nodes import recycle_drain, render_preflight

    drain = recycle_drain.drop_residency_memos
    checkpoint = tmp_path / H3_UNET
    checkpoint.write_bytes(b"x")
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.get_full_path = lambda _kind, _name: str(checkpoint)
    monkeypatch.setitem(sys.modules, "folder_paths", folder_paths)
    monkeypatch.setattr(render_preflight, "sniff_checkpoint",
                        lambda _path: ("minimax_h3", "bf16"))
    monkeypatch.setattr(driver_footprint, "file_size_bytes", lambda _path: H3_DIT_BF16)
    monkeypatch.setattr(driver_footprint, "driver_footprint_preflight",
                        lambda *_a, **_k: None)
    # A GPU-less runner would stand the guard down and pass the refusing test.
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: H3_MEM)
    monkeypatch.delenv(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV, raising=False)
    monkeypatch.setattr(render_preflight, "_LAST_RENDERED_UNET", H3_UNET)
    monkeypatch.setattr(render_preflight, "_LAST_SUBMITTED_UNET", None)
    monkeypatch.setattr(render_preflight, "_LAST_DRIVER_CHARGED_UNET", None)

    def mesh_handle(name):
        handle = SimpleNamespace(name=name, owns_hosts=False, world=1)
        handle.effective_worker_args = lambda requested=None: dict(requested or {})
        return handle

    fresh = mesh_handle("fresh")
    render_submit, state = _render_rig(monkeypatch, heal=lambda _handle, _exc: (
        restart_reattach.Reattached(fresh, 4.25, (A, B), "liveness_release")))
    # The rig counts drains; these renders take the shipped one.
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", drain)
    rig = SimpleNamespace(state=state, prices=[], estimates=[])
    price = render_preflight.activation_footprint_preflight_for_request
    estimate = h3_activation.estimate_h3_activation

    def priced(model, request, latent):
        rig.prices.append((model.mesh.handle.name, state.calls))
        price(model, request, latent)

    def estimated(**kwargs):
        rig.estimates.append(estimate(**kwargs))
        return rig.estimates[-1]

    monkeypatch.setattr(render_preflight, "activation_footprint_preflight_for_request", priced)
    monkeypatch.setattr(h3_activation, "estimate_h3_activation", estimated)
    model = SimpleNamespace(unet_name=H3_UNET, options={}, mesh=SimpleNamespace(
        topology_preset="auto", worker_args={}, world=1, handle=mesh_handle("stale")))
    request = {"positive": [[SimpleNamespace(shape=(1, 20, 4096)), {}]]}
    latent = {"samples": SimpleNamespace(shape=video)}
    # run_render's own price, before submit_render.
    rig.price = lambda: render_preflight.activation_footprint_preflight_for_request(
        model, request, latent)
    rig.submit = lambda: render_submit.submit_render(model, request, latent, None, 1)
    rig.drain = drain
    return rig


def test_a_retry_after_a_restart_prices_again_without_the_old_fleets_credit(
        monkeypatch, tmp_path):
    """Greptile P1 on PR 49: the retry dispatched on the first price, which
    credited the old fleet's weights. The weights and the activations each
    fit, and together they do not. The mutation that drops the second price
    dispatches this render a second time."""
    rig = _h3_retry_rig(monkeypatch, tmp_path, H3_LONG_CLIP)
    rig.price()  # the credit leaves the weights out, so the activations fit
    assert rig.estimates[-1].weight_bytes == 0 and rig.estimates[-1].fits
    with pytest.raises(mesh_safety.StockLoadCapacityError) as refused:
        rig.submit()
    assert rig.state.calls == 1
    assert rig.prices == [("stale", 0), ("fresh", 1)]
    charged = rig.estimates[-1]
    assert charged.weight_bytes == H3_DIT_BF16 and not charged.fits
    assert charged.weight_bytes <= charged.usable
    assert charged.activation_bytes <= charged.usable
    tag = parse_leading_refusal_tag(str(refused.value))
    assert tag is not None and tag.guard == "activation_footprint_preflight"
    # The refusal a first attempt with no credit raises, word for word.
    rig.drain()
    with pytest.raises(mesh_safety.StockLoadCapacityError) as first:
        rig.price()
    assert str(first.value) == str(refused.value)


def test_a_retry_whose_new_price_fits_dispatches_once_on_the_new_mesh(
        monkeypatch, tmp_path):
    rig = _h3_retry_rig(monkeypatch, tmp_path, H3_SHORT_CLIP)
    assert rig.submit() == "pending"
    assert rig.state.calls == 2
    assert rig.prices == [("fresh", 1)]
    assert [(e.weight_bytes, e.fits) for e in rig.estimates] == [(H3_DIT_BF16, True)]


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
