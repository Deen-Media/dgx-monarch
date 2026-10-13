"""A failed Fleet wave must not wedge the next Fleet queue (docs/TROUBLESHOOTING.md #87).

Fleet runs the render path's one-shot heal once per queue, on the live handle,
before its render session binds. When abandoned leases are the only
obstruction it recycles once, drops the residency memos and dispatches on the
fresh fleet. Every other busy state leaves the Fleet refusal as it was: the
queue meets the abandoned samples and refuses with the same error. A recycle
that is not confirmed raises that refusal before the session binds, because a
failed stop can leave the handle retired.
"""
from __future__ import annotations

import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import mesh_setup, telemetry_fleet
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.mesh_session import RetiredRenderHandleError
from dgx_monarch.nodes import (
    abandoned_heal,
    consent_waiver,
    fleet,
    recycle_drain,
    render_submit,
)
from dgx_monarch.nodes.common import MeshSpec, ModelSpec


class _Outcome:
    def __init__(self, ok):
        self.ok = ok
        self.status = type("S", (), {"value": "recycled" if ok else "blocked"})()
        self.detail = "probe"


class _Log:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def info(self, message, *args):
        self.lines.append(("info", message % args))

    def warning(self, message, *args):
        self.lines.append(("warning", message % args))


class _Mesh(MeshHandle):
    """A MeshHandle whose dispatch runs the real abandoned-sample refusal."""

    world = 2
    gpus_per_host = 1
    config = None

    def __init__(self, name, events, *, abandoned=0, active=0, recycle_ok=True):
        self.name = name
        self.events = events
        self.defunct = False
        self.teardown_complete = False
        self.replacement_blocked = None
        self.setup_cleanup_state = None
        self.abandoned_sample_leases = {1: abandoned} if abandoned else {}
        self.sample_leases = {1: active} if active else {}
        self.recycle_ok = recycle_ok

    def recycle_detailed(self):
        self.events.append(("recycle", self.name))
        if not self.recycle_ok:
            # What a timed-out ProcMesh stop leaves (mesh_recycle): the setup
            # stays DIRTY and replacement stays blocked on a retired handle.
            self.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                1, "recycle group teardown", 60.0)
            self.replacement_blocked = "probe: ProcMesh stop timed out"
        self.defunct = True
        return _Outcome(self.recycle_ok)

    def submit_sample_to(self, index, request, **_kwargs):
        # The check _dispatch_sample_bound makes before it leases a sample.
        mesh_setup.require_no_abandoned_samples(self, "dispatch another sample")
        self.events.append(("submit", self.name, index))
        return request["render_seq"]

    def collect_one(self, future, timeout_s):
        self.events.append(("collect", self.name, future))
        return {"latent": future, "host": self.name, "rank": future, "sample_s": 0.1}

    def cancel_sample(self, render_id, wait=False):
        self.events.append(("cancel", self.name))


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch):
    """No pending reads, an empty audit window and a fresh wave ledger."""
    import dgx_monarch.rdma_read_job as rdma_read_job

    monkeypatch.setattr(
        rdma_read_job, "pending_job_count_for_handle", lambda handle: 0)
    monkeypatch.setattr(telemetry_fleet, "fleet_jobs", telemetry_fleet.FleetJobs())
    consent_waiver.reset_pending_audit()
    yield
    consent_waiver.reset_pending_audit()


def _wire_fleet(monkeypatch, events, fresh):
    """Fake every Fleet seam except the heal, the wave collection and dispatch."""

    class Session:
        def bind(self, handle):
            # The refusals RenderSession.bind raises before it claims a handle.
            mesh_setup.require_not_dirty(
                handle, "start a render or residency session")
            if handle.defunct or handle.teardown_complete or handle.replacement_blocked:
                raise RetiredRenderHandleError(
                    "mesh retired while claiming a render session")
            events.append(("bind", handle.name))

        def activate(self):
            return nullcontext()

        def close(self):
            events.append(("close",))

        def track(self, _future):
            return None

    class Progress:
        port = 1234

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def ensure_live(handle):
        events.append(("ensure_live", handle.name))
        return fresh if handle.defunct else handle

    def setup(handle, *_args, **_kwargs):
        events.append(("setup", handle.name))
        return object()

    real_collect = fleet._collect_fleet_wave

    def collect_wave(handle, *args, **kwargs):
        events.append(("wave", handle.name))
        return real_collect(handle, *args, **kwargs)

    log = _Log()
    monkeypatch.setattr(abandoned_heal, "log", log)
    monkeypatch.setattr(fleet, "ensure_live", ensure_live)
    monkeypatch.setattr(fleet, "RenderSession", Session)
    monkeypatch.setattr(
        fleet, "_enforce_persisted_quarantine",
        lambda _spec, handle: events.append(("quarantine", handle.name)))
    monkeypatch.setattr(
        fleet, "_fleet_worker_policy",
        lambda *_args: fleet._FleetAuthorization({}, {}, None))
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", setup)
    monkeypatch.setattr(mesh_setup, "prepare_sample", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mesh_setup, "token_kwargs", lambda _token: {})
    monkeypatch.setattr(fleet, "ProgressReceiver", Progress)
    monkeypatch.setattr(fleet, "pack_latent", lambda value: value)
    monkeypatch.setattr(fleet, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(
        fleet, "read_latent_result",
        lambda index, *_guard: torch.full((1, 1), float(index)))
    monkeypatch.setattr(
        fleet.consent_waiver, "stamp_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fleet, "_collect_fleet_wave", collect_wave)
    monkeypatch.setattr(
        fleet, "mark_defunct_on_supervision_failure",
        lambda handle, exc: events.append(("supervision", handle.name)))
    monkeypatch.setattr(
        recycle_drain, "drop_residency_memos",
        lambda: events.append(("drop_memos",)))
    return log


def _queue(handle):
    """One public Fleet queue: two conditioning jobs on a two-GPU handle."""
    spec = ModelSpec(
        mesh=MeshSpec(handle=handle, topology_preset="uly2",
                      attention="TORCH_FLASH", sync_ulysses=True, worker_args={}),
        unet_name="model.safetensors", options={}, loras=())
    return fleet.DGXMonarchFleetKSampler().fleet(
        [spec], [object()], ["NEG"], [{"samples": torch.zeros(1, 1)}],
        [""], [42], [1], [1.0], ["euler"], ["simple"], positive=["POS-0", "POS-1"])


def _real_wedged_handle(stop_error, *, ready):
    """A real MeshHandle with two abandoned samples whose ProcMesh stop fails.

    The recycle transaction is the real one; only the stop call is faked.
    """

    class Call:
        def __init__(self, exc=None):
            self.exc = exc

        def get(self, timeout=None):
            if self.exc is not None:
                raise self.exc

    handle = object.__new__(MeshHandle)
    handle.name = "wedged"
    handle.lock = threading.RLock()
    handle.config = SimpleNamespace(worker_args={}, hosts=[], source="")
    handle.defunct = False
    handle.teardown_complete = False
    handle.replacement_blocked = None
    handle._replacement_retryable = False
    handle.setup_cleanup_state = None
    handle.setup_generation = 1
    handle.setup_key = ("ready",) if ready else None
    handle.worker_args_key = ("policy",)
    handle.sample_leases = {}
    handle.abandoned_sample_leases = {1: 2}
    handle.deferred_supervision_error = None
    handle.workers = SimpleNamespace(
        teardown_group=SimpleNamespace(call=lambda **_kwargs: Call()))
    handle.procs = SimpleNamespace(stop=lambda _reason: Call(stop_error))
    return handle


def _refusal(monkeypatch, *, heal, abandoned=1, active=0, recycle_ok=True):
    """Run a refused Fleet queue; ``heal=False`` stubs the heal as not applying.

    The ``heal=False`` run is the comparison: the current Fleet code with the
    heal reporting ``SKIPPED``, which is how every queue ran before the heal.
    Call it with ``heal=True`` first: the ``heal=False`` patch stays for the test.
    """
    events: list[tuple] = []
    wedged = _Mesh("wedged", events, abandoned=abandoned, active=active,
                   recycle_ok=recycle_ok)
    fresh = _Mesh("fresh", events)
    log = _wire_fleet(monkeypatch, events, fresh)
    if not heal:
        monkeypatch.setattr(
            fleet, "heal_abandoned_wedge_once",
            lambda *_args, **_kwargs: abandoned_heal.HealOutcome.SKIPPED)
    with pytest.raises(mesh_setup.LifecycleBusyError) as raised:
        _queue(wedged)
    return raised.value, events, log


def test_abandoned_only_wedge_heals_before_the_first_fleet_dispatch(monkeypatch):
    events: list[tuple] = []
    wedged = _Mesh("wedged", events, abandoned=2)
    fresh = _Mesh("fresh", events)
    log = _wire_fleet(monkeypatch, events, fresh)

    result, = _queue(wedged)

    assert events[:9] == [
        ("ensure_live", "wedged"),
        ("recycle", "wedged"),
        ("drop_memos",),
        ("ensure_live", "wedged"),
        ("bind", "fresh"),
        ("quarantine", "fresh"),
        ("setup", "fresh"),
        ("submit", "fresh", 0),
        ("submit", "fresh", 1),
    ]
    assert events[9] == ("wave", "fresh")
    assert sorted(event for event in events if event[0] == "collect") == [
        ("collect", "fresh", 0), ("collect", "fresh", 1)]
    assert events[-1] == ("close",)
    assert [event for event in events if event[0] in ("recycle", "drop_memos")] == [
        ("recycle", "wedged"), ("drop_memos",)]
    assert not any(event[0] == "submit" and event[1] == "wedged" for event in events)
    assert result["samples"].flatten().tolist() == [0.0, 1.0]
    assert log.lines == [(
        "info",
        "healing the abandoned-lease wedge: a crashed render left 2 abandoned "
        "sample lease(s), so Fleet dispatch would raise LifecycleBusyError; "
        "recycling the attached mesh once, then dispatching this Fleet queue on "
        "the new fleet (docs/TROUBLESHOOTING.md #87)",
    )]


def test_an_active_lease_leaves_the_fleet_refusal_unchanged(monkeypatch):
    healed, healed_events, log = _refusal(monkeypatch, heal=True, active=1)
    before, before_events, _log = _refusal(monkeypatch, heal=False, active=1)

    assert not any(event[0] in ("recycle", "drop_memos") for event in healed_events)
    assert [event for event in healed_events if event[0] == "ensure_live"] == [
        ("ensure_live", "wedged")]
    assert ("bind", "wedged") in healed_events
    assert healed_events == before_events
    assert (type(healed), str(healed)) == (type(before), str(before))
    assert "reset the Attached mesh first" in str(healed)
    assert log.lines == []


def test_a_pending_rdma_read_leaves_the_fleet_refusal_unchanged(monkeypatch):
    import dgx_monarch.rdma_read_job as rdma_read_job

    monkeypatch.setattr(
        rdma_read_job, "pending_job_count_for_handle", lambda handle: 1)
    healed, healed_events, log = _refusal(monkeypatch, heal=True)
    before, before_events, _log = _refusal(monkeypatch, heal=False)

    assert not any(event[0] in ("recycle", "drop_memos") for event in healed_events)
    assert [event for event in healed_events if event[0] == "ensure_live"] == [
        ("ensure_live", "wedged")]
    assert ("bind", "wedged") in healed_events
    assert healed_events == before_events
    assert (type(healed), str(healed)) == (type(before), str(before))
    assert "reset the Attached mesh first" in str(healed)
    assert log.lines == []


def test_an_unconfirmed_recycle_keeps_the_fleet_refusal(monkeypatch):
    healed, healed_events, log = _refusal(monkeypatch, heal=True, recycle_ok=False)
    before, _before_events, _log = _refusal(monkeypatch, heal=False, recycle_ok=False)

    # One observed attempt, no memo drop and no second ensure_live. The failed
    # stop retired the handle, so Fleet raises the dispatch refusal before its
    # session binds instead of the retired-handle error a bind would raise.
    assert healed_events == [("ensure_live", "wedged"), ("recycle", "wedged")]
    assert (type(healed), str(healed)) == (type(before), str(before))
    assert "reset the Attached mesh first" in str(healed)
    assert [level for level, _line in log.lines] == ["info", "warning"]
    assert "the original refusal stands" in log.lines[1][1]


@pytest.mark.parametrize("ready", [True, False], ids=["ready-setup", "no-setup"])
@pytest.mark.parametrize(
    ("stop_error", "status"),
    [(TimeoutError("probe stop timed out"), "proc_stop_timed_out"),
     (RuntimeError("probe stop failed"), "proc_stop_failed")],
    ids=["stop-timed-out", "stop-failed"])
def test_a_failed_stop_on_a_real_handle_keeps_the_fleet_refusal(
        monkeypatch, stop_error, status, ready):
    """The real recycle transaction retires the handle when its stop fails."""
    events: list[tuple] = []
    fresh = _Mesh("fresh", events)
    log = _wire_fleet(monkeypatch, events, fresh)
    wedged = _real_wedged_handle(stop_error, ready=ready)

    with pytest.raises(mesh_setup.LifecycleBusyError) as raised:
        _queue(wedged)

    assert wedged.defunct and wedged.replacement_blocked
    assert (wedged.setup_cleanup_state is not None) is ready
    assert str(raised.value).startswith(
        "cannot dispatch another sample: 2 abandoned sample(s)")
    assert str(raised.value).endswith("reset the Attached mesh first")
    assert events == [("ensure_live", "wedged")]
    assert [level for level, _line in log.lines] == ["info", "warning"]
    assert f"({status}: " in log.lines[1][1]


def test_the_fleet_heal_runs_at_most_once_per_queue(monkeypatch):
    """A fresh fleet that still reports abandonment is refused, never recycled again."""
    events: list[tuple] = []
    wedged = _Mesh("wedged", events, abandoned=1)
    fresh = _Mesh("fresh", events, abandoned=1)
    _wire_fleet(monkeypatch, events, fresh)

    with pytest.raises(mesh_setup.LifecycleBusyError, match="reset the Attached mesh first"):
        _queue(wedged)

    assert [event for event in events if event[0] == "recycle"] == [("recycle", "wedged")]
    assert [event for event in events if event[0] == "ensure_live"] == [
        ("ensure_live", "wedged"), ("ensure_live", "wedged")]
    assert ("bind", "fresh") in events
    assert not any(event[0] == "submit" for event in events)


def test_the_fleet_heal_reads_the_live_handle_not_the_cached_init_handle(monkeypatch):
    """A recycled Init handle keeps its abandoned count; Fleet must not heal it.

    ComfyUI keeps the Init node's spec on the retired handle, whose recycle
    answers ok again from its already-stopped branch. Healing that handle
    would drop the residency memos and log a false heal on every queue.
    """
    events: list[tuple] = []
    init = _Mesh("init", events, abandoned=1)
    init.defunct = True
    init.teardown_complete = True
    fresh = _Mesh("fresh", events)
    log = _wire_fleet(monkeypatch, events, fresh)

    for _queue_index in range(2):
        _queue(init)

    assert not any(event[0] in ("recycle", "drop_memos") for event in events)
    assert log.lines == []
    assert [event for event in events if event[0] == "ensure_live"] == [
        ("ensure_live", "init"), ("ensure_live", "init")]
    assert [event for event in events if event[0] in ("bind", "submit")] == [
        ("bind", "fresh"), ("submit", "fresh", 0), ("submit", "fresh", 1)] * 2


def test_the_render_path_keeps_its_heal_and_its_log_line(monkeypatch):
    """Both paths share one heal; the render wrapper's text is unchanged."""
    log = _Log()
    monkeypatch.setattr(abandoned_heal, "log", log)
    monkeypatch.setattr(recycle_drain, "drop_residency_memos", lambda: None)
    events: list[tuple] = []
    handle = _Mesh("wedged", events, abandoned=2)
    model = type("Model", (), {"mesh": type("Spec", (), {"handle": handle})()})()

    assert render_submit._heal_abandoned_wedge_once(
        model, mesh_setup.LifecycleBusyError("cannot dispatch")) is True

    assert events == [("recycle", "wedged")]
    assert log.lines == [(
        "info",
        "healing the abandoned-lease wedge: a crashed render left 2 abandoned "
        "sample lease(s), so dispatch raises LifecycleBusyError; recycling the "
        "attached mesh once, then retrying this submit on the new fleet "
        "(docs/TROUBLESHOOTING.md #87)",
    )]
    assert fleet.heal_abandoned_wedge_once is abandoned_heal.heal_abandoned_wedge_once
