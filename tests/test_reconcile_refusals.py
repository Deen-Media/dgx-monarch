"""The cleanup after a Worker fault when its shutdown is refused before any stop.

A supervision failure starts the reconcile (mesh_teardown.reconcile_stop) on a
thread of its own. Its shutdown can be refused before it sends a stop: a busy
lifecycle lock, a read job still rooted, a live lease. Nothing was stopped
then, so the fleet can still be stopped, and the handle must say so: a cause
of its own and a Reset that works. A lock held by a transition that then
publishes nothing is waited out, and the cleanup sends its stop itself. A
block another stop published is that stop's, and the refusal never touches it.
No test here sleeps to win a race; Events order every step.
"""
from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_safety, mesh_setup, mesh_teardown, rdma_read_job
from dgx_monarch.mesh import mark_defunct_on_supervision_failure
from dgx_monarch.mesh_recycle import RecycleStatus
from dgx_monarch.nodes.render_session import RenderSession
from dgx_monarch.rdma_read_driver import run_owned_off_loop
from topology_transition_helpers import (
    _acks,
    _dispatch_sample,
    _Endpoint,
    _Future,
    _handle,
    _ValueMesh,
    _Workers,
)


class SupervisionError(RuntimeError):
    """Stand-in whose type name carries the classifier marker."""


@pytest.fixture(autouse=True)
def _fresh_teardown_state(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))


def _short_lock(monkeypatch, refused: threading.Event, seconds: float = 0.2,
                refuse_shutdowns: int = 0):
    """Shorten every lifecycle lock wait; ``refused`` fires when a shutdown's wait runs out.

    ``refuse_shutdowns`` refuses that many shutdowns outright, as a lock some
    other transition keeps would.
    """
    real = mesh_setup.lifecycle_lock
    left = [refuse_shutdowns]

    @contextlib.contextmanager
    def short(handle, operation, timeout_s=10.0):
        if operation == "shutdown" and left[0] > 0:
            left[0] -= 1
            refused.set()
            raise mesh_setup.LifecycleBusyError(
                "cannot shutdown: another lifecycle transition still owns the mesh")
        try:
            with real(handle, operation, min(float(timeout_s), seconds)):
                yield
        except mesh_setup.LifecycleBusyError:
            if operation == "shutdown":
                refused.set()
            raise

    monkeypatch.setattr(mesh_setup, "lifecycle_lock", short)


def _join_reconcile(handle):
    thread = getattr(handle, "_supervision_reconcile_thread", None)
    if thread is not None:
        thread.join(10.0)
        assert not thread.is_alive()


def test_a_cleanup_whose_lock_wait_runs_out_stops_the_fleet_once_the_lock_frees(monkeypatch):
    """The transition that held the lock published nothing, so the cleanup still owns the stop."""
    from dgx_monarch import reconcile_running

    refused = threading.Event()
    _short_lock(monkeypatch, refused)
    stamps: list = []
    real_note = reconcile_running.note_started
    monkeypatch.setattr(reconcile_running, "note_started",
                        lambda handle: stamps.append(refused.is_set()) or real_note(handle))
    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    failure = SupervisionError("peer closed")
    token = object()
    handle.defunct = True  # what the supervision publication stores before it starts the cleanup
    handle._supervision_reconcile_token = token
    held = threading.Event()

    def hold_until_refused():
        with handle.lock:
            held.set()
            assert refused.wait(10.0)

    holder = threading.Thread(target=hold_until_refused)
    holder.start()
    assert held.wait(5.0)
    cleanup = threading.Thread(
        target=mesh_teardown.reconcile_thread_main, args=(handle, failure, token))
    cleanup.start()
    holder.join(10.0)
    cleanup.join(10.0)
    assert not cleanup.is_alive()

    assert refused.is_set()
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert handle.procs.reasons == ["dgx-monarch client detach"]
    # The "still stopping" window counts again from the stop sent after the wait.
    assert stamps == [False, True]


def test_a_cleanup_refused_before_any_stop_says_so_and_a_reset_stops_the_fleet(monkeypatch):
    refused = threading.Event()
    _short_lock(monkeypatch, refused, refuse_shutdowns=2)
    handle, _old, _new = _handle(_Workers([_Future(_acks())]))

    mark_defunct_on_supervision_failure(handle, SupervisionError("peer closed"))
    _join_reconcile(handle)

    assert handle.procs.reasons == []  # nothing was stopped
    assert handle.teardown_complete is False
    assert mesh_teardown.block_cause(handle) == mesh_teardown.TAG_NO_STOP_ISSUED
    assert handle._replacement_retryable is True
    assert "LifecycleBusyError" in handle.replacement_blocked
    assert mesh_teardown.block_cause_sentence(handle).startswith("No stop was issued")
    assert not mesh_safety.token_present(id(handle))

    outcome = handle.recycle_detailed()
    assert outcome.status is RecycleStatus.RECYCLED, outcome
    assert handle.procs.reasons == ["dgx-monarch recycle"]


def test_a_refused_cleanup_leaves_a_block_another_stop_published_alone(monkeypatch):
    """The refusal sent nothing, so it can say nothing about another stop's outcome."""
    handle = SimpleNamespace(
        lock=threading.RLock(), teardown_complete=False, defunct=True,
        replacement_blocked="TimeoutError('proc stop did not finish within 100 s')",
        _replacement_retryable=False,
        _block_cause=mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH)

    def shutdown(**_kwargs):
        raise mesh_setup.SampleResultBusyError("cannot shut down: 1 unresolved read")

    handle.shutdown = shutdown
    mesh_teardown.reconcile_stop(handle, SupervisionError("peer closed"), object())

    assert handle._block_cause == mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH
    assert handle._replacement_retryable is False
    assert handle.replacement_blocked == "TimeoutError('proc stop did not finish within 100 s')"


def test_a_refused_cleanup_writes_nothing_while_another_stop_holds_the_token(monkeypatch):
    """A stop still in flight publishes its own outcome; a retryable latch would license a second."""
    handle = SimpleNamespace(
        lock=threading.RLock(), teardown_complete=False, defunct=True,
        replacement_blocked=None, _replacement_retryable=False, _block_cause=None)

    def shutdown(**_kwargs):
        raise mesh_setup.SampleResultBusyError("cannot shut down: 1 unresolved read")

    handle.shutdown = shutdown
    mesh_safety.begin_deliberate_teardown(id(handle))
    try:
        mesh_teardown.reconcile_stop(handle, SupervisionError("peer closed"), object())
        assert handle.replacement_blocked is None
        assert handle._replacement_retryable is False
        assert handle._block_cause is None
    finally:
        mesh_safety.end_deliberate_teardown(id(handle), stop_confirmed=False)


def test_a_worker_death_during_a_read_that_outlives_its_render_leaves_a_working_reset(monkeypatch):
    """The read job resumes the deferred cleanup from its own token retirement.

    The job is still rooted at that instant, so the cleanup's shutdown refuses
    on it. Nothing was stopped: the handle says so, and once the job has gone
    a Reset stops the fleet.
    """
    monkeypatch.setattr(rdma_read_job._registry, "_JOBS", [])
    workers = _Workers([_Future(_acks())])
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    session = RenderSession(timeout_s=0)
    session.bind(handle)
    with session.activate():
        lease = _dispatch_sample(handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    entered, release = threading.Event(), threading.Event()

    def operation(_primary, _before_ack, enter, _publish):
        enter()
        entered.set()
        assert release.wait(10.0)
        return "latent"

    outcome = []

    def runner():
        try:
            outcome.append(run_owned_off_loop(lease, operation, 30.0, "dgxm-latent-read"))
        except BaseException as exc:  # the test reads what the runner ended with
            outcome.append(exc)

    thread = threading.Thread(target=runner)
    thread.start()
    assert entered.wait(10.0)
    mark_defunct_on_supervision_failure(handle, SupervisionError("peer closed"))
    assert handle.deferred_supervision_error is not None  # the read holds the lease
    mesh_setup.abandon_sample(lease)  # the render gave up on the read
    session.close()
    release.set()
    thread.join(10.0)
    assert not thread.is_alive()
    _join_reconcile(handle)
    assert rdma_read_job.pending_job_count_for_handle(handle) == 0

    assert handle.procs.reasons == []
    assert mesh_teardown.block_cause(handle) == mesh_teardown.TAG_NO_STOP_ISSUED
    assert handle._replacement_retryable is True
    assert "unresolved sample result" in handle.replacement_blocked
    reset = handle.recycle_detailed()
    assert reset.status is RecycleStatus.RECYCLED, reset
    assert handle.procs.reasons == ["dgx-monarch recycle"]


def test_a_gate_policy_restore_sends_nothing_to_a_fleet_that_died_in_the_apply(monkeypatch):
    """The restore refuses a defunct fleet at once, so the cleanup takes the lock and stops it."""
    from dgx_monarch import mesh_setup_transition

    refused = threading.Event()
    _short_lock(monkeypatch, refused)
    workers = _Workers([_Future(_acks())])
    workers.apply_worker_args = _Endpoint(
        _Future(error=SupervisionError("peer closed")), _Future(_ValueMesh([{}, {}])))
    handle, _old, _new = _handle(workers)
    # The restore wins the lock: the cleanup starts only once the restore entered.
    applies, restore_entered = [], threading.Event()
    real_apply = mesh_setup_transition.apply_worker_args
    real_reconcile = mesh_teardown.reconcile_stop

    def apply(*args, **kwargs):
        applies.append(1)
        if len(applies) == 2:
            restore_entered.set()
        return real_apply(*args, **kwargs)

    def reconcile_after_the_restore(*args, **kwargs):
        assert restore_entered.wait(10.0)
        return real_reconcile(*args, **kwargs)

    monkeypatch.setattr(mesh_setup_transition, "apply_worker_args", apply)
    monkeypatch.setattr(mesh_teardown, "reconcile_stop", reconcile_after_the_restore)

    with pytest.raises(SupervisionError), \
            handle.temporary_worker_args({}, {"slab_weights": False}, timeout_s=5.0):
        pass
    _join_reconcile(handle)

    assert len(workers.apply_worker_args.calls) == 1  # the restore sent nothing
    assert handle.teardown_complete is True
    assert handle.procs.reasons == ["dgx-monarch client detach"]


def test_setup_and_policy_changes_refuse_a_defunct_fleet_before_any_send():
    workers = _Workers([_Future(_acks())])
    workers.apply_worker_args = _Endpoint()
    handle, _old, new = _handle(workers)
    handle.defunct = True

    with pytest.raises(RuntimeError, match="defunct"):
        handle.apply_worker_args({"slab_weights": False}, timeout_s=5.0)
    with pytest.raises(RuntimeError, match="defunct"):
        handle.ensure_setup(new, "TORCH_FLASH")
    assert workers.apply_worker_args.calls == []
    assert workers.teardown_group.calls == []
    assert handle.setup_cleanup_state is None


class _GatedFuture(_Future):
    """A rollback teardown that answers only once the cleanup's lock wait ran out."""

    def __init__(self, gate: threading.Event, value=None, error=None):
        super().__init__(value, error)
        self.gate = gate

    def get(self, timeout=None):
        assert self.gate.wait(10.0)
        return super().get(timeout)


def test_a_setup_rollback_that_holds_the_lock_past_the_cleanups_wait_is_stopped_after(monkeypatch):
    """Every rank acks the rollback, so it stops nothing; the cleanup then stops the fleet."""
    refused = threading.Event()
    _short_lock(monkeypatch, refused)
    workers = _Workers(
        [_Future(_acks()), _GatedFuture(refused, _acks())],
        setup_futures=[_Future(error=SupervisionError("peer closed")), _Future({"rank": 1})])
    handle, _old, new = _handle(workers)

    with pytest.raises(SupervisionError):
        handle.ensure_setup(new, "TORCH_FLASH")
    _join_reconcile(handle)

    assert refused.is_set()
    assert handle.teardown_complete is True
    assert handle.procs.reasons == ["dgx-monarch client detach"]


def test_a_rollback_whose_own_stop_times_out_keeps_its_cause_past_the_refused_cleanup(monkeypatch):
    """A rank that died cannot ack: the rollback stops the procs itself, and that stop's cause stands."""
    refused = threading.Event()
    _short_lock(monkeypatch, refused)

    class TimingOutProcs:
        def __init__(self):
            self.reasons = []

        def stop(self, reason):
            self.reasons.append(reason)
            return _Future(error=TimeoutError("proc stop timed out"))

    workers = _Workers(
        [_Future(_acks()), _GatedFuture(refused, error=SupervisionError("peer closed"))],
        setup_futures=[_Future(error=SupervisionError("peer closed")), _Future({"rank": 1})])
    handle, _old, new = _handle(workers)
    handle.procs = TimingOutProcs()

    with pytest.raises(SupervisionError):
        handle.ensure_setup(new, "TORCH_FLASH")
    _join_reconcile(handle)

    assert handle.procs.reasons == ["dgx-monarch failed setup rollback"]
    assert mesh_teardown.block_cause(handle) == mesh_teardown.TAG_STOP_TIMEOUT_AFTER_DEATH
    assert "LifecycleBusyError" not in handle.replacement_blocked
    assert "rollback" in handle.replacement_blocked
