"""While the driver still stops a dead fleet, a refused render or Reset says so, and only that.

After a supervision failure (a lost Worker), a stall eviction, or a deliberate
eviction resumed at lease retirement, the supervision reconcile stops the old
fleet, holding the handle's lock across its proc stop. A render queued in that
window, or a Reset pressed in it, is refused as before; the verdicts and the
locks do not change. Only the words do: they say the driver is still stopping
the old fleet, within about the actor stop's cap plus the proc stop's own limit,
and to try again when that ends, never to press Reset or restart ComfyUI, which
cannot help then (reconcile_running.cleanup_running_sentence). They hold only while
the reconcile is alive and within its own wait of its start; a refusal with no
reconcile running, or one stuck past that wait, keeps its old words.
"""
from __future__ import annotations

import logging
import threading
import time

import pytest

from dgx_monarch import mesh, mesh_recycle, mesh_stop, mesh_teardown, reconcile_running
from dgx_monarch.log import get_logger
from test_mesh_creation_recovery import _ProcMesh, _spawn_returning, isolated_mesh_runtime  # noqa: F401
from topology_transition_helpers import _acks, _Future, _handle, _Workers

RUNNING = "the driver is still stopping the previous Worker fleet; that stop ends within about"


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def reconcile():
    """A reconcile thread that is alive until the test ends, as one inside its proc stop is."""
    release = threading.Event()
    thread = threading.Thread(target=release.wait, args=(30,), name="dgxm-reconcile-stop", daemon=True)
    thread.start()
    yield thread
    release.set()
    thread.join(5)


def _running(handle, reconcile_thread, started_ago_s: float = 1.0) -> None:
    """What the launch and the reconcile thread's first act leave on the handle."""
    handle._supervision_reconcile_thread = reconcile_thread
    setattr(handle, reconcile_running.RECONCILE_STARTED, time.monotonic() - started_ago_s)


def _defunct_handle(reconcile_thread=None, started_ago_s: float = 1.0):
    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    handle.defunct = True  # what the publication leaves: the verdict reads "unresolved"
    if reconcile_thread is not None:
        _running(handle, reconcile_thread, started_ago_s)
    return handle


def _says_only_wait(text: str) -> None:
    assert RUNNING in text and "try again when it ends" in text, text
    assert f"within about {mesh_stop.ACTOR_STOP_S + mesh_stop.proc_stop_bound_s():.0f} s" in text, text
    assert "Reset" not in text and "restart ComfyUI" not in text and "lost Worker" not in text, text


def _recycled(handle, lock_timeout_s: float = 10.0):
    logger, capture = get_logger("dgx_monarch.mesh_recycle"), _Records()
    logger.addHandler(capture)
    try:
        outcome = mesh_recycle.recycle_detailed_impl(handle, lambda: None, lambda: None,
                                                     lock_timeout_s=lock_timeout_s)
    finally:
        logger.removeHandler(capture)
    return outcome, capture.records


def test_a_render_queued_while_the_cleanup_runs_is_told_to_try_again(reconcile):
    handle = _defunct_handle(reconcile)
    with pytest.raises(mesh.MeshAttachError) as caught:
        mesh.ensure_live(handle)
    _says_only_wait(str(caught.value))


def test_get_mesh_on_an_unresolved_predecessor_while_its_cleanup_runs(isolated_mesh_runtime, monkeypatch,  # noqa: F811
                                                                     reconcile):
    monkeypatch.setattr(mesh, "spawn_worker_fleet", _spawn_returning(_ProcMesh()))
    monkeypatch.setattr(mesh.mesh_helpers, "start_client_lease", lambda: True)
    first = mesh.get_mesh(mode="local", gpus_per_host=1)
    first.defunct = True
    _running(first, reconcile)
    with pytest.raises(mesh.MeshAttachError) as caught:
        mesh.get_mesh(mode="local", gpus_per_host=1)
    _says_only_wait(str(caught.value))


def test_a_reset_while_the_cleanup_holds_the_lock_waits_and_logs_no_error(reconcile):
    handle = _defunct_handle(reconcile)
    held, release = threading.Event(), threading.Event()

    def hold_the_lock():  # the reconcile holds the handle's lock across its proc stop
        with handle.lock:
            held.set()
            release.wait(10)

    holder = threading.Thread(target=hold_the_lock, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        outcome, records = _recycled(handle, lock_timeout_s=0.2)
    finally:
        release.set()
        holder.join(5)
    assert outcome.status is mesh_recycle.RecycleStatus.LIFECYCLE_BUSY and outcome.retryable
    _says_only_wait(outcome.detail)
    assert [r for r in records if r.levelno >= logging.ERROR] == []
    assert any(r.levelno == logging.WARNING for r in records)


def test_a_reset_refused_by_a_prior_stop_while_the_cleanup_runs_says_wait(reconcile):
    """The Reset's other refusal: the lock is free, but a stop with no known outcome stands."""
    handle = _defunct_handle(reconcile)
    handle.replacement_blocked = "worker ProcMesh stop timed out; its outcome is unknown"
    outcome, records = _recycled(handle)
    assert outcome.status is mesh_recycle.RecycleStatus.PRIOR_TEARDOWN_UNKNOWN
    _says_only_wait(outcome.detail)
    assert [r for r in records if r.levelno >= logging.ERROR] == []
    assert any(r.levelno == logging.WARNING for r in records)


def test_a_reconcile_stuck_past_its_own_wait_gets_the_old_words(reconcile):
    """Past reconcile_join_s of its start the stop is stuck: "ends within about N s" would be untrue."""
    inside = _defunct_handle(reconcile, started_ago_s=mesh_stop.reconcile_join_s() - 5.0)
    stuck = _defunct_handle(reconcile, started_ago_s=mesh_stop.reconcile_join_s() + 5.0)
    assert mesh_teardown.cleanup_running(inside) and not mesh_teardown.cleanup_running(stuck)
    with pytest.raises(mesh.MeshAttachError) as caught:
        mesh.ensure_live(stuck)
    assert RUNNING not in str(caught.value)
    assert str(caught.value).startswith("the cached mesh cannot be used or replaced")


def test_with_no_cleanup_running_the_refusals_keep_their_words():
    handle = _defunct_handle()
    assert not mesh_teardown.cleanup_running(handle)
    with pytest.raises(mesh.MeshAttachError) as caught:
        mesh.ensure_live(handle)
    assert RUNNING not in str(caught.value)
    assert str(caught.value).startswith("the cached mesh cannot be used or replaced")
    assert mesh_teardown.unresolved_refusal(handle).startswith(
        "the previous fleet's group or process teardown is unconfirmed")


def test_the_reconcile_stamps_its_start_as_its_first_act():
    """The real reconcile thread body records when it began, which bounds the words above."""
    handle = _defunct_handle()

    def shutdown(**_kwargs):
        raise RuntimeError("stop failed")

    handle.shutdown = shutdown
    before = time.monotonic()
    mesh_teardown.reconcile_thread_main(handle, RuntimeError("lost"), object())
    started = getattr(handle, reconcile_running.RECONCILE_STARTED, None)
    assert isinstance(started, float) and before <= started <= time.monotonic()
