"""The moment a render's call fails on a lost Worker, the log says so once, in plain words.

The driver stops the dead fleet once no render holds it, and the render that
retires the last lease waits for that stop before its error shows (on loopback
98.5 s after the loss of a box beside a live one, 10 s of it detection).
worker_box.warn_lost_once, called by mesh_lease.collect_sample right after the
words are recorded, says at once what happened and how long a waiting render's
error can take. One line per fleet: in a two-deep Pipeline both renders'
collects fail on the same loss, render 1's error at once and render 2's after
the stop.

These run a real MeshHandle's sample lease, collect_sample and
PendingRender.result under the node layer, with Monarch's own SupervisionError
built from a captured text, as tests/test_lost_worker_words.py does.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import test_worker_loss as captured
from dgx_monarch import mesh_setup, mesh_stop, reconcile_running, worker_loss
from dgx_monarch.log import get_logger
from dgx_monarch.nodes import refusal_lead
from test_lost_worker_words import _supervision_error

WORDS = "The Worker for GPU 1 on this box was stopped by signal 9 (SIGKILL)"


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def warnings_seen():
    logger, capture = get_logger("dgx_monarch.worker_box"), _Records()
    logger.addHandler(capture)
    yield capture.records
    logger.removeHandler(capture)


def _renders(error_texts: list[str]):
    """A real handle with one dispatched render per text, each of whose calls fails with it."""
    import torch

    from dgx_monarch.mesh_session import RenderSession, claim_render_session
    from dgx_monarch.nodes import render_result, render_submit
    from topology_transition_helpers import _acks, _dispatch_sample, _Future, _handle, _Workers

    handle, old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    session, _owned = claim_render_session(handle, RenderSession())  # a Pipeline's one session
    pendings = []
    for index, text in enumerate(error_texts):
        error = _supervision_error(text)
        with session.activate():
            submission = _dispatch_sample(handle, lambda error=error: _Future(error=error), setup_token=token)
            session.track(submission)
        pendings.append(render_submit.PendingRender(
            handle, submission, SimpleNamespace(__exit__=lambda *_args: None), old,
            {"samples": torch.zeros((1, 4, 1, 1))}, 10.0, f"render-{index}", render_result._finish_render))

    def collect(pending):
        with session.activate():
            return refusal_lead.lead_with_refusal(pending.result)()

    return handle, pendings, collect


def _lost_lines(records):
    return [r.getMessage() for r in records if r.levelno == logging.WARNING and WORDS in r.getMessage()]


def test_a_lost_worker_is_logged_at_the_failure_with_the_stops_own_limit(warnings_seen):
    handle, [pending], collect = _renders([captured.KILL_ERROR])
    with pytest.raises(worker_loss.WorkerLostError):
        collect(pending)
    [line] = _lost_lines(warnings_seen)
    assert line.startswith(WORDS + "; the kernel's out-of-memory killer sends signal 9"), line
    # The actor stop's cap plus the proc stop's own limit, read from the stop code.
    bound = mesh_stop.ACTOR_STOP_S + mesh_stop.proc_stop_bound_s()
    assert reconcile_running.lost_fleet_stop_s() == bound
    assert line.endswith("The driver stops that fleet once no render holds it; a render still waiting "
                         f"on it gets its error when that stop ends, within about {bound:.0f} s."), line
    assert handle.defunct


def test_two_renders_failing_on_one_loss_log_it_once(warnings_seen):
    """A two-deep Pipeline: render 1's error is early (render 2 still holds a lease), render 2's follows the stop."""
    handle, pendings, collect = _renders([captured.KILL_ERROR, captured.KILL_ERROR])
    for pending in pendings:
        with pytest.raises(worker_loss.WorkerLostError):
            collect(pending)
    assert len(_lost_lines(warnings_seen)) == 1
    assert handle.defunct


@pytest.mark.parametrize("exiting", [False, True], ids=["running", "exiting"])
def test_at_exit_a_stopped_call_is_no_lost_worker(warnings_seen, monkeypatch, exiting):
    """Monarch's exit step stops a render's call, which then reads "monitor ... stopped before
    reporting a failure", one of the lost-contact texts. While the driver runs that is a lost
    Worker; once it exits, the collect words nothing, warns nothing and flags nothing."""
    from dgx_monarch import driver_exit
    from dgx_monarch.worker_box import LOSS_WARNED, LOST_WORKER

    handle, [pending], collect = _renders([captured.RESTART_ERROR_080])
    if exiting:
        monkeypatch.setattr(driver_exit, "_main_thread_alive", lambda: False)
        driver_exit._begin()
    try:
        with pytest.raises(Exception) as caught:  # the type is what this test reads
            collect(pending)
    finally:
        driver_exit.reset()
    lost = [r for r in warnings_seen if r.levelno == logging.WARNING and "lost contact" in r.getMessage()]
    if exiting:
        assert lost == [] and not getattr(handle, LOSS_WARNED, False)
        assert type(caught.value).__name__ == "SupervisionError", type(caught.value)  # as Monarch wrote it
        assert getattr(caught.value, LOST_WORKER, None) is None
    else:
        assert len(lost) == 1 and getattr(handle, LOSS_WARNED, False) is True
        assert isinstance(caught.value, worker_loss.WorkerLostError)


@pytest.mark.parametrize("text", [captured.RECYCLE_ERROR_070, captured.DRAINING_ERROR_080])
def test_a_deliberate_stops_failure_logs_no_lost_worker(warnings_seen, text):
    from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError

    _handle, [pending], collect = _renders([text])
    with pytest.raises(SupervisionError):  # left as Monarch wrote it (test_lost_worker_words.py)
        collect(pending)
    assert [r for r in warnings_seen if r.levelno >= logging.WARNING] == []
