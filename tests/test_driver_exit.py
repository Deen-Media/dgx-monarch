"""Once the driver's interpreter has begun to exit, lost-fleet work stands down (driver_exit).

Monarch's own exit step stops every mesh the driver owns, so a call still in
flight fails at a normal exit. The flag's exit hook, registered at the first
mesh build after Monarch's own, runs first and logs the one INFO line from the
main thread; from then on the eviction, the reconcile and the fault hook do
nothing. The hook sets nothing while the main thread still runs. Without the
flag the same calls evict, stop the dead fleet and report as they always did.
Here the hook runs by hand with the main thread taken as ended;
test_exit_clean.py proves the order with real processes (a call in flight at
exit, Ctrl+C, a call holding its actor, and a Worker killed while the driver is
not exiting).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import test_worker_loss as captured
from dgx_monarch import driver_exit, mesh_helpers
from dgx_monarch.mesh import mark_defunct_on_supervision_failure
from test_lost_worker_words import LIVE_NAME, _bring_up, _errors, _Fault
from test_lost_worker_words import hook as fault_hook  # noqa: F401  # a fixture: the real fault hook

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(autouse=True)
def _no_exit():
    driver_exit.reset()
    yield
    driver_exit.reset()


def _exit_begins(monkeypatch) -> None:
    """The exit hook as Python runs it: after the main thread has ended."""
    monkeypatch.setattr(driver_exit, "_main_thread_alive", lambda: False)
    driver_exit._begin()


class _Handle:
    def __init__(self):
        self.defunct = False
        self.shutdown_called = False

    def shutdown(self, timeout_s: float = 60.0, **kwargs) -> None:
        self.shutdown_called = True


class SupervisionError(Exception):
    """Stand-in whose type name carries the classifier marker (test_mesh_evict.py)."""


def _notices(records):
    return [r for r in records if r.getMessage() == driver_exit.NOTICE]


def test_once_exit_began_the_hook_says_one_info_line_and_sends_no_toast(fault_hook, monkeypatch):  # noqa: F811
    fault_hook.meshes["cluster"] = _bring_up(LIVE_NAME)
    _exit_begins(monkeypatch)
    fault_hook.fault(_Fault(captured.KILL_FAULT_080))
    fault_hook.fault(_Fault(captured.KILL_FAULT_080.replace("crashed_ranks=[0]", "crashed_ranks=[1]")))
    assert _errors(fault_hook.records) == []
    assert fault_hook.toasts == []
    notices = _notices(fault_hook.records)
    assert len(notices) == 1 and notices[0].levelno == logging.INFO


def test_while_not_exiting_the_hook_reports_the_fault_as_before(fault_hook):  # noqa: F811
    fault_hook.meshes["cluster"] = _bring_up(LIVE_NAME)
    fault_hook.fault(_Fault(captured.KILL_FAULT_080))
    errors = _errors(fault_hook.records)
    assert len(errors) == 1 and errors[0].startswith("cluster fault: The Worker on the sibling")
    assert len(fault_hook.toasts) == 1
    assert _notices(fault_hook.records) == []


def test_once_exit_began_an_eviction_publishes_nothing_and_stops_nothing(monkeypatch):
    handle = _Handle()
    _exit_begins(monkeypatch)
    mark_defunct_on_supervision_failure(handle, SupervisionError("worker 1 proc exited"))
    assert not handle.defunct
    assert not handle.shutdown_called
    assert getattr(handle, "_supervision_reconcile_thread", None) is None


def test_while_not_exiting_an_eviction_marks_the_fleet_and_stops_it():
    handle = _Handle()
    mark_defunct_on_supervision_failure(handle, SupervisionError("worker 1 proc exited"))
    assert handle.defunct
    assert handle.shutdown_called


def test_a_reconcile_that_starts_after_exit_began_stops_nothing(monkeypatch):
    handle = _Handle()
    _exit_begins(monkeypatch)
    assert mesh_helpers._reconcile_thread_main(handle, RuntimeError("lost"), object()) is None
    assert not handle.shutdown_called


def test_the_notice_is_the_exit_hooks_one_line_and_no_thread_that_stands_down_logs(caplog, monkeypatch):
    calls: list[int] = []
    quiet = driver_exit.skipped_at_exit(lambda: calls.append(1))
    with caplog.at_level(logging.INFO, logger="dgx_monarch.mesh"):
        _exit_begins(monkeypatch)
        driver_exit._begin()  # a second run of the hook adds nothing
        threads = [threading.Thread(target=quiet) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
    assert calls == []
    notices = [r for r in caplog.records if r.getMessage() == driver_exit.NOTICE]
    assert len(notices) == 1 and notices[0].threadName == threading.current_thread().name


def test_the_hook_sets_nothing_while_the_main_thread_still_runs(caplog):
    """A stray run of the exit hooks while ComfyUI runs must not leave later lost Workers unevicted."""
    assert driver_exit._main_thread_alive()
    with caplog.at_level(logging.INFO, logger="dgx_monarch.mesh"):
        driver_exit._begin()
    assert not driver_exit.exiting()
    assert [r for r in caplog.records if r.getMessage() == driver_exit.NOTICE] == []
    handle = _Handle()
    mark_defunct_on_supervision_failure(handle, SupervisionError("worker 1 proc exited"))
    assert handle.defunct and handle.shutdown_called


class _Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record):
        self.records.append(record)


def _reconcile_cut_short(monkeypatch, exit_begins: bool) -> list[logging.LogRecord]:
    """The real reconcile_stop, on the reconcile's own thread, inside a stop that then fails.

    The stop blocks until the test lets it fail, as one does when Monarch's exit
    step closes the client under it; the exit begins (or not) while it blocks.
    """
    from dgx_monarch import mesh_teardown
    from dgx_monarch.log import get_logger

    inside, release = threading.Event(), threading.Event()

    class _Stopping:
        defunct, teardown_complete, replacement_blocked = True, False, None
        lock = threading.RLock()

        def shutdown(self, **_kwargs):
            inside.set()
            release.wait(10)
            raise ValueError("error stopping mesh: owner terminated: stopped: client stopped")

    logger = get_logger("dgx_monarch.mesh")
    capture = _Records()
    logger.addHandler(capture)
    try:
        thread = threading.Thread(
            target=mesh_teardown.reconcile_stop,
            args=(_Stopping(), RuntimeError("the lost Worker"), object()),
            name=driver_exit.RECONCILE_THREAD, daemon=True)
        thread.start()
        assert inside.wait(10)
        if exit_begins:
            _exit_begins(monkeypatch)
        release.set()
        thread.join(10)
    finally:
        logger.removeHandler(capture)
    return [r for r in capture.records if r.threadName == driver_exit.RECONCILE_THREAD]


def test_a_reconcile_inside_its_stop_when_exit_begins_logs_none_of_its_lines(monkeypatch):
    assert _reconcile_cut_short(monkeypatch, exit_begins=True) == []


def test_a_reconcile_whose_stop_fails_while_not_exiting_reports_as_before(monkeypatch):
    lines = [r.getMessage() for r in _reconcile_cut_short(monkeypatch, exit_begins=False)
             if r.levelno == logging.ERROR]
    assert lines[0].startswith("the defunct mesh did not shut down; replacement is blocked")
    assert lines[1].startswith("worker fleet failure (")


def test_once_exit_began_only_the_reconcile_thread_is_quiet(monkeypatch):
    from dgx_monarch import mesh_liveness
    from dgx_monarch.log import get_logger

    assert mesh_liveness._RECONCILE_THREAD == driver_exit.RECONCILE_THREAD  # the thread's one name
    logger, capture = get_logger("dgx_monarch.mesh"), _Records()
    logger.addHandler(capture)
    try:
        _exit_begins(monkeypatch)
        logger.error("a line from another thread at exit")
    finally:
        logger.removeHandler(capture)
    assert [r.getMessage() for r in capture.records][-1] == "a line from another thread at exit"


def test_the_exit_wrappers_sit_on_the_functions_the_eviction_and_the_reconcile_run():
    """A rename in mesh_helpers or mesh_teardown must not leave a path unwrapped."""
    from dgx_monarch import mesh_teardown

    assert mesh_helpers._reconcile_thread_main.__wrapped__ is mesh_teardown.reconcile_thread_main
    assert mesh_helpers._publish_defunct.__wrapped__.__name__ == "_publish_defunct"
    import inspect

    source = inspect.getsource(mesh_helpers)
    assert "target=_reconcile_thread_main" in source and "name=driver_exit.RECONCILE_THREAD" in source
    assert source.count("_publish_defunct(handle, exc, text") == 2  # both evictions go through the wrapper


def test_the_hook_registers_once_and_runs_before_a_hook_registered_earlier(tmp_path):
    """The order Monarch's own exit step relies on: last registered, first run."""
    script = tmp_path / "order.py"
    script.write_text(
        "import atexit, json\n"
        "from dgx_monarch import driver_exit\n"
        "seen = {}\n"
        "atexit.register(lambda: print(json.dumps({'earlier_hook_saw_exit': driver_exit.exiting(),"
        " 'added': seen['added']})))\n"
        "before = atexit._ncallbacks()\n"
        "driver_exit.install(); driver_exit.install()\n"
        "seen['added'] = atexit._ncallbacks() - before\n"
        "print(json.dumps({'exiting_before_exit': driver_exit.exiting()}))\n")
    environment = dict(os.environ, HOME=str(tmp_path),
                       PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=60, env=environment)
    assert proc.returncode == 0, proc.stderr[-2000:]
    first, last = (json.loads(line) for line in proc.stdout.strip().splitlines())
    assert first == {"exiting_before_exit": False}
    assert last == {"earlier_hook_saw_exit": True, "added": 1}
