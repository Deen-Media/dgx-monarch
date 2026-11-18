"""The canary runner stops every session a smoke script started, however the script ends.

tests/canary/monarch_lost_rank_smoke.py starts two Worker loops, each leading
its own session so that it can kill one by process group. When the script
hangs past its limit, the runner's kill of the script's own group does not
reach those sessions, and a hang in native code runs neither the script's
``finally`` nor a signal handler; the loops and their procs then held their
ports after the canary gave up. These stand-in scripts start a session-leading
child that starts a grandchild of its own, as a loop starts its procs, and
then hang, end on their own deadline, or exit leaving the child running.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

CANARY = Path(__file__).resolve().parent / "canary"

# A loop stand-in: leads its session, starts a grandchild in its group, writes both pids.
_LOOP = (
    "import pathlib, subprocess, sys, time\n"
    "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
    "pathlib.Path(sys.argv[1]).write_text(f'{grandchild.pid}')\n"
    "time.sleep(600)\n"
)


def _script(tmp_path: Path, ending: str) -> Path:
    """A smoke stand-in that starts one recorded session, then ends as ``ending`` says."""
    script = tmp_path / "standin_smoke.py"
    script.write_text(
        "import pathlib, subprocess, sys, time\n"
        f"sys.path.insert(0, {str(CANARY)!r})\n"
        "import canary_sessions\n"
        f"out = pathlib.Path({str(tmp_path)!r})\n"
        f"child = canary_sessions.start_session([sys.executable, '-c', {_LOOP!r}, str(out / 'grandchild')],\n"
        "                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "(out / 'child').write_text(f'{child.pid}')\n"
        "while not (out / 'grandchild').exists():\n"
        "    time.sleep(0.05)\n"
        + {"hang": "time.sleep(600)\n",
           "deadline": ("canary_sessions.arm_deadline(lambda: canary_sessions.stop_groups([child.pid]))\n"
                        "time.sleep(600)\n"),
           "exit": "sys.exit(0)\n"}[ending])
    return script


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(CANARY))
    spec = importlib.util.spec_from_file_location("monarch_surface_canary_under_test",
                                                  CANARY / "monarch_surface_canary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gone(pid: int, wait_s: float = 5.0) -> bool:
    """No such process, or only its zombie, within ``wait_s``."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        if state == "Z":
            return True
        time.sleep(0.1)
    return False


def _pids(tmp_path: Path) -> list[int]:
    return [int((tmp_path / name).read_text()) for name in ("child", "grandchild")]


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_a_hung_script_leaves_no_session_it_started(runner, tmp_path):
    """The runner kills the script's group at its limit, then every group it recorded."""
    started = time.monotonic()
    returncode, left = runner._run_smoke(_script(tmp_path, "hang"), tmp_path / "out.json",
                                         dict(os.environ), timeout_s=4.0, deadline_s=60.0)
    assert returncode == "timeout" and left == []
    assert time.monotonic() - started < 20.0
    child, grandchild = _pids(tmp_path)
    assert _gone(child) and _gone(grandchild), (child, grandchild)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_the_scripts_own_deadline_ends_it_before_the_runners_limit(runner, tmp_path):
    started = time.monotonic()
    returncode, left = runner._run_smoke(_script(tmp_path, "deadline"), tmp_path / "out.json",
                                         dict(os.environ), timeout_s=30.0, deadline_s=1.5)
    elapsed = time.monotonic() - started
    assert returncode == runner.canary_sessions.DEADLINE_EXIT and left == []
    assert elapsed < 15.0, elapsed  # its own cleanup, long before the 30 s hard kill
    child, grandchild = _pids(tmp_path)
    assert _gone(child) and _gone(grandchild), (child, grandchild)


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs /proc")
def test_a_session_left_by_a_script_that_exited_is_stopped_too(runner, tmp_path):
    returncode, left = runner._run_smoke(_script(tmp_path, "exit"), tmp_path / "out.json",
                                         dict(os.environ), timeout_s=30.0)
    assert returncode == 0 and left == []
    child, grandchild = _pids(tmp_path)
    assert _gone(child) and _gone(grandchild), (child, grandchild)


def test_the_runner_gives_each_script_a_deadline_before_its_limit(runner):
    assert runner.DEADLINE_MARGIN_S == 30.0
    assert 240 - runner.DEADLINE_MARGIN_S > 0 and 300 - runner.DEADLINE_MARGIN_S > 0


def test_nothing_recorded_means_nothing_to_stop(tmp_path):
    sys.path.insert(0, str(CANARY))
    try:
        import canary_sessions
    finally:
        sys.path.remove(str(CANARY))
    assert canary_sessions.recorded(tmp_path / "absent") == []
    assert canary_sessions.stop_groups([]) == []
