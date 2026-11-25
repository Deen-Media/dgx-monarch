"""What a stop whose Future completed without error leaves behind, on the real runtime.

A cluster bring-up whose rollback stop completed without error records no
poison and asks no host before the next fleet (mesh_factory.left_nothing_running):
the signal a normal teardown trusts too. Monarch's ProcMesh.stop returns Ok once
every rank is terminating, which includes failed and timed out, so this pins
the property the lift rests on with a rank that cannot answer: the stop's
Future is still pending, or it completed and that rank's process no longer
runs. A completed Future beside a running rank is the failure. CPU only; the
probe runs in its own interpreter and its own session (stop_future_probe.py),
and every process of that session, recorded or not, is killed here if it
outlives the probe.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dgx_monarch.cli import proc_identity as pi

SRC = Path(__file__).resolve().parents[1] / "src"
PROBE = Path(__file__).resolve().parent / "stop_future_probe.py"
GRACE_S = 15.0  # how long the probe's processes may take to leave after it exits


def _session_members(session: int, born: int) -> set[tuple[int, int]]:
    """Every process of the session the probe leads, by pid and start time.

    The probe starts its own session, and every rank and helper it spawns
    stays in it, so this finds them even when the probe died before it wrote
    the ranks file. Only this uid's processes started no earlier than the
    probe can match.
    """
    members = set()
    for pid in pi.iter_pids():
        stat = pi.read_stat(pid)
        if (stat is not None and stat[1] >= born and pi.read_session(pid) == session
                and pi.read_uid(pid) == os.getuid()):
            members.add((pid, stat[1]))
    return members


def _kill_left_ranks(ranks_file: Path, session: int, born: int) -> list[int]:
    """Clean up the probe's session and every process it recorded; return the ranks that had to be killed.

    Its children (its local host agent among them) leave once it exits;
    whatever still runs after GRACE_S is killed, identity-checked by start
    time. A rank killed here means the probe left it running.
    """
    try:
        recorded = json.loads(ranks_file.read_text())
    except (OSError, ValueError):
        recorded = {}
    ranks = {(int(pid), int(start)) for pid, start in recorded.get("ranks", ())}
    everyone = ranks | {(int(pid), int(start)) for pid, start in recorded.get("children", ())}
    everyone |= _session_members(session, born)

    def running(pid: int, start: int) -> bool:
        return pi.alive(pid, start) and pi.read_state(pid) != "Z"

    deadline = time.monotonic() + GRACE_S
    while any(running(*entry) for entry in everyone) and time.monotonic() < deadline:
        time.sleep(0.1)
    killed = [entry for entry in everyone | _session_members(session, born) if running(*entry)]
    for pid, _start in killed:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return sorted(pid for pid, start in killed if (pid, start) in ranks)


def _run_probe(tmp_path: Path) -> dict:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    ranks_file = tmp_path / "ranks.json"
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MONARCH_", "HYPERACTOR_", "DGXM_MONARCH_POLICY"))}
    environment.update(HOME=str(tmp_path), XDG_RUNTIME_DIR=str(runtime_dir), CUDA_VISIBLE_DEVICES="",
                       PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    probe = subprocess.Popen([sys.executable, str(PROBE), str(ranks_file)], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=environment, start_new_session=True)
    born = (pi.read_stat(probe.pid) or (0, 0))[1]  # the session id is the probe's own pid
    stdout = stderr = ""
    try:
        stdout, stderr = probe.communicate(timeout=300)
    finally:
        if probe.poll() is None:
            probe.kill()
            probe.communicate(timeout=30)
        left = _kill_left_ranks(ranks_file, probe.pid, born)
    assert probe.returncode == 0, (
        f"probe crashed rc={probe.returncode}\nstdout: {stdout[-2000:]}\nstderr: {stderr[-2000:]}")
    assert left == [], f"ranks outlived the probe and were killed here: {left}"
    return json.loads(stdout.strip().splitlines()[-1])


def test_a_stop_that_completes_without_error_leaves_no_rank_running(tmp_path):
    pytest.importorskip("monarch.actor")
    result = _run_probe(tmp_path)
    print("stop contract:", json.dumps(result))  # which branch the runtime took, for the log

    assert result["native_readback"] == "verified", result
    assert result["ranks"] == 2 and result["frozen_state_before_stop"] == "T", result
    while_frozen = result["while_frozen"]
    if while_frozen != "pending" and while_frozen["outcome"] == "ok":
        # Completed while the rank could not answer: the host agent must have ended it.
        assert while_frozen["frozen_state_at_end"] in ("gone", "Z"), result
    final = result["final"]
    assert final != "pending", result
    if final["outcome"] == "ok":
        assert final["frozen_state_at_end"] in ("gone", "Z"), result
    assert result["left_running"] == [], result
