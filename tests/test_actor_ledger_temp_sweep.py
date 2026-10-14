"""Temporary files that dead actor ledger writers leave behind (issue #36).

``write_ledger`` writes ``<ledger>.<pid>.<starttime>.<thread ident>.tmp`` and
then renames it over the ledger. A writer killed between the two steps
strands the file, and each later write removes the stranded files whose
writer has exited. Earlier releases wrote ``<ledger>.<pid>.<thread id>.tmp``,
which the sweep still judges by its pid alone. Every test works in
``tmp_path``. The real ``/proc`` serves where a test needs a live writer (the
test itself, with its real start time) or a dead one (a pid above
``pid_max``, which no process can hold). A synthetic one serves where the
tracker's and the pruner's ``proc_root`` must reach the sweep, and where the
writer cannot read its own stat.
"""
from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from dgx_monarch.cli import actor_ledger, proc_identity

_BOOT = "11111111-2222-3333-4444-555555555555"
_LEDGER = "actor-procs.json"


def _dead_pid() -> int:
    with open("/proc/sys/kernel/pid_max", encoding="utf-8") as handle:
        return int(handle.read()) + 1


def _own_starttime() -> int:
    stat = proc_identity.read_stat(os.getpid())
    assert stat is not None
    return stat[1]


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def test_the_next_write_removes_dead_temps_and_keeps_live_ones(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    path = state / _LEDGER
    path.write_text(json.dumps(actor_ledger.empty_ledger(_BOOT)))
    dead = _dead_pid()
    stranded = [state / f"{_LEDGER}.{dead}.1.tmp",
                state / f"{_LEDGER}.{dead}.281472826552608.tmp",
                state / f"{_LEDGER}.{dead}.4242.281472826552608.tmp"]
    for leftover in stranded:
        leftover.write_text("{}")  # fresh: a dead pid needs no grace period
    # The test's own process is alive, so its files stay in both shapes: it
    # may still rename them. The old shape is judged by the pid alone.
    live = [state / f"{_LEDGER}.{os.getpid()}.7.tmp",
            state / f"{_LEDGER}.{os.getpid()}.{_own_starttime()}.7.tmp"]
    for in_flight in live:
        in_flight.write_text("in flight")
    wide = "".join(chr(0x0660 + int(digit)) for digit in str(dead))
    strangers = [state / name for name in (
        "worker-loop.log",
        f"{_LEDGER}.bak",
        f"other.json.{dead}.1.tmp",
        f"{_LEDGER}.{dead}.tmp",
        f"{_LEDGER}.{dead}.x.tmp",
        f"{_LEDGER}.{dead}.1.tmp.bak",
        f"{_LEDGER}.{dead}.4242.1.tmp.bak",
        f"{_LEDGER}.{dead}.4242.1.2.tmp",
        f"{_LEDGER}.{dead}.nostat.1.tmp",
        f"{_LEDGER}.{wide}.1.tmp",  # Arabic-Indic digits, not ASCII ones
    )]
    for stranger in strangers:
        stranger.write_text("not ours")

    fresh = dict(actor_ledger.empty_ledger(_BOOT), written_at=5.0)
    lines = _Lines()
    level = actor_ledger.log.level
    actor_ledger.log.addHandler(lines)
    actor_ledger.log.setLevel(logging.DEBUG)
    try:
        assert actor_ledger.write_ledger(fresh, str(path)) is True
    finally:
        actor_ledger.log.removeHandler(lines)
        actor_ledger.log.setLevel(level)

    assert not any(leftover.exists() for leftover in stranded)
    assert all(in_flight.read_text() == "in flight" for in_flight in live)
    assert all(stranger.read_text() == "not ours" for stranger in strangers)
    assert json.loads(path.read_text()) == fresh
    assert lines.lines == [f"actor ledger: removed 3 dead temporary file(s) beside {path}"]


def test_a_new_shape_temp_is_judged_by_its_writers_pid_and_start_time(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    own, started = os.getpid(), _own_starttime()
    # This process wrote it and may still rename it.
    writer = state / f"{_LEDGER}.{own}.{started}.1.tmp"
    # A writer with this pid but another start time is gone: this process
    # took its pid later and cannot have written the file.
    reused = state / f"{_LEDGER}.{own}.{started + 1}.1.tmp"
    # No process holds this pid at all.
    dead = state / f"{_LEDGER}.{_dead_pid()}.{started}.1.tmp"
    for leftover in (writer, reused, dead):
        leftover.write_text("{}")

    assert actor_ledger.sweep_dead_temps(str(state / _LEDGER)) == 2

    assert writer.exists()
    assert not reused.exists()
    assert not dead.exists()


def test_the_sweep_ignores_file_times(tmp_path):
    # Greptile's case on PR #38: a clock step made a live writer's file look
    # older than the process that holds its pid. Here each file last changed
    # an hour before this process started, and both stay.
    state = tmp_path / "state"
    state.mkdir()
    started = _own_starttime()
    began = time.time() - proc_identity.age_s(started, proc_identity.uptime_s())
    live = [state / f"{_LEDGER}.{os.getpid()}.{started}.1.tmp",
            state / f"{_LEDGER}.{os.getpid()}.1.tmp"]
    for in_flight in live:
        in_flight.write_text("in flight")
        os.utime(in_flight, (began - 3600.0, began - 3600.0))

    path = state / _LEDGER
    assert actor_ledger.write_ledger(actor_ledger.empty_ledger(_BOOT), str(path)) is True

    assert all(in_flight.read_text() == "in flight" for in_flight in live)
    assert json.loads(path.read_text())["boot_id"] == _BOOT


_KILLED_MID_WRITE = """
import os, signal, sys, threading
from dgx_monarch.cli import actor_ledger, proc_identity

def _killed(_source, _target):
    starttime = proc_identity.read_stat(os.getpid())[1]
    print(starttime, threading.get_ident(), flush=True)
    os.kill(os.getpid(), signal.SIGKILL)

actor_ledger.os.replace = _killed
actor_ledger.write_ledger(actor_ledger.empty_ledger("child"), sys.argv[1])
"""


def test_a_kill_between_write_and_rename_leaves_a_temp_the_next_write_removes(
        tmp_path):
    path = tmp_path / "state" / _LEDGER
    source = str(Path(actor_ledger.__file__).resolve().parents[2])
    inherited = os.environ.get("PYTHONPATH")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        part for part in (source, inherited) if part))
    with subprocess.Popen([sys.executable, "-c", _KILLED_MID_WRITE, str(path)],
                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True) as child:
        printed, errors = child.communicate(timeout=120)
    assert child.returncode == -signal.SIGKILL, errors

    starttime, ident = printed.split()
    stranded = list(path.parent.glob(f"{_LEDGER}.*.tmp"))
    assert [leftover.name for leftover in stranded] == [
        f"{_LEDGER}.{child.pid}.{starttime}.{ident}.tmp"]
    # The write was whole and synced; only the rename never ran.
    assert json.loads(stranded[0].read_text())["boot_id"] == "child"
    assert not path.exists()

    assert actor_ledger.write_ledger(actor_ledger.empty_ledger(_BOOT), str(path)) is True
    assert not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text())["boot_id"] == _BOOT


def test_a_failed_rename_leaves_no_temp_behind(tmp_path, monkeypatch):
    # The writer stays alive after a failed rename, so the sweep would keep
    # its file: the writer must remove it itself.
    def _refused(_source, _target):
        raise OSError("rename refused")

    monkeypatch.setattr(actor_ledger.os, "replace", _refused)
    path = tmp_path / _LEDGER
    assert actor_ledger.write_ledger(actor_ledger.empty_ledger(_BOOT), str(path)) is False
    assert not list(tmp_path.glob("*.tmp"))
    assert not path.exists()


def _plant_stat(root: Path, pid: int, *, starttime: int) -> None:
    directory = root / str(pid)
    directory.mkdir(parents=True)
    filler = " ".join(["0"] * 17)
    (directory / "stat").write_text(f"{pid} (python) S 1 {filler} {starttime} 0 0 0\n")


def _synthetic_root(tmp_path: Path) -> tuple[Path, int]:
    """A proc root that lacks this process and holds a pid the real one cannot."""
    proc_root = tmp_path / "proc"
    (proc_root / "sys" / "kernel" / "random").mkdir(parents=True)
    (proc_root / "sys" / "kernel" / "random" / "boot_id").write_text(f"{_BOOT}\n")
    beyond = _dead_pid()  # no process holds it in the real /proc
    _plant_stat(proc_root, beyond, starttime=400)
    return proc_root, beyond


def test_a_writer_that_cannot_read_its_own_stat_keeps_its_temp_from_every_sweep(
        tmp_path, monkeypatch):
    # Under a root without the writer's stat, either digit shape would read
    # dead there, so the writer uses a name no sweep matches.
    proc_root, _ = _synthetic_root(tmp_path)
    path = tmp_path / "state" / _LEDGER
    real_replace = os.replace
    renamed: list[str] = []

    def _swept_first(source, target):
        renamed.append(os.path.basename(source))
        assert actor_ledger.sweep_dead_temps(target, proc_root=str(proc_root)) == 0
        assert actor_ledger.sweep_dead_temps(target) == 0
        assert os.path.exists(source)
        real_replace(source, target)

    monkeypatch.setattr(actor_ledger.os, "replace", _swept_first)
    assert actor_ledger.write_ledger(actor_ledger.empty_ledger(_BOOT), str(path),
                                     proc_root=str(proc_root)) is True

    assert len(renamed) == 1
    assert re.fullmatch(rf"{re.escape(_LEDGER)}\.{os.getpid()}\.nostat\.\d+\.tmp",
                        renamed[0], re.ASCII)
    assert json.loads(path.read_text())["boot_id"] == _BOOT
    assert not list(path.parent.glob("*.tmp"))


def test_the_tracker_tick_judges_temps_against_its_own_proc_root(tmp_path):
    # Each writer reads alive in one root and dead in the other on any host,
    # so a sweep that falls back to the real /proc fails one of the two checks.
    proc_root, beyond = _synthetic_root(tmp_path)
    _plant_stat(proc_root, 100, starttime=500)  # the loop itself
    state = tmp_path / "state"
    state.mkdir()
    # This process with its real start time, absent from the synthetic root.
    gone = state / f"{_LEDGER}.{os.getpid()}.{_own_starttime()}.1.tmp"
    gone.write_text("{}")
    held = state / f"{_LEDGER}.{beyond}.400.1.tmp"
    held.write_text("{}")

    actor_ledger.tick("tcp://10.0.0.1:26600", loop_pid=100, proc_root=str(proc_root),
                      path=str(state / _LEDGER), now=1000.0)

    assert not gone.exists()
    assert held.exists()
    assert (state / _LEDGER).exists()
    assert sorted(leftover.name for leftover in state.glob("*.tmp")) == [held.name]


def test_the_pruner_judges_temps_against_its_own_proc_root(tmp_path):
    # The tick test's two writers in the older shape, judged by pid alone: a
    # sweep that fell back to the real /proc fails one of the two checks.
    proc_root, beyond = _synthetic_root(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    path = state / _LEDGER
    ledger = actor_ledger.empty_ledger(_BOOT)
    # One row for a process the synthetic root does not hold, so the pruner
    # has a row to drop and writes.
    ledger["actors"] = [{
        "pid": 300, "starttime": 1, "address": "tcp://10.0.0.1:26600",
        "loop_pid": 100, "loop_starttime": 500, "marker": "cmdline",
        "first_seen": 1.0, "last_seen": 2.0, "orphan_since": None,
    }]
    path.write_text(json.dumps(ledger))
    gone = state / f"{_LEDGER}.{os.getpid()}.1.tmp"  # absent from the synthetic root
    gone.write_text("{}")
    held = state / f"{_LEDGER}.{beyond}.1.tmp"
    held.write_text("{}")

    actor_ledger.prune_dead_rows(str(path), proc_root=str(proc_root))

    assert json.loads(path.read_text())["actors"] == []
    assert not gone.exists()
    assert held.exists()
    assert sorted(leftover.name for leftover in state.glob("*.tmp")) == [held.name]


def test_a_write_that_raises_still_sweeps(tmp_path):
    path = tmp_path / _LEDGER
    dead = _dead_pid()
    stranded = [tmp_path / f"{_LEDGER}.{dead}.1.tmp",
                tmp_path / f"{_LEDGER}.{dead}.4242.1.tmp"]
    for leftover in stranded:
        leftover.write_text("{}")
    with pytest.raises(TypeError):
        actor_ledger.write_ledger({"x": object()}, str(path))
    assert not any(leftover.exists() for leftover in stranded)
    assert not list(tmp_path.glob("*.tmp"))
    assert not path.exists()
