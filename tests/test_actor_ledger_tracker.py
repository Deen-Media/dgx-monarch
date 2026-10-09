"""The reap ledger and the loop-side tracker, and the startup wiring that
plants the driver identity and starts the tracker.

The ledger runs against a synthetic ``/proc`` built in ``tmp_path``, and the
wiring tests read source, so no process is ever signalled and no thread is
ever started.
"""
from __future__ import annotations

import inspect
import json
import os
import stat

import pytest

from actor_lifetime_helpers import _Recorder
from dgx_monarch import client_lease
from dgx_monarch.cli import actor_ledger, proc_identity

_BOOT = "11111111-2222-3333-4444-555555555555"
_ACTOR_ARGV = ("/usr/bin/python", "-m", "monarch._src.actor.bootstrap_main")


def _plant(root, pid, *, ppid, starttime, argv, comm="python", environ=()):
    directory = root / str(pid)
    directory.mkdir(parents=True, exist_ok=True)
    filler = " ".join(["0"] * 17)
    (directory / "stat").write_text(
        f"{pid} ({comm}) S {ppid} {filler} {starttime} 0 0 0\n")
    (directory / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (directory / "environ").write_bytes(b"\0".join(e.encode() for e in environ) + b"\0")
    (directory / "status").write_text(f"Name:\t{comm}\nUid:\t1000\t1000\t1000\t1000\n")
    return directory


@pytest.fixture
def proc_root(tmp_path):
    root = tmp_path / "proc"
    (root / "sys" / "kernel" / "random").mkdir(parents=True)
    (root / "sys" / "kernel" / "random" / "boot_id").write_text(f"{_BOOT}\n")
    return root


def _tick(proc_root, tmp_path, *, loop_pid, address="tcp://10.0.0.1:26600", now=1000.0,
          writer=actor_ledger.write_ledger):
    return actor_ledger.tick(
        address,
        loop_pid=loop_pid,
        proc_root=str(proc_root),
        path=str(tmp_path / "state" / "actor-procs.json"),
        now=now,
        writer=writer,
    )


def test_the_tracker_records_only_actor_marked_descendants(proc_root, tmp_path):
    _plant(proc_root, 100, ppid=1, starttime=500,
           argv=("python", "-m", "dgx_monarch.cli.worker_loop", "--address", "tcp://x"))
    _plant(proc_root, 101, ppid=100, starttime=900, argv=_ACTOR_ARGV)
    _plant(proc_root, 102, ppid=101, starttime=910, argv=_ACTOR_ARGV)  # grandchild
    _plant(proc_root, 103, ppid=100, starttime=920, argv=("python", "train.py"))
    _plant(proc_root, 104, ppid=1, starttime=930, argv=_ACTOR_ARGV)  # another loop's
    # A process that only prints the module name is not a candidate: the match
    # compares argv tokens, never a substring of the joined command line.
    _plant(proc_root, 105, ppid=100, starttime=940,
           argv=("python", "-c", "print('monarch._src.actor.bootstrap_main')"))

    ledger = _tick(proc_root, tmp_path, loop_pid=100)
    assert [row["pid"] for row in ledger["actors"]] == [101, 102]
    assert ledger["loop"] == {"pid": 100, "starttime": 500, "address": "tcp://10.0.0.1:26600"}
    assert all(row["orphan_since"] is None for row in ledger["actors"])
    assert ledger["boot_id"] == _BOOT


def test_process_identity_survives_a_comm_with_spaces_and_parentheses(proc_root):
    _plant(proc_root, 200, ppid=7, starttime=4242, argv=_ACTOR_ARGV, comm="py thon (x)")
    assert proc_identity.read_stat(200, str(proc_root)) == (7, 4242)
    assert proc_identity.alive(200, 4242, str(proc_root)) is True
    assert proc_identity.alive(200, 4243, str(proc_root)) is False


def test_the_ledger_file_is_atomic_and_private(proc_root, tmp_path):
    _plant(proc_root, 100, ppid=1, starttime=500, argv=("python", "-m", "loop"))
    _plant(proc_root, 101, ppid=100, starttime=900, argv=_ACTOR_ARGV)
    path = tmp_path / "state" / "actor-procs.json"
    written = _tick(proc_root, tmp_path, loop_pid=100)

    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(path.parent).st_mode) == 0o700
    assert not list(path.parent.glob("*.tmp"))
    on_disk = json.loads(path.read_text())
    assert on_disk == written
    assert on_disk["version"] == actor_ledger.LEDGER_VERSION
    round_trip = actor_ledger.read_ledger(str(path), boot=_BOOT, proc_root=str(proc_root))
    assert [row["pid"] for row in round_trip["actors"]] == [101]


@pytest.mark.parametrize("body", ["", "{not json", '{"version": 99}', '[]'])
def test_a_missing_or_corrupt_ledger_reads_empty(proc_root, tmp_path, body):
    path = tmp_path / "actor-procs.json"
    if body:
        path.write_text(body)
    ledger = actor_ledger.read_ledger(str(path), boot=_BOOT, proc_root=str(proc_root))
    assert ledger == actor_ledger.empty_ledger(_BOOT)


def test_a_foreign_boot_ledger_is_discarded_whole(proc_root, tmp_path):
    path = tmp_path / "actor-procs.json"
    path.write_text(json.dumps({
        "version": 1,
        "boot_id": "99999999-9999-9999-9999-999999999999",
        "written_at": 1.0,
        "loop": {"pid": 100, "starttime": 500, "address": "tcp://x"},
        "actors": [{"pid": 101, "starttime": 900}],
    }))
    ledger = actor_ledger.read_ledger(str(path), proc_root=str(proc_root))
    assert ledger["actors"] == []
    assert ledger["boot_id"] == _BOOT


def test_rows_are_adopted_across_a_loop_restart_and_start_their_dwell_clock(
        proc_root, tmp_path):
    # The docs/TROUBLESHOOTING.md #51 leak: the loop that spawned the actor is
    # gone, a new loop has taken its place, and the actor is still alive.
    _plant(proc_root, 100, ppid=1, starttime=500, argv=("python", "-m", "loop"))
    _plant(proc_root, 101, ppid=100, starttime=900, argv=_ACTOR_ARGV)
    first = _tick(proc_root, tmp_path, loop_pid=100, now=1000.0)
    assert first["actors"][0]["first_seen"] == 1000.0

    # The loop dies, the actor is reparented to pid 1, a new loop starts.
    _plant(proc_root, 101, ppid=1, starttime=900, argv=_ACTOR_ARGV)
    (proc_root / "100").rename(proc_root / "100.gone")
    _plant(proc_root, 300, ppid=1, starttime=5000, argv=("python", "-m", "loop"))

    second = _tick(proc_root, tmp_path, loop_pid=300, now=2000.0)
    row = second["actors"][0]
    assert row["pid"] == 101
    assert row["loop_pid"] == 100          # ownership outlives the loop
    assert row["first_seen"] == 1000.0     # adopted, not re-created
    assert row["orphan_since"] == 2000.0
    assert second["loop"]["pid"] == 300


def test_dead_rows_and_recycled_pids_are_pruned(proc_root, tmp_path):
    _plant(proc_root, 100, ppid=1, starttime=500, argv=("python", "-m", "loop"))
    _plant(proc_root, 101, ppid=100, starttime=900, argv=_ACTOR_ARGV)
    _plant(proc_root, 102, ppid=100, starttime=910, argv=_ACTOR_ARGV)
    _tick(proc_root, tmp_path, loop_pid=100, now=1000.0)

    (proc_root / "101").rename(proc_root / "101.gone")   # exited
    _plant(proc_root, 102, ppid=100, starttime=99999, argv=_ACTOR_ARGV)  # pid reuse
    ledger = _tick(proc_root, tmp_path, loop_pid=100, now=1100.0)
    assert [(row["pid"], row["starttime"]) for row in ledger["actors"]] == [(102, 99999)]
    assert ledger["actors"][0]["first_seen"] == 1100.0


def test_a_reobserved_row_clears_its_dwell_clock(proc_root, tmp_path):
    _plant(proc_root, 100, ppid=1, starttime=500, argv=("python", "-m", "loop"))
    _plant(proc_root, 101, ppid=100, starttime=900, argv=_ACTOR_ARGV)
    _tick(proc_root, tmp_path, loop_pid=100, now=1000.0)
    (proc_root / "100").rename(proc_root / "100.gone")
    _plant(proc_root, 101, ppid=1, starttime=900, argv=_ACTOR_ARGV)
    orphaned = _tick(proc_root, tmp_path, loop_pid=999, now=2000.0)
    assert orphaned["actors"][0]["orphan_since"] == 2000.0

    # The same process observed again under a live loop is not an orphan.
    _plant(proc_root, 400, ppid=1, starttime=7000, argv=("python", "-m", "loop"))
    _plant(proc_root, 101, ppid=400, starttime=900, argv=_ACTOR_ARGV)
    adopted = _tick(proc_root, tmp_path, loop_pid=400, now=3000.0)
    assert adopted["actors"][0]["orphan_since"] is None
    assert adopted["actors"][0]["loop_pid"] == 400
    assert adopted["actors"][0]["first_seen"] == 1000.0


def test_the_ledger_is_rewritten_only_when_it_changes_or_goes_stale(
        proc_root, tmp_path):
    _plant(proc_root, 100, ppid=1, starttime=500, argv=("python", "-m", "loop"))
    _plant(proc_root, 101, ppid=100, starttime=900, argv=_ACTOR_ARGV)
    writes: list[dict] = []

    def _writer(ledger, path, **_kwargs):
        writes.append(ledger)
        actor_ledger.write_ledger(ledger, path)
        return True

    _tick(proc_root, tmp_path, loop_pid=100, now=1000.0, writer=_writer)
    _tick(proc_root, tmp_path, loop_pid=100, now=1005.0, writer=_writer)
    assert len(writes) == 1                                  # nothing changed
    _tick(proc_root, tmp_path, loop_pid=100, now=1000.0 + actor_ledger.HEARTBEAT_S,
          writer=_writer)
    assert len(writes) == 2                                  # liveness heartbeat
    _plant(proc_root, 102, ppid=100, starttime=1200, argv=_ACTOR_ARGV)
    _tick(proc_root, tmp_path, loop_pid=100, now=1070.0, writer=_writer)
    assert len(writes) == 3                                  # a new actor row


def test_an_unwritable_state_directory_never_raises(proc_root, tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    assert actor_ledger.write_ledger(actor_ledger.empty_ledger(_BOOT),
                                     str(blocked / "actor-procs.json")) is False


def test_the_tracker_thread_is_a_daemon():
    recorder = _Recorder()
    actor_ledger.start_tracker("tcp://10.0.0.1:26600", thread_factory=recorder)
    assert len(recorder.created) == 1
    assert recorder.created[0]["daemon"] is True
    assert recorder.created[0]["name"] == "dgxm-actor-ledger"


def test_the_tracker_loop_swallows_a_failing_tick(monkeypatch):
    calls: list[int] = []

    def _boom(_address, **_kwargs):
        calls.append(1)
        if len(calls) >= 2:
            raise _Stop
        raise OSError("procfs hiccup")

    class _Stop(Exception):
        pass

    monkeypatch.setattr(actor_ledger, "tick", _boom)
    naps: list[float] = []

    def _sleeper(seconds: float) -> None:
        naps.append(seconds)
        if len(naps) >= 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        actor_ledger._track("tcp://x", 5.0, _sleeper)
    assert calls == [1, 1]
    assert naps == [5.0, 5.0]


def test_the_driver_plants_its_identity_before_it_spawns_anything():
    # No process can read back its own os.environ writes through
    # /proc/<pid>/environ, so the sweep cannot ask a live driver whether it
    # owns a candidate; it asks the candidate, which inherited this pair.
    from dgx_monarch import mesh

    planted: dict[str, str] = {}
    marker = client_lease.plant_driver_marker(planted)
    assert planted[client_lease.DRIVER_ENV] == marker
    assert proc_identity.parse_owner(marker) == (
        os.getpid(), proc_identity.read_stat(os.getpid())[1])
    assert proc_identity.alive(*proc_identity.parse_owner(marker)) is True

    source = inspect.getsource(mesh.get_mesh)
    assert source.index("plant_driver_marker") < source.index("spawn_worker_fleet(")


def test_the_worker_loop_starts_the_tracker_before_it_blocks_in_native_code():
    from dgx_monarch.cli import worker_loop

    source = inspect.getsource(worker_loop.main)
    assert source.index("start_tracker(") < source.index("run_worker_loop_forever(")


def test_the_ledger_path_is_resolved_in_python_on_both_sides():
    assert actor_ledger.ledger_path("/home/example").endswith(
        "/.local/state/dgx-monarch/actor-procs.json")
    assert actor_ledger.LEDGER_REL == ".local/state/dgx-monarch/actor-procs.json"
