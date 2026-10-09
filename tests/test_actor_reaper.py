"""The actor sweep: identity, orphan policy, the kill ladder, the doctor report,
`dgxm reap` and the liveness answer.

Everything runs against a synthetic /proc built in tmp_path plus a recording
signal sender. No real process is ever signalled, no GPU is touched, and no
pattern ever reaches a process argv: identification happens in Python against
procfs contents.
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path

import pytest

from dgx_monarch.cli import actor_ledger, actor_reaper
from dgx_monarch.cli import proc_identity as pi

UID = 4242
SELF_PID = 999
LOOP_ADDRESS = "tcp://10.0.0.2:26600"
ACTOR_ARGV = ("/venv/bin/python", "-m", "monarch._src.actor.bootstrap_main")
LOOP_ARGV = ("/venv/bin/python", "-m", "dgx_monarch.cli.worker_loop",
             "--address", LOOP_ADDRESS)
# An actor process carries both markers: ownership (planted by dgx-monarch on
# the loop boundary and on the local-mode driver, so every descendant inherits
# it) and the launcher's own bootstrap marker.
OWNED_ENV = ("DGXM_PYTHONPATH=/home/u/.local/share/dgx-monarch/src",
             "HYPERACTOR_MESH_BOOTSTRAP_MODE=1")
# What a live worker loop or a live ComfyUI driver shows in
# /proc/<pid>/environ: not one dgx-monarch name. Both plant their markers with
# `os.environ[...] = ...` at runtime, and that never reaches the exec-time copy
# of the stack procfs serves. Children exec'd afterwards do carry them, which is
# why the marker identifies a candidate and can never identify its owner.
SPAWNER_ENV = ("PATH=/usr/bin:/bin", "HOME=/home/u", "LANG=C.UTF-8")


def make_proc(
    root: Path,
    pid: int,
    *,
    ppid: int = 1,
    starttime: int = 1000,
    session: int = 0,
    state: str = "S",
    uid: int = UID,
    argv: tuple[str, ...] = ACTOR_ARGV,
    env: tuple[str, ...] | None = OWNED_ENV,
    driver: str | None = None,
    comm: str = "py thon (x)",
    anon_gib: float = 0.0,
    file_gib: float = 0.0,
    sockets: tuple[int, ...] = (),
) -> None:
    """One synthetic ``/proc/<pid>`` directory.

    ``env=None`` writes no environ file, which reads like an environ the kernel
    refuses: the open fails, and ``proc_identity`` treats every failed open
    alike.
    """
    directory = root / str(pid)
    (directory / "fd").mkdir(parents=True, exist_ok=True)
    if driver is not None:
        env = (*(env or ()), f"DGXM_DRIVER={driver}")
    # stat fields after the last ')': state, ppid, pgrp, session, then the rest.
    fillers = " ".join(["0", str(session), *["0"] * 15])
    (directory / "stat").write_text(
        f"{pid} ({comm}) {state} {ppid} {fillers} {starttime}\n")
    (directory / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    if env is not None:
        (directory / "environ").write_bytes(b"\0".join(e.encode() for e in env) + b"\0")
    (directory / "status").write_text(f"Name:\tpython\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    pages = int((anon_gib + file_gib) * 2**30 / pi._PAGE_SIZE)
    (directory / "statm").write_text(f"{pages} {pages} 0 0 0 0 0\n")
    anon_kb, file_kb = int(anon_gib * 2**20), int(file_gib * 2**20)
    (directory / "smaps").write_text(
        f"7f0000000000-7f0000001000 rw-p 00000000 00:00 0\nRss:{anon_kb:>20} kB\n"
        f"7f1000000000-7f1000001000 r--p 00000000 08:01 4242 /lib/model.safetensors\n"
        f"Rss:{file_kb:>20} kB\n")
    (directory / "smaps_rollup").write_text(
        f"00000000-ffffffff ---p 00000000 00:00 0 [rollup]\n"
        f"Rss:{anon_kb + file_kb:>20} kB\nAnonymous:{anon_kb:>15} kB\n")
    for index, inode in enumerate(sockets):
        link = directory / "fd" / str(index)
        if not link.is_symlink():
            link.symlink_to(f"socket:[{inode}]")


def make_proc_root(tmp_path: Path, *, boot: str = "boot-abc", established=()) -> Path:
    root = tmp_path / "proc"
    (root / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
    (root / "net").mkdir(parents=True, exist_ok=True)
    (root / "sys/kernel/random/boot_id").write_text(f"{boot}\n")
    (root / "uptime").write_text("100000.00 900000.00\n")
    header = "  sl  local_address rem_address   st tx_queue rx:tm retrnsmt uid timeout inode\n"
    rows = "".join(
        f"  {index}: 0100007F:1F90 0100007F:8080 01 00000000:00000000 "
        f"00:00000000 00000000 {UID} 0 {inode} 1 ffff 20 0 0 10 0\n"
        for index, inode in enumerate(established))
    (root / "net/tcp").write_text(header + rows)
    (root / "net/tcp6").write_text(header)
    return root


def write_ledger(tmp_path: Path, rows: list[dict], *, boot: str = "boot-abc",
                 loop: dict | None = None, written_at: float = 0.0) -> str:
    path = tmp_path / "actor-procs.json"
    path.write_text(json.dumps({
        "version": 1, "boot_id": boot, "written_at": written_at,
        "loop": loop, "actors": rows,
    }))
    return str(path)


class Killer:
    """Records signals. SIGKILL always kills; SIGTERM kills only ``dies_on_term``."""

    def __init__(self, root: Path, dies_on_term: tuple[int, ...] = ()):
        self.root = root
        self.dies_on_term = dies_on_term
        self.calls: list[tuple[int, int]] = []

    def __call__(self, pid: int, sig: int) -> None:
        self.calls.append((pid, sig))
        if sig == signal.SIGKILL or (sig == signal.SIGTERM and pid in self.dies_on_term):
            for name in ("stat", "cmdline", "environ", "status", "smaps", "smaps_rollup"):
                target = self.root / str(pid) / name
                if target.exists():
                    target.unlink()


def sweep(root: Path, **kwargs):
    kwargs.setdefault("killer", Killer(root))
    return actor_reaper.run_sweep(
        proc_root=str(root), uid=UID, self_pid=SELF_PID,
        loop_addresses=(LOOP_ADDRESS,), sleep=lambda _s: None,
        term_wait_s=1.0, kill_wait_s=0.5, **kwargs)


def scan(root: Path, **kwargs):
    return actor_reaper.scan(
        proc_root=str(root), uid=UID, self_pid=SELF_PID,
        loop_addresses=(LOOP_ADDRESS,), **kwargs)


def liveness(root: Path, **kwargs):
    return actor_reaper.liveness(
        proc_root=str(root), uid=UID, self_pid=SELF_PID,
        loop_addresses=(LOOP_ADDRESS,), **kwargs)


def test_stat_parses_field_22_when_comm_contains_spaces_and_parens(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, comm="py thon (x)", ppid=7, starttime=4560123)
    assert pi.read_stat(100, str(root)) == (7, 4560123)


def test_marked_owned_actor_is_a_candidate(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, anon_gib=49.0)
    found = scan(root)
    assert [proc.pid for proc in found] == [100]
    assert found[0].gib == 49.0


def test_plain_python_holding_60_gib_is_not_a_candidate(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, argv=("python", "train.py"), env=(), anon_gib=60.0)
    assert scan(root) == []


def test_another_users_actor_is_never_a_candidate(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, uid=UID + 1)
    assert scan(root) == []


def test_unrelated_monarch_app_needs_no_marker_to_be_seen(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, env=("HYPERACTOR_MESH_BOOTSTRAP_MODE=1",))
    assert scan(root) == []
    assert [proc.pid for proc in scan(root, require_marker=False)] == [100]


def test_rewritten_argv0_is_still_identified(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, argv=("env", "-m", "monarch._src.actor.bootstrap_main"))
    assert [proc.pid for proc in scan(root)] == [100]


def test_process_that_only_mentions_the_module_is_not_a_candidate(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, argv=(
        "python", "-c", "print('monarch._src.actor.bootstrap_main')"), env=())
    assert scan(root) == []


def test_self_and_ancestors_and_the_loop_are_excluded(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 500, argv=("bash", "-s"), env=SPAWNER_ENV)
    make_proc(root, SELF_PID, ppid=500, argv=(
        "python", "-m", "dgx_monarch.cli.actor_reaper", "--sweep"))
    make_proc(root, 300, argv=LOOP_ARGV, env=SPAWNER_ENV, starttime=500)
    make_proc(root, 100, ppid=300, starttime=900)
    plan = scan(root)
    assert [proc.pid for proc in plan] == [100]


def test_reparented_actor_is_an_orphan(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1)
    assert scan(root)[0].provenance == actor_reaper.ORPHAN


def test_actor_reparented_to_a_subreaper_is_still_an_orphan(tmp_path):
    """The docs/TROUBLESHOOTING.md #51 leak: the loop died and systemd --user
    adopted the actor.

    A subreaper is a live parent that started long before its charge, so no
    ordering rule catches it. The session does: it is inherited at fork and only
    a process's own setsid changes it, so a parent in another session did not
    spawn an actor that never called setsid.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 2797, argv=("/lib/systemd/systemd", "--user"), env=(),
              session=2797, starttime=10)
    make_proc(root, 100, ppid=2797, session=4000)
    assert scan(root)[0].provenance == actor_reaper.ORPHAN


def test_live_loop_child_is_not_an_orphan(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 300, argv=LOOP_ARGV, env=SPAWNER_ENV, starttime=500)
    make_proc(root, 100, ppid=300, starttime=900)
    assert scan(root)[0].provenance == actor_reaper.LOOP_CHILD


def test_actor_older_than_its_parent_loop_is_an_orphan(tmp_path):
    """A restarted loop (or a recycled pid) cannot have spawned an older proc."""
    root = make_proc_root(tmp_path)
    make_proc(root, 300, argv=LOOP_ARGV, env=SPAWNER_ENV, starttime=900)
    make_proc(root, 100, ppid=300, starttime=500)
    assert scan(root)[0].provenance == actor_reaper.ORPHAN


def test_actor_under_a_live_driver_is_held_not_orphaned(tmp_path):
    """Local mode: the actor is a child of ComfyUI, which is still running.

    The driver's own environ carries no dgx-monarch name (see SPAWNER_ENV), so
    ownership is read off the child: the driver planted its `(pid, starttime)`
    in the environment this process inherited.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    make_proc(root, 100, ppid=200, starttime=900, driver="200:100")
    assert scan(root)[0].provenance == actor_reaper.HELD


def test_a_live_local_mode_fleet_is_never_swept_and_never_warned_about(tmp_path):
    """A live ComfyUI with two fat ranks under it is a healthy single-box fleet.

    It must survive `dgxm reap` (which runs with grace 0 and no cluster.toml)
    and must not put doctor into a permanent WARN. The ranks hold no ESTABLISHED
    TCP socket here on purpose: a local transport is not required to be TCP, so
    an attachment count is not evidence about a held process.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    for rank, pid in enumerate((601, 602)):
        make_proc(root, pid, ppid=200, starttime=900 + rank, anon_gib=25.0,
                  driver="200:100")
    killer = Killer(root)
    assert sweep(root, killer=killer)["pids"] == []
    assert killer.calls == []
    report = actor_reaper.orphan_report(proc_root=str(root), uid=UID, self_pid=SELF_PID)
    assert report.orphans == () and report.tracked == 2


def test_a_dead_driver_marker_is_positive_proof_of_an_orphan(tmp_path):
    """Same fleet after ComfyUI exits: the marker names a pid that is gone."""
    root = make_proc_root(tmp_path)
    make_proc(root, 2797, argv=("/lib/systemd/systemd", "--user"), env=(),
              session=2797, starttime=10)
    make_proc(root, 100, ppid=2797, session=4000, starttime=900,
              anon_gib=49.0, driver="200:100")
    assert scan(root)[0].provenance == actor_reaper.ORPHAN
    assert sweep(root, killer=Killer(root, dies_on_term=(100,)))["reaped"] == 1


def test_a_recycled_driver_pid_does_not_hold_an_orphan(tmp_path):
    """The owner pair is checked on `(pid, starttime)`, never on the pid."""
    root = make_proc_root(tmp_path)
    make_proc(root, 200, argv=("python", "other.py"), env=SPAWNER_ENV, starttime=7777)
    make_proc(root, 100, ppid=200, starttime=900, driver="200:100")
    assert scan(root)[0].provenance == actor_reaper.ORPHAN


def test_driver_identity_is_empty_rather_than_a_zero_start_time(tmp_path):
    """A pair carrying start time 0 matches no live process.

    A live driver would then read as dead and its own fleet would become a sweep
    target. An empty answer makes the planter skip the marker instead, which
    leaves the candidate on the `held` default every unresolvable case gets.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    assert pi.driver_identity(200, str(root)) == "200:100"
    assert pi.driver_identity(4242, str(root)) == ""


def test_a_ledger_row_naming_no_loop_never_overrides_a_held_verdict(tmp_path):
    """`loop_pid: 0` means "no recorded owner", not "the owner is dead".

    `/proc/0` reads as dead for everybody, so testing it unguarded turns every
    truncated row into a kill order against a process the live evidence held.
    The ledger writer already guards the identical test.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    make_proc(root, 100, ppid=200, starttime=900, driver="200:100", anon_gib=49.0)
    ledger = write_ledger(tmp_path, [{
        "pid": 100, "starttime": 900, "address": LOOP_ADDRESS,
        "loop_pid": 0, "loop_starttime": 0, "orphan_since": None,
    }])
    assert scan(root, ledger_file=ledger)[0].provenance == actor_reaper.HELD
    killer = Killer(root)
    assert sweep(root, killer=killer, ledger_file=ledger)["pids"] == []
    assert killer.calls == []


def test_an_unattributable_candidate_is_reported_by_nobody_and_signalled_by_nobody(
        tmp_path):
    """A driver older than this code plants no `DGXM_DRIVER` marker.

    With a live parent in the same session, nothing in procfs then says whether
    that parent spawned this process or adopted it, and the safe answer is
    `held`: killing a live render is worse than the leak.
    docs/TROUBLESHOOTING.md #51 gives the operator's remedy for an old orphan.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    make_proc(root, 100, ppid=200, starttime=900, anon_gib=49.0)
    assert scan(root)[0].provenance == actor_reaper.HELD
    killer = Killer(root)
    assert sweep(root, killer=killer)["pids"] == []
    assert killer.calls == []
    report = actor_reaper.orphan_report(proc_root=str(root), uid=UID, self_pid=SELF_PID)
    assert report.orphans == ()


def test_ledger_row_with_a_dead_loop_marks_an_orphan(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 500, argv=("bash",), env=SPAWNER_ENV)
    make_proc(root, 100, ppid=500, starttime=900)
    ledger = write_ledger(tmp_path, [{
        "pid": 100, "starttime": 900, "address": LOOP_ADDRESS,
        "loop_pid": 300, "loop_starttime": 500, "orphan_since": None,
    }])
    assert scan(root, ledger_file=ledger)[0].provenance == actor_reaper.ORPHAN


def test_foreign_boot_ledger_is_ignored_whole(tmp_path):
    root = make_proc_root(tmp_path, boot="boot-abc")
    make_proc(root, 100, env=("HYPERACTOR_MESH_BOOTSTRAP_MODE=1",))
    ledger = write_ledger(tmp_path, [{
        "pid": 100, "starttime": 1000, "address": LOOP_ADDRESS,
        "loop_pid": 0, "loop_starttime": 0, "orphan_since": None,
    }], boot="boot-from-last-week")
    assert scan(root, ledger_file=ledger) == []


def test_a_damaged_ledger_reads_as_empty_and_never_raises(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1)
    path = tmp_path / "actor-procs.json"
    path.write_text('{"version": 1, "actors": [tru')
    assert [proc.pid for proc in scan(root, ledger_file=str(path))] == [100]


def test_established_socket_counts_but_listen_does_not(tmp_path):
    root = make_proc_root(tmp_path, established=(7001,))
    make_proc(root, 100, ppid=1, sockets=(7001,))
    make_proc(root, 101, ppid=1, sockets=(9999,), starttime=1001)
    counts = {proc.pid: proc.established for proc in scan(root)}
    assert counts == {100: 1, 101: 0}


def test_pool_gib_counts_only_giant_anonymous_mappings(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, anon_gib=49.0, file_gib=30.0)
    make_proc(root, 101, starttime=1001, anon_gib=2.0, file_gib=0.0)
    assert pi.pool_gib(100, str(root)) == 49.0
    assert pi.pool_gib(101, str(root)) == 0.0
    assert pi.rollup_rss_gib(100, str(root)) == 79.0


def test_pool_gib_sees_slab_residency_in_a_memfd_mapping(tmp_path):
    """A slab-resident actor holds its weights in a memfd and reports almost no
    Anonymous, which is why the row counts ``/memfd:`` mappings and prefilters
    on rollup Rss."""
    root = make_proc_root(tmp_path)
    make_proc(root, 100, anon_gib=0.1)
    (root / "100" / "smaps").write_text(
        "7f0000000000-7f0000001000 rw-s 00000000 00:05 8123 /memfd:dgxm-slab (deleted)\n"
        f"Rss:{50 * 2**20:>20} kB\n"
        "7f1000000000-7f1000001000 rw-p 00000000 00:00 0 [heap]\n"
        f"Rss:{9 * 2**20:>20} kB\n")
    (root / "100" / "smaps_rollup").write_text(
        f"Rss:{59 * 2**20:>20} kB\nAnonymous:{9 * 2**20:>15} kB\n")
    assert pi.pool_gib(100, str(root)) == 50.0


def test_missing_smaps_yields_zero_and_never_raises(tmp_path):
    root = make_proc_root(tmp_path)
    assert pi.pool_gib(4242, str(root)) == 0.0
    assert pi.rollup_rss_gib(4242, str(root)) == 0.0


def test_sweep_targets_orphans_only_and_never_the_held_or_loop_children(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 300, argv=LOOP_ARGV, env=SPAWNER_ENV, starttime=100)
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    make_proc(root, 100, ppid=1, starttime=900, anon_gib=49.0)
    make_proc(root, 101, ppid=300, starttime=901)
    make_proc(root, 102, ppid=200, starttime=902)
    killer = Killer(root, dies_on_term=(100,))
    result = sweep(root, killer=killer)
    assert result["pids"] == [100]
    assert result["reaped"] == 1 and result["left"] == 0
    assert {pid for pid, _sig in killer.calls} == {100}


def test_all_loop_children_adds_live_loop_children_but_not_driver_children(tmp_path):
    """`down` passes this only after it stops the loop. A live driver's procs
    stay untouched, which keeps the driver-side restart path safe."""
    root = make_proc_root(tmp_path)
    make_proc(root, 300, argv=LOOP_ARGV, env=SPAWNER_ENV, starttime=100)
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    make_proc(root, 101, ppid=300, starttime=901)
    make_proc(root, 102, ppid=200, starttime=902)
    killer = Killer(root, dies_on_term=(101,))
    result = sweep(root, killer=killer, all_loop_children=True)
    assert result["pids"] == [101]
    assert {pid for pid, _sig in killer.calls} == {101}


def test_kill_ladder_is_term_then_kill_only_for_survivors(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1, starttime=900)
    make_proc(root, 101, ppid=1, starttime=901)
    killer = Killer(root, dies_on_term=(100,))
    result = sweep(root, killer=killer)
    assert killer.calls[:2] == [(100, signal.SIGTERM), (101, signal.SIGTERM)]
    assert killer.calls[2:] == [(101, signal.SIGKILL)]
    assert result["killed"] == 1 and result["reaped"] == 2 and result["left"] == 0


def test_a_process_that_survives_sigkill_is_reported_left(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1)

    def stubborn(_pid: int, _sig: int) -> None:
        return None

    result = sweep(root, killer=stubborn)
    assert result["left"] == 1 and result["reaped"] == 0


def test_second_sweep_is_a_no_op_and_prunes_the_ledger(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1, starttime=900)
    ledger = write_ledger(tmp_path, [{
        "pid": 100, "starttime": 900, "address": LOOP_ADDRESS,
        "loop_pid": 0, "loop_starttime": 0, "orphan_since": None,
    }])
    first = sweep(root, killer=Killer(root, dies_on_term=(100,)), ledger_file=ledger)
    second = sweep(root, ledger_file=ledger)
    assert first["reaped"] == 1
    assert second == {"tracked": 0, "gib": 0.0, "pids": [], "reaped": 0,
                      "killed": 0, "left": 0}
    assert actor_ledger.read_ledger(ledger, proc_root=str(root))["actors"] == []


def test_a_recycled_pid_in_the_ledger_is_never_signalled(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=200, starttime=5000, argv=("python", "unrelated.py"), env=())
    make_proc(root, 200, argv=("python", "main.py"), env=SPAWNER_ENV, starttime=100)
    ledger = write_ledger(tmp_path, [{
        "pid": 100, "starttime": 900, "address": LOOP_ADDRESS,
        "loop_pid": 0, "loop_starttime": 0, "orphan_since": None,
    }])
    killer = Killer(root)
    assert sweep(root, killer=killer, ledger_file=ledger)["pids"] == []
    assert killer.calls == []


def test_grace_holds_off_a_young_orphan(tmp_path):
    root = make_proc_root(tmp_path)
    # Uptime is 100000 s, so a process started 99990 s after boot is 10 s old.
    make_proc(root, 100, ppid=1, starttime=int(99990 * os.sysconf("SC_CLK_TCK")))
    killer = Killer(root)
    assert sweep(root, killer=killer, grace_s=180.0)["pids"] == []
    assert sweep(root, killer=Killer(root, dies_on_term=(100,)), grace_s=5.0)["pids"] == [100]


def test_a_pid_recycled_between_the_scan_and_the_signal_is_never_signalled(
        tmp_path, monkeypatch):
    """The confirmation closes the scan-to-signal window.

    `scan` walks every pid and then reads smaps for each candidate before any
    signal. If the target exits in that window and the kernel reissues its pid
    to an unrelated process of ours, the stored verdict names somebody else, so
    identity, uid and marker are all re-read immediately before the signal.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1, starttime=900, anon_gib=49.0)
    stale = scan(root)
    assert [proc.pid for proc in stale] == [100]
    make_proc(root, 100, ppid=500, starttime=7777,
              argv=("python", "editor.py"), env=SPAWNER_ENV)
    monkeypatch.setattr(actor_reaper, "scan", lambda **_kwargs: stale)
    killer = Killer(root)
    sweep(root, killer=killer)
    assert killer.calls == []


def test_dry_run_signals_nothing_and_still_reports_the_plan(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1, anon_gib=49.0)
    killer = Killer(root)
    result = sweep(root, killer=killer, dry_run=True)
    assert result["pids"] == [100] and result["left"] == 1 and result["gib"] == 49.0
    assert killer.calls == []


def test_orphan_report_flags_the_fat_clientless_actor_only(tmp_path):
    root = make_proc_root(tmp_path, established=(7001,))
    make_proc(root, 100, ppid=1, starttime=900, anon_gib=49.1)
    make_proc(root, 101, ppid=1, starttime=901, anon_gib=5.0, sockets=(7001,))
    make_proc(root, 102, ppid=1, starttime=902, anon_gib=1.0)
    report = actor_reaper.orphan_report(
        proc_root=str(root), uid=UID, self_pid=SELF_PID)
    assert report.tracked == 3
    assert report.attached == 1
    assert [proc.pid for proc in report.orphans] == [100, 101]
    assert report.orphans[0].gib == 49.1


def test_attached_actor_under_a_live_loop_is_not_reported(tmp_path):
    root = make_proc_root(tmp_path, established=(7001,))
    make_proc(root, 300, argv=LOOP_ARGV, env=SPAWNER_ENV, starttime=100)
    make_proc(root, 100, ppid=300, starttime=900, anon_gib=49.0, sockets=(7001,))
    report = actor_reaper.orphan_report(
        proc_root=str(root), uid=UID, self_pid=SELF_PID)
    assert report.orphans == () and report.tracked == 1 and report.attached == 1


def test_a_loop_child_holding_no_socket_at_all_is_still_not_reported(tmp_path):
    """The row reads provenance, never the socket count.

    A live loop child on the driver box can hold no TCP socket in this table
    (nothing requires the transport to be TCP), and its owner, the worker loop,
    is alive. Reporting it would print `no live owner` about a process whose
    owner is alive, and hand the operator a remedy that will not touch it: the
    sweep leaves live loop children alone unless `down` stopped the loop first.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 300, argv=LOOP_ARGV, env=SPAWNER_ENV, starttime=100)
    make_proc(root, 100, ppid=300, starttime=900, anon_gib=49.0)
    report = actor_reaper.orphan_report(
        proc_root=str(root), uid=UID, self_pid=SELF_PID)
    assert report.orphans == () and report.tracked == 1 and report.attached == 0


def test_an_orphan_whose_residency_is_spread_over_small_mappings_is_reported(tmp_path):
    """The symptom is a process total; the pool rule counts giant mappings.

    `pool_gib` counts a mapping only from 4 GiB up, so 50 GiB arriving as fifty
    1 GiB mappings pools to 0.0. Selecting on the pool would print `none` for
    the leaked process, so the row selects on the process total and prints the
    pool beside it as evidence.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1, starttime=900)
    (root / "100" / "smaps").write_text("".join(
        f"7f{index:010x}-7f{index + 1:010x} rw-p 00000000 00:00 0\n"
        f"Rss:{2**20:>20} kB\n" for index in range(50)))
    (root / "100" / "smaps_rollup").write_text(f"Rss:{50 * 2**20:>20} kB\n")
    proc = scan(root)[0]
    assert proc.rss_gib == 50.0 and proc.gib == 0.0
    report = actor_reaper.orphan_report(
        proc_root=str(root), uid=UID, self_pid=SELF_PID)
    assert [orphan.pid for orphan in report.orphans] == [100]


def test_dwell_text_degrades_to_age_without_a_tracker_row(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 100, ppid=1, starttime=int(77680 * os.sysconf("SC_CLK_TCK")))
    proc = scan(root)[0]
    assert proc.dwell_text == "age 6h12m"
    assert proc.signal_text == "orphan, no client"


def test_summary_line_round_trips_including_the_empty_case():
    empty = {"reaped": 0, "killed": 0, "left": 0, "tracked": 0, "gib": 0.0, "pids": []}
    line = actor_reaper.format_summary(empty)
    assert line.endswith("pids=-")
    assert actor_reaper.parse_summary(f"noise\n{line}\n") == empty
    full = {"reaped": 2, "killed": 1, "left": 1, "tracked": 3, "gib": 50.3,
            "pids": [117707, 118322]}
    assert actor_reaper.parse_summary(actor_reaper.format_summary(full)) == full


def test_parse_summary_ignores_output_without_the_contract_line():
    assert actor_reaper.parse_summary("") is None
    assert actor_reaper.parse_summary("STARTED_NOHUP\n") is None
    with pytest.raises(ValueError, match="malformed"):
        actor_reaper.parse_summary("ACTOR_SWEEP reaped=x left=1\n")


def test_parse_summary_requires_one_exact_final_frame():
    line = actor_reaper.format_summary(
        {"reaped": 0, "killed": 0, "left": 0, "tracked": 0, "gib": 0.0, "pids": []}
    )
    with pytest.raises(ValueError, match="unique and final"):
        actor_reaper.parse_summary(f"{line}\n{line}\n")
    with pytest.raises(ValueError, match="unique and final"):
        actor_reaper.parse_summary(f"{line}\ntrailing output\n")
    with pytest.raises(ValueError, match="malformed"):
        actor_reaper.parse_summary(f"{line} extra=1\n")


def _summary(**fields):
    base = {"reaped": 1, "killed": 0, "left": 0, "tracked": 2, "gib": 49.1,
            "pids": [117707]}
    base.update(fields)
    return base


def test_reap_without_a_cluster_sweeps_this_box(monkeypatch, capsys):
    """A local-mode fleet has no configured hosts, so `down` and `up` iterate
    nothing and `dgxm reap` is its only sweep."""
    from dgx_monarch.cli import actor_sweep

    calls: list[dict] = []

    def fake_sweep(**kwargs):
        calls.append(kwargs)
        return _summary()

    monkeypatch.setattr(actor_reaper, "run_sweep", fake_sweep)
    assert actor_sweep.reap_command(None, grace_s=180.0) is True
    assert calls == [{"grace_s": 180.0, "dry_run": False}]
    assert "117707" in capsys.readouterr().out


@pytest.mark.parametrize(("dry_run", "expected"), [(True, True), (False, False)])
def test_reap_exit_contract_follows_survivors_except_on_a_dry_run(
        monkeypatch, capsys, dry_run, expected):
    from dgx_monarch.cli import actor_sweep

    monkeypatch.setattr(
        actor_reaper, "run_sweep",
        lambda **_kwargs: _summary(reaped=0, left=2, pids=[1, 2]))
    assert actor_sweep.reap_command(None, dry_run=dry_run) is expected
    capsys.readouterr()


def test_reap_sweeps_every_configured_host_and_the_local_box(monkeypatch, capsys):
    """A driver is not required to be one of its own workers.

    The doctor row that names an orphan is box-local, so the remedy it prints
    has to reach that box whether or not it appears in cluster.toml.
    """
    from dgx_monarch.cli import actor_sweep, lifecycle
    from dgx_monarch.config import ClusterConfig, HostConfig

    config = ClusterConfig(
        hosts=(HostConfig("w1", "tcp://10.0.0.2:26600"),
               HostConfig("w2", "tcp://10.0.0.3:26600")),
        transport_security="trusted_fabric")
    swept: list[str] = []

    def fake_host(_config, host, **_kwargs):
        swept.append(host.name)
        return host.name != "w2"

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(actor_sweep, "sweep_host", fake_host)
    monkeypatch.setattr(actor_reaper, "run_sweep", lambda **_kwargs: _summary())
    assert actor_sweep.reap_command(config) is False  # one failing host fails it
    assert swept == ["w1", "w2"]
    assert "local: actor sweep" in capsys.readouterr().out


def test_reap_does_not_sweep_the_local_box_twice(monkeypatch, capsys):
    from dgx_monarch.cli import actor_sweep, lifecycle
    from dgx_monarch.config import ClusterConfig, HostConfig

    config = ClusterConfig(
        hosts=(HostConfig("head", "tcp://127.0.0.1:26600"),),
        transport_security="trusted_fabric")

    def forbidden(**_kwargs):
        raise AssertionError("a configured local host is already swept")

    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: True)
    monkeypatch.setattr(actor_sweep, "sweep_host", lambda _c, _h, **_k: True)
    monkeypatch.setattr(actor_reaper, "run_sweep", forbidden)
    assert actor_sweep.reap_command(config) is True
    assert "local: actor sweep" not in capsys.readouterr().out


def test_remote_sweep_oserror_is_a_failure(monkeypatch, capsys):
    from dgx_monarch.cli import actor_sweep, lifecycle
    from dgx_monarch.config import ClusterConfig, HostConfig

    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")

    def unavailable(*_args, **_kwargs):
        raise OSError("private transport detail")

    # This test exercises runner failure, not host resolution.
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    assert not actor_sweep.sweep_host(config, host, runner=unavailable)
    out = capsys.readouterr().out
    assert "actor sweep unavailable" in out


def test_reap_cli_maps_remote_sweep_oserror_to_exit_one(monkeypatch, capsys):
    from dgx_monarch.cli import lifecycle
    from dgx_monarch.cli import main as cli
    from dgx_monarch.config import ClusterConfig, HostConfig

    host = HostConfig("worker", "tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    monkeypatch.setattr(cli, "_load_config", lambda _args: config)
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: True)
    monkeypatch.setattr(
        lifecycle,
        "run_on_host",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    assert cli.main(["reap"]) == 1
    assert "actor sweep unavailable" in capsys.readouterr().out


def test_the_reap_subcommand_is_wired_and_maps_its_exit_code(monkeypatch, capsys):
    from dgx_monarch.cli import actor_sweep
    from dgx_monarch.cli import main as cli

    seen: list[dict] = []

    def fake_reap(config, **kwargs):
        seen.append({"config": config, **kwargs})
        return kwargs["dry_run"]

    monkeypatch.setattr(cli, "_load_config", lambda _args: None)
    monkeypatch.setattr(actor_sweep, "reap_command", fake_reap)
    assert cli.main(["reap", "--dry-run"]) == 0
    assert cli.main(["reap", "--grace", "300"]) == 1
    assert [call["grace_s"] for call in seen] == [0.0, 300.0]
    assert [call["config"] for call in seen] == [None, None]
    capsys.readouterr()


def test_module_entrypoint_reports_and_accepts_no_pattern_option(capsys):
    rc = actor_reaper.main(["--report", "--loop-address", LOOP_ADDRESS])
    assert rc == 0
    assert isinstance(json.loads(capsys.readouterr().out.splitlines()[-1])["candidates"], list)
    # The actor marker is hardcoded, so no option can put it into an argv and
    # the sweeper can never match itself.
    for rejected in ("--pattern", "--marker", "--match"):
        with pytest.raises(SystemExit):
            actor_reaper.main(["--sweep", rejected, pi.ACTOR_MODULE])


def test_the_liveness_line_carries_schema_one_and_every_named_field(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 4100, ppid=4000, starttime=900, argv=LOOP_ARGV, env=SPAWNER_ENV)
    make_proc(root, 4101, ppid=4100)
    ledger = write_ledger(
        tmp_path,
        [{"pid": 4101, "starttime": 1000, "loop_pid": 4100,
          "loop_starttime": 900}],
        loop={"pid": 4100, "starttime": 900, "address": LOOP_ADDRESS},
        written_at=1.0)
    answer = liveness(root, ledger_file=ledger)
    assert set(answer) == {
        "schema", "actors", "actors_marked", "zombies", "unreadable",
        "ledger_rows_alive", "ledger_boot_ok", "ledger_age_s", "loop_alive",
        "pids"}
    assert answer["schema"] == 1
    assert (answer["actors"], answer["actors_marked"]) == (1, 1)
    assert (answer["ledger_rows_alive"], answer["loop_alive"]) == (1, True)
    assert answer["ledger_boot_ok"] is True
    assert answer["pids"][0]["pid"] == 4101
    assert "bootstrap_main" in answer["pids"][0]["argv_tail"]


def test_a_process_whose_two_reads_are_both_empty_counts_as_unreadable(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 4200, argv=(), env=())
    answer = liveness(root)
    assert (answer["unreadable"], answer["actors"], answer["zombies"]) == (1, 0, 0)


def test_a_process_that_vanished_before_the_recheck_is_not_unreadable(
        tmp_path, monkeypatch):
    root = make_proc_root(tmp_path)
    make_proc(root, 4300, argv=(), env=())
    real = pi.alive
    monkeypatch.setattr(
        pi, "alive",
        lambda pid, starttime, proc_root=pi.PROC_ROOT: (
            False if pid == 4300 else real(pid, starttime, proc_root)))
    assert liveness(root)["unreadable"] == 0


def test_a_zombie_is_counted_as_a_zombie_and_nowhere_else(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 4400, state="Z", argv=(), env=())
    make_proc(root, 4401, state="Z")
    answer = liveness(root)
    assert answer["zombies"] == 2
    assert (answer["unreadable"], answer["actors"]) == (0, 0)


def test_a_ledger_row_for_a_zombie_does_not_count_as_alive(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 4500, state="Z")
    ledger = write_ledger(tmp_path, [{"pid": 4500, "starttime": 1000}])
    assert liveness(root, ledger_file=ledger)["ledger_rows_alive"] == 0


def test_a_stopped_actor_is_alive_because_it_still_holds_its_memory(tmp_path):
    # State T is the frozen-actor shape: it answers nothing and holds every
    # page. Only Z belongs in the zombie rule.
    root = make_proc_root(tmp_path)
    make_proc(root, 4550, state="T")
    ledger = write_ledger(tmp_path, [{"pid": 4550, "starttime": 1000}])
    answer = liveness(root, ledger_file=ledger)
    assert (answer["actors"], answer["zombies"]) == (1, 0)
    assert answer["ledger_rows_alive"] == 1


def test_a_marked_actor_with_no_owner_marker_still_counts_in_actors(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 4600, env=("HYPERACTOR_MESH_BOOTSTRAP_MODE=1",))
    answer = liveness(root)
    assert (answer["actors"], answer["actors_marked"]) == (1, 0)


def test_the_marked_count_matches_a_marker_only_scan_row_for_row(tmp_path):
    # One walk answers both counts, so the strict count is derived rather than
    # re-walked. This pins the derivation against the scan's own rule.
    root = make_proc_root(tmp_path)
    make_proc(root, 4610, env=("HYPERACTOR_MESH_BOOTSTRAP_MODE=1",))
    make_proc(root, 4611, env=("HYPERACTOR_MESH_BOOTSTRAP_MODE=1",))
    make_proc(root, 4612)
    ledger = write_ledger(tmp_path, [{"pid": 4611, "starttime": 1000}])
    answer = liveness(root, ledger_file=ledger)
    strict = scan(root, ledger_file=ledger, require_marker=True)
    assert answer["actors_marked"] == len(strict) == 2
    assert answer["actors"] == 3


def test_an_orphan_actor_with_no_address_attribution_counts_in_actors(tmp_path):
    root = make_proc_root(tmp_path)
    make_proc(root, 4700, ppid=1)
    answer = liveness(root)
    assert answer["actors"] == 1
    assert answer["pids"][0]["provenance"] == actor_reaper.ORPHAN


def test_a_fat_process_that_refuses_its_environ_counts_as_unreadable(tmp_path):
    """A live own-uid process holding tens of GiB, whose command line names no
    python module, whose environ will not read and which has no ledger row,
    counts as unreadable.

    It is not marked, so it is not in `actors`, and only one of its two reads
    comes back empty, so a rule that counts a process unreadable only when both
    reads are empty would let the host answer gone.
    """
    root = make_proc_root(tmp_path)
    make_proc(root, 5100, argv=("/opt/bin/worker-shim",), env=None,
              anon_gib=60.0)
    answer = liveness(root)
    assert (answer["actors"], answer["unreadable"]) == (0, 1)


def test_a_small_process_that_refuses_its_environ_is_not_unreadable(tmp_path):
    # The size floor is load bearing. The kernel refuses the environ of an
    # own-uid process that is not dumpable or holds privileges the reader lacks
    # (in a desktop session, for example systemd --user, sd-pam, fusermount3 and
    # ssh-agent), so without a floor those agents would hold the latch closed
    # for ever and no host could ever answer gone.
    root = make_proc_root(tmp_path)
    make_proc(root, 5110, argv=("/usr/bin/ssh-agent", "-D"), env=None,
              anon_gib=0.01)
    assert liveness(root)["unreadable"] == 0


def test_a_python_module_with_a_refused_environ_is_not_unreadable(tmp_path):
    # A fat worker loop says on its command line what it is, and that is an
    # answer even when its environ is not.
    root = make_proc_root(tmp_path)
    make_proc(root, 5120, env=None, anon_gib=60.0,
              argv=("python", "-m", pi.LOOP_MODULE, "--address", LOOP_ADDRESS))
    answer = liveness(root)
    assert (answer["actors"], answer["unreadable"]) == (0, 0)


def test_an_actor_with_a_refused_environ_is_counted_once(tmp_path):
    # `actors` and `unreadable` come from two separate walks, so the actor's
    # own module token must exclude it here as well. Otherwise every leaked
    # actor with a refused environ would print actors=1 unreadable=1.
    root = make_proc_root(tmp_path)
    make_proc(root, 5140, env=None, anon_gib=60.0)
    answer = liveness(root)
    assert (answer["actors"], answer["unreadable"]) == (1, 0)


def test_an_actor_that_exited_during_the_walk_is_gone_not_unreadable(tmp_path):
    # The pid may already belong to someone else, so the re-check reads the
    # start time and not just the directory.
    root = make_proc_root(tmp_path)
    make_proc(root, 5130, argv=("/opt/bin/worker-shim",), env=None,
              anon_gib=60.0)
    assert pi.unreadable_to_us(5130, 1000, False, str(root)) is True
    assert pi.unreadable_to_us(5130, 4321, False, str(root)) is False


def test_the_liveness_scan_never_walks_a_process_mapping(tmp_path, monkeypatch):
    root = make_proc_root(tmp_path)
    make_proc(root, 4800, anon_gib=60.0)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a liveness question must not price anything")

    monkeypatch.setattr(pi, "rollup_rss_gib", forbidden)
    monkeypatch.setattr(pi, "pool_gib", forbidden)
    assert liveness(root)["actors"] == 1
    # The one size the liveness walk reads is statm, one short line, and only
    # for a process whose environ refused. Both smaps reads stay forbidden,
    # including the rollup, which the kernel gates behind the same ptrace check
    # the environ read just failed.
    make_proc(root, 4801, argv=("/opt/bin/worker-shim",), env=None, anon_gib=60.0)
    assert liveness(root)["unreadable"] == 1


def test_the_liveness_flag_signals_nothing_and_prints_one_line(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("--liveness must never reach the kill ladder")

    monkeypatch.setattr(actor_reaper, "run_sweep", forbidden)
    monkeypatch.setattr(actor_reaper, "_signal", forbidden)
    monkeypatch.setattr(
        actor_reaper, "liveness",
        lambda **kwargs: {"schema": 1, "asked": list(kwargs["loop_addresses"])})
    assert actor_reaper.main(
        ["--liveness", "--sweep", "--loop-address", LOOP_ADDRESS]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"schema": 1, "asked": [LOOP_ADDRESS]}
