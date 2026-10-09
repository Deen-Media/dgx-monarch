"""Find, report and reap worker actor processes that outlived their client.

A teardown panic can disable torchmonarch's client keepalive and parent-death
cleanup, leaving actors holding retained pools (docs/TROUBLESHOOTING.md #51).

This policy serves lifecycle sweeps (``down``, ``up``, and ``restart``),
``dgxm reap``, doctor's read-only orphan report, the driver's post-timeout
``--liveness`` probe, and verified update's ``--report`` candidate count.

Targets are identified through procfs and rechecked by ``(pid, starttime)``,
UID, and marker before signalling. The hardcoded actor marker never appears in
this module's invocation, preventing self-matches.

Ownership markers are read from children: procfs does not reflect a process's
later ``os.environ`` writes. The driver puts its identity in the environment
inherited by children. A live owner or inconclusive evidence yields ``held``;
these candidates are reported but never signalled, to protect live renders.
"""
from __future__ import annotations

import json
import os
import re
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import actor_ledger
from . import proc_identity as pi

FAT_ANON_GIB = 4.0
SWEEP_TERM_WAIT_S = 10.0
SWEEP_KILL_WAIT_S = 5.0
SWEEP_POLL_S = 0.25

# One stdout line, parsed by `actor_sweep.report_result`. `pids` lists the
# processes selected for signalling (`-` when none), so a failing host names them.
SUMMARY_PREFIX = "ACTOR_SWEEP"
_SUMMARY_RE = re.compile(
    rf"{SUMMARY_PREFIX} reaped=(?P<reaped>0|[1-9][0-9]*) "
    r"killed=(?P<killed>0|[1-9][0-9]*) left=(?P<left>0|[1-9][0-9]*) "
    r"tracked=(?P<tracked>0|[1-9][0-9]*) gib=(?P<gib>(?:0|[1-9][0-9]*)\.[0-9]) "
    r"pids=(?P<pids>-|(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*)"
)

# Provenance, decided from live procfs evidence:
#   orphan      no live process could still stop it: reparented to init, a
#               subreaper adopted it, the driver that spawned it is dead, its
#               parent loop started after it, or its ledger row's loop is dead
#   loop-child  its parent is a live worker loop for a given address (any
#               address when none is given)
#   held        a live owner could still stop it. Never signalled, and never
#               named by the doctor row: the default when evidence cannot settle it
ORPHAN = "orphan"
LOOP_CHILD = "loop-child"
HELD = "held"


@dataclass(frozen=True)
class ActorProc:
    """A confirmed worker actor process and the evidence about it."""

    pid: int
    starttime: int
    ppid: int
    gib: float
    rss_gib: float
    established: int
    provenance: str
    age_s: float
    orphan_s: float | None
    argv_tail: str
    # Whether a marker-only scan keeps this one: carried, never re-derived.
    strictly_marked: bool = False

    @property
    def orphan(self) -> bool:
        return self.provenance == ORPHAN

    @property
    def dwell_text(self) -> str:
        """Time since orphaned, or process age when no tracker recorded `orphan_since`."""
        seconds = self.age_s if self.orphan_s is None else self.orphan_s
        clock = "age" if self.orphan_s is None else "orphaned for"
        return f"{clock} {int(seconds // 3600)}h{int(seconds % 3600 // 60):02d}m"

    @property
    def signal_text(self) -> str:
        signals = []
        if self.orphan:
            signals.append("orphan")
        if not self.established:
            signals.append("no client")
        return ", ".join(signals) or self.provenance

    def as_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid, "ppid": self.ppid, "gib": self.gib,
            "established": self.established, "provenance": self.provenance,
            "age_s": round(self.age_s, 1), "orphan_s": self.orphan_s,
        }


@dataclass(frozen=True)
class OrphanReport:
    """Read-only orphan report for the doctor row."""

    tracked: int
    attached: int
    orphans: tuple[ActorProc, ...]


def _ledger_rows(ledger: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    for row in ledger.get("actors") or []:
        try:
            rows[(int(row["pid"]), int(row["starttime"]))] = row
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def _loop_owned(argv: tuple[str, ...], addresses: tuple[str, ...]) -> bool:
    """True when argv is a worker loop for one of the addresses, or any loop when none is given."""
    address = pi.loop_argv_address(argv)
    if address is None:
        return False
    return not addresses or address in addresses


def scan(
    *,
    proc_root: str = pi.PROC_ROOT,
    ledger_file: str | None = None,
    uid: int | None = None,
    self_pid: int | None = None,
    loop_addresses: tuple[str, ...] = (),
    require_marker: bool = True,
    sizes: bool = True,
) -> list[ActorProc]:
    """Return confirmed worker actor processes with their provenance.

    ``sizes=False`` skips ``smaps_rollup`` reads for candidates and ``smaps`` reads
    for large candidates. Count-only callers can avoid these costs.
    """
    uid = os.getuid() if uid is None else uid
    self_pid = os.getpid() if self_pid is None else self_pid
    excluded = {self_pid, 1, 0, *pi.ancestry(self_pid, proc_root)}
    rows = _ledger_rows(actor_ledger.read_ledger(ledger_file, proc_root=proc_root))
    uptime = pi.uptime_s(proc_root)

    procs: dict[int, pi.ProcInfo] = {}
    candidates: list[pi.ProcInfo] = []
    for pid in pi.iter_pids(proc_root):
        stat = pi.read_stat(pid, proc_root)
        if stat is None:
            continue
        proc_uid = pi.read_uid(pid, proc_root)
        bootstrap_env = dgxm_owned = False
        owner: tuple[int, int] | None = None
        if proc_uid == uid:
            bootstrap_env, dgxm_owned, owner = pi.read_env_markers(pid, proc_root)
        argv = pi.read_argv(pid, proc_root)
        info = pi.ProcInfo(
            pid=pid, ppid=stat[0], starttime=stat[1], uid=proc_uid, argv=argv,
            actor_argv=pi.has_module_argv(argv, pi.ACTOR_MODULE),
            bootstrap_env=bootstrap_env, dgxm_owned=dgxm_owned, owner=owner,
        )
        procs[pid] = info
        if pid in excluded or proc_uid != uid or not info.marked:
            continue
        if require_marker and not dgxm_owned and (pid, info.starttime) not in rows:
            # Other Monarch applications can share the bootstrap marker without
            # DGXM_PYTHONPATH. --no-marker allows manual recovery when an orphan has
            # no ledger row and its environment is unreadable or lacks this marker.
            # dgxm never passes it (docs/TROUBLESHOOTING.md #51).
            continue
        candidates.append(info)

    counts = pi.established_counts([info.pid for info in candidates], proc_root)
    return [
        _classify(info, procs, rows, loop_addresses, proc_root, uptime,
                  counts.get(info.pid, 0), sizes)
        for info in candidates
    ]


def _adopted(info: pi.ProcInfo, proc_root: str) -> bool:
    """True when a subreaper adopted this process, so its spawner is gone.

    If neither process called ``setsid`` after the fork, a parent in another
    session did not spawn this one (``proc_identity.read_session``). Both
    readings must succeed: an unreadable session is not evidence of anything,
    and this decides whether a process may be signalled.
    """
    child = pi.read_session(info.pid, proc_root)
    parent = pi.read_session(info.ppid, proc_root)
    return child is not None and parent is not None and child != parent


def _classify(
    info: pi.ProcInfo,
    procs: dict[int, pi.ProcInfo],
    rows: dict[tuple[int, int], dict[str, Any]],
    addresses: tuple[str, ...],
    proc_root: str,
    uptime: float,
    established: int,
    sizes: bool = True,
) -> ActorProc:
    parent = procs.get(info.ppid)
    if info.ppid <= 1 or parent is None:
        provenance = ORPHAN
    elif _loop_owned(parent.argv, addresses):
        # An actor cannot be a child of a loop that started after it: that is
        # a restarted loop (or a recycled pid) inheriting nothing.
        provenance = ORPHAN if parent.starttime > info.starttime else LOOP_CHILD
    elif info.owner is not None:
        # The inherited marker identifies the spawning driver by (pid, starttime).
        # Keep the actor while that driver lives; otherwise classify it as orphaned.
        provenance = (
            HELD if pi.alive(info.owner[0], info.owner[1], proc_root) else ORPHAN)
    elif pi.loop_argv_address(parent.argv) is not None:
        provenance = HELD
    elif _adopted(info, proc_root):
        # Subreaper adoption indicates the original parent has exited.
        provenance = ORPHAN
    else:
        # Older drivers may omit the ownership marker; a same-session parent
        # may still own a live render. Keep it held, including during the
        # driver's mesh_attach auto-heal restart, to avoid supervision failures.
        provenance = HELD

    row = rows.get((info.pid, info.starttime))
    orphan_s: float | None = None
    if row is not None:
        since = row.get("orphan_since")
        if isinstance(since, (int, float)):
            orphan_s = max(time.time() - float(since), 0.0)
        # Check (pid, starttime) to detect PID reuse. Ledger rows only describe
        # loop descendants, so a dead recorded loop establishes orphanhood.
        # Missing loop_pid defaults to alive: /proc/0 would otherwise make a
        # truncated row authorize a kill. This is stricter than the dwell clock.
        loop_pid = int(row.get("loop_pid") or 0)
        loop_alive = loop_pid <= 0 or pi.alive(
            loop_pid, int(row.get("loop_starttime") or 0), proc_root)
        if provenance != LOOP_CHILD and not loop_alive:
            provenance = ORPHAN

    rss_gib = pi.rollup_rss_gib(info.pid, proc_root) if sizes else 0.0
    gib = (pi.pool_gib(info.pid, proc_root)
           if sizes and rss_gib >= FAT_ANON_GIB else 0.0)
    return ActorProc(
        strictly_marked=info.dgxm_owned or (info.pid, info.starttime) in rows,
        pid=info.pid, starttime=info.starttime, ppid=info.ppid, gib=gib,
        rss_gib=rss_gib, established=established, provenance=provenance,
        age_s=pi.age_s(info.starttime, uptime), orphan_s=orphan_s,
        argv_tail=info.argv_tail,
    )


def select_targets(
    candidates: list[ActorProc],
    *,
    all_loop_children: bool = False,
    grace_s: float = 0.0,
) -> list[ActorProc]:
    """Select actors that may be signalled.

    Select orphans, and loop children only with ``--all-loop-children``. ``down``
    passes that flag after stopping the loop, when its children can no longer be
    legitimate. Never select a ``held`` candidate.

    ``ActorProc.established`` is diagnostic only: an actor can retain sockets to
    other peers after its loop stops, so a socket-count filter would skip it.
    """
    targets: list[ActorProc] = []
    for proc in candidates:
        if proc.provenance == HELD:
            continue
        if proc.provenance == LOOP_CHILD and not all_loop_children:
            continue
        # Without a tracker timestamp, use process age. This bounds lifetime,
        # not time since orphaning; --grace is therefore less conservative for
        # untracked actors (documented in help and TROUBLESHOOTING.md #51).
        dwell = proc.orphan_s if proc.orphan_s is not None else proc.age_s
        if grace_s > 0.0 and dwell < grace_s:
            continue
        targets.append(proc)
    return targets


def _still_alive(
    targets: list[ActorProc], wait_s: float, proc_root: str, sleep: Callable[[float], None]
) -> list[ActorProc]:
    remaining = list(targets)
    for _ in range(max(int(wait_s / SWEEP_POLL_S), 1)):
        remaining = [t for t in remaining if pi.alive(t.pid, t.starttime, proc_root)]
        if not remaining:
            break
        sleep(SWEEP_POLL_S)
    return [t for t in remaining if pi.alive(t.pid, t.starttime, proc_root)]


def _signal(
    target: ActorProc, sig: int, killer: Callable[[int, int], None],
    *, proc_root: str, uid: int,
) -> None:
    """Recheck identity, UID and marker against procfs, then signal.

    A process may exit and its PID may be reused between the initial scan and this
    call. Never signal using the scan's identity evidence alone.
    """
    if not pi.confirmed(target.pid, target.starttime, uid, proc_root):
        return
    try:
        killer(target.pid, sig)
    except OSError:
        # It exited between confirmation and the signal, which is the goal.
        pass


def run_sweep(
    *,
    proc_root: str = pi.PROC_ROOT,
    ledger_file: str | None = None,
    uid: int | None = None,
    self_pid: int | None = None,
    loop_addresses: tuple[str, ...] = (),
    all_loop_children: bool = False,
    grace_s: float = 0.0,
    dry_run: bool = False,
    require_marker: bool = True,
    killer: Callable[[int, int], None] = os.kill,
    sleep: Callable[[float], None] = time.sleep,
    term_wait_s: float = SWEEP_TERM_WAIT_S,
    kill_wait_s: float = SWEEP_KILL_WAIT_S,
) -> dict[str, Any]:
    """SIGTERM, wait, SIGKILL survivors, wait, report. Idempotent."""
    uid = os.getuid() if uid is None else uid
    candidates = scan(
        proc_root=proc_root, ledger_file=ledger_file, uid=uid, self_pid=self_pid,
        loop_addresses=loop_addresses, require_marker=require_marker,
    )
    targets = select_targets(
        candidates, all_loop_children=all_loop_children, grace_s=grace_s)
    summary: dict[str, Any] = {
        "tracked": len(candidates),
        "gib": round(sum(t.gib for t in targets), 1),
        "pids": [t.pid for t in targets],
        "reaped": 0, "killed": 0, "left": len(targets),
    }
    if dry_run or not targets:
        # A dry run counts every selected process as `left`, since none was
        # signalled; `dgxm reap --dry-run` still exits 0.
        return summary

    for target in targets:
        _signal(target, signal.SIGTERM, killer, proc_root=proc_root, uid=uid)
    survivors = _still_alive(targets, term_wait_s, proc_root, sleep)
    for target in survivors:
        _signal(target, signal.SIGKILL, killer, proc_root=proc_root, uid=uid)
    left = _still_alive(survivors, kill_wait_s, proc_root, sleep)
    summary.update(
        reaped=len(targets) - len(left), killed=len(survivors), left=len(left))
    actor_ledger.prune_dead_rows(ledger_file, proc_root=proc_root)
    return summary


def orphan_report(
    *,
    proc_root: str = pi.PROC_ROOT,
    ledger_file: str | None = None,
    uid: int | None = None,
    self_pid: int | None = None,
    min_gib: float = FAT_ANON_GIB,
) -> OrphanReport:
    """Report large actor processes with no live owner, without signalling.

    Provenance determines selection; socket counts are diagnostic only. A live
    local-mode rank may have no TCP sockets, while a loop child can keep its loop
    socket after the driver dies. Actors under a live loop remain loop children:
    the client lease or a later ``down`` sweep handles them.

    Use total RSS from ``smaps_rollup`` for the size threshold and report the
    giant-mapping pool separately. Residency spread across mappings below the
    pool rule's 4 GiB floor must still count.
    """
    candidates = scan(
        proc_root=proc_root, ledger_file=ledger_file, uid=uid, self_pid=self_pid)
    orphans = tuple(
        proc for proc in candidates if proc.rss_gib >= min_gib and proc.orphan)
    return OrphanReport(
        tracked=len(candidates),
        attached=sum(1 for proc in candidates if proc.established > 0),
        orphans=orphans,
    )


LIVENESS_SCHEMA = 1
LIVENESS_MAX_PIDS = 32


def liveness(
    *,
    proc_root: str = pi.PROC_ROOT,
    ledger_file: str | None = None,
    uid: int | None = None,
    self_pid: int | None = None,
    loop_addresses: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Report whether this host still holds worker actors, without signals or smaps.

    Count every actor regardless of provenance or address attribution: an orphan
    may have lost both and must still block driver replacement. Disable the marker
    requirement so unreadable environments or missing ledger rows cannot hide
    actors; return the stricter count separately as ``actors_marked``.

    Unidentifiable processes count as unreadable, so the caller reports unknown
    rather than gone. This includes empty cmdline/environ reads and sufficiently
    large processes with a refused environ read and no Python module in argv
    (``proc_identity.unreadable_to_us``).
    """
    uid = os.getuid() if uid is None else uid
    self_pid = os.getpid() if self_pid is None else self_pid
    # One scan gives both `actors` and `actors_marked`; a second scan would
    # double every procfs read on a starved host. A zombie is not an actor here:
    # it has already released every page, so counting one would answer alive.
    actors = [proc for proc in scan(
        proc_root=proc_root, ledger_file=ledger_file, uid=uid,
        self_pid=self_pid, loop_addresses=loop_addresses,
        require_marker=False, sizes=False)
        if pi.read_state(proc.pid, proc_root) != "Z"]
    marked = sum(1 for proc in actors if proc.strictly_marked)

    excluded = {self_pid, 1, 0, *pi.ancestry(self_pid, proc_root)}
    zombies = unreadable = 0
    for pid in pi.iter_pids(proc_root):
        if pid in excluded:
            continue
        stat = pi.read_stat(pid, proc_root)
        if stat is None or pi.read_uid(pid, proc_root) != uid:
            continue
        empty = pi.reads_empty(pid, proc_root)
        if pi.read_state(pid, proc_root) == "Z":
            info = pi.proc_info(pid, proc_root)
            if empty or (info is not None and info.marked):
                zombies += 1
            continue
        if pi.unreadable_to_us(pid, stat[1], empty, proc_root):
            unreadable += 1

    ledger = actor_ledger.read_ledger(ledger_file, proc_root=proc_root)
    rows_alive = sum(
        1 for row in ledger.get("actors") or []
        if pi.alive_and_not_zombie(row["pid"], row["starttime"], proc_root))
    boot = pi.boot_id(proc_root)
    written = float(ledger.get("written_at") or 0.0)
    loop = ledger.get("loop")
    loop_alive: bool | None = None
    if isinstance(loop, dict):
        try:
            loop_alive = pi.alive_and_not_zombie(
                int(loop.get("pid") or 0), int(loop.get("starttime") or 0), proc_root)
        except (TypeError, ValueError):
            loop_alive = None
    return {
        "schema": LIVENESS_SCHEMA,
        "actors": len(actors),
        "actors_marked": marked,
        # Every own-uid zombie, not only actor zombies: the kernel frees a
        # zombie's mm, so both reads come back empty and nothing identifies it
        # any more. A stray zombie from an unrelated command raises this count.
        "zombies": zombies,
        "unreadable": unreadable,
        "ledger_rows_alive": rows_alive,
        "ledger_boot_ok": bool(boot) and str(ledger.get("boot_id", "")) == boot,
        # Age of the tracker's last write. Negative when no ledger has been
        # written at all, which is not the same as an old one.
        "ledger_age_s": (round(max(time.time() - written, 0.0), 1)
                         if written > 0 else -1.0),
        "loop_alive": loop_alive,
        "pids": [
            {"pid": proc.pid, "provenance": proc.provenance,
             "argv_tail": proc.argv_tail}
            for proc in actors[:LIVENESS_MAX_PIDS]
        ],
    }


def summary_fields(summary: dict[str, Any]) -> str:
    pids = ",".join(str(pid) for pid in summary.get("pids") or []) or "-"
    return (
        f"reaped={summary['reaped']} killed={summary['killed']} "
        f"left={summary['left']} tracked={summary['tracked']} "
        f"gib={float(summary['gib']):.1f} pids={pids}"
    )


def format_summary(summary: dict[str, Any]) -> str:
    return f"{SUMMARY_PREFIX} {summary_fields(summary)}"


def parse_summary(text: str) -> dict[str, Any] | None:
    """Parse one exact final contract line; None means no frame was emitted."""
    lines = (text or "").splitlines()
    frames = [line for line in lines if line.startswith(SUMMARY_PREFIX)]
    if not frames:
        return None
    if len(frames) != 1 or frames[0] != lines[-1]:
        raise ValueError("actor sweep summary must be unique and final")
    match = _SUMMARY_RE.fullmatch(frames[0])
    if match is None:
        raise ValueError("actor sweep summary is malformed")
    fields = match.groupdict()
    try:
        parsed: dict[str, Any] = {
            key: int(fields[key]) for key in ("reaped", "killed", "left", "tracked")
        }
        parsed["gib"] = float(fields["gib"])
        parsed["pids"] = [] if fields["pids"] == "-" else [
            int(pid) for pid in fields["pids"].split(",")]
    except (OverflowError, ValueError):
        raise ValueError("actor sweep summary is malformed") from None
    return parsed


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="dgx_monarch.cli.actor_reaper",
        description="Report or reap worker actor processes that outlived their client.")
    parser.add_argument("--sweep", action="store_true", help="signal the selected processes")
    parser.add_argument("--report", action="store_true", help="print the candidates as one JSON line and exit 0")
    parser.add_argument("--liveness", action="store_true",
                        help="print one JSON liveness line and exit 0; signals nothing")
    parser.add_argument("--all-loop-children", action="store_true",
                        help="also select live loop children (only after the loop is stopped)")
    parser.add_argument("--loop-address", action="append", default=[],
                        help="a configured Worker service address (repeatable)")
    parser.add_argument("--grace", type=float, default=0.0,
                        help="minimum seconds a process must have been orphaned, or, "
                             "when no tracker recorded it, alive")
    parser.add_argument("--dry-run", action="store_true", help="select but signal nothing")
    parser.add_argument("--no-marker", action="store_true",
                        help="also accept a monarch actor with no dgx-monarch environ marker and no ledger row")
    args = parser.parse_args(argv)

    addresses = tuple(args.loop_address)
    if args.liveness:
        # Read only, on the same rule as --report: a liveness question must
        # never be able to change what it is measuring.
        print(json.dumps(liveness(loop_addresses=addresses)))
        return 0
    if args.report or not args.sweep:
        candidates = scan(loop_addresses=addresses, require_marker=not args.no_marker)
        print(json.dumps({"candidates": [proc.as_dict() for proc in candidates]}))
        return 0
    summary = run_sweep(
        loop_addresses=addresses, all_loop_children=args.all_loop_children,
        grace_s=args.grace, dry_run=args.dry_run, require_marker=not args.no_marker)
    print(format_summary(summary))
    return 0 if args.dry_run or not summary["left"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
