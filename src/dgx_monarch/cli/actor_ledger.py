"""Persist worker actor identities across loop restarts.

The driver's meshes and the host agent's in-memory PID table do not preserve
actor PIDs after a loop restart. A daemon thread beside the blocking
``run_worker_loop_forever`` call records live descendants every
:data:`TRACKER_TICK_S` seconds and retains live rows from earlier loops on the
same boot. This module only observes; ``cli/actor_reaper.py`` decides what to
signal.

Each row identifies a process by ``(pid, starttime)`` to detect PID reuse.
``boot_id`` scopes the file because start times can repeat across boots.
Writes use a unique temporary file and ``os.replace``, with a 0700 directory
and 0600 file. A writer killed after it creates its temporary file and before
``os.replace`` leaves the file behind, so each write also removes temporary
files whose writer has exited. The file name carries the writer's
``(pid, starttime)``, so the sweep judges it by the same identity check as a
row, with no clock. Missing, corrupt, or foreign-boot ledgers read as empty,
allowing ``dgxm down`` to work on hosts without a usable ledger.

``orphan_since`` records when the owning loop first failed its identity check.
It establishes provenance, not attachment state. Socket counts are read live
through ``cli/proc_identity.established_counts`` when needed.

The loop record also carries the id the loop serves on this start
(``worker_proc_id.py``). ``publish_loop`` writes it before the loop serves, so
the listener-generation probe never finds a listener whose record names the
previous loop. A sweep's ``prune_dead_rows`` writes back the loop record it
read; when that read predates the loop's first write, the next tracker tick
restores the record, because its payload then differs from the file.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..log import get_logger
from . import proc_identity

log = get_logger(__name__)

LEDGER_VERSION = 1
# Share the ledger path between tracker and sweep readers. Like the adjacent
# worker log, it does not use XDG_STATE_HOME.
LEDGER_REL = os.path.join(".local", "state", "dgx-monarch", "actor-procs.json")
TRACKER_TICK_S = 5.0
# Rewrite an unchanged ledger at most this often, so ``written_at`` stays a
# useful liveness signal without a disk write every tick.
HEARTBEAT_S = 60.0
MAX_ROWS = 512
_DIR_MODE = 0o700
_FILE_MODE = 0o600
# A temporary file name of either shape the sweep judges: ``<pid>.<starttime>``
# or, from earlier releases, ``<pid>`` alone, then the thread ident.
_TEMP_NAME = r"\.(\d+)(?:\.(\d+))?\.\d+\.tmp"


def ledger_path(home: str | None = None) -> str:
    return os.path.join(home or os.path.expanduser("~"), LEDGER_REL)


def empty_ledger(boot: str = "") -> dict[str, Any]:
    return {
        "version": LEDGER_VERSION,
        "boot_id": boot,
        "written_at": 0.0,
        "loop": None,
        "actors": [],
    }


def _coerce_row(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    try:
        pid = int(raw["pid"])
        starttime = int(raw["starttime"])
    except (KeyError, TypeError, ValueError):
        return None
    if pid <= 1:
        return None
    orphan_since = raw.get("orphan_since")
    try:
        orphan = None if orphan_since is None else float(orphan_since)
    except (TypeError, ValueError):
        orphan = None
    return {
        "pid": pid,
        "starttime": starttime,
        "address": str(raw.get("address", "")),
        "loop_pid": _int_or_zero(raw.get("loop_pid")),
        "loop_starttime": _int_or_zero(raw.get("loop_starttime")),
        "marker": str(raw.get("marker", "cmdline")),
        "first_seen": _float_or_zero(raw.get("first_seen")),
        "last_seen": _float_or_zero(raw.get("last_seen")),
        "orphan_since": orphan,
    }


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def read_ledger(
    path: str | None = None,
    *,
    home: str | None = None,
    boot: str | None = None,
    proc_root: str = proc_identity.PROC_ROOT,
) -> dict[str, Any]:
    """The ledger for this boot, or an empty one. Never raises."""
    running_boot = proc_identity.boot_id(proc_root) if boot is None else boot
    target = ledger_path(home) if path is None else path
    try:
        with open(target, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError):
        return empty_ledger(running_boot)
    if not isinstance(raw, dict) or raw.get("version") != LEDGER_VERSION:
        return empty_ledger(running_boot)
    if str(raw.get("boot_id", "")) != str(running_boot):
        # Start times repeat across boots and so do pids: a foreign-boot ledger
        # could name an unrelated process. Discard it whole.
        return empty_ledger(running_boot)
    rows = [row for row in map(_coerce_row, raw.get("actors") or []) if row is not None]
    loop = raw.get("loop") if isinstance(raw.get("loop"), dict) else None
    return {
        "version": LEDGER_VERSION,
        "boot_id": str(running_boot),
        "written_at": _float_or_zero(raw.get("written_at")),
        "loop": loop,
        "actors": rows,
    }


def write_ledger(
    ledger: dict[str, Any],
    path: str | None = None,
    *,
    home: str | None = None,
    proc_root: str = proc_identity.PROC_ROOT,
) -> bool:
    """Publish a ledger atomically. Returns False on OSError instead of raising.

    The temporary file is ``<ledger>.<pid>.<starttime>.<thread ident>.tmp``,
    with the writer's own start time read under ``proc_root``. A writer that
    cannot read its own stat there, which happens only under a synthetic root
    or an unusable ``/proc``, puts ``nostat`` in place of the start time. No
    sweep matches that name, so no sweep under any root removes the file of
    a live writer. Every call then removes the temporary files that dead
    writers left beside the ledger (:func:`sweep_dead_temps`), whether its
    own write lands, fails, or raises.
    """
    target = ledger_path(home) if path is None else path
    directory = os.path.dirname(target)
    pid = os.getpid()
    own = proc_identity.read_stat(pid, proc_root)
    identity = f"{pid}.nostat" if own is None else f"{pid}.{own[1]}"
    temp = f"{target}.{identity}.{threading.get_ident()}.tmp"
    try:
        os.makedirs(directory, mode=_DIR_MODE, exist_ok=True)
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, _FILE_MODE)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(ledger, handle)
                handle.flush()
                os.fsync(handle.fileno())
            # A failed rename must not strand the file: the next write from
            # this thread reuses the name, so O_EXCL would refuse every later
            # write, and the sweep keeps the file while its writer lives.
            os.replace(temp, target)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temp)
            raise
        written = True
    except OSError:
        written = False
    finally:
        sweep_dead_temps(target, proc_root=proc_root)
    return written


def _temp_writer(name: str, base: str) -> tuple[int, int | None] | None:
    """The writer ``(pid, starttime)`` in a ledger temporary file name.

    Two shapes count, each a whole name in ASCII digits:
    ``<ledger>.<pid>.<starttime>.<thread ident>.tmp``, which ``write_ledger``
    writes, and ``<ledger>.<pid>.<thread ident>.tmp``, which earlier releases
    wrote and gives None for the start time. Any other name gives None.
    """
    match = re.fullmatch(re.escape(base) + _TEMP_NAME, name, re.ASCII)
    if match is None:
        return None
    starttime = match.group(2)
    return int(match.group(1)), None if starttime is None else int(starttime)


def sweep_dead_temps(
    target: str, *, proc_root: str = proc_identity.PROC_ROOT
) -> int:
    """Remove temporary files beside ``target`` whose writer has exited.

    A writer killed between creating its file and renaming it leaves the file
    behind, for example the tracker thread when a stop kills the Worker
    service mid-write. The name carries the writer's identity, so identity
    decides and the sweep reads no clock and no file time. The sweep removes
    a name with a start time once ``proc_identity.alive`` rejects its
    ``(pid, starttime)``: a live writer passes, and a later process that took
    the pid fails, since its start time differs. That is exact within a
    boot. A process on a later boot with the same pid and start tick keeps
    the file until it exits. It removes a name in the older shape, with no
    start time, only once its pid names no process, so if a later process
    took that pid, the file stays until that process exits. Names of any
    other shape stay, ``nostat`` names included. Only ``proc_root`` answers.
    Returns the count removed and never raises.
    """
    directory, base = os.path.split(target)
    try:
        names = os.listdir(directory or ".")
    except OSError:
        return 0
    removed = 0
    for name in names:
        writer = _temp_writer(name, base)
        if writer is None:
            continue
        pid, starttime = writer
        if starttime is None:
            exited = proc_identity.read_stat(pid, proc_root) is None
        else:
            exited = not proc_identity.alive(pid, starttime, proc_root)
        if not exited:
            continue
        try:
            os.unlink(os.path.join(directory, name))
        except OSError:
            continue  # another sweep removed it first, or it will not go
        removed += 1
    log.debug("actor ledger: removed %d dead temporary file(s) beside %s",
              removed, target)
    return removed


def prune_dead_rows(
    path: str | None = None, *, proc_root: str = proc_identity.PROC_ROOT
) -> None:
    """Remove rows for exited processes so repeated sweeps have no effect.

    The tracker also removes dead rows on each tick. If it overwrites a concurrent
    prune, the next tick repeats the removal. Both writers replace the file
    atomically, so readers never see a partial write.
    """
    ledger = read_ledger(path, proc_root=proc_root)
    rows = [row for row in ledger["actors"]
            if proc_identity.alive(row["pid"], row["starttime"], proc_root)]
    if len(rows) != len(ledger["actors"]):
        ledger["actors"] = rows
        write_ledger(ledger, path, proc_root=proc_root)


def loop_identity(
    address: str, *, pid: int | None = None, proc_root: str = proc_identity.PROC_ROOT,
    proc_id: str | None = None,
) -> dict[str, Any]:
    """Return this loop's identity for actor ownership records.

    ``proc_id`` is the id value the loop serves; a record without one is a
    loop that serves the bare address.
    """
    loop_pid = os.getpid() if pid is None else int(pid)
    stat = proc_identity.read_stat(loop_pid, proc_root)
    identity: dict[str, Any] = {
        "pid": loop_pid,
        "starttime": 0 if stat is None else stat[1],
        "address": str(address),
    }
    if proc_id is not None:
        identity["proc_id"] = str(proc_id)
    return identity


def _children_index(proc_root: str) -> dict[int, list[int]]:
    index: dict[int, list[int]] = {}
    for pid in proc_identity.iter_pids(proc_root):
        stat = proc_identity.read_stat(pid, proc_root)
        if stat is None:
            continue
        index.setdefault(stat[0], []).append(pid)
    return index


def scan_loop_actors(
    loop: dict[str, Any],
    *,
    proc_root: str = proc_identity.PROC_ROOT,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Return ledger rows for live actor descendants of this loop.

    Include indirect descendants so launcher processes cannot hide actors. Match
    ``-m monarch._src.actor.bootstrap_main`` as adjacent argv tokens; a mention
    of the module elsewhere in a command is not enough.
    """
    moment = time.time() if now is None else float(now)
    index = _children_index(proc_root)
    frontier = [int(loop.get("pid", 0))]
    seen: set[int] = set(frontier)
    rows: list[dict[str, Any]] = []
    while frontier:
        for child in index.get(frontier.pop(), ()):
            if child in seen:
                continue
            seen.add(child)
            frontier.append(child)
            info = proc_identity.proc_info(child, proc_root)
            if info is None or not info.actor_argv:
                continue
            rows.append({
                "pid": info.pid,
                "starttime": info.starttime,
                "address": str(loop.get("address", "")),
                "loop_pid": int(loop.get("pid", 0)),
                "loop_starttime": int(loop.get("starttime", 0)),
                "marker": "cmdline",
                "first_seen": moment,
                "last_seen": moment,
                "orphan_since": None,
            })
    return rows


def merge_rows(
    previous: Iterable[dict[str, Any]],
    observed: Sequence[dict[str, Any]],
    *,
    now: float,
    proc_root: str = proc_identity.PROC_ROOT,
) -> list[dict[str, Any]]:
    """Adopt, refresh, age, and prune one tick's worth of rows."""
    kept: dict[tuple[int, int], dict[str, Any]] = {}
    for row in previous:
        key = (row["pid"], row["starttime"])
        if proc_identity.alive(row["pid"], row["starttime"], proc_root):
            kept[key] = dict(row)
    for row in observed:
        key = (row["pid"], row["starttime"])
        existing = kept.get(key)
        if existing is None:
            kept[key] = dict(row)
            continue
        existing.update({
            "address": row["address"],
            "loop_pid": row["loop_pid"],
            "loop_starttime": row["loop_starttime"],
            "marker": row["marker"],
            "last_seen": row["last_seen"],
            "orphan_since": None,
        })
    fresh = {(row["pid"], row["starttime"]) for row in observed}
    for key, row in kept.items():
        if key in fresh:
            continue
        owner_alive = row["loop_pid"] > 0 and proc_identity.alive(
            row["loop_pid"], row["loop_starttime"], proc_root)
        if not owner_alive and row.get("orphan_since") is None:
            # The loop that spawned it is gone: the dwell clock starts here.
            row["orphan_since"] = now
    rows = sorted(kept.values(), key=lambda row: (row["pid"], row["starttime"]))
    if len(rows) > MAX_ROWS:
        rows = sorted(rows, key=lambda row: row["last_seen"], reverse=True)[:MAX_ROWS]
        rows.sort(key=lambda row: (row["pid"], row["starttime"]))
    return rows


def _payload(ledger: dict[str, Any]) -> str:
    """Return ledger fields whose changes require a write.

    ``written_at`` and ``last_seen`` change every tick; persist them only alongside
    another change or at the heartbeat interval.
    """
    stable = {key: value for key, value in ledger.items() if key != "written_at"}
    stable["actors"] = [
        {key: value for key, value in row.items() if key != "last_seen"}
        for row in ledger.get("actors") or []
    ]
    return json.dumps(stable, sort_keys=True)


def tick(
    address: str,
    *,
    loop_pid: int | None = None,
    proc_root: str = proc_identity.PROC_ROOT,
    path: str | None = None,
    home: str | None = None,
    now: float | None = None,
    writer: Callable[..., bool] = write_ledger,
    proc_id: str | None = None,
) -> dict[str, Any]:
    """One tracker observation: read, adopt, refresh, write when it changed."""
    moment = time.time() if now is None else float(now)
    boot = proc_identity.boot_id(proc_root)
    current = read_ledger(path, home=home, boot=boot, proc_root=proc_root)
    loop = loop_identity(address, pid=loop_pid, proc_root=proc_root, proc_id=proc_id)
    merged = {
        "version": LEDGER_VERSION,
        "boot_id": boot,
        "written_at": moment,
        "loop": loop,
        "actors": merge_rows(
            current["actors"],
            scan_loop_actors(loop, proc_root=proc_root, now=moment),
            now=moment,
            proc_root=proc_root,
        ),
    }
    stale = moment - current["written_at"] >= HEARTBEAT_S
    if stale or _payload(merged) != _payload(current):
        writer(merged, path, home=home, proc_root=proc_root)
    return merged


def publish_loop(
    address: str,
    *,
    proc_id: str,
    loop_pid: int | None = None,
    proc_root: str = proc_identity.PROC_ROOT,
    path: str | None = None,
    home: str | None = None,
) -> bool:
    """Write this loop's record, served id included, before the loop serves.

    The tracker thread's first tick can land after the listener exists, and
    the driver's probe would then read the previous loop's record. Returns
    whether the file now names this loop with this id; never raises.
    """
    try:
        tick(address, loop_pid=loop_pid, proc_root=proc_root, path=path,
             home=home, proc_id=proc_id)
        loop = read_ledger(path, home=home, proc_root=proc_root)["loop"]
    except Exception:
        return False
    owner = os.getpid() if loop_pid is None else int(loop_pid)
    return (isinstance(loop, dict) and loop.get("proc_id") == proc_id
            and loop.get("pid") == owner)


def _track(address: str, tick_s: float, sleeper: Callable[[float], None],
           **kwargs: Any) -> None:
    while True:
        try:
            tick(address, **kwargs)
        except BaseException:
            pass  # a failed tick must not end the tracker thread
        sleeper(tick_s)


def start_tracker(
    address: str,
    *,
    tick_s: float = TRACKER_TICK_S,
    sleeper: Callable[[float], None] = time.sleep,
    thread_factory: Any = threading.Thread,
    **kwargs: Any,
) -> Any:
    """Start the loop-side tracker as a daemon thread. Returns the thread."""
    thread = thread_factory(
        target=_track,
        args=(address, tick_s, sleeper),
        kwargs=kwargs,
        name="dgxm-actor-ledger",
        daemon=True,
    )
    thread.start()
    return thread
