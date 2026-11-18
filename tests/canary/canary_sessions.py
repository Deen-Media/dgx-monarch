"""The sessions a canary smoke script starts, so that its runner can always stop them.

A smoke script that starts a child in its own session (a Worker loop the
script later kills by process group) records that group here, in the file the
runner names, the moment the child starts, and removes the record once its own
stop of that group is confirmed (stop_sessions). The runner
(monarch_surface_canary._run_smoke) stops every group still recorded once the
script has ended, however it ended: a script hung in native code runs neither
its ``finally`` nor a signal handler, and the runner's kill of the script's
own group does not reach another session. After a clean run nothing is
recorded, so the runner signals no group, and never one whose id an unrelated
process has taken since. A script also arms its own
deadline, shortly before the runner's limit, so that its cleanup, not the
hard kill, is the usual way a stuck run ends. Monarch's procs need no record:
each leads its own process group but exits within a second of its parent's
death (a loop's, or the script's for this_host()). Standard library only.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any

SESSIONS_ENV = "DGXM_CANARY_SESSIONS"
DEADLINE_ENV = "DGXM_CANARY_DEADLINE_S"
DEADLINE_EXIT = 3
_LOCK = threading.Lock()  # the main thread and the deadline's both stop and write the record
_stopped: set[int] = set()  # groups this script's own stop confirmed gone; never signalled again


def _write(path: str, groups: list[int]) -> None:
    """Replace the record with ``groups``, in one rename."""
    staging = f"{path}.next"
    with open(staging, "w", encoding="ascii") as record:
        record.write("".join(f"{group}\n" for group in groups))
        record.flush()
        os.fsync(record.fileno())
    os.replace(staging, path)


def start_session(argv: list[str], **kwargs: Any) -> subprocess.Popen:
    """Start ``argv`` leading its own session, and record its group for the runner."""
    process = subprocess.Popen(argv, start_new_session=True, **kwargs)
    path = os.environ.get(SESSIONS_ENV)
    with _LOCK:
        _stopped.discard(process.pid)  # a new group under an id an earlier stop freed
        if path:
            _write(path, [*recorded(path), process.pid])
    return process


def stop_sessions(groups: Iterable[int], wait_s: float = 10.0) -> list[int]:
    """A script's own stop of groups it started; returns the ones still alive.

    Each group is signalled until this stop confirms it gone, and then never
    again, by the script or by the runner: its record is removed at once, so a
    later stop in the script, or the runner's after the script ends, cannot hit
    an unrelated process that has since taken the id.
    """
    path = os.environ.get(SESSIONS_ENV)
    with _LOCK:
        groups = [group for group in groups if group not in _stopped]
        alive = stop_groups(groups, wait_s)
        gone = {group for group in groups if group not in alive}
        _stopped.update(gone)
        if path and gone:
            _write(path, [group for group in recorded(path) if group not in gone])
    return alive


def recorded(path: str | os.PathLike[str]) -> list[int]:
    """The process groups a script recorded; none when it recorded nothing."""
    try:
        with open(path, encoding="ascii") as record:
            return [int(line) for line in record.read().split() if line.isdigit()]
    except FileNotFoundError:
        return []


def group_alive(group: int) -> bool:
    """Whether a process other than a zombie is still in ``group``.

    A killed child stays a zombie until its parent reaps it, and signal 0
    reaches a zombie, so on Linux this reads /proc instead.
    """
    if not os.path.isdir("/proc"):
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", encoding="ascii", errors="replace") as stat:
                fields = stat.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if fields[0] != "Z" and int(fields[2]) == group:
            return True
    return False


def stop_groups(groups: Iterable[int], wait_s: float = 10.0) -> list[int]:
    """SIGKILL each group, wait until each is gone, and return the ones still alive."""
    groups = [group for group in groups if group > 1]
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + wait_s
    alive = [group for group in groups if group_alive(group)]
    while alive and time.monotonic() < deadline:
        time.sleep(0.1)
        alive = [group for group in alive if group_alive(group)]
    return alive


def arm_deadline(cleanup: Callable[[], object]) -> threading.Timer | None:
    """End this script through ``cleanup`` once the runner's deadline for it passes."""
    value = os.environ.get(DEADLINE_ENV)
    if not value:
        return None

    def expire() -> None:
        try:
            cleanup()
        finally:
            os._exit(DEADLINE_EXIT)

    timer = threading.Timer(float(value), expire)
    timer.daemon = True
    timer.start()
    return timer
