"""The sessions a canary smoke script starts, so that its runner can always stop them.

A smoke script that starts a child in its own session (a Worker loop the
script later kills by process group) records that group here, in the file the
runner names, the moment the child starts. The runner
(monarch_surface_canary._run_smoke) stops every recorded group once the
script has ended, however it ended: a script hung in native code runs neither
its ``finally`` nor a signal handler, and the runner's kill of the script's
own group does not reach another session. A script also arms its own
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


def start_session(argv: list[str], **kwargs: Any) -> subprocess.Popen:
    """Start ``argv`` leading its own session, and record its group for the runner."""
    process = subprocess.Popen(argv, start_new_session=True, **kwargs)
    path = os.environ.get(SESSIONS_ENV)
    if path:
        with open(path, "a", encoding="ascii") as record:
            record.write(f"{process.pid}\n")
            record.flush()
            os.fsync(record.fileno())
    return process


def recorded(path: str | os.PathLike[str]) -> list[int]:
    """The process groups a script recorded; none when it recorded nothing."""
    try:
        with open(path, encoding="ascii") as record:
            return [int(line) for line in record.read().split() if line.isdigit()]
    except FileNotFoundError:
        return []


def group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
