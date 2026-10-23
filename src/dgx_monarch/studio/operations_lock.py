"""The operations lock: up, down and token --rotate never stop a review or operation the companion admitted.

The companion holds ``operations.lock`` in its state directory shared, on a
descriptor of its own, for each review it prepares and each operation it
applies, from admission until the operation settles. ``dgxm studio up``,
``down`` and ``token --rotate`` take it exclusive, without waiting, before the
busy check, and keep it until their restart or stop has finished. While they
hold it the companion refuses new reviews and operations; while any review or
operation holds it they refuse as busy. Both sides create the file when it is
missing, so neither can miss the other's lock. flock locks belong to open
file descriptions, so the kernel decides every race, and a command that dies
releases its lock with its descriptor, so nothing can leave the companion
refusing. A companion that predates the lock never takes it, and the control
group check alone decides, as before. Standard library only, so the
aiohttp-free CLI can import it.
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import stat
from collections.abc import Iterator
from pathlib import Path

from . import unit_state
from .contracts import StudioError

OPERATIONS_LOCK = "operations.lock"
# The code the page already treats as "not accepted; review a new plan".
DRAINING = ("Studio is stopping or restarting for dgxm studio up, down or token --rotate, so it accepted nothing; "
            "try again once Studio answers.")
_FLAGS = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC


def hold(directory: Path) -> int:
    """A shared hold for one review or operation; refused while up, down or --rotate holds the lock."""
    descriptor = os.open(directory / OPERATIONS_LOCK, _FLAGS, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise StudioError("operation_unavailable", DRAINING, 409) from None
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def release(descriptor: int) -> None:
    with contextlib.suppress(OSError):
        os.close(descriptor)


def _exclusive(state: Path) -> tuple[int | None, bool]:
    """The lock's descriptor and whether this command now holds it exclusive.

    ``(None, True)`` when the state directory does not exist: a companion that
    never made its records holds no review to apply.
    """
    directory = unit_state._state_directory(state)
    if directory is None:
        return None, True
    try:
        descriptor = os.open(directory / OPERATIONS_LOCK, _FLAGS, 0o600)
    except OSError as exc:
        raise SystemExit(f"Cannot use the state directory {state}: {exc.strerror or exc}.") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise SystemExit(unit_state.STATE_PRIVATE)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return descriptor, False
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, True


@contextlib.contextmanager
def quiesced(state: Path, status: unit_state.UnitStatus) -> Iterator[None]:
    """Stop the running companion admitting work, then refuse while any runs; held until the block ends.

    The refusals are ``unit_state.BUSY``, with one line per process in the
    unit's control group other than its main process, and
    ``unit_state.BUSY_UNKNOWN`` when that group cannot be read.
    """
    descriptor, held = _exclusive(state)
    try:
        found = unit_state.busy_processes(status)
        if held and found is None:
            raise SystemExit(unit_state.BUSY_UNKNOWN)
        if not held or found:
            raise SystemExit("\n".join([unit_state.BUSY, *(found or [])]))
        yield
    finally:
        if descriptor is not None:
            release(descriptor)
