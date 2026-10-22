"""What the user manager and the state directory say about the always-on Studio.

Every probe here keeps three answers apart: an observed yes, an observed no,
and no answer, which callers turn into ``probe_certainty.UNKNOWN_EXIT``. The
lock probe never creates a file, the busy check reads the unit's control
group, and readiness counts a token file or sign-in page only when it was
written after the unit's main process started. Standard library only, so the
aiohttp-free CLI can import it.
"""
from __future__ import annotations

import contextlib
import errno
import fcntl
import os
import re
import shutil
import socket
import sqlite3
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .contracts import StudioError
from .driver_service import _direct_command, listen_endpoint, origin_mismatch
from .unit import (
    DRIVER_UNIT_NOTE,
    DRIVER_UNIT_ORIGIN,
    DRIVER_UNIT_STATE,
    DRIVER_UNIT_UNKNOWN,
    NO_ANSWER,
    UNIT_NAME,
)
from .unit_paths import UnitRuntime, make_private_directories, printable, service_command

UNIT_BUSY = "Another dgxm studio up, down or token --rotate is running; wait for it."
BUSY = "Studio is running a management operation; wait for it to finish, or stop the service yourself with systemctl --user stop dgxm-studio.service."
BUSY_UNKNOWN = "Cannot tell whether Studio is running a management operation; wait, or stop the service yourself with systemctl --user stop dgxm-studio.service."
# Filled at print time (design section 6.4 step 7, and the unit lock's own checks).
CHECK_FAILED = "The service command cannot start the companion:"
CHECK_TIMEOUT = "The service command did not finish its check in 60 seconds; nothing was installed."
UNIT_LOCK_UNSAFE = "The Studio unit lock must be a regular file you own, in a directory you own that no other account can write:"
# The store's own refusals (studio/store.py private_directory), for the read-only walk.
STATE_SYMLINK = "State directories cannot be symlinks."
STATE_PRIVATE = "State directory must belong to this user with mode 0700."

ACTIVE_STATES = frozenset({"active", "activating", "reloading", "deactivating"})
LOCK_FILE = "companion.lock"
UNIT_LOCK_FILE = "studio-unit.lock"
_STATUS_PROPERTIES = ("LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup", "FragmentPath",
                      "ExecMainStartTimestampMonotonic")
_PRIVATE_LIMIT = 65536

# Clocks and kernel trees are read through these names, so a test replaces
# them without reading the machine's clocks, control groups or processes.
_monotonic = time.monotonic
_realtime = time.time
_sleep = time.sleep
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC = Path("/proc")

Run = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True)
class UnitStatus:
    """One ``systemctl --user show`` read of ``dgxm-studio.service``."""

    load_state: str
    active_state: str
    sub_state: str
    main_pid: int
    control_group: str
    fragment_path: str
    start_monotonic_us: int


def show_unit(properties: Sequence[str], *, unit: str = UNIT_NAME, run: Run | None = None,
              timeout: float = 10.0) -> dict[str, str] | None:
    """The named properties of a user unit, or ``None`` when the user manager does not answer."""
    run = run or subprocess.run
    try:
        result = run(["systemctl", "--user", "show", "--no-pager", "--property=" + ",".join(properties), "--", unit],
                     stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace",
                     timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    fields: dict[str, str] = {}
    for line in str(result.stdout).splitlines():
        name, separator, value = line.partition("=")
        if separator:
            fields.setdefault(name, value)
    if not set(properties) <= set(fields):
        return None
    return {name: fields[name] for name in properties}


def user_manager(*, run: Run | None = None,
                 which: Callable[[str], str | None] | None = None) -> Literal["ok", "absent", "no_answer"]:
    """Whether a systemd user manager answers here (design section 6.4 step 2)."""
    run, which = run or subprocess.run, which or shutil.which
    if not sys.platform.startswith("linux") or which("systemctl") is None:
        return "absent"
    try:
        result = run(["systemctl", "--user", "show-environment"], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10, check=False)
    except subprocess.TimeoutExpired:
        return "no_answer"
    except OSError:
        return "absent"
    return "ok" if result.returncode == 0 else "absent"


def unit_status(*, run: Run | None = None) -> UnitStatus | None:
    """The Studio unit's state, or ``None`` when the user manager does not answer in 10 seconds."""
    fields = show_unit(_STATUS_PROPERTIES, run=run)
    if fields is None:
        return None
    try:
        return UnitStatus(
            load_state=fields["LoadState"], active_state=fields["ActiveState"], sub_state=fields["SubState"],
            main_pid=int(fields["MainPID"] or 0), control_group=fields["ControlGroup"],
            fragment_path=fields["FragmentPath"],
            start_monotonic_us=int(fields["ExecMainStartTimestampMonotonic"] or 0),
        )
    except ValueError:
        return None


def unit_active(status: UnitStatus) -> bool:
    """Running in the sense of every Studio command: a process may exist, written files or not."""
    return status.active_state in ACTIVE_STATES


def _ready(state: Path, status: UnitStatus, *, management: bool, after_monotonic: float | None,
           previous_token: str | None) -> bool:
    from .sign_in import OWNER_FILE, SIGN_IN_FILE, read_sign_in_page, read_token_file

    if not unit_active(status) or status.main_pid <= 0 or status.start_monotonic_us <= 0:
        return False
    start = status.start_monotonic_us / 1_000_000
    if after_monotonic is not None and start <= after_monotonic:
        return False
    # The main process start in wall time, less one second of slack; a page left
    # by a crashed process is older, because a restart waits RestartSec=3.
    floor_ns = int((start + _realtime() - _monotonic() - 1.0) * 1_000_000_000)
    try:
        token = read_token_file(state / OWNER_FILE)
        page = None if management else read_sign_in_page(state / SIGN_IN_FILE)
    except StudioError:
        return False
    if token is None or token[1] < floor_ns:
        return False
    if previous_token is not None and token[0] == previous_token:
        return False
    return management or (page is not None and page[1] >= floor_ns)


def wait_ready(state: Path, *, management: bool, after_monotonic: float | None, previous_token: str | None,
               timeout: float = 30.0, poll: float = 0.25,
               status_fn: Callable[[], UnitStatus | None] | None = None) -> Literal["ready", "failed", "unknown"]:
    """Wait for the running unit's own token file (and, without management, its page).

    ``failed`` when the unit reads ``failed``, or ``inactive`` after a restart
    the caller stamped with ``after_monotonic``; ``unknown`` at the deadline.
    ``status_fn`` defaults to this module's ``unit_status``, looked up at call
    time like every ``run`` default here, so a test's replacement applies.
    """
    status_fn = status_fn or unit_status
    deadline = _monotonic() + timeout
    while True:
        status = status_fn()
        if status is not None:
            if status.active_state == "failed" or (after_monotonic is not None and status.active_state == "inactive"):
                return "failed"
            if _ready(state, status, management=management, after_monotonic=after_monotonic,
                      previous_token=previous_token):
                return "ready"
        if _monotonic() >= deadline:
            return "unknown"
        _sleep(poll)


def _state_directory(state: Path) -> Path | None:
    """The store's directory checks without its ``mkdir``; ``None`` when the directory does not exist."""
    path = state.expanduser().absolute()
    for entry in [*reversed(path.parents), path]:
        try:
            info = os.lstat(entry)
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as exc:
            raise SystemExit(f"Cannot use the state directory {state}: {exc.strerror or exc}.") from exc
        if stat.S_ISLNK(info.st_mode):
            raise SystemExit(STATE_SYMLINK)
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise SystemExit(STATE_PRIVATE)
    return path


def lock_probe(state: Path) -> Literal["absent", "held", "free"]:
    """Whether a companion holds the state lock, without creating the lock or following a link."""
    directory = _state_directory(state)
    if directory is None:
        return "absent"
    try:
        descriptor = os.open(directory / LOCK_FILE, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return "absent"
    except OSError as exc:
        raise SystemExit(f"Cannot use the state directory {state}: {exc.strerror or exc}.") from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise SystemExit(STATE_PRIVATE)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "held"
        except OSError as exc:
            raise SystemExit(f"Cannot use the state directory {state}: {exc.strerror or exc}.") from exc
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return "free"
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def unit_lock() -> Iterator[None]:
    """Serialize ``up``, ``down`` and ``token --rotate`` for this user; held refuses with ``UNIT_BUSY``."""
    from .launch import default_state_dir

    directory = default_state_dir().parent
    try:
        make_private_directories(directory)
        parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise SystemExit(f"{UNIT_LOCK_UNSAFE} {directory}") from exc
    descriptor = -1
    try:
        info = os.fstat(parent)
        if info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise SystemExit(f"{UNIT_LOCK_UNSAFE} {directory}")
        try:
            descriptor = os.open(UNIT_LOCK_FILE, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600,
                                 dir_fd=parent)
            lock = os.fstat(descriptor)
            if not stat.S_ISREG(lock.st_mode) or lock.st_uid != os.geteuid():
                raise SystemExit(f"{UNIT_LOCK_UNSAFE} {directory / UNIT_LOCK_FILE}")
            os.fchmod(descriptor, 0o600)
        except OSError as exc:
            raise SystemExit(f"{UNIT_LOCK_UNSAFE} {directory / UNIT_LOCK_FILE}") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(UNIT_BUSY) from None
        final = os.stat(UNIT_LOCK_FILE, dir_fd=parent, follow_symlinks=False)
        if (final.st_dev, final.st_ino) != (lock.st_dev, lock.st_ino):
            raise SystemExit(f"{UNIT_LOCK_UNSAFE} {directory / UNIT_LOCK_FILE}")
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    finally:
        os.close(parent)
    try:
        yield
    finally:
        os.close(descriptor)


def _command_summary(words: list[str]) -> str:
    """The first words of a command line; ``python -m dgx_monarch update`` reads ``dgxm update``."""
    if "-m" in words[:-1]:
        index = words.index("-m")
        module = words[index + 1]
        name, rest = ("dgxm" if module == "dgx_monarch" else module), words[index + 2:]
    else:
        name, rest = Path(words[0]).name, words[1:]
    return " ".join(printable(word)[:64] for word in [name, *rest[:2]])


def _describe(pid: int) -> str | None:
    try:
        raw = (_PROC / str(pid) / "cmdline").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None  # it ended after the control group was read
    except OSError:
        return f"pid {pid}"
    words = [word.decode("utf-8", "replace") for word in raw.split(b"\0") if word]
    return f"pid {pid}: {_command_summary(words)}" if words else f"pid {pid}"


def busy_processes(status: UnitStatus) -> list[str] | None:
    """One line per process in the unit's control group other than its main process.

    ``None`` when the control group cannot be read, which callers refuse as
    ``BUSY_UNKNOWN``: unknown is never read as idle.
    """
    group = status.control_group
    parts = group.split("/")
    if not group.startswith("/") or group == "/" or ".." in parts or any(ord(char) < 32 for char in group):
        return None
    try:
        text = (_CGROUP_ROOT / group.lstrip("/") / "cgroup.procs").read_text()
        pids = [int(value) for value in text.split()]
    except (OSError, ValueError):
        return None
    lines = [_describe(pid) for pid in pids if pid != status.main_pid]
    return [line for line in lines if line is not None]


def journal_tail(since_epoch: float, *, run: Run | None = None) -> list[str]:
    """The unit's last 20 journal lines since a wall-clock epoch; empty when the journal does not answer."""
    run = run or subprocess.run
    try:
        result = run(["journalctl", "--user", "-u", UNIT_NAME, "--since", f"@{int(since_epoch)}", "-n", "20",
                      "--no-pager", "-o", "cat"], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                     errors="replace", timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    return [printable(line) for line in str(result.stdout).splitlines()][-20:]


def linger_state(*, run: Run | None = None) -> str | None:
    """``loginctl``'s Linger value for this user, or ``None`` when it does not answer."""
    run = run or subprocess.run
    try:
        result = run(["loginctl", "show-user", str(os.getuid()), "--property=Linger", "--value"],
                     stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="replace", timeout=10,
                     check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return str(result.stdout).strip() or None


def port_in_use(port: int) -> bool:
    """A bind test of Studio's own loopback port, the check the companion makes a moment later."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return True
            raise SystemExit(f"Cannot listen on 127.0.0.1:{port}: {exc.strerror or exc}.") from exc
    return False


def origin_preflight(state: Path, origin: str, from_default: bool) -> tuple[
        Literal["ok", "no_origin", "differs", "unreadable"], str | None]:
    """The state directory's bound driver origin, read only; the bound origin when it differs.

    A missing directory or database counts as no binding. A read that fails
    for any reason is ``unreadable``, never "matches" or "differs".
    """
    directory = _state_directory(state)
    bound: str | None = None
    if directory is not None:
        database = directory / "studio.sqlite3"
        try:
            info = os.lstat(database)
        except FileNotFoundError:
            info = None
        except OSError:
            return "unreadable", None
        if info is not None:
            if not stat.S_ISREG(info.st_mode):
                return "unreadable", None
            try:
                # as_uri() encodes a ? or # in the path, which f"file:{path}" would not.
                connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=1.0)
                try:
                    row = connection.execute("SELECT value FROM meta WHERE key='driver_origin'").fetchone()
                finally:
                    connection.close()
            except (sqlite3.Error, OSError):
                return "unreadable", None
            bound = None if row is None else str(row[0])
    if bound is None:
        return ("no_origin" if from_default else "ok"), None
    return ("ok", None) if bound == origin else ("differs", bound)


def check_preflight(runtime: UnitRuntime, env: dict[str, str], *,
                    run: Run | None = None) -> tuple[Literal["ok", "failed", "timeout"], str]:
    """Run the exact service command with ``--check``: the outcome and the line to print."""
    from .launch import EXTRA_MISSING

    run = run or subprocess.run
    try:
        result = run([*service_command(runtime), "--check"], env=env, cwd="/", stdin=subprocess.DEVNULL,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                     timeout=60, check=False)
    except subprocess.TimeoutExpired:
        return "timeout", CHECK_TIMEOUT
    except OSError as exc:
        return "failed", f"{CHECK_FAILED} {printable(str(exc.strerror or exc))}"
    output = str(result.stdout or "")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    last = printable(lines[-1]) if lines else f"exit status {result.returncode}"
    if result.returncode != 0 or EXTRA_MISSING in output:
        return "failed", f"{CHECK_FAILED} {last}"
    return "ok", last


def driver_unit_check(unit: str, origin: str, *, run: Run | None = None) -> tuple[
        Literal["ok", "note", "unknown_unit", "other_state", "origin", "no_answer"], str, str | None]:
    """Design section 6.4 step 8: the outcome, its line, and the driver unit's ``ActiveState``."""
    fields = show_unit(("LoadState", "ActiveState", "ExecStart"), unit=unit, run=run)
    if fields is None:
        return "no_answer", NO_ANSWER, None
    active = fields["ActiveState"] or None
    load = fields["LoadState"]
    if load == "not-found":
        return "unknown_unit", f"{DRIVER_UNIT_UNKNOWN} {unit}", active
    if load != "loaded":
        return "other_state", DRIVER_UNIT_STATE.format(unit=unit, state=printable(load)), active
    try:
        _executable, _main, argv = _direct_command(fields["ExecStart"])
        endpoint = listen_endpoint(argv)
    except StudioError:
        return "note", DRIVER_UNIT_NOTE, active
    if origin_mismatch(endpoint, origin) is not None:
        return "origin", DRIVER_UNIT_ORIGIN, active
    return "ok", "", active


def _remove_matching(path: Path, pattern: re.Pattern[str]) -> bool:
    """Remove an owner-only regular file whose whole content matches; the read checks its descriptor."""
    from .sign_in import read_private_file

    try:
        before = os.lstat(path)
        found = read_private_file(path, _PRIVATE_LIMIT)
        if found is None or pattern.fullmatch(found[0].decode("utf-8", "replace")) is None:
            return False
        after = os.lstat(path)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            return False
        os.unlink(path)
    except (OSError, ValueError):
        return False
    return True


def remove_leftovers(state: Path) -> list[Path]:
    """After ``down``: remove a dead token file and sign-in page, only while no companion holds the lock."""
    from .sign_in import EARLIER_TOKEN, OWNER_FILE, SIGN_IN_FILE, SIGN_IN_PAGE

    if lock_probe(state) != "free":
        return []
    removed: list[Path] = []
    for name, pattern in ((OWNER_FILE, EARLIER_TOKEN), (SIGN_IN_FILE, SIGN_IN_PAGE)):
        path = state / name
        if _remove_matching(path, pattern):
            removed.append(path)
    return removed


def specifier_values(environ: Mapping[str, str]) -> tuple[str, str, str]:
    """``HOME``, ``USER`` and ``XDG_RUNTIME_DIR`` as the user manager fills ``%h``, ``%u`` and ``%t``."""
    import pwd

    entry = pwd.getpwuid(os.getuid())
    runtime = environ.get("XDG_RUNTIME_DIR", "") or f"/run/user/{os.getuid()}"
    return entry.pw_dir, entry.pw_name, runtime
