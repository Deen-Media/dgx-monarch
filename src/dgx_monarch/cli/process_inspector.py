#!/usr/bin/env python3
"""Temporary, read-only inspector of one user's Monarch processes, run as root; see docs/INSTALL.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import stat
import struct
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import TypedDict

TOKEN = b"dgxm-process-inspection-v1\n"
SCHEMA = 1
MAX_READ = 1 << 20
MAX_PIDS = 65536
MAX_REPORT_PROCESSES = 8192
MAX_REPLY = 1 << 20
MAX_REQUEST = 256
CONNECTION_TIMEOUT = 10.0
MAX_CONNECTIONS = 4
DEFAULT_LIFETIME = 900
MAX_LIFETIME = 1800
ROOT_DIRECTORY = Path("/run/dgxm-process-inspection")

WORKER_MODULE = "dgx_monarch.cli.worker_loop"
ACTOR_MODULE = "monarch._src.actor.bootstrap_main"
ACTOR_ENV_MARKERS = (b"HYPERACTOR_MESH_BOOTSTRAP_MODE=", b"DGXM_PYTHONPATH=")


class ProcessIdentity(TypedDict):
    pid: int
    starttime: int


class WorkerIdentity(ProcessIdentity):
    address_hashes: list[str]


def _read(path: Path) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(MAX_READ + 1)
    if len(value) > MAX_READ:
        raise OSError("oversized proc entry")
    return value


def _starttime(stat_text: bytes) -> int:
    # comm may contain arbitrary bytes and parentheses; only parse the suffix.
    tail = stat_text.rsplit(b")", 1)
    if len(tail) != 2:
        raise ValueError("malformed stat")
    fields = tail[1].split()
    if len(fields) <= 19:
        raise ValueError("short stat")
    starttime = int(fields[19])
    if starttime <= 0:
        raise ValueError("invalid starttime")
    return starttime


def _argv(cmdline: bytes) -> tuple[str, ...]:
    if not cmdline:
        return ()
    if not cmdline.endswith(b"\0"):
        raise ValueError("malformed cmdline")
    # argv is bytes. Decoding with replacement keeps ASCII marker matches exact and
    # does not make a process with a non-UTF-8 argument unknown.
    return tuple(part.decode("utf-8", "replace") for part in cmdline[:-1].split(b"\0"))


def _has_module(argv: tuple[str, ...], module: str) -> bool:
    return any(argv[index : index + 2] == ("-m", module) for index in range(len(argv) - 1))


def _actor_markers(environment: bytes) -> bool:
    if environment and not environment.endswith(b"\0"):
        raise ValueError("malformed environ")
    entries = tuple(item for item in environment.split(b"\0") if item)
    return all(any(item.startswith(marker) for item in entries) for marker in ACTOR_ENV_MARKERS)


def _status_uids(status: bytes) -> tuple[int, int, int, int]:
    for line in status.splitlines():
        if line.startswith(b"Uid:"):
            fields = line.split()[1:]
            if len(fields) != 4:
                break
            try:
                values = tuple(int(field) for field in fields)
            except ValueError as exc:
                raise ValueError("malformed status") from exc
            if any(value < 0 for value in values):
                raise ValueError("malformed status")
            return values[0], values[1], values[2], values[3]
    raise ValueError("missing status uid")


def _process_identity(entry: Path) -> tuple[tuple[int, int, int, int], int]:
    """Return the exact status UID tuple and proc-directory owner."""
    return _status_uids(_read(entry / "status")), entry.stat().st_uid


def _includes_uid(identity: tuple[tuple[int, int, int, int], int], uid: int) -> bool:
    """True when ``uid`` is the real, effective, saved or filesystem UID, or owns the proc directory."""
    uids, owner = identity
    return uid in uids or owner == uid


def _address_hashes(argv: tuple[str, ...]) -> tuple[list[str], bool]:
    values = {argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "--address"}
    return sorted(hashlib.sha256(value.encode()).hexdigest() for value in values)[:16], len(values) > 16


def _boot_id(proc: Path) -> str:
    value = _read(proc / "sys/kernel/random/boot_id").decode("ascii", "strict").strip()
    if not value or len(value) > 128:
        raise ValueError("malformed boot id")
    return value


def _vanished(entry: Path) -> bool:
    try:
        entry.stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def scan(
    uid: int,
    proc: Path = Path("/proc"),
    source_hash: str = "",
    *,
    deadline: float | None = None,
    clock=time.monotonic,
) -> dict[str, object]:
    """Return a bounded report of the UID's worker and actor processes, with no raw argv or environment value.

    The status is ``unknown`` after an unreadable, malformed or oversized observation, a process whose
    UIDs, proc-directory owner or start time changed mid-scan, or a passed deadline. Only a read that raises
    FileNotFoundError, with the proc directory then confirmed gone, drops the process without counting it.
    """
    started = time.monotonic_ns()
    workers: list[WorkerIdentity] = []
    actors: list[ProcessIdentity] = []
    unknown_count = 0
    try:
        entries = list(islice((entry for entry in proc.iterdir() if entry.name.isdigit()), MAX_PIDS + 1))
    except OSError:
        entries = []
        unknown_count += 1
    if len(entries) > MAX_PIDS:
        entries = entries[:MAX_PIDS]
        unknown_count += 1
    for entry in entries:
        if deadline is not None and clock() >= deadline:
            unknown_count += 1
            break
        try:
            before_identity = _process_identity(entry)
            if not _includes_uid(before_identity, uid):
                continue
            before = _starttime(_read(entry / "stat"))
        except FileNotFoundError:
            if _vanished(entry):
                continue
            unknown_count += 1
            continue
        except (OSError, ValueError):
            unknown_count += 1
            continue
        argv: tuple[str, ...] = ()
        try:
            argv = _argv(_read(entry / "cmdline"))
        except FileNotFoundError:
            if _vanished(entry):
                continue
            unknown_count += 1
        except (OSError, ValueError):
            unknown_count += 1
        is_worker = _has_module(argv, WORKER_MODULE)
        is_actor = _has_module(argv, ACTOR_MODULE)
        try:
            is_actor = is_actor or _actor_markers(_read(entry / "environ"))
        except FileNotFoundError:
            if _vanished(entry):
                continue
            unknown_count += 1
        except (OSError, ValueError):
            unknown_count += 1
        try:
            after_identity = _process_identity(entry)
            after = _starttime(_read(entry / "stat"))
            if after_identity != before_identity or before != after:
                unknown_count += 1
                continue
        except FileNotFoundError:
            if _vanished(entry):
                continue
            unknown_count += 1
            continue
        except (OSError, ValueError):
            unknown_count += 1
            continue
        process: ProcessIdentity = {"pid": int(entry.name), "starttime": before}
        if len(workers) + len(actors) + int(is_worker) + int(is_actor) > MAX_REPORT_PROCESSES:
            unknown_count += 1
            continue
        if is_worker:
            address_hashes, address_overflow = _address_hashes(argv)
            if address_overflow:
                unknown_count += 1
            workers.append({**process, "address_hashes": address_hashes})
        if is_actor:
            actors.append(process)
    try:
        if deadline is not None and clock() >= deadline:
            raise ValueError("scan deadline expired")
        boot_id = _boot_id(proc)
    except (OSError, UnicodeDecodeError, ValueError):
        boot_id = ""
        unknown_count += 1
    return {
        "schema": SCHEMA,
        "status": "unknown" if unknown_count else "ok",
        "uid": uid,
        "boot_id": boot_id,
        "start_monotonic_ns": started,
        "end_monotonic_ns": time.monotonic_ns(),
        "source_sha256": source_hash,
        "workers": sorted(workers, key=lambda process: process["pid"]),
        "actors": sorted(actors, key=lambda process: process["pid"]),
        "unknown_count": unknown_count,
    }


def _root_directory() -> Path:
    if os.path.lexists(ROOT_DIRECTORY) and ROOT_DIRECTORY.is_symlink():
        raise RuntimeError("unsafe inspector directory")
    try:
        ROOT_DIRECTORY.mkdir(mode=0o711, parents=False)
    except FileExistsError:
        pass
    else:
        os.chmod(ROOT_DIRECTORY, 0o711)  # noqa: S103 - required root-owned traversal-only directory
    info = ROOT_DIRECTORY.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o711:
        raise RuntimeError("unsafe inspector directory")
    return ROOT_DIRECTORY


def _source_hash() -> str:
    """Hash this file for setup to match against its own copy; refuse a file or parent a non-root user could write."""
    source = Path(__file__)
    if not source.is_absolute() or source.is_symlink():
        raise RuntimeError("inspector source is not a pinned absolute file")
    for path in (source, *source.parents):
        info = path.lstat()
        if path != source and not stat.S_ISDIR(info.st_mode):
            raise RuntimeError("inspector source parent is not a real directory")
        if info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError("inspector source is writable by the client")
    if not stat.S_ISREG(source.lstat().st_mode):
        raise RuntimeError("inspector source is not a regular file")
    return hashlib.sha256(_read(source)).hexdigest()


def _cleanup_socket(path: Path, expected: os.stat_result) -> None:
    """Remove only the socket inode created by this server."""
    try:
        current = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISSOCK(current.st_mode) and (current.st_dev, current.st_ino) == (expected.st_dev, expected.st_ino):
        path.unlink()


def _peer_uid(connection: socket.socket) -> int:
    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return struct.unpack("3i", credentials)[1]


def _read_request(connection: socket.socket, deadline: float, clock=time.monotonic) -> bytes | None:
    chunks: list[bytes] = []
    size = 0
    while True:
        remaining = deadline - clock()
        if remaining <= 0:
            return None
        connection.settimeout(remaining)
        piece = connection.recv(min(MAX_REQUEST - size + 1, MAX_REQUEST))
        if not piece:
            return b"".join(chunks)
        chunks.append(piece)
        size += len(piece)
        if size > MAX_REQUEST:
            return None


def _serve_socket(
    uid: int,
    path: Path,
    lifetime: float,
    source_hash: str,
    *,
    proc: Path = Path("/proc"),
    clock=time.monotonic,
    peer_uid=_peer_uid,
    ready: Callable[[], None] | None = None,
) -> None:
    """Serve on ``path``, which must not exist yet, until ``lifetime`` seconds after the call."""
    if os.path.lexists(path):
        raise RuntimeError("inspector socket already exists")
    server = socket.socket(socket.AF_UNIX)
    created: os.stat_result | None = None
    deadline = clock() + lifetime
    slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
    active: set[socket.socket] = set()
    active_lock = threading.Lock()
    stopping = threading.Event()
    pool = ThreadPoolExecutor(max_workers=MAX_CONNECTIONS, thread_name_prefix="dgxm-inspector")

    def handle(connection: socket.socket, connection_deadline: float) -> None:
        try:
            with connection:
                if stopping.is_set() or peer_uid(connection) != uid:
                    return
                if _read_request(connection, connection_deadline, clock) != TOKEN:
                    return
                report = scan(uid, proc=proc, source_hash=source_hash, deadline=connection_deadline, clock=clock)
                if stopping.is_set() or clock() >= connection_deadline:
                    return
                reply = json.dumps(report, separators=(",", ":"), sort_keys=True).encode() + b"\n"
                if len(reply) <= MAX_REPLY and clock() < connection_deadline:
                    connection.settimeout(max(0.001, connection_deadline - clock()))
                    connection.sendall(reply)
        except OSError:
            pass
        finally:
            with active_lock:
                active.discard(connection)
            slots.release()

    try:
        server.bind(str(path))
        created = path.lstat()
        if not stat.S_ISSOCK(created.st_mode):
            raise RuntimeError("inspector socket bind did not create a socket")
        os.chown(path, uid, -1)
        os.chmod(path, 0o600)
        configured = path.lstat()
        if (
            configured.st_uid != uid
            or stat.S_IMODE(configured.st_mode) != 0o600
            or (configured.st_dev, configured.st_ino) != (created.st_dev, created.st_ino)
        ):
            raise RuntimeError("inspector socket ownership or mode changed")
        server.listen(8)
        if ready is not None:
            ready()
        while clock() < deadline:
            server.settimeout(max(0.001, deadline - clock()))
            try:
                connection, _ = server.accept()
            except TimeoutError:
                continue
            if not slots.acquire(blocking=False):
                connection.close()
                continue
            with active_lock:
                active.add(connection)
            try:
                pool.submit(handle, connection, min(deadline, clock() + CONNECTION_TIMEOUT))
            except BaseException:
                connection.close()
                with active_lock:
                    active.discard(connection)
                slots.release()
                raise
    finally:
        stopping.set()
        server.close()
        with active_lock:
            pending = tuple(active)
        for connection in pending:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        pool.shutdown(wait=True)
        if created is not None:
            _cleanup_socket(path, created)


def serve(uid: int, lifetime: int) -> None:
    """Serve on the fixed per-UID socket, owned by ``uid`` inside the root-owned directory."""
    source_hash = _source_hash()
    directory = _root_directory()
    path = directory / f"{uid}.sock"
    old_term = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    old_int = signal.signal(signal.SIGINT, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    old_hup = signal.signal(signal.SIGHUP, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    try:
        _serve_socket(
            uid,
            path,
            lifetime,
            source_hash,
            ready=lambda: print(f"dgxm process inspector ready: uid={uid} lifetime={lifetime}s", flush=True),
        )
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        signal.signal(signal.SIGHUP, old_hup)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifetime", type=int, default=DEFAULT_LIFETIME)
    args = parser.parse_args()
    try:
        uid = int(os.environ["SUDO_UID"])
    except (KeyError, ValueError):
        raise SystemExit(2) from None
    if os.geteuid() != 0 or uid <= 0 or not 1 <= args.lifetime <= MAX_LIFETIME:
        raise SystemExit(2)
    serve(uid, args.lifetime)


if __name__ == "__main__":
    main()
