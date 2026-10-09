"""Opt-in client for the root process inspector; it trusts only a fresh report from root for this UID and boot."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import socket
import stat
import struct
import time


# privileged_process_source returns these three functions verbatim for the setup scripts to embed; those scripts run
# under python -I -S, so the functions may call only each other and the modules on the import line it emits.
def _checked_process_report(value, uid, boot_id, source_sha, requested_ns, received_ns):
    fields = {"schema", "status", "uid", "boot_id", "start_monotonic_ns", "end_monotonic_ns",
              "source_sha256", "workers", "actors", "unknown_count"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("invalid inspection report")
    if (type(value["schema"]) is not int or value["schema"] != 1
            or type(value["uid"]) is not int or value["uid"] != uid
            or value["boot_id"] != boot_id or value["source_sha256"] != source_sha):
        raise ValueError("inspection identity mismatch")
    start, end = value["start_monotonic_ns"], value["end_monotonic_ns"]
    if (type(start) is not int or type(end) is not int
            or not requested_ns <= start <= end <= received_ns
            or received_ns - end > 5_000_000_000):
        raise ValueError("stale inspection report")
    unknown = value["unknown_count"]
    if type(unknown) is not int or not 0 <= unknown <= 131072:
        raise ValueError("invalid inspection certainty")
    if value["status"] != ("unknown" if unknown else "ok"):
        raise ValueError("inconsistent inspection certainty")
    for name in ("workers", "actors"):
        rows = value[name]
        if not isinstance(rows, list) or len(rows) > 32768:
            raise ValueError("invalid inspection rows")
        seen = set()
        expected = {"pid", "starttime", "address_hashes"} if name == "workers" else {"pid", "starttime"}
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError("invalid process identity")
            pid, birth = row["pid"], row["starttime"]
            if (type(pid) is not int or not 1 <= pid < 2**31 or pid in seen
                    or type(birth) is not int or not 0 < birth < 2**63):
                raise ValueError("invalid process lifetime")
            seen.add(pid)
            if name == "workers":
                hashes = row["address_hashes"]
                if (not isinstance(hashes, list) or len(hashes) > 16
                        or any(not isinstance(item, str) or len(item) != 64
                               or any(c not in "0123456789abcdef" for c in item) for item in hashes)
                        or len(hashes) != len(set(hashes)) or hashes != sorted(hashes)):
                    raise ValueError("invalid worker endpoint fingerprints")
    return value


def _request_process_report(source_sha):
    uid = os.geteuid()
    if uid <= 0:
        raise ValueError("setup inspection requires a non-root target user")
    for path in (pathlib.Path("/run"), pathlib.Path("/run/dgxm-process-inspection")):
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError("unsafe inspection directory")
    path = pathlib.Path(f"/run/dgxm-process-inspection/{uid}.sock")
    before = path.lstat()
    if (not stat.S_ISSOCK(before.st_mode) or before.st_uid != uid
            or stat.S_IMODE(before.st_mode) != 0o600):
        raise ValueError("unsafe inspection socket")
    boot_id = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    requested = time.monotonic_ns()
    deadline = time.monotonic() + 10
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(10)
        connection.connect(str(path))
        credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        if struct.unpack("3i", credentials)[1] != 0:
            raise ValueError("inspection peer is not root")
        connection.sendall(b"dgxm-process-inspection-v1\n")
        connection.shutdown(socket.SHUT_WR)
        parts = []
        size = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("process inspection expired")
            connection.settimeout(remaining)
            chunk = connection.recv(min(65536, (1 << 20) + 1 - size))
            if not chunk:
                break
            parts.append(chunk)
            size += len(chunk)
            if size > 1 << 20:
                raise ValueError("inspection report too large")
    after = path.lstat()
    if (before.st_dev, before.st_ino, before.st_uid, before.st_mode) != (
            after.st_dev, after.st_ino, after.st_uid, after.st_mode):
        raise ValueError("inspection socket changed")
    value = json.loads(b"".join(parts))
    return _checked_process_report(value, uid, boot_id, source_sha, requested, time.monotonic_ns())


def _privileged_process_facts(source_sha, address="", pid=0):
    try:
        report = _request_process_report(source_sha)
    except (OSError, ValueError, TypeError, KeyError, IndexError, UnicodeError, struct.error):
        return None, None, None
    workers, actors = report["workers"], report["actors"]
    address_hash = hashlib.sha256(address.encode("utf-8")).hexdigest()
    selected = any(row["pid"] == pid and address_hash in row["address_hashes"] for row in workers)
    unknown = report["unknown_count"] != 0
    return (True if selected else None if unknown else False,
            True if actors else None if unknown else False,
            True if workers else None if unknown else False)


def privileged_process_source(enabled: bool) -> str:
    """Return the client source and helper digest, or an empty string when off; never start or elevate the helper."""
    if type(enabled) is not bool:
        raise ValueError("process inspection selection must be boolean")
    if not enabled:
        return ""
    helper = pathlib.Path(__file__).with_name("process_inspector.py")
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    return "\n".join([
        "import hashlib,json,os,pathlib,socket,stat,struct,time",
        f"DGXM_INSPECTOR_SHA={digest!r}",
        inspect.getsource(_checked_process_report),
        inspect.getsource(_request_process_report),
        inspect.getsource(_privileged_process_facts),
    ])
