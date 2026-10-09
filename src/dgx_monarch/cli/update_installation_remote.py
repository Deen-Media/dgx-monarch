"""Isolated host-side journal operations for verified Worker installation changes."""
from __future__ import annotations

import base64
import grp
import hashlib
import json
import os
import pwd
import re
import shutil
import stat
import subprocess
from pathlib import Path


def checked(path: Path, *, directory: bool = False) -> os.stat_result:
    info = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    forbidden = 0o022
    if directory and info.st_gid == os.getegid():
        group = grp.getgrgid(info.st_gid)
        user = pwd.getpwuid(os.geteuid()).pw_name
        if group.gr_name == user and set(group.gr_mem) <= {user} and all(entry.pw_uid == os.geteuid() for entry in pwd.getpwall() if entry.pw_gid == info.st_gid):
            forbidden = 0o002
    if not kind(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & forbidden:
        raise ValueError("installation ownership changed")
    return info


def read(path: Path) -> bytes:
    before = checked(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    after = os.fstat(fd)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        os.close(fd)
        raise ValueError("installation file changed while opening")
    with os.fdopen(fd, "rb") as source:
        return source.read()


def sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable(path: Path, content: bytes, mode: int = 0o600) -> None:
    checked(path.parent, directory=True)
    temporary = path.with_name(path.name + ".next")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        os.fchmod(output.fileno(), mode)
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    sync(path.parent)


def mkdir(path: Path) -> None:
    if not path.exists():
        mkdir(path.parent)
        path.mkdir(mode=0o700)
        sync(path.parent)
    checked(path, directory=True)


def unit_state(unit: Path, *, allow_stale: bool = False) -> dict[str, str]:
    properties = ("FragmentPath", "DropInPaths", "MainPID", "ActiveState", "SubState", "Job", "NeedDaemonReload", "ControlGroup")
    command = ["systemctl", "--user", "show", unit.name, "--no-pager"]
    result = subprocess.run([*command, *("--property=" + key for key in properties)], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise ValueError("unit state unavailable")
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if fields.get("FragmentPath") != str(unit) or fields.get("DropInPaths") != "" or fields.get("NeedDaemonReload") not in (("no", "yes") if allow_stale else ("no",)):
        raise ValueError("unit overrides or stale manager state")
    paths = (unit.parent, Path.home() / ".local/share/systemd/user", Path(os.environ.get("XDG_RUNTIME_DIR", "/nonexistent")) / "systemd/user")
    for directory in paths:
        for name in (unit.name + ".d", "dgxm-.service.d", "service.d"):
            dropins = directory / name
            if dropins.exists() and any(dropins.iterdir()):
                raise ValueError("unit drop-ins exist")
    return fields


def link_state(live: Path) -> dict[str, str]:
    if live.is_symlink():
        if live.lstat().st_uid != os.geteuid():
            raise ValueError("source link ownership changed")
        return {"kind": "symlink", "target": os.readlink(live)}
    if live.exists():
        checked(live, directory=True)
        return {"kind": "directory", "target": ""}
    return {"kind": "absent", "target": ""}


def snapshot(unit: Path, live: Path, python_bin: str, address: str) -> dict[str, object]:
    for parent in (unit.parent, unit.parent.parent, unit.parent.parent.parent):
        checked(parent, directory=True)
    state = unit_state(unit)
    pid = state.get("MainPID", "")
    if not pid.isdigit() or int(pid) <= 0 or state.get("ActiveState") != "active" or state.get("SubState") != "running" or state.get("Job") not in ("", "0"):
        raise ValueError("Worker must be observed running before update")
    process = Path("/proc") / pid
    if process.stat().st_uid != os.geteuid():
        raise ValueError("Worker owner differs")
    interpreter = Path(python_bin).expanduser()
    argv = [value.decode() for value in (process / "cmdline").read_bytes().split(b"\0") if value]
    suffix = ["-m", "dgx_monarch.cli.worker_loop", "--address", address]
    if not argv or argv[-4:] != suffix or argv[1:-4] not in ([], ["-S", "-P", "-s", "-B"]) or Path(argv[0]).resolve() != interpreter.resolve() or (process / "exe").resolve() != interpreter.resolve():
        raise ValueError("Worker command differs from configured installation")
    birth = (process / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    paths = [value.removeprefix(b"PYTHONPATH=").decode() for value in (process / "environ").read_bytes().split(b"\0") if value.startswith(b"PYTHONPATH=")]
    if len(paths) != 1 or not paths[0].startswith("/"):
        raise ValueError("Worker source path is ambiguous")
    parts = paths[0].split(":")
    if len(parts) not in (1, 2):
        raise ValueError("Worker source path is ambiguous")
    source = Path(parts[0])
    site_packages = ""
    if len(parts) == 2:
        interpreter = Path(python_bin).expanduser()
        version = subprocess.run([str(interpreter), "-I", "-S", "-c", "import sys;print(str(sys.version_info.major)+chr(46)+str(sys.version_info.minor))"], capture_output=True, text=True, timeout=10)
        if version.returncode or not re.fullmatch(r"[0-9]+\.[0-9]+\n", version.stdout):
            raise ValueError("configured Python version unavailable")
        site_packages = str(interpreter.parent.parent / "lib" / ("python" + version.stdout.strip()) / "site-packages")
        if parts[1] != site_packages:
            raise ValueError("Worker dependency path differs from configured Python")
    if not (source / "dgx_monarch").is_dir():
        raise ValueError("Worker source missing")
    group = state.get("ControlGroup", "")
    if not group.startswith("/user.slice/") or ".." in Path(group).parts:
        raise ValueError("Worker cgroup unavailable")
    members = (Path("/sys/fs/cgroup") / group.lstrip("/") / "cgroup.procs").read_text().split()
    if members != [pid]:
        raise ValueError("Worker cgroup contains other processes")
    for task in (process / "task").iterdir():
        if (task / "children").read_text().strip():
            raise ValueError("Worker has child actors")
    data = read(unit)
    result: dict[str, object] = {"unit": base64.b64encode(data).decode(), "mode": stat.S_IMODE(unit.lstat().st_mode), "sha256": hashlib.sha256(data).hexdigest(), "site": str(source), "site_packages": site_packages, "user": pwd.getpwuid(os.geteuid()).pw_name, "home": str(Path.home()), "live": link_state(live), "pid": pid, "birth": birth}
    if unit_state(unit) != state or (process / "stat").read_text().rsplit(") ", 1)[1].split()[19] != birth:
        raise ValueError("Worker changed during capture")
    return result


def stopped(unit: Path, *, allow_stale: bool = False) -> None:
    fields = unit_state(unit, allow_stale=True) if allow_stale else unit_state(unit)
    if fields.get("MainPID") != "0" or fields.get("ActiveState") != "inactive" or fields.get("SubState") != "dead" or fields.get("Job") not in ("", "0"):
        raise ValueError("Worker stop was not confirmed")


def reload() -> None:
    result = subprocess.run(["systemctl", "--user", "daemon-reload"], capture_output=True, timeout=15)
    if result.returncode:
        raise ValueError("unit reload failed")


def switch_link(live: Path, target: str) -> None:
    temporary = live.with_name(".src-update-next")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("pending source link exists")
    temporary.symlink_to(target)
    os.replace(temporary, live)
    sync(live.parent)


def known(unit: Path, live: Path, backup: Path, prior: dict, new: bytes, target: str) -> bool:
    if read(unit) not in (base64.b64decode(prior["unit"], validate=True), new):
        return False
    current = link_state(live)
    return current in (prior["live"], {"kind": "symlink", "target": target}) or (current["kind"] == "absent" and prior["live"]["kind"] == "directory" and backup.is_dir())


def main(request: dict) -> None:
    base = Path.home() / ".local/share/dgx-monarch"
    unit = Path.home() / ".config/systemd/user/dgxm-worker.service"
    live = base / "src"
    operation = request["operation"]
    if operation == "snapshot":
        print(json.dumps(snapshot(unit, live, request["python_bin"], request["address"]), sort_keys=True))
        return
    if operation == "stop":
        if snapshot(unit, live, request["python_bin"], request["address"]) != request["prior"]:
            raise ValueError("Worker changed since preflight")
        marker = Path.home() / ".local/state/dgx-monarch/worker-generation"
        if marker.exists() or marker.is_symlink():
            marker.unlink()
            sync(marker.parent)
        result = subprocess.run(["systemctl", "--user", "stop", unit.name], capture_output=True, timeout=40)
        if result.returncode:
            raise ValueError("Worker stop failed")
        stopped(unit)
        print("STOPPED")
        return
    token = request["token"]
    if not re.fullmatch(r"u-[0-9a-f]{12}-[0-9a-f]{16}", token):
        raise ValueError("invalid token")
    if operation == "cleanup":
        root = base / "releases" / token
        journal = base / "transactions" / token
        if live.is_symlink() and live.resolve() == root / "site":
            raise ValueError("release is still active")
        if journal.exists():
            if read(journal / "compensated") != b"COMPENSATED\n":
                raise ValueError("unsettled installation journal")
            print("CLEANED")
            return
        if root.exists() or root.is_symlink():
            if read(root / ".reservation").decode("ascii") != request["reservation"]:
                raise ValueError("release reservation belongs to another attempt")
            checked(root.parent, directory=True)
            checked(root, directory=True)
            shutil.rmtree(root)
            sync(root.parent)
        print("CLEANED")
        return
    if operation == "reserve":
        root = base / "releases" / token
        mkdir(root.parent)
        root.mkdir(mode=0o700)
        sync(root.parent)
        reservation = request["reservation"]
        if not re.fullmatch(r"[0-9a-f]{32}", reservation):
            raise ValueError("invalid reservation")
        durable(root / ".reservation", reservation.encode("ascii"))
        (root / "site").mkdir(mode=0o700)
        sync(root)
        print("RESERVED")
        return
    prior = request["prior"]
    old = base64.b64decode(prior["unit"], validate=True)
    new = base64.b64decode(request["unit"], validate=True)
    target = str(base / "releases" / token / "site")
    journal = base / "transactions" / token
    backup = base / "backups" / token / "src"
    if operation == "finalize":
        recorded = json.loads(read(journal / "installation.json"))
        if recorded != {**request, "operation": "activate"} or read(unit) != new or link_state(live) != {"kind": "symlink", "target": target}:
            raise ValueError("final release differs from journal")
        unit_state(unit)
        durable(journal / "finalized", b"FINALIZED\n")
        print("FINALIZED")
        return
    stopped(unit, allow_stale=operation == "compensate")
    if operation == "activate":
        if read(unit) != old or link_state(live) != prior["live"]:
            print("ACTIVATION_PROBE_UNKNOWN")
            return
        checked(Path(target), directory=True)
        mkdir(journal)
        mkdir(backup.parent)
        if (journal / "installation.json").exists() or backup.exists():
            raise ValueError("transaction already exists")
        durable(journal / "installation.json", json.dumps(request, sort_keys=True).encode())
        try:
            if prior["live"]["kind"] == "directory":
                os.rename(live, backup)
                sync(base)
                sync(backup.parent)
            switch_link(live, target)
            durable(unit, new)
            reload()
            stopped(unit)
            if read(unit) != new or link_state(live) != {"kind": "symlink", "target": target}:
                raise ValueError("activation readback differs")
            print("ACTIVATED")
        except (OSError, ValueError, subprocess.SubprocessError):
            print("ACTIVATION_PROBE_PRIOR" if known(unit, live, backup, prior, new, target) else "ACTIVATION_PROBE_UNKNOWN")
        return
    recorded = json.loads(read(journal / "installation.json"))
    if recorded != {**request, "operation": "activate"}:
        raise ValueError("journal differs from captured installation")
    if not known(unit, live, backup, prior, new, target):
        raise ValueError("installation changed outside transaction")
    if operation == "compensate":
        state = link_state(live)
        if state != prior["live"]:
            if prior["live"]["kind"] == "symlink":
                switch_link(live, prior["live"]["target"])
            else:
                if live.is_symlink():
                    live.unlink()
                if prior["live"]["kind"] == "directory":
                    os.rename(backup, live)
                sync(base)
        durable(unit, old, int(prior.get("mode", 0o600)))
        reload()
        stopped(unit)
        if read(unit) != old or link_state(live) != prior["live"]:
            raise ValueError("restoration readback differs")
        durable(journal / "compensated", b"COMPENSATED\n")
        print("COMPENSATED")
        return
    raise ValueError("invalid operation")
