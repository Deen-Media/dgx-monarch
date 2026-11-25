"""Host-side retirement of release slots after a finalized verified update.

Shipped after ``update_installation_remote``'s text and run in that isolated
helper's own globals (``update_installation.retire_script``), under the
lifecycle lock, which every update, setup and lifecycle command takes. It does
not take the Worker generation lock: a Worker that restarts meanwhile must
still pass its start check, and it starts from the kept slot. Stdlib only,
Python 3.9.
"""
from __future__ import annotations

import base64
import importlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Shipped, this text runs in the installation helper's globals; in the package it imports that module.
installation: Any = (sys.modules["__main__"] if __name__ == "__main__"
                     else importlib.import_module(".update_installation_remote", __package__))

RELEASE_NAME = re.compile(r"u-[0-9a-f]{12}-[0-9a-f]{16}|s-[0-9a-f]{16}-[1-9][0-9]*")
RETIRING = ".retiring-"
RETIRE_BUDGET_S = 120.0  # no new removal starts after this; the call allows twice as long
RETIRE_PROC = Path("/proc")  # tests point it at the processes they started


def running_source(unit: Path) -> str | None:
    """The source path the running Worker imports from, read as ``snapshot`` reads it; None when stopped."""
    pid = installation.unit_state(unit).get("MainPID", "")
    if not pid.isdigit() or int(pid) <= 0:
        return None
    process = RETIRE_PROC / pid
    if process.stat().st_uid != os.geteuid():
        raise ValueError("Worker owner differs")
    paths = [value[len(b"PYTHONPATH="):].decode() for value in (process / "environ").read_bytes().split(b"\0")
             if value.startswith(b"PYTHONPATH=")]
    if len(paths) != 1 or not paths[0].startswith("/"):
        raise ValueError("Worker source path is ambiguous")
    return paths[0].split(":")[0]


def settled_slot(base: Path, name: str) -> bool:
    """Whether a slot's own transaction is over: an update's journal finalized or compensated, a setup's closed.

    Setup writes ``closed.json`` only when its cleanup finds nothing installed,
    so a slot from a guided setup that succeeded has no closing mark and stays.
    """
    if name.startswith("u-"):
        journal = base / "transactions" / name
        marks = (("finalized", b"FINALIZED\n"), ("compensated", b"COMPENSATED\n"))
        return any(os.path.lexists(journal / mark) and installation.read(journal / mark) == content for mark, content in marks)
    closed = Path.home() / ".local/state/dgx-monarch/setup" / name / "closed.json"
    if not os.path.lexists(closed):
        return False
    record = json.loads(installation.read(closed))
    token, ordinal = name.rsplit("-", 1)
    return isinstance(record, dict) and (record.get("schema"), record.get("token"), record.get("ordinal")) == (1, token, int(ordinal))


def slots_in_use(releases: Path) -> set[str]:
    """Slot names a process of this user still uses: its folder, a mapped or open file, or PYTHONPATH.

    A program's own file is always mapped, so the maps cover it. A process
    this user cannot read (it exited, or it is not dumpable) adds nothing.
    """
    roots = {str(releases), str(releases.resolve())}
    names: set[str] = set()

    def note(path: str) -> None:
        for root in roots:
            if path.startswith(root + "/"):
                name = path[len(root) + 1:].split("/", 1)[0]
                names.add(name[len(RETIRING):] if name.startswith(RETIRING) else name)

    for process in RETIRE_PROC.iterdir():
        try:
            if not process.name.isdigit() or process.stat().st_uid != os.geteuid():
                continue
            links = [process / "cwd", *(process / "fd").iterdir()]
            maps = (process / "maps").read_text(errors="replace").splitlines()
            environ = (process / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        for link in links:
            try:
                note(os.readlink(link))
            except OSError:
                pass
        for line in maps:
            fields = line.split(None, 5)
            if len(fields) == 6:
                note(fields[5])
        for value in environ:
            if value.startswith(b"PYTHONPATH="):
                for part in value[len(b"PYTHONPATH="):].decode(errors="replace").split(":"):
                    if part.startswith("/"):
                        note(part)
                        note(str(Path(part).resolve()))
    return names


def remove_tree(path: Path) -> int:
    """Remove a renamed slot whole, read-only folders inside included; return the bytes it held."""
    size = 0
    if path.lstat().st_mode & 0o700 != 0o700:
        path.chmod(0o700)
    for current, folders, files in os.walk(path):
        for name in folders:
            folder = Path(current) / name
            if not folder.is_symlink() and folder.lstat().st_mode & 0o700 != 0o700:
                folder.chmod(0o700)
        size += sum((Path(current) / name).lstat().st_size for name in files)
    shutil.rmtree(path)
    return size


def retire(base: Path, token: str, prior: dict) -> None:
    """Remove the release slots this host can prove dead, printing each one, then a closing count.

    It keeps the slot of the finalized update and the slot the update replaced,
    compared by real path, so a rollback still has it. It removes only a slot
    named like one, owned, not a symlink, whose own transaction is over and
    which no process of this user still uses; every other folder stays. Each
    slot is first renamed out of the slot pattern, then removed, so a stop
    halfway leaves nothing that looks like a slot; the next run removes such
    leftovers first. No new removal starts after ``RETIRE_BUDGET_S``, nor
    after a line fails to reach the caller.
    """
    releases = base / "releases"
    root = releases.resolve()
    keep = {root / token}
    # The replaced slot: any slot the old link or the old Worker's source path lies in.
    for named in (prior["live"]["target"] if prior["live"]["kind"] == "symlink" else "", prior.get("site", "")):
        parts = Path(os.path.join(base, named)).resolve().parts if named else ()
        if len(parts) > len(root.parts) and parts[:len(root.parts)] == root.parts:
            keep.add(root / parts[len(root.parts)])
    installation.checked(releases, directory=True)
    deadline, removed, failed, held = time.monotonic() + RETIRE_BUDGET_S, 0, 0, 0
    busy = slots_in_use(releases)
    for entry in sorted(releases.iterdir(), key=lambda entry: (not entry.name.startswith(RETIRING), entry.name)):
        name = entry.name[len(RETIRING):] if entry.name.startswith(RETIRING) else entry.name
        leftover = name != entry.name
        if not RELEASE_NAME.fullmatch(name) or (not leftover and entry.resolve() in keep):
            continue
        try:
            installation.checked(entry, directory=True)
            if not leftover and not settled_slot(base, name):
                continue
        except (OSError, ValueError):
            continue
        if name in busy or time.monotonic() > deadline:
            held += 1  # still in use, or out of time; the next update tries again
            continue
        try:
            renamed = releases / (RETIRING + name)
            if not leftover:
                os.rename(entry, renamed)
            size = remove_tree(renamed)
        except OSError:
            failed += 1
            continue
        removed += 1
        if not say(f"REMOVED {name} {size}"):
            break  # the caller is gone: start no further removal
    if removed:
        installation.sync(releases)
    say(f"RETIRED {removed} {failed} {held}")


def say(line: str) -> bool:
    """Print one line at once; False when the caller no longer reads it."""
    try:
        print(line, flush=True)
    except OSError:
        return False
    return True


def retire_installation(request: dict) -> None:
    """The retire operation: finalize's own check again, under the same locks, then ``retire``."""
    base = Path.home() / ".local/share/dgx-monarch"
    unit = Path.home() / ".config/systemd/user/dgxm-worker.service"
    live = base / "src"
    token = request["token"]
    if request.get("operation") != "retire" or not re.fullmatch(r"u-[0-9a-f]{12}-[0-9a-f]{16}", token):
        raise ValueError("invalid retire request")
    new = base64.b64decode(request["unit"], validate=True)
    target = str(base / "releases" / token / "site")
    journal = base / "transactions" / token
    recorded = json.loads(installation.read(journal / "installation.json"))
    if (installation.read(journal / "finalized") != b"FINALIZED\n" or recorded != {**request, "operation": "activate"}
            or installation.read(unit) != new or installation.link_state(live) != {"kind": "symlink", "target": target}):
        raise ValueError("the release to keep is not the finalized one")
    source = running_source(unit)
    if source is not None and Path(source).resolve() != Path(target).resolve():
        raise ValueError("the running Worker imports from another source")
    retire(base, token, request["prior"])
