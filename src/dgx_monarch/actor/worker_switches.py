"""A Worker's diagnostic switches, from one file its own user owns.

A managed Worker unit starts its loop through ``/usr/bin/env -i``, which
clears every variable a unit drop-in sets, on purpose: nothing ambient may
reach an actor. The Worker-side diagnostic switches are environment variables
all the same (a worker argument reaches every rank and rides in every
render's capability context), so on such a unit there was no way to set one
short of rewriting the unit's start line by hand.

This module is that way. Each actor reads ``~/.config/dgx-monarch/worker.env``
when it starts (``GPUWorker.__init__``), so a recycle, which starts new
actors, applies a change with no service restart and no unit touched. It
takes ``NAME=value`` lines for the six names in ``NAMES`` and nothing else:
each can only turn one of the Worker's own steps off, or, for the two fault
names and only with ``DGXM_ACCEPTANCE``, make one rank refuse. No path, no
credential and no loader variable can come in this way.

- The file must be a regular file owned by the Worker's user and not writable
  by group or others; a link is not followed. Anything else is ignored and
  said, so the cleared environment is not opened to another user.
- A name already set in the actor's environment wins: where a drop-in or the
  shell does reach the loop, it keeps its say.
- It is never silent. Every start logs one WARNING naming each diagnostic
  switch set in this actor and where it came from, the setup reply and the
  status carry the same names, and the driver repeats them at every setup. A
  forgotten off switch turns a safety rule off.

``apply`` never raises: a file that cannot be read sets nothing.
"""
from __future__ import annotations

import errno
import os
import re
import stat
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from ..log import get_logger
from . import backup_agreement, context_room, load_fault, lora_partial_load, resident_ledger

log = get_logger(__name__)

# Each name comes from the module that reads it, so this list cannot drift from them, and a
# name stays written in one place (tests/test_load_fault.py holds the load fault to that).
NAMES = (
    context_room.DISABLE_ENV,
    resident_ledger.COPY_FREE_DROP_DISABLE_ENV,
    lora_partial_load.BACKUP_FREE_DISABLE_ENV,
    load_fault.RANK_ENV,
    backup_agreement.FAULT_RANK_ENV,
    load_fault.ACCEPTANCE_ENV,
)
FILE = Path(".config") / "dgx-monarch" / "worker.env"  # under the home of the Worker's user
DOCS = "docs/CLUSTER.md, Setting a Worker-side variable"
MAX_BYTES = 16 * 1024
_NAME = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_VALUE = re.compile(r"[A-Za-z0-9_.:-]{1,64}")


@dataclass
class Reading:
    """What one read of the file gave: the switches it sets, and what it had to say."""

    values: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def file_path() -> Path:
    return Path.home() / FILE


def _shown(value: str) -> str:
    """A value fit for a log line: these switches take small plain values, anything else is not echoed."""
    return value if _VALUE.fullmatch(value) else "<set>"


def read(path: Path) -> Reading:
    """Read the file. No file is no switch and no word; a file this Worker must not trust says why."""
    found = Reading()
    try:
        handle = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return found
    except OSError as exc:
        why = ("it is a symbolic link, which is not followed" if exc.errno == errno.ELOOP
               else f"it could not be opened ({type(exc).__name__})")
        found.notes.append(f"{path} was ignored: {why}; no switch was taken from it")
        return found
    try:
        state = os.fstat(handle)
        refused = ("it is not a regular file" if not stat.S_ISREG(state.st_mode)
                   else "it is not owned by the Worker's user" if state.st_uid != os.geteuid()
                   else f"it is writable by group or others (mode {stat.S_IMODE(state.st_mode):04o}; make it 0600)"
                   if state.st_mode & 0o022
                   else f"it is larger than {MAX_BYTES} bytes" if state.st_size > MAX_BYTES else "")
        if refused:
            found.notes.append(f"{path} was ignored: {refused}; no switch was taken from it")
            return found
        text = os.read(handle, MAX_BYTES).decode("utf-8", errors="replace")
    finally:
        os.close(handle)
    unknown: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, equals, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not equals or not _NAME.fullmatch(name):
            unknown.append(f"line {number}")
        elif name not in NAMES:
            unknown.append(name)  # the name only: a value here may be something private
        elif not _VALUE.fullmatch(value):
            found.notes.append(f"{path}: {name} has no plain value (letters, digits and _ . : - only), "
                               "so it was not set")
        else:
            found.values[name] = value
    if unknown:
        found.notes.append(f"{path}: ignored {', '.join(dict.fromkeys(unknown))}; only the Worker's "
                           f"diagnostic switches are read from this file ({', '.join(NAMES)})")
    return found


def apply(environ: MutableMapping[str, str] | None = None, path: Path | None = None) -> tuple[str, ...]:
    """Take the file's switches into this actor's environment; return every diagnostic switch now set.

    Each entry reads ``NAME=value (worker.env)`` or ``NAME=value (environment)``.
    Called once as each actor starts, before anything reads a switch.
    """
    target = os.environ if environ is None else environ
    taken: set[str] = set()
    try:
        found = read(file_path() if path is None else path)
        for note in found.notes:
            log.warning("worker.env: %s", note)
        for name, value in found.values.items():
            if target.get(name):
                if target[name] != value:
                    log.warning("worker.env: %s is already set in this Worker's environment, which wins; "
                                "the file's value was not used", name)
                continue
            target[name] = value
            taken.add(name)
    except Exception as exc:
        log.warning("worker.env could not be read (%s); no switch was taken from it", type(exc).__name__)
    active = tuple(f"{name}={_shown(target[name])} ({'worker.env' if name in taken else 'environment'})"
                   for name in NAMES if target.get(name))
    if active:
        log.warning("Worker switches set on this box: %s. Each turns one of the Worker's own steps off or "
                    "arms a debug fault; remove it when the diagnosis is done (%s)", ", ".join(active), DOCS)
    return active


__all__ = ["DOCS", "FILE", "NAMES", "Reading", "apply", "file_path", "read"]
