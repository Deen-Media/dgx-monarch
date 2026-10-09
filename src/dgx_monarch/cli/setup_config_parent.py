"""Create and check, without following symlinks, the directories above a cluster config or its lock file."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path


def prepare_config_parent(path: Path, *, fsync: Callable[[int], None] = os.fsync) -> None:
    """Create missing directories at 0700 and fsync each new entry; raise OSError on a symlink or unsafe component."""
    if not isinstance(path, Path) or not path.is_absolute():
        raise ValueError("setup config parent must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path.anchor, flags)
    pending = -1
    try:
        parts = path.parts[1:]
        for index, name in enumerate(parts):
            created = False
            try:
                child = os.open(name, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(name, 0o700, dir_fd=descriptor)
                created = True
                fsync(descriptor)
                child = os.open(name, flags, dir_fd=descriptor)
            pending = child
            info = os.fstat(child)
            final = index == len(parts) - 1
            safe_owner = info.st_uid == os.geteuid()
            writable = bool(info.st_mode & 0o022)
            sticky = bool(info.st_mode & stat.S_ISVTX)
            safe_nonowner = not writable or sticky
            if (
                not stat.S_ISDIR(info.st_mode)
                or (final and (not safe_owner or writable))
                or (writable and not sticky)
                or (not safe_owner and not safe_nonowner)
                or (created and (not safe_owner or info.st_mode & 0o077))
            ):
                raise OSError("setup config parent chain is unsafe")
            os.close(descriptor)
            descriptor = pending
            pending = -1
    except BaseException:
        for owned in (descriptor, pending):
            if owned >= 0:
                try:
                    os.close(owned)
                except BaseException:
                    pass
        raise
    else:
        try:
            os.close(descriptor)
        except BaseException as primary:
            try:
                os.close(descriptor)
            except BaseException:
                pass
            raise primary


def validate_real_parent(path: Path) -> None:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise OSError("cluster config parent must be a real directory")
    if metadata.st_uid != os.geteuid():
        raise PermissionError("cluster config parent must be owned by the current user")
