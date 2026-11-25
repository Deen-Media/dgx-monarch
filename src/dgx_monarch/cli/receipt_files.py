"""The receipt folder walk and the no-clobber file steps that operator receipts publish through.

One walk serves the writer, which creates missing folders, and the preflight,
which creates nothing, so the two cannot disagree about a path.
"""
from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

_DIRECTORY = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
# What link(2) answers on a filesystem without hard links, such as vfat or exFAT.
NO_HARD_LINKS = frozenset({errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP})


def private_parent(path: Path, *, create: bool = True) -> int:
    """Open the receipt's folder through every part, checking each as the writer needs it.

    With ``create`` a missing part is made at 0700; a folder that already
    exists keeps its mode, since a custom receipt path may sit in a home or a
    group-shared folder. A filesystem without permission bits (vfat, exFAT)
    gives every folder the mount's mode, so a created folder may show more
    than 0700, but never more than the folder it was made in. Without
    ``create`` nothing is created: a missing part ends the walk, the deepest
    existing folder must let this user create in it, and -1 is returned.
    """
    directory_fd = os.open(path.anchor, _DIRECTORY)
    pending_fd = -1
    created = False
    parent_mode = 0
    try:
        for index, name in enumerate(path.parts[1:]):
            created = False
            try:
                child_fd = os.open(name, _DIRECTORY, dir_fd=directory_fd)
            except FileNotFoundError:
                if not create:
                    if not os.access(".", os.W_OK | os.X_OK, dir_fd=directory_fd):
                        raise OSError("operator receipt directory is not writable") from None
                    os.close(directory_fd)
                    return -1
                parent_mode = os.fstat(directory_fd).st_mode
                os.mkdir(name, 0o700, dir_fd=directory_fd)
                created = True
                os.fsync(directory_fd)
                child_fd = os.open(name, _DIRECTORY, dir_fd=directory_fd)
            except OSError as exc:
                if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise OSError("operator receipt parent must be a real directory, not a symlink") from exc
                raise OSError(f"operator receipt parent cannot be opened: {exc.strerror}") from exc
            pending_fd = child_fd
            opened = os.fstat(child_fd)
            final = index == len(path.parts[1:]) - 1
            writable = bool(opened.st_mode & 0o022)
            if not stat.S_ISDIR(opened.st_mode):
                raise OSError("operator receipt parent must be a real directory, not a symlink")
            if final and opened.st_uid != os.geteuid():
                raise OSError("operator receipt directory belongs to another user")
            if (
                (created and (opened.st_uid != os.geteuid() or opened.st_mode & 0o077 & ~parent_mode))
                or (not final and writable and not opened.st_mode & stat.S_ISVTX)
            ):
                raise OSError("operator receipt parent chain is unsafe")
            # Hand the parent over before closing it: a failed close must not
            # leave its number for the handler below to close a second time.
            previous, directory_fd, pending_fd = directory_fd, child_fd, -1
            os.close(previous)
        if created:
            os.fchmod(directory_fd, 0o700)
    except BaseException:
        for descriptor in (pending_fd, directory_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except BaseException:
                    pass
        raise
    return directory_fd


def entry_matches(directory_fd: int, name: str, identity: tuple[int, int]) -> bool:
    try:
        current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return (current.st_dev, current.st_ino) == identity


def unlink_owned(directory_fd: int, name: str, identity: tuple[int, int]) -> bool:
    if not entry_matches(directory_fd, name, identity):
        return False
    os.unlink(name, dir_fd=directory_fd)
    return True


def same_parent_fd(path: Path, identity: tuple[int, int]) -> int:
    descriptor = os.open(path, _DIRECTORY)
    opened = os.fstat(descriptor)
    if (opened.st_dev, opened.st_ino) == identity:
        return descriptor
    os.close(descriptor)
    raise OSError("operator receipt directory identity changed")


def publish_exclusive(directory_fd: int, name: str, payload: bytes, created: list[tuple[int, int]]) -> None:
    """Publish where link(2) is refused: create the final name exclusively and write it whole.

    It still never replaces an existing file. ``created`` receives the new
    file's identity as soon as it exists, so a failure removes only that file.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    file_fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        opened = os.fstat(file_fd)
        created.append((opened.st_dev, opened.st_ino))
        os.fchmod(file_fd, 0o600)
        view = memoryview(payload)
        while view:
            view = view[os.write(file_fd, view):]
        os.fsync(file_fd)
    finally:
        os.close(file_fd)
