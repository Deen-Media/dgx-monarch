"""Read-only evidence for a preserved core package's final-component symlink.

This grants no write, replacement or uninstall authority to either pathname.
Callers must establish the preserved protected distribution and venv boundary.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path


class ProtectedLeafError(ValueError):
    """Bounded refusal when protected link/target evidence cannot be established."""


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    # Access times may change because of this read; identity/content timestamps
    # and ownership/mode must not. Device+inode prevent same-content replacement.
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def protected_leaf_fingerprint(path: Path) -> tuple[str, Path]:
    """Bind logical link identity/text and resolved regular target identity/data."""
    try:
        if any(parent.is_symlink() for parent in path.parents):
            raise ProtectedLeafError("protected core inventory has a symlink ancestor")
        link_before = path.lstat()
        if not stat.S_ISLNK(link_before.st_mode):
            raise ProtectedLeafError("protected core leaf is no longer a symlink")
        spelling = os.readlink(path)
        target = path.resolve(strict=True)
        target_before = target.lstat()
        if not stat.S_ISREG(target_before.st_mode):
            raise ProtectedLeafError("protected core symlink target is not a regular file")
        # Nonblocking avoids waiting if a concurrent replacement becomes a FIFO;
        # nofollow plus fstat refuse an unexpected final symlink or special file.
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or _stamp(opened) != _stamp(target_before):
                raise ProtectedLeafError("protected core target changed before its read")
            content = hashlib.sha256()
            while chunk := os.read(fd, 1 << 20):
                content.update(chunk)
            if _stamp(os.fstat(fd)) != _stamp(opened):
                raise ProtectedLeafError("protected core target changed during its read")
        finally:
            os.close(fd)
        if (_stamp(path.lstat()) != _stamp(link_before) or os.readlink(path) != spelling
                or path.resolve(strict=True) != target or _stamp(target.lstat()) != _stamp(opened)):
            raise ProtectedLeafError("protected core link or target changed after its read")
        record = {"logical_path": str(path), "link_spelling": spelling, "link_identity": _stamp(link_before),
                  "resolved_target": str(target), "target_identity": _stamp(opened),
                  "target_content_sha256": content.hexdigest()}
        digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return digest, target
    except (OSError, RuntimeError):
        raise ProtectedLeafError("protected core link observation unknown") from None
