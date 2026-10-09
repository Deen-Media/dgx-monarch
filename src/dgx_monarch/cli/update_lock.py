"""A per-checkout flock, so one driver-side update, verified or legacy, runs at a time."""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from pathlib import Path


class UpdateLockUnavailable(RuntimeError):
    """Another dgxm update holds this checkout's lock, the lock path is unsafe,
    or open, fstat, fchmod or flock on the lock file raised an OSError."""


class UpdateLock:
    def __init__(self, repo: Path, *, state_root: Path | None = None) -> None:
        identity = str(repo.resolve(strict=True)).encode("utf-8")
        key = hashlib.sha256(identity).hexdigest()
        root = state_root or Path.home() / ".local" / "state" / "dgx-monarch" / "update-locks"
        self._root = root.expanduser().absolute()
        self._path = self._root / f"{key}.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> UpdateLock:
        self._root.mkdir(parents=True, mode=0o700, exist_ok=True)
        metadata = self._root.lstat()
        if self._root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise UpdateLockUnavailable("update lock root is unsafe")
        if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
            raise UpdateLockUnavailable("update lock root has the wrong owner")
        self._root.chmod(0o700)
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(self._path, flags, 0o600)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise UpdateLockUnavailable("update lock is not a regular file")
            if hasattr(os, "geteuid") and opened.st_uid != os.geteuid():
                raise UpdateLockUnavailable("update lock has the wrong owner")
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError, UpdateLockUnavailable) as exc:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise UpdateLockUnavailable(
                "another dgxm update is already active, or the update lock could not be taken safely"
            ) from exc
        self._descriptor = descriptor
        return self

    def __exit__(self, _kind: object, _error: object, _traceback: object) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(descriptor)
            except OSError:
                pass
