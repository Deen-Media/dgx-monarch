"""Advisory per-config flock for dgxm commands that change a cluster config.

The lock name hashes the path, so the state directory and the lock errors never name a config path.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from pathlib import Path

from .setup_config_parent import prepare_config_parent


class SetupConfigLockUnavailable(OSError):
    """Another dgxm command holds the lock, or the lock path failed a safety, open or identity check."""


class SetupConfigLock:
    def __init__(self, config_path: Path, *, state_root: Path | None = None) -> None:
        if not isinstance(config_path, Path):
            raise ValueError("setup config lock requires a Path")
        normalized = os.path.abspath(os.fspath(config_path.expanduser()))
        if os.name == "posix" and normalized.startswith("//"):
            normalized = f"/{normalized.lstrip('/')}"
        target = Path(normalized)
        key = hashlib.sha256(str(target).encode("utf-8")).hexdigest()
        root = state_root or Path.home() / ".local/state/dgx-monarch/config-locks"
        self._root = root.expanduser().absolute()
        self._path = self._root / f"{key}.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> SetupConfigLock:
        try:
            self._prepare_root()
            root = self._root.lstat()
            if (
                self._root.is_symlink()
                or not stat.S_ISDIR(root.st_mode)
                or root.st_uid != os.geteuid()
                or root.st_mode & 0o022
            ):
                raise SetupConfigLockUnavailable("setup config lock root is unsafe")
            parent_identity = root.st_dev, root.st_ino
            descriptor = os.open(
                self._path,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except OSError as exc:
            raise SetupConfigLockUnavailable("setup config lock is unavailable") from exc
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.geteuid()
                or stat.S_IMODE(opened.st_mode) & 0o077
            ):
                raise SetupConfigLockUnavailable("setup config lock is unsafe")
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            parent = self._root.lstat()
            current = self._path.lstat()
            if (parent.st_dev, parent.st_ino) != parent_identity or (
                current.st_dev,
                current.st_ino,
            ) != (opened.st_dev, opened.st_ino):
                raise SetupConfigLockUnavailable("setup config lock identity changed")
        except BaseException as exc:
            try:
                os.close(descriptor)
            except BaseException:
                pass
            if isinstance(exc, OSError):
                raise SetupConfigLockUnavailable(
                    "another setup config operation is active, or the lock could not be taken safely"
                ) from exc
            raise
        self._descriptor = descriptor
        return self

    def _prepare_root(self) -> None:
        try:
            prepare_config_parent(self._root)
        except OSError as exc:
            raise SetupConfigLockUnavailable("setup config lock parent is unsafe") from exc

    def __exit__(self, _kind: object, _error: object, _traceback: object) -> None:
        descriptor, self._descriptor = self._descriptor, None
        if descriptor is None:
            return
        cleanup_error: BaseException | None = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except BaseException as error:
            if not isinstance(error, OSError):
                cleanup_error = error
        finally:
            try:
                os.close(descriptor)
            except BaseException as error:
                if cleanup_error is None and not isinstance(error, OSError):
                    cleanup_error = error
        if _error is None and cleanup_error is not None:
            raise cleanup_error
