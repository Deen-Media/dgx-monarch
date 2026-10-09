"""The `config permissions` doctor row: a read-only check of the loaded config file's mode."""
from __future__ import annotations

import stat
from pathlib import Path

from ..config import ClusterConfig


def config_permissions_row(config: ClusterConfig | None) -> dict[str, str]:
    if config is None or not config.source:
        return {
            "status": "ok",
            "name": "config permissions",
            "detail": "no cluster config loaded",
        }
    try:
        metadata = Path(config.source).lstat()
    except OSError:
        return {
            "status": "WARN",
            "name": "config permissions",
            "detail": "could not read the mode of the loaded config file; check it by hand",
        }
    if not stat.S_ISREG(metadata.st_mode):
        return {
            "status": "WARN",
            "name": "config permissions",
            "detail": "the loaded config is a symlink or another non-regular file; automatic repair is unavailable",
        }
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o077:
        return {
            "status": "WARN",
            "name": "config permissions",
            "detail": f"mode {mode:04o} exposes cluster metadata; run dgxm doctor --repair",
        }
    return {
        "status": "ok",
        "name": "config permissions",
        "detail": f"mode {mode:04o} is owner-only",
    }
