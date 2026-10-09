"""Exact, durable removal of a config already bound by an operator lock."""

from __future__ import annotations

import os

from .setup_config_io import read_snapshot
from .setup_config_types import ConfigSnapshot


def remove_exact_config(expected: ConfigSnapshot) -> bool:
    """Remove only the exact reviewed inode and durably publish its absence."""
    if not isinstance(expected, ConfigSnapshot) or not expected.existed:
        return False
    try:
        current = read_snapshot(expected.path)
        if current != expected:
            return False
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        directory = os.open(expected.path.parent, flags)
        try:
            before = os.fstat(directory)
            os.unlink(expected.path.name, dir_fd=directory)
            after = os.fstat(directory)
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                return False
            if os.path.lexists(expected.path):
                return False
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError:
        return False
    return True
