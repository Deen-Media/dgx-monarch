"""Refuse, before any operation, an operator receipt path that publication would refuse."""
from __future__ import annotations

import os
import stat
from pathlib import Path

from .operator_receipt import receipt_destination


def check_receipt_destination(path: str | os.PathLike[str]) -> Path:
    """Refuse a receipt path that ``write_receipt`` would refuse, reading and creating nothing.

    It applies the writer's rules to the parts that exist: each is a real
    directory, a group- or world-writable one only if sticky, the receipt's own
    folder belongs to this user, the deepest existing folder is writable, and
    the file itself does not exist yet. A path taken after this check is still
    never overwritten at publication.
    """
    destination = receipt_destination(path)
    try:
        os.lstat(destination)
    except (FileNotFoundError, NotADirectoryError):
        pass  # a parent that is not a directory is named by the walk below
    else:
        raise FileExistsError("operator receipt destination already exists")
    current = Path(destination.anchor)
    final = destination.parent
    deepest = current
    for name in final.parts[1:]:
        current = current / name
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            break
        if not stat.S_ISDIR(info.st_mode):
            raise OSError("operator receipt parent must be a real directory, not a symlink")
        if current == final and info.st_uid != os.geteuid():
            raise OSError("operator receipt directory belongs to another user")
        if current != final and info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
            raise OSError("operator receipt parent chain is unsafe")
        deepest = current
    if not os.access(deepest, os.W_OK | os.X_OK):
        raise OSError("operator receipt directory is not writable")
    return destination
