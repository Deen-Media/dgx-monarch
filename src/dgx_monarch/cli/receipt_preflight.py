"""Refuse, before any operation, an operator receipt path that publication would refuse."""
from __future__ import annotations

import os
from pathlib import Path

from .operator_receipt import receipt_destination
from .receipt_files import private_parent


def check_receipt_destination(path: str | os.PathLike[str]) -> Path:
    """Refuse a receipt path that ``write_receipt`` would refuse, creating nothing.

    It walks the folders with the writer's own walk (``private_parent`` without
    creating), so each existing part is opened and checked as publication
    opens it: a real directory it can open, a group- or world-writable one
    only if sticky, the receipt's own folder this user's. The folder that will
    hold the receipt must let this user create in it, and the file must not
    exist yet. A folder without hard links is no reason to refuse: the writer
    then creates the final name exclusively. A path taken after this check is
    still never overwritten at publication.
    """
    destination = receipt_destination(path)
    directory_fd = private_parent(destination.parent, create=False)
    if directory_fd < 0:
        return destination
    try:
        try:
            os.stat(destination.name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("operator receipt destination already exists")
        if not os.access(".", os.W_OK | os.X_OK, dir_fd=directory_fd):
            raise OSError("operator receipt directory is not writable")
    finally:
        os.close(directory_fd)
    return destination
