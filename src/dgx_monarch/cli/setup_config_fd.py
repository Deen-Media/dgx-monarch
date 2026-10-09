"""File-descriptor helpers for setup config I/O; a cleanup error never replaces the primary exception passed in."""

from __future__ import annotations

import os
from pathlib import Path


def write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("setup config write made no progress")
        remaining = remaining[written:]


def read_bounded(descriptor: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def close_preserving(descriptor: int, primary: BaseException | None = None) -> None:
    try:
        os.close(descriptor)
    except BaseException:
        if primary is None:
            raise


def unlink_preserving(path: Path, primary: BaseException | None = None) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except BaseException:
        if primary is None:
            raise
