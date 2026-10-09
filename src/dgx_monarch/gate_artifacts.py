"""Bounded artifact fingerprints used by the identity-gate ledger."""
from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Iterable

_ARTIFACT_CACHE: dict[tuple, str] = {}
_ARTIFACT_CACHE_LOCK = threading.Lock()
_ARTIFACT_CACHE_LIMIT = 128
_SAMPLE_BYTES = 1024 * 1024


def _artifact_stat_key(path: str, stat: os.stat_result) -> tuple:
    return (
        os.path.realpath(path), stat.st_dev, stat.st_ino, stat.st_size,
        stat.st_mtime_ns, stat.st_ctime_ns,
    )


def _sample_offsets(size: int) -> tuple[int, int, int]:
    return (0, max(0, size // 2 - _SAMPLE_BYTES // 2), max(0, size - _SAMPLE_BYTES))


def _advise_sampled_ranges_dontneed(fd: int, offsets: Iterable[int], size: int) -> None:
    """Issue best-effort DONTNEED advice for the sampled file ranges."""
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return
    for offset in set(offsets):
        length = min(_SAMPLE_BYTES, max(0, size - offset))
        if length:
            try:
                advise(fd, offset, length, dontneed)
            except OSError:
                pass


# The three strings that stand where a fingerprint should be: the two this
# module returns when it cannot read a file, and the one the reader-side
# composites write for a name that resolves to no path.
UNRESOLVED_SIGNATURES = frozenset({"missing", "unreadable", "unstable"})


class UnresolvedArtifactIdentityError(RuntimeError):
    """An artifact could not be given a stable fingerprint.

    Keep ``RuntimeError`` compatibility, but let loader preflight distinguish this
    from estimator failure: an identity error after a capacity refusal must not
    be swallowed as permission to load.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__("cannot establish a stable model artifact identity for "
                         + ", ".join(names))
        self.names = names


def refuse_unresolved(labelled: Iterable[tuple[str, str]]) -> None:
    """Raise before a sentinel can enter a gate or consent lookup key.

    None of the three identifies an artifact, so a digest built over one
    matches no ledger row: a quarantine or consent lookup would read "no
    verdict recorded" for a model it never identified, and a persisted FAIL
    would be skipped. The writers refuse them too
    (actor/store_identity.build_request_artifact_identity); this is the same
    rule on the reader side.
    """
    unresolved = [name for name, signature in labelled
                  if signature in UNRESOLVED_SIGNATURES]
    if unresolved:
        raise UnresolvedArtifactIdentityError(tuple(unresolved))


def artifact_signature(path: str) -> str:
    """Return a stable start/middle/end fingerprint with at most 3 MiB I/O."""
    for _attempt in range(2):
        fd = None
        offsets: tuple[int, ...] = ()
        size = 0
        try:
            fd = os.open(path, os.O_RDONLY)
            before = os.fstat(fd)
            size = before.st_size
            key = _artifact_stat_key(path, before)
            with _ARTIFACT_CACHE_LOCK:
                cached = _ARTIFACT_CACHE.get(key)
            if cached is not None:
                # Opening pins an inode, not the path. Re-check both before a
                # cached return so atomic replacement cannot inherit identity.
                after_fd = os.fstat(fd)
                after_path = os.stat(path)
                if (_artifact_stat_key(path, after_fd) == key
                        and _artifact_stat_key(path, after_path) == key):
                    return cached
                continue

            offsets = _sample_offsets(size)
            digest = hashlib.sha256(str(size).encode())
            for offset in offsets:
                remaining = min(_SAMPLE_BYTES, max(0, size - offset))
                position = offset
                while remaining:
                    chunk = os.pread(fd, remaining, position)
                    if not chunk:
                        raise OSError("short artifact sample read")
                    digest.update(chunk)
                    position += len(chunk)
                    remaining -= len(chunk)

            after_fd = os.fstat(fd)
            after_path = os.stat(path)
        except OSError:
            return "unreadable"
        finally:
            if fd is not None:
                _advise_sampled_ranges_dontneed(fd, offsets, size)
                try:
                    os.close(fd)
                except OSError:
                    pass
        if (_artifact_stat_key(path, after_fd) != key
                or _artifact_stat_key(path, after_path) != key):
            continue

        value = digest.hexdigest()[:24]
        with _ARTIFACT_CACHE_LOCK:
            _ARTIFACT_CACHE[key] = value
            while len(_ARTIFACT_CACHE) > _ARTIFACT_CACHE_LIMIT:
                _ARTIFACT_CACHE.pop(next(iter(_ARTIFACT_CACHE)))
        return value
    return "unstable"
