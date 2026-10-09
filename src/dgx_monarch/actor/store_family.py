"""Persist checkpoint families learned from successful local loads.

The primary memo maps stat identities to families. A separate safetensors
header-digest index is consulted only on a complete stat-key miss, allowing
metadata-only changes to preserve a previously learned family. It cannot
restore a deleted or unreadable memo or adopt a peer's detection result.
``_index_family`` defines the identity checks for index hits.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable
from typing import Any

# Absolute on purpose: tests/test_consent_store.py loads this file by path,
# without the actor package, to pin consent_store.file_identity to it.
from dgx_monarch.log import get_logger
from dgx_monarch.safetensors_header import MAX_HEADER_BYTES
from dgx_monarch.transfer_utils import failure_summary, safe_call

log = get_logger(__name__)

# The side index lives in its own file so the memo keeps one key shape for the
# prune, the eviction and every reader.
_DIGEST_INDEX_SUFFIX = ".digests.json"


def _digest_index_path(memo_path: str) -> str:
    base = memo_path[:-len(".json")] if memo_path.endswith(".json") else memo_path
    return base + _DIGEST_INDEX_SUFFIX


def header_digest(path: str) -> str | None:
    """sha256 over a safetensors file's length prefix and its header.

    Content derived and cheap: no tensor byte is read. Anything that is not a
    safetensors container fails the length bound and earns no row.
    """
    try:
        with open(path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8:
                return None
            length = int.from_bytes(prefix, "little")
            if not 0 < length <= MAX_HEADER_BYTES:
                return None
            header = f.read(length)
            if len(header) != length:
                return None
    except OSError:
        return None
    return hashlib.sha256(prefix + header).hexdigest()


def _file_size(path: str) -> int:
    return os.stat(path).st_size


# What a file that will not parse costs, in that file's own terms: the memo's
# rows were earned by loads and nothing rebuilds them, while the index refills
# from the next load of each checkpoint. One sentence for both misreports one.
_MEMO_LOST = ("family memo at %s does not parse (%d bytes, %s); starting a fresh "
              "one and every row it held is lost")
_INDEX_LOST = ("the family memo's header-digest side index at %s does not parse "
               "(%d bytes, %s); starting a fresh one. No memo row is affected "
               "and the index refills as checkpoints load")


def _read_memo(memo_path: str, logger: Any, lost: str = _MEMO_LOST) -> dict:
    """Read the on-disk memo, returning an empty mapping if it cannot be read.

    Warn when nonempty contents cannot be parsed. ``lost`` describes which data
    was lost: family rows require another load to recover, while the side index
    can rebuild from later loads.
    """
    try:
        with open(memo_path) as f:
            raw = f.read()
    except OSError:
        return {}
    if not raw.strip():
        return {}
    try:
        memo = json.loads(raw)
    except ValueError as exc:
        safe_call(logger.warning, lost, os.path.basename(memo_path), len(raw),
                  failure_summary(exc))
        return {}
    if not isinstance(memo, dict):
        safe_call(logger.warning, lost, os.path.basename(memo_path), len(raw),
                  f"a JSON {type(memo).__name__} where an object was expected")
        return {}
    return memo


def file_identity(path: str) -> str:
    """Identity that changes across replacement and same-size in-place rewrites."""
    st = os.stat(path)
    return (
        f"{st.st_dev}:{st.st_ino}:{st.st_size}:"
        f"{st.st_mtime_ns}:{st.st_ctime_ns}"
    )


def memoized_family(
    path: str,
    *,
    identity: Callable[[str], str],
    memo_path: str,
    lock: Any,
) -> str | None:
    """Return the persisted family for the current file identity.

    The stat key answers first and alone where it hits. The side index is
    consulted only on a full miss, the case a metadata sweep makes: bytes this
    box has already loaded, under a new ctime.

    A memo this call could not read is not a miss: no one knows whether the key
    was there, so there is no index lookup and no family, and an index that
    outlived a deleted memo cannot bring back the rows that deletion retired.
    """
    try:
        key = identity(path)
        with lock, open(memo_path) as f:
            memo = json.load(f)
    except (OSError, ValueError) as exc:
        safe_call(log.debug, "family memo or checkpoint unreadable for %s (%s); "
                  "no family and no digest lookup", os.path.basename(path),
                  failure_summary(exc))
        return None
    if not isinstance(memo, dict):
        return None
    family = memo.get(key)
    if family is not None:
        return family
    return _family_from_digest(path, memo_path, lock)


def _family_from_digest(path: str, memo_path: str, lock: Any) -> str | None:
    """The side index, read only after the stat key missed outright."""
    family = _index_family(path, memo_path, lock)
    if family is None:
        return None
    safe_call(log.info, "family memo: the identity key missed for %s and the "
              "header digest resolved it as %s", os.path.basename(path), family)
    return family


def _index_family(path: str, memo_path: str, guard: Any) -> str | None:
    """The index's answer for these bytes, with no log line and no write.

    Two reads of this box's own file, not one: the digest that looks the row up
    and a second that confirms it, so a checkpoint replaced under a running
    price cannot resolve to the family the old bytes carried. A hit claims an
    identity this box earned by loading it, never adopted from a peer. Two files
    whose headers collide agree on every tensor name, shape, dtype and byte
    range, which is what the model class is read off, so they are one family.
    ``guard`` is the memo's lock, or an empty context where the caller holds it
    already and a second acquire would hang the load.
    """
    digest = header_digest(path)
    if digest is None:
        return None
    try:
        with guard, open(_digest_index_path(memo_path)) as f:
            index = json.load(f)
    except (OSError, ValueError):
        return None
    family = index.get(digest) if isinstance(index, dict) else None
    if family is None or header_digest(path) != digest:
        return None
    return str(family)


def _stat_fields(key: Any) -> list[int] | None:
    """The five integers of ``dev:ino:size:mtime_ns:ctime_ns``, or None."""
    parts = str(key).split(":")
    if len(parts) != 5:
        return None
    try:
        return [int(part) for part in parts]
    except ValueError:
        return None


def _ghost_rows(memo: dict, key: str, family: str) -> list:
    """The rows a metadata sweep left behind for the file ``key`` names.

    A sweep moves ctime alone, so a row agreeing on device, inode, size and
    mtime under the family this load detected is this same file. One naming
    another family disagrees rather than ghosts, and a key in the memo is a
    repeat load with no ghosts of its own.
    """
    fields = _stat_fields(key)
    if fields is None or key in memo:
        return []
    group = fields[:4]
    return [row for row, value in memo.items()
            if (_stat_fields(row) or [])[:4] == group and str(value) == family]


def _drop_ghosts(memo: dict, ghosts: list, path: str, family: str, *,
                 memo_path: str, logger: Any) -> int:
    """Retire the sweep's rows for this file, once the digest confirms them.

    An index silent about these bytes keeps every ghost, and the row this load
    adds lets the next one retire them. An index naming another family is a
    header that changed under a sweep, so those rows keep their place.
    """
    resolved = _index_family(path, memo_path, contextlib.nullcontext())
    if resolved is None:
        return 0
    if resolved != family:
        safe_call(logger.warning,
                  "family memo: the identity key missed for %s and the header "
                  "digest names %s where this load detected %s; the older rows "
                  "for these bytes keep their place",
                  os.path.basename(path), resolved, family)
        return 0
    for row in ghosts:
        memo.pop(row, None)
    safe_call(logger.info,
              "family memo: the identity key missed for %s and the header "
              "digest resolved it as %s; dropped %d stale rows a metadata "
              "sweep left behind and gave this load's key their place",
              os.path.basename(path), family, len(ghosts))
    return len(ghosts)


def _prune_below_floor(memo: dict, min_size: int) -> int:
    """Drop rows whose recorded file size is below the checkpoint floor.

    Tiny test fixtures can otherwise displace real checkpoint rows. Read size
    from the identity key because the original file may no longer exist.
    """
    if min_size <= 0:
        return 0
    doomed = []
    for key in memo:
        fields = _stat_fields(key)
        if fields is not None and fields[2] < min_size:
            doomed.append(key)
    for key in doomed:
        memo.pop(key, None)
    return len(doomed)


def _eviction_victim(memo: dict, protected: Any) -> Any:
    """Choose the oldest row from the family with the most rows.

    Evicting globally by age would let one busy family displace every other
    family's slab eligibility. Vouched families receive no special protection
    because fixture rows can also name a vouched family.

    Count ``protected`` (the newly written row) when choosing the largest
    family, but evict it only if no other row remains. Excluding it from counts
    would misidentify the family that exceeded the cap.
    """
    counts: dict[str, int] = {}
    for value in memo.values():
        counts[str(value)] = counts.get(str(value), 0) + 1
    for family in sorted(counts, key=lambda name: -counts[name]):
        victim = next((key for key, value in memo.items()
                       if key != protected and str(value) == family), None)
        if victim is not None:
            return victim
    return protected


def _write_json(path: str, data: dict) -> None:
    """Replace ``path`` atomically, leaving no partial file behind on failure."""
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _memoize_digest(path: str, family: str, *, memo_path: str, limit: int,
                    logger: Any) -> None:
    """Record the header digest beside the memo row, from the same loaded truth.

    It runs where ``memoize_family`` runs and nowhere else, under the same two
    locks, so the index holds no identity a load on this box did not earn.
    """
    digest = header_digest(path)
    if digest is None:
        return
    index_path = _digest_index_path(memo_path)
    index = _read_memo(index_path, logger, _INDEX_LOST)
    if index.get(digest) == family:
        return
    # Re-insert so the fresh row is the newest of its family, as the memo does.
    index.pop(digest, None)
    index[digest] = family
    while len(index) > limit:
        index.pop(_eviction_victim(index, digest))
    _write_json(index_path, index)


def memoize_family(
    path: str,
    family: str,
    *,
    identity: Callable[[str], str],
    memo_path: str,
    lock: Any,
    limit: int,
    logger: Any,
    min_size: int = 0,
    size: Callable[[str], int] | None = None,
) -> None:
    """Persist a detected family atomically, with best-effort process locking.

    ``min_size`` defaults off for synthetic-key tests. When enabled, its stat
    check stays inside the exception handler: a file replaced during loading
    must produce a warning rather than fail an otherwise successful load.

    On a key miss, remove stale metadata-only identities only after the digest
    index confirms them. Do this during writes because explicit slab loads do
    not otherwise read the memo.
    """
    try:
        key = identity(path)
        if min_size > 0 and (size or _file_size)(path) < min_size:
            safe_call(logger.info,
                      "family memo skipped %s: below the %d byte checkpoint floor",
                      os.path.basename(path), min_size)
            return
        os.makedirs(os.path.dirname(memo_path), exist_ok=True)
        with lock:
            lock_fd = -1
            try:
                try:
                    import fcntl

                    lock_fd = os.open(
                        f"{memo_path}.lock", os.O_CREAT | os.O_RDWR, 0o644)
                    fcntl.flock(lock_fd, fcntl.LOCK_EX)
                except (ImportError, OSError):
                    pass  # cross-process serialization is best-effort
                memo = _read_memo(memo_path, logger)
                # Before the index write below: this load's own row would
                # otherwise answer its own question.
                ghosts = _ghost_rows(memo, key, family)
                dropped = _drop_ghosts(
                    memo, ghosts, path, family,
                    memo_path=memo_path, logger=logger) if ghosts else 0
                # Above the early return too: a memo row older than the side
                # index still owes this file its digest row. Its own try,
                # because the index is best effort and must never cost the memo
                # the row this load earned.
                try:
                    _memoize_digest(path, family, memo_path=memo_path,
                                    limit=limit, logger=logger)
                except OSError as exc:
                    safe_call(logger.warning,
                              "family memo side index not persisted (%s)",
                              failure_summary(exc))
                # Unconditional, and above both guards: below the cap the
                # eviction loop never runs, and the common case is a repeat load
                # of an already memoized checkpoint, which returns early.
                below_floor = _prune_below_floor(memo, min_size)
                if below_floor:
                    safe_call(logger.info,
                              "family memo dropped %d rows below the %.0f MiB checkpoint floor",
                              below_floor, min_size / (1 << 20))
                dropped += below_floor
                if memo.get(key) == family and not dropped:
                    return
                # Re-insert so the fresh write is the newest row of its family
                # and cannot be the one the cap evicts.
                memo.pop(key, None)
                memo[key] = family
                while len(memo) > limit:
                    memo.pop(_eviction_victim(memo, key))
                _write_json(memo_path, memo)
            finally:
                if lock_fd >= 0:
                    os.close(lock_fd)
    except OSError as exc:
        safe_call(
            logger.warning,
            "family memo not persisted (%s)",
            failure_summary(exc),
        )
