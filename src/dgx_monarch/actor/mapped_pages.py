"""Release checkpoint pages retained after an FSDP shard build.

FSDP loads use stock safetensors mmap to avoid a full anonymous state dict.
The shard builder copies local rows through a pinned bounce buffer, leaving
reclaimable clean page cache. Copying directly to the device can instead pin
mmap pages with write intent, break copy-on-write, and retain anonymous copies
beside the shards. See measurement M6 in docs/VALIDATION.md.

After sharding, clone remaining file-backed parameters and buffers into
anonymous memory, release their references, and then evict page cache with
``posix_fadvise(POSIX_FADV_DONTNEED)``. Eviction cannot remove mapped pages.
The remaining tensors are usually replicated fp32 islands, scalars, auxiliary
modules, and buffers whose dtype already matched.

For mappings still held elsewhere, ``madvise(MADV_DONTNEED)`` releases resident
pages without removing the mapping; later reads fault the same bytes back in.
Apply this only to regular files with no anonymous pages in ``/proc/self/smaps``.
Anonymous pages may have changed and must be logged, never discarded. The
regular-file check also excludes pinned-memory device mappings, which can
report zero anonymous bytes without containing checkpoint data.

Using pread would require translating checkpoint keys to ComfyUI's converted
parameter names. Mapping-based release avoids that dependency. All operations
are advisory: ``release_materialize_window`` catches exceptions and logs a
warning without failing the load.
"""
from __future__ import annotations

import bisect
import ctypes
import os
import re
import stat
from dataclasses import dataclass
from typing import Any

from ..log import get_logger
from ..transfer_utils import failure_summary, safe_call

log = get_logger(__name__)

# asm-generic/mman-common.h, which x86-64 and arm64 both use.
_MADV_DONTNEED = 4
_KIB = 1024
# One /proc/self/maps row with a pathname. Anonymous rows and the bracketed
# kernel rows carry no pathname that starts with a slash.
_MAPS_ROW = re.compile(
    r"^([0-9a-f]+)-([0-9a-f]+) \S{4} [0-9a-f]+ \S+ +\d+ +(/.*?)(?: \(deleted\))?$")
_SMAPS_HEADER = re.compile(r"^[0-9a-f]+-[0-9a-f]+ ")


@dataclass(frozen=True, slots=True)
class FileMapping:
    """One address range this process has a file mapped into."""

    path: str
    start: int
    end: int

    def holds(self, pointer: int) -> bool:
        return self.start <= pointer < self.end


def _row(line: str) -> FileMapping | None:
    match = _MAPS_ROW.match(line.rstrip("\n"))
    if match is None:
        return None
    return FileMapping(match.group(3), int(match.group(1), 16),
                       int(match.group(2), 16))


def _regular(path: str, seen: dict[str, bool]) -> bool:
    """Whether a mapping's pathname is a real file, cached in ``seen``."""
    known = seen.get(path)
    if known is None:
        try:
            known = stat.S_ISREG(os.stat(path).st_mode)
        except OSError:
            known = False
        seen[path] = known
    return known


def file_mappings() -> list[FileMapping]:
    """Every regular-file range in this process, sorted; empty off Linux."""
    seen: dict[str, bool] = {}
    try:
        with open("/proc/self/maps") as handle:
            rows = [row for row in (_row(line) for line in handle)
                    if row is not None and _regular(row.path, seen)]
    except OSError:
        return []
    rows.sort(key=lambda row: row.start)
    return rows


class _Index:
    """The mappings, searched by address rather than walked per tensor."""

    def __init__(self, mappings: list[FileMapping]):
        self.mappings = mappings
        self.starts = [mapping.start for mapping in mappings]

    def path_at(self, pointer: int) -> str | None:
        position = bisect.bisect_right(self.starts, pointer) - 1
        if position < 0:
            return None
        mapping = self.mappings[position]
        return mapping.path if mapping.holds(pointer) else None


def resident(paths: frozenset[str]) -> list[tuple[FileMapping, int, int]]:
    """``(mapping, resident bytes, anonymous bytes)`` for these paths.

    ``/proc/self/smaps`` repeats each maps row and then names its counters, so
    one walk says both what a drop returns and whether it may happen. A path
    that is not a regular file is dropped here, the one gate keeping
    ``drop_mapped_pages`` off a range no checkpoint lived in.
    """
    rows: list[tuple[FileMapping, int, int]] = []
    seen: dict[str, bool] = {}
    paths = frozenset(path for path in paths if _regular(path, seen))
    current: FileMapping | None = None
    rss = anonymous = 0
    try:
        with open("/proc/self/smaps") as handle:
            for line in handle:
                if _SMAPS_HEADER.match(line):
                    if current is not None:
                        rows.append((current, rss, anonymous))
                    header = _row(line)
                    current = (header if header is not None and header.path in paths
                               else None)
                    rss = anonymous = 0
                    continue
                if current is None:
                    continue
                if line.startswith("Rss:"):
                    rss = int(line.split()[1]) * _KIB
                elif line.startswith("Anonymous:"):
                    anonymous = int(line.split()[1]) * _KIB
    except (OSError, ValueError, IndexError):
        return rows
    if current is not None:
        rows.append((current, rss, anonymous))
    return rows


def resident_bytes(paths: frozenset[str]) -> int:
    """How much of these files this process still holds mapped and resident."""
    return sum(rss for _mapping, rss, _anonymous in resident(paths))


def _host_pointer(tensor: Any) -> int | None:
    """The host address a tensor's storage starts at, or ``None``.

    A sharded parameter answers for its local tensor. A device tensor, a meta
    tensor, and any storage that cannot be read answer ``None``.
    """
    local = getattr(tensor, "_local_tensor", tensor)
    try:
        if local.is_meta or local.device.type != "cpu":
            return None
        return int(local.untyped_storage().data_ptr())
    except Exception:
        return None


def backing_paths(tensors: Any) -> frozenset[str]:
    """Which files back these tensors, read off this process's own mappings.

    This is how a caller learns the real path. The FSDP load opens its
    checkpoint through a ``/proc/self/fd`` alias so the proven inode is the one
    comfy reads (``actor/fsdp_checkpoint_pin.py``), and the kernel names the
    resolved file in ``/proc/self/maps``, never the alias.
    """
    index = _Index(file_mappings())
    if not index.mappings:
        return frozenset()
    found: set[str] = set()
    for tensor in tensors:
        pointer = _host_pointer(tensor)
        if pointer is None:
            continue
        path = index.path_at(pointer)
        if path is not None:
            found.add(path)
    return frozenset(found)


def evict_page_cache(paths: frozenset[str]) -> None:
    """Return these files' clean page cache. Useless while they are mapped."""
    from .unbake import drop_file_cache

    for path in paths:
        try:
            drop_file_cache(path)
        except Exception as exc:  # the kernel reclaims these pages anyway
            safe_call(log.debug, "page cache eviction skipped for %s (%s)",
                      path, failure_summary(exc))


def drop_mapped_pages(paths: frozenset[str]) -> int:
    """Return these files' mapped pages to the pool, keeping the mapping.

    Bytes dropped. A mapping carrying anonymous pages is skipped and named:
    those are rows the driver copied straight out of the mmap (copy-on-write
    broken under its pin), and a drop would discard pages this process cannot
    prove unmodified rather than cost a re-read.
    """
    rows = [row for row in resident(paths) if row[1]]
    if not rows:
        return 0
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except OSError as exc:
        safe_call(log.debug, "madvise unavailable: %s", failure_summary(exc))
        return 0
    dropped = 0
    for mapping, rss, anonymous in rows:
        if anonymous:
            safe_call(log.warning,
                      "mapped page drop skipped for %s at %#x: %.3f GiB of it is "
                      "anonymous (copy-on-write rows) and a drop would discard it",
                      os.path.basename(mapping.path), mapping.start,
                      anonymous / (1 << 30))
            continue
        if libc.madvise(ctypes.c_void_p(mapping.start),
                        ctypes.c_size_t(mapping.end - mapping.start),
                        _MADV_DONTNEED) == 0:
            dropped += rss
    return dropped


def detach_from_files(module: Any) -> tuple[int, int, frozenset[str]]:
    """Copy every file-backed tensor of ``module`` into anonymous memory.

    ``(tensor count, bytes copied, the files they were backed by)``. This is
    what lets the mapping go: one such tensor keeps the whole file mapped.
    """
    index = _Index(file_mappings())
    if not index.mappings:
        return 0, 0, frozenset()

    def backing(tensor: Any) -> str | None:
        pointer = _host_pointer(tensor)
        return None if pointer is None else index.path_at(pointer)

    count = 0
    copied = 0
    paths: set[str] = set()
    for _name, parameter in list(module.named_parameters()):
        path = backing(parameter)
        if path is None:
            continue
        parameter.data = parameter.data.clone()
        paths.add(path)
        count += 1
        copied += parameter.numel() * parameter.element_size()
    # Walk every name a buffer has: the default walk yields a shared buffer
    # once and would leave its other owner holding the file.
    clones: dict[int, Any] = {}
    for name, buffer in list(module.named_buffers(remove_duplicate=False)):
        path = backing(buffer)
        if path is None:
            continue
        if id(buffer) not in clones:
            clones[id(buffer)] = buffer.clone()
            count += 1
            copied += buffer.numel() * buffer.element_size()
        owner_name, _, attr = name.rpartition(".")
        owner = module.get_submodule(owner_name) if owner_name else module
        owner._buffers[attr] = clones[id(buffer)]
        paths.add(path)
    return count, copied, frozenset(paths)
