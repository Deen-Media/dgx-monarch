"""Check whether the kernel excludes slab-backed Shmem from MemAvailable.

Linux keeps Shmem folios on the anonymous LRU, while ``si_mem_available()``
credits the file LRU. Subtracting Shmem again double-charges a resident slab.
In the recorded sample this produced negative availability on 97 of 110
slab-resident rows, reaching -51.91 GiB (docs/VALIDATION.md, 2026-09-03).

``shmem_credited`` checks that assumption on each ``telemetry.host_stats``
reading, including masked meminfo and possible kernel changes. Fixture tests
cover 544 recorded rows but cannot detect a live kernel change; the telemetry
field provides that evidence. This standard-library/logging leaf measures
accounting only. Capacity decisions remain in ``capacity_fit`` and
``mesh_safety``.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .log import get_logger

log = get_logger(__name__)

PROC_MEMINFO = "/proc/meminfo"

# The predicate's inputs. The four LRU counters are absent from a masked or
# minimal meminfo, and the sweep's per-cell sampler records only two fields,
# so "cannot tell" must be a real answer here.
CREDIT_FIELDS: tuple[str, ...] = (
    "Active(file)", "Inactive(file)", "Active(anon)", "Inactive(anon)",
    "Cached", "Buffers", "Shmem",
)


def read_meminfo(path: str = PROC_MEMINFO) -> dict[str, int] | None:
    """Every sized /proc/meminfo row in bytes, or None when it cannot be read.

    None means no reading, never zero: an unreadable meminfo makes no claim in
    either direction, as with every capacity probe. Rows with no `kB` unit are
    counts (`HugePages_Total`), not sizes, and are dropped.
    """
    try:
        with open(path) as handle:
            text = handle.read()
    except OSError as exc:
        log.debug("meminfo unreadable at %s: %s", path, exc)
        return None
    out: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if len(parts) != 2 or parts[1] != "kB":
            continue
        try:
            out[key] = int(parts[0]) * 1024
        except ValueError:
            continue
    return out


@dataclass(frozen=True, slots=True)
class ShmemCredit:
    """Whether this kernel counts shmem as reclaimable file cache.

    `credited` has three values: False is this kernel and the safe case, True
    says the module's assumption no longer holds here, and None says the
    reading lacked the fields to decide. The two LRU sums keep the units of the
    mapping they came from (bytes from `read_meminfo`, GiB from a recorded
    row): the predicate is a ratio, and a caller that logs the sums can see why
    it answered.
    """

    credited: bool | None
    file_lru: float | None
    anon_lru: float | None
    complete: bool


def shmem_credited(meminfo: Mapping[str, float] | None) -> ShmemCredit:
    """Is Shmem on the file LRU, where the kernel's cache credit would find it?

    The file LRU counters and `Cached + Buffers - Shmem` describe the same
    pages on a kernel that keeps shmem on the anon LRU, so the difference sits
    near zero (0.02 to 0.03 percent of the sum, docs/VALIDATION.md, 2026-09-03).
    A kernel that moved shmem across would raise that difference by the whole
    of Shmem, so half of Shmem is a threshold no rounding or accounting drift
    reaches.

    Zero Shmem answers False. With nothing resident the comparison reduces to
    `difference >= 0`, which rounding alone would flip, and a recorded verdict
    that changes at idle is worse than the safe answer.
    """
    if meminfo is None or any(field not in meminfo for field in CREDIT_FIELDS):
        return ShmemCredit(credited=None, file_lru=None, anon_lru=None, complete=False)
    file_lru = meminfo["Active(file)"] + meminfo["Inactive(file)"]
    anon_lru = meminfo["Active(anon)"] + meminfo["Inactive(anon)"]
    shmem = meminfo["Shmem"]
    difference = file_lru - (meminfo["Cached"] + meminfo["Buffers"] - shmem)
    return ShmemCredit(
        credited=bool(shmem > 0 and difference >= shmem / 2),
        file_lru=file_lru,
        anon_lru=anon_lru,
        complete=True,
    )
