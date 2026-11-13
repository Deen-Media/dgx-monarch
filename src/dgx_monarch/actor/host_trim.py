"""Hand freed host memory back to the box where a host copy is given up.

glibc gives a block its own mapping above a threshold and takes smaller ones
from an arena. A mapped block goes back to the box when it is freed. The
threshold starts at 128 KiB and moves up to the size of any mapped block the
process frees, as far as 32 MiB. The Worker loads on its GPU thread, which is
not the process's main thread, so its arena is a chain of 64 MiB heaps, each
its own mapping. A block freed inside such a heap stays resident: glibc gives
a heap back only when it is the chain's last one and empty. Later requests of
that thread reuse the memory; the box never reads it.

Read on hardware (2026-11-12 and 13, one GPU of a DGX Spark under its clock
cap, max 2100 MHz). The same free of ComfyUI's weight backup returned 15.3 GiB
after one load line into a fresh Worker, five times, and 5.4 to 5.8 GiB where
the INT8 ConvRot Wan-Animate 2 file was loaded a second time in the process.
That file holds 5.70 GiB in tensors of 32 MiB and more and 9.78 GiB under it.
A probe then found the kept memory, 9.95 GiB, in 243 anonymous mappings of 50
to 55 MiB at 64 MiB-aligned addresses; a soft and a hard clear returned none
of it; and the next load of the file went into it (a peak of 17.20 GiB, not
25): freed memory kept by the arenas and reused, not a leak.

``trim`` calls ``malloc_trim(0)``, which walks EVERY arena of the process,
the main one and each thread's, takes each arena's lock in turn, and gives
back the whole free pages it finds inside them. So it does not matter which
thread calls it. It is called where a host copy is given up: after the free
of ComfyUI's backup for a sampler (the one-GPU whole-load step, the pair's
act on every rank and the partial-load route's tail,
``lora_partial_load._free_backup``), and after the store drops a slot, which
the Worker's hard clear does for every slot
(``resident_ledger.flush_dropped``). It is NOT called after a load is placed
or after a soft clear. It never raises. On another C library, or where the
symbol is missing, it says so once and does nothing. It does not reach memory
held by torch's own bundled allocator, which the worker loop already sets to
purge at free (``cli/worker_process_env.HOST_ALLOCATOR_PURGE_ENV``).
"""
from __future__ import annotations

import ctypes
import time

from ..log import get_logger
from ..mesh_safety import mem_available_bytes

log = get_logger(__name__)

_unavailable: str | None = None  # proc lifetime: why this process has no malloc_trim, said once


def trim() -> tuple[bool, float, int, int]:
    """Trim glibc's heaps. Returns (ran, seconds, MemAvailable before, MemAvailable after), in bytes."""
    global _unavailable
    before = int(mem_available_bytes() or 0)
    if _unavailable is not None:
        return False, 0.0, before, before
    started = time.perf_counter()
    try:
        libc = ctypes.CDLL(None)
        libc.gnu_get_libc_version.restype = ctypes.c_char_p  # glibc alone has it; another C library raises here
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes, malloc_trim.restype = [ctypes.c_size_t], ctypes.c_int
        malloc_trim(0)
    except Exception as exc:
        _unavailable = f"{type(exc).__name__}: {exc}"
        log.info("host trim: this process's C library offers no malloc_trim (%s), so freed host memory "
                 "under its mapping threshold stays with the process until it is used again", _unavailable)
        return False, 0.0, before, before
    return True, time.perf_counter() - started, before, int(mem_available_bytes() or 0)


__all__ = ["trim"]
