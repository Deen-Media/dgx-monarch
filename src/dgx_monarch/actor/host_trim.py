"""Hand freed host memory back to the box after a host copy is freed for the sampler.

The C library gives a block its own mapping above a threshold and takes
smaller ones from its heap. A mapped block goes back to the box when it is
freed. A heap block does not: its pages stay with the process until the heap's
top is free or something trims it. glibc moves that threshold up to the size
of any mapped block the process frees, as far as 32 MiB. So once a Worker has
freed blocks of some size under 32 MiB, later blocks of that size and smaller
come from the heap, and freeing them returns nothing the box can read.

On record (2026-11-12, one GPU of a DGX Spark under its clock cap, max
2100 MHz): the same free of ComfyUI's weight backup returned 15.3 GiB after
one load line into a fresh Worker, five times, and 5.4 to 5.8 GiB where the
INT8 ConvRot Wan-Animate 2 file was loaded a second time in the process, three
times. That file holds 5.70 GiB in tensors of 32 MiB and more and 9.78 GiB in
tensors under it, and 9.5 to 9.8 GiB stayed.

``trim`` calls glibc's ``malloc_trim(0)``, which returns the free pages of
its heaps wherever they lie. It is called after the three frees of a host copy
made for a sampler: the one-GPU whole-load step, the pair's act on every rank
and the partial-load route's tail (``lora_partial_load._free_backup``). It
never raises. On another C library, or where the symbol is missing, it says so
once and does nothing. It does not reach memory held by torch's own bundled
allocator, which the worker loop already sets to purge at free
(``cli/worker_process_env.HOST_ALLOCATOR_PURGE_ENV``).
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
