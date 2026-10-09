"""Per-wrap allocator trim and memory trace for the FSDP shard build.

Kept beside adapters/fsdp.py so the build loop stays readable; both helpers
are inert without a CUDA allocator.
"""
from __future__ import annotations

from ..log import get_logger

log = get_logger(__name__)


def _log_wrap_memory(index: int) -> None:
    """Per-wrap memory trace for the shard build (first three, then every tenth).

    Names the term that grows: torch-tracked device bytes (allocated and
    reserved) against the process's anonymous RSS from /proc/self/status.
    """
    if index > 3 and index % 10 != 0:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 2**30
        reserved = torch.cuda.memory_reserved() / 2**30
        rss = {}
        with open("/proc/self/status") as status:
            for line in status:
                key = line.split(":")[0]
                if key in ("VmRSS", "RssAnon", "RssFile", "RssShmem"):
                    rss[key] = int(line.split()[1]) / 2**20
        log.info("FSDP wrap %d: torch allocated %.1f GiB, reserved %.1f GiB; process "
                 "VmRSS %.1f, RssAnon %.1f, RssFile %.1f, RssShmem %.1f GiB",
                 index, allocated, reserved, rss.get("VmRSS", 0.0), rss.get("RssAnon", 0.0),
                 rss.get("RssFile", 0.0), rss.get("RssShmem", 0.0))
    except Exception as exc:  # a trace line must never fail a load
        log.debug("FSDP wrap memory trace skipped: %s", exc)


def _allocator_pool_trim():
    """The per-wrap pool trim, or None where no CUDA allocator is in play."""
    import torch

    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None
    return cuda.empty_cache
