"""Timing boundaries that retain asyncio locking and executor context semantics."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal

from .runtime_diagnostics import phase


class MeasuredAsyncLock(asyncio.Lock):
    """The ordinary FIFO lock, with advisory acquisition latency."""

    async def acquire(self) -> Literal[True]:
        with phase("queue_wait"):
            return await super().acquire()


def run_gpu_phase(fn: Callable[..., Any], *args: Any) -> Any:
    # Setup has its own boundary. Other work includes model load/unload,
    # scheduling and sampling; this measures executor occupancy, not just DiT.
    name = "setup" if getattr(fn, "__name__", "") == "_setup_impl" else "compute"
    with phase(name):
        return fn(*args)
