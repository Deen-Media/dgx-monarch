"""Worker-thread calls whose caller cannot outlive the thread."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any


async def blocking(function: Callable[..., Any], *args: Any) -> Any:
    """Run ``function`` on a worker thread; a cancelled caller ends only after the thread does.

    asyncio.to_thread cannot stop a running thread. Without this wait a
    cancelled caller would release its locks, or let shutdown close the store,
    while the thread still writes.
    """
    return await settle(asyncio.to_thread(function, *args))


async def settle(awaitable: Any) -> Any:
    task = asyncio.create_task(awaitable)
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
            continue
        except BaseException:
            if cancellation is not None:
                raise cancellation from None
            raise
        if cancellation is not None:
            raise cancellation
        return result
