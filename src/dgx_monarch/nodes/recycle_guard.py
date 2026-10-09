"""Single-flight ownership for reset work that can outlive its HTTP waiter."""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable


def begin(
    state: dict[str, object], lock: threading.Lock, min_interval_s: float,
    now: float | None = None,
) -> bool:
    """Admit one bounded global reset write without trusting client identity."""
    observed = time.monotonic() if now is None else now
    with lock:
        last_value = state.get("last_finished", float("-inf"))
        last = float(last_value) if isinstance(last_value, int | float) else float("-inf")
        if state.get("active") or observed - last < min_interval_s:
            return False
        state["active"] = True
        return True


def finish(
    state: dict[str, object], lock: threading.Lock, now: float | None = None,
) -> None:
    observed = time.monotonic() if now is None else now
    with lock:
        state.update(active=False, last_finished=observed)


async def await_settlement(
    fn: Callable[[], object],
    timeout_s: float,
    on_boundary: Callable[[], None],
    on_finish: Callable[[], None],
):
    """Keep lifecycle ownership until executor work, not its waiter, settles."""
    on_boundary()

    def settled(future) -> None:
        try:
            # Retrieve a post-timeout exception without exposing its detail in
            # the event loop's unhandled-Future diagnostic.
            if not future.cancelled():
                future.exception()
        finally:
            on_boundary()
            on_finish()

    try:
        future = asyncio.get_running_loop().run_in_executor(None, fn)
    except Exception:
        on_boundary()
        on_finish()
        raise
    future.add_done_callback(settled)
    # Executor work cannot be cancelled. Shield prevents wait_for from marking
    # its Future done at the HTTP deadline and firing settlement prematurely.
    return await asyncio.wait_for(asyncio.shield(future), timeout=timeout_s)
