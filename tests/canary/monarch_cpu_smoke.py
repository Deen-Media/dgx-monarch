#!/usr/bin/env python3
"""Explicit local-process Monarch semantics canary; never attaches to Workers."""
from __future__ import annotations

import asyncio
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from monarch.actor import (
    Actor,
    Future,
    concurrent_endpoint,
    endpoint,
    returns_future,
    shutdown_context,
    this_host,
)


class CpuProbe(Actor):
    def __init__(self):
        self.busy = False

    @endpoint
    async def is_busy(self) -> bool:
        return self.busy

    @endpoint
    async def identity(self) -> str:
        return importlib.metadata.version("torchmonarch")

    @concurrent_endpoint
    async def delayed(self) -> str:
        self.busy = True
        try:
            await asyncio.sleep(2)
            return "completed"
        finally:
            self.busy = False


class SyncProbe(Actor):
    @endpoint
    def has_event_loop(self) -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True


def run() -> dict[str, object]:
    observations: dict[str, object] = {}
    events: list[str] = []

    @returns_future
    async def lazy() -> None:
        events.append("started")
        try:
            await asyncio.sleep(1)
            events.append("completed")
        finally:
            events.append("cancelled")

    try:
        try:
            Future()
        except TypeError:
            observations["direct_future_rejected"] = True
        else:
            raise AssertionError("Future construction unexpectedly succeeded")
        future = lazy()
        assert not events, "returns_future body started before observation"
        try:
            future.get(timeout=0.01)
        except TimeoutError:
            pass
        else:
            raise AssertionError("returns_future did not time out")
        assert events == ["started", "cancelled"], events
        try:
            future.get(timeout=1)
        except TimeoutError:
            observations["returns_future_timeout_terminal"] = True
        else:
            raise AssertionError("cancelled body resumed")
        procs = this_host().spawn_procs(name="dgxm_cpu_canary")
        procs.initialized.get(timeout=15)
        actor = procs.spawn("async_probe", CpuProbe)
        expected = importlib.metadata.version("torchmonarch")
        assert actor.identity.call_one().get(timeout=15) == expected
        observations["child_version"] = expected
        sync = procs.spawn("sync_probe", SyncProbe)
        assert sync.has_event_loop.call_one().get(timeout=15) is False
        observations["sync_actor_has_no_event_loop"] = True
        delayed = actor.delayed.call_one()
        try:
            delayed.get(timeout=0.01)
        except TimeoutError:
            pass
        else:
            raise AssertionError("endpoint observation did not time out")
        assert actor.is_busy.call_one().get(timeout=1) is True
        assert delayed.get(timeout=5) == "completed"
        observations["concurrent_endpoint_timeout_reobservable"] = True
        stop = procs.stop()
        assert actor.identity.call_one().get(timeout=5) == expected
        observations["proc_stop_lazy"] = True
        stop.get(timeout=15)
        observations["proc_stop_completed"] = True
        # A timed-out stop cancels its remaining steps. A zero deadline lands
        # before the native stop starts; any nonzero one races it.
        doomed = this_host().spawn_procs(name="dgxm_cpu_canary_stop")
        doomed.initialized.get(timeout=15)
        survivor = doomed.spawn("stop_probe", CpuProbe)
        assert survivor.identity.call_one().get(timeout=15) == expected
        cancelled = doomed.stop()
        try:
            cancelled.get(timeout=0)
        except TimeoutError:
            pass
        else:
            raise AssertionError("ProcMesh stop finished inside a zero deadline")
        try:
            cancelled.get(timeout=1)
        except TimeoutError:
            observations["proc_stop_timeout_cancelled"] = True
        else:
            raise AssertionError("a timed-out ProcMesh stop resumed")
        time.sleep(1)
        assert survivor.identity.call_one().get(timeout=5) == expected
        observations["proc_stop_timeout_procs_survived"] = True
        doomed.stop().get(timeout=15)
        try:
            survivor.identity.call_one().get(timeout=5)
        except Exception as exc:
            observations["proc_stop_after_timeout_completed"] = type(exc).__name__
        else:
            raise AssertionError("procs answered after a completed stop")
    finally:
        shutdown_context(host_timeout=10).get(timeout=15)
    return observations


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(run(), indent=2) + "\n")
