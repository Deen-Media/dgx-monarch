"""Dispatch-contract probe: out-of-band calls must interleave with sample.

Run it in a fresh interpreter, because monarch allows one transport per
process; test_dispatch_contract.py runs it as a subprocess. The actor copies
GPUWorker's dispatch shape (actor/worker.py, actor/cancellation.py): async
endpoints, one asyncio.Lock serializing GPU work, the blocking body on a
dedicated single-thread executor, and a cancel endpoint outside the lock that
must reach a render already in flight. sample is a @concurrent_endpoint, like
every worker endpoint that takes _gpu_lock (torchmonarch >= 0.6.0 queue
dispatch). Cancellation, live telemetry and the pipelined artifact preflight
all depend on the two latencies measured here.

Prints exactly one JSON object as the last stdout line.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import threading
import time

from monarch.actor import Actor, concurrent_endpoint, endpoint, this_host

SLOW_S = 3.0


def _one_value(value_mesh: object) -> object:
    values = list(value_mesh)  # type: ignore[call-overload]
    first = values[0]
    return first[-1] if isinstance(first, tuple) else first


class DispatchContractWorker(Actor):
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._cancel = threading.Event()

    def _blocking_render(self) -> str:
        deadline = time.monotonic() + SLOW_S
        while time.monotonic() < deadline:
            if self._cancel.is_set():
                return "cancelled"
            time.sleep(0.05)
        return "completed"

    @concurrent_endpoint
    async def sample(self) -> str:
        async with self._lock:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._exec, self._blocking_render)

    @endpoint
    async def cancel(self) -> str:
        self._cancel.set()
        return "cancel-recorded"

    @endpoint
    async def status(self) -> str:
        return "ok"

    @concurrent_endpoint
    async def boom(self) -> str:
        raise RuntimeError("deliberate probe failure")


def main() -> None:
    procs = this_host().spawn_procs(per_host={"procs": 1})
    worker = procs.spawn("dispatch_contract", DispatchContractWorker)
    worker.status.call().get(timeout=60)  # warm-up: spawn + first dispatch

    slow = worker.sample.call()
    time.sleep(0.5)  # let sample reach the executor

    t0 = time.monotonic()
    worker.status.call().get(timeout=60)
    status_latency = time.monotonic() - t0

    t0 = time.monotonic()
    worker.cancel.call().get(timeout=60)
    cancel_latency = time.monotonic() - t0

    sample_outcome = _one_value(slow.get(timeout=60))

    # Error contract: a raising @concurrent_endpoint must reach the caller as
    # ActorError and leave the actor alive. The bare form forwards the error;
    # only an explicit_response_port endpoint fails the actor.
    try:
        worker.boom.call().get(timeout=60)
        error_contract = "no-error"
    except BaseException as exc:
        error_contract = type(exc).__name__
    try:
        worker.status.call().get(timeout=60)
        actor_survives = True
    except BaseException:
        actor_survives = False

    try:
        procs.stop("dispatch contract probe done").get(timeout=60)
    except Exception:
        pass  # teardown noise must not mask the measured result

    print(json.dumps({
        "status_latency_s": round(status_latency, 3),
        "cancel_latency_s": round(cancel_latency, 3),
        "sample_outcome": sample_outcome,
        "error_contract": error_contract,
        "actor_survives": actor_survives,
    }), flush=True)


if __name__ == "__main__":
    main()
