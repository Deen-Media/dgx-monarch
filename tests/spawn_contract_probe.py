"""Spawn and stop probe: what bring-up and teardown rest on under torchmonarch 0.8.0.

Run it in a fresh interpreter, because monarch allows one transport per
process; test_spawn_contract.py runs it as a subprocess. It installs the
driver policy the way monarch_launch.prepare_driver does (apply_policy, the
native import, activate_native_policy), then spawns local CPU actors only.

Four observations:

- The root Python logger's level. 0.8.0 sets it to INFO when the factory's
  ``this_host()`` creates this process's client; the factory puts it back.
- A per-rank bootstrap callable that raises. 0.8.0 runs it on the calling
  thread before anything spawns, so ``spawn_procs`` raises the callable's own
  exception and registers no ProcMesh; mesh_factory relies on that to confirm
  the rollback without stopping anything.
- The real worker-fleet factory with a GPUWorker whose constructor raises.
  Under 0.8.0's direct spawn ``initialized`` fails with a RuntimeError whose
  text opens with Monarch's supervisor chain; the factory's error must open
  with the Worker's own words.
- A call in flight when teardown stops the fleet. Under direct spawn only
  ``ActorMesh.stop`` carries the stop reason to it; ``ProcMesh.stop`` alone
  reaches it as "parent draining". mesh_teardown.stop_with_reason stops the
  actors with the reason first.

Prints exactly one JSON object as the last stdout line.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import types

from dgx_monarch.monarch_policy import MonarchPolicy, activate_native_policy, apply_policy


def _type(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


def main() -> None:
    apply_policy(MonarchPolicy(), role="driver")
    import monarch.actor as monarch_actor
    from monarch._src.actor.proc_mesh import get_active_proc_meshes

    from dgx_monarch import mesh_factory, mesh_teardown
    from dgx_monarch.config import ClusterConfig

    result: dict[str, object] = {"native_readback": activate_native_policy().get("native_readback")}
    faults: list[str] = []
    # The driver's fault hook consumes faults the same way (mesh_helpers.install_unhandled_fault_hook).
    monarch_actor.unhandled_fault_hook = lambda failure: faults.append(str(failure).split("\n", 1)[0])

    class RefusingWorker(monarch_actor.Actor):
        def __init__(self) -> None:
            raise RuntimeError("deliberate constructor refusal")

        @monarch_actor.endpoint
        async def status(self) -> str:
            return "ok"

    class BlockingWorker(monarch_actor.Actor):
        @monarch_actor.endpoint
        async def block(self) -> str:
            import asyncio

            await asyncio.sleep(60)
            return "finished"

    refused: list[BaseException] = []

    def refusing_command(_point: object) -> object:
        error = ValueError("deliberate bootstrap refusal")
        refused.append(error)
        raise error

    # The factory's this_host() is this process's first client-creating call,
    # which on 0.8.0 sets the root logger to INFO unless the pack puts it back.
    logging.getLogger().setLevel(logging.WARNING)
    actor_module = types.ModuleType("dgx_monarch.actor")
    actor_module.GPUWorker = RefusingWorker  # type: ignore[attr-defined]
    sys.modules["dgx_monarch.actor"] = actor_module
    try:
        mesh_factory.spawn_worker_fleet(ClusterConfig(), False, 1, None, None)
    except mesh_factory.WorkerSpawnError as exc:
        result["factory_original"] = _type(exc.original)
        result["factory_text"] = exc.original_text[:4000]
        result["factory_rollback_confirmed"] = exc.rollback_confirmed
    else:
        result["factory_original"] = "returned"
    result["root_level_after_factory"] = logging.getLevelName(logging.getLogger().level)

    host = monarch_actor.this_host()
    before = len(get_active_proc_meshes())
    try:
        host.spawn_procs(per_host={"procs": 1}, bootstrap_command=refusing_command)
    except BaseException as exc:
        result["bootstrap_spawn"] = "raised"
        result["bootstrap_error_is_callables"] = bool(refused) and exc is refused[0]
    else:
        result["bootstrap_spawn"] = "returned"
    result["bootstrap_proc_meshes_added"] = len(get_active_proc_meshes()) - before

    procs = host.spawn_procs(per_host={"procs": 1})
    procs.initialized.get(timeout=60)
    workers = procs.spawn("blocking", BlockingWorker)
    in_flight = workers.block.call()
    time.sleep(1.0)
    outcome: dict[str, str] = {}

    def observe() -> None:
        try:
            in_flight.get(timeout=60)
            outcome["text"] = "returned"
        except BaseException as exc:
            outcome["text"] = str(exc)

    watcher = threading.Thread(target=observe)
    watcher.start()
    handle = types.SimpleNamespace(workers=workers, procs=procs)
    mesh_teardown.stop_with_reason(handle, "dgx-monarch recycle", 60)
    watcher.join(timeout=60)
    result["in_flight_after_stop"] = outcome.get("text", "no answer")[:600]
    result["faults"] = [text[:200] for text in faults]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
