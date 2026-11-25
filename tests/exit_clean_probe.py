"""A driver exit, or a lost Worker, on a fleet the pack's own get_mesh brought up (local, CPU).

test_exit_clean.py runs it as a subprocess, because monarch allows one
transport per process and the exit is the thing under test. It builds a local
one-rank fleet through ``mesh.get_mesh``, proves the Worker answers and prints
the actor process ids as one JSON line. Then, by mode:

- ``idle``: returns, so the interpreter exits the way ComfyUI does when it
  stops with no render running.
- ``inflight``: a daemon thread calls the Worker's status endpoint in a loop
  through the handle's own ``_await_or_evict``, as a render's calls go, and
  the main thread returns while a call is in flight.
- ``sigint``: the same loop, and the main thread waits for SIGINT, catches the
  KeyboardInterrupt and returns, as ComfyUI's main does on Ctrl+C.
- ``kill``: the same loop; the probe kills its own Worker's actor process
  (SIGKILL) while ComfyUI is NOT exiting, waits for the eviction and the
  reconcile, and prints what the handle published as a second JSON line.
- ``hold``: a call that holds its actor's process for 60 s, as a render's
  step does, on an actor spawned in the Worker's own processes, through the
  handle's ``_await_or_evict``; then the main thread waits for SIGINT as in
  ``sigint``. The actor's class is defined here, in ``__main__``, so it
  travels to the actor process by value and needs no module there.
- ``collect``: the same holding actor, spawned under the Worker mesh's own
  name, with the call awaited by the render's own collect
  (``mesh_lease.collect_sample``) on a daemon thread; the main thread returns.
- ``heal``: a render's restart heal (``restart_reattach.heal``) on a daemon
  thread, for a fleet whose Worker processes are gone, as after a restart: its
  stop kills the actor processes as it starts, and the main thread returns half
  a second into that stop. ``heal_sigint``: the same, then the main thread
  waits for SIGINT as in ``sigint``.
- ``queue``: ComfyUI's prompt worker with prompts queued behind a cut render:
  a daemon thread runs the status loop of ``inflight`` (the render the exit
  cuts), then four ``mesh.ensure_live`` calls on the real handle, each marked
  in ``PROBE_ENTRIES`` before it starts. The main thread returns. A hook
  registered before the mesh build (so it runs after Monarch's exit step)
  waits until the first queued entry has started, then writes the time it
  returned to ``PROBE_LAST_HOOK``. ``queue_wrapped``: the same, each queued
  prompt entering a node method under ``refusal_lead.lead_with_refusal`` that
  marks ``inside the node`` before its own ``mesh.ensure_live``.

The last stdout line is always a JSON object.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time

from monarch.actor import Actor, endpoint


class _Busy(Actor):
    """Blocks its process for as long as it is asked, as a render's step does."""

    @endpoint
    async def hold(self, seconds: float) -> float:
        time.sleep(seconds)
        return seconds


def _actor_processes() -> list[int]:
    """This process's descendants that run a Monarch actor bootstrap."""
    found, todo = [], [os.getpid()]
    while todo:
        pid = todo.pop()
        try:
            for task in os.listdir(f"/proc/{pid}/task"):
                with open(f"/proc/{pid}/task/{task}/children") as f:
                    todo.extend(int(child) for child in f.read().split())
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if b"monarch._src.actor.bootstrap_main" in f.read():
                    found.append(pid)
        except OSError:
            pass
    return found


_RENDER: list[threading.Thread] = []


def _join_render() -> None:
    """Registered before the mesh build, so it runs after Monarch's exit step and
    the exit flag's hook: the render's thread gets to finish whatever it does
    after the exit began, so a line it logs then is never lost to the race with
    the interpreter's end."""
    for thread in _RENDER:
        thread.join(30)


def _group_of(pid: int) -> int | None:
    """A process's group, or None once it is gone (``heal`` kills its actors)."""
    try:
        return os.getpgid(pid)
    except ProcessLookupError:
        return None


def _calls(handle, record: dict) -> None:
    try:
        while True:
            handle._await_or_evict(handle.workers.status.call(), 30)
            record["calls"] += 1
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"


def _hold(handle, busy, record: dict) -> None:
    try:
        record["calls"] += 1
        handle._await_or_evict(busy.hold.call(60.0), 120)
        record["held"] = "returned"
    except BaseException as exc:
        record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"


def main() -> None:
    from dgx_monarch import mesh

    mode = sys.argv[1] if len(sys.argv) > 1 else "idle"
    if mode in ("heal", "heal_sigint"):
        import atexit

        atexit.register(_join_render)
    queued = threading.Event()
    if mode in ("queue", "queue_wrapped"):
        import atexit

        def _until_queued() -> None:
            queued.wait(30)
            with open(os.environ["PROBE_LAST_HOOK"], "w") as out:
                out.write(repr(time.time()))

        atexit.register(_until_queued)
    handle = mesh.get_mesh(mode="local", gpus_per_host=1, comfy_dir=os.environ["PROBE_COMFY_DIR"])
    handle.workers.status.call().get(timeout=60)
    actors = _actor_processes()
    record: dict = {"calls": 0, "error": None}
    if mode == "hold":
        busy = handle.procs.spawn("dgxm_busy", _Busy)
        busy.hold.call(0.0).get(timeout=60)  # constructed and answering
        threading.Thread(target=_hold, args=(handle, busy, record), name="inflight", daemon=True).start()
        time.sleep(1.0)  # the 60 s hold is running in the actor's process
        actors = _actor_processes()
    elif mode == "collect":
        from monarch.actor import this_host

        from dgx_monarch import mesh_lease

        # Under the Worker mesh's own name, so the error the exit leaves reads as a render step's would.
        busy = this_host().spawn_procs(per_host={"gpus": 1}).spawn("dgxm_worker", _Busy)
        busy.hold.call(0.0).get(timeout=60)

        def _collect() -> None:  # the render's own collect, as PendingRender.result runs it
            try:
                record["calls"] += 1
                mesh_lease.collect_sample(handle, busy.hold.call(60.0), 120.0)
                record["held"] = "returned"
            except BaseException as exc:
                record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

        threading.Thread(target=_collect, name="render", daemon=True).start()
        time.sleep(1.5)
        actors = _actor_processes()
    elif mode in ("heal", "heal_sigint"):
        from dgx_monarch import restart_reattach
        from dgx_monarch.delivery_faults import WorkerDeliveryError

        entered = threading.Event()
        real_shutdown, gone = handle.shutdown, list(actors)

        def shutdown(*args, **kwargs):  # the heal's own stop, of processes a restart ended
            for pid in gone:
                os.kill(pid, signal.SIGKILL)
            entered.set()
            return real_shutdown(*args, **kwargs)

        handle.shutdown = shutdown
        refused = WorkerDeliveryError("[dgxm:C guard=cross_rank_capacity waivable=0] undeliverable",
                                      moves=restart_reattach.Moves(moved=("tcp://127.0.0.1:1",)))

        def _heal() -> None:
            try:
                record["calls"] += 1
                restart_reattach.heal(handle, refused)
                record["held"] = "healed"
            except BaseException as exc:
                record["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"

        _RENDER.append(threading.Thread(target=_heal, name="render", daemon=True))
        _RENDER[0].start()
        entered.wait(60)
        time.sleep(float(os.environ.get("PROBE_HEAL_DELAY_S", "0.5")))  # inside the heal's 2 s stop
    elif mode in ("queue", "queue_wrapped"):
        from dgx_monarch.nodes.refusal_lead import lead_with_refusal

        @lead_with_refusal
        def _node() -> object:  # a node method under the pack's wrapper, as a sampler's is
            with open(os.environ["PROBE_ENTRIES"], "a") as out:
                out.write("inside the node\n")
            return mesh.ensure_live(handle)

        def _entry() -> object:
            return _node() if mode == "queue_wrapped" else mesh.ensure_live(handle)

        def _prompt_worker() -> None:
            _calls(handle, record)  # the render the exit cuts
            for _prompt in range(4):  # the prompts queued behind it
                with open(os.environ["PROBE_ENTRIES"], "a") as out:
                    out.write(f"entered {time.time()!r}\n")
                queued.set()
                try:
                    _entry()
                except BaseException as exc:
                    with open(os.environ["PROBE_ENTRIES"], "a") as out:
                        out.write(f"raised {type(exc).__name__}\n")

        threading.Thread(target=_prompt_worker, name="prompt-worker", daemon=True).start()
        while record["calls"] < 3 and record["error"] is None:
            time.sleep(0.05)
    elif mode != "idle":
        threading.Thread(target=_calls, args=(handle, record), name="inflight", daemon=True).start()
        while record["calls"] < 3 and record["error"] is None:
            time.sleep(0.05)
    groups = sorted({group for group in map(_group_of, actors) if group is not None})
    ready = json.dumps({"actors": actors, "actor_groups": groups, "group": os.getpgrp()})
    print(ready, flush=True)
    if os.environ.get("PROBE_READY"):  # the caller waits for this file, then signals
        with open(os.environ["PROBE_READY"] + ".part", "w") as f:
            f.write(ready)
        os.replace(os.environ["PROBE_READY"] + ".part", os.environ["PROBE_READY"])
    if mode in ("sigint", "hold", "heal_sigint"):
        try:
            time.sleep(60)
        except KeyboardInterrupt:
            print(json.dumps({"interrupted": True, "calls": record["calls"],
                              "error": record["error"], "held": record.get("held")}), flush=True)
        return
    if mode == "kill":
        os.kill(actors[0], signal.SIGKILL)
        deadline = time.monotonic() + 150
        while time.monotonic() < deadline and not (
                getattr(handle, "teardown_complete", False)
                or getattr(handle, "replacement_blocked", None)):
            time.sleep(0.2)
        print(json.dumps({
            "defunct": bool(getattr(handle, "defunct", False)),
            "teardown_complete": bool(getattr(handle, "teardown_complete", False)),
            "replacement_blocked": getattr(handle, "replacement_blocked", None),
            "error": record["error"]}), flush=True)
        return
    print(json.dumps({"calls": record["calls"], "error": record["error"]}), flush=True)


if __name__ == "__main__":
    main()
