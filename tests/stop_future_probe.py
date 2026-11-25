"""Stop probe: what a stop whose Future completed without error says about its ranks.

A bring-up whose rollback stop completed without error lifts no poison and
asks no host (mesh_factory.left_nothing_running), on the same signal a normal
teardown trusts before it attaches a replacement. Monarch's own words are
wider: the Future completes once every proc has stopped, failed or timed
out, and the host agent can mark a rank failed. This probe pins the property
the lift rests on: when the Future completes without error, no rank process
of that ProcMesh still runs.

Run it in a fresh interpreter, because monarch allows one transport per
process; test_stop_future_contract.py runs it as a subprocess. It spawns a
local CPU ProcMesh of two ranks, freezes one rank's process (SIGSTOP; this
probe started it), sends the stop, and watches its Future for WATCH_S. It
then resumes that rank and lets the stop finish. The rank pids and start
times go to the file named in argv[1] as soon as they are known, so the test
can kill any rank that outlives the probe.

Prints exactly one JSON object as the last stdout line.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time

from dgx_monarch.cli import proc_identity as pi
from dgx_monarch.monarch_policy import MonarchPolicy, activate_native_policy, apply_policy

# The phases (with the 5 s freeze wait) add up to 260 s, inside the test's 300 s
# bound on the whole probe, which leaves room for the interpreter and policy start.
READY_S = 45.0     # each of: every rank's proc started; every rank answered its pid
WATCH_S = 40.0     # how long the frozen rank's stop is watched
FINISH_S = 90.0    # how long the stop may take once the rank resumes
GONE_S = 30.0      # how long every rank may take to leave once the stop ended


def _state(pid: int, starttime: int) -> str:
    """The rank's procfs state, or "gone" when its pid no longer names it."""
    if not pi.alive(pid, starttime):
        return "gone"
    return pi.read_state(pid) or "gone"


def _running(state: str) -> bool:
    """A rank that still runs, as against one gone or exiting (state Z)."""
    return state not in ("gone", "Z")


def main(ranks_file: str) -> None:
    apply_policy(MonarchPolicy(), role="driver")
    import monarch.actor as monarch_actor

    result: dict[str, object] = {"native_readback": activate_native_policy().get("native_readback")}

    class Ident(monarch_actor.Actor):
        @monarch_actor.endpoint
        async def pid(self) -> int:
            return os.getpid()

    procs = monarch_actor.this_host().spawn_procs(per_host={"procs": 2})
    procs.initialized.get(timeout=READY_S)
    pids = sorted(procs.spawn("ident", Ident).pid.call().get(timeout=READY_S).values())
    ranks = [(pid, (pi.read_stat(pid) or (0, 0))[1]) for pid in pids]
    children = [(pid, stat[1]) for pid in pi.iter_pids()
                if (stat := pi.read_stat(pid)) is not None and stat[0] == os.getpid()]
    with open(ranks_file, "w", encoding="utf-8") as handle:
        json.dump({"ranks": ranks, "children": children}, handle)
    result["ranks"] = len(ranks)
    frozen, frozen_start = ranks[0]
    os.kill(frozen, signal.SIGSTOP)
    deadline = time.monotonic() + 5.0
    while _state(frozen, frozen_start) != "T" and time.monotonic() < deadline:
        time.sleep(0.05)
    result["frozen_state_before_stop"] = _state(frozen, frozen_start)

    box: dict[str, object] = {}
    ended = threading.Event()
    sent = time.monotonic()
    future = procs.stop("dgx-monarch stop contract probe")

    def wait() -> None:
        # One get with a long bound: a returns_future stop is cancelled by a get that times out.
        try:
            future.get(timeout=WATCH_S + FINISH_S)
            box["outcome"] = "ok"
        except BaseException as exc:
            box["outcome"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        box["frozen_state_at_end"] = _state(frozen, frozen_start)  # read the moment it ended
        box["at_s"] = round(time.monotonic() - sent, 2)
        ended.set()

    threading.Thread(target=wait, daemon=True).start()
    result["while_frozen"] = dict(box) if ended.wait(WATCH_S) else "pending"
    if _running(_state(frozen, frozen_start)):
        os.kill(frozen, signal.SIGCONT)
    result["final"] = dict(box) if ended.wait(FINISH_S) else "pending"
    deadline = time.monotonic() + GONE_S
    while any(_running(_state(pid, start)) for pid, start in ranks) and time.monotonic() < deadline:
        time.sleep(0.1)
    result["left_running"] = [pid for pid, start in ranks if _running(_state(pid, start))]
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
    os._exit(0)
