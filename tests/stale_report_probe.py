"""A real Recycle of a local fleet, with another ProcMesh's stop report arriving during its stop.

Run it in a fresh interpreter (monarch allows one transport per process);
tests/test_proc_stop_reports.py runs it as a subprocess. It brings a local CPU
fleet up through the pack's own get_mesh, then runs Recycle. With ``stale``,
the controller report a restart heal's 2 s stop left for another ProcMesh
(captured on loopback) goes through the pack's real fault hook the moment the
Recycle's proc stop registers; with ``control``, nothing is fed. Prints one
JSON object as the last stdout line.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
STALE = ("MeshFailure(mesh_name=<none>, crashed_ranks=[], event=Supervision event: actor "
         "proc_mesh_controller_anon-wnpcwboqf6n<cVAgKdNmYsM>.local<9yH82kDpLa4>@tcp://127.0.0.1:33255 "
         "failed:\n  error stopping proc mesh: statuses: Timeout(30.000805711s)=0..2)")


def main(mode: str) -> None:
    from dgx_monarch import mesh, mesh_teardown, proc_stops

    handle = mesh.get_mesh(mode="local", gpus_per_host=1, comfy_dir="/nonexistent-comfy")
    handle.workers.status.call().get(timeout=60)
    import monarch.actor as monarch_actor

    out: dict[str, object] = {"mode": mode, "name": proc_stops.name_of(handle.procs)}

    def feed() -> None:
        until = time.monotonic() + 60
        while time.monotonic() < until and not proc_stops._STOPS:
            time.sleep(0.001)
        out["stops_in_flight_when_fed"] = len(proc_stops._STOPS)
        monarch_actor.unhandled_fault_hook(STALE)

    if mode == "stale":
        threading.Thread(target=feed, daemon=True).start()
    started = time.monotonic()
    outcome = handle.recycle_detailed()
    out.update(recycle_s=round(time.monotonic() - started, 2), status=str(outcome.status),
               teardown_complete=bool(handle.teardown_complete),
               replacement_blocked=handle.replacement_blocked,
               block_cause=mesh_teardown.block_cause(handle))
    print(json.dumps(out, default=str), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
    os._exit(0)
