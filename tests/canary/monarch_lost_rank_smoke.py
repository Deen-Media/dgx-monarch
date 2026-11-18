#!/usr/bin/env python3
"""Explicit local-process canary: the stop report for a lost rank beside a live one.

Two Worker loops on 127.0.0.1 (plain Monarch, never a configured Worker), a
named ProcMesh across them, one loop's process group killed, then the mesh
stopped. Monarch's ProcMesh controller cannot confirm the lost rank, ends the
stop with "error stopping proc mesh" after proc_stop_max_idle, and fails; the
stop's Future does not complete. dgx-monarch ties that report to its own stop
by the controller's label, ``proc_mesh_controller_<ProcMesh name>-<uid>``
(src/dgx_monarch/proc_stops.py). The real report goes through that very
matcher here, a stop registered for this mesh's name taking it, so a
rewording or a respelling upstream turns this red instead of leaving the
pack's wait quietly at its bound. ``loop ADDRESS`` runs one Worker loop;
the canary runs this with dgx-monarch's src on PYTHONPATH.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

NAME = "dgxm-procs-canary00"
MARKERS = ("actor proc_mesh_controller_", "error stopping proc mesh", "Timeout(")


def loop(address: str) -> None:
    from monarch.actor import enable_transport, run_worker_loop_forever

    enable_transport("tcp://127.0.0.1:0")
    run_worker_loop_forever(address=address, ca="trust_all_connections")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def run() -> dict[str, object]:
    import monarch.actor as monarch_actor
    from monarch.actor import attach_to_workers, enable_transport, shutdown_context

    from dgx_monarch import proc_stops

    observations: dict[str, object] = {}
    ports = [_free_port(), _free_port()]
    loops = [subprocess.Popen([sys.executable, __file__, "loop", f"tcp://127.0.0.1:{port}"],
                              start_new_session=True, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL) for port in ports]
    faults: list[tuple[float, str]] = []
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not all(map(_listening, ports)):
            time.sleep(0.2)
        assert all(map(_listening, ports)), "the Worker loops never listened"
        monarch_actor.unhandled_fault_hook = lambda failure: faults.append(
            (time.monotonic(), str(failure)))
        enable_transport("tcp://127.0.0.1:0")
        hosts = attach_to_workers(ca="trust_all_connections",
                                  workers=[f"tcp://127.0.0.1:{port}" for port in ports])
        hosts.initialized.get(timeout=60)
        procs = hosts.spawn_procs(per_host={"gpus": 1}, name=NAME)
        procs.initialized.get(timeout=60)
        os.killpg(loops[0].pid, signal.SIGKILL)
        loops[0].wait(10)
        proc_stops.remember(procs, NAME)
        registered = proc_stops.started(procs, keep_s=300)  # what mesh_stop.stop_procs does
        future = procs.stop("dgx-monarch canary stop")
        done: dict[str, float] = {}

        def wait() -> None:
            try:
                future.get(timeout=150)
            except BaseException:
                pass
            done["at"] = time.monotonic()

        started = time.monotonic()
        threading.Thread(target=wait, daemon=True).start()
        report = None
        while time.monotonic() < started + 90 and report is None:
            report = next((entry for entry in faults if MARKERS[1] in entry[1]), None)
            time.sleep(0.2)
        assert report is not None, f"no stop report within 90 s: {[text[:200] for _t, text in faults]}"
        reported_at, text = report
        time.sleep(5.0)
        assert all(marker in text for marker in MARKERS), text[:600]
        # The pack's own matcher must take the real report for this stop.
        assert proc_stops.claim(text) is True, text[:600]
        assert registered.report == text and registered.ended.is_set()
        observations["stop_report_claimed_by_proc_stops"] = True
        # The stop's own Future does not finish once the controller has failed.
        assert "at" not in done or done["at"] > reported_at + 5.0, (done, reported_at)
        observations["stop_report_markers"] = True
        observations["stop_report_label_names_the_proc_mesh"] = True
        observations["stop_report_future_pending_5s_after"] = "at" not in done
        observations["stop_report_s"] = round(reported_at - started, 1)
        observations["stop_report_text"] = " ".join(text.split())[:600]
    finally:
        for process in loops:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(10)
        try:
            shutdown_context(host_timeout=10).get(timeout=15)
        except BaseException:
            pass
    return observations


if __name__ == "__main__":
    if sys.argv[1] == "loop":
        loop(sys.argv[2])
    else:
        Path(sys.argv[1]).write_text(json.dumps(run(), indent=2) + "\n")
        os._exit(0)
