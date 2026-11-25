"""A normal driver exit is clean whether or not a call is in flight; a lost Worker still reconciles.

At a normal interpreter exit (ComfyUI on Ctrl+C or a closed server),
torchmonarch's own exit hook stops every mesh the driver owns, Workers' actors
included (meta-pytorch/monarch#5152, #5070). The pack used to register its own
stop at import, so it ran after Monarch's, against a closed client: on 0.7.0 it
logged an ERROR "proc stop on detach" line ("timed out" or "failed") at an exit
with no call running and left the handle blocked. On 0.8.0 that late stop also
ran the actor stop added for the stop reason (mesh_stop.stop_with_reason),
whose native side panicked during interpreter exit. The pack now registers no
exit stop.

A call in flight at exit fails when Monarch's hook stops its mesh. The pack's
exit flag (driver_exit), whose hook runs before Monarch's and logs one INFO
line from the main thread, makes the eviction, the reconcile and the fault hook
do nothing, where they logged three ERROR lines that said replacement was
blocked and ComfyUI must restart (and, once on CPU, a fatal error as that
reconcile logged during interpreter shutdown). Monarch
starts each local actor process in its own process group, so a terminal's
Ctrl+C reaches only the driver, and the probe's ``sigint`` and ``hold`` modes
send it to the driver's whole group as a terminal does; ``hold`` does so while
a call holds its actor's process, as a render's step does. In ``collect`` the
render's own collect (mesh_lease.collect_sample) waits on such a call under the
Worker mesh's name: the call the exit stops reads like a lost contact, and the
collect must word no loss and warn of none. The ``kill`` mode is the other
direction: a Worker killed while the driver is not exiting is still evicted and
its fleet stopped.

A SIGTERM'd interpreter runs no exit hook at all, on either version, and its
actors wait for the actor lease (docs/TROUBLESHOOTING.md #51).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROBE = Path(__file__).with_name("exit_clean_probe.py")
SRC = Path(__file__).resolve().parents[1] / "src"
NOTICE = "The driver is exiting: Monarch's exit step stops the Worker actors"


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return False


def _run(tmp_path: Path, mode: str,
         probe: Path = PROBE) -> tuple[subprocess.CompletedProcess, list[dict], float]:
    runtime_dir = tmp_path / "run"
    runtime_dir.mkdir(mode=0o700)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MONARCH_", "HYPERACTOR_", "OTEL_", "DGXM_"))}
    environment.update(
        HOME=str(tmp_path), XDG_RUNTIME_DIR=str(runtime_dir), CUDA_VISIBLE_DEVICES="",
        PROBE_COMFY_DIR=str(tmp_path / "comfy"), PROBE_READY=str(tmp_path / "ready.json"),
        PROBE_ENTRIES=str(tmp_path / "entries.txt"), PROBE_LAST_HOOK=str(tmp_path / "last_hook.txt"),
        PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))))
    ready = tmp_path / "ready.json"
    started = time.monotonic()
    child = subprocess.Popen([sys.executable, str(probe), mode], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, env=environment,
                             start_new_session=True)
    try:
        if mode in ("sigint", "hold", "heal_sigint"):
            while not ready.exists() and child.poll() is None and time.monotonic() < started + 120:
                time.sleep(0.1)
            if ready.exists():
                os.killpg(child.pid, signal.SIGINT)  # as a terminal's Ctrl+C reaches its foreground group
        out, err = child.communicate(timeout=240)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate()
    exited = time.monotonic()
    proc = subprocess.CompletedProcess(child.args, child.returncode, out, err)
    records = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
    assert exited - started < 200
    return proc, records, exited


def _actors_gone(actors: list[int], exited: float) -> None:
    assert actors, "the probe found no actor process to watch"
    deadline = exited + 15.0
    while any(_alive(pid) for pid in actors) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not any(_alive(pid) for pid in actors), actors


def _error_lines(stderr: str) -> list[str]:
    return [line for line in stderr.splitlines() if " ERROR " in line]


@pytest.mark.parametrize("mode", ["idle", "inflight", "sigint", "hold", "collect"])
def test_a_normal_exit_is_clean_with_or_without_a_call_in_flight(tmp_path, mode):
    pytest.importorskip("monarch.actor")
    proc, records, exited = _run(tmp_path, mode)
    detail = f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-3000:]}"
    assert proc.returncode == 0, detail
    first, last = records[0], records[-1]
    # Nothing ran against a closed client: no native panic, no failed stop, no error.
    assert "panicked" not in proc.stderr, detail
    assert "proc stop on detach" not in proc.stderr, detail
    assert "NCCL teardown on detach" not in proc.stderr, detail
    assert _error_lines(proc.stderr) == [], detail
    # No line says a Worker was lost: the exit stopped the call (collect: the render's own collect).
    assert [line for line in proc.stderr.splitlines() if " WARNING " in line] == [], detail
    assert proc.stderr.count(NOTICE) == 1, detail  # the exit hook's one line, from the main thread
    if mode != "idle":
        # Calls were still running (hold, collect: the one call had not returned).
        assert last["calls"] >= (1 if mode in ("hold", "collect") else 3) and last["error"] is None, detail
        assert last.get("held") is None, detail
    if mode in ("sigint", "hold"):
        assert last.get("interrupted") is True, detail
        # Monarch starts each local actor process in its own group: Ctrl+C reaches the driver only.
        assert first["group"] not in first["actor_groups"], detail
    _actors_gone(first["actors"], exited)


@pytest.mark.parametrize("mode", ["heal", "heal_sigint"])
def test_an_exit_inside_a_heals_own_stop_is_quiet(tmp_path, mode):
    """ComfyUI exits while a render's restart heal is stopping a fleet whose
    Worker processes are gone, on the render's own thread. Monarch's exit step
    ends that stop ("client stopped"); the render's thread then logs nothing,
    probes no host and attaches nothing, so the exit shows the one notice and
    no WARNING or ERROR. Before, that thread logged the failed stop, the
    liveness proof and the refused or retired outcome after the notice."""
    pytest.importorskip("monarch.actor")
    proc, records, _exited = _run(tmp_path, mode)
    detail = f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-3000:]}"
    assert proc.returncode == 0, detail
    assert "panicked" not in proc.stderr, detail
    assert "proc stop on detach" not in proc.stderr, detail
    assert "restart_reattach outcome=" not in proc.stderr, detail
    assert "liveness fold" not in proc.stderr, detail
    assert _error_lines(proc.stderr) == [], detail
    assert [line for line in proc.stderr.splitlines() if " WARNING " in line] == [], detail
    assert proc.stderr.count(NOTICE) == 1, detail
    assert records[-1]["calls"] == 1 and records[-1].get("held") is None, detail  # the heal never finished
    if mode == "heal_sigint":
        assert records[-1].get("interrupted") is True, detail


@pytest.mark.parametrize("mode", ["queue", "queue_wrapped"])
def test_a_queue_behind_the_cut_render_holds_one_entry_and_prints_nothing(tmp_path, mode):
    """A real interpreter exit with ComfyUI's prompt worker holding a queue: the
    render the exit cuts, then four queued prompts entering ``mesh.ensure_live``
    on the real handle, directly or (``queue_wrapped``) through a node method
    under the pack's wrapper, which holds before the node's own work. Exactly
    one entry starts and holds at the entry (driver_exit.hold_at_entry); none
    raises, none prints, the exit shows its one notice and no WARNING or ERROR,
    and the process ends right after its last exit hook, the held daemon thread
    with it, with code 0. The 5 s bound leaves a slow runner room; a held thread
    that lengthened the exit would add the 30 s hold."""
    pytest.importorskip("monarch.actor")
    proc, _records, _exited = _run(tmp_path, mode)
    ended = time.time()
    detail = f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-3000:]}"
    assert proc.returncode == 0, detail
    entries = (tmp_path / "entries.txt").read_text().splitlines()
    assert len(entries) == 1 and entries[0].startswith("entered "), entries  # one enters; it holds
    # (queue_wrapped: no "inside the node" line, so the wrapper held before the node's own work)
    assert "panicked" not in proc.stderr and "Fatal Python error" not in proc.stderr, detail
    assert _error_lines(proc.stderr) == [], detail
    assert [line for line in proc.stderr.splitlines() if " WARNING " in line] == [], detail
    assert "Traceback" not in proc.stderr and "DriverExitingError" not in proc.stderr, detail
    assert proc.stderr.count(NOTICE) == 1, detail
    last_hook = float((tmp_path / "last_hook.txt").read_text())
    assert ended - last_hook < 5.0, ended - last_hook  # the held thread lengthens nothing


def test_a_worker_killed_while_the_driver_is_not_exiting_is_evicted_and_stopped(tmp_path):
    pytest.importorskip("monarch.actor")
    proc, records, exited = _run(tmp_path, "kill")
    detail = f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-3000:]}"
    assert proc.returncode == 0, detail
    first, last = records[0], records[-1]
    # The in-flight call reads Monarch's kill text, or, when the kill lands
    # between status checks, the status-request timeout about 10 s later.
    error = last["error"] or ""
    assert "Killed(sig=9)" in error or "status request to monitored actor" in error, detail
    assert last["defunct"] is True and last["teardown_complete"] is True, detail
    assert last["replacement_blocked"] is None, detail
    assert proc.stderr.count(NOTICE) == 1, detail  # the exit that follows; the kill came while it ran
    _actors_gone(first["actors"], exited)
