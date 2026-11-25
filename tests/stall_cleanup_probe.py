"""A stall eviction's cleanup on a fleet the pack's own get_mesh brought up (local, CPU).

For tests/test_cleanup_running_wording.py. The fleet runs the pack's real
GPUWorker with one step added that holds its GPU lock on the GPU thread, as a
render's sample stalled in a collective does, so the reconcile's group teardown
(the Worker's real ``teardown_group``) waits on that lock. The render's own
collect (``mesh_lease.collect_sample``) awaits the step under the stall guard;
the stall eviction (``mark_defunct_deliberate``) starts the reconcile, and the
probe reads ``mesh.ensure_live``'s refusal while it runs. With no other
render's lease held, the collect itself waits for that cleanup before it
raises, as the render that retires the last lease does.

Four seams, each named where it is set: the fleet's class is that Worker; the
stall budget and the guard's slice are cut to seconds (the test's
environment drops every ``DGXM_`` variable, so the budget is set here); and
``setup_key`` is set by hand, the state a render's setup leaves, which a CPU
box cannot run. Without it the shutdown skips the group teardown.

The last stdout line is a JSON object.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time

from monarch.actor import concurrent_endpoint

from dgx_monarch.actor import GPUWorker

HOLD_S = 600.0


class _StallWorker(GPUWorker):
    """The pack's Worker, plus a step that holds the GPU lock on the GPU thread, as a stalled sample does."""

    @concurrent_endpoint
    async def stall(self, seconds: float) -> float:
        async with self._gpu_lock:
            return await self._on_gpu(_sleep, seconds)


def _sleep(seconds: float) -> float:
    time.sleep(seconds)
    return seconds


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[tuple[float, int, str]] = []

    def emit(self, record):
        self.lines.append((time.monotonic(), record.levelno, record.getMessage()[:400]))


def main() -> None:
    import dgx_monarch.actor as actor_package
    from dgx_monarch import mesh, mesh_lease, reconcile_running, worker_box, worker_loss
    from dgx_monarch.log import get_logger

    actor_package.GPUWorker = _StallWorker  # seam: spawn_worker_fleet spawns the fleet's class from here
    os.environ[mesh_lease.SAMPLE_STALL_ENV] = "2"  # seam: the stall budget, in seconds
    mesh_lease._STALL_SLICE_S = 0.5  # seam: the guard's poll slice
    lines = _Lines()
    for name in ("dgx_monarch.mesh", "dgx_monarch.worker_box", "dgx_monarch.mesh_helpers"):
        get_logger(name).addHandler(lines)

    handle = mesh.get_mesh(mode="local", gpus_per_host=1, comfy_dir=os.environ["PROBE_COMFY_DIR"])
    handle.workers.status.call().get(timeout=60)
    handle.setup_key = ("stall-probe",)  # seam: what a render's setup leaves; the group teardown runs
    record: dict = {"refusals": []}
    done = threading.Event()

    def queued_renders() -> None:  # a render queued while the cleanup runs, every 2 s
        while not done.is_set():
            thread = getattr(handle, "_supervision_reconcile_thread", None)
            if thread is not None and thread.is_alive():
                try:
                    mesh.ensure_live(handle)
                    words = "live"
                except BaseException as exc:
                    words = str(exc)
                if thread.is_alive():  # given while the cleanup still ran
                    record["refusals"].append(words)
                done.wait(2.0)
            else:
                done.wait(0.1)

    threading.Thread(target=queued_renders, name="queued", daemon=True).start()
    started = time.monotonic()
    future = handle.workers.stall.call(HOLD_S)
    try:  # with no other lease held, the collect waits for the cleanup before it raises
        mesh_lease.collect_sample(handle, future, HOLD_S + 60.0, activity_fn=lambda: started)
        record["stall"] = None
    except BaseException as exc:
        record["stall"] = f"{type(exc).__name__}: {exc}"
        record["stall_words"] = worker_box.lost_worker_words(exc)
        record["stall_loss"] = worker_loss.read(str(exc)) is not None
    record["collect_s"] = round(time.monotonic() - started, 1)
    thread = getattr(handle, "_supervision_reconcile_thread", None)
    record["reconcile_started"] = thread is not None
    if thread is not None:
        thread.join(timeout=200.0)
    ended = time.monotonic()
    done.set()
    stamp = getattr(handle, reconcile_running.RECONCILE_STARTED, None)
    group = [at for at, _level, line in lines.lines if "NCCL teardown on detach" in line]
    try:
        future.get(timeout=10)
        record["held_call"] = "returned"
    except BaseException as exc:
        text = f"{type(exc).__name__}: {exc}"
        record["held_call"] = text[:600]
        record["held_call_loss"] = worker_loss.read(str(exc)) is not None
    record.update(
        reconcile_at_s=round(stamp - started, 1) if isinstance(stamp, float) else None,
        cleanup_s=round(ended - stamp, 1) if isinstance(stamp, float) else None,
        group_wait_s=round(group[0] - stamp, 1) if group and isinstance(stamp, float) else None,
        teardown_complete=bool(getattr(handle, "teardown_complete", False)),
        replacement_blocked=getattr(handle, "replacement_blocked", None),
        warned=bool(getattr(handle, worker_box.LOSS_WARNED, False)),
        figure_s=reconcile_running.reconcile_stop_limit_s(),
        warnings=[line for _at, level, line in lines.lines if level >= logging.WARNING])
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
