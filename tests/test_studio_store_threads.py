"""The Studio store's one SQLite connection is shared by the event loop and
worker threads; CPU fakes only.

Settlement, live previews and operation previews write on worker threads.
Every use of the connection, close included, holds the store lock, and a
cancelled caller ends only after its thread does, so shutdown returns before
the owner closes the store only when no thread still writes. Closing the connection under
a statement on another thread is a segmentation fault, so the race runs in a
child interpreter.
"""
from __future__ import annotations

import asyncio
import os
import sqlite3
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path

from aiohttp import web

import dgx_monarch
from dgx_monarch.studio import reconcile
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.wire_records import snapshot_window

TOKEN = "store-threads-owner-token-" + "x" * 32
CLOSE_RACE = r"""
import sqlite3, sys, tempfile, threading, time
from pathlib import Path
from dgx_monarch.studio.store import Store

errors = set()
for attempt in range(int(sys.argv[1])):
    store = Store(Path(tempfile.mkdtemp(dir=sys.argv[2])) / "state")
    started = threading.Event()

    def write():
        started.set()
        try:
            while True:
                store.create("artifacts", {"job_id": "job", "kind": "image", "source": "driver_history",
                                           "intermediate": False})
        except sqlite3.ProgrammingError as exc:
            errors.add(str(exc))

    thread = threading.Thread(target=write)
    thread.start()
    started.wait()
    time.sleep(0.001)
    store.close()
    thread.join()
print(sorted(errors))
"""


class Driver:
    """ComfyUI-shaped queue and history; a lookup for an unknown prompt answers empty."""

    origin = "http://127.0.0.1:8188"

    def __init__(self):
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = {}

    async def start(self):
        return None

    async def close(self):
        return None

    async def request(self, method, path, data=None):
        if path == "/queue":
            return deepcopy(self.queue)
        if path == "/history":
            return deepcopy(self.history)
        return {}

    async def history_prompt(self, identity):
        return {identity: deepcopy(self.history[identity])} if identity in self.history else {}


def completed(prompt_id, outputs=1):
    return {"prompt": [1, prompt_id, {}, {}, []],
            "status": {"completed": True, "status_str": "success", "messages": [["execution_success", {}]]},
            "outputs": {str(node): {"images": [{"filename": f"{prompt_id}-{node}.png", "subfolder": "", "type": "output"}]}
                        for node in range(outputs)}}


def queued(store, identity, prompt_id):
    store.create("jobs", {"id": identity, "state": "queued", "prompt_id": prompt_id,
                          "submission": {"prompt": {}, "workflow": {}, "digest": "d"}})


def test_close_waits_for_a_write_in_flight_on_another_thread(tmp_path):
    store = Store(tmp_path / "state")
    holding, release = threading.Event(), threading.Event()

    def write():
        with store.lock, store.transaction():
            store._put("jobs", {"id": "held", "state": "queued"}, 1)
            holding.set()
            release.wait(10)

    writer = threading.Thread(target=write)
    writer.start()
    assert holding.wait(10)
    closer = threading.Thread(target=store.close)
    closer.start()
    closer.join(0.3)
    waited = closer.is_alive()
    release.set()
    writer.join(10)
    closer.join(10)
    assert waited, "close ran while another thread held an open transaction"
    assert not closer.is_alive()
    try:
        store.create("jobs", {"id": "late"})
    except sqlite3.ProgrammingError as exc:
        assert "closed database" in str(exc)
    else:
        raise AssertionError("a write after close reached the closed connection")
    store.close()  # A second close is harmless.
    reopened = Store(tmp_path / "state")
    try:
        assert reopened.get("jobs", "held")["state"] == "queued"
    finally:
        reopened.close()


def test_close_racing_a_writer_thread_never_crashes_the_interpreter(tmp_path):
    """The CI segmentation fault: close on one thread while another runs a statement."""
    source = str(Path(dgx_monarch.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [source, os.environ.get("PYTHONPATH")]))}
    result = subprocess.run([sys.executable, "-X", "faulthandler", "-c", CLOSE_RACE, "60", str(tmp_path)],
                            capture_output=True, text=True, env=env, timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip() == "['Cannot operate on a closed database.']", result.stdout


def test_settlement_on_a_worker_thread_while_the_loop_writes_and_reads(tmp_path, monkeypatch):
    """Each cycle settles new jobs on a worker thread while the loop keeps writing and reading.

    The loop thread runs in asyncio debug mode, which raises on any loop call
    from a foreign thread. Each settlement waits until the loop has written at
    least once inside it, so every cycle overlaps.
    """
    cycles, per_cycle, loop_jobs = 12, 6, 4
    settling, overlapped = threading.Event(), threading.Event()
    settle_threads: set[int] = set()
    real_settle = reconcile._settle

    def overlapping_settle(*args):
        settle_threads.add(threading.get_ident())
        overlapped.clear()
        settling.set()
        try:
            overlapped.wait(10)
            real_settle(*args)
        finally:
            settling.clear()

    monkeypatch.setattr(reconcile, "_settle", overlapping_settle)

    async def scenario():
        loop_thread = threading.get_ident()
        store = Store(tmp_path / "state")
        client = Driver()
        jobs = Jobs(store, client)
        for index in range(loop_jobs):
            store.create("jobs", {"id": f"loop-{index}", "state": "queued", "progress": 0,
                                  "submission": {"prompt": {}, "workflow": {}, "digest": "d"}})
        done = asyncio.Event()
        loop_writes = overlapping_writes = 0

        async def loop_side():
            nonlocal loop_writes, overlapping_writes
            cursor = 0
            while not done.is_set():
                inside = settling.is_set()
                loop_writes += 1
                store.update("jobs", f"loop-{loop_writes % loop_jobs}", {"progress": loop_writes})
                page = store.events(cursor)
                cursors = [row["cursor"] for row in page["events"]]
                assert cursors == sorted(set(cursors)) and all(value > cursor for value in cursors)
                cursor = page["cursor"]
                snapshot_window(store)
                if inside:
                    overlapping_writes += 1
                    overlapped.set()
                await asyncio.sleep(0)

        async def settle_side():
            try:
                for cycle in range(cycles):
                    for index in range(per_cycle):
                        prompt = f"prompt-{cycle}-{index}"
                        queued(store, f"job-{cycle}-{index}", prompt)
                        client.history[prompt] = completed(prompt, outputs=2)
                    await jobs.reconcile()
                    assert jobs.observer_failures == 0 and jobs.driver["state"] == "connected", jobs.driver
            finally:
                done.set()

        try:
            await asyncio.gather(settle_side(), loop_side())
            settled = [store.get("jobs", f"job-{cycle}-{index}")["state"]
                       for cycle in range(cycles) for index in range(per_cycle)]
            artifacts = store.all("artifacts")
            progress = {store.get("jobs", f"loop-{index}").get("progress") for index in range(loop_jobs)}
            integrity = store.db.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            store.close()
        return loop_thread, settled, artifacts, progress, integrity, loop_writes, overlapping_writes

    loop_thread, settled, artifacts, progress, integrity, loop_writes, overlapping = asyncio.run(scenario(), debug=True)
    assert settle_threads and loop_thread not in settle_threads
    assert overlapping >= cycles and loop_writes >= overlapping
    assert settled == ["succeeded"] * (cycles * per_cycle)
    assert len(artifacts) == cycles * per_cycle * 2
    assert loop_writes in progress
    assert integrity == "ok"


def test_shutdown_waits_for_a_settlement_thread_before_the_store_closes(tmp_path, monkeypatch):
    """Cancelling the observer cannot stop its thread; shutdown returns only after the thread does."""
    entered, release = threading.Event(), threading.Event()
    real_import = reconcile.import_artifacts

    def held_import(*args, **kwargs):
        entered.set()
        release.wait(10)
        real_import(*args, **kwargs)

    monkeypatch.setattr(reconcile, "import_artifacts", held_import)

    async def scenario():
        store = Store(tmp_path / "state")
        client = Driver()
        queued(store, "job", "prompt")
        client.history["prompt"] = completed("prompt")
        runner = web.AppRunner(create_app(store, client, TOKEN, allowed_hosts=set()), access_log=None)
        await runner.setup()
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            cleanup = asyncio.create_task(runner.cleanup())
            await asyncio.sleep(0.3)
            returned_early = cleanup.done()
            release.set()
            await asyncio.wait_for(cleanup, 10)
            return returned_early, store.get("jobs", "job"), store.all("artifacts")
        finally:
            release.set()
            store.close()

    returned_early, job, artifacts = asyncio.run(scenario(), debug=True)
    assert not returned_early, "shutdown returned while settlement was still writing"
    assert job["state"] == "succeeded"
    assert [artifact["job_id"] for artifact in artifacts] == ["job"]
