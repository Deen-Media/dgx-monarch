"""The polling lifecycle settles jobs on its own and survives faults outside the driver protocol."""
from __future__ import annotations

import asyncio
import sqlite3
import time
import uuid

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch.studio import SCHEMA_VERSION, reconcile
from dgx_monarch.studio import server as server_module
from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store

SCHEMA = {"Render": {"input": {"required": {"steps": ["INT", {"min": 1, "max": 20}]}},
                     "input_order": {"required": ["steps"]}, "output": [], "output_node": True}}


class LiveDriver:
    """In-process ComfyUI stand-in: queue, history, websocket events and the companion's polled routes."""

    def __init__(self):
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = {}
        self.events = []
        self.sockets = set()
        self.run = asyncio.Event()
        self.prompts = []

    async def broadcast(self, message):
        self.events.append(message)
        for socket in list(self.sockets):
            try:
                await socket.send_json(message)
            except ConnectionError:
                self.sockets.discard(socket)

    async def execute(self, prompt_id, row):
        await self.run.wait()
        self.queue["queue_pending"].remove(row)
        self.queue["queue_running"].append(row)
        await self.broadcast({"type": "execution_start", "data": {"prompt_id": prompt_id, "timestamp": 1_000}})
        finished = ["execution_success", {"prompt_id": prompt_id, "timestamp": 2_000}]
        self.queue["queue_running"].remove(row)
        self.history[prompt_id] = {"prompt": row, "status": {"status_str": "success", "completed": True,
                                                              "messages": [finished]},
                                   "outputs": {"1": {"images": [{"filename": "done.png", "subfolder": "", "type": "output"}]}}}
        await self.broadcast({"type": finished[0], "data": finished[1]})

    def app(self):
        app = web.Application()

        async def websocket(request):
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            first = await socket.receive()
            assert first.type == WSMsgType.TEXT and "feature_flags" in first.data
            # A reconnecting client receives the events it may have missed.
            for message in list(self.events):
                await socket.send_json(message)
            self.sockets.add(socket)
            async for _ in socket:
                pass
            self.sockets.discard(socket)
            return socket

        async def submit(request):
            body = await request.json()
            # ComfyUI queues under the client's canonical identity, or mints one when none is sent.
            identity = body.get("prompt_id") or str(uuid.uuid4())
            row = [len(self.prompts), identity, body["prompt"], body["extra_data"], ["1"]]
            self.prompts.append(body)
            self.queue["queue_pending"].append(row)
            asyncio.get_running_loop().create_task(self.execute(identity, row))
            return web.json_response({"prompt_id": identity, "number": row[0], "node_errors": {}})

        async def one(request):
            identity = request.match_info["identity"]
            return web.json_response({identity: self.history[identity]} if identity in self.history else {})

        async def cancel(request):
            return web.json_response({"cancelled": False})

        def fixed(value):
            async def handler(request):
                return web.json_response(value() if callable(value) else value)
            return handler

        app.router.add_get("/ws", websocket)
        app.router.add_get("/dgxm/execution-events", fixed({"schema_version": 1, "producer_id": "driver-test",
                                                            "events": [], "next_sequence": 0, "omitted_events": 0}))
        app.router.add_get("/history/{identity}", one)
        app.router.add_post("/api/jobs/{identity}/cancel", cancel)
        app.router.add_get("/queue", fixed(lambda: self.queue))
        app.router.add_get("/history", fixed(lambda: self.history))
        app.router.add_post("/prompt", submit)
        app.router.add_get("/object_info", fixed(SCHEMA))
        app.router.add_get("/system_stats", fixed({"system": {"comfyui_version": "lifecycle"}}))
        app.router.add_get("/dgxm/telemetry", fixed({"workers": []}))
        return app


async def until(check, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "lifecycle did not reach the expected state"
        await asyncio.sleep(0.05)


def test_polling_lifecycle_settles_a_job_and_survives_a_full_disk(tmp_path, monkeypatch, caplog):
    instances = []

    def jobs_factory(store, client):
        jobs = Jobs(store, client)
        jobs.event_pause, jobs.backoff_base, jobs.backoff_limit = 0.05, 0.05, 0.2
        instances.append(jobs)
        return jobs
    monkeypatch.setattr(server_module, "Jobs", jobs_factory)

    async def scenario():
        driver = LiveDriver()
        async with TestServer(driver.app()) as upstream:
            store = Store(tmp_path / "state")
            client = ComfyClient(str(upstream.make_url("/")).rstrip("/"))
            token = "lifecycle-owner-" + "x" * 32
            hosts: set[str] = set()
            app = create_app(store, client, token, allowed_hosts=hosts, poll=True)
            try:
                async with TestClient(TestServer(app)) as browser:
                    hosts.add(str(browser.make_url("/")).split("://", 1)[1].rstrip("/"))
                    headers = {"Authorization": "Bearer " + token, "X-DGXM-Schema": SCHEMA_VERSION}
                    jobs = instances[0]
                    draft = {"workflow": {"1": {"class_type": "Render", "inputs": {"steps": 2}}}, "values": {},
                             "seed_policy": "fixed", "operation": "image"}
                    preview = await (await browser.post("/api/v1/jobs/preview", headers=headers, json=draft)).json()
                    response = await browser.post("/api/v1/jobs", headers=headers,
                                                  json={"preview_id": preview["id"], "idempotency_key": "once"})
                    job = await response.json()
                    assert response.status == 202
                    await until(lambda: store.get("jobs", job["id"])["state"] == "queued")
                    await until(lambda: bool(driver.sockets))
                    # Every record write fails as on a full disk until the fault is cleared.
                    full = {"on": True}
                    put = store._put

                    def failing_put(*args):
                        if full["on"]:
                            raise sqlite3.OperationalError("database or disk is full")
                        return put(*args)
                    monkeypatch.setattr(store, "_put", failing_put)
                    driver.run.set()
                    await until(lambda: jobs.observer_failures >= 1 and jobs.events_fault is not None)
                    assert jobs.driver["state"] == "unavailable" and jobs.driver["code"] == "studio_storage"
                    assert isinstance(jobs.driver["since"], float)
                    full["on"] = False
                    # No explicit reconcile: the observer and the watcher both carry on.
                    await until(lambda: store.get("jobs", job["id"]).get("reconciliation") is not None)
                    value = store.get("jobs", job["id"])
                    assert value["state"] == "succeeded" and value["prompt_id"] == job["client_prompt_id"]
                    assert any(row["job_id"] == job["id"] for row in store.all("artifacts"))
                    await until(lambda: jobs.events_fault is None and jobs.driver["state"] == "connected")
                    assert len(driver.prompts) == 1
            finally:
                store.close()
    asyncio.run(scenario())
    messages = [record.getMessage() for record in caplog.records]
    assert any("Studio reconciliation failed (studio_storage)" in message for message in messages)
    assert any("Studio event watcher failed (studio_storage)" in message for message in messages)


def test_observer_backs_off_after_an_unexpected_fault_and_recovers(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")
    try:
        class Quiet:
            calls = 0

            async def request(self, method, path, data=None):
                Quiet.calls += 1
                return {"queue_running": [], "queue_pending": []} if path == "/queue" else {}
        jobs = Jobs(store, Quiet())
        jobs.backoff_base, jobs.backoff_limit = 30.0, 30.0
        original = reconcile.known_jobs

        def full(selected):
            raise sqlite3.OperationalError("database or disk is full")
        monkeypatch.setattr(reconcile, "known_jobs", full)
        asyncio.run(jobs.reconcile())
        assert jobs.driver["state"] == "unavailable" and jobs.driver["code"] == "studio_storage"
        since = jobs.driver["since"]
        monkeypatch.setattr(reconcile, "known_jobs", original)
        asyncio.run(jobs.reconcile())  # Inside the back-off window: no driver traffic.
        assert Quiet.calls == 0 and jobs.driver["since"] == since
        jobs.observer_retry_at = 0.0
        asyncio.run(jobs.reconcile())
        assert jobs.driver["state"] == "connected" and "code" not in jobs.driver
        assert jobs.observer_failures == 0
    finally:
        store.close()


@pytest.mark.parametrize("fault", [RecursionError, OverflowError, sqlite3.OperationalError, KeyError])
def test_watcher_survives_any_non_cancellation_fault(tmp_path, monkeypatch, fault):
    store = Store(tmp_path / "state")
    try:
        class Stream:
            connections = 0

            async def stream_events(self, client_id):
                Stream.connections += 1
                if Stream.connections > 3:
                    raise asyncio.CancelledError
                yield {"type": "execution_start", "data": {"prompt_id": "p"}}
        jobs = Jobs(store, Stream())
        jobs.event_pause, jobs.backoff_base, jobs.backoff_limit = 0.0, 0.0, 0.0

        def broken(message, **kwargs):
            raise fault("synthetic")
        monkeypatch.setattr(jobs, "consume_event", broken)
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(jobs.watch_events())
        assert Stream.connections == 4
        assert jobs.events_fault["state"] == "unavailable" and jobs.driver["events"]["code"] == jobs.events_fault["code"]
    finally:
        store.close()
