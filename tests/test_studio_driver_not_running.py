"""A refused connection at the driver origin reads as "driver not running"; every other failure stays unknown."""
from __future__ import annotations

import asyncio
import errno
import logging
import socket
import ssl
import time
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch.studio import SCHEMA_VERSION
from dgx_monarch.studio import server as server_module
from dgx_monarch.studio.comfy import DRIVER_NOT_RUNNING, ComfyClient
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store

TOKEN = "first-run-owner-" + "x" * 32
HEADERS = {"Authorization": "Bearer " + TOKEN, "X-DGXM-Schema": SCHEMA_VERSION}
KEY = SimpleNamespace(host="127.0.0.1", port=9, ssl=True)


def closed_port() -> int:
    """A loopback port nothing listens on: bound once to pick it, then closed."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Failing:
    """A driver session whose every request fails before a response, as a connect failure does."""

    def __init__(self, error: BaseException):
        self.error = error

    def request(self, *args, **kwargs):
        error = self.error

        class Context:
            async def __aenter__(self):
                raise error

            async def __aexit__(self, *exc):
                return False
        return Context()


async def failure(error: BaseException) -> StudioError:
    client = ComfyClient("http://127.0.0.1:9")
    client.session = Failing(error)  # type: ignore[assignment]
    with pytest.raises(StudioError) as caught:
        await client.request("GET", "/queue")
    return caught.value


def test_a_refused_connection_is_driver_not_running():
    async def scenario():
        client = ComfyClient(f"http://127.0.0.1:{closed_port()}")
        await client.start()
        try:
            for path in ("/queue", "/object_info"):
                with pytest.raises(StudioError) as caught:
                    await client.request("GET", path)
                assert (caught.value.code, str(caught.value), caught.value.status) == (
                    "driver_not_running", DRIVER_NOT_RUNNING, 503)
        finally:
            await client.close()
        refused = aiohttp.ClientConnectorError(KEY, ConnectionRefusedError(errno.ECONNREFUSED, "Connection refused"))
        mocked = await failure(refused)
        assert (mocked.code, mocked.status) == ("driver_not_running", 503)
        assert mocked.__cause__ is refused
    asyncio.run(scenario())


@pytest.mark.parametrize("error", [
    TimeoutError(),
    aiohttp.ClientConnectorError(KEY, ConnectionResetError(errno.ECONNRESET, "Connection reset by peer")),
    # aiohappyeyeballs merges failures from several addresses into one OSError with no errno.
    aiohttp.ClientConnectorError(KEY, OSError("Multiple exceptions: [Errno 111] refused, [Errno 99] address")),
    aiohttp.ClientConnectorSSLError(KEY, ssl.SSLError(1, "handshake failed")),
    aiohttp.ClientOSError(errno.ECONNREFUSED, "refused after the connection was made"),
    aiohttp.ServerDisconnectedError(),
], ids=["timeout", "reset", "merged", "tls", "not_a_connect", "disconnected"])
def test_other_connect_failures_stay_unknown(error):
    caught = asyncio.run(failure(error))
    assert (caught.code, caught.status) == ("driver_unknown", 503)


def test_one_reconcile_cycle_degrades_without_logging(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    store = Store(tmp_path / "state")

    async def scenario():
        client = ComfyClient(f"http://127.0.0.1:{closed_port()}")
        await client.start()
        try:
            jobs = Jobs(store, client)
            await jobs.reconcile()
            first = jobs.driver["since"]
            await jobs.reconcile()
            return jobs, first
        finally:
            await client.close()
    try:
        jobs, first = asyncio.run(scenario())
    finally:
        store.close()
    assert jobs.driver["state"] == "unavailable" and jobs.driver["code"] == "driver_not_running"
    # The second cycle keeps when the outage began, and an expected outage grows no back-off.
    assert jobs.driver["since"] == first and isinstance(first, float)
    assert (jobs.observer_failures, jobs.observer_retry_at) == (0, 0.0)
    assert caplog.records == []


@pytest.mark.parametrize("start", [
    None,
    {"command": "systemctl --user start comfyui-driver.service", "source": "driver_service",
     "config": "/home/user/.config/dgx-monarch/cluster.toml"},
])
def test_capabilities_carry_driver_start(tmp_path, start):
    async def scenario():
        store = Store(tmp_path / "state")
        client = ComfyClient("http://127.0.0.1:8191")
        hosts: set[str] = set()
        operations = Operations(store, client, driver_start=start)
        app = create_app(store, client, TOKEN, allowed_hosts=hosts, operations=operations, poll=False)
        try:
            async with TestClient(TestServer(app)) as browser:
                hosts.add(str(browser.make_url("/")).split("://", 1)[1].rstrip("/"))
                response = await browser.get("/api/v1/capabilities", headers=HEADERS)
                assert response.status == 200
                return await response.json()
        finally:
            store.close()
    body = asyncio.run(scenario())
    assert body["driver_origin"] == "http://127.0.0.1:8191"
    assert body["driver_start"] == (start or {"command": None, "source": None, "config": None})
    assert set(body["driver_start"]) == {"command", "source", "config"}


def fixture_driver() -> web.Application:
    """The routes the observer reads, answering as an idle driver."""
    def fixed(value):
        async def handler(request):
            return web.json_response(value)
        return handler

    app = web.Application()
    app.router.add_get("/queue", fixed({"queue_running": [], "queue_pending": []}))
    app.router.add_get("/history", fixed({}))
    app.router.add_get("/dgxm/telemetry", fixed({"workers": []}))
    app.router.add_get("/dgxm/execution-events", fixed({"schema_version": 1, "producer_id": "driver-test", "events": [],
                                                        "next_sequence": 0, "omitted_events": 0}))
    app.router.add_get("/object_info", fixed({}))
    return app


def test_snapshot_reports_driver_not_running_then_connected(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "OBSERVE_SECONDS", 0.2)
    cycles = {"count": 0, "connected": None}
    reconcile = Jobs.reconcile

    async def counted(self):
        await reconcile(self)
        cycles["count"] += 1
        if self.driver["state"] == "connected" and cycles["connected"] is None:
            cycles["connected"] = cycles["count"]
    monkeypatch.setattr(Jobs, "reconcile", counted)
    port = closed_port()

    async def scenario():
        store = Store(tmp_path / "state")
        client = ComfyClient(f"http://127.0.0.1:{port}")
        hosts: set[str] = set()
        app = create_app(store, client, TOKEN, allowed_hosts=hosts, poll=True)
        runner = web.AppRunner(fixture_driver())
        try:
            async with TestClient(TestServer(app)) as browser:
                hosts.add(str(browser.make_url("/")).split("://", 1)[1].rstrip("/"))

                async def driver_state():
                    response = await browser.get("/api/v1/snapshot", headers=HEADERS)
                    return (await response.json())["driver"]

                async def wait_for(check):
                    deadline = time.monotonic() + 20
                    while not check(state := await driver_state()):
                        assert time.monotonic() < deadline, state
                        await asyncio.sleep(0.05)
                    return state

                state = await wait_for(lambda state: state.get("code") == "driver_not_running")
                assert state["state"] == "unavailable" and isinstance(state["since"], float)
                for response in (await browser.get("/api/v1/catalog", headers=HEADERS),
                                 await browser.post("/api/v1/workflows/inspect", headers=HEADERS,
                                                    json={"workflow": {}, "values": {}})):
                    assert response.status == 503
                    assert (await response.json())["error"] == {"code": "driver_not_running", "message": DRIVER_NOT_RUNNING}
                await runner.setup()
                await web.TCPSite(runner, "127.0.0.1", port).start()
                started = cycles["count"]
                state = await wait_for(lambda state: state["state"] == "connected")
                # One cycle may have begun before the driver listened; the next one connects.
                assert cycles["connected"] is not None and cycles["connected"] - started <= 2
                assert "code" not in state
                assert (await browser.get("/api/v1/catalog", headers=HEADERS)).status == 200
        finally:
            await runner.cleanup()
            store.close()
    asyncio.run(scenario())
