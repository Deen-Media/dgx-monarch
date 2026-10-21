"""Companion boundary guards that a browser or a hostile client can reach.

Raw sockets send bytes the aiohttp client never would, under both HTTP parsers.
The background-loop tests run the real ``create_app(poll=True)`` lifecycle.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import sqlite3
import subprocess

import aiohttp.http_parser
import aiohttp.web_protocol
import pytest
from aiohttp.test_utils import TestServer
from PIL import Image

from dgx_monarch.studio import server as server_module
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.media_assets import UPLOAD_BYTES
from dgx_monarch.studio.store import Store
from test_studio_http_quality import preview_request
from test_studio_integration import PNG, companion

PARSERS = {"python": aiohttp.http_parser.HttpRequestParserPy,
           "c": getattr(aiohttp.http_parser, "HttpRequestParserC", None)}


@pytest.fixture(autouse=True)
def no_management_processes(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("boundary checks must not run management processes")
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)


async def raw_get(browser, path: bytes, authorization: bytes) -> tuple[int, dict[bytes, bytes], bytes]:
    host, port = browser.server.host, browser.server.port
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(b"GET " + path + b" HTTP/1.1\r\nHost: " + f"{host}:{port}".encode()
                 + b"\r\nConnection: close\r\nAuthorization: " + authorization + b"\r\n\r\n")
    await writer.drain()
    raw = await reader.read()
    writer.close()
    await writer.wait_closed()
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    headers = {name.strip().lower(): value.strip() for name, _, value in (line.partition(b":") for line in lines[1:])}
    return int(lines[0].split(b" ", 2)[1]), headers, body


@pytest.mark.parametrize("parser", sorted(PARSERS))
@pytest.mark.parametrize("value", ["latin1_e_acute", "byte_ff", "token_then_ff"])
def test_non_utf8_authorization_bytes_get_the_json_401(tmp_path, monkeypatch, parser, value):
    if PARSERS[parser] is None:
        pytest.skip("aiohttp was built without its C parser")
    monkeypatch.setattr(aiohttp.web_protocol, "HttpRequestParser", PARSERS[parser])

    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            sent = {"latin1_e_acute": b"Bearer caf\xe9", "byte_ff": b"Bearer \xff",
                    "token_then_ff": system.headers["Authorization"].encode() + b"\xff"}[value]
            status, headers, body = await raw_get(system.browser, b"/api/v1/capabilities", sent)
            assert status == 401
            assert headers[b"content-type"].startswith(b"application/json")
            assert json.loads(body)["error"]["code"] == "authentication"
            # The same raw path still accepts the real token.
            status, _, body = await raw_get(system.browser, b"/api/v1/capabilities",
                                            system.headers["Authorization"].encode())
            assert status == 200 and json.loads(body)["schema_version"]
    asyncio.run(scenario())


@pytest.mark.parametrize(("after", "status"), [
    ("-1", 400), ("9223372036854775808", 400), ("99999999999999999999999", 400), ("x", 400),
    ("9223372036854775807", 200), ("0", 200),
])
def test_event_cursor_is_bounded_to_the_sqlite_integer_range(tmp_path, monkeypatch, after, status):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            response = await system.browser.get("/api/v1/events", params={"after": after}, headers=system.headers)
            body = await response.json()
            assert response.status == status, body
            if status == 400:
                assert body["error"]["code"] == "cursor"
            else:
                assert isinstance(body["events"], list)
    asyncio.run(scenario())


def test_non_chunked_oversize_media_is_a_clean_413(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            monkeypatch.setattr(server_module, "MEDIA_LIMIT", len(PNG) - 1)
            system.protocol.chunked_media = False
            system.store.create("artifacts", {"id": "big", "filename": "result.png", "subfolder": "studio", "type": "output"})
            response = await system.browser.get("/api/v1/artifacts/big/file", headers=system.headers)
            assert response.status == 413
            assert (await response.json())["error"]["code"] == "media_limit"
    asyncio.run(scenario())


def png_of_size(total: int) -> bytes:
    data = io.BytesIO()
    Image.new("RGB", (64, 64), (1, 2, 3)).save(data, "PNG")
    raw = data.getvalue()
    return raw + b"\0" * (total - len(raw))  # Pillow ignores bytes after IEND.


@pytest.mark.parametrize(("size", "status"), [(UPLOAD_BYTES, 201), (UPLOAD_BYTES + 1, 413)])
def test_upload_limit_is_exactly_two_mebibytes_through_http(tmp_path, monkeypatch, size, status):
    assert UPLOAD_BYTES + 1 == 2_097_153

    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            body = {"kind": "input", "media_type": "image/png",
                    "data_base64": base64.b64encode(png_of_size(size)).decode()}
            response = await system.browser.post("/api/v1/assets", headers=system.headers, json=body)
            value = await response.json()
            assert response.status == status, value
            if status == 413:
                assert value["error"]["code"] == "asset_limit"
                assert system.protocol.uploaded == {}
            else:
                assert len(system.protocol.uploaded) == 1
    asyncio.run(scenario())


def test_snapshot_cursor_comes_from_the_row_read_without_decoding_events(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            empty = await system.request("GET", "/snapshot")
            assert empty["cursor"] == 0
            for index in range(3):
                system.store.create("artifacts", {"id": f"a{index}", "filename": "result.png",
                                                  "subfolder": "studio", "type": "output"})
            system.store.update("artifacts", "a0", {"note": "changed"})
            with system.store.lock:
                latest = system.store.db.execute("SELECT MAX(cursor) FROM events").fetchone()[0]

            def decoded(*args, **kwargs):
                raise AssertionError("the snapshot must not decode the event log")
            monkeypatch.setattr(system.store, "events", decoded)
            snapshot = await system.request("GET", "/snapshot")
            assert snapshot["cursor"] == latest
    asyncio.run(scenario())


def failed_retry_scenario(tmp_path, monkeypatch, kind, change="choices"):
    """Create a failed job or grid cell, then change the driver schema.

    ``choices`` adds a sampler choice, which a retry's live revalidation accepts;
    ``definition`` lowers the steps maximum, which it refuses.
    """
    async def setup(system):
        preview = await system.request("POST", f"/{kind}/preview", json=preview_request(kind))
        accepted = await system.request("POST", f"/{kind}", expected=202,
                                        json={"preview_id": preview["id"], "idempotency_key": f"{kind}-once"})
        job_id = accepted["id"] if kind == "jobs" else accepted["cells"][0]["job_id"]
        await system.store.wait_state("jobs", job_id, "queued")
        system.store.update("jobs", job_id, {"state": "failed"})
        sampler = system.protocol.schema["Sampler"]["input"]["required"]
        if change == "choices":
            sampler["sampler"] = [["euler", "heun", "dpmpp_2m"]]
        else:
            sampler["steps"][1]["max"] = 99
        path = f"/jobs/{job_id}/retry" if kind == "jobs" else f"/grids/{accepted['id']}/retry"
        body = {"confirmed": True, "idempotency_key": f"{kind}-retry"}
        return path, body if kind == "jobs" else {**body, "indices": [0]}
    return setup


@pytest.mark.parametrize("kind", ["jobs", "grids"])
def test_confirmed_retry_refreshes_the_driver_schema_first_and_replays_offline(tmp_path, monkeypatch, kind):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            path, body = await failed_retry_scenario(tmp_path, monkeypatch, kind)(system)
            assert system.jobs.schema["Sampler"]["input"]["required"]["sampler"] == [["euler", "heun"]]
            accepted = await system.request("POST", path, expected=202, json=body)
            assert system.jobs.schema == system.protocol.schema
            await system.upstream.close()
            replay = await system.request("POST", path, expected=202, json=body)
            assert replay["id"] == accepted["id"]
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["jobs", "grids"])
def test_confirmed_retry_revalidates_against_the_refreshed_schema(tmp_path, monkeypatch, kind):
    # The cached schema still matches the frozen review, so only a refresh before
    # the retry can see the changed definition and refuse it.
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            path, body = await failed_retry_scenario(tmp_path, monkeypatch, kind, "definition")(system)
            assert system.jobs.schema["Sampler"]["input"]["required"]["steps"][1]["max"] == 100
            refused = await system.request("POST", path, expected=409, json=body)
            assert refused["error"]["code"] == "schema_changed"
            assert system.jobs.schema == system.protocol.schema
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["jobs", "grids"])
def test_unconfirmed_retry_is_refused_without_contacting_the_driver(tmp_path, monkeypatch, kind):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            path, body = await failed_retry_scenario(tmp_path, monkeypatch, kind)(system)
            await system.upstream.close()
            refused = await system.request("POST", path, expected=409, json={**body, "confirmed": False})
            assert refused["error"]["code"] in {"retry_scope", "grid_retry", "confirmation"}
    asyncio.run(scenario())


class Driver:
    origin = "http://127.0.0.1:8188"

    def __init__(self):
        self.started = self.closed = False

    async def start(self):
        self.started = True

    async def close(self):
        self.closed = True


async def until(condition, timeout=5.0):
    async def wait():
        while not condition():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait(), timeout)


def lifecycle_app(tmp_path, monkeypatch, driver):
    captured = {}

    def make_jobs(store, client):
        captured["jobs"] = Jobs(store, client)
        return captured["jobs"]

    monkeypatch.setattr(server_module, "Jobs", make_jobs)
    monkeypatch.setattr(server_module, "OBSERVE_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(server_module, "BACKOFF_SECONDS", 0.02, raising=False)
    store = Store(tmp_path / "state")
    app = server_module.create_app(store, driver, "lifecycle-owner-" + "x" * 32, allowed_hosts={"studio"})
    return app, store, captured["jobs"]


def test_background_loops_survive_unexpected_errors_and_report_the_driver_degraded(tmp_path, monkeypatch, caplog):
    driver = Driver()
    app, store, jobs = lifecycle_app(tmp_path, monkeypatch, driver)
    reconciles, watches, degraded = [], [], []

    async def reconcile():
        reconciles.append(dict(jobs.driver))
        if len(reconciles) == 1:
            raise sqlite3.OperationalError("database or disk is full")
        if len(reconciles) == 2:
            degraded.append(dict(jobs.driver))

    async def watch_events():
        watches.append(True)
        if len(watches) == 1:
            raise RecursionError("maximum recursion depth exceeded")
        await asyncio.Event().wait()

    jobs.reconcile = reconcile
    jobs.watch_events = watch_events

    async def scenario():
        with caplog.at_level(logging.ERROR, logger=server_module.__name__):
            async with TestServer(app):
                await until(lambda: len(reconciles) >= 4 and len(watches) >= 2)
        assert driver.started and driver.closed

    try:
        asyncio.run(scenario())
    finally:
        store.close()
    assert degraded[0]["state"] == "unavailable" and degraded[0]["code"] == "observer_failed"
    assert isinstance(degraded[0]["since"], (int, float, str))
    messages = [record.getMessage() for record in caplog.records]
    assert any("Studio observer failed" in message for message in messages)
    assert any("Studio event_watch failed" in message for message in messages)


def test_shutdown_survives_a_failed_listener_and_still_closes_the_driver(tmp_path, monkeypatch, caplog):
    driver = Driver()

    async def broken(*args, **kwargs):
        raise RuntimeError("listener failed")

    app, store, jobs = lifecycle_app(tmp_path, monkeypatch, driver)
    monkeypatch.setattr(server_module, "supervise", broken, raising=False)
    pending = {}

    async def scenario():
        with caplog.at_level(logging.ERROR, logger=server_module.__name__):
            async with TestServer(app):
                pending["task"] = asyncio.create_task(asyncio.sleep(3600))
                jobs.tasks.add(pending["task"])
                await asyncio.sleep(0.05)
        assert pending["task"].cancelled()
        assert driver.closed

    try:
        asyncio.run(scenario())
    finally:
        store.close()
    assert sum("background listener ended with an error" in record.getMessage() for record in caplog.records) == 2
