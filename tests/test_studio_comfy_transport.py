"""HTTP and event protocol tests use a loopback mock server, never a real driver."""
from __future__ import annotations

import asyncio
import json
import os
import struct
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dgx_monarch import finite_json
from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.contracts import (
    MAX_BODY_BYTES,
    OBJECT_INFO_BYTES,
    StudioError,
    canonical,
    schema_document,
)

ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def driver(*routes):
    app = web.Application()
    for method, path, handler in routes:
        app.router.add_route(method, path, handler)
    async with TestServer(app) as server:
        client = ComfyClient(str(server.make_url("/")))
        await client.start()
        try:
            yield client
        finally:
            await client.close()


def raw_json(body: bytes, status: int = 200):
    async def handler(request):
        return web.Response(body=body, status=status, content_type="application/json")
    return handler


def test_atomic_cancel_endpoint_has_no_global_fallback_and_binds_safe_path():
    async def scenario():
        seen = []

        async def cancel(request):
            seen.append(request.match_info["id"])
            if request.match_info["id"] == "unsupported":
                raise web.HTTPNotFound()
            return web.json_response({"cancelled": True})

        async def global_interrupt(request):
            pytest.fail("Global interrupt must never be reached")

        app = web.Application()
        app.router.add_post("/api/jobs/{id}/cancel", cancel)
        app.router.add_post("/interrupt", global_interrupt)
        async with TestServer(app) as server:
            client = ComfyClient(str(server.make_url("/")))
            await client.start()
            try:
                assert await client.cancel_prompt("safe-id") == {"cancelled": True}
                with pytest.raises(StudioError, match="HTTP 404"):
                    await client.cancel_prompt("unsupported")
                with pytest.raises(StudioError, match="prompt id"):
                    await client.cancel_prompt("../escape")
            finally:
                await client.close()
        assert seen == ["safe-id", "unsupported"]
    asyncio.run(scenario())


def test_websocket_negotiates_identity_metadata_and_ignores_legacy_preview():
    async def scenario():
        async def websocket(request):
            assert request.query["clientId"] == "stable-client"
            socket = web.WebSocketResponse()
            await socket.prepare(request)
            assert (await socket.receive_json())["data"]["supports_preview_metadata"] is True
            await socket.send_json({"type": "execution_start", "data": {"prompt_id": "bound"}})
            await socket.send_bytes(struct.pack(">II", 1, 2) + b"unbound")
            metadata = json.dumps({"prompt_id": "bound", "image_type": "image/png"}).encode()
            await socket.send_bytes(struct.pack(">II", 4, len(metadata)) + metadata + b"image")
            await socket.close()
            return socket

        app = web.Application()
        app.router.add_get("/ws", websocket)
        async with TestServer(app) as server:
            client = ComfyClient(str(server.make_url("/")))
            await client.start()
            try:
                messages = [row async for row in client.stream_events("stable-client")]
            finally:
                await client.close()
        assert [row["type"] for row in messages] == ["execution_start", "preview_frame"]
        assert messages[1]["data"]["prompt_id"] == "bound"
    asyncio.run(scenario())


def test_websocket_redirect_is_never_followed():
    async def scenario():
        touched = []

        async def redirect(request):
            raise web.HTTPFound("/elsewhere")

        async def destination(request):
            touched.append(True)
            return web.Response()

        app = web.Application()
        app.router.add_get("/ws", redirect)
        app.router.add_get("/elsewhere", destination)
        async with TestServer(app) as server:
            client = ComfyClient(str(server.make_url("/")))
            await client.start()
            try:
                with pytest.raises(StudioError):
                    _ = [row async for row in client.stream_events("client")]
            finally:
                await client.close()
        assert touched == []
    asyncio.run(scenario())


def test_driver_client_sends_no_credentials():
    # No flag ever set a driver token; HTTPS with a server certificate works
    # and a proxy that checks credentials refuses the companion.
    with pytest.raises(TypeError):
        ComfyClient("https://driver.example:8188", bearer="private")
    seen = []

    async def stats(request):
        seen.append(request.headers.get("Authorization"))
        return web.json_response({"system": {}})

    async def scenario():
        async with driver(("GET", "/system_stats", stats)) as client:
            assert await client.request("GET", "/system_stats") == {"system": {}}
    asyncio.run(scenario())
    assert seen == [None]


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e400", "-1E+400", "1" + "0" * 400 + ".5",
                                   "9" * 300 + "e9"])
def test_finite_loads_replaces_every_nonfinite_source(token):
    assert finite_json.loads('{"value": [' + token + ', 1.5, "NaN"]}') == {"value": [None, 1.5, "NaN"]}
    assert finite_json.loads(('[' + token + ']').encode()) == [None]


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-32"])
def test_finite_loads_walks_wide_encodings(encoding):
    assert finite_json.loads('{"value": [NaN, Infinity, 2.5]}'.encode(encoding)) == {"value": [None, None, 2.5]}


def test_finite_loads_keeps_large_finite_numbers():
    assert finite_json.loads('[18446744073709551615, 1.7e308, 1e-400]') == [18446744073709551615, 1.7e308, 0.0]


def test_transport_replaces_nonfinite_numbers_with_null():
    # An API client may send "nan" or "inf" for a FLOAT input; ComfyUI stores
    # the converted value and serves it back in /queue and /history.
    history = (b'{"prompt-1": {"prompt": [0, "prompt-1", {"1": {"class_type": "Render", "inputs": {"cfg": NaN, '
               b'"denoise": Infinity, "shift": -Infinity, "scale": 1e400}}}, {}], "status": {"completed": true}, '
               b'"outputs": {}}}')
    queue = b'{"queue_running": [[0, "prompt-2", {"1": {"inputs": {"cfg": NaN}}}, {}]], "queue_pending": []}'

    async def websocket(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.receive_json()
        await socket.send_str('{"type": "progress", "data": {"prompt_id": "prompt-2", "value": NaN, "max": Infinity}}')
        metadata = b'{"prompt_id": "prompt-2", "node_id": NaN, "image_type": "image/png"}'
        await socket.send_bytes(struct.pack(">II", 4, len(metadata)) + metadata + b"image")
        await socket.close()
        return socket

    async def scenario():
        async with driver(("GET", "/history", raw_json(history)), ("GET", "/queue", raw_json(queue)),
                          ("GET", "/ws", websocket)) as client:
            observed = await client.request("GET", "/history")
            assert observed["prompt-1"]["prompt"][2]["1"]["inputs"] == {"cfg": None, "denoise": None, "shift": None,
                                                                       "scale": None}
            pending = await client.request("GET", "/queue")
            assert pending["queue_running"][0][2]["1"]["inputs"]["cfg"] is None
            # Canonical encoding refuses non-finite numbers; these now store cleanly.
            canonical([observed, pending])
            messages = [row async for row in client.stream_events("client")]
        assert messages[0]["data"] == {"prompt_id": "prompt-2", "value": None, "max": None}
        assert messages[1]["type"] == "preview_frame" and messages[1]["data"]["node_id"] is None
    asyncio.run(scenario())


def test_companion_transport_does_not_load_the_comfy_node_surface():
    code = ("import sys, dgx_monarch.studio.comfy, dgx_monarch.studio.live_preview, dgx_monarch.finite_json; "
            "print(sorted(name for name in sys.modules if name.startswith('dgx_monarch.nodes')))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert result.stdout.strip() == "[]"


def test_driver_refusals_carry_status_and_cleaned_detail():
    private = "/home/user/models/private-model.safetensors"
    bodies = {
        "values": {"error": {"type": "prompt_outputs_failed_validation", "message": "m", "details": private,
                             "extra_info": {}},
                   "node_errors": {"3": {"errors": [{"type": "value_not_in_list", "message": f"Value not in list: {private}",
                                                     "details": private, "extra_info": {"input_name": "ckpt_name",
                                                                                         "received_value": private}}],
                                         "dependent_outputs": ["9"], "class_type": "CheckpointLoaderSimple"}}},
        "many": {"error": {"type": "prompt_outputs_failed_validation"},
                 "node_errors": {str(index): {"errors": [{"type": "required_input_missing",
                                                          "extra_info": {"input_name": "image\n\x1b[2J"}}],
                                              "class_type": "Node"} for index in range(70)}},
        "graph": {"error": {"type": "missing_node_type", "message": "Node 'Gone' not found.", "details": "#7",
                            "extra_info": {"node_id": "7", "class_type": "Gone", "node_title": "Gone"}},
                  "node_errors": {}},
    }
    mode = {"value": "values"}

    async def prompt(request):
        if mode["value"] == "text":
            return web.Response(text="Internal Server Error", status=500)
        if mode["value"] == "huge":
            return web.json_response({"node_errors": {}, "padding": "x" * (MAX_BODY_BYTES + 1024)}, status=400)
        return web.json_response(bodies[mode["value"]], status=400)

    async def consent(request):
        return web.json_response({"ok": False, "detail": "no pending consent\x00 for that request;\n" + "y" * 500},
                                 status=409)

    async def recycle(request):
        raise web.HTTPForbidden(text="cross-origin action refused")

    async def queue(request):
        raise web.HTTPNotFound()

    async def scenario():
        async with driver(("POST", "/prompt", prompt), ("POST", "/dgxm/consent", consent),
                          ("POST", "/dgxm/recycle", recycle), ("POST", "/queue", queue)) as client:
            async def refusal(path):
                with pytest.raises(StudioError) as caught:
                    await client.request("POST", path, {})
                assert caught.value.code == "driver_rejected" and caught.value.status == 502
                return caught.value

            error = await refusal("/prompt")
            assert "HTTP 400" in str(error) and error.upstream_status == 400
            assert error.upstream_detail == [{"node_id": "3", "class_type": "CheckpointLoaderSimple",
                                              "input_name": "ckpt_name", "type": "value_not_in_list"}]
            assert "private" not in canonical(error.upstream_detail)
            mode["value"] = "many"
            rows = (await refusal("/prompt")).upstream_detail
            assert len(rows) == 64
            assert all(row["input_name"] == "image [2J" for row in rows)
            mode["value"] = "graph"
            assert (await refusal("/prompt")).upstream_detail == [
                {"node_id": "7", "class_type": "Gone", "input_name": None, "type": "missing_node_type"}]
            mode["value"] = "text"
            error = await refusal("/prompt")
            assert (error.upstream_status, error.upstream_detail) == (500, None)
            # A refusal stays a refusal at any size; only its detail is dropped.
            mode["value"] = "huge"
            error = await refusal("/prompt")
            assert (error.upstream_status, error.upstream_detail) == (400, None)
            error = await refusal("/dgxm/consent")
            assert error.upstream_status == 409 and error.upstream_detail.startswith("no pending consent for that request; y")
            assert len(error.upstream_detail) == 300 and all(ord(char) >= 32 for char in error.upstream_detail)
            error = await refusal("/dgxm/recycle")
            assert (error.upstream_status, error.upstream_detail) == (403, "cross-origin action refused")
            error = await refusal("/queue")
            assert (error.upstream_status, error.upstream_detail) == (404, None)
    asyncio.run(scenario())
    local = StudioError("local", "A companion refusal names no driver status.")
    assert (local.upstream_status, local.upstream_detail) == (None, None)


def test_unreadable_driver_bodies_are_unknown_responses():
    deep = b"[" * 100_000 + b"]" * 100_000
    digits = b'{"value": ' + b"1" * 5000 + b"}"

    async def websocket(request):
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        await socket.receive_json()
        await socket.send_str("[" * 30_000 + "]" * 30_000)
        await socket.close()
        return socket

    async def scenario():
        async with driver(("GET", "/queue", raw_json(deep)), ("GET", "/system_stats", raw_json(digits)),
                          ("GET", "/ws", websocket)) as client:
            for path in ("/queue", "/system_stats"):
                with pytest.raises(StudioError) as caught:
                    await client.request("GET", path)
                assert caught.value.code == "driver_unknown"
            with pytest.raises(StudioError) as caught:
                _ = [row async for row in client.stream_events("client")]
            assert caught.value.code == "driver_events"
    asyncio.run(scenario())


def test_object_info_has_its_own_published_budget():
    assert schema_document()["limits"]["object_info_bytes"] == OBJECT_INFO_BYTES == 16 * 1024 * 1024
    schema = json.dumps({"Render": {"description": "d" * (MAX_BODY_BYTES + 1024)}}).encode()
    oversized = json.dumps({"Render": {"description": "d" * OBJECT_INFO_BYTES}}).encode()
    bodies = {"/object_info": schema}

    async def object_info(request):
        return web.Response(body=bodies["/object_info"], content_type="application/json")

    async def scenario():
        async with driver(("GET", "/object_info", object_info), ("GET", "/queue", raw_json(schema))) as client:
            assert set(await client.request("GET", "/object_info")) == {"Render"}
            with pytest.raises(StudioError, match=r"from /queue exceeds its 4 MiB budget") as caught:
                await client.request("GET", "/queue")
            assert caught.value.code == "driver_response_limit"
            bodies["/object_info"] = oversized
            with pytest.raises(StudioError, match=r"from /object_info exceeds its 16 MiB budget") as caught:
                await client.request("GET", "/object_info")
            assert caught.value.code == "driver_response_limit"
    asyncio.run(scenario())
