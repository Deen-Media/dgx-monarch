"""HTTP and event protocol tests use a loopback mock server, never a real driver."""
from __future__ import annotations

import asyncio
import json
import struct

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.contracts import StudioError


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


def test_websocket_redirect_cannot_forward_credentials():
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
            client = ComfyClient(str(server.make_url("/")), bearer="private")
            await client.start()
            try:
                with pytest.raises(StudioError):
                    _ = [row async for row in client.stream_events("client")]
            finally:
                await client.close()
        assert touched == []
    asyncio.run(scenario())
