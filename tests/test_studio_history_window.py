"""The /history window shrinks for large rows and grows back once they age out."""
from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestServer

from dgx_monarch.studio.comfy import HISTORY_REGROW_BYTES, HISTORY_WINDOW, ComfyClient
from dgx_monarch.studio.contracts import MAX_BODY_BYTES


def entry(index: int, size: int) -> tuple[str, dict]:
    identity = f"prompt-{index}"
    return identity, {"prompt": [index, identity, {"1": {"class_type": "Text", "inputs": {"text": "x" * size}}}, {}],
                      "status": {"completed": True}, "outputs": {}}


def test_history_window_shrinks_then_regrows_step_by_step():
    rows: list[tuple[str, dict]] = []
    limits: list[int] = []

    async def history(request):
        limit = int(request.query["max_items"])
        limits.append(limit)
        return web.json_response(dict(rows[-limit:]))

    def fill(old_size: int, new_size: int) -> None:
        # The newest 100 rows use new_size; the 100 before them use old_size.
        rows[:] = [entry(index, old_size if index < 100 else new_size) for index in range(200)]

    async def scenario():
        app = web.Application()
        app.router.add_get("/history", history)
        async with TestServer(app) as upstream:
            client = ComfyClient(str(upstream.make_url("/")))
            await client.start()
            try:
                async def poll() -> tuple[list[int], int]:
                    limits.clear()
                    await client.request("GET", "/history")
                    return list(limits), client.history_limit

                # Every row is large: the full window overflows and half of
                # it is still more than half the cap, so it stays shrunk.
                fill(30_000, 30_000)
                assert 100 * 30_000 > HISTORY_REGROW_BYTES and 200 * 30_000 > MAX_BODY_BYTES
                assert await poll() == ([200, 100], 100)
                assert await poll() == ([100], 100)
                # The large rows age out: three small pages in a row double it.
                fill(1_000, 1_000)
                assert [await poll() for _ in range(3)] == [([100], 100), ([100], 100), ([100], 200)]
                assert await poll() == ([200], 200)
                # Older rows are large again. The full window overflows once,
                # halves, and needs three small pages before it tries again.
                fill(45_000, 1_000)
                assert await poll() == ([200, 100], 100)
                assert [await poll() for _ in range(2)] == [([100], 100), ([100], 200)]
                # A doubled window that overflows halves within the same poll,
                # and the next attempt waits for twice as many small pages
                # (the page that follows the overflow counts). No poll loops.
                assert await poll() == ([200, 100], 100)
                assert [await poll() for _ in range(5)] == [([100], 100)] * 4 + [([100], 200)]
                assert await poll() == ([200, 100], 100)
                assert [await poll() for _ in range(11)] == [([100], 100)] * 10 + [([100], 200)]
                # Once the large rows are gone the full window holds again.
                fill(1_000, 1_000)
                assert await poll() == ([200], HISTORY_WINDOW)
                assert await poll() == ([200], HISTORY_WINDOW)
            finally:
                await client.close()
    asyncio.run(scenario())


def test_history_window_shrinks_to_one_row_and_refuses_below_it():
    async def history(request):
        limit = int(request.query["max_items"])
        return web.json_response({"only": {"padding": "x" * (MAX_BODY_BYTES + limit)}})

    async def scenario():
        app = web.Application()
        app.router.add_get("/history", history)
        async with TestServer(app) as upstream:
            client = ComfyClient(str(upstream.make_url("/")))
            await client.start()
            try:
                try:
                    await client.request("GET", "/history")
                except ValueError as exc:
                    assert getattr(exc, "code", None) == "driver_response_limit"
                else:
                    raise AssertionError("An oversized single row must be refused")
                assert client.history_limit == 1
            finally:
                await client.close()
    asyncio.run(scenario())
