"""A retained artifact must never resolve against another configured driver."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store


@pytest.mark.parametrize("first,second", [
    ("http://LOCALHOST:80/", "http://localhost"),
    ("https://localhost:443", "https://localhost/"),
    ("http://[::1]:80", "http://[::1]/"),
])
def test_equivalent_origins_share_persistent_binding(tmp_path, first, second):
    store = Store(tmp_path / "state")
    store.bind_driver(first)
    store.close()
    store = Store(tmp_path / "state")
    try:
        store.bind_driver(second)
        with pytest.raises(StudioError) as caught:
            store.bind_driver("http://different-driver:8188")
        assert caught.value.code == "driver_identity_mismatch"
        store.bind_driver(first)
    finally:
        store.close()


def test_restarting_with_another_driver_cannot_rebind_saved_media(tmp_path):
    async def scenario():
        accesses = []

        def upstream(name):
            driver = web.Application()

            async def view(request):
                accesses.append(name)
                assert request.query["filename"] == "ComfyUI_00001_.png"
                return web.Response(body=name.encode())

            driver.router.add_get("/view", view)
            return TestServer(driver)

        token = "x" * 32
        headers = {"Host": "studio", "Authorization": "Bearer " + token}

        async def read_media(store, origin):
            app = create_app(store, ComfyClient(origin), token, allowed_hosts={"studio"}, poll=False)
            async with TestClient(TestServer(app)) as client:
                response = await client.get("/api/v1/artifacts/old/file", headers=headers)
                assert response.status == 200
                return await response.read()

        async with upstream("driver-a") as first, upstream("driver-b") as second:
            origin_a, origin_b = str(first.make_url("")), str(second.make_url(""))
            store = Store(tmp_path / "state")
            artifact = {"id": "old", "filename": "ComfyUI_00001_.png", "type": "output",
                        "subfolder": "", "job_id": "old-job", "provenance": {"prompt_id": "old-prompt"}}
            store.create("artifacts", artifact)
            assert await read_media(store, origin_a) == b"driver-a"
            store.close()
            store = Store(tmp_path / "state")
            try:
                assert await read_media(store, origin_a) == b"driver-a"
                with pytest.raises(StudioError, match="different driver origin"):
                    create_app(store, ComfyClient(origin_b), token, allowed_hosts={"studio"}, poll=False)
                assert store.get("artifacts", "old") == artifact
                assert accesses == ["driver-a", "driver-a"]
            finally:
                store.close()
            separate = Store(tmp_path / "separate")
            try:
                separate.create("artifacts", artifact)
                assert await read_media(separate, origin_b) == b"driver-b"
            finally:
                separate.close()

    asyncio.run(scenario())
