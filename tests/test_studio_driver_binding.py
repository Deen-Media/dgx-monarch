"""A retained artifact must never resolve against another configured driver."""

import asyncio
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch.config_schema import canonical_driver_origin
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


# Each spelling and the row Store.bind_driver stored for it before the rule
# moved to config_schema, so the move is proved to change nothing.
SPELLINGS = [
    ("http://127.0.0.1:8191", "http://127.0.0.1:8191"), ("http://127.0.0.1:8191/", "http://127.0.0.1:8191"),
    ("HTTP://127.0.0.1:8191", "http://127.0.0.1:8191"), ("http://LOCALHOST:80/", "http://localhost"),
    ("https://localhost:443", "https://localhost"), ("https://localhost/", "https://localhost"),
    ("https://Driver.Example:8443/", "https://driver.example:8443"), ("http://[::1]:80", "http://[::1]"),
    ("http://[::1]/", "http://[::1]"), ("http://[::1]:8188", "http://[::1]:8188"),
    ("http://[0:0::1]:8188", "http://[0:0::1]:8188"), ("http://127.0.0.1", "http://127.0.0.1"),
    ("http://localhost", "http://localhost"), ("http://driver.example:08188", "http://driver.example:8188"),
]


def _stored(store):
    with store.lock:
        row = store.db.execute("SELECT value FROM meta WHERE key='driver_origin'").fetchone()
    return None if row is None else row[0]


@pytest.mark.parametrize(("spelling", "stored"), SPELLINGS)
def test_config_canonical_rule_is_the_store_rule(tmp_path, spelling, stored):
    store = Store(tmp_path / "state")
    try:
        store.bind_driver(spelling)
        assert _stored(store) == canonical_driver_origin(spelling) == stored
    finally:
        store.close()


@pytest.mark.parametrize("earlier", ["http://127.0.0.1:8191/", "HTTP://127.0.0.1:8191", "http://127.0.0.1:8191"])
def test_a_config_origin_binds_the_row_any_equivalent_driver_spelling_bound(tmp_path, earlier):
    config_origin = "http://127.0.0.1:8191"
    store = Store(tmp_path / "state")
    store.bind_driver(earlier)
    store.close()
    store = Store(tmp_path / "state")
    try:
        store.bind_driver(config_origin)
        assert store.bound_driver() == config_origin
        with pytest.raises(StudioError) as caught:
            store.bind_driver("http://127.0.0.1:8188")
        assert caught.value.code == "driver_identity_mismatch"
        assert store.bound_driver() == config_origin
    finally:
        store.close()


def test_bound_driver_reads_the_stored_origin_under_the_store_lock(tmp_path):
    store = Store(tmp_path / "state")
    try:
        assert store.bound_driver() is None
        store.bind_driver("http://LOCALHOST:8188/")
        assert store.bound_driver() == "http://localhost:8188"
        held, release, done = threading.Event(), threading.Event(), threading.Event()
        seen = []

        def hold():
            with store.lock:
                held.set()
                release.wait(5)

        def read():
            seen.append(store.bound_driver())
            done.set()

        holder = threading.Thread(target=hold)
        holder.start()
        assert held.wait(5)
        reader = threading.Thread(target=read)
        reader.start()
        # The read waits for the lock another thread holds.
        assert not done.wait(0.2)
        release.set()
        assert done.wait(5)
        holder.join(5)
        reader.join(5)
        assert seen == ["http://localhost:8188"]
    finally:
        store.close()
