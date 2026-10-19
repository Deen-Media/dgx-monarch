"""Full HTTP companion flows against an in-process Comfy protocol fixture.

No ComfyUI runtime, workers, model execution, external network or real management
subprocess is used. These are protocol tests, not hardware acceptance evidence.
"""
from __future__ import annotations

import asyncio
import base64
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch import __version__
from dgx_monarch.studio import SCHEMA_VERSION
from dgx_monarch.studio import server as server_module
from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store

# A real tiny PNG makes the media route test byte preservation, not image creation.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j8n8AAAAASUVORK5CYII="
)


class EventStore(Store):
    """Observe real durable updates without adding polling sleeps to tests."""

    def __init__(self, path):
        super().__init__(path)
        self.changed = asyncio.Event()

    def _event(self, kind, object_id, data):
        super()._event(kind, object_id, data)
        self.changed.set()

    async def wait_state(self, kind, object_id, state):
        async def observe():
            while True:
                self.changed.clear()
                document = self.get(kind, object_id)
                if document.get("state") == state:
                    return document
                await self.changed.wait()
        return await asyncio.wait_for(observe(), timeout=5)


class DriverProtocol:
    def __init__(self):
        self.submissions = []
        self.media_requests = []
        self.uploaded = {}
        self.chunked_upload = False
        self.chunked_media = False
        self.queue = {"queue_pending": [], "queue_running": []}
        self.history = {}
        self.schema = {
            "Text": {"input": {"required": {"text": ["STRING", {"default": ""}]}}, "output": ["STRING"]},
            "Sampler": {"input": {"required": {
                "text": ["STRING", {"forceInput": True}], "seed": ["INT", {"min": 0, "max": 2**64 - 1}],
                "steps": ["INT", {"min": 1, "max": 100}], "sampler": [["euler", "heun"]],
            }}, "output": ["IMAGE"]},
            "Save": {"input": {"required": {"image": ["IMAGE"]}}, "output": [], "output_node": True},
            "DGXMonarchLoraLoader": {"input": {"required": {"lora_name": [["local-style.safetensors"]]}}, "output": []},
        }

    def app(self):
        app = web.Application()

        async def object_info(request):
            return web.json_response(self.schema)

        async def queue(request):
            return web.json_response(self.queue)

        async def history(request):
            return web.json_response(self.history)

        async def telemetry(request):
            return web.json_response({"render": {"active": bool(self.queue["queue_running"])}, "workers": []})

        async def submit(request):
            body = await request.json()
            self.submissions.append(body)
            prompt_id = f"prompt-{len(self.submissions)}"
            self.queue["queue_pending"].append([1, prompt_id, body["prompt"], body["extra_data"], ["3"]])
            return web.json_response({"prompt_id": prompt_id, "number": 1, "node_errors": {}})

        async def media(request):
            self.media_requests.append(dict(request.query))
            if request.query.get("filename") in self.uploaded:
                return web.Response(body=self.uploaded[request.query["filename"]], content_type="image/png")
            if dict(request.query) != {"filename": "result.png", "subfolder": "studio", "type": "output"}:
                raise web.HTTPNotFound()
            if self.chunked_media:
                response = web.StreamResponse(headers={"Content-Type": "image/png"})
                await response.prepare(request)
                await response.write(PNG)
                await response.write_eof()
                return response
            return web.Response(body=PNG, content_type="image/png")

        async def upload(request):
            parts = await request.multipart()
            part = await parts.next()
            assert part.name == "image" and part.filename.endswith(".png")
            self.uploaded[part.filename] = bytes(await part.read())
            value = {"name": part.filename, "subfolder": "studio", "type": "input"}
            if self.chunked_upload:
                raw = json.dumps(value).encode()
                response = web.StreamResponse(headers={"Content-Type": "application/json"})
                await response.prepare(request)
                await response.write(raw[:8])
                await asyncio.sleep(0.01)
                await response.write(raw[8:])
                await response.write_eof()
                return response
            return web.json_response(value)

        app.router.add_get("/object_info", object_info)
        app.router.add_get("/queue", queue)
        app.router.add_get("/history", history)
        app.router.add_get("/dgxm/telemetry", telemetry)
        app.router.add_post("/prompt", submit)
        app.router.add_get("/view", media)
        app.router.add_post("/upload/image", upload)
        return app


@asynccontextmanager
async def companion(tmp_path, monkeypatch):
    protocol = DriverProtocol()
    async with TestServer(protocol.app()) as upstream:
        store = EventStore(tmp_path / "state")
        client = ComfyClient(str(upstream.make_url("/")).rstrip("/"))
        jobs = Jobs(store, client)
        operations = Operations(store, client)
        # Inject the same real Jobs instance for explicit HTTP reconciliation.
        # All requests and business methods still execute production code.
        monkeypatch.setattr(server_module, "Jobs", lambda supplied_store, supplied_client: jobs)
        allowed_hosts = set()
        token = "integration-owner-" + "x" * 32
        app = server_module.create_app(store, client, token, allowed_hosts=allowed_hosts,
                                       operations=operations, poll=False)
        try:
            async with TestClient(TestServer(app)) as browser:
                origin = str(browser.make_url("/")).rstrip("/")
                allowed_hosts.add(origin.split("://", 1)[1])
                headers = {"Authorization": "Bearer " + token, "X-DGXM-Schema": SCHEMA_VERSION}

                async def request(method, path, *, expected=200, **kwargs):
                    response = await browser.request(method, "/api/v1" + path, headers=headers, **kwargs)
                    body = await response.json()
                    assert response.status == expected, body
                    return body

                yield SimpleNamespace(protocol=protocol, upstream=upstream, store=store, jobs=jobs,
                                      operations=operations, browser=browser, headers=headers, request=request)
        finally:
            store.close()


def test_complete_http_generation_provenance_and_media_flow(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            denied = await system.browser.get("/api/v1/capabilities")
            assert denied.status == 401
            capabilities = await system.request("GET", "/capabilities")
            assert capabilities["schema_version"] == SCHEMA_VERSION
            assert any(row["id"] == "version" and row["available"] for row in capabilities["operations"])
            catalog = await system.request("GET", "/catalog")
            assert catalog["object_info"] == system.protocol.schema
            assert catalog["loras"][0]["name"] == "local-style.safetensors"
            assert catalog["loras"][0]["compatibility"] == "unknown"
            bundled = await system.request("GET", "/workflows")
            assert len(bundled["workflows"]) == 66

            workflow = json.loads((Path(__file__).parent / "fixtures/studio/app_mode_canonical.json").read_text())
            inspection = await system.request("POST", "/workflows/inspect", json={"workflow": workflow})
            assert inspection["errors"] == []
            assert any(row["id"] == "main:1:text" for row in inspection["bindings"])
            system.store.create("artifacts", {"id": "parent-output", "kind": "image", "job_id": None,
                                              "filename": "parent.png", "subfolder": "", "type": "output"})
            draft = {"workflow": workflow, "values": {"main:1:text": "integration prompt", "main:2:seed": 123},
                     "seed_policy": "fixed", "operation": "image", "project_id": "project-one",
                     "parent_artifact_ids": ["parent-output"]}
            preview = await system.request("POST", "/jobs/preview", json=draft)
            assert preview["prompt"]["1"]["inputs"]["text"] == "integration prompt"
            assert preview["effective"]["seeds"] == {"2.seed": 123}
            submit = {"preview_id": preview["id"], "idempotency_key": "generation-once"}
            accepted = await system.request("POST", "/jobs", expected=202, json=submit)
            job_id = accepted["id"]
            await system.store.wait_state("jobs", job_id, "queued")
            repeat = await system.request("POST", "/jobs", expected=202, json=submit)
            assert repeat["id"] == job_id
            assert len(system.protocol.submissions) == 1
            request_state = await system.request("GET", "/jobs/request/generation-once")
            assert request_state["id"] == job_id and request_state["state"] == "queued"
            conflict = await system.request("POST", "/jobs", expected=409,
                                            json={**submit, "preview_id": "different-preview"})
            assert conflict["error"]["code"] == "idempotency_conflict"
            stream = await system.request("GET", "/events")
            assert any(row["kind"] == "jobs.accepted" and row["job_id"] == job_id for row in stream["events"])

            entry = system.protocol.queue["queue_pending"].pop()
            system.protocol.queue["queue_running"].append(entry)
            await system.jobs.reconcile()
            running = await system.request("GET", f"/jobs/{job_id}")
            assert running["state"] == "running"
            system.protocol.queue["queue_running"].clear()
            system.protocol.history[entry[1]] = {
                "prompt": entry,
                "status": {"completed": True, "status_str": "success", "messages": [["execution_success", {}]]},
                "outputs": {"3": {"images": [
                    {"filename": "result.png", "subfolder": "studio", "type": "output", "width": 1, "height": 1},
                    {"filename": "../private.png", "subfolder": "", "type": "output"},
                ]}},
            }
            system.protocol.queue["queue_pending"] = [[2, "external-prompt", {}, {"studio": "unrelated extension metadata"}, []]]
            await system.jobs.reconcile()
            snapshot = await system.request("GET", "/snapshot")
            completed = next(row for row in snapshot["jobs"] if row["id"] == job_id)
            assert completed["state"] == "succeeded"
            assert completed["summary"] and "workflow" not in completed["submission"]
            completed = await system.request("GET", f"/jobs/{job_id}")
            assert completed["submission"]["requested"] == draft
            assert completed["submission"]["workflow"] == workflow
            assert completed["submission"]["parent_artifact_ids"] == ["parent-output"]
            assert any(row.get("external") and row["prompt_id"] == "external-prompt" for row in snapshot["jobs"])
            assert len(snapshot["artifacts"]) == 2
            artifact = next(row for row in snapshot["artifacts"] if row["id"] != "parent-output")
            assert artifact["job_id"] == job_id and artifact["metadata"]["width"] == 1
            media = await system.browser.get(f"/api/v1/artifacts/{artifact['id']}/file", headers=system.headers)
            assert media.status == 200 and media.content_type == "image/png"
            assert await media.read() == PNG
            assert system.protocol.media_requests == [{"filename": "result.png", "subfolder": "studio", "type": "output"}]
            assert len(system.protocol.submissions) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("chunked", [False, True])
def test_input_upload_and_bounded_thumbnail_flow(tmp_path, monkeypatch, chunked):
    import io

    from PIL import Image

    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            system.protocol.chunked_upload = chunked
            data = io.BytesIO()
            Image.new("RGBA", (640, 480), (255, 0, 0, 0)).save(data, "PNG")
            asset = await system.request("POST", "/assets", expected=201, json={
                "filename": "../../untrusted.png", "kind": "mask", "media_type": "image/png",
                "data_base64": base64.b64encode(data.getvalue()).decode(),
            })
            assert asset["width"] == 640 and asset["height"] == 480
            assert asset["type"] == "input" and asset["binding_value"].startswith("studio/")
            assert "untrusted" not in asset["filename"]
            response = await system.browser.get(f"/api/v1/artifacts/{asset['id']}/thumbnail", headers=system.headers)
            assert response.status == 200
            with Image.open(io.BytesIO(await response.read())) as preview:
                assert preview.size == (320, 240)
            assert len(system.protocol.uploaded) == 1
    asyncio.run(scenario())


def test_oversize_chunked_media_aborts_instead_of_returning_a_partial_file(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            monkeypatch.setattr(server_module, "MEDIA_LIMIT", len(PNG) - 1)
            system.protocol.chunked_media = True
            system.store.create("artifacts", {"id": "oversize", "filename": "result.png", "subfolder": "studio", "type": "output"})
            response = await system.browser.get("/api/v1/artifacts/oversize/file", headers=system.headers)
            with pytest.raises(aiohttp.ClientPayloadError):
                await response.read()
    asyncio.run(scenario())


def test_project_revision_and_offline_driver_keep_host_and_version_available(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            data = {"name": "Protocol project", "extension": {"unknown": [1, 2, 3]}}
            saved = await system.request("PUT", "/records/projects/project-one", json={"revision": 0, "data": data})
            assert saved["revision"] == 1
            conflict = await system.request("PUT", "/records/projects/project-one", expected=409,
                                            json={"revision": 0, "data": {"name": "stale tab"}})
            assert conflict["error"]["code"] == "revision_conflict"
            current = await system.request("GET", "/records/projects")
            assert current["records"][0]["data"] == data
            before = await system.browser.get("/")
            static_bytes = await before.read()
            assert before.status == 200 and b"<html" in static_bytes
            await system.jobs.reconcile()
            assert (await system.request("GET", "/snapshot"))["driver"]["state"] == "connected"
            await system.upstream.close()
            await system.jobs.reconcile()
            offline = await system.request("GET", "/snapshot")
            assert offline["driver"]["state"] == "unavailable"
            assert offline["telemetry"]["state"] == "stale"
            assert offline["records"]["projects"][0]["summary"]
            assert offline["records"]["projects"][0]["data"]["name"] == data["name"]
            assert (await system.request("GET", "/records/projects"))["records"][0]["data"] == data
            retained = await system.browser.get("/")
            assert retained.status == 200 and await retained.read() == static_bytes
            await system.request("GET", "/catalog", expected=503)
            capabilities = await system.request("GET", "/capabilities")
            assert any(row["id"] == "version" and row["available"] for row in capabilities["operations"])

            calls = []

            async def fake_subprocess(*argv, **kwargs):
                assert argv == (sys.executable, "-m", "dgx_monarch", "version")
                calls.append(argv)
                stream = asyncio.StreamReader()
                stream.feed_data(b"version output from an isolated process fixture\n")
                stream.feed_eof()

                async def wait():
                    return 0
                return SimpleNamespace(pid=12345, stdout=stream, wait=wait)

            monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_subprocess)
            review = await system.request("POST", "/operations/preview", json={"action": "version"})
            request = {"digest": review["digest"], "idempotency_key": "offline-version"}
            operation = await system.request("POST", f"/operations/{review['id']}/apply", expected=202, json=request)
            result = await system.store.wait_state("operations", operation["id"], "succeeded")
            assert result["receipt"]["version"] == __version__
            assert result["receipt"]["source"] == "existing_cli"
            repeated = await system.request("POST", f"/operations/{review['id']}/apply", expected=202, json=request)
            assert repeated["id"] == operation["id"] and len(calls) == 1
            final = await system.request("GET", "/snapshot")
            assert any(row["id"] == operation["id"] and row["state"] == "succeeded" for row in final["operations"])
    asyncio.run(scenario())
