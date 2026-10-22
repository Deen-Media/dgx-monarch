"""A stored Studio upload binds to an image upload widget; nothing else widens a choice list."""
from __future__ import annotations

import asyncio
import base64
import io
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from PIL import Image

from dgx_monarch.studio import SCHEMA_VERSION
from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.submission_guard import schema_identity, upload_view


def png(color="red"):
    image = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(image, format="PNG")
    return base64.b64encode(image.getvalue()).decode()


class Driver:
    """LoadImage lists only top-level input files, as ComfyUI's nodes.py does."""

    def __init__(self, form="legacy"):
        self.inputs = ["example.png"]
        self.loras = ["first.safetensors"]
        self.form = form
        self.posts = []
        self.uploads = []

    def object_info(self):
        files = sorted(self.inputs)
        image = [files, {"image_upload": True}] if self.form == "legacy" else ["COMBO", {"options": files, "image_upload": True}]
        return {
            "LoadImage": {"input": {"required": {"image": image}}, "input_order": {"required": ["image"]},
                          "output": ["IMAGE", "MASK"], "output_node": False},
            "LoadImageOutput": {"input": {"required": {"image": ["COMBO", {"options": ["old.png"], "image_upload": True,
                                                                          "image_folder": "output"}]}},
                                "output": ["IMAGE", "MASK"], "output_node": False},
            "LoraLoaderModelOnly": {"input": {"required": {"lora_name": [sorted(self.loras)]}}, "output": ["MODEL"]},
            "Preview": {"input": {"required": {"images": ["IMAGE"]}}, "output": [], "output_node": True},
        }

    def app(self):
        app = web.Application()

        async def object_info(request):
            return web.json_response(self.object_info())

        async def upload(request):
            parts = await request.multipart()
            fields = {}
            async for part in parts:
                fields[part.name] = part.filename if part.filename else await part.text()
            self.uploads.append(fields)
            return web.json_response({"name": fields["image"], "subfolder": fields["subfolder"], "type": "input"})

        async def prompt(request):
            body = await request.json()
            self.posts.append(body)
            return web.json_response({"prompt_id": body["prompt_id"], "number": 1, "node_errors": {}})

        async def empty(request):
            return web.json_response({})

        async def queue(request):
            return web.json_response({"queue_running": [], "queue_pending": []})

        app.router.add_get("/object_info", object_info)
        app.router.add_post("/upload/image", upload)
        app.router.add_post("/prompt", prompt)
        app.router.add_get("/queue", queue)
        app.router.add_get("/history", empty)
        app.router.add_get("/system_stats", empty)
        return app


@asynccontextmanager
async def companion(tmp_path, form="legacy"):
    driver = Driver(form)
    async with TestServer(driver.app()) as upstream:
        store = Store(tmp_path / "state")
        client = ComfyClient(str(upstream.make_url("/")).rstrip("/"))
        token = "upload-binding-" + "x" * 32
        hosts: set[str] = set()
        app = create_app(store, client, token, allowed_hosts=hosts, poll=False)
        try:
            async with TestClient(TestServer(app)) as browser:
                hosts.add(str(browser.make_url("/")).split("://", 1)[1].rstrip("/"))
                headers = {"Authorization": "Bearer " + token, "X-DGXM-Schema": SCHEMA_VERSION}

                async def call(method, path, **kwargs):
                    response = await browser.request(method, "/api/v1" + path, headers=headers, **kwargs)
                    return response.status, await response.json()
                yield SimpleNamespace(driver=driver, store=store, call=call)
        finally:
            store.close()


def draft(value, node="LoadImage"):
    return {"workflow": {"1": {"class_type": node, "inputs": {"image": "example.png" if node == "LoadImage" else "old.png"}},
                         "2": {"class_type": "Preview", "inputs": {"images": ["1", 0]}}},
            "values": {"root:1:image": value}, "seed_policy": "fixed", "operation": "edit"}


@pytest.mark.parametrize("form", ["legacy", "options"])
def test_upload_bind_preview_and_submit_through_the_companion(tmp_path, form):
    async def scenario():
        async with companion(tmp_path, form) as system:
            status, uploaded = await system.call("POST", "/assets", json={
                "kind": "input", "media_type": "image/png", "data_base64": png()})
            assert status == 201 and uploaded["binding_value"].startswith("studio/")
            assert system.driver.uploads[0]["subfolder"] == "studio"
            binding = uploaded["binding_value"]
            status, preview = await system.call("POST", "/jobs/preview", json=draft(binding))
            assert status == 200, preview
            assert preview["prompt"]["1"]["inputs"]["image"] == binding
            # The driver's own schema, not the widened view, names the review.
            assert preview["schema_digest"] != "" and binding not in str(system.driver.object_info())
            status, job = await system.call("POST", "/jobs", json={"preview_id": preview["id"], "idempotency_key": "submit"})
            assert status == 202, job
            for _ in range(50):
                if system.driver.posts:
                    break
                await asyncio.sleep(0.02)
            assert system.driver.posts[0]["prompt"]["1"]["inputs"]["image"] == binding
            assert system.driver.posts[0]["prompt_id"] == job["client_prompt_id"]
    asyncio.run(scenario())


@pytest.mark.parametrize("value", ["studio/never-uploaded.png", "../studio/escape.png", "/etc/passwd", "studio\\x.png",
                                   "studio/\x01.png", "example.png [input]"])
@pytest.mark.parametrize("stored", [False, True])
def test_values_that_match_no_safe_stored_upload_are_still_refused(tmp_path, value, stored):
    async def scenario():
        async with companion(tmp_path) as system:
            if stored and value not in {"studio/never-uploaded.png", "example.png [input]"}:
                # Even a record carrying an unsafe binding (never produced by upload) cannot widen a choice.
                system.store.create("artifacts", {"id": "unsafe", "kind": "input", "job_id": None, "binding_value": value})
            status, body = await system.call("POST", "/jobs/preview", json=draft(value))
            assert status == 400 and "available choice" in body["error"]["message"]
    asyncio.run(scenario())


def test_an_upload_never_widens_a_widget_that_reads_another_store(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            _, uploaded = await system.call("POST", "/assets", json={
                "kind": "input", "media_type": "image/png", "data_base64": png()})
            status, body = await system.call("POST", "/jobs/preview", json=draft(uploaded["binding_value"], "LoadImageOutput"))
            assert status == 400 and "available choice" in body["error"]["message"]
    asyncio.run(scenario())


def test_job_output_artifacts_never_widen_upload_choices(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            system.store.create("artifacts", {"id": "output", "kind": "input", "job_id": "some-job",
                                              "binding_value": "studio/output.png"})
            status, _ = await system.call("POST", "/jobs/preview", json=draft("studio/output.png"))
            assert status == 400
    asyncio.run(scenario())


def test_new_upload_or_lora_file_does_not_void_a_pending_review(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            status, preview = await system.call("POST", "/jobs/preview", json=draft("example.png"))
            assert status == 200
            # Another client adds a LoRA file, and this companion uploads an input.
            system.driver.loras.append("second.safetensors")
            status, _ = await system.call("POST", "/assets", json={
                "kind": "mask", "media_type": "image/png", "data_base64": png()})
            system.driver.inputs.append("other.png")
            assert status == 201
            status, job = await system.call("POST", "/jobs", json={"preview_id": preview["id"], "idempotency_key": "after"})
            assert status == 202, job
    asyncio.run(scenario())


def test_removed_choice_and_changed_node_definition_are_refused_with_their_own_codes(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            _, preview = await system.call("POST", "/jobs/preview", json=draft("example.png"))
            system.driver.inputs.remove("example.png")
            status, body = await system.call("POST", "/jobs", json={"preview_id": preview["id"], "idempotency_key": "gone"})
            assert status == 409 and body["error"]["code"] == "choice_unavailable"
            assert "1.image" in body["error"]["message"]
            system.driver.inputs.append("example.png")
            original = system.driver.object_info

            def changed():
                schema = original()
                schema["LoadImage"]["output"] = ["IMAGE"]
                return schema
            system.driver.object_info = changed
            status, body = await system.call("POST", "/jobs", json={"preview_id": preview["id"], "idempotency_key": "changed"})
            assert status == 409 and body["error"]["code"] == "schema_changed"
    asyncio.run(scenario())


def test_schema_identity_covers_only_used_classes_without_choice_lists():
    driver = Driver()
    prompt = {"1": {"class_type": "LoadImage", "inputs": {"image": "example.png"}}}
    before = schema_identity(prompt, driver.object_info())
    driver.inputs.append("new.png")
    driver.loras.append("unused.safetensors")
    assert schema_identity(prompt, driver.object_info()) == before
    changed = driver.object_info()
    changed["LoadImage"]["input"]["required"]["image"][1]["tooltip"] = "changed"
    assert schema_identity(prompt, changed) != before
    assert schema_identity(prompt, {}) != before


def test_upload_view_leaves_the_driver_schema_untouched():
    schema = Driver().object_info()
    view = upload_view(schema, frozenset({"studio/a.png"}))
    assert view["LoadImage"]["input"]["required"]["image"][0] == ["example.png", "studio/a.png"]
    assert schema["LoadImage"]["input"]["required"]["image"][0] == ["example.png"]
    assert view["LoraLoaderModelOnly"] is schema["LoraLoaderModelOnly"]
    assert view["LoadImageOutput"] is schema["LoadImageOutput"]
    assert upload_view(schema, frozenset()) is schema


@pytest.mark.parametrize("form", ["legacy", "options"])
def test_inspect_accepts_a_workflow_that_embeds_a_stored_upload(tmp_path, form):
    # Reusing a job's workflow inspects the compiled prompt, which names the
    # upload; inspect must see the same choices preview does.
    async def scenario():
        async with companion(tmp_path, form) as system:
            status, uploaded = await system.call("POST", "/assets", json={
                "kind": "input", "media_type": "image/png", "data_base64": png()})
            assert status == 201
            workflow = draft(uploaded["binding_value"])["workflow"]
            workflow["1"]["inputs"]["image"] = uploaded["binding_value"]
            status, inspected = await system.call("POST", "/workflows/inspect", json={"workflow": workflow})
            assert status == 200 and inspected["errors"] == []
            assert inspected["prompt"]["1"]["inputs"]["image"] == uploaded["binding_value"]
            workflow["1"]["inputs"]["image"] = "studio/never-uploaded.png"
            status, inspected = await system.call("POST", "/workflows/inspect", json={"workflow": workflow})
            assert status == 200 and inspected["errors"] == ["1.image: Value is not an available choice"]
    asyncio.run(scenario())


@pytest.mark.parametrize("form", ["legacy", "options"])
def test_inspect_with_draft_values_keeps_an_attached_upload_a_valid_choice(tmp_path, form):
    # A reload inspects the saved workflow with the draft's values. An upload the
    # draft attached must stay among the field's choices, or the field shows an error.
    async def scenario():
        async with companion(tmp_path, form) as system:
            status, uploaded = await system.call("POST", "/assets", json={
                "kind": "input", "media_type": "image/png", "data_base64": png()})
            assert status == 201
            body = draft(uploaded["binding_value"])
            status, inspected = await system.call("POST", "/workflows/inspect", json={
                "workflow": body["workflow"], "values": body["values"]})
            assert status == 200 and inspected["errors"] == []
            image = next(row for row in inspected["bindings"] if row["id"] == "root:1:image")
            assert uploaded["binding_value"] in image["schema"]["choices"]
            # The saved graph still names its own value.
            assert image["value"] == "example.png"
            status, refused = await system.call("POST", "/workflows/inspect", json={
                "workflow": body["workflow"], "values": ["root:1:image"]})
            assert status == 400 and refused["error"]["code"] == "invalid_object"
    asyncio.run(scenario())


@pytest.mark.parametrize("form", ["legacy", "options"])
def test_a_grid_axis_over_stored_uploads_previews_and_submits(tmp_path, form):
    # The axis schemas come from the inspection, so it must list the stored
    # uploads the axes name, as each cell's preview does.
    async def scenario():
        async with companion(tmp_path, form) as system:
            values = []
            for color in ("red", "blue"):
                status, uploaded = await system.call("POST", "/assets", json={
                    "kind": "input", "media_type": "image/png", "data_base64": png(color)})
                assert status == 201
                values.append(uploaded["binding_value"])
            body = {"draft": draft(values[0]), "axes": [{"name": "root:1:image", "values": values}]}
            status, grid = await system.call("POST", "/grids/preview", json=body)
            assert status == 200, grid
            assert grid["ready"] and [cell["errors"] for cell in grid["cells"]] == [[], []]
            # The review keeps the driver's own schema digest, not the widened view's.
            assert all(value not in str(system.driver.object_info()) for value in values)
            status, run = await system.call("POST", "/grids", json={"preview_id": grid["id"], "idempotency_key": "upload-grid"})
            assert status == 202, run
            for _ in range(100):
                if len(system.driver.posts) == 2:
                    break
                await asyncio.sleep(0.02)
            assert sorted(post["prompt"]["1"]["inputs"]["image"] for post in system.driver.posts) == sorted(values)
            # A value that names no stored upload still fails its own cell.
            body["axes"][0]["values"] = [values[0], "studio/never-uploaded.png"]
            status, grid = await system.call("POST", "/grids/preview", json=body)
            assert status == 200 and not grid["ready"]
            assert grid["cells"][0]["errors"] == [] and grid["cells"][1]["errors"] == [
                "root:1:image: Value is not an available choice"]
    asyncio.run(scenario())
