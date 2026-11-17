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
        # Files the driver saved as job outputs, served through /view as ComfyUI does.
        self.outputs: dict[str, bytes] = {}
        # When set, /view drops the connection mid-request, as a driver going away does.
        self.drop_view = False
        # Seconds /view waits before it answers, as a busy driver does.
        self.view_delay = 0.0
        # The type of each /view request, in order.
        self.views: list[str] = []
        # When set, /view answers a held input file with this status, as a server fault or a
        # proxy in front of the driver can.
        self.input_status: int | None = None

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
                if part.filename:
                    fields[part.name], fields["bytes"] = part.filename, await part.read()
                else:
                    fields[part.name] = await part.text()
            self.uploads.append(fields)
            return web.json_response({"name": fields["image"], "subfolder": fields["subfolder"], "type": "input"})

        async def view(request):
            self.views.append(request.query.get("type", ""))
            await asyncio.sleep(self.view_delay)
            if self.drop_view:
                request.transport.close()
                return web.Response(status=500)
            if request.query.get("type") == "input":
                # The files its input store holds, as ComfyUI serves them.
                held = {(upload["subfolder"], upload["image"]): upload["bytes"] for upload in self.uploads}
                data = held.get((request.query.get("subfolder", ""), request.query.get("filename", "")))
                if data is None:
                    return web.json_response({"error": "missing"}, status=404)
                if self.input_status is not None:
                    return web.json_response({"error": "unavailable"}, status=self.input_status)
                return web.Response(body=data, content_type="image/png")
            data = self.outputs.get(request.query.get("filename", ""))
            if request.query.get("type") != "output" or data is None:
                return web.json_response({"error": "missing"}, status=404)
            return web.Response(body=data, content_type="image/png")

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
        app.router.add_get("/view", view)
        app.router.add_post("/prompt", prompt)
        app.router.add_get("/queue", queue)
        app.router.add_get("/history", empty)
        app.router.add_get("/system_stats", empty)
        return app


class CompanionServer(TestServer):
    """The companion as `dgxm-studio` serves it (web.run_app): a handler runs on when its client
    goes away, where aiohttp's test server would cancel it."""

    async def _make_runner(self, **kwargs):
        return await super()._make_runner(**{**kwargs, "handler_cancellation": False})


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
            async with TestClient(CompanionServer(app)) as browser:
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


def saved_output(system, name="out.png", data=None, **fields):
    """A job's saved image output, as the companion records one from driver history."""
    system.driver.outputs[name] = data if data is not None else base64.b64decode(png("green"))
    record = {"id": "output-" + name, "job_id": "job", "kind": "image", "filename": name, "subfolder": "",
              "type": "output", "metadata": {}, **fields}
    system.store.create("artifacts", record)
    return record["id"]


@pytest.mark.parametrize("form", ["legacy", "options"])
def test_send_to_edit_copies_an_output_into_the_input_store_and_binds_it(tmp_path, form):
    # Send to edit binds a job's output as the actual image input: the companion copies
    # it into the driver's input store as an upload of its own, which preview accepts.
    async def scenario():
        async with companion(tmp_path, form) as system:
            output = saved_output(system)
            status, copied = await system.call("POST", "/assets", json={"kind": "input", "artifact_id": output})
            assert status == 201, copied
            binding = copied["binding_value"]
            assert binding.startswith("studio/") and copied["source_artifact_id"] == output
            assert (copied["kind"], copied["job_id"], copied["width"], copied["height"]) == ("input", None, 8, 6)
            (upload,) = system.driver.uploads
            assert (upload["type"], upload["subfolder"], upload["image"]) == ("input", "studio", binding.split("/")[1])
            with Image.open(io.BytesIO(upload["bytes"])) as stored:
                assert (stored.format, stored.mode, stored.size) == ("PNG", "RGBA", (8, 6))
                assert stored.getpixel((0, 0)) == (0, 128, 0, 255)
            # The output itself still widens nothing; its copy binds through preview and submit.
            status, _ = await system.call("POST", "/jobs/preview", json=draft("output.png"))
            assert status == 400
            status, preview = await system.call("POST", "/jobs/preview", json=draft(binding))
            assert status == 200, preview
            assert preview["prompt"]["1"]["inputs"]["image"] == binding
            status, job = await system.call("POST", "/jobs", json={"preview_id": preview["id"], "idempotency_key": "edit"})
            assert status == 202, job
    asyncio.run(scenario())


def test_send_to_edit_copies_an_output_larger_than_the_browser_upload_limit(tmp_path):
    # The bytes never cross the browser, so the 2 MiB upload contract does not bind;
    # the pixel rule does.
    noise = Image.frombytes("RGB", (1100, 1100), bytes(range(256)) * (1100 * 1100 * 3 // 256) + bytes(1100 * 1100 * 3 % 256))
    data = io.BytesIO()
    noise.save(data, "PNG", compress_level=0)
    assert len(data.getvalue()) > 2 * 1024 * 1024

    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system, "large.png", data.getvalue())
            status, copied = await system.call("POST", "/assets", json={"kind": "input", "artifact_id": output})
            assert status == 201, copied
            assert (copied["width"], copied["height"]) == (1100, 1100)
    asyncio.run(scenario())


@pytest.mark.parametrize(("body", "fields", "code", "status"), [
    ({"artifact_id": "missing"}, None, "not_found", 404),
    ({"artifact_id": 7}, None, "asset_source", 400),
    ({"kind": "mask"}, {}, "asset_source", 400),
    ({}, {"kind": "video", "filename": "out.mp4"}, "asset_source", 400),
    ({}, {"job_id": None, "kind": "input"}, "asset_source", 400),
    ({}, {"source": "live_preview"}, "asset_source", 400),
    ({}, {"filename": "gone.png"}, "artifact_unavailable", 404),
])
def test_send_to_edit_refuses_anything_but_a_saved_image_output(tmp_path, body, fields, code, status):
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system, **(fields or {})) if fields is not None else None
            request = {"kind": "input", **({"artifact_id": output} if output else {}), **body}
            answer, refused = await system.call("POST", "/assets", json=request)
            assert (answer, refused["error"]["code"]) == (status, code), refused
            assert system.driver.uploads == []
    asyncio.run(scenario())


def test_send_to_edit_names_a_dropped_driver_connection(tmp_path):
    # A driver that goes away while the output is read is a copy failure the user can act on,
    # with its own code, not an internal error; no upload is made.
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            system.driver.drop_view = True
            status, refused = await system.call("POST", "/assets", json={"kind": "input", "artifact_id": output})
            assert (status, refused["error"]["code"]) == (503, "asset_copy"), refused
            assert "Send to edit again" in refused["error"]["message"]
            assert system.driver.uploads == []
    asyncio.run(scenario())


def _copies(system, output):
    return [row for row in system.store.all("artifacts") if row.get("source_artifact_id") == output]


def test_a_copy_the_browser_gave_up_on_is_bound_by_the_next_press_and_never_made_twice(tmp_path):
    # The browser abandons a companion request after 20 seconds (studio/src/api.ts), queued
    # behind thumbnails or not, and the companion finishes the handler it left. Send to edit
    # stores one copy per output, so the next press binds the copy that finished.
    import aiohttp

    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            system.driver.view_delay = 0.5
            body = {"kind": "input", "artifact_id": output}
            with pytest.raises(TimeoutError):
                await system.call("POST", "/assets", json=body, timeout=aiohttp.ClientTimeout(total=0.1))
            await asyncio.sleep(1.0)
            (stored,) = _copies(system, output)
            status, copied = await system.call("POST", "/assets", json=body)
            assert status == 201 and copied["id"] == stored["id"], copied
            assert len(system.driver.uploads) == 1 and len(_copies(system, output)) == 1
            status, preview = await system.call("POST", "/jobs/preview", json=draft(copied["binding_value"]))
            assert status == 200, preview
    asyncio.run(scenario())


def test_two_presses_for_one_output_make_one_copy_and_both_name_it(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            system.driver.view_delay = 0.3
            body = {"kind": "input", "artifact_id": output}
            answers = await asyncio.gather(*(system.call("POST", "/assets", json=body) for _ in range(2)))
            assert [status for status, _ in answers] == [201, 201], answers
            assert answers[0][1]["id"] == answers[1][1]["id"]
            assert len(system.driver.uploads) == 1 and len(_copies(system, output)) == 1
    asyncio.run(scenario())


def test_a_copy_that_failed_leaves_nothing_the_next_press_reuses(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            body = {"kind": "input", "artifact_id": output}
            system.driver.drop_view = True
            status, refused = await system.call("POST", "/assets", json=body)
            assert (status, refused["error"]["code"]) == (503, "asset_copy"), refused
            system.driver.drop_view = False
            status, copied = await system.call("POST", "/assets", json=body)
            assert status == 201 and _copies(system, output) == [copied]
            assert len(system.driver.uploads) == 1
    asyncio.run(scenario())


def test_a_stored_copy_the_driver_still_serves_costs_one_check_and_no_copy(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            body = {"kind": "input", "artifact_id": output}
            status, first = await system.call("POST", "/assets", json=body)
            assert status == 201, first
            system.driver.views.clear()
            status, again = await system.call("POST", "/assets", json=body)
            assert status == 201 and again == first
            assert system.driver.views == ["input"] and len(system.driver.uploads) == 1
    asyncio.run(scenario())


def test_a_stored_copy_the_driver_lost_is_copied_again_once_and_the_next_press_reuses_it(tmp_path):
    # The driver's input folder was cleaned, or ComfyUI was reinstalled: the copy's file is gone.
    # The new copy takes the old record's place, so no binding names the missing file.
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            body = {"kind": "input", "artifact_id": output}
            status, first = await system.call("POST", "/assets", json=body)
            assert status == 201, first
            system.driver.uploads.clear()
            status, renewed = await system.call("POST", "/assets", json=body)
            assert status == 201, renewed
            assert renewed["id"] == first["id"] and renewed["binding_value"] != first["binding_value"]
            assert len(system.driver.uploads) == 1 and _copies(system, output) == [renewed]
            system.driver.views.clear()
            status, again = await system.call("POST", "/assets", json=body)
            assert status == 201 and again == renewed
            assert system.driver.views == ["input"] and len(system.driver.uploads) == 1
            status, preview = await system.call("POST", "/jobs/preview", json=draft(renewed["binding_value"]))
            assert status == 200, preview
    asyncio.run(scenario())


def test_two_presses_after_the_driver_lost_the_copy_make_one_check_and_one_copy(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            body = {"kind": "input", "artifact_id": output}
            status, first = await system.call("POST", "/assets", json=body)
            assert status == 201, first
            system.driver.uploads.clear()
            system.driver.views.clear()
            system.driver.view_delay = 0.2
            answers = await asyncio.gather(*(system.call("POST", "/assets", json=body) for _ in range(2)))
            assert [status for status, _ in answers] == [201, 201], answers
            assert answers[0][1] == answers[1][1] and answers[0][1]["id"] == first["id"]
            assert system.driver.views == ["input", "output"] and len(system.driver.uploads) == 1
    asyncio.run(scenario())


def test_a_stored_copy_is_not_used_when_the_driver_cannot_be_asked(tmp_path):
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            body = {"kind": "input", "artifact_id": output}
            status, first = await system.call("POST", "/assets", json=body)
            assert status == 201, first
            system.driver.drop_view = True
            status, refused = await system.call("POST", "/assets", json=body)
            assert (status, refused["error"]["code"]) == (503, "asset_copy"), refused
            assert "Send to edit again" in refused["error"]["message"]
    asyncio.run(scenario())


@pytest.mark.parametrize("answer", [500, 403])
def test_only_a_404_means_the_driver_lost_the_copy(tmp_path, answer):
    # Stock ComfyUI's /view answers 404 for a missing input file. Any other answer to the check,
    # with the file still held, means the driver could not be asked: the press fails, binds
    # nothing and makes no second copy.
    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            body = {"kind": "input", "artifact_id": output}
            status, first = await system.call("POST", "/assets", json=body)
            assert status == 201, first
            system.driver.input_status = answer
            status, refused = await system.call("POST", "/assets", json=body)
            assert (status, refused["error"]["code"]) == (503, "asset_copy"), refused
            assert "could not be asked" in refused["error"]["message"]
            assert len(system.driver.uploads) == 1 and _copies(system, output) == [first]
    asyncio.run(scenario())


@pytest.mark.parametrize("held_in", ["driver read", "image conversion",
                                     "image conversion, let go once the session closed"])
def test_a_copy_alive_at_shutdown_sends_nothing_after_the_driver_session_closes_and_writes_nothing(
        tmp_path, monkeypatch, caplog, held_in):
    """A Send to edit copy outlives its request and is still running when the companion shuts
    down as dgxm-studio does: web.run_app's runner.cleanup() (site stop, the graceful wait for
    requests, then the app's cleanup, which stops the copies still running and then closes the
    driver session), then run_app's cancel of every task still alive, and only then the store's
    close. After the session closes nothing reaches the driver, nothing writes to the store
    during or after shutdown, the copy ends cancelled, and nothing is logged as an error, even
    when the copy is let go between the session closing and run_app's last cancel."""
    import logging
    import socket
    import threading

    import aiohttp

    from dgx_monarch.studio import media_assets

    caplog.set_level(logging.ERROR)
    events: list[str] = []  # every one appended on the event loop, so its order is the events'
    driver = Driver()
    # The copy is held by events, not by time: until the test lets it go, the driver does not
    # answer the output read, or the image conversion does not end.
    converting, let_go = threading.Event(), threading.Event()
    convert = media_assets.normalized

    def held_conversion(raw):
        converting.set()
        let_go.wait(30)
        return convert(raw)

    if held_in.startswith("image conversion"):
        monkeypatch.setattr(media_assets, "normalized", held_conversion)
    assets: list = []
    make_assets = media_assets.MediaAssets.__init__

    def recorded(self, *args, **kwargs):
        make_assets(self, *args, **kwargs)
        assets.append(self)
    monkeypatch.setattr(media_assets.MediaAssets, "__init__", recorded)
    copies: list[asyncio.Future] = []

    @web.middleware
    async def logged(request, handler):
        events.append(f"driver {request.method} {request.path}")
        if held_in == "driver read" and request.path == "/view":
            await asyncio.get_running_loop().run_in_executor(None, let_go.wait, 30)
        return await handler(request)

    async def until(condition):
        for _ in range(3000):
            if condition():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the copy never reached the point the test holds it at")

    async def scenario():
        upstream_app = driver.app()
        upstream_app.middlewares.append(logged)
        async with TestServer(upstream_app) as upstream:
            store = Store(tmp_path / "state")
            client = ComfyClient(str(upstream.make_url("/")).rstrip("/"))
            output = saved_output(SimpleNamespace(driver=driver, store=store))
            close = client.close

            async def logged_close():
                events.append("client.close")
                await close()
                if held_in.endswith("closed"):
                    # A copy that resumes now, before run_app's last cancel, meets a closed session.
                    let_go.set()
                    await until(copies[0].done)
                    for _ in range(20):  # let a request handler waiting on the copy answer
                        await asyncio.sleep(0)
            client.close = logged_close
            for name in ("create", "update"):
                def logged_write(*args, _write=getattr(store, name), _name=name, **kwargs):
                    events.append(f"store.{_name}")
                    return _write(*args, **kwargs)
                monkeypatch.setattr(store, name, logged_write)
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            host = f"127.0.0.1:{listener.getsockname()[1]}"
            token = "shutdown-" + "x" * 32
            app = create_app(store, client, token, allowed_hosts={host}, poll=False)
            # web.run_app's own runner: handlers run on when their client leaves; a short graceful wait.
            runner = web.AppRunner(app, handler_cancellation=False, shutdown_timeout=0.3)
            await runner.setup()
            await web.SockSite(runner, listener).start()
            headers = {"Authorization": "Bearer " + token, "X-DGXM-Schema": SCHEMA_VERSION}
            async with aiohttp.ClientSession() as browser:
                with pytest.raises(TimeoutError):
                    await browser.post(f"http://{host}/api/v1/assets", headers=headers,
                                       json={"kind": "input", "artifact_id": output},
                                       timeout=aiohttp.ClientTimeout(total=0.2))
            await until(converting.is_set if held_in.startswith("image conversion")
                        else lambda: "driver GET /view" in events)
            copies.extend(assets[0].copying.values())
            events.append("shutdown")
            try:
                await runner.cleanup()
                alive = [task for task in asyncio.all_tasks()
                         if task is not asyncio.current_task() and not task.done()]
                for task in alive:
                    task.cancel()
                await asyncio.gather(*alive, return_exceptions=True)
                store.close()
                events.append("store.close")
            finally:
                let_go.set()

    asyncio.run(scenario())
    shutdown, closed = events.index("shutdown"), events.index("client.close")
    # The copy had started before shutdown and had not finished: it read the output, uploaded nothing.
    assert "driver GET /view" in events[:shutdown], events
    assert "driver POST /upload/image" not in events, events
    assert not [event for event in events[closed:] if event.startswith("driver ")], events
    assert not [event for event in events[shutdown:] if event.startswith("store.") and event != "store.close"], events
    (copy,) = copies
    assert copy.cancelled()
    assert not [record.getMessage() for record in caplog.records if record.levelno >= logging.ERROR]


def _lookup_plan(store: Store) -> str:
    """The query plan of the lookup Store.upload_copy runs, read from the statement it executes."""
    statements: list[str] = []
    store.db.set_trace_callback(statements.append)
    try:
        assert store.upload_copy("output-x") is None
    finally:
        store.db.set_trace_callback(None)
    (lookup,) = [statement for statement in statements if "source_artifact_id" in statement]
    return " ".join(str(row["detail"]) for row in store.db.execute("EXPLAIN QUERY PLAN " + lookup))


def test_the_stored_copy_lookup_reads_an_index_in_a_new_store_and_in_an_older_one(tmp_path):
    # Every Send to edit press looks up its output's copy; without an index that scans every
    # stored document under the store lock.
    store = Store(tmp_path / "fresh")
    try:
        assert "artifacts_by_source" in _lookup_plan(store)
    finally:
        store.close()
    older = Store(tmp_path / "older")
    older.db.execute("DROP INDEX IF EXISTS artifacts_by_source")
    older.close()
    reopened = Store(tmp_path / "older")
    try:
        assert "artifacts_by_source" in _lookup_plan(reopened)
    finally:
        reopened.close()


def test_send_to_edit_keeps_the_pixel_rule(tmp_path, monkeypatch):
    from dgx_monarch.studio import media_assets
    monkeypatch.setattr(media_assets, "IMAGE_PIXELS", 10)

    async def scenario():
        async with companion(tmp_path) as system:
            output = saved_output(system)
            status, refused = await system.call("POST", "/assets", json={"kind": "input", "artifact_id": output})
            assert (status, refused["error"]["code"]) == (413, "image_limit")
            assert "Downscale the image to at most 16 megapixels" in refused["error"]["message"]
            assert system.driver.uploads == []
    asyncio.run(scenario())
