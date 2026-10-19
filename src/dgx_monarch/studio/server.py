"""Authenticated standalone host. No import or startup of ComfyUI is needed."""
from __future__ import annotations

import asyncio
import hmac
import json
import mimetypes
from contextlib import suppress
from importlib.resources import files
from typing import Any

from aiohttp import web

from . import SCHEMA_VERSION
from .comfy import ComfyClient
from .contracts import MAX_BODY_BYTES, RECORD_KINDS, StudioError, identifier, object_value, schema_document, timestamp
from .experiments import GridExperiments
from .history import page_artifacts, page_jobs
from .jobs import Jobs
from .library import Library
from .media_assets import MediaAssets
from .operations import Operations
from .runtime_history import page_runtime
from .store import Store
from .workflows import WorkflowError, inspect_workflow

MEDIA_LIMIT = 64 * 1024 * 1024


def create_app(store: Store, client: ComfyClient, token: str, *, allowed_hosts: set[str],
               operations: Operations | None = None, poll: bool = True) -> web.Application:
    if len(token) < 32:
        raise ValueError("Authentication token must contain at least 32 characters")
    jobs = Jobs(store, client)
    grids = GridExperiments(jobs)
    library = Library(store)
    media_assets = MediaAssets(store, client)
    ops = operations or Operations(store, client)
    instance = identifier()
    resources = files("dgx_monarch.studio")
    # Keep this version's assets available if a reviewed update replaces files on disk.
    static = resources.joinpath("static")
    assets: dict[str, bytes] = {}
    if static.is_dir():
        def collect(directory: Any, prefix: str = "") -> None:
            for entry in directory.iterdir():
                name = prefix + entry.name
                if entry.is_dir():
                    collect(entry, name + "/")
                elif entry.is_file() and not name.endswith(".map"):
                    assets[name] = entry.read_bytes()
        collect(static)

    @web.middleware
    async def boundary(request: web.Request, handler: Any) -> web.StreamResponse:
        try:
            if request.host not in allowed_hosts:
                raise StudioError("host", "Unrecognized companion host.", 403)
            origin = request.headers.get("Origin")
            if origin is not None and origin != f"{request.scheme}://{request.host}":
                raise StudioError("origin", "Cross-origin access is forbidden.", 403)
            if request.path.startswith("/api/"):
                authorization = request.headers.get("Authorization", "")
                if not hmac.compare_digest(authorization, f"Bearer {token}"):
                    raise StudioError("authentication", "Enter this companion's owner token.", 401)
                version = request.headers.get("X-DGXM-Schema", SCHEMA_VERSION)
                if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
                    raise StudioError("schema_version", "Client schema is incompatible; reload this companion's bundled client.", 409)
                if request.method in {"POST", "PUT", "DELETE"} and request.content_type != "application/json":
                    raise StudioError("content_type", "Mutations require application/json.", 415)
            response = await handler(request)
        except StudioError as exc:
            response = web.json_response({"error": {"code": exc.code, "message": str(exc)}}, status=exc.status)
        except WorkflowError as exc:
            response = web.json_response({"error": {"code": "workflow", "message": str(exc)}}, status=400)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response = web.json_response({"error": {"code": "invalid_json", "message": "Invalid JSON request."}}, status=400)
        response.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                                 "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
                                 "Cache-Control": "no-store"})
        return response

    app = web.Application(middlewares=[boundary], client_max_size=MAX_BODY_BYTES)

    async def payload(request: web.Request) -> dict[str, Any]:
        return object_value(await request.json())

    async def capabilities(request: web.Request) -> web.Response:
        from .presentation import presentation_contract
        return web.json_response({**schema_document(), "instance_id": instance, "operations": ops.capabilities(),
                                  "presentation": presentation_contract(), "driver_origin": client.origin,
                                  "support_status": "draft; see docs/STUDIO_ACCEPTANCE.md for unimplemented acceptance criteria"})

    async def snapshot(request: web.Request) -> web.Response:
        return web.json_response({"schema_version": SCHEMA_VERSION, "instance_id": instance,
                                  "server_time": timestamp(), "cursor": store.events(0)["latest_cursor"],
                                  "jobs": store.list("jobs"), "artifacts": store.list("artifacts"),
                                  "operations": store.list("operations"), "records": {kind: store.list(kind) for kind in RECORD_KINDS},
                                  "telemetry": jobs.telemetry, "driver": jobs.driver})

    async def events(request: web.Request) -> web.Response:
        try:
            cursor = int(request.query.get("after", "0"))
        except ValueError as exc:
            raise StudioError("cursor", "Cursor must be an integer.") from exc
        if cursor < 0:
            raise StudioError("cursor", "Cursor must be nonnegative.")
        return web.json_response(store.events(cursor))

    async def catalog(request: web.Request) -> web.Response:
        return web.json_response(await jobs.catalog())

    async def workflows(request: web.Request) -> web.Response:
        return web.Response(body=resources.joinpath("data/workflows.json").read_bytes(), content_type="application/json")

    async def inspect(request: web.Request) -> web.Response:
        body = await payload(request)
        await jobs.catalog()
        return web.json_response(inspect_workflow(object_value(body.get("workflow")), jobs.schema))

    async def preview(request: web.Request) -> web.Response:
        return web.json_response(await jobs.preview(await payload(request)))

    async def submit(request: web.Request) -> web.Response:
        return web.json_response(jobs.submit(await payload(request)), status=202)

    async def job_action(request: web.Request) -> web.Response:
        body = await payload(request)
        if request.match_info["action"] == "cancel":
            value = await jobs.cancel(request.match_info["id"], body)
        elif request.match_info["action"] == "retry":
            value = jobs.retry(request.match_info["id"], body)
        else:
            raise StudioError("job_action", "Unknown job action.", 404)
        return web.json_response(value, status=202)

    async def job(request: web.Request) -> web.Response:
        return web.json_response(store.get("jobs", request.match_info["id"]))

    async def job_preview(request: web.Request) -> web.Response:
        return web.json_response(store.get("previews", request.match_info["id"]))

    async def runtime_history(request: web.Request) -> web.Response:
        return web.json_response(page_runtime(store, request.match_info["id"], dict(request.query)))

    async def history(request: web.Request) -> web.Response:
        page = page_artifacts if request.path.endswith("/artifacts") else page_jobs
        return web.json_response(page(store, dict(request.query)))

    async def grid_preview(request: web.Request) -> web.Response:
        return web.json_response(await grids.preview(await payload(request)))

    async def grid_submit(request: web.Request) -> web.Response:
        return web.json_response(grids.submit(await payload(request)), status=202)

    async def grid_get(request: web.Request) -> web.Response:
        if "id" in request.match_info:
            return web.json_response(grids.get(request.match_info["id"]))
        return web.json_response({"items": [grids.get(item["id"]) for item in store.list("grid_runs")]})

    async def grid_retry(request: web.Request) -> web.Response:
        return web.json_response(grids.retry(request.match_info["id"], await payload(request)), status=202)

    async def request_status(request: web.Request) -> web.Response:
        with store.lock:
            row = store.db.execute("SELECT kind,entity_id FROM requests WHERE key=?", (request.match_info["key"],)).fetchone()
        if row is None:
            raise StudioError("not_found", "No accepted request exists for this idempotency key.", 404)
        return web.json_response(store.get(row["kind"], row["entity_id"]))

    async def records(request: web.Request) -> web.Response:
        kind = request.match_info["kind"]
        if kind not in RECORD_KINDS:
            raise StudioError("record_kind", "Unknown record kind.", 404)
        if request.method == "GET":
            return web.json_response(library.page(kind, dict(request.query)))
        body = await payload(request)
        value = library.save(kind, request.match_info["id"], object_value(body.get("data")), body.get("revision"))
        return web.json_response(value)

    async def library_export(request: web.Request) -> web.Response:
        return web.json_response(library.export(request.query.get("project_id")))

    async def library_preview(request: web.Request) -> web.Response:
        return web.json_response(library.preview(await payload(request)))

    async def library_import(request: web.Request) -> web.Response:
        return web.json_response(library.apply(await payload(request)))

    async def membership(request: web.Request) -> web.Response:
        return web.json_response(library.membership(request.match_info["id"], await payload(request)))

    async def operation_preview(request: web.Request) -> web.Response:
        body = await payload(request)
        return web.json_response(await ops.prepare(body))

    async def operation_apply(request: web.Request) -> web.Response:
        return web.json_response(ops.apply(request.match_info["id"], await payload(request)), status=202)

    async def media(request: web.Request) -> web.StreamResponse:
        artifact = store.get("artifacts", request.match_info["id"])
        if artifact.get("source") == "live_preview":
            from .live_preview import preview_frame
            data, content_type = preview_frame(store, artifact["id"])
            return web.Response(body=data, content_type=content_type)
        upstream = await client.media(artifact)
        limit = MEDIA_LIMIT
        if upstream.content_length and upstream.content_length > limit:
            upstream.release()
            raise StudioError("media_limit", "Media exceeds the 64 MiB display budget.", 413)
        response = web.StreamResponse(headers={"Content-Type": mimetypes.guess_type(artifact["filename"])[0] or "application/octet-stream",
                                               "X-Content-Type-Options": "nosniff", "Cache-Control": "private, no-store"})
        try:
            await response.prepare(request)
            count = 0
            async for chunk in upstream.content.iter_chunked(65536):
                count += len(chunk)
                if count > limit:
                    # Headers have been sent. End the transport without a valid
                    # chunk terminator so clients reject, not save a partial file.
                    if request.transport:
                        request.transport.close()
                    return response
                await response.write(chunk)
            await response.write_eof()
        finally:
            upstream.release()
        return response

    async def upload_asset(request: web.Request) -> web.Response:
        return web.json_response(await media_assets.upload(await payload(request)), status=201)

    async def thumbnail(request: web.Request) -> web.Response:
        return web.Response(body=await media_assets.preview(request.match_info["id"]), content_type="image/jpeg")

    async def static_file(request: web.Request) -> web.Response:
        path = request.match_info.get("path", "") or "index.html"
        if path not in assets:
            raise StudioError("not_found", "Bundled asset not found.", 404)
        return web.Response(body=assets[path], content_type=mimetypes.guess_type(path)[0] or "application/octet-stream")

    app.router.add_get("/api/v1/capabilities", capabilities)
    app.router.add_get("/api/v1/snapshot", snapshot)
    app.router.add_get("/api/v1/events", events)
    app.router.add_get("/api/v1/catalog", catalog)
    app.router.add_get("/api/v1/workflows", workflows)
    app.router.add_post("/api/v1/workflows/inspect", inspect)
    app.router.add_post("/api/v1/jobs/preview", preview)
    app.router.add_post("/api/v1/jobs", submit)
    app.router.add_get("/api/v1/jobs", history)
    app.router.add_get("/api/v1/jobs/request/{key}", request_status)
    app.router.add_get("/api/v1/jobs/previews/{id}", job_preview)
    app.router.add_get("/api/v1/jobs/{id}", job)
    app.router.add_get("/api/v1/jobs/{id}/runtime", runtime_history)
    app.router.add_post("/api/v1/jobs/{id}/{action}", job_action)
    app.router.add_get("/api/v1/grids", grid_get)
    app.router.add_post("/api/v1/grids/preview", grid_preview)
    app.router.add_post("/api/v1/grids", grid_submit)
    app.router.add_get("/api/v1/grids/{id}", grid_get)
    app.router.add_post("/api/v1/grids/{id}/retry", grid_retry)
    app.router.add_get("/api/v1/records/{kind}", records)
    app.router.add_put("/api/v1/records/{kind}/{id}", records)
    app.router.add_get("/api/v1/library/export", library_export)
    app.router.add_post("/api/v1/library/import/preview", library_preview)
    app.router.add_post("/api/v1/library/import", library_import)
    app.router.add_post("/api/v1/operations/preview", operation_preview)
    app.router.add_post("/api/v1/operations/{id}/apply", operation_apply)
    app.router.add_post("/api/v1/assets", upload_asset)
    app.router.add_get("/api/v1/artifacts", history)
    app.router.add_get("/api/v1/artifacts/{id}/thumbnail", thumbnail)
    app.router.add_get("/api/v1/artifacts/{id}/file", media)
    app.router.add_put("/api/v1/artifacts/{id}/projects", membership)
    app.router.add_get("/{path:.*}", static_file)

    async def lifecycle(application: web.Application) -> Any:
        store.recover()
        await client.start()
        async def observe() -> None:
            while True:
                await jobs.reconcile()
                await asyncio.sleep(2)
        task = asyncio.create_task(observe()) if poll else None
        stream = asyncio.create_task(jobs.watch_events()) if poll else None
        yield
        for listener in (task, stream):
            if not listener:
                continue
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener
        # Do not kill an in-flight management subprocess on companion shutdown.
        for pending in [*jobs.tasks, *ops.tasks]:
            pending.cancel()
        await asyncio.gather(*jobs.tasks, *ops.tasks, return_exceptions=True)
        await client.close()

    app.cleanup_ctx.append(lifecycle)
    return app
