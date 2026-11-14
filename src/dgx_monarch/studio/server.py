"""Authenticated standalone host. No import or startup of ComfyUI is needed."""
from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import mimetypes
from importlib.resources import files
from typing import Any

from aiohttp import web

from . import SCHEMA_VERSION
from .comfy import ComfyClient
from .contracts import (
    MAX_BODY_BYTES,
    RECORD_KINDS,
    StudioError,
    canonical,
    identifier,
    object_value,
    schema_document,
    timestamp,
)
from .experiments import GridExperiments
from .grid_views import read as read_grid
from .grid_views import summary as grid_summary
from .history import page_artifacts, page_grids, page_jobs
from .jobs import Jobs
from .library import Library
from .media_assets import MediaAssets
from .operations import Operations
from .runtime_history import page_runtime
from .store import Store
from .wire_records import snapshot_window, telemetry_summary
from .workflows import WorkflowError, inspect_workflow

LOG = logging.getLogger(__name__)
MEDIA_LIMIT = 64 * 1024 * 1024
CURSOR_LIMIT = 2**63 - 1  # SQLite's largest integer; a larger cursor would overflow the query.
OBSERVE_SECONDS = 2.0
BACKOFF_SECONDS = 30.0
HTTP_CODES = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "request_limit"}


def json_response(value: Any, *, status: int = 200) -> web.Response:
    return web.json_response(value, status=status, dumps=canonical)


def error_response(code: str, message: str, status: int) -> web.Response:
    return json_response({"error": {"code": code, "message": message}}, status=status)


def http_error(exc: web.HTTPException) -> web.Response:
    """Give aiohttp's own refusals, such as the request size limit, the JSON error envelope."""
    if exc.status == 413:
        message = f"Request body exceeds the {MAX_BODY_BYTES // (1024 * 1024)} MiB companion limit."
    else:
        message = f"{exc.reason}."
    response = error_response(HTTP_CODES.get(exc.status, "http_error"), message, exc.status)
    if "Allow" in exc.headers:
        response.headers["Allow"] = exc.headers["Allow"]
    return response


def degrade(jobs: Jobs, code: str) -> None:
    """Show an unexpected background failure as an unavailable driver, keeping when it began."""
    current = jobs.driver
    since = current.get("since") if current.get("state") == "unavailable" and current.get("code") == code else None
    jobs.driver = {**current, "state": "unavailable", "code": code, "since": since or timestamp()}


async def supervise(name: str, step: Any, *, repeat: bool, degrades: Jobs | None = None,
                    wake: asyncio.Event | None = None) -> None:
    """Run ``step`` until cancelled; an unexpected error is logged and retried after a growing pause.

    asyncio.CancelledError is not an Exception, so shutdown still ends the loop.
    Only the observer marks the driver unavailable: reconciliation, not the event
    feed, is what keeps the snapshot's driver state current. Setting ``wake``
    ends the pause between steps early, so one more step runs after it was set;
    it never cuts short the pause after an error.
    """
    delay = OBSERVE_SECONDS
    while True:
        if wake is not None:
            wake.clear()  # The step that starts now serves every wake set before it.
        try:
            await step()
        except Exception:
            LOG.exception("Studio %s failed; retrying in %.0f s", name, delay)
            if degrades is not None:
                degrade(degrades, f"{name}_failed")
            await asyncio.sleep(delay)
            delay = min(delay * 2, BACKOFF_SECONDS)
            continue
        if not repeat:
            return
        delay = OBSERVE_SECONDS
        if wake is None:
            await asyncio.sleep(OBSERVE_SECONDS)
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wake.wait(), OBSERVE_SECONDS)


def create_app(store: Store, client: ComfyClient, token: str, *, allowed_hosts: set[str],
               operations: Operations | None = None, poll: bool = True) -> web.Application:
    if len(token) < 32:
        raise ValueError("Authentication token must contain at least 32 characters")
    store.bind_driver(client.origin)
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
                # aiohttp decodes header bytes with surrogateescape; encode them back the same way.
                authorization = request.headers.get("Authorization", "").encode("utf-8", "surrogateescape")
                if not hmac.compare_digest(authorization, f"Bearer {token}".encode()):
                    raise StudioError("authentication", "Enter this companion's owner token.", 401)
                version = request.headers.get("X-DGXM-Schema", SCHEMA_VERSION)
                if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
                    raise StudioError("schema_version", "Client schema is incompatible; reload this companion's bundled client.", 409)
                if request.method in {"POST", "PUT", "DELETE"} and request.content_type != "application/json":
                    raise StudioError("content_type", "Mutations require application/json.", 415)
            response = await handler(request)
        except StudioError as exc:
            response = error_response(exc.code, str(exc), exc.status)
        except WorkflowError as exc:
            response = error_response("workflow", str(exc), 400)
        except (json.JSONDecodeError, UnicodeDecodeError):
            response = error_response("invalid_json", "Invalid JSON request.", 400)
        except web.HTTPException as exc:
            if exc.status < 400:
                raise
            response = http_error(exc)
        except Exception:
            LOG.exception("Studio request %s %s failed", request.method, request.path)
            response = error_response("internal", "The companion could not complete this request; "
                                                  "its log records the cause.", 500)
        return response

    async def response_headers(request: web.Request, response: web.StreamResponse) -> None:
        # Streaming handlers prepare their response before middleware returns.
        response.headers.update({"X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
                                 "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'",
                                 "Cache-Control": "no-store"})

    app = web.Application(middlewares=[boundary], client_max_size=MAX_BODY_BYTES)
    app.on_response_prepare.append(response_headers)

    async def payload(request: web.Request) -> dict[str, Any]:
        return object_value(await request.json())

    async def capabilities(request: web.Request) -> web.Response:
        from .presentation import presentation_contract
        return json_response({**schema_document(), "instance_id": instance, "operations": ops.capabilities(),
                                  "presentation": presentation_contract(), "driver_origin": client.origin,
                                  "driver_start": ops.driver_start,
                                  "support_status": "supported; docs/STUDIO_ACCEPTANCE.md lists the acceptance items that remain open"})

    async def snapshot(request: web.Request) -> web.Response:
        rows, omitted, cursor = snapshot_window(store)
        return json_response({"schema_version": SCHEMA_VERSION, "instance_id": instance,
                                  "server_time": timestamp(), "cursor": cursor,
                                  "jobs": rows["jobs"], "artifacts": rows["artifacts"], "operations": rows["operations"],
                                  "records": {kind: rows[kind] for kind in RECORD_KINDS}, "omitted": omitted,
                                  "jobs_truncated": bool(omitted.get("jobs")),
                                  "telemetry": telemetry_summary(jobs.telemetry), "driver": jobs.driver})

    async def events(request: web.Request) -> web.Response:
        try:
            cursor = int(request.query.get("after", "0"))
        except ValueError as exc:
            raise StudioError("cursor", "Cursor must be an integer.") from exc
        if cursor < 0:
            raise StudioError("cursor", "Cursor must be nonnegative.")
        if cursor > CURSOR_LIMIT:
            raise StudioError("cursor", "Cursor must be below 2^63.")
        return json_response(store.events(cursor))

    async def catalog(request: web.Request) -> web.Response:
        return json_response(await jobs.catalog())

    async def workflows(request: web.Request) -> web.Response:
        return web.Response(body=resources.joinpath("data/workflows.json").read_bytes(), content_type="application/json")

    async def inspect(request: web.Request) -> web.Response:
        body = await payload(request)
        catalog_value = await jobs.catalog()
        workflow = object_value(body.get("workflow"))
        # Draft values select dynamic choices; stored uploads the workflow or the
        # values name join image upload choices, as at preview.
        values = object_value(body.get("values", {}), "values")
        view = jobs.upload_view({"workflow": workflow, "values": values}, catalog_value["object_info"])
        return json_response(inspect_workflow(workflow, view, values))

    async def preview(request: web.Request) -> web.Response:
        return json_response(await jobs.preview(await payload(request)))

    async def submit(request: web.Request) -> web.Response:
        body = await payload(request)
        existing = store.request("jobs", body.get("idempotency_key"), body)
        if existing is None:
            await jobs.catalog()
        return json_response(existing if existing is not None else jobs.submit(body), status=202)

    async def refresh_before_retry(kind: str, body: dict[str, Any], bound: dict[str, Any]) -> None:
        """Refresh the driver schema before a new confirmed retry, as submit does.

        An accepted retry replays offline. An unconfirmed retry never reaches the
        driver. A key conflict is left to the retry itself, which owns that rule.
        """
        try:
            if store.request(kind, body.get("idempotency_key"), bound) is not None:
                return
        except StudioError as exc:
            if exc.code != "idempotency_conflict":
                raise
        if body.get("confirmed") is True:
            await jobs.catalog()

    async def job_action(request: web.Request) -> web.Response:
        body = await payload(request)
        if request.match_info["action"] == "cancel":
            from .contracts import bounded_key
            bounded_key(body.get("idempotency_key"))
            value = await jobs.cancel(request.match_info["id"], body)
        elif request.match_info["action"] == "retry":
            await refresh_before_retry("jobs", body, {**body, "retry_of": request.match_info["id"]})
            value = jobs.retry(request.match_info["id"], body)
        else:
            raise StudioError("job_action", "Unknown job action.", 404)
        return json_response(value, status=202)

    async def job(request: web.Request) -> web.Response:
        return json_response(store.get("jobs", request.match_info["id"]))

    async def job_preview(request: web.Request) -> web.Response:
        return json_response(store.get("previews", request.match_info["id"]))

    async def runtime_history(request: web.Request) -> web.Response:
        return json_response(page_runtime(store, request.match_info["id"], dict(request.query)))

    async def history(request: web.Request) -> web.Response:
        page = page_artifacts if request.path.endswith("/artifacts") else page_jobs
        return json_response(page(store, dict(request.query)))

    async def grid_preview(request: web.Request) -> web.Response:
        compact = compact_grid(request)
        value = await grids.preview(await payload(request))
        return json_response(grid_summary(value) if compact else value)

    def compact_grid(request: web.Request) -> bool:
        value = request.query.get("summary", "false")
        if value not in {"true", "false"}:
            raise StudioError("grid_summary", "Summary must be true or false.")
        return value == "true"

    async def grid_submit(request: web.Request) -> web.Response:
        compact = compact_grid(request)
        body = await payload(request)
        existing = store.request("grid_runs", body.get("idempotency_key"), body)
        if existing is None:
            await jobs.catalog()
        value = grids.get(existing["id"], compact=compact) if existing is not None else grids.submit(body)
        return json_response(grid_summary(value) if compact else value, status=202)

    async def grid_get(request: web.Request) -> web.Response:
        if "id" in request.match_info:
            return json_response(grids.get(request.match_info["id"], compact=compact_grid(request)))
        return json_response(page_grids(store, dict(request.query)))

    async def grid_preview_get(request: web.Request) -> web.Response:
        return json_response(read_grid(store, "grid_previews", request.match_info["id"], compact=compact_grid(request)))

    async def grid_retry(request: web.Request) -> web.Response:
        compact = compact_grid(request)
        body = await payload(request)
        await refresh_before_retry("grid_retries", body, {**body, "grid_id": request.match_info["id"]})
        value = grids.retry(request.match_info["id"], body)
        return json_response(grid_summary(value) if compact else value, status=202)

    async def request_status(request: web.Request) -> web.Response:
        with store.lock:
            row = store.db.execute("SELECT kind,entity_id FROM requests WHERE key=?", (request.match_info["key"],)).fetchone()
        if row is None:
            raise StudioError("not_found", "No accepted request exists for this idempotency key.", 404)
        value = store.get(row["kind"], row["entity_id"])
        if row["kind"] == "job_cancellations":
            value = store.get("jobs", value["job_id"])
        return json_response(value)

    async def records(request: web.Request) -> web.Response:
        kind = request.match_info["kind"]
        if kind not in RECORD_KINDS:
            raise StudioError("record_kind", "Unknown record kind.", 404)
        if request.method == "GET":
            return json_response(library.page(kind, dict(request.query)))
        body = await payload(request)
        value = library.save(kind, request.match_info["id"], object_value(body.get("data")), body.get("revision"))
        return json_response(value)

    async def library_export(request: web.Request) -> web.Response:
        return json_response(library.export(request.query.get("project_id")))

    async def library_preview(request: web.Request) -> web.Response:
        return json_response(library.preview(await payload(request)))

    async def library_import(request: web.Request) -> web.Response:
        return json_response(library.apply(await payload(request)))

    async def membership(request: web.Request) -> web.Response:
        return json_response(library.membership(request.match_info["id"], await payload(request)))

    async def operation_preview(request: web.Request) -> web.Response:
        body = await payload(request)
        return json_response(await ops.prepare(body))

    async def operation_apply(request: web.Request) -> web.Response:
        return json_response(ops.apply(request.match_info["id"], await payload(request)), status=202)

    async def operation_get(request: web.Request) -> web.Response:
        return json_response(store.get("operations", request.match_info["id"]))

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
        except Exception as exc:
            if not response.prepared:
                raise
            # Headers are out, so a JSON error cannot follow; end the stream the same way.
            LOG.warning("Studio media stream for %s ended early: %s", artifact["id"], type(exc).__name__)
            if request.transport:
                request.transport.close()
        finally:
            upstream.release()
        return response

    async def upload_asset(request: web.Request) -> web.Response:
        return json_response(await media_assets.upload(await payload(request)), status=201)

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
    app.router.add_get("/api/v1/grids/previews/{id}", grid_preview_get)
    app.router.add_get("/api/v1/grids/{id}", grid_get)
    app.router.add_post("/api/v1/grids/{id}/retry", grid_retry)
    app.router.add_get("/api/v1/records/{kind}", records)
    app.router.add_put("/api/v1/records/{kind}/{id}", records)
    app.router.add_get("/api/v1/library/export", library_export)
    app.router.add_post("/api/v1/library/import/preview", library_preview)
    app.router.add_post("/api/v1/library/import", library_import)
    app.router.add_post("/api/v1/operations/preview", operation_preview)
    app.router.add_get("/api/v1/operations/{id}", operation_get)
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
        listeners = [asyncio.create_task(supervise("observer", jobs.reconcile, repeat=True, degrades=jobs, wake=jobs.wake)),
                     asyncio.create_task(supervise("event_watch", jobs.watch_events, repeat=False))] if poll else []
        yield
        try:
            for listener in listeners:
                listener.cancel()
            # A listener that already failed must not stop the rest of shutdown.
            for outcome in await asyncio.gather(*listeners, return_exceptions=True):
                if isinstance(outcome, Exception):
                    LOG.error("Studio background listener ended with an error", exc_info=outcome)
            # Do not kill an in-flight management subprocess on companion shutdown.
            for pending in [*jobs.tasks, *ops.tasks]:
                pending.cancel()
            await asyncio.gather(*jobs.tasks, *ops.tasks, return_exceptions=True)
        finally:
            await client.close()

    app.cleanup_ctx.append(lifecycle)
    return app
