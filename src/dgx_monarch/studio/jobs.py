"""Immutable submissions and reconciliation against the driver's authoritative queue."""
from __future__ import annotations

import asyncio
import logging
import math
import secrets
import sqlite3
import time
from copy import deepcopy
from typing import Any

from .. import __version__
from . import SCHEMA_VERSION
from .comfy import ComfyClient
from .contracts import (
    MAX_BODY_BYTES,
    StudioError,
    bounded_key,
    canonical,
    digest,
    identifier,
    object_value,
    timestamp,
)
from .job_observations import model_identities, observation, prompt_ended, requested_topology
from .preview_drafts import prepare_preview
from .reconcile import job_for_prompt, reconcile_cycle
from .runtime_observations import RuntimeObservations
from .store import Store
from .submission_guard import (
    check_reviewed,
    node_error_summary,
    referenced_uploads,
    schema_identity,
    stored_upload_names,
    upload_view,
)
from .threads import blocking
from .workflows import node_inputs

LOG = logging.getLogger(__name__)
TELEMETRY_FRESH_SECONDS = 30
# Review-only preview fields that a submitted job's frozen snapshot leaves out.
SNAPSHOT_OMITTED = frozenset({"id", "created_at", "expires_at", "errors", "warnings"})
# Seconds a preview stays submittable. Submit checks the node definitions and each chosen file
# against the live driver whatever the age, so this only bounds how old the frozen runtime
# observation can be when the job is accepted, and lets the sweep delete unused previews.
PREVIEW_SECONDS = 600


def _random_seed(schema: dict[str, Any]) -> int:
    """Sample the installed range that browser JSON can round-trip exactly."""
    bounds = [schema.get("min", 0), schema.get("max", 2**64 - 1)]
    if any(type(bound) not in (int, float) or (type(bound) is float and not math.isfinite(bound)) for bound in bounds):
        raise StudioError("seed_bounds", "Random seeds require finite numeric bounds in the installed input schema.")
    safe_integer = 2**53 - 1
    low, high = max(math.ceil(bounds[0]), -safe_integer), min(math.floor(bounds[1]), safe_integer)
    if low > high:
        raise StudioError("seed_bounds", "The installed seed input has no integer range that browser JSON can represent exactly.")
    return low + secrets.randbelow(high - low + 1)


def fault_code(exc: BaseException) -> str:
    """Name an unexpected background failure without exposing its text."""
    if isinstance(exc, StudioError):
        return exc.code
    if isinstance(exc, sqlite3.Error):
        return "studio_storage"
    if isinstance(exc, (RecursionError, OverflowError)):
        return "driver_data"
    return "studio_internal"


class Jobs:
    def __init__(self, store: Store, client: ComfyClient):
        self.store, self.client = store, client
        self.schema: dict[str, Any] = {}
        self.telemetry: dict[str, Any] = {}
        self.telemetry_clock: float | None = None
        self.driver: dict[str, Any] = {"state": "unavailable", "timestamp": None}
        self.tasks: set[asyncio.Task[Any]] = set()
        self.runtime: dict[str, Any] = {}
        self.observation_lock = asyncio.Lock()
        self.history_cursor = 0
        self.runtime_events = RuntimeObservations(store, client)
        # Jobs whose cancellation request is on its way to the driver.
        self.cancel_dispatches: set[str] = set()
        self.scoped_cancel_unavailable = False
        self.events_fault: dict[str, Any] | None = None
        # The watcher sets this when the driver reports that a prompt ended, so
        # the observer settles it from history without waiting out its pause.
        self.wake = asyncio.Event()
        # Seconds: the normal event reconnect pause, then the background back-off range.
        self.event_pause, self.backoff_base, self.backoff_limit = 2.0, 2.0, 30.0
        self.observer_failures, self.observer_retry_at = 0, 0.0

    def launch(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    def degrade(self, code: str) -> None:
        """Report a background failure in the vocabulary the client knows (state, code, since)."""
        since = self.driver.get("since") if self.driver.get("state") == "unavailable" else None
        self.driver = {**self.driver, "state": "unavailable", "code": code, "since": since or timestamp()}
        self.telemetry = {**self.telemetry, "state": "stale" if self.telemetry else "unavailable"}

    def _backoff(self, failures: int) -> float:
        return min(self.backoff_limit, self.backoff_base * 2 ** max(failures - 1, 0))

    async def catalog(self) -> dict[str, Any]:
        self.schema = object_value(await self.client.request("GET", "/object_info"), "object_info")
        entries = []
        for node in ("DGXMonarchLoraLoader", "LoraLoader", "LoraLoaderModelOnly"):
            choices = self.schema.get(node, {}).get("input", {}).get("required", {}).get("lora_name", [[]])[0]
            if isinstance(choices, list):
                for name in choices:
                    if isinstance(name, str):
                        entries.append({"id": name, "name": name, "kind": "lora", "compatibility": "unknown", "source_node": node})
        return {"object_info": self.schema, "loras": entries}

    async def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        schema = (await self.catalog())["object_info"]
        runtime = await self.capture_runtime()
        return self.build_preview(request, schema=schema, runtime=runtime)

    async def capture_runtime(self) -> dict[str, Any]:
        try:
            stats = object_value(await self.client.request("GET", "/system_stats"))
            system = object_value(stats.get("system", {}))
            runtime = {"observed_at": timestamp(), "source": "driver_system_stats", "reported": {
                key: system[key] for key in ("comfyui_version", "python_version", "pytorch_version", "required_frontend_version")
                if key in system}}
            # Freshness uses the monotonic clock: a wall-clock step back must
            # not make an old diagnostic identity look current.
            fresh = self.telemetry_clock is not None and time.monotonic() - self.telemetry_clock < TELEMETRY_FRESH_SECONDS
            if self.telemetry.get("state") == "connected" and fresh:
                sample = self.telemetry.get("sample", {})
                if isinstance(sample, dict):
                    runtime["monarch_diagnostic_identity"] = deepcopy(sample.get("diagnostic_identity"))
                    runtime["comfy_source"] = deepcopy(sample.get("comfy"))
                    runtime["identity_observed_at"] = self.telemetry["observed_at"]
        except StudioError:
            runtime = {"observed_at": timestamp(), "source": "unavailable", "reported": {}}
        self.runtime = runtime
        return runtime

    def upload_view(self, request: Any, schema: dict[str, Any]) -> dict[str, Any]:
        """The schema a draft validates against: stored uploads it names join image upload choices."""
        return upload_view(schema, referenced_uploads(request, stored_upload_names(self.store)))

    def build_preview(self, request: dict[str, Any], *, schema: dict[str, Any] | None = None,
                      runtime: dict[str, Any] | None = None, persist: bool = True,
                      view: dict[str, Any] | None = None) -> dict[str, Any]:
        schema = self.schema if schema is None else schema
        runtime = self.runtime if runtime is None else runtime
        workflow = object_value(request.get("workflow"), "workflow")
        parents = request.get("parent_artifact_ids", [])
        if not isinstance(parents, list) or len(parents) > 128 or any(not isinstance(parent, str) for parent in parents):
            raise StudioError("artifact_lineage", "Parent artifacts must be at most 128 saved artifact identities.")
        for parent in parents:
            self.store.get("artifacts", parent)
        if view is None:
            view = self.upload_view({"values": request.get("values"), "workflow": workflow}, schema)
        prompt, values, transformations = prepare_preview(request, schema, view=view)
        effective = deepcopy(request)
        effective["values"] = deepcopy(values)
        effective.update(transformations)
        policy = request.get("seed_policy", "fixed")
        if policy not in {"fixed", "random"}:
            raise StudioError("seed_policy", "Seed policy must be fixed or random.")
        seeds: dict[str, int] = {}
        for node_id, node in prompt.items():
            fields = {name: field for name, field, _ in node_inputs(schema[node["class_type"]])}
            for name in ("seed", "noise_seed"):
                if type(node["inputs"].get(name)) is int and fields.get(name, {}).get("type") == "INT":
                    if policy == "random":
                        node["inputs"][name] = _random_seed(fields[name])
                    seeds[f"{node_id}.{name}"] = node["inputs"][name]
        effective["seeds"] = seeds
        operation = request.get("operation", "workflow")
        if operation not in {"image", "video", "edit", "upscale", "workflow", "fleet"}:
            raise StudioError("operation", "Unknown generation operation.")
        # The whole-schema digest stays as provenance; review freshness binds
        # to the identity of the classes this prompt uses.
        snapshot = {"requested": deepcopy(request), "effective": effective, "prompt": prompt, "workflow": deepcopy(workflow),
                    "workflow_digest": digest(workflow), "schema_digest": digest(schema),
                    "schema_identity": schema_identity(prompt, schema),
                    "adapter_version": SCHEMA_VERSION, "schema_version": SCHEMA_VERSION,
                    "versions": {"dgx_monarch": __version__, "driver": runtime.get("reported", {}).get("comfyui_version", "unknown")},
                    "runtime_observation": deepcopy(runtime), "model_identities": model_identities(prompt),
                    "requested_topology": requested_topology(prompt),
                    "project_id": request.get("project_id"), "parent_artifact_ids": request.get("parent_artifact_ids", []),
                    "model_fingerprints": "unavailable", "effective_topology": "pending_backend_admission"}
        # A warning asks a person to decide before the job runs. A review raises none yet, so
        # Generate submits at once; the checks that run at dispatch are not warnings.
        value = {"id": identifier(), "digest": digest(snapshot), **snapshot, "created_at": timestamp(),
                 "expires_at": timestamp() + PREVIEW_SECONDS, "errors": [], "warnings": []}
        if len(canonical(value).encode()) > MAX_BODY_BYTES:
            raise StudioError("preview_limit", "Expanded submission preview exceeds the 4 MiB JSON budget.", 413)
        return self.store.create("previews", value) if persist else value

    def reviewed_snapshot(self, preview_id: str, *, uploads: frozenset[str] | None = None) -> dict[str, Any]:
        try:
            preview = self.store.get("previews", preview_id)
        except StudioError as exc:
            # The expiry sweep deletes expired previews, so a missing one is expired too.
            if exc.code != "not_found":
                raise
            raise StudioError("preview_expired", "Preview expired; review a fresh submission.", 409) from exc
        if preview["expires_at"] < timestamp():
            raise StudioError("preview_expired", "Preview expired; review a fresh submission.", 409)
        check_reviewed(preview, self.schema, stored_upload_names(self.store) if uploads is None else uploads)
        return {key: value for key, value in preview.items() if key not in SNAPSHOT_OMITTED}

    def revalidate(self, submission: dict[str, Any], *, uploads: frozenset[str] | None = None) -> None:
        """A confirmed retry re-dispatches a reviewed submission; the live schema replaces review freshness."""
        check_reviewed(submission, self.schema, stored_upload_names(self.store) if uploads is None else uploads)

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        existing = self.store.request("jobs", request.get("idempotency_key"), request)
        if existing is not None:
            return existing
        snapshot = self.reviewed_snapshot(request.get("preview_id", ""))
        job = {"id": identifier(), "state": "submitting", "submission": snapshot, "prompt_id": None,
               "client_prompt_id": identifier(), "idempotency_key": request.get("idempotency_key"),
               "created_at": timestamp(), "updated_at": timestamp()}
        value, created = self.store.reserve("jobs", request.get("idempotency_key"), request, job)
        if created:
            self.launch(self.dispatch(value))
        return value

    async def retry_runtime(self, prompt: Any, runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        """The runtime a retry actually dispatches against, kept outside its frozen submission."""
        runtime = await self.capture_runtime() if runtime is None else runtime
        return {"observed_at": timestamp(), "runtime_observation": deepcopy(runtime),
                "versions": {"dgx_monarch": __version__, "driver": runtime.get("reported", {}).get("comfyui_version", "unknown")},
                "schema_digest": digest(self.schema), "schema_identity": schema_identity(prompt, self.schema)}

    async def dispatch(self, job: dict[str, Any], *, runtime: dict[str, Any] | None = None) -> None:
        try:
            await self._dispatch(job, runtime)
        except Exception as exc:  # A dispatch fault must settle the job, never strand it as submitting.
            code = fault_code(exc)
            LOG.error("Studio dispatch for job %s failed (%s).", job.get("id"), code, exc_info=True)
            self.degrade(code)
            try:
                if self.store.get("jobs", job["id"])["state"] == "submitting":
                    self.store.update("jobs", job["id"], {"state": "unknown", "error": {
                        "code": code, "message": "Dispatch stopped before its outcome was recorded; reconcile before retrying."}})
            except Exception:
                LOG.error("Studio could not record the dispatch fault for job %s.", job.get("id"), exc_info=True)

    async def _dispatch(self, job: dict[str, Any], runtime: dict[str, Any] | None) -> None:
        await self.runtime_events.poll()
        snapshot = job["submission"]
        client_prompt_id = job.get("client_prompt_id")
        if not isinstance(client_prompt_id, str):
            # The companion issues the identity before the driver sees the prompt,
            # so a lost or malformed acknowledgement still reconciles exactly.
            client_prompt_id = identifier()
            self.store.update("jobs", job["id"], {"client_prompt_id": client_prompt_id})
        if job.get("retry_of"):
            self.store.update("jobs", job["id"], {"retry_runtime": await self.retry_runtime(snapshot.get("prompt"), runtime)})
        try:
            result = await self.client.request("POST", "/prompt", {
                "prompt": snapshot["prompt"], "prompt_id": client_prompt_id, "client_id": self.store.client_id,
                "extra_data": {"studio": {"job_id": job["id"], "digest": snapshot["digest"]},
                               "extra_pnginfo": {"workflow": snapshot["workflow"]}},
            })
        except StudioError as exc:
            self._refused(job, exc)
            return
        try:
            self._accepted(job, result)
        except StudioError as exc:
            # A late acknowledgement cannot rebind a job reconciliation already bound.
            current = self.store.get("jobs", job["id"])
            self.store.update("jobs", job["id"], {"state": "unknown" if current["state"] == "submitting" else current["state"],
                                                  "error": {"code": exc.code, "message": str(exc)}})

    def _refused(self, job: dict[str, Any], exc: StudioError) -> None:
        current = self.store.get("jobs", job["id"])
        status = getattr(exc, "upstream_status", None)
        if type(status) is int and 400 <= status <= 499:
            # ComfyUI validates before queueing, so a client error proves the
            # prompt was not queued; transport failures and 5xx may follow acceptance.
            summary = node_error_summary(getattr(exc, "upstream_detail", None))
            self.store.update("jobs", job["id"], {
                "state": "failed" if current["state"] == "submitting" else current["state"],
                # Settled by the refusal itself: there is no driver history to look up.
                "reconciliation": {"source": "driver_refusal"},
                "error": {"code": "prompt_rejected", "status": status, "node_errors": summary,
                          "message": f"Driver refused the prompt before queueing it (HTTP {status})."}})
            return
        self.store.update("jobs", job["id"], {"state": "unknown" if current["state"] == "submitting" else current["state"],
                                              "error": {"code": exc.code, "message": str(exc)}})

    def _accepted(self, job: dict[str, Any], result: Any) -> None:
        try:
            prompt_id = bounded_key(result.get("prompt_id") if isinstance(result, dict) else None, "prompt id")
        except StudioError as exc:
            raise StudioError("submission_unknown", "Driver acceptance did not return a usable prompt identity.", 503) from exc
        updates: dict[str, Any] = {"prompt_id": prompt_id, "state": "queued", "accepted_at": timestamp()}
        errors = result.get("node_errors")
        if isinstance(errors, dict) and errors:
            # ComfyUI queued the outputs that validated and dropped the rest.
            updates["driver_warnings"] = [{"code": "node_errors", "node_errors": node_error_summary(errors),
                                           "message": "The driver queued this prompt without some outputs that failed validation."}]
        self.store.update("jobs", job["id"], updates)

    async def reconcile(self) -> None:
        if time.monotonic() < self.observer_retry_at:
            return  # Backing off after an unexpected fault.
        async with self.observation_lock:
            try:
                await reconcile_cycle(self)
            except Exception as exc:  # Never end the observer; CancelledError still propagates.
                self.observer_failures += 1
                delay = self._backoff(self.observer_failures)
                self.observer_retry_at = time.monotonic() + delay
                code = fault_code(exc)
                LOG.error("Studio reconciliation failed (%s); next attempt in %.0f s.", code, delay, exc_info=True)
                self.degrade(code)
                return
            self.observer_failures, self.observer_retry_at = 0, 0.0

    async def cancel(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
        from .cancellation import cancel_job
        return await cancel_job(self, job_id, request)

    def consume_event(self, message: dict[str, Any], *, known: dict[str, Any] | None = None, live: bool = False) -> None:
        parsed = observation(message)
        if parsed is None:
            return
        event, updates = parsed
        if live and event["type"] == "execution_success":
            # ComfyUI sends this before it writes the history entry that lists
            # the outputs. History settlement stores them and then marks the
            # job succeeded, so a job never reads succeeded without its artifacts.
            updates = {key: value for key, value in updates.items() if key != "state"}
        prompt_id = event["data"]["prompt_id"]
        job = job_for_prompt(self.store, prompt_id) if known is None else known.get(prompt_id)
        if job is None:
            return
        if job.get("prompt_id") is None:
            # The driver used the identity this companion sent; bind it now.
            updates = {**updates, "prompt_id": prompt_id}
        self.store.observe(job["id"], event, updates, live=live)

    async def watch_events(self) -> None:
        stream = getattr(self.client, "stream_events", None)
        if not callable(stream):
            return
        failures = 0
        while True:
            try:
                async for message in stream(self.store.client_id):
                    async with self.observation_lock:
                        if message.get("type") == "preview_frame":
                            from .live_preview import accept_preview
                            await blocking(accept_preview, self.store, message)
                        else:
                            self.consume_event(message, live=True)
                            if prompt_ended(message):
                                self.wake.set()
                    failures, self.events_fault = 0, None
                delay = self.event_pause
            except StudioError:
                delay = self.event_pause
            except Exception as exc:  # Never end the watcher; CancelledError still propagates.
                failures += 1
                delay = self._backoff(failures)
                code = fault_code(exc)
                LOG.error("Studio event watcher failed (%s); reconnecting in %.0f s.", code, delay, exc_info=True)
                since = self.events_fault.get("since") if self.events_fault else None
                self.events_fault = {"state": "unavailable", "code": code, "since": since or timestamp()}
                self.degrade(code)
                self.driver["events"] = dict(self.events_fault)
            await asyncio.sleep(delay)

    def retry(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
        key_request = {**request, "retry_of": job_id}
        existing = self.store.request("jobs", request.get("idempotency_key"), key_request)
        if existing is not None:
            return existing
        old = self.store.get("jobs", job_id)
        if request.get("confirmed") is not True or old["state"] not in {"failed", "cancelled"}:
            raise StudioError("retry_scope", "Only a confirmed failed or cancelled graph task may be retried.", 409)
        if old.get("external"):
            raise StudioError("retry_provenance", "Reuse and review the imported workflow before resubmitting.", 409)
        self.revalidate(old["submission"])
        value = {"id": identifier(), "state": "submitting", "submission": deepcopy(old["submission"]),
                 "retry_of": job_id, "prompt_id": None, "client_prompt_id": identifier(),
                 "idempotency_key": request.get("idempotency_key"), "created_at": timestamp(), "updated_at": timestamp()}
        value, created = self.store.reserve("jobs", request.get("idempotency_key"), key_request, value)
        if created:
            self.launch(self.dispatch(value))
        return value
