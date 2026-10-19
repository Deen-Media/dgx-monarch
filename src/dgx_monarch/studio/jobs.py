"""Immutable submissions and reconciliation against the driver's authoritative queue."""
from __future__ import annotations

import asyncio
import secrets
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

from .. import __version__
from . import SCHEMA_VERSION
from .comfy import ComfyClient, safe_descriptor
from .contracts import TERMINAL_STATES, StudioError, digest, identifier, object_value, timestamp
from .job_observations import model_identities, observation, requested_topology
from .preview_drafts import prepare_preview
from .runtime_observations import RuntimeObservations
from .store import Store


class Jobs:
    def __init__(self, store: Store, client: ComfyClient):
        self.store, self.client = store, client
        self.schema: dict[str, Any] = {}
        self.telemetry: dict[str, Any] = {}
        self.driver: dict[str, Any] = {"state": "unavailable", "timestamp": None}
        self.tasks: set[asyncio.Task[Any]] = set()
        self.runtime: dict[str, Any] = {}
        self.observation_lock = asyncio.Lock()
        self.history_cursor = 0
        self.runtime_events = RuntimeObservations(store, client)

    def launch(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

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
        await self.catalog()
        await self.capture_runtime()
        return self.build_preview(request)

    async def capture_runtime(self) -> None:
        try:
            stats = object_value(await self.client.request("GET", "/system_stats"))
            system = object_value(stats.get("system", {}))
            self.runtime = {"observed_at": timestamp(), "source": "driver_system_stats", "reported": {
                key: system[key] for key in ("comfyui_version", "python_version", "pytorch_version", "required_frontend_version")
                if key in system}}
            if self.telemetry.get("state") == "connected" and timestamp() - self.telemetry.get("observed_at", 0) < 30:
                sample = self.telemetry.get("sample", {})
                if isinstance(sample, dict):
                    self.runtime["monarch_diagnostic_identity"] = deepcopy(sample.get("diagnostic_identity"))
                    self.runtime["comfy_source"] = deepcopy(sample.get("comfy"))
                    self.runtime["identity_observed_at"] = self.telemetry["observed_at"]
        except StudioError:
            self.runtime = {"observed_at": timestamp(), "source": "unavailable", "reported": {}}

    def build_preview(self, request: dict[str, Any]) -> dict[str, Any]:
        workflow = object_value(request.get("workflow"), "workflow")
        parents = request.get("parent_artifact_ids", [])
        if not isinstance(parents, list) or len(parents) > 128 or any(not isinstance(parent, str) for parent in parents):
            raise StudioError("artifact_lineage", "Parent artifacts must be at most 128 saved artifact identities.")
        for parent in parents:
            self.store.get("artifacts", parent)
        prompt, values, transformations = prepare_preview(request, self.schema)
        effective = deepcopy(request)
        effective["values"] = deepcopy(values)
        effective.update(transformations)
        policy = request.get("seed_policy", "fixed")
        if policy not in {"fixed", "random"}:
            raise StudioError("seed_policy", "Seed policy must be fixed or random.")
        seeds: dict[str, int] = {}
        for node_id, node in prompt.items():
            for name in ("seed", "noise_seed"):
                if name in node["inputs"] and isinstance(node["inputs"][name], int):
                    if policy == "random":
                        node["inputs"][name] = secrets.randbelow(2**63)
                    seeds[f"{node_id}.{name}"] = node["inputs"][name]
        effective["seeds"] = seeds
        operation = request.get("operation", "workflow")
        if operation not in {"image", "video", "edit", "upscale", "workflow", "fleet"}:
            raise StudioError("operation", "Unknown generation operation.")
        snapshot = {"requested": deepcopy(request), "effective": effective, "prompt": prompt, "workflow": deepcopy(workflow),
                    "workflow_digest": digest(workflow), "schema_digest": digest(self.schema),
                    "adapter_version": SCHEMA_VERSION, "schema_version": SCHEMA_VERSION,
                    "versions": {"dgx_monarch": __version__, "driver": self.runtime.get("reported", {}).get("comfyui_version", "unknown")},
                    "runtime_observation": deepcopy(self.runtime), "model_identities": model_identities(prompt),
                    "requested_topology": requested_topology(prompt),
                    "project_id": request.get("project_id"), "parent_artifact_ids": request.get("parent_artifact_ids", []),
                    "model_fingerprints": "unavailable", "effective_topology": "pending_backend_admission"}
        value = {"id": identifier(), "digest": digest(snapshot), **snapshot, "created_at": timestamp(),
                 "expires_at": timestamp() + 600, "errors": [],
                 "warnings": ["Schema validation is complete. Memory, topology, identity and consent guards run in the backend at execution."]}
        return self.store.create("previews", value)

    def reviewed_snapshot(self, preview_id: str) -> dict[str, Any]:
        preview = self.store.get("previews", preview_id)
        if preview["expires_at"] < timestamp():
            raise StudioError("preview_expired", "Preview expired; review a fresh submission.", 409)
        if digest(self.schema) != preview["schema_digest"]:
            raise StudioError("schema_changed", "Driver capabilities changed; review a fresh preview.", 409)
        return {key: value for key, value in preview.items() if key not in {"id", "created_at", "expires_at", "errors", "warnings"}}

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        existing = self.store.request("jobs", request.get("idempotency_key"), request)
        if existing is not None:
            return existing
        snapshot = self.reviewed_snapshot(request.get("preview_id", ""))
        job = {"id": identifier(), "state": "submitting", "submission": snapshot, "prompt_id": None,
               "idempotency_key": request.get("idempotency_key"), "created_at": timestamp(), "updated_at": timestamp()}
        value, created = self.store.reserve("jobs", request.get("idempotency_key"), request, job)
        if created:
            self.launch(self.dispatch(value))
        return value

    async def dispatch(self, job: dict[str, Any]) -> None:
        await self.runtime_events.poll()
        snapshot = job["submission"]
        try:
            result = await self.client.request("POST", "/prompt", {
                "prompt": snapshot["prompt"], "client_id": self.store.client_id,
                "extra_data": {"studio": {"job_id": job["id"], "digest": snapshot["digest"]},
                               "extra_pnginfo": {"workflow": snapshot["workflow"]}},
            })
            if not isinstance(result, dict) or not isinstance(result.get("prompt_id"), str):
                raise StudioError("submission_unknown", "Driver acceptance did not return a prompt identity.", 503)
            self.store.update("jobs", job["id"], {"prompt_id": result["prompt_id"], "state": "queued", "accepted_at": timestamp()})
        except StudioError as exc:
            # HTTP refusal can be final, but transport failure may follow acceptance.
            current = self.store.get("jobs", job["id"])
            self.store.update("jobs", job["id"], {"state": "unknown" if current["state"] == "submitting" else current["state"],
                                                  "error": {"code": exc.code, "message": str(exc)}})

    async def reconcile(self) -> None:
        async with self.observation_lock:
            await self._reconcile()

    async def _reconcile(self) -> None:
        try:
            queue = await self.client.request("GET", "/queue")
            history = await self.client.request("GET", "/history")
            if not isinstance(queue, dict) or not isinstance(history, dict):
                raise StudioError("driver_schema", "Driver queue snapshot is malformed.", 502)
            self.driver = {"state": "connected", "timestamp": timestamp()}
            known = {str(job["prompt_id"]): job for job in self.store.all("jobs") if job.get("prompt_id")}
            lookup = getattr(self.client, "history_prompt", None)
            if callable(lookup):
                active = {entry[1] for key in ("queue_running", "queue_pending")
                          for entry in (queue.get(key, []) if isinstance(queue.get(key), list) else [])
                          if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], str)}
                missing = [key for key, job in known.items() if key not in history and key not in active
                           and (job["state"] not in TERMINAL_STATES or "reconciliation" not in job)]
                offset = self.history_cursor % len(missing) if missing else 0
                for key in (missing[offset:] + missing[:offset])[:32]:
                    try:
                        older = await lookup(key)
                        if isinstance(older, dict) and key in older:
                            history[key] = older[key]
                    except StudioError:
                        pass
                self.history_cursor = offset + 32
            for state, rows in (("running", queue.get("queue_running", [])), ("queued", queue.get("queue_pending", []))):
                for entry in rows if isinstance(rows, list) else []:
                    job = self._discover(entry, known)
                    if job:
                        self.store.update("jobs", job["id"], {"state": state, "prompt_id": entry[1]})
            for prompt_id, value in history.items():
                if not isinstance(value, dict):
                    continue
                job = self._discover(value.get("prompt", []), known)
                if job is None:
                    job = known.get(prompt_id)
                if not job:
                    continue
                status = value.get("status", {})
                if not isinstance(status, dict):
                    continue
                messages = status.get("messages", [])
                if not isinstance(messages, list):
                    messages = []
                types = {row[0] for row in messages if isinstance(row, list) and row and isinstance(row[0], str)}
                state = "cancelled" if "execution_interrupted" in types else "failed" if status.get("status_str") == "error" else (
                    "succeeded" if status.get("completed") is True else "unknown")
                for message in messages:
                    if isinstance(message, (list, tuple)) and len(message) == 2 and isinstance(message[1], dict):
                        self.consume_event({"type": message[0], "data": message[1]}, known=known)
                self.store.update("jobs", job["id"], {"state": state, "prompt_id": prompt_id,
                                                      "backend_status": status, "progress": None,
                                                      "reconciliation": {"source": "driver_history", "digest": digest(value)}})
                self._artifacts(job["id"], value.get("outputs", {}))
            await self.runtime_events.poll()
            self.telemetry["runtime_events"] = self.runtime_events.status
            try:
                sample = await self.client.request("GET", "/dgxm/telemetry")
                self.telemetry = {"sample": sample, "observed_at": timestamp(), "state": "connected",
                                  "runtime_events": self.runtime_events.status}
            except StudioError:
                self.telemetry = {**self.telemetry, "state": "stale" if self.telemetry else "unavailable"}
        except StudioError:
            self.driver = {**self.driver, "state": "unavailable"}
            self.telemetry = {**self.telemetry, "state": "stale" if self.telemetry else "unavailable"}

    def _discover(self, entry: Any, known: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(entry, (list, tuple)) or len(entry) < 4 or not isinstance(entry[1], str):
            return None
        if entry[1] in known:
            return known[entry[1]]
        extra = entry[3] if isinstance(entry[3], dict) else {}
        studio = extra.get("studio")
        studio = studio if isinstance(studio, dict) else {}
        studio_id = studio.get("job_id")
        if isinstance(studio_id, str):
            try:
                job = self.store.get("jobs", studio_id)
                if studio.get("digest") == job["submission"].get("digest") and entry[2] == job["submission"].get("prompt"):
                    known[entry[1]] = job
                    return job
            except StudioError:
                pass
        png_info = extra.get("extra_pnginfo")
        png_info = png_info if isinstance(png_info, dict) else {}
        job = {"id": identifier(), "state": "unknown", "prompt_id": entry[1], "created_at": timestamp(),
               "updated_at": timestamp(), "external": True,
               "submission": {"prompt": entry[2], "workflow": png_info.get("workflow"),
                              "requested": None, "effective": {"prompt": entry[2]},
                              "provenance_status": "imported_driver_snapshot", "versions": {}, "parent_artifact_ids": []}}
        self.store.create("jobs", job)
        known[entry[1]] = job
        return job

    def _artifacts(self, job_id: str, outputs: Any) -> None:
        if not isinstance(outputs, dict):
            return
        job = self.store.get("jobs", job_id)
        submission = job["submission"]
        requested = submission.get("requested") or {}
        for node_id, output in outputs.items():
            if not isinstance(output, dict):
                continue
            for key in ("images", "gifs", "videos"):
                for value in output.get(key, []) if isinstance(output.get(key), list) else []:
                    if not isinstance(value, dict):
                        continue
                    try:
                        descriptor = safe_descriptor(value)
                    except StudioError:
                        continue
                    suffix = PurePosixPath(descriptor["filename"]).suffix.lower()
                    if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov"}:
                        continue
                    object_id = digest([job_id, node_id, descriptor])
                    artifact = {"id": object_id, "job_id": job_id, "node_id": node_id,
                                "kind": "video" if suffix in {".mp4", ".webm", ".mov"} else "image", **descriptor,
                                "metadata": {key: value.get(key) for key in ("width", "height", "frame_rate", "duration", "format")},
                                "created_at": timestamp(), "intermediate": descriptor["type"] == "temp",
                                "operation": requested.get("operation", "workflow"), "project_id": submission.get("project_id"),
                                "status": job["state"], "model_ids": submission.get("model_identities", []),
                                "search_text": str(submission.get("effective", {})) + " " + descriptor["filename"],
                                "provenance": {"submission_digest": submission.get("digest"), "prompt_id": job.get("prompt_id"),
                                               "parent_artifact_ids": submission.get("parent_artifact_ids", []),
                                               "content_sha256": None, "source": "driver_history"}}
                    try:
                        self.store.create("artifacts", artifact)
                    except StudioError as exc:
                        if exc.code != "exists":
                            raise

    async def cancel(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("confirmed") is not True:
            raise StudioError("confirmation", "Explicit cancellation confirmation is required.", 409)
        await self.reconcile()
        job = self.store.get("jobs", job_id)
        if job["state"] in TERMINAL_STATES:
            return job
        scoped_cancel = getattr(self.client, "cancel_prompt", None)
        if not job.get("prompt_id") or job["state"] not in {"queued", "running", "cancel_requested"}:
            raise StudioError("cancellation_scope", "Only queued graph tasks can be cancelled safely; running driver interruption is global.", 409)
        if job["state"] != "queued" and not callable(scoped_cancel):
            raise StudioError("cancellation_scope", "Only queued graph tasks are supported by this driver adapter; atomic prompt cancellation is unavailable.", 409)
        self.store.update("jobs", job_id, {"state": "cancel_requested", "cancellation_scope":
                                          "atomic_prompt" if callable(scoped_cancel) else "queued_only"})
        try:
            if callable(scoped_cancel):
                receipt = await scoped_cancel(job["prompt_id"])
            else:
                await self.client.request("POST", "/queue", {"delete": [job["prompt_id"]]})
                receipt = {"deletion_requested": True}
            updates: dict[str, Any] = {"cancellation": {"prompt_id": job["prompt_id"], "receipt": receipt,
                                                        "observed_at": timestamp(), "completion": "awaiting_driver_evidence"}}
            if receipt.get("cancelled") is False:
                updates["state"] = "unknown"
                updates["cancellation"]["completion"] = "not_dispatched"
            self.store.update("jobs", job_id, updates)
        except StudioError as exc:
            self.store.update("jobs", job_id, {"state": "unknown", "cancellation": {"code": exc.code, "completion": "unknown"}})
            raise
        return self.store.get("jobs", job_id)

    def consume_event(self, message: dict[str, Any], *, known: dict[str, Any] | None = None) -> None:
        parsed = observation(message)
        if parsed is None:
            return
        event, updates = parsed
        prompt_id = event["data"]["prompt_id"]
        if known is None:
            known = {job["prompt_id"]: job for job in self.store.all("jobs") if job.get("prompt_id")}
        job = known.get(prompt_id)
        if job is None:
            return
        self.store.observe(job["id"], event, updates)

    async def watch_events(self) -> None:
        stream = getattr(self.client, "stream_events", None)
        if not callable(stream):
            return
        while True:
            try:
                async for message in stream(self.store.client_id):
                    async with self.observation_lock:
                        if message.get("type") == "preview_frame":
                            from .live_preview import accept_preview
                            await asyncio.to_thread(accept_preview, self.store, message)
                        else:
                            self.consume_event(message)
            except StudioError:
                pass
            await asyncio.sleep(2)

    def retry(self, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
        key_request = {"retry_of": job_id, **request}
        existing = self.store.request("jobs", request.get("idempotency_key"), key_request)
        if existing is not None:
            return existing
        old = self.store.get("jobs", job_id)
        if request.get("confirmed") is not True or old["state"] not in {"failed", "cancelled"}:
            raise StudioError("retry_scope", "Only a confirmed failed or cancelled graph task may be retried.", 409)
        if old.get("external"):
            raise StudioError("retry_provenance", "Reuse and review the imported workflow before resubmitting.", 409)
        value = {"id": identifier(), "state": "submitting", "submission": deepcopy(old["submission"]),
                 "retry_of": job_id, "prompt_id": None, "idempotency_key": request.get("idempotency_key"),
                 "created_at": timestamp(), "updated_at": timestamp()}
        value, created = self.store.reserve("jobs", request.get("idempotency_key"), key_request, value)
        if created:
            self.launch(self.dispatch(value))
        return value
