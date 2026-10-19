"""Durable exact-prompt joins for opt-in Monarch runtime evidence.

This journal never changes immutable submissions or Comfy graph terminal state.
Runtime outputs, refused/unknown outcomes and native owner ACKs are separate
facts. Late evidence may enrich a terminal graph without reviving it.
"""
from __future__ import annotations

import asyncio
import json
import math
from copy import deepcopy
from typing import Any

from ..execution_events import KINDS, MAX_EVENT_BYTES
from .contracts import StudioError, canonical, digest, timestamp
from .store import Store

RENDER_SUMMARY_LIMIT = 32
RENDER_SUMMARY_BYTES = 4096


def valid_event(event: Any) -> bool:
    if not isinstance(event, dict) or type(event.get("schema_version")) is not int or event.get("schema_version") != 1:
        return False
    if type(event.get("sequence")) is not int or event["sequence"] <= 0 or not isinstance(event.get("kind"), str) or event["kind"] not in KINDS:
        return False
    if any(not isinstance(event.get(key), str) or not event[key] or len(event[key]) > 512
           for key in ("producer_id", "prompt_id", "node_id", "render_id")):
        return False
    if (type(event.get("timestamp")) not in (int, float) or not 0 <= event["timestamp"] <= 1e12 or not math.isfinite(event["timestamp"])
            or any(event.get(key) is not None and (type(event[key]) is not int or event[key] < 0)
                   for key in ("list_index", "fleet_index"))):
        return False
    if not isinstance(event.get("facts"), dict):
        return False
    facts = event["facts"]
    if event["kind"] == "worker_results" and (
            not isinstance(facts.get("workers"), list)
            or any(not isinstance(row, dict) for row in facts["workers"])
            or not isinstance(facts.get("authority"), str) or facts["authority"] not in {"validated_render_results", "validated_fleet_output"}):
        return False
    if event["kind"] == "native_ack":
        token, statuses = facts.get("owner_token_sha256"), facts.get("statuses")
        if (facts.get("authority") != "validated_full_world_owner_ack"
                or type(facts.get("setup_generation")) is not int or facts["setup_generation"] < 1
                or not isinstance(token, str) or len(token) != 64 or any(char not in "0123456789abcdef" for char in token)
                or not isinstance(statuses, list) or not statuses
                or any(not isinstance(status, str) or status not in {"released", "already_released", "unknown"} for status in statuses)
                or sum(status in {"released", "already_released"} for status in statuses) != 1):
            return False
    try:
        return len(canonical(event).encode()) <= MAX_EVENT_BYTES + 128
    except (TypeError, ValueError):
        return False


def record_event(store: Store, event: dict, job_id: str | None) -> None:
    """Deduplicate the journal row and materialized evidence in one transaction."""
    if not valid_event(event):
        raise ValueError("Malformed runtime event")
    key = digest([event["producer_id"], event["sequence"]])
    with store.lock, store.transaction():
        row = store.db.execute("SELECT document FROM objects WHERE kind='runtime_events' AND id=?", (key,)).fetchone()
        existing = json.loads(row[0]) if row else None
        if existing and existing["event"] != event:
            raise ValueError("Runtime event identity has conflicting payloads")
        if existing and existing.get("job_id") is not None:
            return
        value = {"id": key, "event": event, "job_id": job_id,
                 "created_at": existing["created_at"] if existing else timestamp()}
        store._put("runtime_events", value, 1)
        if job_id is None:
            return
        job = store.get("jobs", job_id)
        if job.get("prompt_id") != event["prompt_id"]:
            raise ValueError("Runtime observation prompt identity mismatch")
        state = deepcopy(job.get("monarch", {"schema_version": 1, "renders": {}, "event_count": 0, "watermarks": {},
                                               "render_count": 0, "render_status_counts": {}}))
        render_key = digest([job_id, event["producer_id"], event["render_id"], event["node_id"], event.get("list_index")])
        old = store.db.execute("SELECT document FROM objects WHERE kind='runtime_renders' AND id=?", (render_key,)).fetchone()
        render = json.loads(old[0]) if old else {
            "id": render_key, "job_id": job_id, "render_id": event["render_id"], "producer_id": event["producer_id"],
            "node_id": event["node_id"], "list_index": event.get("list_index"), "fleet_index": event.get("fleet_index"),
            "status": "unknown", "acks": [], "created_at": timestamp()}
        old_status = render["status"] if old else None
        state["event_count"] += 1
        _watermark(store, state, job_id, event)
        _apply(render, event)
        if event["sequence"] > render.get("last_sequence", 0):
            render["last_sequence"] = event["sequence"]
            render["observed_at"] = event["timestamp"]
        store._put("runtime_renders", render, 1)
        if old is None:
            state["render_count"] += 1
        counts = state["render_status_counts"]
        if old_status != render["status"]:
            if old_status is not None:
                counts[old_status] -= 1
            counts[render["status"]] = counts.get(render["status"], 0) + 1
        state["renders"][render_key] = _summary(render)
        while len(state["renders"]) > RENDER_SUMMARY_LIMIT:
            oldest = min(state["renders"], key=lambda key: state["renders"][key].get("observed_at", 0))
            del state["renders"][oldest]
        state["omitted_render_count"] = state["render_count"] - len(state["renders"])
        job["monarch"] = state
        job["updated_at"] = timestamp()
        store._put("jobs", job, 1)
        store._event("jobs.runtime", job_id, {"render_id": event["render_id"], "kind": event["kind"],
                                             "sequence": event["sequence"], "producer_id": event["producer_id"]})


def _watermark(store: Store, state: dict, job_id: str, event: dict) -> None:
    producer, sequence = event["producer_id"], event["sequence"]
    key = digest([job_id, producer])
    old = store.db.execute("SELECT document FROM objects WHERE kind='runtime_job_watermarks' AND id=?", (key,)).fetchone()
    previous = json.loads(old[0])["sequence"] if old else 0
    value = {"id": key, "job_id": job_id, "producer_id": producer, "sequence": max(sequence, previous)}
    store._put("runtime_job_watermarks", value, 1)
    state["watermarks"][producer] = value["sequence"]
    if len(state["watermarks"]) > RENDER_SUMMARY_LIMIT:
        del state["watermarks"][next(key for key in state["watermarks"] if key != producer)]
        state["watermarks_truncated"] = True


def _summary(render: dict) -> dict:
    keys = ("id", "render_id", "producer_id", "node_id", "list_index", "fleet_index", "status", "last_sequence", "observed_at")
    summary = {key: render[key] for key in keys if key in render}
    summary.update(worker_count=len(render.get("workers", [])), ack_count=len(render["acks"]), details_available=True)
    # Keep common small render evidence inline. Oversized arrays remain in the
    # durable per-render record and event pages, never in each snapshot.
    for key in ("admission", "effective_topology", "setup_generation", "attention", "attention_source", "worker_acceptance",
                "output_validated", "completed_at", "worker_result_authority", "config_fingerprint",
                "failure", "workers", "acks", "model_fingerprints", "worker_runtime"):
        if key not in render:
            continue
        candidate = {**summary, key: render[key]}
        if len(canonical(candidate).encode()) <= RENDER_SUMMARY_BYTES - 64:
            summary = candidate
        else:
            summary["details_truncated"] = True
    return summary


def _apply(render: dict, event: dict) -> None:
    kind, facts = event["kind"], event["facts"]
    newer = event["sequence"] > render.get("last_sequence", 0)
    if kind == "dispatch_submitted":
        render.update(admission=facts.get("admission"), effective_topology=facts.get("effective_topology"),
                      setup_generation=facts.get("setup_generation"), attention=facts.get("attention"),
                      attention_source=facts.get("attention_source"),
                      model_fingerprints=facts.get("model_fingerprints"), worker_runtime=facts.get("worker_runtime"),
                      config_fingerprint=facts.get("config_fingerprint"))
        render.setdefault("worker_acceptance", facts.get("worker_acceptance"))
        if newer and render["status"] == "unknown":
            render["status"] = "submitted"
    elif kind == "worker_results" and event["sequence"] > render.get("worker_sequence", 0):
        render["worker_sequence"] = event["sequence"]
        render.update(workers=facts.get("workers", []), worker_result_authority=facts.get("authority"),
                      worker_acceptance="validated_result_observed")
    elif kind == "native_ack":
        token = facts.get("owner_token_sha256")
        if facts.get("authority") == "validated_full_world_owner_ack" and token and not any(
                ack.get("owner_token_sha256") == token for ack in render["acks"]):
            render["acks"].append(facts)
    elif kind == "result_validated":
        render["output_validated"] = True
        if newer and render["status"] != "completed":
            render["status"] = "output_validated"
    elif kind in {"render_completed", "fleet_item_completed"}:
        if facts.get("outcome") == "validated_output_and_retired_lease":
            render.update(status="completed", completed_at=event["timestamp"])
    elif newer and kind in {"render_failed", "fleet_item_failed", "submission_failed"}:
        render["failure"] = facts
        if render["status"] != "completed":
            render["status"] = "refused" if facts.get("worker_refusal_verified") is True else "unknown"


class RuntimeObservations:
    def __init__(self, store: Store, client: Any):
        self.store, self.client = store, client
        self.status: dict[str, Any] = {"state": "unavailable", "reason": "not_observed"}
        self._lock = asyncio.Lock()
        self.cursor_id = digest(["monarch_runtime", getattr(client, "origin", "configured_driver")])
        with store.lock:
            store.db.execute("CREATE INDEX IF NOT EXISTS runtime_pending_prompt ON objects "
                             "(json_extract(document,'$.event.prompt_id')) WHERE kind='runtime_events' "
                             "AND json_extract(document,'$.job_id') IS NULL")
            store.db.execute("CREATE INDEX IF NOT EXISTS runtime_jobs_prompt ON objects "
                             "(json_extract(document,'$.prompt_id')) WHERE kind='jobs'")

    def _cursor(self) -> dict:
        try:
            return self.store.get("runtime_cursors", self.cursor_id)
        except StudioError:
            return {"id": self.cursor_id, "sequence": 0, "producer_id": None}

    async def poll(self) -> None:
        async with self._lock:
            await self._poll()

    async def _poll(self) -> None:
        fetch = getattr(self.client, "execution_events", None)
        if not callable(fetch):
            return
        cursor = self._cursor()
        try:
            for _page in range(8):
                page = await fetch(cursor["sequence"], cursor.get("producer_id"))
                if (not isinstance(page, dict) or page.get("schema_version") != 1
                        or not isinstance(page.get("events"), list) or not isinstance(page.get("producer_id"), str)
                        or not 0 < len(page["producer_id"]) <= 128
                        or type(page.get("next_sequence")) is not int):
                    raise ValueError("Malformed runtime event replay")
                previous = cursor["sequence"] if cursor.get("producer_id") == page["producer_id"] else 0
                if page["next_sequence"] < previous or len(page["events"]) > 128:
                    raise ValueError("Runtime replay cursor regressed")
                known = {job["prompt_id"]: job["id"] for job in self.store.all("jobs") if job.get("prompt_id")}
                sequence_gap = False
                for event in page["events"]:
                    if not valid_event(event) or event["producer_id"] != page["producer_id"]:
                        raise ValueError("Malformed runtime event")
                    if event["sequence"] <= previous or event["sequence"] > page["next_sequence"]:
                        raise ValueError("Runtime events are unordered")
                    sequence_gap = sequence_gap or event["sequence"] != previous + 1
                    previous = event["sequence"]
                    record_event(self.store, event, known.get(event["prompt_id"]))
                sequence_gap = sequence_gap or page["next_sequence"] != previous
                omitted = page.get("omitted_events", 0)
                if type(omitted) is not int or omitted < 0:
                    raise ValueError("Malformed omitted event count")
                coverage = page.get("coverage_epoch", 0)
                if type(coverage) is not int or coverage < 0:
                    raise ValueError("Malformed observation coverage epoch")
                missing = (bool(page.get("gap")) or sequence_gap or omitted > cursor.get("omitted_events", 0)
                           or ("coverage_epoch" in cursor and coverage != cursor["coverage_epoch"]))
                if missing:
                    gap = {"id": digest([self.cursor_id, cursor, page["producer_id"], page["next_sequence"]]),
                           "previous": cursor, "producer_id": page["producer_id"], "created_at": timestamp(),
                           "oldest_sequence": page.get("oldest_sequence"), "omitted_events": omitted, "reason": "observation_coverage_gap"}
                    try:
                        self.store.create("runtime_gaps", gap)
                    except StudioError as exc:
                        if exc.code != "exists":
                            raise
                cursor = {"id": self.cursor_id, "sequence": page["next_sequence"], "producer_id": page["producer_id"], "omitted_events": omitted,
                          "has_gaps": bool(cursor.get("has_gaps")) or missing, "coverage_epoch": coverage}
                with self.store.lock, self.store.transaction():
                    self.store._put("runtime_cursors", cursor, 1)
                self.status = {"state": "connected", "observed_at": timestamp(), "cursor": cursor,
                               "gap": missing, "has_gaps": cursor["has_gaps"], "capture": page.get("capture"),
                               "context_available": page.get("context_available"), "coverage_started_at": page.get("coverage_started_at")}
                if not page.get("more", len(page["events"]) >= 128):
                    break
            self._join_pending()
        except (StudioError, ValueError, TypeError):
            self.status = {**self.status, "state": "unavailable", "reason": "runtime_event_route_unavailable_or_invalid"}

    def _join_pending(self) -> None:
        # Join only matching pending identities. Partial indexes prevent old
        # unrelated records from starving joins or requiring per-job scans.
        for _batch in range(8):
            with self.store.lock:
                rows = self.store.db.execute(
                    "SELECT event.document, job.id FROM objects AS event JOIN objects AS job "
                    "ON job.kind='jobs' AND json_extract(job.document,'$.prompt_id')="
                    "json_extract(event.document,'$.event.prompt_id') WHERE event.kind='runtime_events' "
                    "AND json_extract(event.document,'$.job_id') IS NULL ORDER BY event.updated LIMIT 512").fetchall()
            for row in rows:
                record_event(self.store, json.loads(row[0])["event"], row[1])
            if len(rows) < 512:
                return
