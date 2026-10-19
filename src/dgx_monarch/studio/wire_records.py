"""Bounded list projections; full provenance remains available by identity."""
from __future__ import annotations

import json
from typing import Any

from .contracts import RECORD_KINDS, canonical
from .grid_views import projection as grid_projection
from .store import Store

PAGE_BYTES = 8 * 1024 * 1024


def _object(fields: tuple[str, ...], prefix: str = "$") -> str:
    return "json_object(" + ",".join(f"'{key}',json_extract(document,'{prefix}.{key}')" for key in fields) + ")"


def projection(kind: str) -> str:
    """SQL only projects fixed fields, avoiding decoding large graph payloads."""
    if kind == "grid_runs":
        return grid_projection()
    if kind == "jobs":
        base = _object(("id", "state", "created_at", "updated_at", "accepted_at", "prompt_id", "idempotency_key",
                        "progress", "error", "cancellation", "cancellation_scope", "external", "retry_of",
                        "experiment_id", "cell_index", "preview_artifact_id"))
        submission = _object(("project_id", "versions", "model_identities", "digest"), "$.submission")
        requested = _object(("operation", "project_id"), "$.submission.requested")
        monarch = _object(("schema_version", "event_count", "render_count", "render_status_counts",
                           "omitted_render_count"), "$.monarch")
        renders = """(SELECT COALESCE(json_group_object(key,json_object(
            'fleet_index',json_extract(value,'$.fleet_index'),'status',json_extract(value,'$.status'))),'{}')
            FROM json_each(document,'$.monarch.renders'))"""
        return (f"json_set({base},'$.summary',json('true'),'$.submission',"
                f"json_set({submission},'$.requested',{requested}),'$.monarch',"
                f"json_set({monarch},'$.renders',json({renders})))")
    if kind == "operations":
        return "json_set(" + _object(("id", "action", "state", "status", "created_at", "updated_at",
                                       "idempotency_key", "digest", "description", "available", "reason",
                                       "requires_confirmation")) + ",'$.summary',json('true'))"
    if kind == "artifacts":
        return "json_remove(document,'$.search_text')"
    if kind in RECORD_KINDS:
        return ("json_set(" + _object(("id", "revision", "project_id", "project_ids"))
                + ",'$.summary',json('true'),'$.data'," + _object(("name", "project_id", "project_ids"), "$.data") + ")")
    raise ValueError("Unsupported list projection")


def snapshot_rows(store: Store) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Share a byte budget across compact windows, prioritizing unresolved jobs."""
    result: dict[str, list[dict[str, Any]]] = {}
    omitted: dict[str, int] = {}
    remaining = PAGE_BYTES // 2  # Leave half for driver/telemetry and the response envelope.
    with store.lock:
        for kind in ("jobs", "operations", *RECORD_KINDS, "artifacts"):
            count = store.db.execute("SELECT count(*) FROM objects WHERE kind=?", (kind,)).fetchone()[0]
            priority = ("CASE WHEN json_extract(document,'$.state') IN "
                        "('submitting','accepted','queued','running','cancel_requested','unknown') THEN 0 ELSE 1 END,"
                        if kind == "jobs" else "")
            rows = store.db.execute(f"SELECT {projection(kind)} FROM objects WHERE kind=? "  # noqa: S608
                                    f"ORDER BY {priority}updated DESC,id LIMIT 200", (kind,))
            result[kind] = []
            for row in rows:
                value = json.loads(row[0])
                size = len(canonical(value).encode()) + 1
                if size > remaining:
                    continue
                result[kind].append(value)
                remaining -= size
            if count > len(result[kind]):
                omitted[kind] = count - len(result[kind])
    return result, omitted


def telemetry_summary(value: dict[str, Any]) -> dict[str, Any]:
    if len(canonical(value).encode()) <= PAGE_BYTES // 4:
        return value
    return {key: val for key, val in value.items() if key != "sample"} | {
        "sample_omitted": "Driver telemetry exceeds the 2 MiB snapshot budget."}
