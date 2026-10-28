"""Bounded list projections; full provenance remains available by identity."""
from __future__ import annotations

import json
from typing import Any

from .contracts import RECORD_KINDS, canonical
from .grid_views import projection as grid_projection
from .store import Store

PAGE_BYTES = 8 * 1024 * 1024
# A browser reads a JSON number as a double, which holds whole numbers exactly only up to
# 2**53 - 1, its Number.MAX_SAFE_INTEGER. ComfyUI seeds reach 2**64 - 1.
SAFE_INTEGER = 2**53 - 1
# An imported job has no frozen seed list, so the projection hands decode_row the inputs of
# each prompt node with an integer seed input. An object extracts as its exact text, while a
# json_each or scalar value turns an integer above 2**63 into a rounded real. Only object nodes
# are read, since a JSON function refuses a node that is plain text.
PROMPT_SEED_INPUTS = """(SELECT json_group_object(key,json_extract(node,'$.inputs'))
    FROM (SELECT key, CASE WHEN type = 'object' THEN value END AS node
          FROM json_each(document,'$.submission.prompt'))
    WHERE json_type(node,'$.inputs.seed') = 'integer' OR json_type(node,'$.inputs.noise_seed') = 'integer')"""


def _object(fields: tuple[str, ...], prefix: str = "$") -> str:
    return "json_object(" + ",".join(f"'{key}',json_extract(document,'{prefix}.{key}')" for key in fields) + ")"


def projection(kind: str) -> str:
    """SQL only projects fixed fields, avoiding decoding large graph payloads."""
    if kind == "grid_runs":
        return grid_projection()
    if kind == "jobs":
        # started_at and ended_at are both driver-clock times from ComfyUI's events, so
        # their difference is the driver's own measure of a job's run time.
        # running_observed_at is the companion's clock, like server_time, so a
        # running job's elapsed time never mixes the two hosts' clocks.
        base = _object(("id", "state", "created_at", "updated_at", "accepted_at", "prompt_id", "idempotency_key",
                        "progress", "error", "cancellation", "cancellation_scope", "external", "retry_of",
                        "experiment_id", "cell_index", "preview_artifact_id", "started_at", "ended_at",
                        "running_observed_at"))
        submission = _object(("project_id", "versions", "model_identities", "digest"), "$.submission")
        requested = _object(("operation", "project_id"), "$.submission.requested")
        # The seeds the driver ran with, at the path the full record uses. decode_row fills an
        # imported job's from its prompt and gives each the form a browser reads exactly.
        effective = ("json_object('seeds',json_extract(document,'$.submission.effective.seeds'),'prompt_seeds',"
                     "CASE WHEN json_type(document,'$.submission.effective.seeds') IS NULL "
                     f"THEN {PROMPT_SEED_INPUTS} END)")
        monarch = _object(("schema_version", "event_count", "render_count", "render_status_counts",
                           "omitted_render_count"), "$.monarch")
        renders = """(SELECT COALESCE(json_group_object(key,json_object(
            'fleet_index',json_extract(value,'$.fleet_index'),'status',json_extract(value,'$.status'))),'{}')
            FROM json_each(document,'$.monarch.renders'))"""
        return (f"json_set({base},'$.summary',json('true'),'$.submission',"
                f"json_set({submission},'$.requested',{requested},'$.effective',{effective}),'$.monarch',"
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


def browser_seeds(seeds: Any) -> Any:
    """Seeds a browser reads exactly: a JSON number up to 2**53 - 1, its decimal digits as a string above."""
    if not isinstance(seeds, dict):
        return seeds
    return {key: str(seed) if type(seed) is int and abs(seed) > SAFE_INTEGER else seed for key, seed in seeds.items()}


def prompt_seeds(nodes: Any) -> dict[str, int]:
    """The integer seed and noise_seed inputs of a prompt, keyed as build_preview keys effective seeds."""
    if not isinstance(nodes, dict):
        return {}
    return {f"{node_id}.{name}": inputs[name] for node_id, inputs in nodes.items() if isinstance(inputs, dict)
            for name in ("seed", "noise_seed") if type(inputs.get(name)) is int}


def decode_row(kind: str, text: str) -> dict[str, Any]:
    """One projected row; a job summary's recorded seeds keep every digit on the wire."""
    value: dict[str, Any] = json.loads(text)
    if kind == "jobs":
        effective = value["submission"]["effective"]
        nodes = effective.pop("prompt_seeds")
        seeds = effective["seeds"] if effective["seeds"] is not None else prompt_seeds(nodes)
        effective["seeds"] = browser_seeds(seeds)
    return value


def snapshot_rows(store: Store) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    """Share a byte budget across compact windows, prioritizing unresolved jobs."""
    rows, omitted, _ = snapshot_window(store)
    return rows, omitted


def snapshot_window(store: Store) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], int]:
    """Return the compact rows, the omitted counts and the event cursor they reflect.

    The cursor is read under the same lock as the rows, so a client that resumes
    from it neither skips an event the rows lack nor decodes any event payload.
    """
    result: dict[str, list[dict[str, Any]]] = {}
    omitted: dict[str, int] = {}
    remaining = PAGE_BYTES // 2  # Leave half for driver/telemetry and the response envelope.
    with store.lock:
        cursor = store.db.execute("SELECT COALESCE(MAX(cursor),0) FROM events").fetchone()[0]
        for kind in ("jobs", "operations", *RECORD_KINDS, "artifacts"):
            count = store.db.execute("SELECT count(*) FROM objects WHERE kind=?", (kind,)).fetchone()[0]
            priority = ("CASE WHEN json_extract(document,'$.state') IN "
                        "('submitting','accepted','queued','running','cancel_requested','unknown') THEN 0 ELSE 1 END,"
                        if kind == "jobs" else "")
            rows = store.db.execute(f"SELECT {projection(kind)} FROM objects WHERE kind=? "  # noqa: S608
                                    f"ORDER BY {priority}updated DESC,id LIMIT 200", (kind,))
            result[kind] = []
            for row in rows:
                value = decode_row(kind, row[0])
                size = len(canonical(value).encode()) + 1
                if size > remaining:
                    continue
                result[kind].append(value)
                remaining -= size
            if count > len(result[kind]):
                omitted[kind] = count - len(result[kind])
    return result, omitted, int(cursor)


def telemetry_summary(value: dict[str, Any]) -> dict[str, Any]:
    if len(canonical(value).encode()) <= PAGE_BYTES // 4:
        return value
    return {key: val for key, val in value.items() if key != "sample"} | {
        "sample_omitted": "Driver telemetry exceeds the 2 MiB snapshot budget."}
