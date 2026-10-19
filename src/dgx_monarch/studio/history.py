"""Bounded, durable keyset pagination for jobs and generated or uploaded media."""
from __future__ import annotations

import base64
import json
import math
from typing import Any

from .contracts import StudioError, canonical, digest
from .store import Store
from .wire_records import PAGE_BYTES, projection


def _page(store: Store, kind: str, query: dict[str, Any]) -> dict[str, Any]:
    allowed = {"limit", "cursor", "sort", "q", "operation", "project_id", "model", "status", "kind",
               "after", "before", "width", "height", "intermediate", "job_id"}
    if set(query) - allowed:
        raise StudioError("history_filter", "Unknown history filter.")
    try:
        limit = int(query.get("limit", 50))
    except (TypeError, ValueError) as exc:
        raise StudioError("history_limit", "Page size must be 1-200.") from exc
    if not 1 <= limit <= 200:
        raise StudioError("history_limit", "Page size must be 1-200.")
    order = query.get("sort", "newest")
    if order not in {"newest", "oldest"}:
        raise StudioError("history_sort", "Sort must be newest or oldest.")
    fingerprint = digest({key: val for key, val in query.items() if key not in {"cursor", "limit"}})
    clauses = ["kind=?"]
    params: list[Any] = [kind]
    created = "COALESCE(json_extract(document,'$.created_at'),0)"
    fields = {"project_id": "$.project_id", "operation": "$.operation", "status": "$.status",
              "kind": "$.kind", "job_id": "$.job_id", "width": "$.metadata.width", "height": "$.metadata.height"}
    if kind == "jobs":
        fields.update(project_id="$.submission.project_id", operation="$.submission.requested.operation", status="$.state")
    for key, path in fields.items():
        if query.get(key) not in (None, ""):
            value = str(query[key])
            if len(value) > 256:
                raise StudioError("history_filter", "Filter value is too long.")
            if key == "project_id" and kind == "artifacts":
                clauses.append("(EXISTS (SELECT 1 FROM json_each(json_extract(document,'$.project_ids')) WHERE value=?) "
                               "OR (json_extract(document,'$.project_ids') IS NULL AND json_extract(document,'$.project_id')=?))")
                params.extend([value, value])
            else:
                clauses.append(f"CAST(json_extract(document,'{path}') AS TEXT)=?")
                params.append(value)
    for key, operator in (("after", ">="), ("before", "<=")):
        if query.get(key) not in (None, ""):
            try:
                bound = float(query[key])
            except (TypeError, ValueError) as exc:
                raise StudioError("history_date", "Date bounds must be Unix timestamps.") from exc
            if not math.isfinite(bound):
                raise StudioError("history_date", "Date bounds must be finite.")
            clauses.append(f"{created}{operator}?")
            params.append(bound)
    if query.get("intermediate") not in (None, ""):
        if str(query["intermediate"]).lower() not in {"true", "false"}:
            raise StudioError("history_filter", "Intermediate filter must be true or false.")
        clauses.append("COALESCE(json_extract(document,'$.intermediate'),0)=?")
        params.append(int(str(query["intermediate"]).lower() == "true"))
    for key in ("q", "model"):
        if query.get(key):
            value = str(query[key])
            if len(value) > 512:
                raise StudioError("history_filter", "Search is too long.")
            path = "$.search_text" if key == "q" else "$.model_ids"
            if kind == "jobs":
                path = "$.submission" if key == "q" else "$.submission.model_identities"
            clauses.append(f"instr(lower(COALESCE(json_extract(document,'{path}'),'')),lower(?))>0")
            params.append(value)
    if query.get("cursor"):
        try:
            encoded = query["cursor"]
            if not isinstance(encoded, str) or len(encoded) > 2048:
                raise ValueError
            cursor = json.loads(base64.urlsafe_b64decode(encoded))
            if cursor["filter"] != fingerprint or not isinstance(cursor["id"], str) or not math.isfinite(cursor["time"]):
                raise ValueError
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise StudioError("history_cursor", "Invalid cursor or changed filters; restart pagination.") from exc
        op = "<" if order == "newest" else ">"
        clauses.append(f"({created}{op}? OR ({created}=? AND id{op}?))")
        params.extend([cursor["time"], cursor["time"], cursor["id"]])
    direction = "DESC" if order == "newest" else "ASC"
    with store.lock:
        # All SQL fragments are constants selected above; user values are bound parameters.
        rows = store.db.execute(f"SELECT {projection(kind)} FROM objects WHERE {' AND '.join(clauses)} "  # noqa: S608
                                f"ORDER BY {created} {direction},id {direction} LIMIT ?", [*params, limit + 1])
        items: list[dict[str, Any]] = []
        byte_count = 4096  # Response envelope and cursor.
        more = False
        for row in rows:
            value = json.loads(row[0])
            size = len(canonical(value).encode()) + 1
            if len(items) == limit or byte_count + size > PAGE_BYTES:
                if not items:
                    raise StudioError("history_size", "One history record exceeds the 8 MiB page budget.", 413)
                more = True
                break
            items.append(value)
            byte_count += size
    next_cursor = None
    if more:
        last = items[-1]
        next_cursor = base64.urlsafe_b64encode(canonical({"filter": fingerprint, "time": last.get("created_at") or 0,
                                                        "id": last["id"]}).encode()).decode()
    return {"items": items, "next_cursor": next_cursor}


def page_artifacts(store: Store, query: dict[str, Any]) -> dict[str, Any]:
    return _page(store, "artifacts", query)


def page_jobs(store: Store, query: dict[str, Any]) -> dict[str, Any]:
    return _page(store, "jobs", query)


def page_grids(store: Store, query: dict[str, Any]) -> dict[str, Any]:
    from .grid_views import live_view

    if set(query) - {"limit", "cursor", "sort"}:
        raise StudioError("history_filter", "Unknown grid history filter.")
    result = _page(store, "grid_runs", query)
    items: list[dict[str, Any]] = []
    for value in result["items"]:
        live = live_view(store, value)
        items.append({key: live[key] for key in ("id", "created_at", "updated_at", "state", "summary")} |
                     {"cell_count": len(live["cells"])})
    return {**result, "items": items}
