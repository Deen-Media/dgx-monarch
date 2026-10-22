"""Bounded pages of durable per-render outcomes and exact runtime events."""
from __future__ import annotations

import base64
import json
import math
from typing import Any

from .contracts import StudioError, canonical
from .store import Store


def page_runtime(store: Store, job_id: str, query: dict[str, str]) -> dict[str, Any]:
    store.get("jobs", job_id)
    if set(query) - {"kind", "limit", "cursor"} or query.get("kind", "renders") not in {"renders", "events"}:
        raise StudioError("runtime_query", "Select renders or events and a bounded page size.")
    kind = "runtime_renders" if query.get("kind", "renders") == "renders" else "runtime_events"
    try:
        limit = int(query.get("limit", "40"))
    except ValueError as exc:
        raise StudioError("runtime_query", "Page size must be an integer.") from exc
    if not 1 <= limit <= 100:
        raise StudioError("runtime_query", "Page size must be between 1 and 100.")
    # Old journals did not store a first-observation timestamp. Keep them at
    # a stable zero epoch rather than sorting by mutable SQL updated time.
    created = "COALESCE(json_extract(document,'$.created_at'),0)"
    clauses, parameters = ["kind=?", "json_extract(document,'$.job_id')=?"], [kind, job_id]
    if query.get("cursor"):
        try:
            if len(query["cursor"]) > 2048:
                raise ValueError
            cursor = json.loads(base64.urlsafe_b64decode(query["cursor"]))
            if cursor["kind"] != kind or cursor["job"] != job_id or type(cursor["time"]) not in (int, float) or not math.isfinite(cursor["time"]) or not isinstance(cursor["id"], str):
                raise ValueError
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise StudioError("runtime_cursor", "Invalid runtime history cursor.") from exc
        clauses.append(f"({created}>? OR ({created}=? AND id>?))")
        parameters.extend([cursor["time"], cursor["time"], cursor["id"]])
    with store.lock:
        rows = store.db.execute(f"SELECT document,{created},id FROM objects WHERE " + " AND ".join(clauses) +  # noqa: S608
                                f" ORDER BY {created},id LIMIT ?", [*parameters, limit + 1]).fetchall()
    items = [json.loads(row[0]) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        # The cursor names the key SQLite sorted by. Its own parse of the
        # stored text can differ from Python's by an ulp, and a Python-side
        # key would serve the last row again or skip the next one.
        last = rows[limit - 1]
        next_cursor = base64.urlsafe_b64encode(canonical({"kind": kind, "job": job_id, "time": last[1], "id": last[2]}).encode()).decode()
    return {"items": items, "next_cursor": next_cursor}
