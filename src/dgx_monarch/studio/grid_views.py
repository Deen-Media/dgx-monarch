"""Compact grid views with live job outcomes and explicit full-record reads."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .contracts import StudioError, canonical
from .store import Store

GRID_BYTES = 32 * 1024 * 1024
VIEW_BYTES = 8 * 1024 * 1024
GRID_FIELDS = ("id", "state", "digest", "created_at", "updated_at", "expires_at", "ready", "total", "job_count", "preview_id")
CELL_FIELDS = ("index", "coordinates", "excluded", "errors", "preview_id", "job_id", "state", "attempts",
               "artifact_ids", "submission_digest")


def summary(value: dict[str, Any]) -> dict[str, Any]:
    return {**{key: value[key] for key in GRID_FIELDS if key in value}, "summary": True,
            "cells": [{key: cell[key] for key in CELL_FIELDS if key in cell} for cell in value.get("cells", [])]}


def projection() -> str:
    fields = ",".join(f"'{key}',json_extract(document,'$.{key}')" for key in GRID_FIELDS)
    cells = ",".join(f"'{key}',json_extract(value,'$.{key}')" for key in CELL_FIELDS)
    # Both interpolations are closed field lists above; no request input enters SQL.
    return (f"json_object({fields},'summary',json('true'),'cells',json((SELECT "  # noqa: S608
            f"COALESCE(json_group_array(json_object({cells})),'[]') FROM json_each(document,'$.cells'))))")


def read(store: Store, kind: str, identity: str, *, compact: bool) -> dict[str, Any]:
    if not compact:
        return store.get(kind, identity)
    with store.lock:
        row = store.db.execute(f"SELECT {projection()} FROM objects WHERE kind=? AND id=?", (kind, identity)).fetchone()  # noqa: S608
    if row is None:
        raise StudioError("not_found", "Grid not found.", 404)
    value = json.loads(row[0])
    if len(canonical(value).encode()) > VIEW_BYTES:
        raise StudioError("grid_view_limit", "Grid details exceed the 8 MiB display budget; export the full record.", 413)
    return value


def live_view(store: Store, value: dict[str, Any]) -> dict[str, Any]:
    cells = deepcopy(value["cells"])
    identities = list({identity for cell in cells for identity in [cell.get("job_id"), *(cell.get("attempts") or [])] if identity})
    jobs: dict[str, Any] = {}
    artifacts: dict[str, list[str]] = {}
    with store.lock:
        for start in range(0, len(identities), 256):
            batch = identities[start:start + 256]
            placeholders = ",".join("?" for _ in batch)
            for row in store.db.execute("SELECT id,json_extract(document,'$.state'),json_extract(document,'$.prompt_id'),"  # noqa: S608
                                        "json_extract(document,'$.retry_runtime') "
                                        f"FROM objects WHERE kind='jobs' AND id IN ({placeholders})", batch):
                jobs[row[0]] = {"state": row[1], "prompt_id": row[2], "retry_runtime": row[3]}
            for row in store.db.execute("SELECT id,json_extract(document,'$.job_id') FROM objects "  # noqa: S608
                                        f"WHERE kind='artifacts' AND json_extract(document,'$.job_id') IN ({placeholders})", batch):
                artifacts.setdefault(row[1], []).append(row[0])
    full = not value.get("summary")
    for cell in cells:
        if cell.get("excluded"):
            continue
        outcome = dict(jobs.get(cell.get("job_id"), {"state": "unknown", "prompt_id": None}))
        retried = outcome.pop("retry_runtime", None)
        if full and isinstance(retried, str):
            # A retried cell's manifest names the runtime that attempt ran against.
            cell["retry_runtime"] = json.loads(retried)
        cell.update(**outcome, artifact_ids=[identity for attempt in cell.get("attempts") or [] for identity in artifacts.get(attempt, [])])
    states = {cell["state"] for cell in cells if not cell.get("excluded")}
    state = ("unknown" if "unknown" in states else "running" if states & {"running", "cancel_requested"} else
             "submitting" if states & {"submitting", "accepted"} else "queued" if "queued" in states else
             "succeeded" if states == {"succeeded"} else "completed_with_failures")
    result = {**value, "cells": cells, "state": state}
    if result.get("summary") and len(canonical(result).encode()) > VIEW_BYTES:
        raise StudioError("grid_view_limit", "Grid details exceed the 8 MiB display budget; export the full record.", 413)
    return result
