"""Bind library experiments to durable grids without restoring executable jobs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import StudioError, bounded_key, digest, object_value, timestamp
from .store import Store


def bind_grid_record(store: Store, data: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(data)
    identity = result.get("grid_id")
    if not identity:
        if "grid_id" in result:
            result["grid_source"] = {"state": "unlinked"}
        return result
    bounded_key(identity, "grid id")
    try:
        grid = store.get("grid_runs", identity)
    except StudioError as exc:
        if exc.code != "not_found":
            raise
        result["grid_source"] = {"state": "unavailable", "grid_id": identity, "reason": "source_grid_missing"}
        return result
    submitted = object_value(grid.get("submission"), "grid submission")
    definition = deepcopy(object_value(submitted.get("request"), "grid request"))
    cells = deepcopy(grid["cells"])
    for cell in cells:
        if not cell.get("job_id"):
            continue
        job = store.get("jobs", cell["job_id"])
        cell.update(state=job["state"], prompt_id=job.get("prompt_id"),
                    submission_digest=job["submission"]["digest"])
    result.update(grid_request=definition,
                  grid_source={"state": "available", "grid_id": identity, "definition_digest": digest(definition)},
                  grid_snapshot={"id": identity, "cells": cells, "observed_at": timestamp(),
                                 "preview_digest": submitted.get("digest"),
                                 "created_at": grid.get("created_at")})
    return result
