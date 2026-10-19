"""Durable grid expansion and graph submission through ComfyUI's existing queue."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import StudioError, canonical, digest, identifier, object_value, timestamp
from .drafts import DraftError, expand_grid
from .grid_views import GRID_BYTES, VIEW_BYTES, live_view, read, summary
from .jobs import Jobs
from .workflows import inspect_workflow


class GridExperiments:
    def __init__(self, jobs: Jobs):
        self.jobs, self.store = jobs, jobs.store

    async def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        draft = object_value(request.get("draft"), "draft")
        if draft.get("seed_policy", "fixed") != "fixed":
            raise StudioError("grid_seed", "Grid previews require resolved fixed or incrementing seeds.")
        schema = (await self.jobs.catalog())["object_info"]
        runtime = await self.jobs.capture_runtime()
        inspection = inspect_workflow(object_value(draft.get("workflow")), schema)
        bindings = {row["id"]: row for row in inspection["bindings"] if row["resolved"]}
        schemas = {key: row["schema"] for key, row in bindings.items()}
        policy = request.get("seed_policy", "fixed")
        seed_binding = request.get("seed_binding") if policy != "fixed" else None
        if not isinstance(policy, str) or (seed_binding is not None and not isinstance(seed_binding, str)):
            raise StudioError("grid_seed", "Seed policy and binding must be text.")
        if seed_binding is not None and (seed_binding not in bindings or bindings[seed_binding]["schema"].get("type") != "INT"
                                         or bindings[seed_binding]["input_name"] not in {"seed", "noise_seed"}):
            raise StudioError("grid_seed", "Seed binding must identify an editable integer seed input.")
        if policy == "increment" and not seed_binding:
            raise StudioError("grid_seed", "Incrementing grids require an explicit seed binding.")
        axes, excluded = request.get("axes", []), request.get("excluded", [])
        if not isinstance(axes, list) or any(not isinstance(axis, dict) for axis in axes) or not isinstance(excluded, list):
            raise StudioError("grid_axes", "Axes and exclusions must be arrays.")
        for axis in axes:
            if axis.get("name") == "styles":
                schemas["styles"] = {"type": "COMBO", "choices": axis.get("values", [])}
        try:
            expanded = expand_grid(object_value(draft.get("values", {})), axes, schemas, seed_policy=policy,
                                   base_seed=request.get("base_seed", 0) if policy != "fixed" else 0, excluded=excluded)
        except DraftError as exc:
            raise StudioError("grid_axes", str(exc)) from exc
        cells, previews, byte_count = [], [], 0
        style_cache: dict[str, dict[str, Any]] = {}
        axis_names = {axis["name"] for axis in axes}
        for cell in expanded["cells"]:
            row = {key: deepcopy(cell[key]) for key in ("index", "coordinates", "excluded", "errors")}
            values = {key: val for key, val in cell["parameters"].items() if key not in {"seed", "styles"}}
            if policy == "increment" and seed_binding and seed_binding not in axis_names:
                values[seed_binding] = cell["parameters"]["seed"]
            if not row["errors"]:
                try:
                    cell_draft = {**deepcopy(draft), "values": values}
                    if "styles" in axis_names:
                        selection = cell["parameters"]["styles"]
                        if (not isinstance(selection, list) or len(selection) > 64
                                or any(not isinstance(identity, str) or not identity for identity in selection)):
                            raise StudioError("grid_styles", "Each style-axis value must be an ordered list of up to 64 saved style IDs.")
                        records = []
                        for identity in selection:
                            if identity not in style_cache:
                                record = self.store.get("styles", identity)
                                style_cache[identity] = {"id": identity, "revision": record["revision"],
                                                         "data": deepcopy(record["data"]), "digest": digest(record["data"])}
                            records.append(deepcopy(style_cache[identity]))
                        cell_draft.update(style_records=records, styles=[{**row["data"], "id": row["id"]} for row in records])
                        row["style_records"] = records
                    preview = self.jobs.build_preview(cell_draft, schema=schema, runtime=runtime, persist=False)
                    byte_count += len(canonical(preview).encode())
                    if byte_count > GRID_BYTES:
                        raise StudioError("grid_size", "Expanded submissions exceed the 32 MiB grid budget.")
                    previews.append(preview)
                    row.update(preview_id=preview["id"], submission_digest=preview["digest"])
                except ValueError as exc:
                    if isinstance(exc, StudioError) and exc.code == "grid_size":
                        raise
                    row["errors"].append(str(exc))
            cells.append(row)
        value = {"id": identifier(), "request": deepcopy(request), "cells": cells, "total": expanded["total"],
                 "job_count": expanded["job_count"], "created_at": timestamp(), "expires_at": timestamp() + 600,
                 "schema_digest": digest(schema), "ready": bool(expanded["job_count"]) and all(
                     row["excluded"] or not row["errors"] for row in cells)}
        value["digest"] = digest(value)
        if byte_count + len(canonical(value).encode()) > GRID_BYTES or len(canonical(summary(value)).encode()) > VIEW_BYTES:
            raise StudioError("grid_size", "Expanded grid exceeds its 32 MiB storage or 8 MiB display budget.")
        with self.store.lock, self.store.transaction():
            for preview in previews:
                self.store._put("previews", preview, 1)
            self.store._put("grid_previews", value, 1)
        return value

    def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        existing = self.store.request("grid_runs", request.get("idempotency_key"), request)
        if existing is not None:
            return self.get(existing["id"])
        preview = self.store.get("grid_previews", request.get("preview_id", ""))
        if not preview["ready"] or preview["expires_at"] < timestamp():
            raise StudioError("grid_preview", "A current fully validated grid preview is required.", 409)
        if preview["schema_digest"] != digest(self.jobs.schema):
            raise StudioError("schema_changed", "Driver capabilities changed; preview the grid again.", 409)
        identity = identifier()
        cells, children = [], []
        for cell in preview["cells"]:
            row = deepcopy(cell)
            if not cell["excluded"]:
                submission = self.jobs.reviewed_snapshot(cell["preview_id"])
                child = self._child(identity, cell["index"], submission)
                children.append(child)
                row.update(job_id=child["id"], attempts=[child["id"]], state="submitting")
            else:
                row.update(job_id=None, attempts=[], state="excluded")
            cells.append(row)
        value = {"id": identity, "state": "submitting", "cells": cells, "preview_id": preview["id"],
                 "submission": deepcopy(preview), "created_at": timestamp(), "updated_at": timestamp()}
        value, created = self.store.reserve_group("grid_runs", request.get("idempotency_key"), request, value, children)
        if created:
            self.jobs.launch(self._dispatch(children))
        return value

    async def _dispatch(self, children: list[dict[str, Any]]) -> None:
        # This is bounded transport, not a scheduler: the driver owns admission and execution order.
        for child in children:
            await self.jobs.dispatch(child)

    @staticmethod
    def _child(parent: str, index: int, submission: dict[str, Any], retry_of: str | None = None) -> dict[str, Any]:
        return {"id": identifier(), "state": "submitting", "prompt_id": None, "submission": deepcopy(submission),
                "experiment_id": parent, "cell_index": index, "retry_of": retry_of,
                "created_at": timestamp(), "updated_at": timestamp()}

    def get(self, identity: str, *, compact: bool = False) -> dict[str, Any]:
        return live_view(self.store, read(self.store, "grid_runs", identity, compact=compact))

    def retry(self, identity: str, request: dict[str, Any]) -> dict[str, Any]:
        bound_request = {**request, "grid_id": identity}
        existing = self.store.request("grid_retries", request.get("idempotency_key"), bound_request)
        if existing is not None:
            return self.get(identity)
        indices = request.get("indices")
        if (request.get("confirmed") is not True or not isinstance(indices, list) or not indices
                or any(type(index) is not int for index in indices) or len(set(indices)) != len(indices)):
            raise StudioError("grid_retry", "Confirm a nonempty unique list of failed or cancelled cell indices.", 409)
        value = self.get(identity)
        cells = deepcopy(value["cells"])
        if any(index < 0 or index >= len(cells) for index in indices):
            raise StudioError("grid_retry", "Cell index is out of range.")
        children = []
        for index in indices:
            cell = cells[index]
            if cell["state"] not in {"failed", "cancelled"}:
                raise StudioError("grid_retry", "Only explicitly failed or cancelled cells can be retried.", 409)
            old = self.store.get("jobs", cell["job_id"])
            child = self._child(identity, index, old["submission"], old["id"])
            children.append(child)
            cell.update(job_id=child["id"], attempts=[*cell["attempts"], child["id"]], state="submitting")
        retry = {"id": identifier(), "grid_id": identity, "indices": indices, "job_ids": [row["id"] for row in children]}
        _, created = self.store.reserve_group("grid_retries", request.get("idempotency_key"), bound_request, retry, children,
                                              replace={**value, "cells": cells, "state": "submitting", "updated_at": timestamp()})
        if created:
            self.jobs.launch(self._dispatch(children))
        return self.get(identity)
