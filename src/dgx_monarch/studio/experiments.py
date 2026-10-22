"""Durable grid expansion and graph submission through ComfyUI's existing queue."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import StudioError, canonical, digest, identifier, object_value, timestamp
from .drafts import DraftError, expand_grid
from .grid_views import GRID_BYTES, VIEW_BYTES, live_view, read, summary
from .jobs import SNAPSHOT_OMITTED, Jobs
from .preview_storage import stored_bytes
from .submission_guard import stored_upload_names
from .workflows import inspect_workflow

# A grid run's cell adds a job identity, its attempts and a state to the
# preview's cell; a job row's two times vary in decimal length by a few bytes.
_RUN_CELL_BYTES = len(canonical({"job_id": "0" * 36, "attempts": ["0" * 36], "state": "submitting"}))
_TIME_SLACK = 16
_RUN_BYTES = 256


class _Budget:
    """The bytes a grid writes when it is previewed and again when it is submitted.

    The store keeps one copy of a workflow for every preview and job that
    shares it, so a cell costs its preview row or its job row without the
    workflow, and each distinct workflow is counted once. Submission writes
    every cell's job row and a grid run that copies the grid preview, so both
    totals must fit the grid budget.
    """

    def __init__(self) -> None:
        self.preview, self.submit, self.workflows = 0, _RUN_BYTES, set[str]()

    def cell(self, row: dict[str, Any], preview: dict[str, Any] | None, child: dict[str, Any] | None) -> None:
        self.submit += len(canonical(row).encode()) + _RUN_CELL_BYTES
        if preview is not None and child is not None:
            self.preview += stored_bytes("previews", preview)
            self.submit += stored_bytes("jobs", child) + _TIME_SLACK
            if preview["workflow_digest"] not in self.workflows:
                self.workflows.add(preview["workflow_digest"])
                self.preview += len(canonical(preview["workflow"]).encode())

    def exceeds(self, grid_bytes: int = 0) -> bool:
        return max(self.preview, self.submit) + grid_bytes > GRID_BYTES


class GridExperiments:
    def __init__(self, jobs: Jobs):
        self.jobs, self.store = jobs, jobs.store

    async def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        draft = object_value(request.get("draft"), "draft")
        if draft.get("seed_policy", "fixed") != "fixed":
            raise StudioError("grid_seed", "Grid previews require resolved fixed or incrementing seeds.")
        schema = (await self.jobs.catalog())["object_info"]
        runtime = await self.jobs.capture_runtime()
        # One schema view serves the inspection, the axes and every cell: it adds
        # the stored uploads the draft or an axis names to the upload choices.
        # Digests and identities keep the driver's own schema.
        view = self.jobs.upload_view({"draft": draft, "axes": request.get("axes", [])}, schema)
        # The draft's values select dynamic choices, so an axis can name a child they reveal.
        inspection = inspect_workflow(object_value(draft.get("workflow")), view,
                                      object_value(draft.get("values", {}), "values"))
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
        cells, previews, budget = [], [], _Budget()
        style_cache: dict[str, dict[str, Any]] = {}
        axis_names = {axis["name"] for axis in axes}
        parent = identifier()
        for cell in expanded["cells"]:
            row = {key: deepcopy(cell[key]) for key in ("index", "coordinates", "excluded", "errors")}
            values = {key: val for key, val in cell["parameters"].items() if key not in {"seed", "styles"}}
            if policy == "increment" and seed_binding and seed_binding not in axis_names:
                values[seed_binding] = cell["parameters"]["seed"]
            # An excluded cell never dispatches, so it is neither built nor charged for a preview or job.
            preview: dict[str, Any] | None = None
            child: dict[str, Any] | None = None
            if not row["errors"] and not row["excluded"]:
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
                    preview = self.jobs.build_preview(cell_draft, schema=schema, runtime=runtime, persist=False, view=view)
                    row.update(preview_id=preview["id"], submission_digest=preview["digest"])
                    child = {**self._fields(parent, row["index"]),
                             "submission": {key: value for key, value in preview.items() if key not in SNAPSHOT_OMITTED}}
                    previews.append(preview)
                except ValueError as exc:
                    preview = child = None
                    row["errors"].append(str(exc))
            budget.cell(row, preview, child)
            if budget.exceeds():
                raise StudioError("grid_size", "Expanded submissions exceed the 32 MiB grid budget.")
            cells.append(row)
        value = {"id": identifier(), "request": deepcopy(request), "cells": cells, "total": expanded["total"],
                 "job_count": expanded["job_count"], "created_at": timestamp(), "expires_at": timestamp() + 600,
                 "schema_digest": digest(schema), "ready": bool(expanded["job_count"]) and all(
                     row["excluded"] or not row["errors"] for row in cells)}
        value["digest"] = digest(value)
        if budget.exceeds(len(canonical(value).encode())) or len(canonical(summary(value)).encode()) > VIEW_BYTES:
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
        try:
            preview = self.store.get("grid_previews", request.get("preview_id", ""))
        except StudioError as exc:
            # The expiry sweep deletes expired grid previews; refuse as for any stale preview.
            if exc.code != "not_found":
                raise
            raise StudioError("grid_preview", "A current fully validated grid preview is required.", 409) from exc
        if not preview["ready"] or preview["expires_at"] < timestamp():
            raise StudioError("grid_preview", "A current fully validated grid preview is required.", 409)
        # Each cell checks the classes its prompt uses and its selected choices;
        # the grid's whole-schema digest stays as provenance only.
        uploads = stored_upload_names(self.store)
        identity = identifier()
        cells, children = [], []
        for cell in preview["cells"]:
            row = deepcopy(cell)
            if not cell["excluded"]:
                try:
                    submission = self.jobs.reviewed_snapshot(cell["preview_id"], uploads=uploads)
                except StudioError as exc:
                    if exc.code == "schema_changed":
                        raise StudioError("schema_changed", "Driver capabilities changed; preview the grid again.", 409) from exc
                    raise
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

    async def _dispatch(self, children: list[dict[str, Any]], *, retry: bool = False) -> None:
        # This is bounded transport, not a scheduler: the driver owns admission and execution order.
        # Retried cells record the runtime they run against, observed once per retry.
        runtime = await self.jobs.capture_runtime() if retry else None
        for child in children:
            await self.jobs.dispatch(child, runtime=runtime)

    @staticmethod
    def _fields(parent: str, index: int, retry_of: str | None = None) -> dict[str, Any]:
        return {"id": identifier(), "state": "submitting", "prompt_id": None, "client_prompt_id": identifier(),
                "experiment_id": parent, "cell_index": index, "retry_of": retry_of,
                "created_at": timestamp(), "updated_at": timestamp()}

    @classmethod
    def _child(cls, parent: str, index: int, submission: dict[str, Any], retry_of: str | None = None) -> dict[str, Any]:
        return {**cls._fields(parent, index, retry_of), "submission": deepcopy(submission)}

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
        uploads = stored_upload_names(self.store)
        for index in indices:
            cell = cells[index]
            if cell["state"] not in {"failed", "cancelled"}:
                raise StudioError("grid_retry", "Only explicitly failed or cancelled cells can be retried.", 409)
            old = self.store.get("jobs", cell["job_id"])
            self.jobs.revalidate(old["submission"], uploads=uploads)
            child = self._child(identity, index, old["submission"], old["id"])
            children.append(child)
            cell.update(job_id=child["id"], attempts=[*cell["attempts"], child["id"]], state="submitting")
        retry = {"id": identifier(), "grid_id": identity, "indices": indices, "job_ids": [row["id"] for row in children]}
        _, created = self.store.reserve_group("grid_retries", request.get("idempotency_key"), bound_request, retry, children,
                                              replace={**value, "cells": cells, "state": "submitting", "updated_at": timestamp()})
        if created:
            self.jobs.launch(self._dispatch(children, retry=True))
        return self.get(identity)
