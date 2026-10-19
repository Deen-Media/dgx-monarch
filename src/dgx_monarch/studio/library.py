"""Project membership and reviewed, atomic portable-library import/export."""
from __future__ import annotations

import json
from collections import Counter
from contextlib import closing
from copy import deepcopy
from typing import Any

from . import SCHEMA_VERSION
from .contracts import (
    RECORD_KINDS,
    RECORD_LIMIT,
    StudioError,
    bounded_key,
    canonical,
    digest,
    identifier,
    object_value,
    timestamp,
)
from .store import Store

FORMAT = "dgxm.studio.library"
MAX_LIBRARY_BYTES = 4 * 1024 * 1024
PAGE_BYTES = 8 * 1024 * 1024
AUXILIARY_LIMIT = RECORD_LIMIT
AUXILIARY_BYTES = 16 * 1024 * 1024
PREVIEW_LIMIT = 256
PREVIEW_BYTES = 32 * 1024 * 1024
IMPORT_LIMIT = RECORD_LIMIT
IMPORT_BYTES = 16 * 1024 * 1024


def _auxiliary_rows(plan: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    metadata = list(plan["metadata_rows"])
    if plan["metadata"]:
        metadata.append({"data": plan["metadata"], "project_ids": plan["project_ids"]})
    references = []
    for reference in plan["references"] + plan["repair_list"]:
        value = {key: value for key, value in reference.items() if key != "id"}
        if "id" in reference:
            value["source_id"] = reference["id"]
        references.append(value)
    return {kind: list({digest(row): {**row, "id": digest(row)} for row in rows}.values())
            for kind, rows in (("library_metadata", metadata), ("library_references", references))}


def project_ids(data: dict[str, Any]) -> list[str]:
    values = data.get("project_ids")
    if values is None:
        draft = data.get("draft", {})
        legacy = data.get("project_id") or (draft.get("project_id") if isinstance(draft, dict) else None)
        values = [legacy] if legacy else []
    if not isinstance(values, list) or len(values) > 64 or any(not isinstance(value, str) for value in values):
        raise StudioError("project_membership", "Project membership must be an array of up to 64 project IDs.")
    return list(dict.fromkeys(bounded_key(value, "project id") for value in values))


def _remap_draft(draft: Any, mapping: dict[str, str], styles: dict[str, str]) -> None:
    if isinstance(draft, dict):
        if isinstance(draft.get("project_id"), str):
            draft["project_id"] = mapping.get(draft["project_id"], draft["project_id"])
        stack = draft.get("styles", [])
        if not isinstance(stack, list):
            raise StudioError("library_references", "Draft styles must be an array.")
        for style in stack:
            if isinstance(style, dict) and isinstance(style.get("id"), str):
                style["id"] = styles.get(style["id"], style["id"])


def _remap(data: dict[str, Any], mapping: dict[str, str], styles: dict[str, str] | None = None) -> dict[str, Any]:
    result = deepcopy(data)
    if "project_ids" in result:
        result["project_ids"] = [mapping.get(value, value) for value in project_ids(result)]
    if isinstance(result.get("project_id"), str):
        result["project_id"] = mapping.get(result["project_id"], result["project_id"])
    style_ids = styles or {}
    _remap_draft(result.get("draft"), mapping, style_ids)
    grid = result.get("grid_request")
    if isinstance(grid, dict):
        _remap_draft(grid.get("draft"), mapping, style_ids)
        axes = grid.get("axes", [])
        if not isinstance(axes, list):
            raise StudioError("library_references", "Grid axes must be an array.")
        for axis in axes:
            if not isinstance(axis, dict) or axis.get("name") != "styles":
                continue
            selections = axis.get("values")
            if (not isinstance(selections, list) or any(not isinstance(selection, list)
                    or any(not isinstance(identity, str) for identity in selection) for selection in selections)):
                raise StudioError("library_references", "Style axes require arrays of saved style IDs.")
            axis["values"] = [[style_ids.get(identity, identity) for identity in selection] for selection in selections]
    return result


class Library:
    def __init__(self, store: Store):
        self.store = store

    def _check_capacity(self, kind: str, rows: list[dict[str, Any]], count_limit: int, byte_limit: int) -> None:
        # Call under the store transaction so overlapping reviews are checked
        # again against the exact committed collection at apply time.
        sizes = dict(self.store.db.execute(
            "SELECT id,length(CAST(document AS BLOB)) FROM objects WHERE kind=?", (kind,)))
        sizes.update({row["id"]: len(canonical(row).encode()) for row in rows})
        if len(sizes) > count_limit:
            raise StudioError("library_limit", f"The {kind} collection has reached its record limit.", 409)
        if sum(sizes.values()) > byte_limit:
            raise StudioError("library_limit", f"The {kind} collection has reached its storage limit.", 413)

    def save(self, kind: str, identity: str, data: dict[str, Any], revision: Any) -> dict[str, Any]:
        if kind == "experiments":
            from .experiment_records import bind_grid_record

            data = bind_grid_record(self.store, data)
        members = project_ids(data)
        if kind == "projects" and members:
            raise StudioError("project_membership", "Projects cannot be nested as library members.")
        for project in members:
            self.store.get("projects", project)
        value = {**data, "project_ids": members} if members or "project_ids" in data else data
        return self.store.save_record(kind, identity, value, revision)

    def page(self, kind: str, query: dict[str, str]) -> dict[str, Any]:
        if kind not in RECORD_KINDS or set(query) - {"project_id", "offset", "limit"}:
            raise StudioError("record_kind", "Unknown record kind or filter.")
        try:
            offset, limit = int(query.get("offset", "0")), int(query.get("limit", "100"))
        except ValueError as exc:
            raise StudioError("library_page", "Offset and limit must be integers.") from exc
        if not 0 <= offset <= RECORD_LIMIT or not 1 <= limit <= 200:
            raise StudioError("library_page", "Offset must be 0-2000 and page size 1-200.")
        project = query.get("project_id")
        clauses, params = ["kind=?"], [kind]
        if project:
            bounded_key(project, "project id")
            if kind == "projects":
                clauses.append("id=?")
                params.append(project)
            else:
                clauses.append("(EXISTS (SELECT 1 FROM json_each(json_extract(document,'$.data.project_ids')) WHERE value=?) "
                               "OR (json_extract(document,'$.data.project_ids') IS NULL AND "
                               "COALESCE(json_extract(document,'$.data.project_id'),json_extract(document,'$.data.draft.project_id'))=?))")
                params.extend([project, project])
        records: list[dict[str, Any]] = []
        size, more = 64, False
        with self.store.lock, closing(self.store.db.execute(
                "SELECT document FROM objects WHERE " + " AND ".join(clauses) +  # noqa: S608
                " ORDER BY updated DESC,id LIMIT ? OFFSET ?", [*params, limit + 1, offset])) as cursor:
            for row in cursor:
                if len(records) == limit:
                    more = True
                    break
                record = json.loads(row[0])
                record_size = len(json.dumps(record).encode()) + 2
                if size + record_size > PAGE_BYTES:
                    if not records:
                        raise StudioError("library_page_limit", "This record exceeds the 8 MiB library page budget.", 413)
                    more = True
                    break
                records.append(record)
                size += record_size
        return {"records": records, "next_offset": offset + len(records) if more else None}

    def export(self, project: str | None = None) -> dict[str, Any]:
        if project:
            self.store.get("projects", project)
        records: dict[str, list[dict[str, Any]]] = {kind: [] for kind in sorted(RECORD_KINDS)}
        result: dict[str, Any] = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "records": records,
                                  "artifact_references": [], "unresolved_references": [], "import_metadata": []}
        size = len(json.dumps(result).encode())
        collections = [(kind, records[kind]) for kind in records] + [
            ("artifacts", result["artifact_references"]), ("library_references", result["unresolved_references"]),
            ("library_metadata", result["import_metadata"])]
        with self.store.lock:
            for kind, target in collections:
                cursor = self.store.db.execute("SELECT document FROM objects WHERE kind=? ORDER BY id", (kind,))
                try:
                    for raw in cursor:
                        row = json.loads(raw[0])
                        if project:
                            if kind == "projects":
                                selected = row["id"] == project
                            elif kind == "library_metadata":
                                selected = project_ids(row) == [project]
                            else:
                                selected = project in project_ids(row["data"] if kind in RECORD_KINDS else row)
                            if not selected:
                                continue
                        size += len(json.dumps(row).encode()) + 2
                        if size > MAX_LIBRARY_BYTES:
                            raise StudioError("library_limit", "Export exceeds 4 MiB; export a single project.", 413)
                        target.append(row)
                finally:
                    cursor.close()
        return result

    def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        document = object_value(request.get("library"), "library")
        if len(canonical(document).encode()) > MAX_LIBRARY_BYTES:
            raise StudioError("library_limit", "Library imports are limited to 4 MiB.", 413)
        if document.get("format", FORMAT) != FORMAT or str(document.get("schema_version", "")).split(".")[0] != "1":
            raise StudioError("library_schema", "This library format or schema version is unsupported.")
        records = object_value(document.get("records"), "library records")
        if set(records) - RECORD_KINDS:
            raise StudioError("library_kind", "Library contains an unsupported record kind.")
        mode = request.get("mode", "merge")
        if not isinstance(mode, str) or mode not in {"merge", "copy"}:
            raise StudioError("library_mode", "Choose merge or copy import.")
        choices = object_value(request.get("choices", {}), "conflict choices")
        rows: list[dict[str, Any]] = []
        identities: set[tuple[str, str]] = set()
        mappings: dict[str, dict[str, str]] = {kind: {} for kind in RECORD_KINDS}
        for kind, collection in records.items():
            if not isinstance(collection, list) or len(collection) > RECORD_LIMIT:
                raise StudioError("library_records", "Each record collection must be an array of at most 2000 records.")
            for raw in collection:
                row = object_value(raw, "library record")
                identity = bounded_key(row.get("id"), "record id")
                if (kind, identity) in identities:
                    raise StudioError("library_identity", "Duplicate record identity in import.")
                identities.add((kind, identity))
                object_value(row.get("data"), "record data")
                if mode == "copy":
                    mappings[kind][identity] = identifier()
                rows.append({"kind": kind, "record": deepcopy(row)})
        changes, conflicts = [], []
        with self.store.lock:
            for row in rows:
                kind, record = row["kind"], row["record"]
                identity = mappings[kind].get(record["id"], record["id"])
                data = _remap(record["data"], mappings["projects"], mappings["styles"])
                if kind == "experiments" and data.get("grid_id"):
                    grid_id = bounded_key(data["grid_id"], "grid id")
                    try:
                        self.store.get("grid_runs", grid_id)
                    except StudioError as exc:
                        if exc.code != "not_found":
                            raise
                        data["grid_source"] = {"state": "unavailable", "grid_id": grid_id,
                                               "reason": "source_grid_missing"}
                members = project_ids(data)
                if kind == "projects" and members:
                    raise StudioError("project_membership", "Projects cannot be nested as library members.")
                try:
                    current = self.store.get(kind, identity)
                except StudioError as exc:
                    if exc.code != "not_found":
                        raise
                    current = None
                extensions = {key: value for key, value in record.items() if key not in {"id", "revision", "data"}}
                existing_extensions = {key: value for key, value in (current or {}).items() if key not in {"id", "revision", "data"}}
                action = "create" if current is None else "skip" if current["data"] == data and extensions == existing_extensions else "conflict"
                key = kind + ":" + record["id"]
                choice = choices.get(key)
                if choice is not None:
                    if (not isinstance(choice, dict) or not isinstance(choice.get("action"), str)
                            or choice["action"] not in {"skip", "replace"}):
                        raise StudioError("library_choice", "Conflict choices must select skip or replace with a revision.")
                    if current is None or choice.get("revision") != current["revision"]:
                        raise StudioError("revision_conflict", "Conflict resolution is stale; review the current library.", 409)
                    action = choice["action"]
                change = {"kind": kind, "id": identity, "source_id": record["id"], "data": data,
                          "extensions": extensions,
                          "action": action, "revision": current["revision"] if current else 0}
                changes.append(change)
                if action == "conflict" and current is not None:
                    conflicts.append({"key": key, "kind": kind, "id": identity, "revision": current["revision"],
                                      "current": current["data"], "incoming": data})
            for kind in RECORD_KINDS:
                count = self.store.db.execute("SELECT count(*) FROM objects WHERE kind=?", (kind,)).fetchone()[0]
                if count + sum(row["kind"] == kind and row["action"] == "create" for row in changes) > RECORD_LIMIT:
                    raise StudioError("record_limit", f"Import would exceed the {kind} record limit.", 409)
            projects = {row["id"] for row in self.store.all("projects")} | {
                row["id"] for row in changes if row["kind"] == "projects" and row["action"] != "skip"}
            missing = [{"kind": row["kind"], "id": row["id"], "project_id": project,
                        "project_ids": project_ids(row["data"]), "reason": "project_missing"}
                       for row in changes for project in project_ids(row["data"]) if project not in projects]
            references = document.get("artifact_references", [])
            if not isinstance(references, list) or len(references) > RECORD_LIMIT:
                raise StudioError("library_references", "Artifact references must be a bounded array.")
            retained_references = []
            for raw in references:
                reference = _remap(object_value(raw, "artifact reference"), mappings["projects"])
                identity = bounded_key(reference.get("id"), "artifact id")
                retained = {"artifact_id": identity, "reference": deepcopy(reference),
                            "project_ids": project_ids(reference)}
                retained_references.append(retained)
                try:
                    artifact = self.store.get("artifacts", identity)
                except StudioError as exc:
                    if exc.code != "not_found":
                        raise
                    missing.append({**retained, "reason": "source_artifact_missing"})
                else:
                    if project_ids(reference) != project_ids(artifact):
                        missing.append({**retained, "reason": "source_membership_review"})
            for change in changes:
                data = change["data"]
                draft = data.get("draft", {})
                assets = ([data["selected_artifact_id"]] if data.get("selected_artifact_id") else [])
                parents = draft.get("parent_artifact_ids", []) if isinstance(draft, dict) else []
                jobs = data.get("active_job_ids", [])
                if not isinstance(parents, list) or not isinstance(jobs, list):
                    raise StudioError("library_references", "Record references must be arrays.")
                assets += parents
                grid_ids = [data["grid_id"]] if change["kind"] == "experiments" and data.get("grid_id") else []
                for kind, reference_ids in (("artifacts", assets), ("jobs", jobs), ("grid_runs", grid_ids)):
                    for identity in reference_ids:
                        bounded_key(identity, "reference id")
                        try:
                            self.store.get(kind, identity)
                        except StudioError as exc:
                            if exc.code != "not_found":
                                raise
                            missing.append({"kind": kind, "id": identity, "project_ids": project_ids(data),
                                            "reason": "source_reference_missing"})
        preserved_metadata = document.get("import_metadata", [])
        preserved_references = document.get("unresolved_references", [])
        if any(not isinstance(rows, list) or len(rows) > RECORD_LIMIT
               for rows in (preserved_metadata, preserved_references)):
            raise StudioError("library_metadata", "Preserved metadata and references must be bounded arrays.")
        metadata_rows = []
        for raw in preserved_metadata:
            row = _remap(object_value(raw, "preserved metadata"), mappings["projects"])
            object_value(row.get("data"), "preserved metadata data")
            metadata_rows.append({**{key: value for key, value in row.items() if key != "id"},
                                  "project_ids": project_ids(row)})
        for raw in preserved_references:
            row = _remap(object_value(raw, "preserved reference"), mappings["projects"])
            if isinstance(row.get("reference"), dict):
                row["reference"] = _remap(row["reference"], mappings["projects"])
            retained_references.append({key: value for key, value in row.items() if key != "id"})
        metadata = {key: deepcopy(value) for key, value in document.items()
                    if key not in {"format", "schema_version", "records", "artifact_references",
                                   "import_metadata", "unresolved_references"}}
        # Only unambiguous IDs belong in the legacy flat lookup. Typed IDs
        # remain authoritative when two collections use the same source ID.
        occurrences = Counter(key for mapping in mappings.values() for key in mapping)
        plan = {"id": identifier(), "mode": mode, "changes": changes, "conflicts": conflicts,
                "repair_list": missing, "references": retained_references, "metadata": metadata,
                "metadata_rows": metadata_rows,
                "project_ids": sorted({project for row in changes for project in
                                       ([row["id"]] if row["kind"] == "projects" else project_ids(row["data"]))}),
                "id_mapping": {key: value for mapping in mappings.values() for key, value in mapping.items()
                               if occurrences[key] == 1},
                "typed_id_mapping": mappings,
                "ready": not conflicts, "created_at": timestamp(), "expires_at": timestamp() + 600}
        plan["digest"] = digest(plan)
        with self.store.lock, self.store.transaction():
            for kind, values in _auxiliary_rows(plan).items():
                self._check_capacity(kind, values, AUXILIARY_LIMIT, AUXILIARY_BYTES)
            self.store.db.execute("DELETE FROM objects WHERE kind='library_previews' "
                                  "AND json_extract(document,'$.expires_at') < ?", (timestamp(),))
            self._check_capacity("library_previews", [plan], PREVIEW_LIMIT, PREVIEW_BYTES)
            self.store._put("library_previews", plan, 1)
        return plan

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        key = request.get("idempotency_key")
        with self.store.lock, self.store.transaction():
            existing = self.store.request("library_imports", key, request)
            if existing is not None:
                return existing
            try:
                plan = self.store.get("library_previews", request.get("preview_id", ""))
            except StudioError as exc:
                if exc.code != "not_found":
                    raise
                raise StudioError("library_preview", "Import review is unavailable; review the library again.", 409) from exc
            if request.get("confirmed") is not True or request.get("digest") != plan["digest"]:
                raise StudioError("library_confirmation", "Confirm this exact reviewed library import.", 409)
            if not plan["ready"] or plan["expires_at"] < timestamp():
                raise StudioError("library_preview", "Resolve conflicts and review a current import plan.", 409)
            for change in plan["changes"]:
                row = self.store.db.execute("SELECT revision FROM objects WHERE kind=? AND id=?",
                                            (change["kind"], change["id"])).fetchone()
                if (row[0] if row else 0) != change["revision"]:
                    raise StudioError("revision_conflict", "Library changed after review; no records were imported.", 409)
            auxiliary = _auxiliary_rows(plan)
            for kind, values in auxiliary.items():
                self._check_capacity(kind, values, AUXILIARY_LIMIT, AUXILIARY_BYTES)
            identity = identifier()
            result = {"id": identity, "state": "succeeded", "created_at": timestamp(),
                      "imported": sum(change["action"] in {"create", "replace"} for change in plan["changes"]),
                      "repair_list": plan["repair_list"], "id_mapping": plan["id_mapping"],
                      "typed_id_mapping": plan["typed_id_mapping"]}
            # Receipts are never evicted: an old idempotency key must continue
            # to reconcile even when new imports reach the storage budget.
            self._check_capacity("library_imports", [result], IMPORT_LIMIT, IMPORT_BYTES)
            counts = {kind: self.store.db.execute("SELECT count(*) FROM objects WHERE kind=?", (kind,)).fetchone()[0]
                      for kind in RECORD_KINDS}
            for change in plan["changes"]:
                if change["action"] == "skip":
                    continue
                kind = change["kind"]
                if change["revision"] == 0:
                    counts[kind] += 1
                    if counts[kind] > RECORD_LIMIT:
                        raise StudioError("record_limit", "Library grew after review; no records were imported.", 409)
                value = {**change["extensions"], "id": change["id"], "revision": change["revision"] + 1, "data": change["data"]}
                self.store._put(kind, value, value["revision"])
                self.store._event("record.saved", change["id"], {"record_kind": kind, "revision": value["revision"]})
            for kind, values in auxiliary.items():
                for value in values:
                    self.store._put(kind, value, 1)
            self.store._put("library_imports", result, 1)
            self.store.db.execute("INSERT INTO requests VALUES (?,?,?,?)", (bounded_key(key), digest(request), "library_imports", identity))
            self.store._event("library.imported", identity, {"imported": result["imported"]})
            return result

    def membership(self, artifact_id: str, request: dict[str, Any]) -> dict[str, Any]:
        if set(request) - {"project_ids", "revision"} or "project_ids" not in request:
            raise StudioError("project_membership", "Specify project_ids and the current membership revision.")
        members = project_ids(request)
        for project in members:
            self.store.get("projects", project)
        revision = request.get("revision")
        if type(revision) is not int or revision < 0:
            raise StudioError("revision", "Artifact membership needs its current revision.")
        with self.store.lock, self.store.transaction():
            artifact = self.store.get("artifacts", artifact_id)
            if artifact.get("membership_revision", 0) != revision:
                raise StudioError("revision_conflict", "Artifact membership changed; reload before saving.", 409)
            value = {**artifact, "project_ids": members, "membership_revision": revision + 1}
            self.store._put("artifacts", value, 1)
            self.store._event("artifact.membership", artifact_id, {"project_ids": members, "revision": revision + 1})
            return value
