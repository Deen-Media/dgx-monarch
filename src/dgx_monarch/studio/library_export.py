"""Portable library export that stays within what one import accepts."""
from __future__ import annotations

import json
from typing import Any

from . import SCHEMA_VERSION
from .contracts import RECORD_KINDS, RECORD_LIMIT, StudioError
from .store import Store

# Import refuses a collection longer than this, so export never writes one.
EXPORT_ROWS = RECORD_LIMIT
# The companion derives these from the job's submission when it observes an
# output, and the gallery searches them in place. A portable reference never
# carries them: import keeps references for repair and never searches them.
DERIVED_ARTIFACT_FIELDS = ("search_text",)
LABELS = {"artifacts": "artifact references", "library_references": "unresolved references",
          "library_metadata": "import metadata rows"}


def portable_reference(artifact: dict[str, Any]) -> dict[str, Any]:
    """An artifact reference without fields derived from its source job."""
    return {key: value for key, value in artifact.items() if key not in DERIVED_ARTIFACT_FIELDS}


def _documents(kind: str) -> str:
    # Both paths come from the closed field tuple above; no request input enters SQL.
    prefix = {"artifacts": "$", "library_references": "$.reference"}.get(kind)
    if prefix is None:
        return "document"
    return "json_remove(document," + ",".join(f"'{prefix}.{key}'" for key in DERIVED_ARTIFACT_FIELDS) + ")"


def _mib(size: int) -> str:
    return f"{size / (1024 * 1024):g} MiB"


def _refusal(kind: str, project: str | None, *, rows: bool, byte_limit: int) -> StudioError:
    """Name the limit and a remedy that a project-scoped caller has not already tried."""
    label = LABELS.get(kind, kind)
    if rows:
        limit = f"more than {EXPORT_ROWS} {label}, the most one import accepts"
        if project:
            return StudioError("library_limit", f"This project would export {limit}. Move some of them into "
                               "another project and export each project on its own.", 413)
        return StudioError("library_limit", f"Export would hold {limit}; export one project at a time.", 413)
    if project:
        return StudioError("library_limit", f"This project's export passed {_mib(byte_limit)}, the most one import "
                           f"accepts, while adding {label}. Move some records or artifacts into another project "
                           "and export each project on its own.", 413)
    return StudioError("library_limit", f"Export exceeds {_mib(byte_limit)}; export one project at a time.", 413)


def export_library(store: Store, project: str | None, *, byte_limit: int) -> dict[str, Any]:
    from .library import FORMAT, project_ids

    if project:
        store.get("projects", project)
    records: dict[str, list[dict[str, Any]]] = {kind: [] for kind in sorted(RECORD_KINDS)}
    result: dict[str, Any] = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "records": records,
                              "artifact_references": [], "unresolved_references": [], "import_metadata": []}
    # Spaced separators overstate the compact body a client sends back.
    size = len(json.dumps(result).encode())
    collections = [(kind, records[kind]) for kind in records] + [
        ("artifacts", result["artifact_references"]), ("library_references", result["unresolved_references"]),
        ("library_metadata", result["import_metadata"])]
    with store.lock:
        for kind, target in collections:
            cursor = store.db.execute(f"SELECT {_documents(kind)} FROM objects WHERE kind=? ORDER BY id", (kind,))  # noqa: S608
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
                    if len(target) == EXPORT_ROWS:
                        raise _refusal(kind, project, rows=True, byte_limit=byte_limit)
                    size += len(json.dumps(row).encode()) + 2
                    if size > byte_limit:
                        raise _refusal(kind, project, rows=False, byte_limit=byte_limit)
                    target.append(row)
            finally:
                cursor.close()
    return result
