"""One stored copy of each workflow, and retention for expired previews.

A job preview and the job submitted from it hold the workflow under requested,
effective and workflow, and every cell of a grid repeats the same graph. The
store keeps one copy per workflow digest and puts it back on every read, so the
stored and wire shape of a preview or a job does not change. An expired preview
leaves the store unless a grid run still reads it: the grid preview the run
came from, which a restored browser session reloads, and the previews of its
excluded cells, whose details have no job to read instead. A stored workflow
stays while any preview, grid preview or job names it; jobs are kept, so the
workflows they name are too.

Every function here runs under the store lock; the writers run inside the
caller's transaction.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from .contracts import StudioError, canonical

PACKED_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "previews": (("workflow",), ("requested", "workflow"), ("effective", "workflow")),
    "grid_previews": (("request", "draft", "workflow"),),
    "jobs": (("submission", "workflow"), ("submission", "requested", "workflow"),
             ("submission", "effective", "workflow")),
}
# Only previews expire; jobs name stored workflows but are never swept.
SWEPT_KINDS = frozenset({"previews", "grid_previews"})
STORAGE_KEY = "workflow_storage"
SWEEP_SECONDS = 300.0
SWEEP_ROWS = 4096
_KINDS = "kind IN ('previews','grid_previews')"
_NAMING = "kind IN ('previews','grid_previews','jobs')"

SCHEMA = f"""
    CREATE TABLE IF NOT EXISTS preview_workflows (digest TEXT PRIMARY KEY, document TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS preview_holds (
        kind TEXT NOT NULL, preview_id TEXT NOT NULL, grid_id TEXT NOT NULL, PRIMARY KEY(kind,preview_id,grid_id));
    CREATE INDEX IF NOT EXISTS previews_by_expiry ON objects(json_extract(document,'$.expires_at')) WHERE {_KINDS};
    DROP INDEX IF EXISTS previews_by_workflow;
    CREATE INDEX IF NOT EXISTS workflows_by_digest
        ON objects(json_extract(document,'$.{STORAGE_KEY}.digest')) WHERE {_NAMING};
"""


def _slot(value: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _without(value: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    """A copy of value without the slot at path; the caller's value is unchanged."""
    copy = dict(value)
    if len(path) == 1:
        del copy[path[0]]
    else:
        copy[path[0]] = _without(value[path[0]], path[1:])
    return copy


def _stripped(kind: str, value: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]] | None:
    """The document without the slots that repeat its source graph, those slots, and the graph."""
    paths = PACKED_PATHS.get(kind)
    if not paths or STORAGE_KEY in value:
        return None
    found, workflow = _slot(value, paths[0])
    if not found or not isinstance(workflow, dict):
        return None
    stored, packed = value, []
    for path in paths:
        present, slot = _slot(value, path)
        # A slot that differs from the source graph stays inline.
        if present and slot == workflow:
            stored = _without(stored, path)
            packed.append(".".join(path))
    return stored, packed, workflow


def stored_bytes(kind: str, value: dict[str, Any]) -> int:
    """Bytes of the document pack would store for value, not counting the shared workflow it names."""
    stripped = _stripped(kind, value)
    if stripped is None:
        return len(canonical(value).encode())
    stored, packed, _ = stripped
    return len(canonical({**stored, STORAGE_KEY: {"digest": "0" * 64, "paths": packed}}).encode())


def pack(db: sqlite3.Connection, kind: str, value: dict[str, Any]) -> dict[str, Any]:
    """Store the workflow once and return the document that refers to it."""
    stripped = _stripped(kind, value)
    if stripped is None:
        return value
    stored, packed, workflow = stripped
    text = canonical(workflow)
    identity = hashlib.sha256(text.encode()).hexdigest()
    db.execute("INSERT OR IGNORE INTO preview_workflows VALUES (?,?)", (identity, text))
    return {**stored, STORAGE_KEY: {"digest": identity, "paths": packed}}


def unpack(db: sqlite3.Connection, kind: str, value: dict[str, Any]) -> dict[str, Any]:
    """Put the stored workflow back into every slot it was taken from."""
    storage = value.pop(STORAGE_KEY, None) if kind in PACKED_PATHS else None
    if not isinstance(storage, dict):
        return value
    row = db.execute("SELECT document FROM preview_workflows WHERE digest=?", (storage.get("digest"),)).fetchone()
    if row is None:
        raise StudioError("preview_workflow_missing", "The stored workflow for this preview is missing; "
                          "review a fresh preview.", 404)
    for dotted in storage.get("paths", []):
        *parents, last = dotted.split(".")
        target = value
        for key in parents:
            target = target[key]
        target[last] = json.loads(row[0])
    return value


def hold_grid(db: sqlite3.Connection, grid: dict[str, Any]) -> None:
    """Keep the previews a dispatched grid run still reads."""
    if isinstance(grid.get("preview_id"), str):
        db.execute("INSERT OR IGNORE INTO preview_holds VALUES ('grid_previews',?,?)", (grid["preview_id"], grid["id"]))
    cells = grid.get("cells")
    for cell in cells if isinstance(cells, list) else []:
        if isinstance(cell, dict) and cell.get("excluded") is True and isinstance(cell.get("preview_id"), str):
            db.execute("INSERT OR IGNORE INTO preview_holds VALUES ('previews',?,?)", (cell["preview_id"], grid["id"]))


def rebuild_holds(db: sqlite3.Connection) -> None:
    """Derive every hold from the stored grid runs, including runs older than the hold table."""
    db.execute("DELETE FROM preview_holds")
    db.execute("INSERT OR IGNORE INTO preview_holds SELECT 'grid_previews',json_extract(document,'$.preview_id'),id "
               "FROM objects WHERE kind='grid_runs' AND json_type(document,'$.preview_id')='text'")
    db.execute("INSERT OR IGNORE INTO preview_holds SELECT 'previews',json_extract(cell.value,'$.preview_id'),run.id "
               "FROM objects AS run, json_each(run.document,'$.cells') AS cell WHERE run.kind='grid_runs' "
               "AND json_type(cell.value,'$.excluded')='true' AND json_type(cell.value,'$.preview_id')='text'")


def sweep(db: sqlite3.Connection, now: float, *, rows: int = SWEEP_ROWS, orphans: bool = False) -> int:
    """Delete expired previews no grid run holds, then workflows no preview or job names."""
    expired = db.execute(
        f"SELECT kind,id,json_extract(document,'$.{STORAGE_KEY}.digest') FROM objects "  # noqa: S608
        f"WHERE {_KINDS} AND json_extract(document,'$.expires_at') < ? AND NOT EXISTS "
        "(SELECT 1 FROM preview_holds AS hold WHERE hold.kind=objects.kind AND hold.preview_id=objects.id) LIMIT ?",
        (now, rows)).fetchall()
    for kind, identity, _workflow in expired:
        db.execute("DELETE FROM objects WHERE kind=? AND id=?", (kind, identity))
    named = f"SELECT 1 FROM objects WHERE {_NAMING} AND json_extract(document,'$.{STORAGE_KEY}.digest')=?"  # noqa: S608
    candidates = {row[2] for row in expired if row[2]}
    if orphans:
        candidates |= {row[0] for row in db.execute("SELECT digest FROM preview_workflows")}
    for workflow in candidates:
        if db.execute(named + " LIMIT 1", (workflow,)).fetchone() is None:
            db.execute("DELETE FROM preview_workflows WHERE digest=?", (workflow,))
    return len(expired)
