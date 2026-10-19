"""Owner-private durable records, immutable submissions and bounded event replay."""
from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from pathlib import Path
from typing import Any

from .contracts import (
    EVENT_BYTE_LIMIT,
    EVENT_LIMIT,
    MAX_EVENT_BYTES,
    RECORD_KINDS,
    RECORD_LIMIT,
    TERMINAL_STATES,
    StudioError,
    bounded_key,
    canonical,
    digest,
    identifier,
    timestamp,
)


def private_directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    # Walk rather than resolve: a configured symlink must not silently select another owner boundary.
    for parent in [*reversed(path.parents), path]:
        if parent.is_symlink():
            raise StudioError("unsafe_state", "State directories cannot be symlinks.")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise StudioError("unsafe_state", "State directory must belong to this user with mode 0700.")
    return path


class Store:
    def __init__(self, path: Path):
        self.directory = private_directory(path)
        db_path = self.directory / "studio.sqlite3"
        fd = os.open(db_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise StudioError("unsafe_state", "State database must be an owner-private regular file.")
        finally:
            os.close(fd)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS objects (
                kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                document TEXT NOT NULL, updated REAL NOT NULL, PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS requests (
                key TEXT PRIMARY KEY, digest TEXT NOT NULL, kind TEXT NOT NULL, entity_id TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                cursor INTEGER PRIMARY KEY AUTOINCREMENT, document TEXT NOT NULL, size INTEGER NOT NULL);
        """)
        row = self.db.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
        if row is not None and row[0] != "1":
            self.db.close()
            raise StudioError("storage_version", "Unsupported storage schema; export with the matching companion.")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('schema','1')")

    def close(self) -> None:
        self.db.close()

    def get(self, kind: str, object_id: str) -> dict[str, Any]:
        with self.lock:
            row = self.db.execute("SELECT document FROM objects WHERE kind=? AND id=?", (kind, object_id)).fetchone()
        if row is None:
            raise StudioError("not_found", "Record not found.", 404)
        return json.loads(row[0])

    def list(self, kind: str, *, limit: int = 200, offset: int = 0) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT document FROM objects WHERE kind=? ORDER BY updated DESC,id LIMIT ? OFFSET ?",
                                   (kind, min(max(limit, 1), RECORD_LIMIT), max(offset, 0))).fetchall()
        return [json.loads(row[0]) for row in rows]

    def _put(self, kind: str, value: dict[str, Any], revision: int) -> None:
        self.db.execute("INSERT OR REPLACE INTO objects VALUES (?,?,?,?,?)",
                        (kind, value["id"], revision, canonical(value), timestamp()))

    def save_record(self, kind: str, object_id: str, data: dict[str, Any], revision: Any) -> dict[str, Any]:
        if kind not in RECORD_KINDS:
            raise StudioError("record_kind", "Unknown record kind.")
        bounded_key(object_id, "record id")
        if type(revision) is not int or revision < 0:
            raise StudioError("revision", "A nonnegative integer revision is required.")
        with self.lock, self.transaction():
            old = self.db.execute("SELECT revision FROM objects WHERE kind=? AND id=?", (kind, object_id)).fetchone()
            if (old[0] if old else 0) != revision:
                raise StudioError("revision_conflict", "A newer version exists; reload before saving.", 409)
            if not old and self.db.execute("SELECT count(*) FROM objects WHERE kind=?", (kind,)).fetchone()[0] >= RECORD_LIMIT:
                raise StudioError("record_limit", "Export or remove records before adding more.", 409)
            value = {"id": object_id, "revision": revision + 1, "data": data}
            self._put(kind, value, revision + 1)
            self._event("record.saved", object_id, {"record_kind": kind, "revision": revision + 1})
        return value

    def create(self, kind: str, value: dict[str, Any]) -> dict[str, Any]:
        value = {**value, "id": value.get("id", identifier())}
        with self.lock, self.transaction():
            if self.db.execute("SELECT 1 FROM objects WHERE kind=? AND id=?", (kind, value["id"])).fetchone():
                raise StudioError("exists", "The record already exists.", 409)
            self._put(kind, value, 1)
        return value

    def reserve(self, kind: str, key: Any, request: Any, value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        key = bounded_key(key)
        request_digest = digest(request)
        with self.lock, self.transaction():
            old = self.db.execute("SELECT * FROM requests WHERE key=?", (key,)).fetchone()
            if old:
                if old["digest"] != request_digest or old["kind"] != kind:
                    raise StudioError("idempotency_conflict", "This key belongs to a different request.", 409)
                return self.get(kind, old["entity_id"]), False
            self.db.execute("INSERT INTO requests VALUES (?,?,?,?)", (key, request_digest, kind, value["id"]))
            self._put(kind, value, 1)
            self._event(f"{kind}.accepted", value["id"], {"state": value.get("state")})
        return value, True

    def request(self, kind: str, key: Any, value: Any) -> dict[str, Any] | None:
        """Reconcile a timed-out request before validating a new dispatch."""
        key = bounded_key(key)
        with self.lock:
            old = self.db.execute("SELECT * FROM requests WHERE key=?", (key,)).fetchone()
            if old is None:
                return None
            if old["digest"] != digest(value) or old["kind"] != kind:
                raise StudioError("idempotency_conflict", "This key belongs to a different request.", 409)
            return self.get(kind, old["entity_id"])

    def update(self, kind: str, object_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        with self.lock, self.transaction():
            old = self.get(kind, object_id)
            if "submission" in updates and updates["submission"] != old.get("submission"):
                raise StudioError("immutable_submission", "Accepted submission settings are immutable.", 409)
            if old.get("state") in TERMINAL_STATES and updates.get("state", old["state"]) != old["state"]:
                return old
            if old.get("state") in {"running", "cancel_requested"} and updates.get("state") == "queued":
                updates = {key: value for key, value in updates.items() if key != "state"}
            if all(old.get(key) == value for key, value in updates.items()):
                return old
            value = {**old, **updates, "id": old["id"], "updated_at": timestamp()}
            if value == old:
                return old
            self._put(kind, value, 1)
            self._event(f"{kind}.updated", object_id, {key: value[key] for key in ("state", "prompt_id", "progress") if key in value})
            return value

    def _event(self, kind: str, object_id: str, data: dict[str, Any]) -> None:
        document = canonical({"kind": kind, "entity_id": object_id, "job_id": object_id if kind.startswith("jobs.") else None,
                              "worker_id": None, "server_time": timestamp(), "data": data})
        size = len(document.encode())
        if size > MAX_EVENT_BYTES:
            raise StudioError("event_limit", "Event exceeds the retention budget.")
        self.db.execute("INSERT INTO events(document,size) VALUES (?,?)", (document, size))
        self.db.execute("DELETE FROM events WHERE cursor <= (SELECT COALESCE(MAX(cursor),0)-? FROM events)", (EVENT_LIMIT,))
        while self.db.execute("SELECT COALESCE(SUM(size),0) FROM events").fetchone()[0] > EVENT_BYTE_LIMIT:
            self.db.execute("DELETE FROM events WHERE cursor=(SELECT MIN(cursor) FROM events)")

    def events(self, after: int = 0) -> dict[str, Any]:
        with self.lock:
            bounds = self.db.execute("SELECT MIN(cursor),MAX(cursor) FROM events").fetchone()
            rows = self.db.execute("SELECT cursor,document FROM events WHERE cursor>? ORDER BY cursor LIMIT 200", (after,)).fetchall()
        return {"cursor": rows[-1]["cursor"] if rows else (bounds[1] or 0), "latest_cursor": bounds[1] or 0,
                "reset": bool(after and ((bounds[0] and after < bounds[0] - 1) or after > (bounds[1] or 0))),
                "events": [{"cursor": row["cursor"], **json.loads(row["document"])} for row in rows]}

    def recover(self) -> None:
        """A new process cannot infer whether an earlier dispatch completed."""
        for kind in ("jobs", "operations"):
            for value in self.list(kind, limit=RECORD_LIMIT):
                if value.get("state") in {"submitting", "running", "applying"}:
                    self.update(kind, value["id"], {"state": "unknown", "recovery": "Reconcile before any retry."})

    def transaction(self) -> Any:
        from contextlib import contextmanager

        @contextmanager
        def begin() -> Any:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.db.execute("COMMIT")
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
        return begin()
