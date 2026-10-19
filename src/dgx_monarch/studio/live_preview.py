"""Bounded previews carrying an explicit ComfyUI prompt identity."""
from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

from .contracts import StudioError, digest, timestamp
from .store import Store

FRAME_BYTES = 1024 * 1024
FRAME_COUNT = 16


def parse_frame(raw: bytes) -> dict[str, Any] | None:
    if not 8 < len(raw) <= FRAME_BYTES + 16392:
        return None
    event_type, metadata_size = struct.unpack(">II", raw[:8])
    if event_type != 4 or not 0 < metadata_size <= 16384 or 8 + metadata_size >= len(raw):
        return None
    try:
        metadata = json.loads(raw[8:8 + metadata_size])
    except (ValueError, UnicodeError):
        return None
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("prompt_id"), str)
            or not isinstance(metadata.get("image_type"), str) or metadata["image_type"] not in {"image/png", "image/jpeg"}):
        return None
    data = raw[8 + metadata_size:]
    if len(data) > FRAME_BYTES:
        return None
    return {"type": "preview_frame", "data": metadata, "image": data}


def accept_preview(store: Store, message: dict[str, Any]) -> None:
    from .media_assets import thumbnail

    metadata = message["data"]
    with store.lock:
        matches = store.db.execute("SELECT id,json_extract(document,'$.state') AS state FROM objects "
                                   "WHERE kind='jobs' AND json_extract(document,'$.prompt_id')=? LIMIT 2",
                                   (metadata["prompt_id"],)).fetchall()
    if len(matches) != 1 or matches[0]["state"] not in {"running", "queued", "cancel_requested"}:
        return
    job = matches[0]
    try:
        image = thumbnail(message["image"])
    except StudioError:
        return
    identity = digest(["live_preview", job["id"]])
    checksum = hashlib.sha256(image).hexdigest()
    with store.lock, store.transaction():
        row = store.db.execute("SELECT document FROM objects WHERE kind='artifacts' AND id=?", (identity,)).fetchone()
        previous = json.loads(row[0]) if row else {}
        artifact = {**previous, "id": identity, "job_id": job["id"], "node_id": metadata.get("node_id"),
                    "source": "live_preview", "kind": "image", "intermediate": True,
                    "filename": identity + ".jpg", "subfolder": "", "type": "temp",
                    "created_at": previous.get("created_at", timestamp()), "updated_at": timestamp(),
                    "metadata": {**previous.get("metadata", {}), "media_type": "image/jpeg", "sha256": checksum},
                    "provenance": {**previous.get("provenance", {}), "prompt_id": metadata["prompt_id"],
                                   "status": "approximate_live_preview"}}
        store.db.execute("CREATE TABLE IF NOT EXISTS preview_frames (id TEXT PRIMARY KEY,data BLOB NOT NULL,updated REAL NOT NULL)")
        store.db.execute("INSERT OR REPLACE INTO preview_frames VALUES (?,?,?)", (identity, image, timestamp()))
        store.db.execute("DELETE FROM preview_frames WHERE id NOT IN (SELECT id FROM preview_frames ORDER BY updated DESC LIMIT ?)",
                         (FRAME_COUNT,))
        store._put("artifacts", artifact, 1)
        store._event("artifacts.preview", identity, {"job_id": job["id"], "sha256": checksum, "intermediate": True})
    store.update("jobs", job["id"], {"preview_artifact_id": identity})


def preview_frame(store: Store, artifact_id: str) -> tuple[bytes, str]:
    artifact = store.get("artifacts", artifact_id)
    if artifact.get("source") != "live_preview":
        raise StudioError("preview_source", "This artifact is not a live preview.", 404)
    with store.lock:
        row = store.db.execute("SELECT data FROM preview_frames WHERE id=?", (artifact_id,)).fetchone()
    if row is None:
        raise StudioError("preview_expired", "This temporary preview has left the bounded cache.", 404)
    return bytes(row[0]), "image/jpeg"
