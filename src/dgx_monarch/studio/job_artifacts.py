"""Import driver-reported outputs as artifacts of one reconciled job."""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from .comfy import safe_descriptor
from .contracts import StudioError, digest, timestamp
from .store import Store

MEDIA_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov"})
VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mov"})


def import_artifacts(store: Store, job_id: str, outputs: Any, *, state: str | None = None,
                     completed_at: float | None = None, prompt_id: str | None = None) -> None:
    """Record each safe output once.

    `created_at` is the driver's completion time when its history reports one,
    and `observed_at` is when the companion imported the output.
    """
    if not isinstance(outputs, dict):
        return
    job = store.get("jobs", job_id)
    observed_at = timestamp()
    created_source = "driver_completion" if completed_at is not None else "companion_observation"
    submission = job["submission"]
    requested = submission.get("requested") or {}
    for node_id, output in outputs.items():
        if not isinstance(output, dict):
            continue
        for key in ("images", "gifs", "videos"):
            for value in output.get(key, []) if isinstance(output.get(key), list) else []:
                if not isinstance(value, dict):
                    continue
                try:
                    descriptor = safe_descriptor(value)
                except StudioError:
                    continue
                suffix = PurePosixPath(descriptor["filename"]).suffix.lower()
                if suffix not in MEDIA_SUFFIXES:
                    continue
                object_id = digest([job_id, node_id, descriptor])
                artifact = {"id": object_id, "job_id": job_id, "node_id": node_id,
                            "kind": "video" if suffix in VIDEO_SUFFIXES else "image", **descriptor,
                            "metadata": {key: value.get(key) for key in ("width", "height", "frame_rate", "duration", "format")},
                            "created_at": observed_at if completed_at is None else completed_at,
                            "observed_at": observed_at, "created_at_source": created_source,
                            "intermediate": descriptor["type"] == "temp",
                            "operation": requested.get("operation", "workflow"), "project_id": submission.get("project_id"),
                            "status": state or job["state"], "model_ids": submission.get("model_identities", []),
                            "search_text": str(submission.get("effective", {})) + " " + descriptor["filename"],
                            "provenance": {"submission_digest": submission.get("digest"), "prompt_id": prompt_id or job.get("prompt_id"),
                                           "parent_artifact_ids": submission.get("parent_artifact_ids", []),
                                           "content_sha256": None, "source": "driver_history"}}
                try:
                    store.create("artifacts", artifact)
                except StudioError as exc:
                    if exc.code != "exists":
                        raise
