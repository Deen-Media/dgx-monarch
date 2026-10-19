"""Versioned wire rules shared by the companion and its compiled client."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from typing import Any

from . import SCHEMA_VERSION

MAX_BODY_BYTES = 4 * 1024 * 1024
MAX_EVENT_BYTES = 65536
EVENT_LIMIT = 4096
EVENT_BYTE_LIMIT = 8 * 1024 * 1024
RECORD_LIMIT = 2000
TERMINAL_STATES = frozenset({"succeeded", "failed", "cancelled"})
RECORD_KINDS = frozenset({"projects", "presets", "styles", "sessions", "workflows", "experiments"})


class StudioError(ValueError):
    """A bounded, client-readable refusal with an HTTP status."""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


def timestamp() -> float:
    return time.time()


def identifier() -> str:
    return str(uuid.uuid4())


def canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise StudioError("invalid_json", "Only finite JSON values are accepted.") from exc


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def object_value(value: Any, name: str = "request") -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StudioError("invalid_object", f"{name} must be a JSON object.")
    canonical(value)
    return value


def bounded_key(value: Any, name: str = "idempotency_key") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
        raise StudioError("invalid_key", f"{name} must contain 1-128 letters, digits, '.', '_', ':' or '-'.")
    return value


def finite_number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StudioError("invalid_number", f"{name} must be finite.")
    if not low <= value <= high:
        raise StudioError("invalid_number", f"{name} must be between {low} and {high}.")
    return float(value)


def schema_document() -> dict[str, Any]:
    """Publish record identities and preservation policy, without promising determinism."""
    return {
        "schema_version": SCHEMA_VERSION,
        "compatibility": "Exact major version; unknown metadata is retained in durable records.",
        "records": {
            "capabilities": ["schema_version", "operations", "limits", "instance_id"],
            "generation_draft": ["workflow", "values", "styles", "lora_stack", "seed_policy"],
            "submission": ["requested", "effective", "prompt", "workflow", "workflow_digest", "versions", "parent_artifact_ids"],
            "job": ["id", "state", "submission", "prompt_id", "created_at", "updated_at"],
            "artifact": ["id", "job_id", "kind", "filename", "metadata", "created_at"],
            "catalog_entry": ["id", "name", "kind", "compatibility"],
            "workflow_binding": ["id", "source_identity", "graph_id", "node_id", "schema", "resolved"],
            "telemetry": ["sample", "observed_at", "state"],
            "operation": ["id", "action", "parameters", "digest", "state", "receipt"],
            **{kind: ["id", "revision", "data"] for kind in sorted(RECORD_KINDS)},
            "event": ["cursor", "kind", "entity_id", "worker_id", "server_time", "data"],
        },
        "limits": {"request_bytes": MAX_BODY_BYTES, "event_count": EVENT_LIMIT,
                   "event_bytes": EVENT_BYTE_LIMIT, "list_page": 200, "grid_cells": 1024,
                   "console_lines": 1000, "console_bytes": 262144,
                   "chart_samples": 600, "chart_seconds": 3600, "thumbnail_bytes": 33554432},
        "replay": "Cursors are durable. A reset response requires a fresh snapshot before applying later events.",
        "reproduction": "Reconstructs configuration; hardware and kernel determinism are not guaranteed.",
    }
