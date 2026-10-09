"""Strict JSON normalization for public HTTP payloads."""
from __future__ import annotations

import json
import math


def _replace_nonfinite(value):
    """Replace non-finite floats without discarding the rest of a snapshot."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _replace_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_nonfinite(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_nonfinite(item) for item in value)
    return value


def _json_payload(value):
    """Round-trip a payload through RFC-compliant JSON primitives."""
    return json.loads(json.dumps(
        _replace_nonfinite(value), default=str, allow_nan=False,
    ))
