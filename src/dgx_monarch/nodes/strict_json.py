"""Strict JSON normalization for public HTTP payloads."""
from __future__ import annotations

import json

from ..finite_json import replace_nonfinite as _replace_nonfinite


def _json_payload(value):
    """Round-trip a payload through RFC-compliant JSON primitives."""
    return json.loads(json.dumps(
        _replace_nonfinite(value), default=str, allow_nan=False,
    ))
