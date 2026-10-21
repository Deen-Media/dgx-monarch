"""Finite JSON values from parsers that accept NaN and Infinity.

Python's json module reads NaN, Infinity and float literals past the double
range as non-finite floats, and ComfyUI can serve them. Canonical encoding
refuses them, so callers replace each one with None after parsing. This module
imports only the standard library: the Studio companion uses it without
loading the ComfyUI node surface that dgx_monarch.nodes imports.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

# Only these tokens can yield a non-finite float: the NaN and Infinity
# literals, an exponent of three or more digits, or a run of 100 digits. A
# number with fewer digits and a two-digit exponent stays below 1e198. Text
# that matches nowhere skips the walk; a false match only costs the walk.
_CANDIDATE_TEXT = re.compile(r"NaN|Infinity|[eE][+]?[0-9]{3}|[0-9]{100}")
_CANDIDATE_BYTES = re.compile(rb"NaN|Infinity|[eE][+]?[0-9]{3}|[0-9]{100}")


def replace_nonfinite(value: Any) -> Any:
    """Replace non-finite floats without discarding the rest of a snapshot."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: replace_nonfinite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_nonfinite(item) for item in value]
    if isinstance(value, tuple):
        return tuple(replace_nonfinite(item) for item in value)
    return value


def loads(raw: str | bytes | bytearray) -> Any:
    """Parse JSON and replace every non-finite number with None.

    Deeply nested input raises RecursionError, as json.loads does; callers
    treat that like any other malformed body.
    """
    value = json.loads(raw)
    if isinstance(raw, str):
        candidate = _CANDIDATE_TEXT.search(raw) is not None
    else:
        # json.loads also reads UTF-16 and UTF-32 bytes, whose tokens the ASCII
        # pattern cannot see. UTF-8 JSON never holds a NUL byte, so walk then.
        candidate = b"\x00" in raw or _CANDIDATE_BYTES.search(raw) is not None
    return replace_nonfinite(value) if candidate else value
