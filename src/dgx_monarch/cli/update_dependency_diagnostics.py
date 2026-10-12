"""Bounded local dependency remediation; never copied into operator receipts."""
from __future__ import annotations

import json
import re
import sys
from typing import Any

_REASONS = frozenset({
    "missing", "incompatible Python version", "requested extra is unavailable",
    "direct URL requirements cannot be verified",
})
_FIXED_ERRORS = frozenset({
    "candidate wheel could not be verified", "candidate wheel download failed",
    "candidate dependency inspection failed", "invalid dependency contract",
    "dependency closure exceeds inspection limit", "invalid dependency requirement",
    "dependency metadata could not be verified",
})


def safe_errors(value: Any) -> list[str]:
    """Bound local remediation text; never forward raw process or metadata output."""
    if not isinstance(value, list) or len(value) > 512:
        return ["candidate dependency inspection failed"]
    result = []
    for error in value[:20]:
        if not isinstance(error, str):
            return ["candidate dependency inspection failed"]
        if error in _FIXED_ERRORS:
            result.append(error)
            continue
        name, separator, reason = error.partition(": ")
        if (separator and re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", name)
                and (reason in _REASONS or re.fullmatch(
                    r"installed [a-zA-Z0-9.+!_-]{1,64}, requires [a-zA-Z0-9.,!<>=+*_-]{1,128}", reason))):
            result.append(error)
        else:
            result.append("candidate dependency inspection failed")
    if len(value) > 20:
        result.append("additional dependency failures omitted")
    return result



def report_result(result: Any, label: str) -> bool:
    """Accept only a successful empty report, and disclose bounded failure reasons."""
    try:
        if not isinstance(result.stdout, str) or len(result.stdout) > 128 * 1024:
            raise ValueError("invalid report")
        errors = safe_errors(json.loads(result.stdout))
        if result.returncode != 0 and not errors:
            errors = ["candidate dependency inspection failed"]
    except (ValueError, TypeError):
        errors = ["candidate dependency inspection failed"]
    for error in errors:
        print(f"  {label}: {error}", file=sys.stderr)
    return not errors
