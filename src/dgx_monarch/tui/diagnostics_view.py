"""Fixed, bounded display of recent driver timings; no readiness inference."""
from __future__ import annotations

import math

from ..runtime_diagnostics import valid_labels


def diagnostics_view(payload: object) -> dict:
    if not isinstance(payload, dict) or payload.get("telemetry_error"):
        return {"available": False, "summary": "Phase timings unavailable or stale"}
    block = payload.get("diagnostics")
    if not isinstance(block, dict) or block.get("available") is not True or block.get("schema") != 1:
        return {"available": False, "summary": "Phase timings unavailable"}
    recent = block.get("recent")
    rows = []
    for row in recent[-4:] if isinstance(recent, list) else []:
        if not isinstance(row, dict) or not valid_labels(
                row.get("phase"), row.get("outcome"), row.get("return_mode")):
            continue
        value = row.get("duration_s")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        try:
            seconds = float(value)
        except OverflowError:
            continue
        if not math.isfinite(seconds) or not 0 <= seconds <= 86400:
            continue
        suffix = "" if row["outcome"] == "ok" else f" ({row['outcome']})"
        rows.append(f"{row['phase'].replace('_', ' ')} {seconds:.2f}s{suffix}")
    return {"available": True, "summary": "Recent driver phases: " + " · ".join(rows)
            if rows else "No completed driver phases recorded"}
