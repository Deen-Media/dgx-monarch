"""Bounded, strict-JSON projection for telemetry event rings."""
from __future__ import annotations

import math

_MAX_EVENT_TEXT = 192
_MAX_EVENT_ITEMS = 16
# JavaScript reads this JSON too, so integers stay in the range it holds
# exactly. The bound also caps what hostile telemetry can make the response
# serialize: Python refuses to convert an integer of thousands of digits.
_MAX_SAFE_INTEGER = (1 << 53) - 1
_PUBLIC_EVENT_FIELDS = {
    "audit_fail": ("key", "actual", "expected"),
    "auto_heal": ("phase", "reason"),
    "clear_vram": ("level",),
    "gate": ("verdict", "origin", "run_id", "model", "loras", "max_diff"),
    "latch_release": ("cause", "hosts", "report"),
    "load": ("model", "quant", "total_s"),
    "notice": ("phase", "note"),
    "quarantine": ("model", "lever", "origin"),
    "swap": ("model", "from", "to", "total_s"),
    "uma_reserve_breach": ("available_gib", "reserve_gib"),
    "verify": ("checked", "of"),
}


def _event_scalar(value: object) -> object | None:
    """Bound one browser/TUI event value to a strict JSON scalar or short list."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value if -_MAX_SAFE_INTEGER <= value <= _MAX_SAFE_INTEGER else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:_MAX_EVENT_TEXT]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value[:_MAX_EVENT_ITEMS]:
            scalar = _event_scalar(item)
            if isinstance(scalar, (dict, list, tuple)):
                continue
            out.append(scalar)
        return out
    return f"<{type(value).__name__}>"


def _event_time(value: object) -> int | float:
    """The event time as a finite JSON number, else 0.0 (also for an int no float holds)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return 0.0
    return value if finite else 0.0


def _public_event(event: object) -> dict | None:
    """Project an event onto its fixed, bounded browser-facing schema."""
    if not isinstance(event, dict):
        return None
    raw_kind = event.get("kind")
    if not isinstance(raw_kind, str) or not raw_kind:
        return None
    kind = raw_kind[:48]
    public: dict[str, object] = {"kind": kind}
    public["t"] = _event_time(event.get("t"))
    sequence = _event_scalar(event.get("seq"))
    if isinstance(sequence, int) and not isinstance(sequence, bool) and sequence >= 0:
        public["seq"] = sequence
    for field in _PUBLIC_EVENT_FIELDS.get(kind, ()):
        if field in event:
            public[field] = _event_scalar(event[field])
    return public


def public_event_tail(events: object, n: int = 64) -> list[dict]:
    """Return at most ``n`` fixed-schema events from a local or remote tail."""
    if not isinstance(events, (list, tuple)):
        return []
    limit = max(0, min(int(n), 256))
    public = [_public_event(event) for event in events[-limit:]] if limit else []
    return [event for event in public if event is not None]
