"""Pure health/activity presentation; no lifecycle or device authority."""
from __future__ import annotations

from collections.abc import Sequence


def telemetry_presentation(
    overall: str, states: Sequence[str], *, legacy: bool = False,
) -> dict[str, object]:
    """Separate occupancy from health without changing the v1 aggregate.

    Call only after validating the telemetry lifecycle and its aggregate. Doctor
    warnings have independent meaning and must not use the modern projection.
    Legacy reports retain degradation when choosing maintenance guidance, since
    activity alone does not establish why their aggregate was degraded.
    Maintenance is an inert enum, never permission to change lifecycle state.
    """
    health = "ready" if not legacy and overall == "degraded" and "active" in states else overall
    if states[2] == "active":
        activity = "rendering"
    elif states[1] == "active":
        activity = "busy"
    elif states[1] in {"ready", "idle"} and states[2] == "idle":
        activity = "idle"
    else:
        activity = "unknown"
    occupied = activity in {"rendering", "busy"}
    maintenance = ("wait" if occupied else "none") if health == "ready" else (
        "inspect_and_wait" if occupied else "inspect")
    return {"schema_version": 1, "health": health, "activity": activity,
            "maintenance": maintenance}
