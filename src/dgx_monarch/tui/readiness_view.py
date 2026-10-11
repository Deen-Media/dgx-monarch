"""Bounded, read-only view of the `operator_readiness` block; a malformed one reads unknown."""
from __future__ import annotations

from collections.abc import Mapping

from rich.text import Text

from ..operator_activity import telemetry_presentation

_LAYERS = (
    ("worker_service", "Worker service"),
    ("attached_mesh", "Attached mesh"),
    ("render_session", "Render session"),
)
_OVERALL = frozenset(("ready", "degraded", "blocked", "unknown"))
_STATES = frozenset(("ready", "idle", "active", "blocked", "unknown"))
_UNKNOWN_DETAIL = "Readiness telemetry is missing or malformed."


def _plain_detail(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    detail = " ".join(value.split())
    return detail[:400] if detail else None


def _required_overall(rows: list[tuple[str, str, str]]) -> str:
    states = {state for _label, state, _detail in rows}
    if "blocked" in states:
        return "blocked"
    if "unknown" in states:
        return "unknown"
    if "active" in states:
        return "degraded"
    return "ready"


def _normalize(tick: object) -> tuple[str, tuple[tuple[str, str, str], ...], dict | None]:
    if not isinstance(tick, Mapping):
        return _unknown()
    report = tick.get("readiness")
    if not isinstance(report, Mapping) or report.get("schema_version") != 1:
        return _unknown()
    overall = report.get("overall")
    lifecycle = report.get("lifecycle")
    if not isinstance(overall, str) or overall not in _OVERALL:
        return _unknown()
    if not isinstance(lifecycle, Mapping):
        return _unknown()

    rows: list[tuple[str, str, str]] = []
    for key, label in _LAYERS:
        layer = lifecycle.get(key)
        if not isinstance(layer, Mapping):
            return _unknown()
        state = layer.get("state")
        detail = _plain_detail(layer.get("detail"))
        if not isinstance(state, str) or state not in _STATES or detail is None:
            return _unknown()
        rows.append((label, state, detail))
    required = _required_overall(rows)
    if overall != required and not (overall == "unknown" and required in {"ready", "degraded"}):
        return _unknown()
    presentation = None
    if "presentation" in report:
        if (rows[0][1] not in {"ready", "blocked", "unknown"}
                or rows[2][1] not in {"idle", "active", "unknown"}):
            return _unknown()
        supplied = report["presentation"]
        expected = telemetry_presentation(overall, [state for _, state, _ in rows])
        if (not isinstance(supplied, Mapping)
                or type(supplied.get("schema_version")) is not int
                or dict(supplied) != expected):
            return _unknown()
        presentation = expected
    return overall, tuple(rows), presentation


def _unknown() -> tuple[str, tuple[tuple[str, str, str], ...], None]:
    return "unknown", tuple(
        (label, "unknown", _UNKNOWN_DETAIL) for _key, label in _LAYERS), None


def _maintenance_text(presentation: Mapping[str, object]) -> str:
    wait = ("Wait for the active render to finish." if presentation["activity"] == "rendering"
            else "Wait for active mesh work to finish.")
    return {
        "none": "No maintenance action suggested.",
        "wait": wait,
        "inspect": "Inspect health and activity before maintenance.",
        "inspect_and_wait": f"Inspect health; {wait[0].lower()}{wait[1:]}",
    }[str(presentation["maintenance"])]


def readiness_text(tick: object, palette: Mapping[str, str]) -> Text:
    """Format lifecycle readiness without interpreting any supplied action."""
    overall, rows, presentation = _normalize(tick)
    colors = {
        "ready": palette["good"],
        "idle": palette["good"],
        "active": palette["warn"],
        "degraded": palette["warn"],
        "blocked": palette["bad"],
        "unknown": palette["bad"],
        "rendering": palette["good"],
        "busy": palette["good"],
    }
    health = str(presentation["health"]) if presentation else overall
    out = Text("\n Health " if presentation else "\n Readiness ", style="dim")
    out.append(health.upper(), style=f"bold {colors[health]}")
    activity_view = presentation or telemetry_presentation(
        overall, [state for _, state, _ in rows], legacy=True)
    activity = str(activity_view["activity"])
    out.append("\n Activity ", style="dim")
    out.append(activity.upper(), style=f"bold {colors[activity]}")
    out.append(f"\n Maintenance {_maintenance_text(activity_view)}", style="dim")
    if presentation is None and overall == "degraded":
        out.append("\n Legacy readiness includes active work; update the server for separate health.",
                   style="dim")
    for label, state, detail in rows:
        out.append(f"\n {label} ", style="bold")
        out.append(state.upper(), style=f"bold {colors[state]}")
        out.append(f"  {detail}", style="dim")
    return out
