"""Bounded, read-only view of the `operator_readiness` block; a malformed one reads unknown."""
from __future__ import annotations

from collections.abc import Mapping

from rich.text import Text

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


def _normalize(tick: object) -> tuple[str, tuple[tuple[str, str, str], ...]]:
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
    return overall, tuple(rows)


def _unknown() -> tuple[str, tuple[tuple[str, str, str], ...]]:
    return "unknown", tuple(
        (label, "unknown", _UNKNOWN_DETAIL) for _key, label in _LAYERS)


def readiness_text(tick: object, palette: Mapping[str, str]) -> Text:
    """Format lifecycle readiness without interpreting any supplied action."""
    overall, rows = _normalize(tick)
    colors = {
        "ready": palette["good"],
        "idle": palette["good"],
        "active": palette["warn"],
        "degraded": palette["warn"],
        "blocked": palette["bad"],
        "unknown": palette["bad"],
    }
    out = Text("\n Readiness ", style="dim")
    out.append(overall.upper(), style=f"bold {colors[overall]}")
    for label, state, detail in rows:
        out.append(f"\n {label} ", style="bold")
        out.append(state.upper(), style=f"bold {colors[state]}")
        out.append(f"  {detail}", style="dim")
    return out
