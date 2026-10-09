"""Small rendering and recording helpers for the Textual dashboard."""
from __future__ import annotations

import json

from rich.text import Text

from ..config import tcp_endpoint
from .braille import EVENT_GLYPHS, EVENT_STYLES
from .data import RingStore, _private_record_stream


def _loop_label(address: str) -> str:
    try:
        host, _ = tcp_endpoint(address)
        return host
    except (TypeError, ValueError):
        return address


def _loop_indicator(
    state: object, palette: dict[str, str],
) -> tuple[str, str]:
    if state is True:
        return "●", palette["good"]
    if state is False:
        return "●", palette["bad"]
    return "?", "dim"


def _append_record(path: str, tick: dict) -> None:
    with _private_record_stream(path, append=True) as stream:
        stream.write(json.dumps(tick, default=str) + "\n")


def _series(ring: RingStore, upto: int, picker, n: int) -> list[float]:
    ticks = list(ring.ticks)[: len(ring.ticks) + upto + 1][-n:]
    out = []
    for tick in ticks:
        try:
            out.append(float(picker(tick) or 0.0))
        except Exception:
            out.append(0.0)
    return out


def _redraw_panels(app, panel_type, tick, panel_ids) -> None:
    """Update every panel even when an earlier query/draw/update fails."""
    for wid in panel_ids:
        panel = None
        try:
            panel = app.query_one(f"#{wid}", panel_type)
            panel.update(getattr(app, f"draw_{wid}")(tick))
        except Exception as exc:
            if panel is None:
                continue
            try:
                panel.update(Text(f"[{wid}] {exc!r}", style="red"))
            except Exception:
                continue


def _marker_lane(ring: RingStore, upto: int, width: int) -> Text:
    """One char row aligned under a braille graph: events at their tick column."""
    ticks = list(ring.ticks)[: len(ring.ticks) + upto + 1][-width * 2:]
    lane = [" "] * width
    styles: dict[int, str] = {}
    for i, tick in enumerate(ticks):
        for event in tick.get("events", []):
            kind = event.get("kind")
            if kind in EVENT_GLYPHS and kind != "verify":
                col = min(width - 1, i // 2)
                lane[col] = EVENT_GLYPHS[kind]
                styles[col] = EVENT_STYLES[kind]
    text = Text()
    for i, char in enumerate(lane):
        text.append(char, style=styles.get(i, "dim"))
    return text


def gate_chip_style(palette: dict, verdict: str) -> str:
    """Choose a color for one gate count using the browser panel's status roles.

    PASS uses the good color, FAIL the bad color, and INCONCLUSIVE, RETESTING and
    ERROR the warning color. Color counts individually so a row with no FAIL
    cannot hide inconclusive results.
    """
    if verdict == "PASS":
        return palette["good"]
    return palette["bad"] if verdict == "FAIL" else palette["warn"]
