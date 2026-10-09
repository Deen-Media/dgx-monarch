"""Terminal artwork for the Monarch butterfly. Telemetry owns status colors."""
from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping

from rich.console import Console
from rich.text import Text

GREEN = "#87f900"
# Diagonal box characters join the filled wings to the circuit rings.
UNICODE_WINGS = (
    " ◢█◣  ╲ ╱  ◢█◣ ",  # noqa: RUF001
    " ████◣ ● ◢████ ",
    " ▀██○━╲┃╱━○██▀ ",  # noqa: RUF001
    "  ▄█○━╱┃╲━○█▄  ",  # noqa: RUF001
    "  ██○━╱┃╲━○██  ",  # noqa: RUF001
    "  ◥██╱ ╹ ╲██◤  ",  # noqa: RUF001
)
ASCII_WINGS = (
    r" /##  \ /  ##\ ",
    " ####  o  #### ",
    r" ###o-\|/-o### ",
    r"  ##o-/|\-o##  ",
    r"  ##o-/|\-o##  ",
    r"  \##/ | \##/  ",
)


def supports_unicode() -> bool:
    """Use ASCII when the terminal or output encoding cannot carry the mark."""
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        encoding = (getattr(sys.stdout, "encoding", None)
                    or getattr(sys.__stdout__, "encoding", None) or "ascii")
        "".join(UNICODE_WINGS).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def measured_activity(tick: Mapping | None, *, now: float, interval: float) -> bool:
    """Only a recent live driver report may light the circuit animation."""
    if not tick or tick.get("driver_up") is not True or tick.get("telemetry_error"):
        return False
    stamp = tick.get("t")
    render = tick.get("render")
    if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
            or not math.isfinite(stamp) or not isinstance(render, Mapping)):
        return False
    return 0 <= now - stamp <= max(3.0, interval * 2) and render.get("active") is True


def butterfly_header(*, width: int, frame: int, animate: bool, paused: bool,
                     replay: bool, active: bool, mono: bool, unicode: bool) -> Text:
    """Render a brief entrance, then pulse the circuits only during live work."""
    moving = animate and not paused and not replay and os.environ.get("TERM") != "dumb"
    color = not mono and not os.environ.get("NO_COLOR")
    if width < 64:
        # Leave the available width to telemetry on small screens.
        return Text(" DGX MONARCH ", style="bold")
    rows = UNICODE_WINGS if unicode else ASCII_WINGS
    out = Text()
    entrance = moving and frame < 4
    for index, row in enumerate(rows):
        for column, char in enumerate(row):
            visible = not entrance or index < max(1, math.ceil(frame * len(rows) / 3))
            style = "bold"
            if index in (2, 3, 4) and char in "○o━-╲╱/\\":  # noqa: RUF001
                style = GREEN if color else "bold"
                distance = abs(column - 7)
                if moving and active and distance == frame % 4:
                    style = f"bold reverse {GREEN}" if color else "bold reverse"
            out.append(char if visible else " ", style=style)
        out.append("\n")
    return out


def header_layout(status: Text, *, width: int, **art_options) -> Text:
    """Place the butterfly beside status lines without changing their styles."""
    if width < 100:
        return status
    artwork = butterfly_header(width=width, **art_options).split("\n")
    lines = status.wrap(Console(), max(1, width - 4 - 18), overflow="fold")
    out = Text()
    for index in range(max(len(artwork), len(lines))):
        if index:
            out.append("\n")
        left = artwork[index].copy() if index < len(artwork) else Text()
        left.pad_right(max(0, 18 - left.cell_len))
        out.append_text(left)
        if index < len(lines):
            out.append_text(lines[index])
    return out
