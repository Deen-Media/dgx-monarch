"""Braille graphs, bars and event glyphs for `dgxm top`, with no plotting dependency.

A braille cell is a 2x4 dot grid, so a width x height character panel gives a
(2*width) x (4*height) pixel canvas. Event glyphs never draw on a graph:
`view_helpers._marker_lane` puts each one but the verify dot in a row under it,
at its tick's column.
"""
from __future__ import annotations

import math

_BRAILLE_BASE = 0x2800
# dot bit positions per (x within 2, y within 4)
_DOTS = {(0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
         (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80}

EVENT_GLYPHS = {"swap": "S", "gate": "G", "load": "L", "verify": "·",
                "quarantine": "Q", "audit_fail": "!"}
EVENT_STYLES = {"swap": "yellow", "gate": "green", "load": "cyan",
                "verify": "dim", "quarantine": "red bold",
                "audit_fail": "red bold", "notice": "cyan"}


def braille_rows(values: list[float], width: int, height: int,
                 vmax: float | None = None) -> list[str]:
    """Render the last (2*width) samples as an area graph, newest right."""
    if width <= 0 or height <= 0:
        return []
    px_w, px_h = width * 2, height * 4
    samples = list(values)[-px_w:]

    def finite_nonnegative(value: object) -> float:
        if not isinstance(value, int | float):
            return 0.0
        number = float(value)
        return max(0.0, number) if math.isfinite(number) else 0.0

    samples = [0.0] * (px_w - len(samples)) + [
        finite_nonnegative(value) for value in samples
    ]
    top = max(finite_nonnegative(vmax), max(samples, default=0.0), 1e-9)
    heights = [min(px_h, round(v / top * px_h)) for v in samples]
    rows = []
    for row in range(height):
        chars = []
        for col in range(width):
            code = 0
            for dx in (0, 1):
                h = heights[col * 2 + dx]
                for dy in range(4):
                    # dy counts down from the cell top; y_from_bottom counts up from the canvas base
                    y_from_bottom = (height - 1 - row) * 4 + (3 - dy)
                    if h > y_from_bottom:
                        code |= _DOTS[(dx, dy)]
            chars.append(chr(_BRAILLE_BASE + code) if code else " ")
        rows.append("".join(chars))
    return rows


def bar(fraction: float, width: int, filled: str = "█", empty: str = " ") -> str:
    """A bar drawn to an eighth of a character with partial blocks."""
    fraction = max(0.0, min(1.0, fraction))
    cells = fraction * width
    full = int(cells)
    rem = cells - full
    partial = " ▏▎▍▌▋▊▉█"[round(rem * 8)] if full < width else ""
    return (filled * full + partial).ljust(width, empty)


def segmented_bar(segments: list[tuple[float, str]], total: float, width: int) -> list[tuple[str, str]]:
    """[(value, style)] -> [(chars, style)] spans for a stacked bar."""
    spans, used = [], 0
    for value, style in segments:
        cells = round(max(0.0, value) / max(total, 1e-9) * width)
        cells = min(cells, width - used)
        if cells > 0:
            spans.append(("█" * cells, style))
            used += cells
    if used < width:
        spans.append((" " * (width - used), "dim"))
    return spans
