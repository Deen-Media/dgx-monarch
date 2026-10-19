"""Pure terminal adapters for the shared Studio presentation contract."""
from __future__ import annotations

import math
from typing import Any, TypeGuard

from ..studio.presentation import (
    MEMORY_ANSI,
    MEMORY_ANSI256,
    MEMORY_COLORS,
    STATUS_COLORS,
    format_metric,
)

# Branding themes preserve semantic colors. Terminal adapters select the
# contract's exact truecolor, 256-color or basic ANSI representation.
_MEMORY_KEYS = {"model": "gpu", "slab": "slab", "pool": "pool", "anon": "anonymous", "other": "other", "cache": "cache"}
_MEMORY_PALETTE = {key: MEMORY_COLORS[category] for key, category in _MEMORY_KEYS.items()}
THEMES = {
    "spark": {"accent": "bright_cyan", **_MEMORY_PALETTE,
              "good": "green", "warn": "yellow", "bad": "red", "bolt": "bright_yellow"},
    "ember": {"accent": "orange1", **_MEMORY_PALETTE,
              "good": "green", "warn": "yellow", "bad": "red", "bolt": "orange1"},
    "mono": {"accent": "white", "model": "grey100", "slab": "grey54",
             "pool": "grey85", "anon": "grey42", "other": "grey70",
             "cache": "grey30", "good": "white", "warn": "grey85", "bad": "white", "bolt": "white"},
}


def known_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def gpu_summary(stats: dict[str, Any]) -> str:
    gpu = stats.get("gpu") or {}
    return (f"GPU {format_metric(gpu.get('util_pct'), '%')} "
            f"{format_metric(gpu.get('clock_mhz'), 'GHz', source_units=True)} "
            f"{format_metric(gpu.get('power_w'), 'W')} {format_metric(gpu.get('temp_c'), '°C')}")


def terminal_palette(theme: str, color_system: str | None) -> dict[str, str]:
    palette = dict(THEMES[theme])
    if theme == "mono" or color_system is None:
        return dict(THEMES["mono"])
    for key, category in _MEMORY_KEYS.items():
        palette[key] = (MEMORY_COLORS[category] if color_system == "truecolor" else
                        f"color({MEMORY_ANSI256[category]})" if color_system == "256" else MEMORY_ANSI[category])
    if color_system == "truecolor":
        palette.update(good=STATUS_COLORS["ready"], warn=STATUS_COLORS["degraded"], bad=STATUS_COLORS["unknown"])
    return palette


def last_render_facts(last: dict[str, Any]) -> str:
    from ..attention_pins import describe

    parts = []
    if isinstance(last.get("topology"), str) and last["topology"]:
        parts.append(last["topology"])
    row = last.get("auto_row")
    if type(row) is int and row >= 1:
        parts.append(f"auto row {row}")
    if last.get("sec_per_step") is not None:
        parts.append(format_metric(last["sec_per_step"], "s") + "/step")
    pins = describe(last)
    if pins:
        parts.append(pins)
    return " · ".join(parts)
