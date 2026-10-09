"""Keep dgxm top's six memory segments visually distinct.

Segment colors must be distinct, theme-defined, and outside the terminal's
repaintable standard slots. The spark theme matches the browser sidebar.
Gate chips use the same three verdict-color roles on both surfaces.
"""
from __future__ import annotations

import ast
import itertools
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("rich")

from rich.color import ANSI_COLOR_NAMES, Color

from dgx_monarch.tui.app import THEMES

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "dgx_monarch" / "tui" / "app.py"
PANEL = REPO / "web" / "js" / "dgx_monarch_panel.js"

# The bar's stacking order, left to right. `model` is the gpu segment.
SEGMENTS = ("model", "slab", "pool", "anon", "other", "cache")

# The browser panel's legend, one call per segment, color then label.
LEGEND_CALL = re.compile(r'legendItem\(\w+,\s*"(#[0-9a-fA-F]{6})",\s*`(\w+) ')

# Straight RGB distance. The grayscale theme sets the floor: six levels spread
# across the range a dark terminal can show land about 52 apart, and they
# cannot spread further. Rich's table gives the color an operator sees only
# because the test below bars the sixteen standard slots.
MIN_DISTANCE = 48.0

HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _rgb(name: str) -> tuple[int, int, int]:
    triplet = Color.parse(name).get_truecolor()
    return (triplet.red, triplet.green, triplet.blue)


def _panel_swatches() -> dict[str, str]:
    """The sidebar's segment colors, keyed by this module's palette names."""
    found = {label: value for value, label in LEGEND_CALL.findall(PANEL.read_text())}
    gpu = found.pop("gpu", None)
    if gpu is not None:
        found["model"] = gpu
    return found


def _draw_hosts() -> ast.FunctionDef:
    for node in ast.walk(ast.parse(APP.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name == "draw_hosts":
            return node
    raise AssertionError("tui/app.py: no function named draw_hosts")


def test_every_theme_colors_all_six_segments():
    for theme, palette in THEMES.items():
        missing = [key for key in SEGMENTS if key not in palette]
        assert not missing, f"theme {theme!r} leaves {missing} to the draw site"


def test_no_two_segments_in_a_theme_look_alike():
    failures = []
    for theme, palette in THEMES.items():
        for first, second in itertools.combinations(SEGMENTS, 2):
            distance = math.dist(_rgb(palette[first]), _rgb(palette[second]))
            if distance < MIN_DISTANCE:
                failures.append(
                    f"{theme}: {first} ({palette[first]}) and {second} "
                    f"({palette[second]}) are {distance:.0f} apart"
                )
    assert not failures, (
        "memory-bar segments a reader cannot tell apart:\n" + "\n".join(failures)
    )


def test_the_bar_takes_every_color_from_the_palette():
    literals = sorted({
        node.value
        for node in ast.walk(_draw_hosts())
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and (node.value in ANSI_COLOR_NAMES or HEX.match(node.value))
    })
    assert not literals, (
        f"draw_hosts hardcodes the colors {literals}; a hardcoded segment "
        "ignores the theme and can collide with a palette one"
    )


def test_no_segment_sits_in_a_slot_the_terminal_can_repaint():
    themable = []
    for theme, palette in THEMES.items():
        for key in SEGMENTS:
            number = Color.parse(palette[key]).number
            if number is not None and number < 16:
                themable.append(f"{theme}: {key} ({palette[key]}) is ANSI {number}")
    assert not themable, (
        "an operator's terminal owns the first sixteen slots and can paint two "
        "of these into one, so the distance floor above proves nothing for "
        "them:\n" + "\n".join(themable)
    )


def test_spark_is_the_browser_panel_s_palette():
    swatches = _panel_swatches()
    assert set(swatches) == set(SEGMENTS), (
        f"cannot read the sidebar legend out of {PANEL.name}: got {sorted(swatches)}"
    )
    spark = THEMES["spark"]
    drift = [
        f"{key}: terminal {spark[key]} vs browser {swatches[key]}"
        for key in SEGMENTS
        if _rgb(spark[key]) != _rgb(swatches[key])
    ]
    assert not drift, (
        "docs promise one box reads the same in the terminal and in the "
        "browser:\n" + "\n".join(drift)
    )


# The panel's three colors, one ternary in dgx_monarch_panel.js.
PANEL_CHIP = re.compile(
    r'verdict === "PASS" \? "(#[0-9a-fA-F]{6})" : '
    r'verdict === "FAIL" \? "(#[0-9a-fA-F]{6})" : "(#[0-9a-fA-F]{6})"')


@pytest.mark.parametrize("theme", sorted(THEMES))
@pytest.mark.parametrize(
    ("verdict", "key"),
    [("PASS", "good"), ("FAIL", "bad"), ("INCONCLUSIVE", "warn"),
     ("RETESTING", "warn"), ("ERROR", "warn"), ("WAIVER", "warn")],
)
def test_a_gate_chip_takes_its_theme_color_per_verdict(theme, verdict, key):
    from dgx_monarch.tui.view_helpers import gate_chip_style

    assert gate_chip_style(THEMES[theme], verdict) == THEMES[theme][key]


def test_the_terminal_and_the_browser_agree_on_the_three_chip_roles():
    """PASS green, FAIL red, everything else amber, in both surfaces."""
    match = PANEL_CHIP.search(PANEL.read_text())
    assert match, f"cannot read the gate chip colors out of {PANEL.name}"
    good, bad, other = (_rgb(value) for value in match.groups())
    assert good[1] > good[0] and good[1] > good[2], "PASS is the green chip"
    assert bad[0] > bad[1] and bad[0] > bad[2], "FAIL is the red chip"
    assert other[0] > other[2] and other[1] > other[2], "the rest are amber"
    spark = THEMES["spark"]
    assert (spark["good"], spark["bad"], spark["warn"]) == ("green", "red", "yellow")


def test_the_render_row_paints_every_verdict_and_not_the_row_at_once():
    """One style for the whole row would paint an INCONCLUSIVE ledger as a PASS."""
    source = APP.read_text()
    assert "gate_chip_style(pal, verdict)" in source
    assert 'pal["bad"] if counts.get("FAIL") else pal["good"]' not in source
