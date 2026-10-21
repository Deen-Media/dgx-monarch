"""Keep dgxm top's six memory segments visually distinct.

Segment colors must be distinct and theme-defined. The guards run over the
colors dgxm top draws, terminal_palette(theme, color_system) as Rich renders
them, for every color system Rich reports. On 256-color and truecolor
terminals the segments stay outside the terminal's repaintable standard slots.
A 16-color terminal has only standard slots, so there spark and ember take six
different ones. The spark theme matches the browser sidebar. Gate chips use the
same three verdict-color roles on both surfaces.
"""
from __future__ import annotations

import ast
import itertools
import json
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("rich")

from rich.color import ANSI_COLOR_NAMES, Color, ColorSystem

from dgx_monarch.tui.app import THEMES, terminal_palette

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "src" / "dgx_monarch" / "tui" / "app.py"
PANEL = REPO / "web" / "js" / "dgx_monarch_panel.js"

# The bar's stacking order, left to right. `model` is the gpu segment.
SEGMENTS = ("model", "slab", "pool", "anon", "other", "cache")

# The browser panel's legend, one call per segment, color then label.
LEGEND_CALL = re.compile(r'legendItem\(\w+,\s*MEMORY\.(\w+),\s*`(\w+) ')

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
    contract = json.loads((REPO / "src/dgx_monarch/studio/data/presentation.json").read_text())
    found = {label: contract["memory"][key]["hex"] for key, label in LEGEND_CALL.findall(PANEL.read_text())}
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


# Each color system Rich reports for a terminal, and the table it renders with.
RENDERED = {"truecolor": ColorSystem.TRUECOLOR, "256": ColorSystem.EIGHT_BIT,
            "standard": ColorSystem.STANDARD, "windows": ColorSystem.WINDOWS}
BASIC = ("standard", "windows")


def _drawn(theme: str, system: str) -> dict[str, Color]:
    """The segment colors dgxm top draws: the adapter's palette as Rich renders it."""
    palette = terminal_palette(theme, system)
    return {key: Color.parse(palette[key]).downgrade(RENDERED[system]) for key in SEGMENTS}


@pytest.mark.parametrize("system", ("truecolor", "256"))
@pytest.mark.parametrize("theme", sorted(THEMES))
def test_drawn_segments_stay_apart_and_out_of_repaintable_slots(theme, system):
    drawn = _drawn(theme, system)
    close = [f"{a} and {b}" for a, b in itertools.combinations(SEGMENTS, 2)
             if math.dist(drawn[a].get_truecolor(), drawn[b].get_truecolor()) < MIN_DISTANCE]
    assert not close, f"{theme} on {system}: segments a reader cannot tell apart: {close}"
    themable = [key for key, color in drawn.items() if color.number is not None and color.number < 16]
    assert not themable, f"{theme} on {system}: {themable} sit in a slot the terminal can repaint"


@pytest.mark.parametrize("system", BASIC)
@pytest.mark.parametrize("theme", ("spark", "ember"))
def test_a_16_color_terminal_gets_six_different_slots(theme, system):
    """Pool and cache once shared yellow here, where Rich's own downgrade kept them apart."""
    slots = [_drawn(theme, system)[key].number for key in SEGMENTS]
    assert len(set(slots)) == len(SEGMENTS), f"{theme} on {system}: slots {slots}"


@pytest.mark.parametrize("system", BASIC)
def test_mono_on_a_16_color_terminal_has_three_greys_and_the_guide_says_so(system):
    """Sixteen colors hold three visible greys, so six mono segments cannot all differ.

    This predates the shared palette, and no assignment of three greys keeps
    every pair of neighbors apart once empty segments drop out of the bar.
    """
    slots = [_drawn("mono", system)[key].number for key in SEGMENTS]
    assert len(set(slots)) == 3
    guide = " ".join((REPO / "docs/TUI.md").read_text().split())
    assert "has only three visible greys" in guide


def test_a_terminal_without_color_gets_the_mono_names():
    for theme in THEMES:
        assert terminal_palette(theme, None) == THEMES["mono"]


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
    r'verdict === "PASS" \? STATUS\.(\w+) : '
    r'verdict === "FAIL" \? STATUS\.(\w+) : STATUS\.(\w+)')


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
    contract = json.loads((REPO / "src/dgx_monarch/studio/data/presentation.json").read_text())
    colors = [contract["status"][key]["hex"] for key in match.groups()]
    good, bad, other = (_rgb(value) for value in colors)
    assert good[1] > good[0] and good[1] > good[2], "PASS is the green chip"
    assert bad[0] > bad[1] and bad[0] > bad[2], "FAIL is the red chip"
    assert other[0] > other[2] and other[1] > other[2], "the rest are amber"
    spark = terminal_palette("spark", "truecolor")
    assert (spark["good"], spark["bad"], spark["warn"]) == tuple(colors)


def test_the_render_row_paints_every_verdict_and_not_the_row_at_once():
    """One style for the whole row would paint an INCONCLUSIVE ledger as a PASS."""
    source = APP.read_text()
    assert "gate_chip_style(pal, verdict)" in source
    assert 'pal["bad"] if counts.get("FAIL") else pal["good"]' not in source
