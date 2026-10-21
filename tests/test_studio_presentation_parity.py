"""The terminal and browser consume the same units, thresholds and colors."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from dgx_monarch.studio.presentation import available_memory_status, format_metric, presentation_contract
from dgx_monarch.tui.app import DgxmTopApp, gpu_summary, last_render_facts, terminal_palette

ROOT = Path(__file__).resolve().parents[1]


def browser_values():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the generated browser contract check")
    source = ROOT / "web/js/dgx_monarch_presentation.js"
    program = (
        f"import {{ PRESENTATION, formatMetric, availableMemoryStatus }} from {json.dumps(source.as_uri())};\n"
        "console.log(JSON.stringify({contract:PRESENTATION, "
        "values:[[1073741824,'GiB',true],[1000000000,'GB/s',true],[1750,'GHz',true],[null,'W',false]]"
        ".map(([v,u,s])=>formatMetric(v,u,s)), states:[null,20,20.01].map(availableMemoryStatus)}));"
    )
    result = subprocess.run([node, "--input-type=module", "-e", program], capture_output=True,
                            text=True, timeout=30, check=True)
    return json.loads(result.stdout)


def test_generated_browser_and_python_contracts_and_formatters_match():
    browser = browser_values()
    assert browser["contract"] == presentation_contract()
    assert browser["values"] == [format_metric(v, unit, source_units=source) for v, unit, source in (
        (2**30, "GiB", True), (10**9, "GB/s", True), (1750, "GHz", True), (None, "W", False),
    )]
    assert browser["states"] == [available_memory_status(v) for v in (None, 20, 20.01)]


def test_terminal_color_adapters_preserve_approved_memory_order_and_status():
    keys = ("model", "slab", "pool", "anon", "other", "cache")
    for theme in ("spark", "ember"):
        color = terminal_palette(theme, "truecolor")
        assert [color[key] for key in keys] == ["#22d3ee", "#34d399", "#fb923c", "#c084fc", "#64748b", "#facc15"]
        assert [color[key] for key in ("good", "warn", "bad")] == ["#4ade80", "#facc15", "#f87171"]
        indexed = terminal_palette(theme, "256")
        assert [indexed[key] for key in keys] == [f"color({n})" for n in (45, 79, 208, 141, 244, 220)]
        for basic_system in ("standard", "windows"):
            basic = terminal_palette(theme, basic_system)
            # Pool and cache once shared yellow, so a 16-color terminal merged them.
            assert [basic[key] for key in keys] == ["cyan", "green", "bright_red", "magenta", "white", "yellow"]
            assert len({basic[key] for key in keys}) == len(keys)


def test_last_render_facts_include_topology_auto_row_last_step_and_attention_pins():
    text = last_render_facts({"topology": "uly2", "auto_row": 11, "sec_per_step": 1.92,
                              "attention_pins": {"comfy_kitchen_sol": 40}, "attention_pins_ran": True})
    assert "uly2" in text and "auto row 11" in text and "1.92 s/step" in text
    assert "comfy_kitchen_sol x 40 modules, ran" in text
    assert last_render_facts({}) == ""


def test_missing_gpu_metrics_are_unknown_instead_of_zero():
    assert gpu_summary({"gpu": {"util_pct": 0}}) == "GPU 0 % unknown unknown unknown"


def test_tui_preserves_every_host_and_both_rail_directions():
    app = DgxmTopApp()
    tick = {"hosts": {"host-a": {"rails": {"rail-a": {"rx_gbs": 1, "tx_gbs": 2}}},
                      "host-b": {"rails": {"rail-b": {"rx_gbs": 3, "tx_gbs": None}}}}}
    text = app.draw_rails(tick).plain
    assert "host-a rail-a" in text and "host-b rail-b" in text
    assert text.count("rx ") == 2 and text.count("tx ") == 2
    assert "1.00 GB/s" in text and "2.00 GB/s" in text and "3.00 GB/s" in text
    assert "unknown" in text


def _sized_app(monkeypatch, columns):
    from textual.geometry import Size

    app = DgxmTopApp()
    monkeypatch.setattr(type(app), "size", property(lambda self: Size(columns, 46)))
    return app


@pytest.mark.parametrize("columns", [100, 120, 160])
def test_tui_rows_fit_the_panel_with_every_host_and_both_directions(monkeypatch, columns):
    """GiB units and per-host rx and tx made rows wrap mid-reading at 120 columns."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("dgxm_top_image", ROOT / "tools/gen_dgxm_top_image.py")
    assert spec and spec.loader
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    app = _sized_app(monkeypatch, columns)
    tick = tool.sample_ticks()[-1]
    room = columns - 4  # the panel's round border and one column of padding a side
    hosts = app.draw_hosts(tick).plain.splitlines()
    bars = [line for line in hosts if "[" in line]
    assert len(bars) == 2
    for line in bars:
        assert len(line) <= room and line.endswith(" GiB") and " avail " in line
    rails = [line for line in app.draw_rails(tick).plain.splitlines() if " rx " in line]
    assert len(rails) == 2 and all(len(line) <= room and " tx " in line for line in rails)


def test_dashboard_image_shows_the_current_dashboard_text():
    """docs/media/dgxm-top.svg comes from tools/gen_dgxm_top_image.py, not an old capture."""
    import html
    import re

    svg = (ROOT / "docs/media/dgxm-top.svg").read_text()
    text = html.unescape("".join(re.findall(r"<text[^>]*>([^<]*)</text>", svg))).replace("\xa0", " ")
    assert "avail 59.6 GiB" in text and "avail 61.6 GiB" in text
    assert not re.search(r"\d\.\dG\b", text), "an old capture still shows G units"
    assert text.count("mlx5_0") == 2 and text.count(" rx ") == 2 and text.count(" tx ") == 2
    assert "Generated with" not in svg and "<!--" not in svg


def test_tui_missing_memory_does_not_invent_capacity_or_a_zero_breakdown():
    app = DgxmTopApp()
    text = app.draw_hosts({"hosts": {"one": {"stats": {"mem_gib": {}}}}}).plain
    assert "memory unknown" in text
    assert "121.6" not in text
    text = app.draw_hosts({"hosts": {"one": {"stats": {"mem_gib": {"MemTotal": 128, "MemAvailable": 20}}}}}).plain
    assert "20.0 GiB" in text and "memory breakdown unknown" in text
    assert "gpu 0.0" not in text


def test_tui_unknown_progress_does_not_invent_a_denominator():
    app = DgxmTopApp()
    text = app.draw_render({"render": {"active": True}}).plain
    assert "step unknown/unknown" in text
    assert "eta" not in text and "estimated remaining" not in text
