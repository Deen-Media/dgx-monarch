"""Butterfly rendering never invents activity or changes telemetry colors."""
from __future__ import annotations

import pytest

pytest.importorskip("rich")

from dgx_monarch.tui.brand import butterfly_header, measured_activity, supports_unicode


def _logo(**overrides):
    args = {"width": 100, "frame": 8, "animate": True, "paused": False, "replay": False,
            "active": False, "mono": False, "unicode": True}
    args.update(overrides)
    return butterfly_header(**args)


def test_idle_and_disabled_animation_are_static(monkeypatch):
    monkeypatch.delenv("TERM", raising=False)
    for args in ({}, {"animate": False, "active": True},
                 {"paused": True, "active": True}, {"replay": True, "active": True}):
        assert _logo(frame=8, **args) == _logo(frame=10, **args)


def test_entrance_finishes_and_only_live_activity_pulses(monkeypatch):
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert _logo(frame=1).plain != _logo(frame=3).plain
    assert _logo(frame=3).plain == _logo(frame=20).plain
    assert _logo(frame=8, active=True) != _logo(frame=10, active=True)
    assert _logo(frame=8, active=True).plain == _logo(frame=10, active=True).plain


@pytest.mark.parametrize("overrides", [
    {"driver_up": False}, {"t": 90}, {"t": 101}, {"t": float("nan")},
    {"t": True}, {"t": "100"}, {"render": None}, {"render": {"active": False}},
    {"telemetry_error": "unavailable"},
])
def test_incomplete_stale_and_idle_reports_cannot_pulse(overrides):
    tick = {"driver_up": True, "t": 100, "render": {"active": True}, **overrides}
    assert not measured_activity(tick, now=100, interval=1)


def test_fresh_activity_obeys_poll_interval():
    tick = {"driver_up": True, "t": 100, "render": {"active": True}}
    assert measured_activity(tick, now=101, interval=1)
    assert measured_activity(tick, now=110, interval=10)
    assert not measured_activity(tick, now=121, interval=10)


def test_small_and_ascii_variants():
    assert _logo(width=40).plain == " DGX MONARCH "
    assert _logo(unicode=False).plain.isascii()
    assert "○" in _logo().plain


def test_mono_and_no_color_remove_brand_color(monkeypatch):
    assert all("#" not in str(span.style) for span in _logo(mono=True).spans)
    monkeypatch.setenv("NO_COLOR", "1")
    assert all("#" not in str(span.style) for span in _logo().spans)


def test_dumb_terminal_uses_static_ascii(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    assert not supports_unicode()
    assert _logo(frame=1, active=True) == _logo(frame=10, active=True)


def test_replay_navigation_and_narrow_header(tmp_path, monkeypatch):
    import asyncio
    import importlib

    pytest.importorskip("textual")
    from textual.geometry import Region

    monkeypatch.setenv("TERM", "xterm-256color")
    from dgx_monarch.tui.data import RingStore

    # Other tests replace the optional UI modules with minimal doubles.
    app_module = importlib.reload(importlib.import_module("dgx_monarch.tui.app"))
    store = RingStore()
    store.push({"t": 100, "events": [], "hosts": {}, "loops": [], "driver_up": True,
                "render": {"active": True, "step": 1, "steps": 20}})
    path = str(tmp_path / "replay.jsonl")
    store.save(path)

    async def check():
        app = app_module.DgxmTopApp(replay=path)
        assert app.poller is None
        async with app.run_test(size=(100, 42)) as pilot:
            await pilot.pause()
            app.redraw()
            await pilot.pause()
            panel = app.query_one("#header")
            visible = "\n".join(strip.text for strip in panel.render_lines(
                Region(0, 0, panel.region.width, panel.region.height)))
            assert 5 <= panel.size.height <= 7
            assert "DGX MONARCH" in visible
            assert "REPLAY" in visible
            assert "readiness" in visible.lower()
            assert "Render session" in visible
            assert "unavailable" not in visible
            before = app.draw_header(app.current())
            app.frame += 2
            assert app.draw_header(app.current()) == before
            await pilot.press("a")
            assert not app.animate
            await pilot.press("t", "t")
            assert app.theme_name == "mono"
            await pilot.resize_terminal(44, 32)
            header = app.draw_header(app.current()).plain
            assert header.startswith(" DGX MONARCH ")
            assert "REPLAY" in header
            assert "\n" not in header.split("REPLAY")[0]

    asyncio.run(check())


def test_butterfly_has_one_center_axis_and_equal_width_rows():
    from rich.cells import cell_len

    from dgx_monarch.tui.brand import ASCII_WINGS, UNICODE_WINGS

    mirror = str.maketrans({"◢": "◣", "◣": "◢", "◤": "◥", "◥": "◤",
                           "╲": "╱", "╱": "╲", "/": "\\", "\\": "/"})  # noqa: RUF001
    for rows in (UNICODE_WINGS, ASCII_WINGS):
        assert all(cell_len(row) == 15 for row in rows)
        for row in rows:
            assert row[:7] == row[8:][::-1].translate(mirror)
        assert all(row[7] in "┃|" for row in rows[2:5])


def test_long_status_wraps_inside_the_right_column(tmp_path, monkeypatch):
    import asyncio
    import importlib

    pytest.importorskip("textual")
    from textual.geometry import Region

    from dgx_monarch.tui.brand import UNICODE_WINGS
    from dgx_monarch.tui.data import RingStore

    monkeypatch.setenv("TERM", "xterm-256color")
    app_module = importlib.reload(importlib.import_module("dgx_monarch.tui.app"))
    store = RingStore()
    store.push({"t": 100, "events": [], "hosts": {}, "driver_up": True,
                "comfy": "very-long-illustrative-comfy-version-" * 3,
                "loops": [{"addr": "tcp://very-long-example-worker-hostname:1234", "up": True}],
                "readiness": {"schema_version": 1, "overall": "ready", "lifecycle": {
                    key: {"state": "ready", "detail": "Long readiness explanation. " * 5}
                    for key in ("worker_service", "attached_mesh", "render_session")}}})
    path = str(tmp_path / "long-replay.jsonl")
    store.save(path)

    async def check():
        app = app_module.DgxmTopApp(replay=path)
        async with app.run_test(size=(100, 70)) as pilot:
            app.redraw()
            await pilot.pause()
            panel = app.query_one("#header")
            rows = [strip.text for strip in panel.render_lines(
                Region(0, 0, panel.region.width, panel.region.height))]
            # Border and padding precede the fixed 18-cell artwork column.
            assert [row[2:17] for row in rows[1:7]] == list(UNICODE_WINGS)
            assert all(row[2:20] == " " * 18 for row in rows[7:-1])
            visible = "\n".join(rows)
            assert "Render session" in visible
            assert "very-long-example-worker-hostname" in visible

    asyncio.run(check())
