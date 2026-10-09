"""TUI poll publication and redraw settlement without optional Textual deps."""
from __future__ import annotations

import asyncio
import importlib
import sys
import types


def _app_module(monkeypatch):
    rich = types.ModuleType("rich")
    rich_text = types.ModuleType("rich.text")
    textual = types.ModuleType("textual")
    textual_app = types.ModuleType("textual.app")
    textual_binding = types.ModuleType("textual.binding")
    textual_widgets = types.ModuleType("textual.widgets")

    class Text:
        def __init__(self, text="", style=None):
            self.text = text
            self.style = style

        def append(self, text, style=None):
            self.text += str(text)
            return self

    class App:
        def __init__(self):
            pass

    class Binding:
        def __init__(self, *_args, **_kwargs):
            pass

    class Static:
        pass

    rich.text = rich_text
    rich_text.Text = Text
    textual_app.App = App
    textual_binding.Binding = Binding
    textual_widgets.Static = Static
    monkeypatch.setitem(sys.modules, "rich", rich)
    monkeypatch.setitem(sys.modules, "rich.text", rich_text)
    monkeypatch.setitem(sys.modules, "textual", textual)
    monkeypatch.setitem(sys.modules, "textual.app", textual_app)
    monkeypatch.setitem(sys.modules, "textual.binding", textual_binding)
    monkeypatch.setitem(sys.modules, "textual.widgets", textual_widgets)
    monkeypatch.delitem(sys.modules, "dgx_monarch.tui.app", raising=False)
    return importlib.import_module("dgx_monarch.tui.app")


def _tick():
    return {"t": 1.0, "events": [], "hosts": {}, "loops": [], "driver_up": False}


def test_inflight_poll_is_discarded_after_pause_or_scrub(monkeypatch):
    tui_app = _app_module(monkeypatch)
    application = tui_app.DgxmTopApp()
    application.poller = types.SimpleNamespace(tick=_tick)

    async def to_thread(function, *args):
        application.action_back()
        return function(*args)

    monkeypatch.setattr(tui_app.asyncio, "to_thread", to_thread)
    asyncio.run(application.refresh_data())

    assert application.paused is True
    assert list(application.ring.ticks) == []


def test_post_poll_failure_publishes_and_records_unknown_cycle(monkeypatch):
    tui_app = _app_module(monkeypatch)
    application = tui_app.DgxmTopApp(record="recording.jsonl")
    application.poller = types.SimpleNamespace(
        tick=_tick,
        unknown_tick=lambda marker: {**_tick(), "telemetry_error": marker},
    )
    application.notify = lambda *_args, **_kwargs: None
    application.ring.dedupe_events = lambda _events: (_ for _ in ()).throw(
        ValueError("malformed post-poll event"))
    recorded = []
    monkeypatch.setattr(
        tui_app, "_append_record",
        lambda path, tick: recorded.append((path, dict(tick))),
    )

    async def to_thread(function, *args):
        return function(*args)

    monkeypatch.setattr(tui_app.asyncio, "to_thread", to_thread)
    asyncio.run(application.refresh_data())

    assert len(application.ring.ticks) == 1
    assert application.ring.ticks[0]["telemetry_error"] == "ValueError"
    assert recorded == [("recording.jsonl", application.ring.ticks[0])]


def test_redraw_continues_after_a_missing_or_broken_panel(monkeypatch):
    tui_app = _app_module(monkeypatch)
    updates = []

    class Panel:
        def __init__(self, name):
            self.name = name

        def update(self, value):
            updates.append((self.name, getattr(value, "text", value)))

    class FakeApp:
        def query_one(self, selector, _panel_type):
            name = selector.removeprefix("#")
            if name == "missing":
                raise LookupError("not mounted")
            return Panel(name)

        def draw_broken(self, _tick):
            raise RuntimeError("draw failed")

        def draw_later(self, _tick):
            return "later content"

    tui_app._redraw_panels(
        FakeApp(), Panel, {"t": 1.0}, ("missing", "broken", "later"))

    assert updates[-1] == ("later", "later content")
