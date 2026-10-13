"""The normal render path attributes steps to its guard's token without owning it.

render_submit starts the guard's telemetry token and PendingRender (or the
submit guard) retires it. The receiver borrows that token for step attribution
only, so a step pumped after retirement cannot revive an ownerless render.
"""
from __future__ import annotations

import queue
import sys
import threading
import types

import pytest

from dgx_monarch import first_render, telemetry
from dgx_monarch.progress import ProgressReceiver


@pytest.fixture
def borrowed(monkeypatch):
    tracker = telemetry.RenderProgress()
    monkeypatch.setattr(telemetry, "render_progress", tracker)
    monkeypatch.setattr(first_render, "note_proof_render", lambda: None)

    class Future:
        def __init__(self, messages):
            self.messages = messages

        def get(self, timeout=None):
            return self.messages.get(timeout=timeout)

    class Receiver:
        def __init__(self, messages):
            self.messages = messages

        def recv(self):
            return Future(self.messages)

    class Port:
        def __init__(self, messages):
            self.messages = messages

        def send(self, value):
            self.messages.put(value)

    class Channel:
        @staticmethod
        def open():
            messages = queue.Queue()
            return Port(messages), Receiver(messages)

    # The actual receiver class runs with an inert channel, never a native client.
    actor = types.ModuleType("monarch.actor")
    actor.Channel = Channel
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    receivers = []

    def make(token):
        stepped = threading.Event()
        receiver = ProgressReceiver(
            8, on_step=lambda _message: stepped.set(), step_token=token)
        receivers.append(receiver)
        return receiver, stepped

    yield tracker, make
    for receiver in receivers:
        receiver.__exit__(None, None, None)


def _step(receiver, stepped, step):
    stepped.clear()
    receiver.port.send({"step": step, "total": 8})
    assert stepped.wait(1)


def test_borrowed_token_steps_the_guard_render_but_never_retires_it(borrowed):
    tracker, make = borrowed
    guard_token = object()
    tracker.start(8, {"model": "model.safetensors"}, token=guard_token)
    receiver, stepped = make(guard_token)
    receiver.__enter__()
    assert tracker.snapshot()["active_renders"] == 1
    _step(receiver, stepped, 3)
    assert tracker.snapshot()["step"] == 3
    receiver.__exit__(None, None, None)
    # The receiver's cleanup leaves the render to the token's owner.
    assert tracker.snapshot()["active_renders"] == 1
    tracker.finish(guard_token)
    snapshot = tracker.snapshot()
    assert snapshot["active"] is False and snapshot["last"]["model"] == "model.safetensors"


def test_a_step_pumped_after_the_guard_retires_cannot_create_a_phantom_render(borrowed):
    tracker, make = borrowed
    guard_token = object()
    tracker.start(8, token=guard_token)
    receiver, stepped = make(guard_token)
    receiver.__enter__()
    # PendingRender retires the token while the pump is still alive, as when
    # the done sentinel was lost or the pump outlived its 2 s join.
    tracker.finish(guard_token)
    idle = tracker.snapshot()
    _step(receiver, stepped, 5)
    assert tracker.snapshot() == idle
    assert tracker.snapshot()["active"] is False
    assert tracker._state["_legacy_count"] == 0
    # The next render is unaffected by the late step and drains normally.
    other = object()
    tracker.start(4, token=other)
    active = tracker.snapshot()
    _step(receiver, stepped, 6)
    assert tracker.snapshot() == active
    tracker.finish(other)
    assert tracker.snapshot()["active"] is False
    assert tracker.snapshot()["active_renders"] == 0


def test_an_untokened_step_with_no_live_render_is_ignored():
    tracker = telemetry.RenderProgress()
    tracker.step(5, 8)
    assert tracker.snapshot()["active"] is False
    assert tracker._state["_legacy_count"] == 0
    token = tracker.start(8, token=object())
    tracker.finish(token)
    idle = tracker.snapshot()
    tracker.step(5, 8)
    assert tracker.snapshot() == idle


def test_untokened_steps_still_update_a_live_legacy_render():
    tracker = telemetry.RenderProgress()
    tracker.start(8)
    tracker.step(2, 8)
    assert tracker.snapshot()["step"] == 2 and tracker.snapshot()["active"] is True
    tracker.finish()
    assert tracker.snapshot()["active"] is False


def test_a_receiver_cannot_both_own_and_borrow_a_token():
    with pytest.raises(ValueError, match="owns its telemetry token or borrows one"):
        ProgressReceiver(8, track_render=True, step_token=object())


def test_normal_submit_lends_the_guard_token_to_its_receiver(monkeypatch):
    import dgx_monarch.nodes.common as common
    import dgx_monarch.nodes.render_submit as submit_mod
    from dgx_monarch.nodes.pending import PendingRenderHandoff
    from render_sessions_helpers import _Progress, _real_handle, _stub_direct_submit

    handle = _real_handle()
    lent, started = [], []

    class TrackingProgress(_Progress):
        def __init__(self, *_args, step_token=None, **kwargs):
            assert "track_render" not in kwargs
            lent.append(step_token)

    spec = _stub_direct_submit(monkeypatch, handle, lambda *_args, **_kwargs: object())
    monkeypatch.setattr(submit_mod, "ProgressReceiver", TrackingProgress)
    monkeypatch.setattr(
        telemetry.render_progress, "start",
        lambda *_args, token=None, **_kwargs: started.append(token))
    pending = common.submit_render(
        spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
        handoff=PendingRenderHandoff())
    try:
        assert len(lent) == 1 and lent[0] is not None
        assert started == lent
        assert pending._telemetry_token is lent[0]
    finally:
        pending.abandon()
