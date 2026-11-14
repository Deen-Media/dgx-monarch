"""One progress receiver shared by every DP group's result leader.

Normal submission hands one receiver's port to every rank. Each DP group has
its own result leader, and each leader sends its steps and then its own done.
The first leader's done therefore says nothing about the other groups: the
pump must keep draining, and keep the liveness stamp fresh, until the driver
that owns the receiver closes it.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
import types

import pytest

from dgx_monarch import first_render, mesh_lease, mesh_runtime, telemetry
from dgx_monarch.progress import ProgressReceiver


@pytest.fixture
def inert_channel(monkeypatch):
    """The real receiver over a queue-backed channel, never a native client."""
    monkeypatch.setattr(telemetry, "render_progress", telemetry.RenderProgress())
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

    actor = types.ModuleType("monarch.actor")
    actor.Channel = Channel
    monkeypatch.setitem(sys.modules, "monarch.actor", actor)
    opened: list[ProgressReceiver] = []
    yield opened
    for receiver in opened:
        receiver.__exit__(None, None, None)


def _open(opened, on_step):
    receiver = ProgressReceiver(8, on_step=on_step)
    opened.append(receiver)
    receiver.__enter__()
    return receiver


def test_a_leaders_done_does_not_stop_the_shared_pump(inert_channel):
    """The second DP group's steps still reach the bar after the first's done."""
    seen: list[dict] = []
    second_step = threading.Event()

    def on_step(message):
        seen.append(message)
        if message.get("leader") == "b":
            second_step.set()

    receiver = _open(inert_channel, on_step)
    receiver.port.send({"step": 8, "total": 8, "leader": "a"})
    receiver.port.send({"done": True})
    receiver.port.send({"step": 3, "total": 8, "leader": "b"})
    assert second_step.wait(2.0), "the pump stopped at the first leader's done"
    assert receiver._thread.is_alive()
    stamp = receiver.activity()
    receiver.port.send({"step": 4, "total": 8, "leader": "b"})
    deadline = time.monotonic() + 2.0
    while receiver.activity() == stamp and time.monotonic() < deadline:
        time.sleep(0.01)
    assert receiver.activity() > stamp
    receiver.__exit__(None, None, None)
    assert not receiver._thread.is_alive()
    assert [m.get("leader") for m in seen] == ["a", "b", "b"]


def test_a_late_dp_group_outlives_the_silence_budget_and_still_collects(
        inert_channel, monkeypatch):
    """The review's case: one shared receiver, the first leader done, the
    second leader stepping for longer than the stall budget, then success."""
    monkeypatch.setenv(mesh_lease.SAMPLE_STALL_ENV, "0.5")
    monkeypatch.setattr(mesh_lease, "_STALL_SLICE_S", 0.02)
    landed = threading.Event()

    def fake_get(future, timeout_s, thread_name="x"):
        if landed.wait(timeout_s):
            return "value-mesh"
        raise TimeoutError("slice elapsed")

    monkeypatch.setattr(mesh_runtime, "get_off_loop", fake_get)
    receiver = _open(inert_channel, lambda _message: None)
    receiver.port.send({"step": 8, "total": 8})
    receiver.port.send({"done": True})

    def second_leader():
        # Three budgets of steps, each well inside the budget, then its done.
        end = time.monotonic() + 1.5
        step = 0
        while time.monotonic() < end:
            step += 1
            receiver.port.send({"step": step, "total": 64})
            time.sleep(0.05)
        receiver.port.send({"done": True})
        landed.set()

    worker = threading.Thread(target=second_leader, daemon=True)
    worker.start()
    try:
        out = mesh_lease.collect_with_liveness(
            None, object(), 30.0, receiver.activity)
    finally:
        landed.set()
        worker.join(5.0)
    assert out == "value-mesh"


def test_a_zero_step_render_still_ends_its_pump_at_close(inert_channel):
    """No leader message at all (a one-element custom schedule): the owner's
    close alone ends the pump, as before."""
    receiver = _open(inert_channel, lambda _message: None)
    assert receiver._thread.is_alive()
    receiver.__exit__(None, None, None)
    assert not receiver._thread.is_alive()


def test_a_leader_message_after_close_is_dropped(inert_channel):
    """Close is final: a step queued behind the owner's close never reaches
    the bar, so a late leader cannot drive a retired render."""
    seen: list[dict] = []
    receiver = _open(inert_channel, seen.append)
    receiver.port.send({"done": True})
    receiver.__exit__(None, None, None)
    receiver.port.send({"step": 1, "total": 8})
    time.sleep(0.05)
    assert seen == []
    assert not receiver._thread.is_alive()
