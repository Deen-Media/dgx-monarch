"""Fleet receiver telemetry owns tokens independently of other render lifetimes."""
from __future__ import annotations

import queue
import sys
import threading
import types

import pytest

from dgx_monarch import first_render, telemetry
from dgx_monarch.progress import ProgressReceiver


@pytest.fixture
def receiver_factory(monkeypatch):
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

    def make(*, track_render=True):
        stepped = threading.Event()
        receiver = ProgressReceiver(
            8, on_step=lambda _message: stepped.set(), track_render=track_render)
        receivers.append(receiver)
        return receiver, stepped

    yield tracker, make, Channel
    for receiver in receivers:
        receiver.__exit__(None, None, None)


def test_fleet_two_receivers_drain_without_finishing_another_render(receiver_factory):
    tracker, make, _channel = receiver_factory
    other = object()
    tracker.start(5, token=other)
    first, stepped = make()
    second, _ = make()
    first.__enter__()
    second.__enter__()
    assert tracker.snapshot()["active_renders"] == 3
    first.port.send({"step": 8, "total": 8})
    assert stepped.wait(1)
    first.port.send({"done": True})
    first._thread.join(1)
    assert not first._thread.is_alive()
    # A progress done message is not proof that the sample future settled.
    assert tracker.snapshot()["active_renders"] == 3
    first.__exit__(None, None, None)
    assert tracker.snapshot()["active_renders"] == 2
    first.__exit__(None, None, None)
    assert tracker.snapshot()["active_renders"] == 2
    second.__exit__(None, None, None)
    assert tracker.snapshot()["active_renders"] == 1
    tracker.finish(other)
    assert tracker.snapshot()["active"] is False


def test_fleet_late_steps_cannot_resurrect_or_update_another_render(receiver_factory):
    tracker, make, _channel = receiver_factory
    receiver, _ = make()
    receiver.__enter__()
    receiver.__exit__(None, None, None)
    completed = tracker.snapshot()
    tracker.step(8, 8, token=receiver._telemetry_token)
    assert tracker.snapshot() == completed
    other = object()
    tracker.start(5, token=other)
    active = tracker.snapshot()
    tracker.step(8, 8, token=receiver._telemetry_token)
    assert tracker.snapshot() == active
    tracker.finish(other)


def test_fleet_zero_step_receiver_retires_without_any_actor_message(receiver_factory):
    tracker, make, _channel = receiver_factory
    receiver, _ = make()
    receiver.__enter__()
    assert tracker.snapshot()["active_renders"] == 1
    receiver.__exit__(None, None, None)
    assert not receiver._thread.is_alive()
    assert tracker.snapshot()["active_renders"] == 0


def test_fleet_failed_channel_open_guard_can_retire_token(receiver_factory, monkeypatch):
    tracker, make, channel = receiver_factory
    receiver, _ = make()

    def fail_open():
        raise RuntimeError("channel unavailable")

    monkeypatch.setattr(channel, "open", fail_open)
    with pytest.raises(RuntimeError, match="channel unavailable"):
        receiver.__enter__()
    receiver.__exit__(None, None, None)
    assert tracker.snapshot()["active"] is False


class Cancelled(BaseException):
    def __bool__(self):
        raise AssertionError("cancellation must never be tested for truth")


@pytest.mark.parametrize("failure_type", [RuntimeError, Cancelled])
def test_fleet_join_failure_still_retires_owned_telemetry(
    receiver_factory, monkeypatch, failure_type,
):
    tracker, make, _channel = receiver_factory
    receiver, _ = make()
    receiver.__enter__()
    receiver.port.send({"done": True})
    receiver._thread.join(1)
    assert not receiver._thread.is_alive()
    failure = failure_type("join interrupted")

    def fail_join(*_args, **_kwargs):
        raise failure

    with monkeypatch.context() as scoped:
        scoped.setattr(receiver._thread, "join", fail_join)
        with pytest.raises(failure_type) as raised:
            receiver.__exit__(None, None, None)
    assert raised.value is failure
    assert tracker.snapshot()["active_renders"] == 0
    receiver.__exit__(None, None, None)
    assert tracker.snapshot()["active"] is False


def test_fleet_telemetry_failure_preserves_primary_cancel_and_can_retry(
    receiver_factory, monkeypatch,
):
    tracker, make, _channel = receiver_factory
    receiver, _ = make()
    receiver.__enter__()
    primary = Cancelled("sampling cancelled")

    def fail_finish(_token):
        raise RuntimeError("telemetry interrupted before commit")

    with monkeypatch.context() as scoped:
        scoped.setattr(tracker, "finish", fail_finish)
        with pytest.raises(Cancelled) as raised:
            receiver.__exit__(type(primary), primary, None)
    assert raised.value is primary
    assert any("telemetry interrupted" in note for note in primary.__notes__)
    assert not receiver._thread.is_alive()
    receiver.__exit__(None, None, None)
    assert tracker.snapshot()["active_renders"] == 0


def test_existing_receiver_does_not_own_normal_render_telemetry(receiver_factory):
    tracker, make, _channel = receiver_factory
    normal = object()
    tracker.start(8, token=normal)
    receiver, stepped = make(track_render=False)
    receiver.__enter__()
    receiver.port.send({"step": 1, "total": 8})
    assert stepped.wait(1)
    receiver.__exit__(None, None, None)
    assert tracker.snapshot()["active_renders"] == 1
    tracker.finish(normal)
    assert tracker.snapshot()["active"] is False


@pytest.mark.parametrize("failure_type", [None, RuntimeError, Cancelled])
def test_actual_fleet_dispatch_pairs_receiver_lifetime(
    receiver_factory, monkeypatch, failure_type,
):
    import torch

    from dgx_monarch.nodes import fleet
    from dgx_monarch.nodes.common import MeshSpec, ModelSpec

    tracker, _make, _channel = receiver_factory
    failure = failure_type("second job failed") if failure_type else None
    starts = []

    class Handle:
        world = 2

        def effective_worker_args(self, args):
            return dict(args)

        def ensure_setup(self, *_args, **_kwargs):
            pass

        def submit_sample_to(self, index, _request, *, progress_port, **_kwargs):
            starts.append(tracker.snapshot()["active_renders"])
            progress_port.send({"step": 8, "total": 8})
            progress_port.send({"done": True})
            return index

        def collect_one(self, future, timeout_s):
            if future == 1 and failure is not None:
                raise failure
            return {"latent": "wire", "host": "fixture", "sample_s": 0.1}

        def cancel_sample(self, _render_id, *, wait):
            pass

    handle = Handle()
    mesh = MeshSpec(
        handle=handle, topology_preset="auto", attention="TORCH_FLASH",
        sync_ulysses=True, worker_args={})
    spec = ModelSpec(mesh=mesh, unet_name="model.safetensors")
    monkeypatch.setattr(fleet, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        fleet, "_fleet_worker_policy", lambda *_args: fleet._FleetAuthorization({}, {}, None))
    monkeypatch.setattr(fleet.consent_waiver, "stamp_request", lambda *_args: None)
    monkeypatch.setattr(fleet, "pack_latent", lambda value: value)
    monkeypatch.setattr(fleet, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(
        fleet, "read_latent_result", lambda _value, _guard=None: torch.zeros(1, 1, 1, 1))

    def invoke():
        return fleet.DGXMonarchFleetKSampler().fleet(
            [spec], [object()], ["NEG"], [{"samples": torch.zeros(1, 1, 1, 1)}],
            [""], [42], [8], [1.0], ["euler"], ["simple"], positive=["A", "B"])

    if failure_type is None:
        result, = invoke()
        assert result["samples"].shape[0] == 2
    else:
        with pytest.raises(failure_type) as raised:
            invoke()
        assert raised.value is failure
    assert starts == [1, 2]
    assert tracker.snapshot()["active_renders"] == 0
    assert tracker.snapshot()["active"] is False
