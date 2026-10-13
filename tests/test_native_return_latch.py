"""The driver latches native latent return off after a failed native read.

A failed native read keeps its ownership rules (no drop or ACK without proof),
but the next setup must not ask fresh actors for native return again: the same
fault would repeat on every render and root one more latent each time.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import mesh_setup, native_return_latch, transfer
from dgx_monarch.native_return_latch import NativeReturnLatchedError
from dgx_monarch.refusal import parse_refusal_tag
from topology_transition_helpers import (
    _acks,
    _dispatch_sample,
    _Future,
    _handle,
    _ValueMesh,
    _Workers,
)

_TOKEN = "a" * 32


def _native_descriptor():
    return {"kind": "rdma", "parts": [], "shape": [1, 4, 1, 1], "dtype": "float32",
            "owner_token": _TOKEN, "setup_generation": 1}


def _failing_read(monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error
    monkeypatch.setattr(transfer, "run_blocking_off_loop", fail)


def _fresh_native_handle():
    """A handle as a recycle leaves it: no setup yet, native configured."""
    workers = _Workers([])
    handle, _old, new = _handle(workers)
    handle.config.rdma_latent_return = True
    handle.setup_key = handle.worker_args_key = handle.topology = None
    return handle, workers, new


def _warnings(monkeypatch):
    lines = []
    monkeypatch.setattr(native_return_latch.log, "warning",
                        lambda message, *args: lines.append(message % args))
    return lines


def test_a_fresh_driver_requests_what_the_config_says():
    assert native_return_latch.snapshot() == {
        "schema": 1, "latched": False, "reason": None, "latched_unix_s": None,
        "fault_count": 0, "last_fault_unix_s": None, "clears_on": "driver_restart"}
    assert native_return_latch.native_enabled(True) is True
    assert native_return_latch.native_enabled(False) is False


def test_the_first_fault_latches_and_warns_once_later_faults_only_count(monkeypatch):
    lines = _warnings(monkeypatch)
    native_return_latch.record(TimeoutError("read timed out"))
    native_return_latch.record(RuntimeError("second fault"))

    state = native_return_latch.snapshot()
    assert state["latched"] is True and state["fault_count"] == 2
    assert state["reason"] == "TimeoutError"
    assert state["latched_unix_s"] <= state["last_fault_unix_s"]
    assert native_return_latch.native_enabled(True) is False
    assert native_return_latch.native_enabled(False) is False
    assert len(lines) == 1
    assert "latched off" in lines[0] and "#115" in lines[0]
    # Only the type name reaches telemetry; exception text can carry peer addresses.
    assert "read timed out" not in str(state)


def test_recording_never_raises_even_when_the_log_does(monkeypatch):
    def broken(*_args):
        raise RuntimeError("log sink down")
    monkeypatch.setattr(native_return_latch.log, "warning", broken)
    native_return_latch.record(TimeoutError())
    assert native_return_latch.latched()


def test_a_failed_native_read_latches_and_reraises_the_same_error(monkeypatch):
    error = TimeoutError("native read timed out")
    _failing_read(monkeypatch, error)
    with pytest.raises(TimeoutError) as raised:
        transfer.read_latent_result(_native_descriptor())
    assert raised.value is error
    assert native_return_latch.latched()
    assert native_return_latch.snapshot()["reason"] == "TimeoutError"


def test_cancellation_and_message_returns_leave_the_latch_alone(monkeypatch):
    _failing_read(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        transfer.read_latent_result(_native_descriptor())
    tensor = torch.ones(2)
    assert transfer.read_latent_result({"kind": "message", "tensor": tensor}) is tensor
    with pytest.raises(ValueError, match="unknown latent descriptor kind"):
        transfer.read_latent_result({"kind": "other"})
    assert not native_return_latch.latched()


def test_the_next_setup_after_a_fault_asks_fresh_actors_for_messaging(monkeypatch):
    first, first_workers, topology = _fresh_native_handle()
    first.ensure_setup(topology, "TORCH_FLASH")
    assert [env["rdma_latent_return"] for endpoint in first_workers.setup_endpoints
            for env in endpoint.envs] == [True, True]

    _failing_read(monkeypatch, RuntimeError("QP init timed out"))
    with pytest.raises(RuntimeError, match="QP init"):
        transfer.read_latent_result(_native_descriptor())

    # The recycle (the abandoned-lease heal or Reset attached mesh) leaves a
    # fresh handle; its first setup is where the latch takes effect.
    replacement, workers, topology = _fresh_native_handle()
    replacement.ensure_setup(topology, "TORCH_FLASH")
    assert [env["rdma_latent_return"] for endpoint in workers.setup_endpoints
            for env in endpoint.envs] == [False, False]
    assert native_return_latch.latched()


def test_the_render_fails_typed_and_its_lease_stays_abandoned(monkeypatch):
    from dgx_monarch import telemetry
    from dgx_monarch.nodes import render_result, render_submit

    handle, old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    results = _ValueMesh([
        {"rank": 0, "host": "rank0", "dp_rank": 0, "sample_s": 1.0,
         "gpu_load_s": 0.0, "transition": "reuse", "latent_stats": {},
         "latent": _native_descriptor(), "latent_extra": {}},
        {"rank": 1, "host": "rank1", "sample_s": 1.0, "gpu_load_s": 0.0,
         "transition": "reuse", "latent_stats": {}, "latent": None},
    ])
    submission = _dispatch_sample(handle, lambda: _Future(results), setup_token=token)
    error = RuntimeError("ibverbs read failed")
    _failing_read(monkeypatch, error)
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    progress = SimpleNamespace(__exit__=lambda *_args: None)
    pending = render_submit.PendingRender(
        handle, submission, progress, old, {"samples": torch.zeros((1, 4, 1, 1))},
        10.0, "render", render_result._finish_render)

    with pytest.raises(NativeReturnLatchedError) as raised:
        pending.result()

    assert raised.value.__cause__ is error
    text = str(raised.value)
    assert "native RDMA latent return" in text and "latched native return off" in text
    assert "ibverbs read failed" in text and "#115" in text
    assert parse_refusal_tag(text) is None  # driver-side, rank-asymmetric: untagged
    # A driver-side finish failure never consumes the lease.
    assert handle.sample_leases == {}
    assert sum(handle.abandoned_sample_leases.values()) == 1
    assert native_return_latch.snapshot()["fault_count"] == 1


def test_a_fleet_job_fails_typed_and_abandons_its_lease(monkeypatch):
    from dgx_monarch.nodes import fleet

    class Progress:
        def __exit__(self, *_args):
            pass

    class Handle:
        def collect_one(self, future, timeout_s):
            descriptor = _native_descriptor() if future == "native" else {"kind": "message"}
            return {"latent": descriptor, "host": "rank0", "sample_s": 0.1}

    error = RuntimeError("QP init timed out")
    _failing_read(monkeypatch, error)
    retired = []
    monkeypatch.setattr(fleet, "_retire_fleet_future",
                        lambda future, *, consumed: retired.append((future, consumed)))
    monkeypatch.setattr(fleet, "mark_defunct_on_supervision_failure", lambda *_args: None)

    with pytest.raises(NativeReturnLatchedError) as raised:
        fleet._collect_fleet_wave(
            Handle(), [(0, "native", Progress())], [None], 10.0, 1)

    assert raised.value.__cause__ is error
    assert "latched native return off" in str(raised.value)
    assert retired == [("native", False)]  # abandoned, never consumed
    assert native_return_latch.snapshot()["fault_count"] == 1


@pytest.mark.parametrize("kind,failed_job_cancelled", [("rdma", False), ("message", True)])
def test_a_wrapped_fleet_read_timeout_leaves_the_cancel_list(monkeypatch, kind, failed_job_cancelled):
    """Current behaviour, kept on purpose.

    Fleet's cancel broadcast treats a raw TimeoutError as ambiguous and cancels
    the failed job too. A native read's TimeoutError arrives wrapped in the
    typed error, a RuntimeError, so the failed job leaves the list and only live
    peers get a cancel. That job's collection already completed, so a cancel
    could reach only a finished render, and its lease ends abandoned either way.
    """
    import threading

    from dgx_monarch.nodes import fleet

    class Progress:
        def __exit__(self, *_args):
            pass

    peer_released, cancelled = threading.Event(), []

    class Handle:
        def collect_one(self, future, timeout_s):
            if future == "peer":  # still running until its cancel arrives
                peer_released.wait(5)
                return {"latent": {"kind": "message", "tensor": torch.zeros(1)}, "sample_s": 0.1}
            failed = _native_descriptor() if kind == "rdma" else {"kind": "message", "tensor": None}
            return {"latent": failed, "sample_s": 0.1}

        def cancel_sample(self, render_id, wait):
            cancelled.append(render_id)
            if render_id == "render-peer":
                peer_released.set()

    timeout = TimeoutError("read timed out")
    if kind == "rdma":
        _failing_read(monkeypatch, timeout)
    else:
        real_read = fleet.read_latent_result

        def read(descriptor, *guard):
            if descriptor.get("tensor") is None:
                raise timeout
            return real_read(descriptor, *guard)
        monkeypatch.setattr(fleet, "read_latent_result", read)
    retired = []
    monkeypatch.setattr(fleet, "_retire_fleet_future",
                        lambda future, *, consumed: retired.append((future, consumed)))
    monkeypatch.setattr(fleet, "_retire_fleet_audit", lambda _render_id: None)
    monkeypatch.setattr(fleet, "mark_defunct_on_supervision_failure", lambda *_args: None)

    with pytest.raises(NativeReturnLatchedError if kind == "rdma" else TimeoutError) as raised:
        fleet._collect_fleet_wave(
            Handle(), [(0, "failed", Progress(), "render-failed"),
                       (1, "peer", Progress(), "render-peer")], [None, None], 10.0, 2)

    assert (raised.value.__cause__ if kind == "rdma" else raised.value) is timeout
    assert cancelled == (["render-failed", "render-peer"] if failed_job_cancelled
                         else ["render-peer"])
    assert sorted(retired) == [("failed", False), ("peer", True)]


def test_a_fleet_message_failure_keeps_its_own_error(monkeypatch):
    from dgx_monarch.nodes import fleet

    error = RuntimeError("message read failed")

    def fail(_descriptor, *_guard):
        raise error
    monkeypatch.setattr(fleet, "read_latent_result", fail)
    native_return_latch.record(TimeoutError())  # an earlier render latched
    with pytest.raises(RuntimeError) as raised:
        native_return_latch.typed_read(fleet.read_latent_result, {"kind": "message"})
    assert raised.value is error


def test_a_message_descriptor_failure_keeps_its_own_error(monkeypatch):
    from dgx_monarch.nodes import render_result

    error = RuntimeError("message read failed")

    def fail(_descriptor, _guard=None):
        raise error
    monkeypatch.setattr(render_result, "read_latent_result", fail)
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None)
    leader = {"rank": 0, "dp_rank": 0, "latent": {"kind": "message"}}
    with pytest.raises(RuntimeError) as raised:
        render_result._finish_render([leader], SimpleNamespace(dp=1), {})
    assert raised.value is error
    assert not native_return_latch.latched()


def test_an_actor_error_text_stays_out_of_the_typed_message():
    actor_error = type("ActorError", (RuntimeError,), {})
    error = native_return_latch.latched_error(actor_error("remote traceback text"))
    assert "remote traceback text" not in str(error)
    assert "ActorError" in str(error)


def _supervision_error():
    from monarch._rust_bindings.monarch_hyperactor.supervision import SupervisionError
    return SupervisionError("proc died")


def _context_held():
    """The receiver's shape: context kept, cause cleared by raise_with_distinct_cause."""
    try:
        try:
            raise _supervision_error()
        except BaseException:
            raise RuntimeError("RDMA handoff ACK failed") from None
    except RuntimeError as failure:
        return failure


def _cause_held():
    failure = RuntimeError("RDMA read failed")
    failure.__cause__ = _supervision_error()
    return failure


def _late_marker():
    return RuntimeError("RDMA read failed on actor " + "a" * 600 + " (peer closed)")


def _plain():
    return RuntimeError("QP init timed out")


def _actor_error():
    from monarch.actor import ActorError
    error = ActorError(RuntimeError("peer closed"))  # an endpoint failure: the actor lives
    error.__context__ = _supervision_error()
    return error


def _wrapped(failure):
    try:
        try:
            raise failure
        except Exception as caught:
            raise native_return_latch.latched_error(caught) from caught
    except NativeReturnLatchedError as wrapped:
        return wrapped


@pytest.mark.parametrize("make,dead", [
    (_context_held, True), (_cause_held, True), (_late_marker, True),
    (_plain, False), (_actor_error, False)])
def test_a_wrapped_failure_classifies_exactly_like_the_raw_failure(monkeypatch, make, dead):
    from monarch.actor import ActorError

    from dgx_monarch import mesh_helpers

    published, formatted = [], []
    monkeypatch.setattr(mesh_helpers, "_publish_defunct",
                        lambda _handle, exc, text, holding_lock=False: published.append((exc, text)))

    def refuse(_self):
        formatted.append(True)
        raise AssertionError("an ActorError was formatted")
    monkeypatch.setattr(ActorError, "__str__", refuse)
    monkeypatch.setattr(ActorError, "__repr__", refuse)

    raw, wrapped = make(), _wrapped(make())
    mesh_helpers.mark_defunct_on_supervision_failure(object(), raw)
    mesh_helpers.mark_defunct_on_supervision_failure(object(), wrapped)

    assert [exc for exc, _text in published] == ([raw, wrapped] if dead else [])
    if dead:  # the same evidence, read from the wrapped failure, not the wrapper
        assert published[0][1] == published[1][1]
        assert "latched native return off" not in published[1][1]
    assert formatted == []


def test_telemetry_reports_the_latch_beside_return_ownership(monkeypatch):
    from dgx_monarch.nodes import routes

    monkeypatch.setattr(routes, "_worker_snapshot", lambda: ([], {}))
    monkeypatch.setattr(routes, "_ledger_summary", lambda: {})
    native_return_latch.record(TimeoutError())
    payload = routes._telemetry_uncached()
    keys = list(payload)
    assert keys.index("native_return_latch") == keys.index("return_ownership") + 1
    assert payload["native_return_latch"]["latched"] is True
    assert payload["native_return_latch"]["clears_on"] == "driver_restart"
