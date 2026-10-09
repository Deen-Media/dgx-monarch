"""Pending render and fleet close, telemetry, timeout and typed-refusal lease regressions."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

import dgx_monarch.nodes.fleet as fleet_mod
import dgx_monarch.nodes.pending as pending_mod
from dgx_monarch import mesh_setup, telemetry
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.pending import PendingRender
from dgx_monarch.nodes.render_session import (
    ConcurrentRenderSessionError,
    RenderSession,
)
from render_sessions_helpers import _Progress, _real_handle


def test_fleet_invocation_holds_one_session_across_bound_waves(monkeypatch):
    handle = _real_handle()
    handle.lock = threading.RLock()
    handle.sample_leases = {}
    handle.abandoned_sample_leases = {}
    handle.deferred_supervision_error = None
    handle.defunct = False
    handle.setup_cleanup_state = None
    handle.setup_generation = 1
    handle.setup_key = ("fleet",)
    handle.worker_args_key = ("policy",)
    setup_token = mesh_setup.current_setup_token(handle)
    handle.world = 2
    submissions = []
    blocked = RenderSession(timeout_s=0)

    def submit(index, _request, **_kwargs):
        if len(submissions) == 2:
            # The first wave is consumed and holds no lease, but the fleet call
            # is one session and holds its bind on the handle until every wave is collected.
            with pytest.raises(ConcurrentRenderSessionError):
                blocked.bind(handle)
        authority = _kwargs.pop("authority")
        future = mesh_setup.dispatch_sample(
            handle,
            object,
            setup_token=_kwargs.pop("setup_token"),
            authority=authority,
        )
        submissions.append((index, future))
        return future

    handle.submit_sample_to = submit
    handle.collect_one = lambda *_args, **_kwargs: {
        "latent": "wire", "host": "test", "sample_s": 0.1}
    handle.cancel_sample = lambda *_args, **_kwargs: None
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle, topology_preset="uly1", attention="TORCH_FLASH",
            sync_ulysses=False, worker_args={}),
        unet_name="model.safetensors",
    )

    monkeypatch.setattr(fleet_mod, "_enforce_persisted_quarantine", lambda *_args: None)
    monkeypatch.setattr(fleet_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "_fleet_worker_policy",
        lambda *_args: fleet_mod._FleetAuthorization({}, {}, None))
    monkeypatch.setattr(
        mesh_setup, "ensure_request_setup", lambda *_args, **_kwargs: setup_token)
    monkeypatch.setattr(fleet_mod, "ProgressReceiver", _Progress)
    monkeypatch.setattr(fleet_mod, "pack_latent", lambda value: value)
    monkeypatch.setattr(fleet_mod, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(
        fleet_mod, "read_latent_result",
        lambda _value, _guard=None: torch.zeros(1, 1, 1, 1))

    result, = fleet_mod.DGXMonarchFleetKSampler().fleet(
        [spec], [object()], ["NEG"], [{"samples": torch.zeros(1)}], [""], [42],
        [2], [1.0], ["euler"], ["simple"],
        positive=["POS-0", "POS-1", "POS-2"],
    )

    assert tuple(result["samples"].shape) == (3, 1, 1, 1)
    assert [index for index, _future in submissions] == [0, 1, 0]
    assert all(future.state == "consumed" for _index, future in submissions)
    assert handle.sample_leases == {}
    blocked.bind(handle)
    blocked.close()


def test_pending_primary_survives_all_baseexception_cleanup(monkeypatch):
    class Primary(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = Primary("collection stopped")
    session_closed = []

    class Progress:
        def __exit__(self, *_args):
            raise CleanupStop("progress cleanup stopped")

    def collect(*_args, **_kwargs):
        raise primary

    def abandon(_future):
        raise CleanupStop("lease cleanup stopped")

    def finish():
        raise CleanupStop("telemetry cleanup stopped")

    handle = _real_handle()
    handle.collect_sample = collect
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)
    owner.bind(handle)
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)
    monkeypatch.setattr(mesh_setup, "abandon_sample", abandon)
    monkeypatch.setattr(telemetry.render_progress, "finish", finish)
    pending = PendingRender(
        handle, object(), Progress(), None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: (session_closed.append(True), owner.close()),
    )

    with pytest.raises(Primary) as caught:
        pending.result()

    assert caught.value is primary
    assert session_closed == [True]
    assert pending._state == "closing"
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize("cancel_first", [False, True])
def test_pending_close_publication_cancellation_outranks_ordinary_failure(
    monkeypatch, cancel_first,
):
    class Cancelled(KeyboardInterrupt):
        def __bool__(self):
            raise AssertionError("close exception truthiness must not be evaluated")

    ordinary = RuntimeError("first close publication failed")
    cancellation = Cancelled("close publication cancelled")
    failures = iter(
        (cancellation, ordinary) if cancel_first else (ordinary, cancellation))
    pending = PendingRender(
        SimpleNamespace(), object(), SimpleNamespace(), None, {}, 1.0,
        "render", lambda *_args: {},
    )

    def fail_begin_close(_abandoned):
        raise next(failures)

    monkeypatch.setattr(pending, "_begin_close", fail_begin_close)

    with pytest.raises(Cancelled) as raised:
        pending._close(abandoned=True)

    assert raised.value is cancellation
    assert raised.value.__cause__ is ordinary
    assert pending._state == "open"


def test_pending_close_publication_same_exception_never_self_chains(monkeypatch):
    failure = RuntimeError("same close publication failure")
    calls = 0
    pending = PendingRender(
        SimpleNamespace(), object(), SimpleNamespace(), None, {}, 1.0,
        "render", lambda *_args: {},
    )

    def fail_begin_close(_abandoned):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(pending, "_begin_close", fail_begin_close)

    with pytest.raises(RuntimeError) as raised:
        pending._close(abandoned=True)

    assert calls == 2
    assert raised.value is failure
    assert raised.value.__cause__ is not failure
    assert raised.value.__context__ is not failure
    assert pending._state == "open"


@pytest.mark.parametrize("explosive", [False, True])
def test_pending_progress_cleanup_preserves_exact_first_baseexception(
    monkeypatch, explosive
):
    class FirstFailure(BaseException):
        def __bool__(self):
            if explosive:
                raise AssertionError("exception truthiness must not be evaluated")
            return False

    first = FirstFailure("first progress cleanup failure")

    class Progress:
        def __init__(self):
            self.calls = 0

        def __exit__(self, *_args):
            self.calls += 1
            if self.calls == 1:
                raise first
            raise BaseException("second progress cleanup failure")

    monkeypatch.setattr(mesh_setup, "abandon_sample", lambda _future: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = PendingRender(
        SimpleNamespace(), object(), Progress(), None, {}, 1.0, "render",
        lambda *_args: {},
    )

    with pytest.raises(BaseException) as raised:
        pending._close(abandoned=True)

    assert raised.value is first


def test_pending_release_failure_surfaces_without_primary(monkeypatch):
    class LeaseFailure(BaseException):
        pass

    failure = LeaseFailure("release failed")
    handle = SimpleNamespace(collect_sample=lambda *_args, **_kwargs: [])
    progress = SimpleNamespace(__exit__=lambda *_args: None)

    def release(_future):
        raise failure

    monkeypatch.setattr(mesh_setup, "release_sample", release)
    monkeypatch.setattr(pending_mod, "_throw_if_comfy_interrupted", lambda: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = PendingRender(
        handle, object(), progress, None, {}, 1.0, "render", lambda *_args: {})

    with pytest.raises(LeaseFailure) as caught:
        pending.result()

    assert caught.value is failure
    assert pending._state == "closing"


def test_pending_retries_tokenized_telemetry_before_terminal_close(monkeypatch):
    from dgx_monarch.telemetry import RenderProgress

    class StopNow(BaseException):
        pass

    interruption = StopNow("telemetry finish interrupted")
    progress_state = RenderProgress()
    token = object()
    progress_state.start(2, token=token)
    actual_finish = progress_state.finish
    calls = []

    def interrupt_first_finish(received_token=None):
        calls.append(received_token)
        if len(calls) == 1:
            raise interruption
        actual_finish(received_token)

    monkeypatch.setattr(progress_state, "finish", interrupt_first_finish)
    monkeypatch.setattr(telemetry, "render_progress", progress_state)
    monkeypatch.setattr(mesh_setup, "release_sample", lambda _future: None)
    pending = PendingRender(
        SimpleNamespace(collect_sample=lambda *_args, **_kwargs: []),
        object(), SimpleNamespace(__exit__=lambda *_args: None), None, {},
        1.0, "render", lambda *_args: {"ok": True}, telemetry_token=token)

    with pytest.raises(StopNow) as caught:
        pending.result()

    assert caught.value is interruption
    assert calls == [token, token]
    assert pending._state == "closed"
    assert progress_state.snapshot()["active"] is False


def test_duplicate_result_cannot_abandon_another_collecting_thread(monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    retired = []
    outputs = []

    def collect(*_args, **_kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return []

    handle = SimpleNamespace(collect_sample=collect)
    monkeypatch.setattr(
        mesh_setup, "release_sample", lambda _future: retired.append("release"))
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: retired.append("abandon"))
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda *_args: None)
    pending = PendingRender(
        handle, object(), SimpleNamespace(__exit__=lambda *_args: None),
        None, {}, 1.0, "render", lambda *_args: {"ok": True})

    collector = threading.Thread(
        target=lambda: outputs.append(pending.result()))
    collector.start()
    assert entered.wait(timeout=1)

    with pytest.raises(RuntimeError, match="already collecting"):
        pending.result()
    # Only the collecting thread owns the close, so abandon here is a no-op.
    pending.abandon()

    assert retired == []
    assert pending._state == "collecting"
    release.set()
    collector.join(timeout=2)
    assert not collector.is_alive()
    assert outputs == [{"ok": True}]
    assert retired == ["release"]
    assert pending._state == "closed"


def test_pending_timeout_cancel_baseexception_wins_after_cleanup(monkeypatch):
    class CancelStop(BaseException):
        pass

    timeout = TimeoutError("render timed out")
    cancel_stop = CancelStop("cancel interrupted")
    cancelled = []
    session_closed = []

    def collect(*_args, **_kwargs):
        raise timeout

    def cancel(*_args, **_kwargs):
        cancelled.append(True)
        raise cancel_stop

    handle = _real_handle()
    handle.collect_sample = collect
    handle.cancel_sample = cancel
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)
    owner.bind(handle)
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)
    progress = SimpleNamespace(__exit__=lambda *_args: None)
    monkeypatch.setattr(mesh_setup, "abandon_sample", lambda _future: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = PendingRender(
        handle, object(), progress, None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: (session_closed.append(True), owner.close()),
    )

    with pytest.raises(CancelStop) as caught:
        pending.result()

    assert caught.value is cancel_stop
    assert cancel_stop.__cause__ is timeout
    assert cancel_stop.__cause__ is not cancel_stop
    assert cancel_stop.__context__ is not cancel_stop
    assert cancelled == [True, True]
    assert session_closed == [True]
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize("cancel_recovers", [True, False])
def test_pending_timeout_reports_only_terminal_cancel_failure(
    monkeypatch, cancel_recovers,
):
    timeout = TimeoutError("render timed out")
    cancel_failures = [
        RuntimeError("first cancel attempt failed"),
        RuntimeError("second cancel attempt failed"),
    ]
    cancel_calls = 0

    def collect(*_args, **_kwargs):
        raise timeout

    def cancel(*_args, **_kwargs):
        nonlocal cancel_calls
        failure = cancel_failures[cancel_calls]
        cancel_calls += 1
        if cancel_calls == 1 or not cancel_recovers:
            raise failure

    handle = _real_handle()
    handle.collect_sample = collect
    handle.cancel_sample = cancel
    monkeypatch.setattr(mesh_setup, "abandon_sample", lambda _future: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = PendingRender(
        handle, object(), SimpleNamespace(__exit__=lambda *_args: None),
        None, {}, 1.0, "render", lambda *_args: {},
    )

    with pytest.raises(TimeoutError) as caught:
        pending.result()

    notes = getattr(timeout, "__notes__", ())
    assert caught.value is timeout
    assert cancel_calls == 2
    assert any("timeout cancellation failed" in note for note in notes) is (
        not cancel_recovers)
    assert pending._state == "closed"


def test_fleet_timeout_cancel_baseexception_wins_after_wave_drain(monkeypatch):
    class CancelStop(BaseException):
        pass

    timeout = TimeoutError("fleet job timed out")
    cancel_stop = CancelStop("cancel interrupted")
    retired = []

    class Handle:
        def __init__(self):
            self.collected = []
            self.cancelled = []

        def collect_one(self, future, timeout_s):
            self.collected.append(future)
            if future == "slow":
                raise timeout
            return {"latent": future, "host": "test", "sample_s": 0.1}

        def cancel_sample(self, render_id, wait=False):
            self.cancelled.append((render_id, wait))
            raise cancel_stop

    progress = [
        SimpleNamespace(__exit__=lambda *_args: None),
        SimpleNamespace(__exit__=lambda *_args: None),
    ]
    handle = Handle()
    results = [None, None]
    monkeypatch.setattr(fleet_mod, "read_latent_result", lambda value: value)
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda future: retired.append(("abandon", future)))
    monkeypatch.setattr(
        mesh_setup, "release_sample", lambda future: retired.append(("release", future)))
    monkeypatch.setattr(
        fleet_mod, "mark_defunct_on_supervision_failure", lambda *_args: None)

    with pytest.raises(CancelStop) as caught:
        fleet_mod._collect_fleet_wave(
            handle,
            [(0, "slow", progress[0], "render-0"),
             (1, "ok", progress[1], "render-1")],
            results, 1.0, 2,
        )

    assert caught.value is cancel_stop
    assert cancel_stop.__cause__ is not cancel_stop
    assert cancel_stop.__context__ is not cancel_stop
    assert handle.cancelled[0] == ("render-0", False)
    assert set(handle.cancelled) <= {
        ("render-0", False), ("render-1", False)}
    assert sorted(handle.collected) == ["ok", "slow"]
    assert retired == [("abandon", "slow"), ("release", "ok")]
    assert results == [None, "ok"]


def _retiring_pending(monkeypatch, collect, finish=lambda *_args: {"ok": True}):
    """A PendingRender whose only observable effect is how it retires."""
    retired: list[str] = []
    monkeypatch.setattr(
        mesh_setup, "release_sample", lambda _future: retired.append("release"))
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: retired.append("abandon"))
    monkeypatch.setattr(pending_mod, "_throw_if_comfy_interrupted", lambda: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda *_args: None)
    handle = SimpleNamespace(collect_sample=collect, cancel_sample=lambda *_a, **_k: None)
    pending = PendingRender(
        handle, object(), SimpleNamespace(__exit__=lambda *_args: None),
        None, {}, 1.0, "render", finish)
    return pending, retired


def _typed_refusal(message: str = "padded ring on a maskless kernel."):
    """A worker-raised dgxm refusal exactly as the collect path receives it."""
    from monarch.actor import ActorError

    from dgx_monarch.refusal import PanelAction, RefusalClass, refusal

    return ActorError(RuntimeError(refusal(
        RefusalClass.KNOWN_WRONG, message, guard="ring_pad", waivable=True,
        panel_action=PanelAction("Render under waiver (output stamped)",
                                 "DGXM_WAIVE_KNOWN_WRONG", env_value="ring_pad"),
        troubleshooting=21)))


def test_a_typed_worker_refusal_consumes_its_lease_instead_of_abandoning_it(
        monkeypatch):
    """A refusal ends a render as completely as a result does.

    Abandoning its lease would block every later sample until the fleet is
    recycled, the re-queue the refusal asks for included. typed_worker_refusal
    in nodes/pending.py says why nothing is left running.
    """
    from monarch.actor import ActorError

    refusal_exc = _typed_refusal()

    def collect(*_args, **_kwargs):
        raise refusal_exc

    pending, retired = _retiring_pending(monkeypatch, collect)
    with pytest.raises(ActorError):
        pending.result()

    assert retired == ["release"]
    assert pending._state == "closed"


def test_every_close_retires_the_dispatch_audit_facts(monkeypatch):
    """Every close retires its own render's waiver audit facts and no other's.

    A refused, failed or abandoned render never reaches ``stamp_result``, which
    retires the audit facts only of a render that spent a waiver. Without
    ``consent_waiver.retire_audit`` on close, leaked entries fill the
    ``_PENDING_AUDIT_LIMIT`` window and later renders are dispatched without their waivers.
    """
    from monarch.actor import ActorError

    from dgx_monarch.nodes import consent_waiver

    consent_waiver.reset_pending_audit()
    try:
        consent_waiver._PENDING_AUDIT["render"] = {"combo_key": "c"}
        consent_waiver._PENDING_AUDIT["other-render"] = {"combo_key": "c"}
        refusal_exc = _typed_refusal()

        def collect(*_args, **_kwargs):
            raise refusal_exc

        pending, _retired = _retiring_pending(monkeypatch, collect)
        with pytest.raises(ActorError):
            pending.result()
        assert "render" not in consent_waiver._PENDING_AUDIT
        assert "other-render" in consent_waiver._PENDING_AUDIT

        consent_waiver._PENDING_AUDIT["render"] = {"combo_key": "c"}
        abandoned, _ = _retiring_pending(monkeypatch, lambda *_a, **_k: [])
        abandoned.abandon()
        assert "render" not in consent_waiver._PENDING_AUDIT
        assert "other-render" in consent_waiver._PENDING_AUDIT
    finally:
        consent_waiver.reset_pending_audit()


def test_an_untagged_actor_endpoint_failure_still_abandons_its_lease(monkeypatch):
    """An ActorError alone does not release the lease.

    An untagged endpoint failure after packing may leave the actor live with
    unresolved generation-bound result or descriptor ownership. Only a tagged
    guard refusal is known to fire before any such change.
    """
    from monarch.actor import ActorError

    crash = ActorError(RuntimeError("CUDA error: an illegal memory access"))

    def collect(*_args, **_kwargs):
        raise crash

    pending, retired = _retiring_pending(monkeypatch, collect)
    with pytest.raises(ActorError):
        pending.result()

    assert retired == ["abandon"]


def test_a_middle_tag_in_an_actor_failure_still_abandons_its_lease(monkeypatch):
    """Only a leading tag counts; one inside an artifact name is text, not a guard's refusal."""
    from monarch.actor import ActorError

    crash = ActorError(RuntimeError(
        "CUDA error while loading models/[dgxm:P]checkpoint.safetensors"))

    def collect(*_args, **_kwargs):
        raise crash

    pending, retired = _retiring_pending(monkeypatch, collect)
    with pytest.raises(ActorError):
        pending.result()

    assert retired == ["abandon"]


def test_a_collect_timeout_still_abandons_its_lease(monkeypatch):
    """A timeout is ambiguous: the render may still be running."""
    def collect(*_args, **_kwargs):
        raise TimeoutError("collect timed out")

    pending, retired = _retiring_pending(monkeypatch, collect)
    with pytest.raises(TimeoutError):
        pending.result()

    assert retired == ["abandon"]


def test_a_refusal_raised_after_collect_still_abandons_its_lease(monkeypatch):
    """Past the collect the workers have packed, so the backing is live.

    A refusal tag here is not evidence of a clean render: the failure comes from the
    driver's own result assembly, and the latents it was reading are the
    backing an abandoned lease protects.
    """
    from monarch.actor import ActorError

    late = _typed_refusal("cross-rank identity gate failed.")

    def finish(*_args):
        raise late

    pending, retired = _retiring_pending(
        monkeypatch, lambda *_a, **_k: [], finish=finish)
    with pytest.raises(ActorError):
        pending.result()

    assert retired == ["abandon"]
