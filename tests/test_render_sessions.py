"""Logical render-session exclusion and terminal cleanup regressions."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import dgx_monarch.mesh_session as session_mod
import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.gate as gate_mod
import dgx_monarch.nodes.gate_identity as gate_identity_mod
import dgx_monarch.nodes.render_submit as submit_mod
from dgx_monarch import mesh_setup
from dgx_monarch.nodes.common import MeshSpec, ModelSpec
from dgx_monarch.nodes.pending import PendingRenderHandoff
from dgx_monarch.nodes.pipeline import RenderPipeline
from dgx_monarch.nodes.render_session import (
    ConcurrentRenderSessionError,
    RenderSession,
    RetiredRenderHandleError,
    claim_render_session,
)
from render_sessions_helpers import (
    _Progress,
    _ready_real_handle,
    _real_handle,
    _stub_direct_submit,
)


def test_real_handle_allows_one_logical_session_at_a_time():
    handle = _real_handle()
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)

    owner.bind(handle)
    owner.bind(handle)  # pipeline submits may bind the shared session repeatedly
    with pytest.raises(ConcurrentRenderSessionError, match="another render session"):
        contender.bind(handle)

    owner.close()
    contender.bind(handle)
    contender.close()


def test_render_session_rejects_handle_retired_at_claim_linearization():
    handle, _token = _ready_real_handle()
    handle.teardown_complete = False
    handle.replacement_blocked = None
    handle.defunct = True
    session = RenderSession(timeout_s=0)

    with pytest.raises(RetiredRenderHandleError, match="mesh retired"):
        session.bind(handle)

    # Failed bind compensated the shared owner, so a healed handle is usable.
    handle.defunct = False
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_direct_submit_resolves_once_after_claim_detects_retired_handle(
    monkeypatch,
):
    retired = _real_handle()
    replacement = _real_handle()
    replacement.world = 1
    replacement.cancel_sample = lambda *_args, **_kwargs: None
    submitted = []
    spec = _stub_direct_submit(
        monkeypatch,
        retired,
        lambda handle, *_args, **_kwargs: submitted.append(handle) or object(),
    )
    resolved = []

    def ensure(handle):
        resolved.append(handle)
        return retired if len(resolved) == 1 else replacement

    real_claim = session_mod.claim_render_session
    claims = []

    def claim(handle, candidate):
        claims.append(handle)
        if len(claims) == 1:
            retired.defunct = True
            raise RetiredRenderHandleError("mesh retired while claiming")
        return real_claim(handle, candidate)

    monkeypatch.setattr(common, "ensure_live", ensure)
    monkeypatch.setattr(submit_mod, "claim_render_session", claim)
    pending = common.submit_render(
        spec,
        {},
        {"samples": object()},
        cfg_value=1.0,
        steps_hint=2,
        handoff=PendingRenderHandoff(),
    )
    pending.abandon()

    assert claims == [retired, replacement]
    assert submitted == [replacement]


def test_progress_cancellation_uses_fire_and_forget_broadcast(monkeypatch):
    handle = _real_handle()
    callbacks = []
    cancellations = []

    class TrackingProgress(_Progress):
        def __init__(self, *_args, on_cancel=None, **_kwargs):
            callbacks.append(on_cancel)

    spec = _stub_direct_submit(
        monkeypatch, handle, lambda *_args, **_kwargs: object())
    handle.cancel_sample = lambda render_id, *, wait: cancellations.append(
        (render_id, wait))
    monkeypatch.setattr(submit_mod, "ProgressReceiver", TrackingProgress)

    pending = common.submit_render(
        spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
        handoff=PendingRenderHandoff())
    assert len(callbacks) == 1 and callbacks[0] is not None
    callbacks[0]()

    assert cancellations == [(pending._render_id, False)]
    handle.cancel_sample = lambda *_args, **_kwargs: None
    pending.abandon()


def test_render_session_timeout_includes_handle_lock_acquisition():
    handle = _real_handle()
    locked = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    outcome = []

    def hold_setup_lock():
        with handle.lock:
            locked.set()
            release.wait(timeout=1)

    def contend():
        session = RenderSession(timeout_s=0.01)
        try:
            session.bind(handle)
        except BaseException as exc:
            outcome.append(exc)
        finally:
            session.close()
            finished.set()

    holder = threading.Thread(target=hold_setup_lock)
    holder.start()
    assert locked.wait(timeout=1)
    contender = threading.Thread(target=contend)
    contender.start()

    # Release only after observing the bounded result. With an unbounded
    # condition-lock acquisition this remains false until release is set.
    bounded = finished.wait(timeout=0.2)
    release.set()
    holder.join(timeout=1)
    contender.join(timeout=1)

    assert bounded is True
    assert len(outcome) == 1
    assert isinstance(outcome[0], ConcurrentRenderSessionError)
    assert "did not drain" in str(outcome[0])


def test_direct_contender_cannot_change_setup_before_session_claim(monkeypatch):
    """A direct render cannot run setup while another session owns the handle.

    Fleet's session owns it through every zero-lease gap between waves."""
    handle = _real_handle()
    owner = RenderSession(timeout_s=0)
    owner.bind(handle)
    spec = _stub_direct_submit(
        monkeypatch, handle, lambda *_args, **_kwargs: object())
    setup_calls = []
    monkeypatch.setattr(
        mesh_setup,
        "ensure_request_setup",
        lambda *_args, **_kwargs: setup_calls.append(True),
    )
    monkeypatch.setattr(session_mod, "_DEFAULT_CLAIM_TIMEOUT_S", 0)

    with pytest.raises(ConcurrentRenderSessionError, match="another render session"):
        common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
            handoff=PendingRenderHandoff())

    assert setup_calls == []
    owner.close()


def test_real_submit_requires_handoff_before_session_or_setup(monkeypatch):
    handle = _real_handle()
    spec = _stub_direct_submit(
        monkeypatch, handle, lambda *_args, **_kwargs: object())
    setup_calls = []
    monkeypatch.setattr(
        mesh_setup,
        "ensure_request_setup",
        lambda *_args, **_kwargs: setup_calls.append(True),
    )

    with pytest.raises(ValueError, match="PendingRenderHandoff"):
        common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2)

    assert setup_calls == []
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_claim_return_interruption_keeps_caller_cleanup_authority(monkeypatch):
    """An interrupt after the claim binds but before it returns cannot strand the owner."""
    class StopNow(BaseException):
        pass

    handle = _real_handle()
    spec = _stub_direct_submit(
        monkeypatch, handle, lambda *_args, **_kwargs: object())

    def bind_then_stop(bound_handle, candidate):
        candidate.bind(bound_handle)
        raise StopNow("claim return interrupted")

    monkeypatch.setattr(submit_mod, "claim_render_session", bind_then_stop)
    with pytest.raises(StopNow):
        common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
            handoff=PendingRenderHandoff())

    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_bind_condition_acquire_interrupt_leaves_no_owner(monkeypatch):
    class StopNow(BaseException):
        pass

    handle = _real_handle()
    session = RenderSession(timeout_s=0)

    def stop_before_claim(*_args):
        raise StopNow("condition acquire interrupted")

    original_claim = session_mod._claim_owner
    monkeypatch.setattr(session_mod, "_claim_owner", stop_before_claim)
    with pytest.raises(StopNow):
        session.bind(handle)

    monkeypatch.setattr(session_mod, "_claim_owner", original_claim)
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_bind_interrupted_after_publish_compensates_owner(monkeypatch):
    class StopNow(BaseException):
        pass

    handle = _real_handle()
    session = RenderSession(timeout_s=0)
    original_publish = session_mod._publish_owner

    def publish_then_stop(state, owner):
        original_publish(state, owner)
        raise StopNow("owner publication interrupted")

    monkeypatch.setattr(session_mod, "_publish_owner", publish_then_stop)
    with pytest.raises(StopNow):
        session.bind(handle)

    assert session_mod._handle_state(handle).owner is None
    monkeypatch.setattr(session_mod, "_publish_owner", original_publish)
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_close_retries_interrupted_owner_clear(monkeypatch):
    class StopNow(BaseException):
        pass

    handle = _real_handle()
    session = RenderSession(timeout_s=0)
    session.bind(handle)
    original_clear = session_mod._clear_owner
    calls = []

    def interrupt_once(state, owner):
        calls.append(True)
        if len(calls) == 1:
            raise StopNow("owner clear interrupted")
        original_clear(state, owner)

    monkeypatch.setattr(session_mod, "_clear_owner", interrupt_once)
    with pytest.raises(StopNow):
        session.close()

    assert len(calls) == 2
    assert session_mod._handle_state(handle).owner is None
    monkeypatch.setattr(session_mod, "_clear_owner", original_clear)
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_mutation_context_retries_first_cleanup_interruption(monkeypatch):
    class StopNow(BaseException):
        pass

    handle = _real_handle()
    actual_close = session_mod._close_candidate_preserving_primary
    cleanup_calls = []

    def interrupt_first_cleanup(candidate, primary):
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            raise StopNow("first mutation-session cleanup interrupted")
        actual_close(candidate, primary)

    monkeypatch.setattr(
        session_mod, "_close_candidate_preserving_primary", interrupt_first_cleanup)

    with pytest.raises(StopNow, match="first mutation-session cleanup interrupted"):
        with session_mod.mutation_render_session(handle):
            pass

    assert cleanup_calls == [True, True]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_mutation_primary_survives_first_cleanup_helper_interruption(monkeypatch):
    class Primary(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = Primary("mutation body interrupted")
    handle = _real_handle()
    actual_close = session_mod._close_candidate_preserving_primary
    cleanup_calls = []

    def interrupt_first_cleanup(candidate, received_primary):
        cleanup_calls.append(received_primary)
        if len(cleanup_calls) == 1:
            raise CleanupStop("first mutation cleanup helper call interrupted")
        actual_close(candidate, received_primary)

    monkeypatch.setattr(
        session_mod, "_close_candidate_preserving_primary", interrupt_first_cleanup)

    with pytest.raises(BaseException) as raised:
        with session_mod.mutation_render_session(handle):
            raise primary

    assert cleanup_calls == [primary, primary]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()
    assert raised.value is primary


def test_dispatch_sample_retries_first_cleanup_interruption(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, token = _ready_real_handle()
    authority = mesh_setup.SetupBoundFuture.prepared(handle, token.generation)
    actual_close = mesh_setup._close_sample_candidate
    cleanup_calls = []

    def interrupt_first_cleanup(candidate, primary):
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            raise StopNow("first sample-session cleanup interrupted")
        actual_close(candidate, primary)

    monkeypatch.setattr(
        mesh_setup, "_close_sample_candidate", interrupt_first_cleanup)

    with pytest.raises(StopNow, match="first sample-session cleanup interrupted"):
        mesh_setup.dispatch_sample(
            handle, object, setup_token=token, authority=authority)

    assert cleanup_calls == [True, True]
    contender = RenderSession(timeout_s=0)
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)
    authority.abandon()
    contender.bind(handle)
    contender.close()


def test_real_sample_dispatch_requires_prepared_authority_before_claim():
    handle, token = _ready_real_handle()
    sent = []

    with pytest.raises(ValueError, match="caller-prepared authority"):
        mesh_setup.dispatch_sample(
            handle, lambda: sent.append(True), setup_token=token)

    assert sent == []
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_sample_dispatch_outer_boundary_abandons_tracked_authority(
    monkeypatch,
):
    class StopNow(BaseException):
        pass

    handle, token = _ready_real_handle()
    authority = mesh_setup.prepare_sample(handle, token)
    assert authority is not None
    actual_abandon = authority.abandon_after_dispatch
    abandon_calls = []

    def interrupt_first_abandon(exc):
        abandon_calls.append(exc)
        if len(abandon_calls) == 1:
            raise StopNow("authority cleanup call boundary interrupted")
        actual_abandon(exc)

    monkeypatch.setattr(
        authority, "abandon_after_dispatch", interrupt_first_abandon)
    monkeypatch.setattr(
        mesh_setup,
        "_dispatch_sample_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StopNow("dispatch helper call boundary interrupted")),
    )

    with pytest.raises(StopNow, match="dispatch helper call boundary interrupted"):
        mesh_setup.dispatch_sample(
            handle, object, setup_token=token, authority=authority)

    assert authority.state == "abandoned"
    assert len(abandon_calls) == 2
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {}
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_abandon_after_dispatch_cannot_mask_cancellation_with_diagnostics(
    monkeypatch,
):
    class BrokenCancellation(KeyboardInterrupt):
        def add_note(self, _note):
            raise RuntimeError("note publication failed")

    class BrokenCleanup(RuntimeError):
        def __repr__(self):
            raise RuntimeError("cleanup repr failed")

    handle, token = _ready_real_handle()
    authority = mesh_setup.SetupBoundFuture.prepared(handle, token.generation)
    cleanup = BrokenCleanup("sample authority cleanup failed")
    calls = []

    def fail_abandon():
        calls.append(True)
        raise cleanup

    monkeypatch.setattr(authority, "abandon", fail_abandon)
    primary = BrokenCancellation("dispatch cancelled")
    with pytest.raises(BrokenCancellation) as exc_info:
        try:
            raise primary
        except BaseException as exc:
            authority.abandon_after_dispatch(exc)
            raise

    assert exc_info.value is primary and calls == [True, True]


def test_gate_retries_first_session_cleanup_interruption(monkeypatch):
    class StopNow(BaseException):
        pass

    handle = _real_handle()
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle,
            topology_preset="uly1",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args={},
        ),
        unet_name="model.safetensors",
    )
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "_run_identity_ceremony_bound",
        lambda *_args, **_kwargs: {"verdict": "INCONCLUSIVE"},
    )
    actual_close = gate_mod._close_gate_session
    cleanup_calls = []

    def interrupt_first_cleanup(session, primary):
        cleanup_calls.append(True)
        if len(cleanup_calls) == 1:
            raise StopNow("first Gate session cleanup interrupted")
        actual_close(session, primary)

    monkeypatch.setattr(gate_mod, "_close_gate_session", interrupt_first_cleanup)

    with pytest.raises(StopNow, match="first Gate session cleanup interrupted"):
        gate_mod.run_identity_ceremony(
            spec, {}, {}, 1.0, 1, "manual")

    assert cleanup_calls == [True, True]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_gate_bind_failure_never_mutates_workers_without_authority(monkeypatch):
    handle = _real_handle()
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle,
            topology_preset="uly1",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args={"lora_low_rss": True, "slab_weights": True},
        ),
        unet_name="model.safetensors",
    )
    owner = RenderSession()
    owner.bind(handle)
    policy_calls = []

    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(gate_mod, "RenderSession", lambda: RenderSession(timeout_s=0))
    monkeypatch.setattr(
        gate_identity_mod,
        "apply_worker_policy",
        lambda *_args, **_kwargs: policy_calls.append(True),
    )

    with pytest.raises(ConcurrentRenderSessionError):
        gate_mod.run_identity_ceremony(
            spec, {}, {}, 1.0, 1, "auto_first_use")

    assert policy_calls == []
    owner.close()


def test_auto_gate_quarantines_before_releasing_ceremony_owner(monkeypatch):
    handle = _real_handle()
    handle.setup_key = ("ready",)
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle,
            topology_preset="uly1",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args={"lora_low_rss": True, "slab_weights": True},
        ),
        unet_name="model.safetensors",
    )
    contender = RenderSession(timeout_s=0)
    observed = []
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "_run_identity_ceremony_bound",
        lambda *_args, **_kwargs: {"verdict": "INCONCLUSIVE"},
    )

    def apply_policy(mesh_handle, worker_args, timeout_s=600):
        assert mesh_handle is handle and timeout_s == 600
        assert worker_args["lora_low_rss"] is False
        assert worker_args["slab_weights"] is False
        with pytest.raises(ConcurrentRenderSessionError):
            contender.bind(handle)
        observed.append(dict(worker_args))
        return []

    monkeypatch.setattr(gate_identity_mod, "apply_worker_policy", apply_policy)

    result = gate_mod.run_identity_ceremony(
        spec, {}, {}, 1.0, 1, "auto_first_use")

    assert result["verdict"] == "INCONCLUSIVE"
    assert observed == [{"lora_low_rss": False, "slab_weights": False}]
    contender.bind(handle)
    contender.close()


def test_auto_gate_retries_interrupted_atomic_quarantine_before_owner_release(
    monkeypatch,
):
    class StopNow(BaseException):
        pass

    handle = _real_handle()
    handle.setup_key = ("ready",)
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle,
            topology_preset="uly1",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args={"lora_low_rss": True, "slab_weights": True},
        ),
        unet_name="model.safetensors",
    )
    contender = RenderSession(timeout_s=0)
    actual_quarantine = gate_identity_mod.quarantine_unproven_paths
    quarantine_calls = []
    policy_calls = []

    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "_run_identity_ceremony_bound",
        lambda *_args, **_kwargs: {"verdict": "INCONCLUSIVE"},
    )

    def interrupt_first_quarantine(model, cause=None):
        actual_quarantine(model, cause)
        quarantine_calls.append(True)
        if len(quarantine_calls) == 1:
            raise StopNow("quarantine publication return interrupted")

    def apply_policy(mesh_handle, worker_args, timeout_s=600):
        assert mesh_handle is handle
        assert worker_args["lora_low_rss"] is False
        assert worker_args["slab_weights"] is False
        with pytest.raises(ConcurrentRenderSessionError):
            contender.bind(handle)
        policy_calls.append(dict(worker_args))
        return []

    monkeypatch.setattr(
        gate_identity_mod, "quarantine_unproven_paths",
        interrupt_first_quarantine)
    monkeypatch.setattr(gate_identity_mod, "apply_worker_policy", apply_policy)

    with pytest.raises(StopNow, match="quarantine publication return interrupted"):
        gate_mod.run_identity_ceremony(
            spec, {}, {}, 1.0, 1, "auto_first_use")

    assert quarantine_calls == [True, True]
    assert policy_calls == [{"lora_low_rss": False, "slab_weights": False}]
    assert spec.mesh.worker_args == {
        "lora_low_rss": False, "slab_weights": False}
    contender.bind(handle)
    contender.close()


def test_gate_primary_survives_first_cleanup_helper_interruption(monkeypatch):
    class Primary(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = Primary("Gate ceremony interrupted")
    handle = _real_handle()
    spec = ModelSpec(
        mesh=MeshSpec(
            handle=handle,
            topology_preset="uly1",
            attention="TORCH_FLASH",
            sync_ulysses=False,
            worker_args={},
        ),
        unet_name="model.safetensors",
    )
    monkeypatch.setattr(gate_mod, "ensure_live", lambda value: value)
    monkeypatch.setattr(
        gate_mod,
        "_run_identity_ceremony_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
    )
    actual_close = gate_mod._close_gate_session
    cleanup_calls = []

    def interrupt_first_cleanup(session, received_primary):
        cleanup_calls.append(received_primary)
        if len(cleanup_calls) == 1:
            raise CleanupStop("first Gate cleanup helper call interrupted")
        actual_close(session, received_primary)

    monkeypatch.setattr(gate_mod, "_close_gate_session", interrupt_first_cleanup)

    with pytest.raises(BaseException) as raised:
        gate_mod.run_identity_ceremony(spec, {}, {}, 1.0, 1, "manual")

    assert cleanup_calls == [primary, primary]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()
    assert raised.value is primary


def test_submit_primary_survives_first_guard_cleanup_interruption(monkeypatch):
    class Primary(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    primary = Primary("submit body interrupted")
    handle = _real_handle()

    def fail_after_claim(*args, **kwargs):
        guard = args[-1] if args else kwargs["guard"]
        guard.handle = handle
        guard.candidate.bind(handle)
        raise primary

    monkeypatch.setattr(submit_mod, "_submit_render_guarded", fail_after_claim)
    actual_cleanup = submit_mod._SubmitRenderGuard.cleanup
    cleanup_calls = []

    def interrupt_first_cleanup(guard, received_primary):
        cleanup_calls.append(received_primary)
        if len(cleanup_calls) == 1:
            raise CleanupStop("first submit guard cleanup call interrupted")
        actual_cleanup(guard, received_primary)

    monkeypatch.setattr(
        submit_mod._SubmitRenderGuard, "cleanup", interrupt_first_cleanup)

    with pytest.raises(BaseException) as raised:
        common.submit_render(object(), {}, {}, None, 1)

    assert cleanup_calls == [primary, primary]
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()
    assert raised.value is primary


def test_lightweight_handle_doubles_remain_lock_free():
    handle = SimpleNamespace()
    first = RenderSession(timeout_s=0)
    second = RenderSession(timeout_s=0)

    first.bind(handle)
    second.bind(handle)
    first.close()
    second.close()

    assert vars(handle) == {}


def test_pipeline_depth_shares_session_until_drain(monkeypatch):
    handle = _real_handle()
    claims = []

    class Pending:
        def __init__(self, seq):
            self.seq = seq

        def result(self):
            return {"seq": self.seq}

    def submit(_model, _request, _latent, _cfg, _steps, *, seq=0, **_kwargs):
        session, owned = claim_render_session(handle, RenderSession())
        claims.append((session, owned))
        return Pending(seq)

    monkeypatch.setattr(common, "submit_render", submit)
    pipeline = RenderPipeline(depth=2)
    pipeline.push(None, {}, {}, 1.0, 2)
    pipeline.push(None, {}, {}, 1.0, 2)

    contender = RenderSession(timeout_s=0)
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)
    assert claims == [(pipeline._session, False), (pipeline._session, False)]

    assert pipeline.drain() == [{"seq": 0}, {"seq": 1}]
    contender.bind(handle)
    contender.close()


def test_pipeline_baseexception_abort_releases_session(monkeypatch):
    class StopNow(BaseException):
        pass

    handle = _real_handle()

    class Pending:
        def result(self):
            raise StopNow("collection interrupted")

        def abandon(self):
            return None

    def submit(*_args, **_kwargs):
        _session, owned = claim_render_session(handle, RenderSession())
        assert owned is False
        return Pending()

    monkeypatch.setattr(common, "submit_render", submit)
    pipeline = RenderPipeline(depth=2)
    pipeline.push(None, {}, {}, 1.0, 2)
    contender = RenderSession(timeout_s=0)
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)

    with pytest.raises(StopNow):
        pipeline.drain()

    contender.bind(handle)
    contender.close()


def test_direct_submit_owns_session_until_pending_result(monkeypatch):
    handle = _real_handle()
    handle.lock = threading.RLock()
    handle.sample_leases = {1: 1}
    handle.abandoned_sample_leases = {}
    handle.deferred_supervision_error = None
    handle.collect_sample = lambda *_args, **_kwargs: [{"latent": "wire"}]
    future = mesh_setup.SetupBoundFuture(object(), handle, 1)
    spec = _stub_direct_submit(
        monkeypatch, handle, lambda *_args, **_kwargs: future)

    handoff = PendingRenderHandoff()
    pending = common.submit_render(
        spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
        handoff=handoff)
    handoff.clear(pending)
    contender = RenderSession(timeout_s=0)
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)

    assert pending.result() == {"samples": "done"}
    assert future.state == "consumed"
    contender.bind(handle)
    contender.close()


def test_reused_request_cannot_carry_a_stale_normal_residency_grant(monkeypatch):
    handle = _real_handle()
    submitted = []

    def submit(_handle, request, *_args, **_kwargs):
        submitted.append(dict(request))
        return object()

    spec = _stub_direct_submit(monkeypatch, handle, submit)
    request = {
        "_dgxm_normal_residency_grant": {"stale": True},
        "_dgxm_normal_policy": {"attention": "stale"},
        "_dgxm_normal_residency_mode": "required",
        "_dgxm_artifact_sets": [{"digest": "stale"}],
        "_dgxm_artifact_preflight": {"generation": -1},
    }
    pending = common.submit_render(
        spec,
        request,
        {"samples": object()},
        cfg_value=1.0,
        steps_hint=2,
        handoff=PendingRenderHandoff(),
    )
    pending.abandon()

    assert submitted[0]["_dgxm_normal_residency_mode"] == "stock"
    assert "_dgxm_normal_residency_grant" not in submitted[0]
    assert "_dgxm_normal_policy" not in submitted[0]
    assert "_dgxm_artifact_sets" not in submitted[0]


def test_direct_submission_baseexception_releases_session(monkeypatch):
    class SubmitStop(BaseException):
        pass

    failure = SubmitStop("submission interrupted")
    handle = _real_handle()

    def submit(*_args, **_kwargs):
        raise failure

    spec = _stub_direct_submit(monkeypatch, handle, submit)
    with pytest.raises(SubmitStop) as caught:
        common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
            handoff=PendingRenderHandoff())

    assert caught.value is failure
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_direct_submit_owns_authority_before_return_handoff(monkeypatch):
    """An interrupt after dispatch returns still has a reachable authority."""
    class StopNow(BaseException):
        pass

    handle, token = _ready_real_handle()
    handle.world = 1
    handle.cancel_sample = lambda *_args, **_kwargs: None
    handle._verify_request_artifacts = lambda *_args, **_kwargs: None
    accepted = []
    handle.workers = SimpleNamespace(
        sample=SimpleNamespace(
            call=lambda *_args, **_kwargs: (accepted.append(True), object())[1]
        )
    )
    actual_submit = mesh_setup.submit_sample
    authorities = []

    def submit_then_stop(
        mesh_handle, request, progress_port, setup_token, authority=None,
    ):
        submitted = actual_submit(
            mesh_handle, request, progress_port, setup_token, authority)
        authorities.append(submitted)
        raise StopNow("dispatch returned before caller result publication")

    spec = _stub_direct_submit(monkeypatch, handle, submit_then_stop)
    monkeypatch.setattr(
        mesh_setup, "ensure_request_setup", lambda *_args, **_kwargs: token)

    with pytest.raises(StopNow):
        common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
            handoff=PendingRenderHandoff())

    assert accepted == [True]
    assert len(authorities) == 1
    assert authorities[0].state == "abandoned"
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()
