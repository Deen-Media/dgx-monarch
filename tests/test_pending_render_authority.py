"""Pending render handoff, authority retirement and lease retry regressions."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import dgx_monarch.nodes.common as common
import dgx_monarch.nodes.render_submit as submit_mod
from dgx_monarch import mesh_setup, telemetry
from dgx_monarch.nodes.pending import PendingRender, PendingRenderHandoff
from dgx_monarch.nodes.pipeline import RenderPipeline
from dgx_monarch.nodes.render_session import (
    ConcurrentRenderSessionError,
    RenderSession,
    claim_render_session,
)
from render_sessions_helpers import (
    _Progress,
    _ready_real_handle,
    _real_handle,
    _stub_direct_submit,
)


def test_prepared_authority_baseexception_releases_direct_session(monkeypatch):
    class PrepareStop(BaseException):
        pass

    failure = PrepareStop("authority preparation interrupted")
    handle = _real_handle()
    spec = _stub_direct_submit(monkeypatch, handle, lambda *_args: object())
    monkeypatch.setattr(
        mesh_setup,
        "prepare_sample",
        lambda *_args: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(PrepareStop) as caught:
        common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
            handoff=PendingRenderHandoff(),
        )

    assert caught.value is failure
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_pending_constructor_return_gap_retires_direct_authority(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, token = _ready_real_handle()
    handle.world = 1
    handle.cancel_sample = lambda *_args, **_kwargs: None
    submitted = []
    progress = []

    class TrackingProgress(_Progress):
        def __init__(self, *_args, **_kwargs):
            self.closed = 0
            progress.append(self)

        def __exit__(self, *_args):
            self.closed += 1

    def submit(mesh_handle, _request, _port, setup_token, authority=None):
        future = mesh_setup.dispatch_sample(
            mesh_handle, object, setup_token=setup_token, authority=authority)
        submitted.append(future)
        return future

    spec = _stub_direct_submit(monkeypatch, handle, submit)
    monkeypatch.setattr(
        mesh_setup, "ensure_request_setup", lambda *_args, **_kwargs: token)
    monkeypatch.setattr(submit_mod, "ProgressReceiver", TrackingProgress)
    real_pending = submit_mod.PendingRender
    constructed = []

    def construct_then_stop(*args, **kwargs):
        constructed.append(real_pending(*args, **kwargs))
        raise StopNow("pending constructor returned before local publication")

    monkeypatch.setattr(submit_mod, "PendingRender", construct_then_stop)

    with pytest.raises(StopNow):
        common.submit_render(
            spec, {}, {"samples": object()}, cfg_value=1.0, steps_hint=2,
            handoff=PendingRenderHandoff())

    assert len(constructed) == 1
    assert len(submitted) == 1 and submitted[0].state == "abandoned"
    assert handle.sample_leases == {}
    assert progress[0].closed == 1
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


def test_run_render_handoff_abandons_return_lost_before_store(monkeypatch):
    class StopNow(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    abandoned = []
    abort_calls = []

    class Pending:
        _state = "open"

        def abandon(self):
            self._state = "closed"
            abandoned.append(True)

    def submit(*_args, handoff, **_kwargs):
        pending = Pending()
        handoff.publish(pending)
        raise StopNow("submit return interrupted before caller store")

    monkeypatch.setattr(common, "_maybe_auto_gate", lambda *_args: None)
    monkeypatch.setattr(common, "submit_render", submit)
    actual_abort = PendingRenderHandoff.abort

    def interrupt_first_abort(self):
        abort_calls.append(True)
        if len(abort_calls) == 1:
            raise CleanupStop("handoff abort call boundary interrupted")
        actual_abort(self)

    monkeypatch.setattr(PendingRenderHandoff, "abort", interrupt_first_abort)

    with pytest.raises(StopNow) as caught:
        common.run_render(object(), {}, {}, None, 1)

    assert isinstance(caught.value, StopNow)
    assert abort_calls == [True, True]
    assert abandoned == [True]


def test_pipeline_handoff_abandons_return_lost_before_deque_append(monkeypatch):
    class StopNow(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    handle = _real_handle()
    abandoned = []
    abort_calls = []

    class Pending:
        _state = "open"

        def abandon(self):
            self._state = "closed"
            abandoned.append(True)

    def submit(*_args, handoff, **_kwargs):
        _session, owned = claim_render_session(handle, RenderSession())
        assert owned is False
        pending = Pending()
        handoff.publish(pending)
        raise StopNow("submit return interrupted before deque append")

    monkeypatch.setattr(common, "submit_render", submit)
    actual_abort = PendingRenderHandoff.abort

    def interrupt_first_abort(self):
        abort_calls.append(True)
        if len(abort_calls) == 1:
            raise CleanupStop("pipeline handoff abort call boundary interrupted")
        actual_abort(self)

    monkeypatch.setattr(PendingRenderHandoff, "abort", interrupt_first_abort)
    pipeline = RenderPipeline(depth=2)

    with pytest.raises(StopNow) as caught:
        pipeline.push(None, {}, {}, 1.0, 2)

    assert isinstance(caught.value, StopNow)
    assert abort_calls == [True, True]
    assert abandoned == [True]
    assert list(pipeline._inflight) == []
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize("transition", ["collecting", "abandoning"])
def test_pending_transition_interruption_still_retires_authority(
    monkeypatch, transition,
):
    class StopNow(BaseException):
        pass

    class InterruptingPending(PendingRender):
        armed = False

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if self.armed and name == "_state" and value == transition:
                self.armed = False
                raise StopNow(f"{transition} publication interrupted")

    handle = SimpleNamespace(
        lock=threading.RLock(),
        sample_leases={1: 1},
        abandoned_sample_leases={},
        deferred_supervision_error=None,
        cancel_sample=lambda *_args, **_kwargs: None,
        collect_sample=lambda *_args, **_kwargs: pytest.fail(
            "collection must not start after an interrupted state publication"),
    )
    future = mesh_setup.SetupBoundFuture(object(), handle, 1)
    progress_closed = []
    progress = SimpleNamespace(
        __exit__=lambda *_args: progress_closed.append(True))
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = InterruptingPending(
        handle, future, progress, None, {}, 1.0, "render", lambda *_args: {})
    pending.armed = True

    with pytest.raises(StopNow):
        if transition == "collecting":
            pending.result()
        else:
            pending.abandon()

    assert pending._state == "closed"
    assert future.state == "abandoned"
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert progress_closed == [True]


def test_abandon_trace_interruption_retries_every_post_publish_line(monkeypatch):
    import dis
    import inspect
    import sys

    class StopNow(BaseException):
        pass

    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    source, start = inspect.getsourcelines(PendingRender._abandon_once)
    cleanup_start = next(
        start + offset for offset, line in enumerate(source)
        if '_cancel_best_effort("during abandon")' in line
    )
    cleanup_end = next(
        start + offset for offset, line in enumerate(source)
        if "self._close(abandoned=True" in line
    )
    retry_error_start = next(
        start + offset for offset, line in enumerate(source)
        if "except BaseException as helper_exc:" in line
    )
    retry_error_end = next(
        start + offset for offset, line in enumerate(source)
        if "cancel_cause = cause" in line
    )
    retry_error = {
        line for _offset, line in dis.findlinestarts(PendingRender._abandon_once.__code__)
        if retry_error_start <= line <= retry_error_end
    }
    retry_error.add(next(
        start + offset for offset, line in enumerate(source)
        if line.strip() == "cancel_error = None"
    ))
    targets = sorted({
        line for _offset, line in dis.findlinestarts(PendingRender._abandon_once.__code__)
        if cleanup_start <= line <= cleanup_end
    })

    for target in targets:
        handle = SimpleNamespace(
            lock=threading.RLock(),
            sample_leases={1: 1},
            abandoned_sample_leases={},
            deferred_supervision_error=None,
            cancel_sample=lambda *_args, **_kwargs: None,
        )
        future = mesh_setup.SetupBoundFuture(object(), handle, 1)
        progress_closed = []
        pending = PendingRender(
            handle,
            future,
            SimpleNamespace(
                __exit__=lambda *_args, closed=progress_closed: closed.append(True)),
            None,
            {},
            1.0,
            "render",
            lambda *_args: {},
        )
        if target in retry_error:
            actual_cancel = pending._cancel_best_effort
            cancel_calls = 0

            def fail_helper_once(context, actual=actual_cancel):
                nonlocal cancel_calls
                cancel_calls += 1
                if cancel_calls == 1:
                    raise RuntimeError("exercise abandon cancel-retry branch")
                return actual(context)

            monkeypatch.setattr(pending, "_cancel_best_effort", fail_helper_once)
        triggered = False

        def interrupt_abandon(frame, event, _arg, target_line=target):
            nonlocal triggered
            if (event == "line" and frame.f_code is PendingRender._abandon_once.__code__
                    and frame.f_lineno == target_line):
                triggered = True
                raise StopNow("abandon interrupted after ownership publication")
            return interrupt_abandon

        sys.settrace(interrupt_abandon)
        interruption = None
        try:
            pending.abandon()
        except StopNow as exc:
            interruption = exc
        finally:
            sys.settrace(None)
        assert interruption is not None, (
            f"abandon interruption target line {target} was not reached")

        assert triggered is True
        assert pending._state == "closed"
        assert future.state == "abandoned"
        assert handle.sample_leases == {}
        assert handle.abandoned_sample_leases == {1: 1}
        assert progress_closed == [True]


def test_result_post_start_line_interruption_still_retires_authority(monkeypatch):
    import dis
    import inspect
    import sys

    class StopNow(BaseException):
        pass

    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    source, start = inspect.getsourcelines(PendingRender.result)
    collect_start = next(
        start + offset for offset, line in enumerate(source)
        if line.strip() == 'phase = "collect"'
    )
    collect_end = next(
        start + offset for offset, line in enumerate(source)
        if line.strip() == 'phase = "finish"'
    )
    targets = sorted({
        line for _offset, line in dis.findlinestarts(PendingRender.result.__code__)
        if collect_start <= line < collect_end
    })

    for target in targets:
        handle = SimpleNamespace(
            lock=threading.RLock(),
            sample_leases={1: 1},
            abandoned_sample_leases={},
            deferred_supervision_error=None,
            cancel_sample=lambda *_args, **_kwargs: None,
            collect_sample=lambda *_args, **_kwargs: pytest.fail(
                "collection must not start after the line interruption"),
        )
        future = mesh_setup.SetupBoundFuture(object(), handle, 1)
        progress_closed = []
        pending = PendingRender(
            handle,
            future,
            SimpleNamespace(
                __exit__=lambda *_args, closed=progress_closed: closed.append(True)),
            None,
            {},
            1.0,
            "render",
            lambda *_args: {},
        )
        triggered = False

        def interrupt_after_start(frame, event, _arg, target_line=target):
            nonlocal triggered
            if (event == "line" and frame.f_code is PendingRender.result.__code__
                    and frame.f_lineno == target_line):
                triggered = True
                raise StopNow("result interrupted after start returned")
            return interrupt_after_start

        sys.settrace(interrupt_after_start)
        with pytest.raises(StopNow):
            try:
                pending.result()
            finally:
                sys.settrace(None)

        assert triggered is True
        assert pending._state == "closed"
        assert future.state == "abandoned"
        assert handle.sample_leases == {}
        assert progress_closed == [True]


def test_outer_rdma_timeout_holds_owner_until_future_finalization():
    """Session exclusion covers live sample, reader, and ACK authority."""
    handle = _real_handle()
    handle.lock = threading.RLock()
    handle.sample_leases = {1: 1}
    handle.abandoned_sample_leases = {}
    handle.deferred_supervision_error = None
    future = mesh_setup.SetupBoundFuture(object(), handle, 1)
    timed_out_session = RenderSession(timeout_s=0)
    competing_session = RenderSession(timeout_s=0)

    timed_out_session.bind(handle)
    timed_out_session.track(future)
    read_token = future.begin_read()  # A scratch reader entered before the join timed out.
    future.abandon()
    timed_out_session.close()

    assert future.state == "active"
    assert future.retire_to == "abandoned"
    assert future.readers == 1
    with pytest.raises(ConcurrentRenderSessionError):
        competing_session.bind(handle)

    future.end_read(read_token)

    assert future.state == "abandoned"
    assert handle.sample_leases == {}
    competing_session.bind(handle)
    competing_session.close()
    with pytest.raises(mesh_setup.LifecycleBusyError, match="abandoned sample"):
        mesh_setup.require_no_abandoned_samples(handle, "dispatch competing render")


def test_track_retries_finalizer_proof_and_removes_unregistered_hold(
    monkeypatch,
):
    class RegistrationStop(BaseException):
        pass

    class ProofStop(BaseException):
        pass

    handle, token = _ready_real_handle()
    future = mesh_setup.SetupBoundFuture.prepared(handle, token.generation)
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)
    owner.bind(handle)
    add_calls = []
    proof_calls = []

    def fail_add(_self, _callback):
        add_calls.append(True)
        raise RegistrationStop("finalizer was not installed")

    def flaky_proof(_self, _callback):
        proof_calls.append(True)
        if len(proof_calls) == 1:
            raise ProofStop("finalizer proof interrupted")
        return False

    monkeypatch.setattr(mesh_setup.SetupBoundFuture, "add_finalizer", fail_add)
    monkeypatch.setattr(mesh_setup.SetupBoundFuture, "has_finalizer", flaky_proof)

    with pytest.raises(RegistrationStop, match="not installed"):
        owner.track(future)
    owner.close()

    assert add_calls == [True, True]
    assert proof_calls == [True, True]
    contender.bind(handle)
    contender.close()


def test_ambiguous_enqueue_failure_is_owned_and_blocks_seq0_retry():
    class StopNow(BaseException):
        pass

    handle, token = _ready_real_handle()
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)
    owner.bind(handle)
    authority = mesh_setup.prepare_sample(handle, token)
    assert authority is not None
    owner.track(authority)
    enqueued = []

    def enqueue_then_stop():
        enqueued.append("accepted")
        raise StopNow("interrupted after remote acceptance became ambiguous")

    with pytest.raises(StopNow):
        with owner.activate():
            mesh_setup.dispatch_sample(
                handle,
                enqueue_then_stop,
                setup_token=token,
                authority=authority,
            )
    owner.close()

    assert enqueued == ["accepted"]
    assert authority.state == "abandoned"
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    contender.bind(handle)
    contender.close()

    retry_calls = []
    retry_authority = mesh_setup.prepare_sample(handle, token)
    with pytest.raises(mesh_setup.LifecycleBusyError, match="abandoned sample"):
        mesh_setup.dispatch_sample(
            handle,
            lambda: retry_calls.append("seq0"),
            setup_token=token,
            authority=retry_authority,
        )
    assert retry_calls == []


def test_terminal_completion_retries_without_losing_owner_or_supervision(
    monkeypatch,
):
    class StopNow(BaseException):
        pass

    class SupervisionError(RuntimeError):
        pass

    handle, _token = _ready_real_handle()
    handle.sample_leases = {1: 1}
    deferred = SupervisionError("peer closed")
    handle.deferred_supervision_error = deferred
    future = mesh_setup.SetupBoundFuture(object(), handle, 1)
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)
    owner.bind(handle)
    owner.track(future)
    owner.close()
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)

    completions = []
    resumed = []
    original_complete = mesh_setup.SetupBoundFuture._complete_finalization

    def interrupt_once(self, deferred_error, finalizers):
        completions.append(True)
        if len(completions) == 1:
            raise StopNow("terminal commit-to-callback handoff interrupted")
        return original_complete(self, deferred_error, finalizers)

    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture, "_complete_finalization", interrupt_once)
    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture,
        "_resume_deferred_supervision",
        lambda _self, error: resumed.append(error),
    )

    with pytest.raises(StopNow):
        future.release()

    assert completions == [True, True]
    assert resumed == [deferred]
    assert future.state == "consumed"
    assert future._pending_completion is None
    assert handle.sample_leases == {}
    assert handle.deferred_supervision_error is None
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize(
    "path", ["drain_finalization", "retire", "end_read", "read_authority",
             "abandon"])
@pytest.mark.parametrize("reuse_error", [False, True])
def test_lease_retries_prefer_cancellation_without_self_chaining(
    monkeypatch, path, reuse_error,
):
    cancellation = KeyboardInterrupt(f"{path} retry cancelled")
    first = (cancellation if reuse_error else
             RuntimeError(f"{path} first attempt failed"))
    failures = [first, cancellation]
    calls = 0

    def fail(*_args, **_kwargs):
        nonlocal calls
        error = failures[calls]
        calls += 1
        raise error

    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={1: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)
    future = mesh_setup.SetupBoundFuture(object(), handle, 1)
    if path == "abandon":
        monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
        pending = PendingRender(
            handle, future, SimpleNamespace(__exit__=lambda *_args: None),
            None, {}, 1.0, "render", lambda *_args: {},
        )
        monkeypatch.setattr(pending, "_cancel_best_effort", fail)
        invoke = pending.abandon
    elif path == "drain_finalization":
        future._pending_completion = (None, [])
        monkeypatch.setattr(
            mesh_setup.SetupBoundFuture, "_complete_finalization", fail)
        invoke = future._drain_finalization
    elif path == "retire":
        monkeypatch.setattr(
            mesh_setup.SetupBoundFuture, "_retire_and_drain_once", fail)

        def invoke():
            future._retire("consumed")
    elif path == "end_read":
        # Patch this future, not the class: a delayed _ReadToken destructor for
        # another lease would consume the two scripted failures (seen on Python 3.11).
        monkeypatch.setattr(future, "_end_read_token", fail)

        def invoke():
            future.end_read(object())
    else:
        monkeypatch.setattr(
            mesh_setup.SetupBoundFuture, "_begin_read_token",
            lambda *_args: None)
        monkeypatch.setattr(
            mesh_setup.SetupBoundFuture, "_capture_end_read", fail)

        def invoke():
            with future.read_authority():
                pass

    with pytest.raises(KeyboardInterrupt) as exc_info:
        invoke()

    assert exc_info.value is cancellation and calls == 2
    if reuse_error:
        assert cancellation.__cause__ is not cancellation
        assert cancellation.__context__ is not cancellation
    else:
        assert cancellation.__cause__ is first


@pytest.mark.parametrize(
    "ordering", ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"])
def test_lease_finalizers_preserve_cancellation_and_retry_authority(ordering):
    class CleanupCancellation(KeyboardInterrupt):
        def __bool__(self):
            raise AssertionError("exception truthiness must not be evaluated")

    ordinary = RuntimeError("sample finalizer failed")
    cancellation = CleanupCancellation("sample finalizer cleanup cancelled")
    pair = {
        "ordinary-cancellation": [ordinary, cancellation],
        "cancellation-ordinary": [cancellation, ordinary],
        "same-cancellation": [cancellation, cancellation],
    }[ordering]
    failures = pair
    calls: list[str] = []
    recovered = False

    def finalizer() -> None:
        calls.append("finalizer")
        if not recovered:
            error = failures[(calls.count("finalizer") - 1) % len(failures)]
            raise error

    def later_finalizer() -> None:
        calls.append("later")

    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={1: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)
    future = mesh_setup.SetupBoundFuture(object(), handle, 1)
    future.add_finalizer(finalizer)
    future.add_finalizer(later_finalizer)

    with pytest.raises(CleanupCancellation) as exc_info:
        future.release()

    assert exc_info.value is cancellation
    assert calls == ["finalizer", "finalizer", "later"] * 4
    assert future.state == "consumed"
    assert future._pending_completion is not None
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation

    recovered = True
    future.release()
    assert calls[-2:] == ["finalizer", "later"]
    assert future._pending_completion is None


@pytest.mark.parametrize("path", ["finalize", "register"])
@pytest.mark.parametrize(
    "ordering", ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"])
def test_lease_inner_commit_retries_preserve_cancellation(
    monkeypatch, path, ordering,
):
    cancellation = KeyboardInterrupt(f"{path} commit cancelled")
    ordinary = RuntimeError(f"{path} ordinary failure")
    errors = {
        "ordinary-cancellation": [ordinary, cancellation],
        "cancellation-ordinary": [cancellation, ordinary],
        "same-cancellation": [cancellation, cancellation],
    }[ordering]
    calls: list[BaseException] = []

    def fail(*_args):
        error = errors[len(calls)]
        calls.append(error)
        raise error

    handle = SimpleNamespace(
        lock=threading.RLock(), sample_leases={1: 1},
        abandoned_sample_leases={}, deferred_supervision_error=None)
    future = mesh_setup.SetupBoundFuture(object(), handle, 1)
    if path == "finalize":
        monkeypatch.setattr(
            mesh_setup.SetupBoundFuture, "_build_retirement_plan_locked", fail)

        def invoke():
            future._finalize_locked("consumed")
    else:
        future = mesh_setup.SetupBoundFuture.prepared(handle, 1)
        monkeypatch.setattr(
            mesh_setup.SetupBoundFuture, "_apply_registration_locked", fail)
        invoke = future._register_locked

    with handle.lock, pytest.raises(KeyboardInterrupt) as exc_info:
        invoke()

    assert exc_info.value is cancellation and calls == errors
    expected_cause = None if ordering == "same-cancellation" else ordinary
    assert cancellation.__cause__ is expected_cause
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation
