"""Sample lease, read authority and pending render lease regressions."""
from __future__ import annotations

import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from dgx_monarch import mesh as mesh_mod
from dgx_monarch import mesh_lease, mesh_setup
from dgx_monarch.adapters.base import UnsupportedModelError
from dgx_monarch.nodes.render_session import RenderSession
from topology_transition_helpers import (
    _acks,
    _dispatch_sample,
    _Endpoint,
    _Future,
    _handle,
    _ObservedLock,
    _ValueMesh,
    _Workers,
)


def test_setup_state_check_and_enqueue_share_handle_lock():
    workers = _Workers([])
    workers.compute_sigmas = _Endpoint(_Future(_ValueMesh(["rank0", "rank1"])))
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    observed = _ObservedLock()
    handle.lock = observed
    errors = []

    observed.inner.acquire()

    def dispatch():
        try:
            handle.call_all("compute_sigmas", setup_token=token)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=dispatch)
    thread.start()
    assert observed.attempted.wait(timeout=1.0)
    handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
        handle.setup_generation, "topology teardown", 600.0)
    observed.inner.release()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], mesh_setup.TopologyTransitionError)
    assert workers.compute_sigmas.calls == []


def test_sample_lease_blocks_transition_until_driver_releases_result():
    workers = _Workers([_Future(_acks())])
    handle, _old, new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)

    assert handle.sample_leases == {1: 1}
    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        handle.ensure_setup(new, "TORCH_FLASH")
    assert handle.setup_generation == 1
    assert handle.topology != new
    assert workers.teardown_group.calls == []

    mesh_setup.release_sample(submission)
    mesh_setup.release_sample(submission)  # release is idempotent
    assert handle.sample_leases == {}
    handle.ensure_setup(new, "TORCH_FLASH")
    assert handle.topology == new


def test_interrupted_registration_does_not_steal_an_existing_lease(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    session = RenderSession(timeout_s=0)
    session.bind(handle)
    with session.activate():
        first = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    second = mesh_setup.SetupBoundFuture.prepared(handle, token.generation)
    original_setattr = mesh_setup.SetupBoundFuture.__setattr__
    armed = True

    def interrupt_registered_store(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if self is second and name == "_registered" and value and armed:
            armed = False
            raise StopNow("registered publication interrupted")

    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture, "__setattr__", interrupt_registered_store)
    sent = []
    with pytest.raises(StopNow):
        with session.activate():
            _dispatch_sample(
                handle,
                lambda: sent.append(True),
                setup_token=token,
                authority=second,
            )

    assert sent == []
    assert first.state == "active"
    assert second.state == "abandoned"
    assert handle.sample_leases == {token.generation: 1}
    assert handle.abandoned_sample_leases == {}
    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        mesh_setup.require_no_sample_leases(handle, "change setup")
    first.release()
    assert handle.sample_leases == {}
    session.close()


def test_interrupted_terminal_store_resumes_finalizers(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    finalized = []
    submission.add_finalizer(lambda: finalized.append(True))
    original_setattr = mesh_setup.SetupBoundFuture.__setattr__
    armed = True

    def interrupt_terminal_store(self, name, value):
        nonlocal armed
        original_setattr(self, name, value)
        if self is submission and name == "state" and value == "consumed" and armed:
            armed = False
            raise StopNow("terminal state publication interrupted")

    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture, "__setattr__", interrupt_terminal_store)
    with pytest.raises(StopNow):
        submission.release()

    assert finalized == [True]
    assert submission.state == "consumed"
    assert submission.finalizers == []
    assert submission._retirement_plan is None
    assert submission._pending_completion is None
    assert handle.sample_leases == {}
    submission.release()
    assert finalized == [True]


def test_interrupted_retirement_plan_construction_is_retried(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    finalized = []
    submission.add_finalizer(lambda: finalized.append(True))
    original_build = mesh_setup.SetupBoundFuture._build_retirement_plan_locked
    builds = []

    def build_then_stop(self, state):
        plan = original_build(self, state)
        builds.append(plan)
        if len(builds) == 1:
            raise StopNow("plan built before publication was interrupted")
        return plan

    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture,
        "_build_retirement_plan_locked",
        build_then_stop,
    )
    with pytest.raises(StopNow):
        submission.release()

    assert len(builds) == 2
    assert finalized == [True]
    assert submission.state == "consumed"
    assert submission._retirement_plan is None
    assert submission._pending_completion is None
    assert handle.sample_leases == {}


def test_consumed_apply_return_interrupt_preserves_pending_finalizer(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    finalized = []
    submission.add_finalizer(lambda: finalized.append(True))
    original_apply = mesh_setup.SetupBoundFuture._apply_retirement_plan_locked
    applies = []

    def apply_then_stop(self):
        had_plan = self._retirement_plan is not None
        original_apply(self)
        if had_plan:
            applies.append(True)
        if had_plan and len(applies) == 1:
            raise StopNow("consumed plan applied before return was interrupted")

    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture,
        "_apply_retirement_plan_locked",
        apply_then_stop,
    )
    with pytest.raises(StopNow):
        submission.release()

    assert applies == [True]
    assert submission.state == "consumed"
    assert finalized == [True]
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {}
    assert submission._retirement_plan is None
    assert submission._pending_completion is None


def test_abandoned_apply_return_interrupt_counts_and_finalizes_once(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    finalized = []
    submission.add_finalizer(lambda: finalized.append(True))
    original_apply = mesh_setup.SetupBoundFuture._apply_retirement_plan_locked
    applies = []

    def apply_then_stop(self):
        had_plan = self._retirement_plan is not None
        original_apply(self)
        if had_plan:
            applies.append(True)
        if had_plan and len(applies) == 1:
            raise StopNow("abandoned plan applied before return was interrupted")

    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture,
        "_apply_retirement_plan_locked",
        apply_then_stop,
    )
    with pytest.raises(StopNow):
        submission.abandon()

    assert applies == [True]
    assert submission.state == "abandoned"
    assert finalized == [True]
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {token.generation: 1}
    assert submission._retirement_plan is None
    assert submission._pending_completion is None


def test_retire_retries_every_executable_finalization_handoff_line():
    import dis
    import inspect
    import sys

    class StopNow(BaseException):
        pass

    helper = mesh_setup.SetupBoundFuture._retire_and_drain_once
    source, start = inspect.getsourcelines(helper)
    handoff_start = next(
        start + offset for offset, line in enumerate(source)
        if line.strip() == "try:"
    )
    handoff_end = next(
        start + offset for offset, line in enumerate(source)
        if "self._drain_finalization()" in line
    )
    targets = sorted({
        line for _offset, line in dis.findlinestarts(helper.__code__)
        if handoff_start <= line <= handoff_end
    })
    assert handoff_start in targets
    assert handoff_end in targets

    for target in targets:
        handle, _old, _new = _handle(_Workers([]))
        token = mesh_setup.current_setup_token(handle)
        submission = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
        finalized = []
        submission.add_finalizer(
            lambda completed=finalized: completed.append(True))
        triggered = False

        def interrupt_handoff(frame, event, _arg, target_line=target):
            nonlocal triggered
            if (event == "line" and frame.f_code is helper.__code__
                    and frame.f_lineno == target_line):
                triggered = True
                raise StopNow("retirement finalization handoff interrupted")
            return interrupt_handoff

        sys.settrace(interrupt_handoff)
        with pytest.raises(StopNow):
            try:
                submission.release()
            finally:
                sys.settrace(None)

        assert triggered is True
        assert submission.state == "consumed"
        assert submission._pending_completion is None
        assert submission._retirement_plan is None
        assert handle.sample_leases == {}
        assert finalized == [True]


def test_interrupted_reader_retire_intent_is_published(monkeypatch):
    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    read_token = submission.begin_read()
    finalized = []
    submission.add_finalizer(lambda: finalized.append(True))
    original_setattr = mesh_setup.SetupBoundFuture.__setattr__
    armed = True

    def stop_before_retire_intent(self, name, value):
        nonlocal armed
        if self is submission and name == "retire_to" and value == "consumed" and armed:
            armed = False
            raise StopNow("reader retire intent interrupted before publication")
        original_setattr(self, name, value)

    monkeypatch.setattr(
        mesh_setup.SetupBoundFuture, "__setattr__", stop_before_retire_intent)
    with pytest.raises(StopNow):
        submission.release()

    assert submission.state == "active"
    assert submission.retire_to == "consumed"
    assert submission.readers == 1
    assert handle.sample_leases == {token.generation: 1}
    submission.end_read(read_token)
    assert submission.state == "consumed"
    assert finalized == [True]
    assert handle.sample_leases == {}
    del read_token
    assert submission.state == "consumed"
    assert finalized == [True]
    assert handle.sample_leases == {}


@pytest.mark.skipif(not hasattr(sys, "monitoring"),
                    reason="requires Python 3.12")
def test_begin_read_call_handoff_cannot_orphan_a_reader_token():
    import dis

    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([]))
    setup_token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=setup_token)

    def direct_begin():
        read_token = submission.begin_read()
        return read_token

    instructions = list(dis.get_instructions(direct_begin))
    load_index = next(
        index for index, instruction in enumerate(instructions)
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "begin_read"
    )
    call_index = next(
        index for index in range(load_index + 1, len(instructions))
        if instructions[index].opname == "CALL"
    )
    assert instructions[call_index + 1].opname == "STORE_FAST"
    target = instructions[call_index + 1].offset
    primary = StopNow("begin_read return handoff interrupted")
    monitoring = sys.monitoring
    tool_id = next(index for index in range(6)
                   if monitoring.get_tool(index) is None)

    def interrupt(_code, instruction_offset):
        if instruction_offset == target:
            monitoring.set_local_events(tool_id, direct_begin.__code__, 0)
            raise primary

    monitoring.use_tool_id(tool_id, "dgxm-read-token-handoff-test")
    monitoring.register_callback(
        tool_id, monitoring.events.INSTRUCTION, interrupt)
    monitoring.set_local_events(
        tool_id, direct_begin.__code__, monitoring.events.INSTRUCTION)
    with pytest.raises(StopNow) as exc_info:
        try:
            direct_begin()
        finally:
            monitoring.set_local_events(tool_id, direct_begin.__code__, 0)
            monitoring.register_callback(
                tool_id, monitoring.events.INSTRUCTION, None)
            monitoring.free_tool_id(tool_id)

    assert exc_info.value is primary
    assert submission.readers == 0
    submission.abandon()
    assert submission.state == "abandoned"
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {setup_token.generation: 1}


def test_lost_read_token_retries_cleanup_helper_entry(monkeypatch):
    class CleanupStop(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([]))
    setup_token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=setup_token)
    read_token = submission.begin_read()
    original_end = submission.end_read
    calls = 0

    def interrupt_once(token):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise CleanupStop("lost-token cleanup entry interrupted")
        original_end(token)

    monkeypatch.setattr(submission, "end_read", interrupt_once)
    del read_token

    assert calls == 2
    assert submission.readers == 0
    submission.abandon()
    assert submission.state == "abandoned"
    assert handle.sample_leases == {}


def test_read_authority_retries_every_executable_cleanup_line():
    import dis
    import inspect
    import sys

    class StopNow(BaseException):
        pass

    generator = mesh_setup.SetupBoundFuture.read_authority.__wrapped__
    source, start = inspect.getsourcelines(generator)
    cleanup_start = next(
        start + offset for offset, line in enumerate(source)
        if "self._capture_end_read(token)" in line
    )
    cleanup_end = next(
        start + offset for offset, line in reversed(list(enumerate(source)))
        if "self._capture_end_read(token)" in line
    )
    executable = {line for _offset, line in dis.findlinestarts(generator.__code__)}
    targets = [cleanup_start, cleanup_end]
    assert cleanup_start != cleanup_end
    assert all(target in executable for target in targets)

    for target in targets:
        handle, _old, _new = _handle(_Workers([]))
        token = mesh_setup.current_setup_token(handle)
        submission = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
        triggered = False

        def interrupt_cleanup(frame, event, _arg, target_line=target):
            nonlocal triggered
            if (event == "line"
                    and frame.f_code is generator.__code__
                    and frame.f_lineno == target_line):
                triggered = True
                raise StopNow("read authority cleanup interrupted")
            return interrupt_cleanup

        authority = submission.read_authority()
        authority.__enter__()
        submission.abandon()
        sys.settrace(interrupt_cleanup)
        with pytest.raises(StopNow):
            try:
                authority.__exit__(None, None, None)
            finally:
                sys.settrace(None)

        assert triggered is True
        assert submission.readers == 0
        assert submission.state == "abandoned"
        assert handle.sample_leases == {}
        assert handle.abandoned_sample_leases == {token.generation: 1}


def test_read_authority_preserves_body_failure_when_cleanup_also_fails(
    monkeypatch,
):
    class BodyFailure(BaseException):
        pass

    class CleanupFailure(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    primary = BodyFailure("read body failed")
    cleanup_failure = CleanupFailure("cleanup failed after retiring token")
    original_end = submission.end_read
    calls = 0

    def retire_then_fail(read_token):
        nonlocal calls
        calls += 1
        original_end(read_token)
        if calls == 1:
            raise cleanup_failure

    monkeypatch.setattr(submission, "end_read", retire_then_fail)

    with pytest.raises(BodyFailure) as exc_info:
        with submission.read_authority():
            submission.abandon()
            raise primary

    assert exc_info.value is primary
    assert calls == 2
    assert submission.readers == 0
    assert submission.state == "abandoned"
    assert handle.sample_leases == {}
    assert any("cleanup failed after retiring token" in note
               for note in primary.__notes__)


def test_read_authority_preserves_body_failure_at_cleanup_helper_entry():
    import inspect

    class BodyFailure(BaseException):
        pass

    class CleanupStop(BaseException):
        pass

    generator = mesh_setup.SetupBoundFuture.read_authority.__wrapped__
    source, start = inspect.getsourcelines(generator)
    target = next(
        start + offset for offset, line in enumerate(source)
        if "first_cleanup = self._capture_end_read(token)" in line
    )
    handle, _old, _new = _handle(_Workers([]))
    setup_token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=setup_token)
    primary = BodyFailure("read body failed")
    cleanup = CleanupStop("cleanup helper entry interrupted")
    triggered = False

    def interrupt_cleanup(frame, event, _arg):
        nonlocal triggered
        if (not triggered and event == "line"
                and frame.f_code is generator.__code__
                and frame.f_lineno == target):
            triggered = True
            raise cleanup
        return interrupt_cleanup

    sys.settrace(interrupt_cleanup)
    with pytest.raises(BodyFailure) as exc_info:
        try:
            with submission.read_authority():
                submission.abandon()
                raise primary
        finally:
            sys.settrace(None)

    assert triggered is True
    assert exc_info.value is primary
    assert submission.readers == 0
    assert submission.state == "abandoned"
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {setup_token.generation: 1}
    assert any("cleanup helper entry interrupted" in note
               for note in primary.__notes__)


def test_read_authority_never_suppresses_a_falsey_cleanup_failure(monkeypatch):
    class FalseyCleanup(BaseException):
        def __bool__(self):
            return False

    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    primary = FalseyCleanup("cleanup failed after retiring token")
    original_end = submission.end_read
    calls = 0

    def retire_then_fail(read_token):
        nonlocal calls
        calls += 1
        original_end(read_token)
        if calls == 1:
            raise primary

    monkeypatch.setattr(submission, "end_read", retire_then_fail)

    with pytest.raises(FalseyCleanup) as exc_info:
        with submission.read_authority():
            pass

    assert exc_info.value is primary
    assert calls == 2
    assert submission.readers == 0


def test_end_read_preserves_first_failure_when_retry_logging_fails(monkeypatch):
    class RetirementFailure(BaseException):
        pass

    class LogFailure(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    read_token = submission.begin_read()
    failures = [
        RetirementFailure("first retirement failed"),
        RetirementFailure("retry retirement failed"),
    ]
    original_end_token = submission._end_read_token
    calls = 0

    def retire_then_fail(received_token):
        nonlocal calls
        original_end_token(received_token)
        failure = failures[calls]
        calls += 1
        raise failure

    def fail_log(*_args, **_kwargs):
        raise LogFailure("cleanup logger failed")

    monkeypatch.setattr(submission, "_end_read_token", retire_then_fail)
    monkeypatch.setattr(mesh_lease.log, "error", fail_log)

    with pytest.raises(RetirementFailure) as exc_info:
        submission.end_read(read_token)

    assert exc_info.value is failures[0]
    assert calls == 2
    assert submission.readers == 0
    notes = exc_info.value.__notes__
    assert any("retry retirement failed" in note for note in notes)
    assert any("cleanup logger failed" in note for note in notes)


def test_pending_render_holds_lease_through_raw_latent_read(monkeypatch):
    from dgx_monarch import telemetry
    from dgx_monarch.nodes import render_result, render_submit

    workers = _Workers([_Future(_acks())])
    handle, old, new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    read_entered, read_release = threading.Event(), threading.Event()
    results = _ValueMesh([
        {"rank": 0, "host": "rank0", "dp_rank": 0, "sample_s": 1.0,
         "gpu_load_s": 0.0, "transition": "reuse", "latent_stats": {},
         "latent": {"kind": "rdma"}, "latent_extra": {}},
        {"rank": 1, "host": "rank1", "sample_s": 1.0, "gpu_load_s": 0.0,
         "transition": "reuse", "latent_stats": {}, "latent": None},
    ])
    submission = _dispatch_sample(
        handle, lambda: _Future(results), setup_token=token)

    def delayed_read(_descriptor, _guard=None):
        read_entered.set()
        assert read_release.wait(timeout=2.0)
        return torch.ones((1, 4, 1, 1))

    monkeypatch.setattr(render_result, "read_latent_result", delayed_read)
    monkeypatch.setattr(
        render_result, "verify_cross_rank_signatures", lambda _results, _topo: None
    )
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    progress = SimpleNamespace(__exit__=lambda *_args: None)
    pending = render_submit.PendingRender(
        handle, submission, progress, old, {"samples": torch.zeros((1, 4, 1, 1))},
        10.0, "render", render_result._finish_render)
    output = []
    thread = threading.Thread(target=lambda: output.append(pending.result()))
    thread.start()
    assert read_entered.wait(timeout=1.0)

    with pytest.raises(mesh_setup.LifecycleBusyError, match="sample result lease"):
        handle.ensure_setup(new, "TORCH_FLASH")
    read_release.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive() and torch.equal(
        output[0]["samples"], torch.ones((1, 4, 1, 1)))
    assert handle.sample_leases == {}
    with pytest.raises(RuntimeError, match="already closed"):
        pending.result()
    handle.ensure_setup(new, "TORCH_FLASH")


def test_pending_render_timeout_is_terminal_and_requires_recycle(monkeypatch):
    from dgx_monarch import telemetry
    from dgx_monarch.nodes.pending import PendingRender

    handle, _old, new = _handle(_Workers([_Future(_acks())]))
    token = mesh_setup.current_setup_token(handle)

    class TerminalTimeoutFuture:
        def get(self, timeout=None):
            raise TimeoutError("driver wait expired")

    submission = _dispatch_sample(
        handle, TerminalTimeoutFuture, setup_token=token)
    closed = []
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = PendingRender(
        handle, submission, SimpleNamespace(__exit__=lambda *_args: closed.append(1)),
        None, {}, 10.0, "render", lambda results, *_args: results[0])

    with pytest.raises(TimeoutError, match="driver wait expired"):
        pending.result(timeout_s=0.01)
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert closed == [1]
    with pytest.raises(mesh_setup.LifecycleBusyError, match="abandoned sample"):
        handle.ensure_setup(new, "TORCH_FLASH")
    with pytest.raises(RuntimeError, match="already closed"):
        pending.result()
    with pytest.raises(RuntimeError, match="already abandoned"):
        submission.get()


def test_pending_collect_baseexception_terminally_abandons(monkeypatch):
    from dgx_monarch import telemetry
    from dgx_monarch.nodes.pending import PendingRender

    class StopNow(BaseException):
        pass

    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(error=StopNow("stop")), setup_token=token)
    closed = []
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)
    pending = PendingRender(
        handle, submission, SimpleNamespace(__exit__=lambda *_args: closed.append(1)),
        None, {}, 10.0, "render", lambda *_args: {})

    with pytest.raises(StopNow, match="stop"):
        pending.result()

    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert closed == [1]
    with pytest.raises(RuntimeError, match="already closed"):
        pending.result()


@pytest.mark.parametrize(
    "inner",
    [
        UnsupportedModelError("reference_latent is unsupported"),
        RuntimeError("connection lost to worker"),
    ],
)
def test_pending_collect_never_formats_actor_error(monkeypatch, inner):
    from monarch.actor import ActorError

    from dgx_monarch import telemetry
    from dgx_monarch.nodes.pending import PendingRender

    wrapper = ActorError(inner)
    handle, _old, _new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(error=wrapper), setup_token=token)
    closed = []
    formatted = []
    telemetry_finished = []
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda: telemetry_finished.append(1))
    pending = PendingRender(
        handle, submission, SimpleNamespace(__exit__=lambda *_args: closed.append(1)),
        None, {}, 10.0, "render", lambda *_args: {})

    def reject_formatting(_self):
        formatted.append(True)
        raise AssertionError("ActorError wrapper was formatted")

    monkeypatch.setattr(ActorError, "__str__", reject_formatting)

    with pytest.raises(ActorError) as caught:
        pending.result()

    assert caught.value is wrapper
    assert caught.value.exception is inner
    assert formatted == []
    assert handle.defunct is False
    assert handle.procs.reasons == []
    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert closed == [1]
    assert telemetry_finished == [1]
    assert pending._state == "closed"


def test_pending_finish_failure_abandons_unread_result(monkeypatch):
    from dgx_monarch import telemetry
    from dgx_monarch.nodes.pending import PendingRender

    handle, _old, new = _handle(_Workers([]))
    token = mesh_setup.current_setup_token(handle)
    submission = _dispatch_sample(
        handle, lambda: _Future(_ValueMesh([{"latent": "wire"}])),
        setup_token=token)
    closed = []
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda: None)

    def fail_finish(*_args):
        raise RuntimeError("latent validation failed")

    pending = PendingRender(
        handle, submission, SimpleNamespace(__exit__=lambda *_args: closed.append(1)),
        None, {}, 10.0, "render", fail_finish)

    with pytest.raises(RuntimeError, match="latent validation failed"):
        pending.result()

    assert handle.sample_leases == {}
    assert handle.abandoned_sample_leases == {1: 1}
    assert closed == [1]
    with pytest.raises(mesh_setup.LifecycleBusyError, match="abandoned sample"):
        handle.ensure_setup(new, "TORCH_FLASH")


def test_supervision_cleanup_waits_for_every_sample_lease():
    class SupervisionError(RuntimeError):
        pass

    workers = _Workers([_Future(_acks())])
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    session = RenderSession(timeout_s=0)
    session.bind(handle)
    with session.activate():
        first = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
        second = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    error = SupervisionError("peer closed")

    mesh_mod.mark_defunct_on_supervision_failure(handle, error)

    assert handle.deferred_supervision_error is error
    assert handle.sample_leases == {1: 2}
    assert handle.procs.reasons == []
    mesh_setup.abandon_sample(first)
    assert handle.sample_leases == {1: 1}
    assert handle.procs.reasons == []

    mesh_setup.abandon_sample(second)
    session.close()

    assert handle.sample_leases == {}
    assert handle.deferred_supervision_error is None
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert handle.procs.reasons == ["dgx-monarch client detach"]


def _evicted_behind_one_lease(exc):
    """Publish one eviction while a sample lease is still live."""
    from dgx_monarch import mesh_helpers

    workers = _Workers([_Future(_acks())])
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    session = RenderSession(timeout_s=0)
    session.bind(handle)
    with session.activate():
        lease = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    mesh_helpers.mark_defunct_deliberate(handle, exc)
    return handle, lease, session


def test_a_deferred_deliberate_eviction_resumes_through_its_own_route(monkeypatch):
    """The stall guard's eviction must reach a terminal teardown outcome.

    Resumed through the supervision classifier, it would leave the handle
    defunct with no outcome, and every later attach and recycle would refuse
    until ComfyUI restarts. resume_deferred_supervision in
    mesh_lease_retirement.py says why (CHANGELOG.md, 2026-08-25).
    """
    from dgx_monarch import mesh_helpers, mesh_safety

    stall = mesh_lease.SampleStallError("render stalled: the fleet reported no progress")
    handle, lease, session = _evicted_behind_one_lease(stall)

    assert handle.defunct is True
    assert handle.deferred_supervision_error is stall
    assert handle.teardown_complete is False
    assert handle.procs.reasons == []
    # Read with a default and asserted after the outcome, so a missing route
    # flag fails on the teardown outcome rather than on an AttributeError.
    route_at_defer = getattr(handle, "_deferred_eviction_deliberate", False)

    classifier: list = []
    monkeypatch.setattr(
        mesh_helpers, "mark_defunct_on_supervision_failure",
        lambda *args, **kwargs: classifier.append(args))

    mesh_setup.abandon_sample(lease)

    assert classifier == []  # the classifier is not this eviction's route
    assert handle.teardown_complete is True
    assert mesh_safety.coherent_lifecycle_verdict(handle, RuntimeError) == "completed"
    assert handle.replacement_blocked is None
    assert handle.deferred_supervision_error is None
    assert route_at_defer is True
    assert getattr(handle, "_deferred_eviction_deliberate", False) is False
    session.close()


def test_a_deferred_supervision_failure_still_resumes_through_the_classifier():
    """A supervision failure carries no route flag and resumes through the classifier."""
    class SupervisionError(RuntimeError):
        pass

    workers = _Workers([_Future(_acks())])
    handle, _old, _new = _handle(workers)
    token = mesh_setup.current_setup_token(handle)
    session = RenderSession(timeout_s=0)
    session.bind(handle)
    with session.activate():
        lease = _dispatch_sample(
            handle, lambda: _Future(_ValueMesh([])), setup_token=token)
    error = SupervisionError("peer closed")

    mesh_mod.mark_defunct_on_supervision_failure(handle, error)

    assert getattr(handle, "_deferred_eviction_deliberate", False) is False
    mesh_setup.abandon_sample(lease)
    assert handle.teardown_complete is True
    session.close()


def test_the_deliberate_route_is_cleared_with_the_deferred_error():
    """A stale route flag must not steer a later, unrelated failure."""
    from dgx_monarch import mesh_helpers

    stall = mesh_lease.SampleStallError("render stalled")
    handle, lease, session = _evicted_behind_one_lease(stall)
    mesh_setup.abandon_sample(lease)
    session.close()

    assert getattr(handle, "_deferred_eviction_deliberate", False) is False

    # An ordinary endpoint error on the same handle is a no-op: the classifier
    # decides it, not a leftover route flag.
    mesh_helpers.mark_defunct_on_supervision_failure(
        handle, ValueError("bad sampler input"))
    assert handle.procs.reasons == ["dgx-monarch client detach"]
