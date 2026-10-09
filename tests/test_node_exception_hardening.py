"""Cross-node invariants for cleanup cancellation and hostile diagnostics."""
from __future__ import annotations

import threading
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_rpc, mesh_setup, telemetry
from dgx_monarch.nodes import (
    auto_gate,
    common,
    consent_waiver,
    fleet,
    gate,
    gate_cross_mode,
    gate_fsdp,
    gate_session,
    pending,
    pipeline,
    submit_guard,
)


class _Cancelled(KeyboardInterrupt):
    def __bool__(self):
        raise AssertionError("exception truthiness must not be evaluated")


class _Exited(SystemExit):
    def __bool__(self):
        raise AssertionError("exception truthiness must not be evaluated")


_TERMINATIONS = (_Cancelled, _Exited)


def _ordered_failures(ordering: str) -> tuple[list[BaseException], _Cancelled]:
    ordinary = RuntimeError("ordinary cleanup failure")
    cancellation = _Cancelled("cleanup cancelled")
    failures = {
        "ordinary-cancellation": [ordinary, cancellation],
        "cancellation-ordinary": [cancellation, ordinary],
        "same-cancellation": [cancellation, cancellation],
    }[ordering]
    return failures, cancellation


@pytest.mark.parametrize("operation", ["fleet-future", "fleet-progress"])
@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_fleet_cleanup_retries_preserve_exact_cancellation(
    monkeypatch, operation, ordering,
):
    failures, cancellation = _ordered_failures(ordering)
    calls = []

    def fail(*_args):
        error = failures[len(calls)]
        calls.append(error)
        raise error

    if operation == "fleet-future":
        monkeypatch.setattr(mesh_setup, "abandon_sample", fail)
        result = fleet._retire_fleet_future(object(), consumed=False)
    else:
        result = fleet._close_fleet_progress(
            SimpleNamespace(__exit__=fail)
        )

    assert result is cancellation
    assert calls == failures
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("operation", ["pipeline-abandon", "pipeline-session"])
@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_pipeline_cleanup_retries_preserve_exact_cancellation(
    operation, ordering,
):
    failures, cancellation = _ordered_failures(ordering)
    calls = []

    def fail(*_args):
        error = failures[len(calls)]
        calls.append(error)
        raise error

    with pytest.raises(_Cancelled) as caught:
        if operation == "pipeline-abandon":
            pipeline._abandon_all_inflight(
                deque([SimpleNamespace(abandon=fail)])
            )
        else:
            pipeline._close_pipeline_session_retry(
                SimpleNamespace(close=fail)
            )

    assert caught.value is cancellation
    assert calls == failures
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


def test_pipeline_cancel_and_drain_surfaces_cancellation_after_full_pass():
    cancellation = _Cancelled("pipeline drain cancelled")
    calls = []
    item = SimpleNamespace(
        _state="open",
        cancel=lambda: calls.append("cancel"),
        result=lambda: (_ for _ in ()).throw(cancellation),
    )

    with pytest.raises(_Cancelled) as caught:
        pipeline._cancel_and_drain_inflight(deque([item]))

    assert caught.value is cancellation
    assert calls == ["cancel"]


@pytest.mark.parametrize(
    "stage", ["lease", "audit", "progress", "telemetry", "session"]
)
@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_pending_cleanup_stages_preserve_exact_cancellation(
    monkeypatch, stage, ordering,
):
    failures, cancellation = _ordered_failures(ordering)
    calls = []

    def fail(*_args):
        error = failures[len(calls)]
        calls.append(error)
        raise error

    monkeypatch.setattr(mesh_setup, "abandon_sample", lambda _future: None)
    monkeypatch.setattr(consent_waiver, "retire_audit", lambda _render_id: None)
    monkeypatch.setattr(telemetry.render_progress, "finish", lambda *_args: None)
    progress = SimpleNamespace(__exit__=lambda *_args: None)
    session_close = None
    if stage == "lease":
        monkeypatch.setattr(mesh_setup, "abandon_sample", fail)
    elif stage == "audit":
        monkeypatch.setattr(consent_waiver, "retire_audit", fail)
    elif stage == "progress":
        progress = SimpleNamespace(__exit__=fail)
    elif stage == "telemetry":
        monkeypatch.setattr(telemetry.render_progress, "finish", fail)
    else:
        session_close = fail
    render = pending.PendingRender(
        SimpleNamespace(), object(), progress, None, {}, 1.0, "render",
        lambda *_args: {}, session_close=session_close,
    )

    with pytest.raises(_Cancelled) as caught:
        render._close(abandoned=True)

    assert caught.value is cancellation
    assert calls == failures
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_submit_cleanup_retry_preserves_exact_cancellation(
    monkeypatch, ordering,
):
    failures, cancellation = _ordered_failures(ordering)
    calls = []

    def fail(_future):
        error = failures[len(calls)]
        calls.append(error)
        raise error

    guard = submit_guard.SubmitRenderGuard()
    guard.future = object()
    monkeypatch.setattr(mesh_setup, "abandon_sample", fail)

    with pytest.raises(_Cancelled) as caught:
        submit_guard.run_guarded_submit(
            guard, lambda: "submitted", lambda _exc: None
        )

    assert caught.value is cancellation
    assert calls == failures
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_submit_cleanup_cancellation_outranks_ordinary_body_primary(
    monkeypatch, termination_type,
):
    primary = RuntimeError("submit body failed")
    cancellation = termination_type("cleanup cancelled")
    guard = submit_guard.SubmitRenderGuard()
    guard.future = object()
    monkeypatch.setattr(
        mesh_setup, "abandon_sample",
        lambda _future: (_ for _ in ()).throw(cancellation),
    )

    with pytest.raises(termination_type) as caught:
        submit_guard.run_guarded_submit(
            guard,
            lambda: (_ for _ in ()).throw(primary),
            lambda _exc: None,
        )

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_common_cleanup_cancellation_wins_after_every_owner_attempt(
    monkeypatch, termination_type,
):
    primary = RuntimeError("render result failed")
    cancellation = termination_type("handoff cleanup cancelled")
    calls: list[str] = []

    class Handoff:
        def abort(self):
            calls.append("abort")
            raise cancellation

        def clear(self, _pending):
            calls.append("clear")

    class Pending:
        def result(self):
            raise primary

        def abandon(self):
            calls.append("abandon")

    monkeypatch.setattr(common.consent_waiver, "validate_inherited_stamps", lambda *_: [])
    monkeypatch.setattr(
        common.render_preflight, "activation_footprint_preflight_for_request",
        lambda *_: None,
    )
    monkeypatch.setattr(common, "_claim_adoption_context", lambda: object())
    monkeypatch.setattr(common, "_bind_packed_render_model", lambda model, *_: (model, None))
    monkeypatch.setattr(common, "_suspend_adoption_context", nullcontext)
    monkeypatch.setattr(common, "_maybe_auto_gate", lambda *_: None)
    monkeypatch.setattr(common, "model_for_request", lambda model, _request: model)
    monkeypatch.setattr(common, "PendingRenderHandoff", Handoff)
    monkeypatch.setattr(common, "submit_render", lambda *_args, **_kwargs: Pending())

    with pytest.raises(termination_type) as caught:
        common.run_render(SimpleNamespace(unet_name=None), {}, {}, None, 1)

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert calls == ["abort", "abort", "abandon", "clear"]


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_pending_cleanup_cancellation_wins_after_earlier_stages_retire(
    monkeypatch, termination_type,
):
    primary = RuntimeError("render collection failed")
    cancellation = termination_type("session release cancelled")
    calls: list[str] = []

    monkeypatch.setattr(mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    render = pending.PendingRender(
        SimpleNamespace(), object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: (calls.append("session"), (_ for _ in ()).throw(cancellation)),
    )

    with pytest.raises(termination_type) as caught:
        render._close(abandoned=True, primary=primary)

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert calls == ["lease", "audit", "progress", "telemetry", "session", "session"]


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_pending_timeout_cancel_cancellation_wins_after_full_close(
    monkeypatch, termination_type,
):
    primary = TimeoutError("render collection timed out")
    cancellation = termination_type("timeout cancellation interrupted")
    calls: list[str] = []

    def cancel(*_args, **_kwargs):
        calls.append("cancel")
        raise cancellation

    handle = SimpleNamespace(
        collect_sample=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        cancel_sample=cancel,
    )
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: calls.append("session"),
    )

    with pytest.raises(termination_type) as caught:
        render.result()

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation
    assert calls == [
        "cancel", "cancel", "lease", "audit", "progress", "telemetry", "session"]
    assert render._state == "closed"


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_pending_timeout_cancel_helper_boundary_retries_full_cleanup(
    monkeypatch, termination_type,
):
    primary = TimeoutError("render collection timed out")
    cancellation = termination_type("timeout cancel helper interrupted")
    calls: list[str] = []
    handle = SimpleNamespace(
        collect_sample=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        cancel_sample=lambda *_args, **_kwargs: calls.append("cancel"),
    )
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: calls.append("session"),
    )
    actual_cancel = render._cancel_best_effort
    helper_calls = 0

    def interrupt_first_cancel(context):
        nonlocal helper_calls
        helper_calls += 1
        calls.append("cancel-helper")
        if helper_calls == 1:
            raise cancellation
        return actual_cancel(context)

    monkeypatch.setattr(render, "_cancel_best_effort", interrupt_first_cancel)

    with pytest.raises(termination_type) as caught:
        render.result()

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert calls == [
        "cancel-helper", "cancel-helper", "cancel", "lease", "audit",
        "progress", "telemetry", "session",
    ]
    assert render._state == "closed"
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_pending_result_close_helper_boundary_retries_full_cleanup(
    monkeypatch, termination_type,
):
    primary = RuntimeError("render collection failed")
    cancellation = termination_type("close helper interrupted")
    calls: list[str] = []
    handle = SimpleNamespace(
        collect_sample=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        cancel_sample=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    monkeypatch.setattr(
        pending, "evict_render_fleet",
        lambda _handle, error: calls.append("supervision") if error is primary else None,
    )
    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: calls.append("session"),
    )
    actual_close = render._close
    close_calls = 0

    def interrupt_first_close(**kwargs):
        nonlocal close_calls
        close_calls += 1
        calls.append("close-helper")
        if close_calls == 1:
            raise cancellation
        actual_close(**kwargs)

    monkeypatch.setattr(render, "_close", interrupt_first_close)

    with pytest.raises(termination_type) as caught:
        render.result()

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert calls == [
        "close-helper", "close-helper", "lease", "audit", "progress",
        "telemetry", "session", "supervision",
    ]
    assert render._state == "closed"
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pending_abandon_cancel_boundary_keeps_first_over_session_cleanup(
    monkeypatch, first_type, later_type,
):
    first = first_type("abandon cancel helper interrupted first")
    later = later_type("session cleanup interrupted later")
    calls: list[str] = []
    handle = SimpleNamespace(
        cancel_sample=lambda *_args, **_kwargs: calls.append("cancel"))
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    session_calls = 0

    def close_session():
        nonlocal session_calls
        session_calls += 1
        calls.append("session")
        if session_calls <= 2:
            raise later

    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=close_session,
    )
    actual_cancel = render._cancel_best_effort
    helper_calls = 0

    def interrupt_first_cancel(context):
        nonlocal helper_calls
        helper_calls += 1
        calls.append("cancel-helper")
        if helper_calls == 1:
            raise first
        return actual_cancel(context)

    monkeypatch.setattr(render, "_cancel_best_effort", interrupt_first_cancel)

    with pytest.raises(first_type) as caught:
        render.abandon()

    assert caught.value is first
    assert helper_calls == 3
    assert calls.count("cancel") == 2
    assert calls.count("lease") == calls.count("audit") == 1
    assert calls.count("progress") == calls.count("telemetry") == 1
    assert calls.count("session") == 3
    assert render._state == "closed"
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pending_abandon_keeps_first_over_close_helper_boundary(
    monkeypatch, first_type, later_type,
):
    first = first_type("abandon cancel helper interrupted first")
    later = later_type("close helper interrupted later")
    calls: list[str] = []
    handle = SimpleNamespace(
        cancel_sample=lambda *_args, **_kwargs: calls.append("cancel"))
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: calls.append("session"),
    )
    actual_cancel = render._cancel_best_effort
    cancel_calls = 0

    def interrupt_first_cancel(context):
        nonlocal cancel_calls
        cancel_calls += 1
        calls.append("cancel-helper")
        if cancel_calls == 1:
            raise first
        return actual_cancel(context)

    actual_close = render._close
    close_calls = 0

    def interrupt_first_close(**kwargs):
        nonlocal close_calls
        close_calls += 1
        calls.append("close-helper")
        if close_calls == 1:
            raise later
        actual_close(**kwargs)

    monkeypatch.setattr(render, "_cancel_best_effort", interrupt_first_cancel)
    monkeypatch.setattr(render, "_close", interrupt_first_close)

    with pytest.raises(first_type) as caught:
        render.abandon()

    assert caught.value is first
    assert cancel_calls == close_calls == 2
    assert calls == [
        "cancel-helper", "cancel-helper", "cancel", "close-helper", "close-helper",
        "lease", "audit", "progress", "telemetry", "session",
    ]
    assert render._state == "closed"
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pending_post_cleanup_interrupt_keeps_first_cleanup_cancellation(
    monkeypatch, first_type, later_type,
):
    primary = RuntimeError("render collection failed")
    first = first_type("session cleanup cancelled first")
    later = later_type("Comfy interruption check cancelled later")
    calls: list[str] = []
    handle = SimpleNamespace(
        collect_sample=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        cancel_sample=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    monkeypatch.setattr(
        pending, "evict_render_fleet",
        lambda _handle, error: calls.append("supervision") if error is primary else None,
    )
    monkeypatch.setattr(
        pending, "_throw_if_comfy_interrupted",
        lambda: (calls.append("interrupt-check"), (_ for _ in ()).throw(later)),
    )
    session_calls = 0

    def close_session():
        nonlocal session_calls
        session_calls += 1
        calls.append("session")
        if session_calls <= 2:
            raise first

    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=close_session,
    )

    with pytest.raises(first_type) as caught:
        render.result()

    assert caught.value is first
    assert first.__cause__ is primary
    assert calls == [
        "lease", "audit", "progress", "telemetry", "session", "session",
        "session", "supervision", "interrupt-check",
    ]
    assert render._state == "closed"
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pending_supervision_boundary_keeps_first_and_still_checks_interrupt(
    monkeypatch, first_type, later_type,
):
    primary = RuntimeError("render collection failed")
    first = first_type("session cleanup cancelled first")
    later = later_type("supervision helper cancelled later")
    calls: list[str] = []
    handle = SimpleNamespace(
        collect_sample=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        cancel_sample=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))

    supervision_calls = 0

    def fail_supervision(_handle, error):
        nonlocal supervision_calls
        assert error is primary
        supervision_calls += 1
        calls.append("supervision-helper")
        if supervision_calls == 1:
            raise later
        calls.append("supervision-publication")

    monkeypatch.setattr(pending, "evict_render_fleet", fail_supervision)
    monkeypatch.setattr(
        pending, "_throw_if_comfy_interrupted",
        lambda: calls.append("interrupt-check"),
    )
    session_calls = 0

    def close_session():
        nonlocal session_calls
        session_calls += 1
        calls.append("session")
        if session_calls <= 2:
            raise first

    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=close_session,
    )

    with pytest.raises(first_type) as caught:
        render.result()

    assert caught.value is first
    assert first.__cause__ is primary
    assert calls == [
        "lease", "audit", "progress", "telemetry", "session", "session",
        "session", "supervision-helper", "supervision-helper",
        "supervision-publication", "interrupt-check",
    ]
    assert supervision_calls == 2
    assert render._state == "closed"
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pending_result_state_snapshot_retries_and_keeps_first_cancellation(
    monkeypatch, first_type, later_type,
):
    primary = TimeoutError("render collection timed out")
    first = first_type("timeout cancellation interrupted first")
    later = later_type("ownership state snapshot interrupted later")
    calls: list[str] = []

    class InterruptingLock:
        enters = 0

        def __enter__(self):
            self.enters += 1
            calls.append(f"state-{self.enters}")
            if self.enters == 2:
                raise later
            return self

        def __exit__(self, *_args):
            return False

    cancel_calls = 0

    def cancel(*_args, **_kwargs):
        nonlocal cancel_calls
        cancel_calls += 1
        calls.append(f"cancel-{cancel_calls}")
        if cancel_calls == 1:
            raise first

    handle = SimpleNamespace(
        collect_sample=lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        cancel_sample=cancel,
    )
    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    render = pending.PendingRender(
        handle, object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: calls.append("session"),
    )
    render._state_lock = InterruptingLock()

    with pytest.raises(first_type) as caught:
        render.result()

    assert caught.value is first
    assert first.__cause__ is primary
    assert calls == [
        "state-1", "cancel-1", "cancel-2", "state-2", "state-3", "state-4",
        "lease", "audit", "progress", "telemetry", "session", "state-5",
    ]
    assert render._state == "closed"
    assert all((render._lease_retired, render._audit_retired,
                render._progress_closed, render._telemetry_finished,
                render._session_closed))
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pending_close_terminal_snapshot_keeps_first_stage_cancellation(
    monkeypatch, first_type, later_type,
):
    first = first_type("lease retirement interrupted first")
    later = later_type("terminal state publication interrupted later")
    calls: list[str] = []

    class InterruptingLock:
        enters = 0

        def __enter__(self):
            self.enters += 1
            calls.append(f"state-{self.enters}")
            if self.enters == 2:
                raise later
            return self

        def __exit__(self, *_args):
            return False

    lease_calls = 0

    def abandon_sample(_future):
        nonlocal lease_calls
        lease_calls += 1
        calls.append(f"lease-{lease_calls}")
        if lease_calls == 1:
            raise first

    monkeypatch.setattr(mesh_setup, "abandon_sample", abandon_sample)
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    render = pending.PendingRender(
        SimpleNamespace(), object(),
        SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {},
        session_close=lambda: calls.append("session"),
    )
    render._state_lock = InterruptingLock()

    with pytest.raises(first_type) as caught:
        render._close(abandoned=True)

    assert caught.value is first
    assert calls == [
        "state-1", "lease-1", "lease-2", "audit", "progress", "telemetry",
        "session", "state-2", "state-3",
    ]
    assert render._state == "closed"
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pending_handoff_clear_retries_and_keeps_first_cancellation(
    first_type, later_type,
):
    first = first_type("handoff abandon interrupted first")
    later = later_type("handoff clear interrupted later")
    calls: list[str] = []

    class Render:
        _state = "closed"
        abandons = 0

        def abandon(self):
            self.abandons += 1
            calls.append(f"abandon-{self.abandons}")
            if self.abandons == 1:
                raise first

    render = Render()
    handoff = pending.PendingRenderHandoff()
    handoff._pending = render
    clear = handoff.clear
    clear_calls = 0

    def interrupt_clear(candidate):
        nonlocal clear_calls
        clear_calls += 1
        calls.append(f"clear-{clear_calls}")
        if clear_calls == 1:
            raise later
        clear(candidate)

    handoff.clear = interrupt_clear

    with pytest.raises(first_type) as caught:
        handoff.abort()

    assert caught.value is first
    assert calls == ["abandon-1", "abandon-2", "clear-1", "clear-2"]
    assert handoff._pending is None
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    (
        (RuntimeError, _Cancelled),
        (RuntimeError, _Exited),
        (_Cancelled, _Exited),
        (_Exited, _Cancelled),
    ),
)
def test_pending_abandon_claim_error_reconciles_later_cleanup_cancellation(
    monkeypatch, first_type, later_type,
):
    first = first_type("abandon claim boundary failed first")
    later = later_type("session cleanup cancelled later")
    calls: list[str] = []

    class InterruptingLock:
        exits = 0

        def __enter__(self):
            calls.append("state-enter")
            return self

        def __exit__(self, *_args):
            self.exits += 1
            calls.append(f"state-exit-{self.exits}")
            if self.exits == 1:
                raise first
            return False

    session_calls = 0

    def close_session():
        nonlocal session_calls
        session_calls += 1
        calls.append(f"session-{session_calls}")
        if session_calls == 1:
            raise later

    monkeypatch.setattr(
        mesh_setup, "abandon_sample", lambda _future: calls.append("lease"))
    monkeypatch.setattr(
        consent_waiver, "retire_audit", lambda _rid: calls.append("audit"))
    monkeypatch.setattr(
        telemetry.render_progress, "finish", lambda *_args: calls.append("telemetry"))
    render = pending.PendingRender(
        SimpleNamespace(cancel_sample=lambda *_args, **_kwargs: calls.append("cancel")),
        object(), SimpleNamespace(__exit__=lambda *_args: calls.append("progress")),
        None, {}, 1.0, "render", lambda *_args: {}, session_close=close_session,
    )
    render._state_lock = InterruptingLock()
    expected = later if first_type is RuntimeError else first

    with pytest.raises(type(expected)) as caught:
        render.abandon()

    assert caught.value is expected
    assert calls.count("cancel") == 1
    assert calls.count("session-1") == calls.count("session-2") == 1
    assert render._state == "closed"
    if first_type is RuntimeError:
        assert later.__cause__ is first
    assert expected.__cause__ is not expected
    assert expected.__context__ is not expected


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_fleet_later_cancellation_wins_after_full_wave_drain(
    monkeypatch, termination_type,
):
    ordinary = RuntimeError("job zero failed")
    cancellation = termination_type("job one cancelled")
    failures = iter((ordinary, cancellation))
    collected: list[object] = []

    class Handle:
        def collect_one(self, future, *, timeout_s):
            del timeout_s
            collected.append(future)
            if len(collected) <= 2:
                raise next(failures)
            return {"latent": "ok", "host": "test", "sample_s": 0.0}

    monkeypatch.setattr(fleet, "_retire_fleet_audit", lambda _rid: None)
    monkeypatch.setattr(
        fleet, "_retire_fleet_future", lambda _future, *, consumed: None)
    monkeypatch.setattr(fleet, "_close_fleet_progress", lambda _progress: None)
    pending_jobs = [
        fleet._FleetPending(index, object(), object(), f"r{index}")
        for index in range(3)
    ]

    with pytest.raises(termination_type) as caught:
        fleet._collect_fleet_wave(
            Handle(), pending_jobs, [None] * 3, 1.0, 3)

    assert caught.value is cancellation
    assert len(collected) == 3
    assert pending_jobs == []
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_fleet_timeout_cancel_cancellation_wins_after_full_wave(
    monkeypatch, termination_type,
):
    primary = TimeoutError("fleet job timed out")
    cancellation = termination_type("fleet timeout cancellation interrupted")
    calls: list[tuple[str, object]] = []

    class Handle:
        def collect_one(self, future, *, timeout_s):
            del timeout_s
            calls.append(("collect", future))
            if future == "slow":
                raise primary
            return {"latent": future, "host": "test", "sample_s": 0.0}

        def cancel_sample(self, render_id, *, wait):
            calls.append(("cancel", (render_id, wait)))
            raise cancellation

    monkeypatch.setattr(fleet, "read_latent_result", lambda value: value)
    monkeypatch.setattr(fleet, "_retire_fleet_audit", lambda _rid: None)
    monkeypatch.setattr(
        fleet, "_retire_fleet_future", lambda _future, *, consumed: None)
    monkeypatch.setattr(fleet, "_close_fleet_progress", lambda _progress: None)
    pending_jobs = [
        fleet._FleetPending(0, "slow", object(), "render-0"),
        fleet._FleetPending(1, "ok", object(), "render-1"),
    ]

    with pytest.raises(termination_type) as caught:
        fleet._collect_fleet_wave(
            Handle(), pending_jobs, [None, None], 1.0, 2)

    assert caught.value is cancellation
    assert sorted(
        value for operation, value in calls if operation == "collect"
    ) == ["ok", "slow"]
    cancellations = [value for operation, value in calls if operation == "cancel"]
    assert cancellations[0] == ("render-0", False)
    assert set(cancellations) <= {
        ("render-0", False), ("render-1", False)}
    assert pending_jobs == []
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_fleet_first_cross_domain_cancellation_wins_after_drain(
    monkeypatch, first_type, later_type,
):
    first = first_type("first audit cancellation")
    later = later_type("later collection cancellation")
    collected: list[str] = []

    class Handle:
        def collect_one(self, future, *, timeout_s):
            del timeout_s
            collected.append(future)
            if future == "later":
                raise later
            return {"latent": future, "host": "test", "sample_s": 0.0}

    monkeypatch.setattr(fleet, "read_latent_result", lambda value: value)
    monkeypatch.setattr(
        fleet, "_retire_fleet_audit",
        lambda render_id: first if render_id == "render-0" else None,
    )
    monkeypatch.setattr(
        fleet, "_retire_fleet_future", lambda _future, *, consumed: None)
    monkeypatch.setattr(fleet, "_close_fleet_progress", lambda _progress: None)
    pending_jobs = [
        fleet._FleetPending(0, "ok", object(), "render-0"),
        fleet._FleetPending(1, "later", object(), "render-1"),
    ]

    with pytest.raises(first_type) as caught:
        fleet._collect_fleet_wave(
            Handle(), pending_jobs, [None, None], 1.0, 2)

    assert caught.value is first
    assert set(collected) == {"ok", "later"}
    assert len(pending_jobs) == 1
    assert pending_jobs[0].future == "ok"
    assert pending_jobs[0].retired and pending_jobs[0].progress_closed
    assert not pending_jobs[0].audit_retired
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_fleet_wave_helper_boundary_preserves_first_and_drains(
    monkeypatch, first_type, later_type,
):
    first = first_type("collection cancelled first")
    later = later_type("audit helper interrupted later")
    collected: list[str] = []
    audit_calls: list[str] = []

    class Handle:
        def collect_one(self, future, *, timeout_s):
            del timeout_s
            collected.append(future)
            if future == "first":
                raise first
            return {"latent": future, "host": "test", "sample_s": 0.0}

    def retire_audit(render_id):
        audit_calls.append(render_id)
        if len(audit_calls) == 1:
            raise later
        return None

    monkeypatch.setattr(fleet, "read_latent_result", lambda value: value)
    monkeypatch.setattr(fleet, "_retire_fleet_audit", retire_audit)
    monkeypatch.setattr(
        fleet, "_retire_fleet_future", lambda _future, *, consumed: None)
    monkeypatch.setattr(fleet, "_close_fleet_progress", lambda _progress: None)
    pending_jobs = [
        fleet._FleetPending(0, "first", object(), "render-0"),
        fleet._FleetPending(1, "second", object(), "render-1"),
    ]

    with pytest.raises(first_type) as caught:
        fleet._collect_fleet_wave(
            Handle(), pending_jobs, [None, None], 1.0, 2)

    assert caught.value is first
    assert set(collected) == {"first", "second"}
    assert audit_calls == ["render-0", "render-0", "render-1"]
    assert pending_jobs == []
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_fleet_wave_flag_publication_boundary_retries_and_drains(
    monkeypatch, termination_type,
):
    cancellation = termination_type("audit completion publication interrupted")
    events: list[str] = []

    class InterruptingPending(fleet._FleetPending):
        armed = False

        def __setattr__(self, name, value):
            super().__setattr__(name, value)
            if self.armed and name == "audit_retired" and value is True:
                self.armed = False
                raise cancellation

    item = InterruptingPending(0, "future", object(), "render")
    item.armed = True
    monkeypatch.setattr(
        fleet, "_retire_fleet_audit", lambda _rid: events.append("audit"))
    monkeypatch.setattr(
        fleet, "_retire_fleet_future",
        lambda _future, *, consumed: events.append("lease"),
    )
    monkeypatch.setattr(
        fleet, "_close_fleet_progress", lambda _progress: events.append("progress"))
    monkeypatch.setattr(fleet, "read_latent_result", lambda value: value)
    pending_jobs = [item]

    with pytest.raises(termination_type) as caught:
        fleet._collect_fleet_wave(
            SimpleNamespace(collect_one=lambda *_args, **_kwargs: {
                "latent": "ok", "host": "test", "sample_s": 0.0}),
            pending_jobs, [None], 1.0, 1,
        )

    assert caught.value is cancellation
    assert events == ["audit", "audit", "lease", "progress"]
    assert pending_jobs == []
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_fleet_submission_helper_boundary_runs_every_later_stage(
    monkeypatch, termination_type,
):
    primary = RuntimeError("fleet submission failed")
    cancellation = termination_type("lease helper interrupted")
    future = object()
    events: list[object] = []

    class Progress:
        port = 1234

        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            events.append("progress-enter")

    class Handle:
        world = 1

        def submit_sample_to(self, *_args, **_kwargs):
            raise primary

    def retire_future(received, *, consumed):
        events.append(("lease", received, consumed))
        if sum(event[0] == "lease" for event in events if isinstance(event, tuple)) == 1:
            raise cancellation
        return None

    def retire_audit(render_id):
        events.append(("audit", render_id))
        return None

    monkeypatch.setattr(
        fleet, "_fleet_worker_policy",
        lambda *_args: fleet._FleetAuthorization({}, {}, None),
    )
    monkeypatch.setattr(mesh_setup, "ensure_request_setup", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(mesh_setup, "prepare_sample", lambda *_args, **_kwargs: future)
    monkeypatch.setattr(mesh_setup, "token_kwargs", lambda _token: {})
    monkeypatch.setattr(fleet, "pack_latent", lambda value: value)
    monkeypatch.setattr(fleet, "conditioning_for_wire", lambda value: value)
    monkeypatch.setattr(fleet, "ProgressReceiver", Progress)
    monkeypatch.setattr(fleet.consent_waiver, "stamp_request", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fleet, "_retire_fleet_future", retire_future)
    monkeypatch.setattr(
        fleet, "_close_fleet_progress",
        lambda _progress: events.append("progress-close"),
    )
    monkeypatch.setattr(fleet, "_retire_fleet_audit", retire_audit)
    monkeypatch.setattr(
        fleet, "mark_defunct_on_supervision_failure",
        lambda _handle, error: events.append(("supervision", error)),
    )
    spec = SimpleNamespace(mesh=SimpleNamespace(attention="flash", sync_ulysses=False))
    session = SimpleNamespace(track=lambda owned: events.append(("track", owned)))

    with pytest.raises(termination_type) as caught:
        fleet.DGXMonarchFleetKSampler()._fleet_bound(
            spec,
            {"samples": object()},
            [{"positive": "pos", "negative": "neg", "seed": 1}],
            1, 1.0, "euler", "simple", Handle(), session,
        )

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert events[:5] == [
        "progress-enter", ("track", future),
        ("lease", future, False), ("lease", future, False), "progress-close",
    ]
    assert events[5][0] == "audit" and events[5][1] is not None
    assert events[6] == ("supervision", primary)
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
@pytest.mark.parametrize("ordering", ["ordinary-termination", "termination-ordinary"])
def test_pipeline_multi_render_preserves_strongest_after_full_abort(
    termination_type, ordering,
):
    ordinary = RuntimeError("ordinary render failure")
    termination = termination_type("render cancelled")
    failures = (
        (ordinary, termination)
        if ordering == "ordinary-termination"
        else (termination, ordinary)
    )
    calls: list[tuple[str, BaseException]] = []

    class Pending:
        def __init__(self, failure):
            self.failure = failure
            self._state = "open"

        def result(self, *_args, **_kwargs):
            calls.append(("result", self.failure))
            raise self.failure

        def cancel(self):
            calls.append(("cancel", self.failure))

        def abandon(self):
            calls.append(("abandon", self.failure))
            self._state = "closed"

    pipe = pipeline.RenderPipeline(depth=2)
    pipe._inflight.extend(Pending(failure) for failure in failures)

    with pytest.raises(termination_type) as caught:
        pipe._collect_oldest()

    assert caught.value is termination
    assert [failure for action, failure in calls if action == "result"] == list(failures)
    assert not pipe._inflight
    assert termination.__cause__ is not termination
    assert termination.__context__ is not termination


@pytest.mark.parametrize(
    ("first_type", "later_type"),
    ((_Cancelled, _Exited), (_Exited, _Cancelled)),
)
def test_pipeline_first_cancellation_wins_after_full_abort(
    first_type, later_type,
):
    first = first_type("first render cancelled")
    later = later_type("later render cancelled")
    calls: list[BaseException] = []

    class Pending:
        def __init__(self, failure):
            self.failure = failure
            self._state = "open"

        def result(self, *_args, **_kwargs):
            calls.append(self.failure)
            raise self.failure

        def cancel(self):
            return None

        def abandon(self):
            self._state = "closed"

    pipe = pipeline.RenderPipeline(depth=2)
    pipe._inflight.extend((Pending(first), Pending(later)))

    with pytest.raises(first_type) as caught:
        pipe._collect_oldest()

    assert caught.value is first
    assert calls == [first, later]
    assert not pipe._inflight
    assert first.__cause__ is not first
    assert first.__context__ is not first


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
@pytest.mark.parametrize("path", ("collect", "push"))
def test_pipeline_abort_helper_boundary_retries_and_drains(
    monkeypatch, termination_type, path,
):
    primary = RuntimeError(f"pipeline {path} failed")
    cancellation = termination_type("pipeline abort helper interrupted")
    calls: list[str] = []

    class Pending:
        _state = "open"

        def result(self, *_args, **_kwargs):
            calls.append("result")
            raise primary

        def cancel(self):
            calls.append("cancel")

        def abandon(self):
            calls.append("abandon")
            self._state = "closed"

    pipe = pipeline.RenderPipeline(depth=2)
    if path == "collect":
        pipe._inflight.extend((Pending(), Pending()))
    else:
        pipe._inflight.append(Pending())
        monkeypatch.setattr(
            pipe, "_push_bound", lambda *_args, **_kwargs: (_ for _ in ()).throw(primary))
    actual_abort = pipe._abort
    abort_calls = []

    def interrupt_first_abort():
        abort_calls.append(True)
        if len(abort_calls) == 1:
            raise cancellation
        actual_abort()

    monkeypatch.setattr(pipe, "_abort", interrupt_first_abort)

    with pytest.raises(termination_type) as caught:
        if path == "collect":
            pipe._collect_oldest()
        else:
            pipe.push(object(), {}, {}, None, 1)

    assert caught.value is cancellation
    assert abort_calls == [True, True]
    assert not pipe._inflight
    assert "abandon" in calls
    assert cancellation.__cause__ is primary
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_gate_session_close_cancellation_outranks_ordinary_primary(termination_type):
    primary = RuntimeError("ceremony failed")
    cancellation = termination_type("session close cancelled")

    with pytest.raises(termination_type) as caught:
        gate._close_gate_session(
            SimpleNamespace(close=lambda: (_ for _ in ()).throw(cancellation)),
            primary,
        )

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary


@pytest.mark.parametrize("termination_type", _TERMINATIONS)
def test_gate_quarantine_cancellation_outranks_ordinary_ceremony(
    monkeypatch, termination_type,
):
    primary = RuntimeError("ceremony failed")
    cancellation = termination_type("quarantine cancelled")
    handle = object()
    model = SimpleNamespace(mesh=SimpleNamespace(handle=handle))
    calls: list[str] = []

    class Session:
        def bind(self, _handle):
            calls.append("bind")

        def activate(self):
            return nullcontext()

    monkeypatch.setattr(common, "_bind_packed_render_model", lambda *_args, **_kwargs: (model, handle))
    monkeypatch.setattr(gate_session, "model_declares_fsdp_proof", lambda *_: False)
    runtime = {
        "ensure_live": lambda value: value,
        "RenderSession": Session,
        "_run_identity_ceremony_bound": lambda *_args, **_kwargs: (_ for _ in ()).throw(primary),
        "_force_stock_quarantine": lambda *_args, **_kwargs: (
            calls.append("quarantine"), (_ for _ in ()).throw(cancellation)
        ),
        "_close_gate_session": lambda *_args: calls.append("close"),
        "log": SimpleNamespace(error=lambda *_args: None),
    }

    with pytest.raises(termination_type) as caught:
        gate_session.run_identity_ceremony(
            model, {}, {}, 1.0, 1, "manual", "run", runtime=runtime)

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary
    assert calls == ["bind", "quarantine", "close", "close"]


def test_cross_mode_hostile_capacity_diagnostics_are_total(monkeypatch):
    class HostileCapacity(RuntimeError):
        def __repr__(self):
            raise _Cancelled("capacity repr cancelled")

        def __str__(self):
            raise _Exited("capacity str exited")

    class HostileLogger:
        def warning(self, *_args):
            raise _Exited("capacity logging exited")

    failure = HostileCapacity("stock load unavailable")
    monkeypatch.setattr(gate_cross_mode.first_render, "cross_residency_check", lambda: None)
    monkeypatch.setattr(gate_cross_mode, "parse_measured", lambda _exc: None)
    runtime = {
        "_temporary_worker_policy": lambda *_args, **_kwargs: nullcontext([]),
        "_model_with_worker_overrides": lambda model, *_args, **_kwargs: model,
        "copy_transaction": lambda value: value,
        "is_artifact_binding_error": lambda _exc: False,
        "is_stock_load_capacity_error": lambda _exc: True,
        "is_memory_exhaustion": lambda _exc: False,
        "log": HostileLogger(),
    }

    verdict, latent = gate_cross_mode.run_cross_residency_reference(
        runtime=runtime,
        ceremony_model=object(),
        handle=SimpleNamespace(call_all=lambda *_args, **_kwargs: None, world=1),
        original_worker_args={},
        slab_proof=SimpleNamespace(error=None, expected=True, cycle=None),
        frozen_request={},
        frozen_latent={},
        artifact_binding={},
        cfg_value=1.0,
        steps_hint=1,
        stock_latent={"samples": object()},
        transaction_render=lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        bind_request=lambda request, _binding: request,
    )

    assert verdict is not None and verdict["verdict"] == "CAPACITY"
    assert verdict["detail"] == "<HostileCapacity>"
    assert latent is None


def test_cross_mode_hostile_error_diagnostics_are_total(monkeypatch):
    class HostileFailure(RuntimeError):
        def __repr__(self):
            raise _Cancelled("error repr cancelled")

        def __str__(self):
            raise _Exited("error str exited")

    class HostileLogger:
        def warning(self, *_args):
            raise _Exited("error logging exited")

    failure = HostileFailure("stock reference failed")
    monkeypatch.setattr(gate_cross_mode.first_render, "cross_residency_check", lambda: None)
    runtime = {
        "_temporary_worker_policy": lambda *_args, **_kwargs: nullcontext([]),
        "_model_with_worker_overrides": lambda model, *_args, **_kwargs: model,
        "copy_transaction": lambda value: value,
        "is_artifact_binding_error": lambda _exc: False,
        "is_stock_load_capacity_error": lambda _exc: False,
        "is_memory_exhaustion": gate.is_memory_exhaustion,
        "log": HostileLogger(),
    }

    verdict, latent = gate_cross_mode.run_cross_residency_reference(
        runtime=runtime,
        ceremony_model=object(),
        handle=SimpleNamespace(call_all=lambda *_args, **_kwargs: None),
        original_worker_args={},
        slab_proof=SimpleNamespace(error=None, expected=True, cycle=None),
        frozen_request={},
        frozen_latent={},
        artifact_binding={},
        cfg_value=1.0,
        steps_hint=1,
        stock_latent={"samples": object()},
        transaction_render=lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
        bind_request=lambda request, _binding: request,
    )

    assert verdict == {"verdict": "ERROR", "detail": "<HostileFailure>"}
    assert latent is None


def test_auto_gate_cleanup_cancellation_outranks_ordinary_primary():
    primary = RuntimeError("ceremony failed")
    cancellation = _Cancelled("active-context restoration cancelled")

    with pytest.raises(_Cancelled) as caught:
        auto_gate._raise_cleanup_cancellation(
            primary, cancellation, "active-context restoration failed"
        )

    assert caught.value is cancellation
    assert cancellation.__cause__ is primary


def test_fsdp_fail_closed_publication_survives_hostile_diagnostics():
    class HostilePrimary(RuntimeError):
        def add_note(self, _note):
            raise _Cancelled("note cancelled")

    class HostileCleanup(RuntimeError):
        def __repr__(self):
            raise _Cancelled("repr cancelled")

    class BrokenLogger:
        @staticmethod
        def error(*_args):
            raise _Cancelled("logging cancelled")

    class Handle:
        def __init__(self):
            self.attempts = 0
            self.defunct = False

        def _latch_ambiguous_mutation(self, *_args):
            self.attempts += 1
            raise _Cancelled("latch cancelled")

    handle = Handle()
    gate_fsdp._latch_or_retire_failed_cleanup(
        handle,
        HostilePrimary("proof failed"),
        HostileCleanup("unload failed"),
        logger=BrokenLogger(),
        timeout_s=1.0,
    )

    assert handle.attempts == 2
    assert handle.defunct is True


def test_ambiguous_rpc_dirty_publication_survives_hostile_diagnostics():
    class HostilePrimary(TimeoutError):
        def add_note(self, _note):
            raise _Cancelled("note cancelled")

        def __repr__(self):
            raise _Cancelled("repr cancelled")

    class BrokenLogger:
        @staticmethod
        def error(*_args):
            raise _Cancelled("logging cancelled")

    primary = HostilePrimary("mutation completion unknown")
    handle = SimpleNamespace(
        lock=threading.RLock(),
        setup_generation=3,
        setup_cleanup_state=mesh_setup.cleanup_in_progress(
            3, "unload RPC completion", 1.0
        ),
    )

    mesh_rpc.latch_ambiguous_mutation(
        handle, "unload", 1.0, primary, logger=BrokenLogger()
    )

    assert handle.setup_cleanup_state is not None
    assert (
        handle.setup_cleanup_state.outcome
        is mesh_setup.SetupCleanupOutcome.TIMEOUT_UNKNOWN
    )
