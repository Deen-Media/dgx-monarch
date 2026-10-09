"""Cancellation-precedence regressions for render-session cleanup."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import dgx_monarch.mesh_session as session_mod
from dgx_monarch import mesh_setup
from dgx_monarch.mesh import MeshHandle
from dgx_monarch.mesh_lease import SetupBoundFuture
from dgx_monarch.nodes.render_session import (
    ConcurrentRenderSessionError,
    RenderSession,
)


def _real_handle() -> MeshHandle:
    """Build an identity-valid handle without constructing Monarch actors."""
    handle = object.__new__(MeshHandle)
    handle.lock = threading.RLock()
    handle.config = SimpleNamespace(worker_args={})
    return handle


def _ready_handle() -> tuple[MeshHandle, mesh_setup.SetupToken]:
    handle = _real_handle()
    handle.defunct = False
    handle.setup_cleanup_state = None
    handle.setup_generation = 1
    handle.setup_key = ("ready",)
    handle.worker_args_key = ("policy",)
    handle.sample_leases = {}
    handle.abandoned_sample_leases = {}
    handle.deferred_supervision_error = None
    return handle, mesh_setup.current_setup_token(handle)


def _ordered_failures(
    ordering: str, label: str,
) -> tuple[BaseException, BaseException, KeyboardInterrupt]:
    ordinary = RuntimeError(f"{label} ordinary failure")
    cancellation = KeyboardInterrupt(f"{label} cancelled")
    if ordering == "ordinary-cancellation":
        return ordinary, cancellation, cancellation
    if ordering == "cancellation-ordinary":
        return cancellation, ordinary, cancellation
    assert ordering == "same-cancellation"
    return cancellation, cancellation, cancellation


def _assert_exact_cancellation(
    raised: pytest.ExceptionInfo[BaseException], cancellation: KeyboardInterrupt,
) -> None:
    assert raised.value is cancellation
    assert cancellation.__cause__ is not cancellation
    assert cancellation.__context__ is not cancellation


def _assert_handle_reusable(handle: MeshHandle) -> None:
    contender = RenderSession(timeout_s=0)
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_bind_compensation_prefers_exact_cancellation_and_clears_owner(
    monkeypatch: pytest.MonkeyPatch, ordering: str,
) -> None:
    primary, cleanup, cancellation = _ordered_failures(ordering, "bind")
    handle = _real_handle()
    session = RenderSession(timeout_s=0)
    original_require_not_dirty = mesh_setup.require_not_dirty
    original_clear = session_mod._clear_owner
    clear_calls = 0

    def fail_after_claim(_handle: object, _operation: str) -> None:
        raise primary

    def fail_then_clear(state: object, owner: object) -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            raise cleanup
        original_clear(state, owner)

    monkeypatch.setattr(mesh_setup, "require_not_dirty", fail_after_claim)
    monkeypatch.setattr(session_mod, "_clear_owner", fail_then_clear)

    with pytest.raises(KeyboardInterrupt) as raised:
        session.bind(handle)

    _assert_exact_cancellation(raised, cancellation)
    assert clear_calls == 2
    assert session_mod._handle_state(handle).owner is None
    assert session._handle_state is None
    assert session._handle_id is None

    monkeypatch.setattr(mesh_setup, "require_not_dirty", original_require_not_dirty)
    monkeypatch.setattr(session_mod, "_clear_owner", original_clear)
    _assert_handle_reusable(handle)


@pytest.mark.parametrize("hostility", ["repr", "add_note"])
def test_bind_compensation_diagnostics_cannot_mask_cancellation(
    monkeypatch: pytest.MonkeyPatch, hostility: str,
) -> None:
    class HostileOrdinary(RuntimeError):
        def __repr__(self) -> str:
            if hostility == "repr":
                raise RuntimeError("ordinary repr failed")
            return super().__repr__()

    class HostileCancellation(KeyboardInterrupt):
        def add_note(self, _note: str) -> None:
            if hostility == "add_note":
                raise RuntimeError("cancellation note failed")
            super().add_note(_note)

    primary = HostileOrdinary("bind primary")
    cancellation = HostileCancellation("bind cleanup cancelled")
    handle = _real_handle()
    session = RenderSession(timeout_s=0)
    original_require_not_dirty = mesh_setup.require_not_dirty
    original_clear = session_mod._clear_owner
    clear_calls = 0

    def fail_after_claim(_handle: object, _operation: str) -> None:
        raise primary

    def fail_then_clear(state: object, owner: object) -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            raise cancellation
        original_clear(state, owner)

    monkeypatch.setattr(mesh_setup, "require_not_dirty", fail_after_claim)
    monkeypatch.setattr(session_mod, "_clear_owner", fail_then_clear)

    with pytest.raises(HostileCancellation) as raised:
        session.bind(handle)

    _assert_exact_cancellation(raised, cancellation)
    assert clear_calls == 2
    assert session_mod._handle_state(handle).owner is None

    monkeypatch.setattr(mesh_setup, "require_not_dirty", original_require_not_dirty)
    monkeypatch.setattr(session_mod, "_clear_owner", original_clear)
    _assert_handle_reusable(handle)


@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_track_registration_prefers_cancellation_and_removes_unregistered_hold(
    monkeypatch: pytest.MonkeyPatch, ordering: str,
) -> None:
    first, second, cancellation = _ordered_failures(ordering, "registration")
    errors = [first, second]
    handle, token = _ready_handle()
    future = SetupBoundFuture.prepared(handle, token.generation)
    owner = RenderSession(timeout_s=0)
    owner.bind(handle)
    calls = 0

    def fail_registration(_self: SetupBoundFuture, _callback: object) -> None:
        nonlocal calls
        error = errors[calls]
        calls += 1
        raise error

    monkeypatch.setattr(SetupBoundFuture, "add_finalizer", fail_registration)

    with pytest.raises(KeyboardInterrupt) as raised:
        owner.track(future)

    _assert_exact_cancellation(raised, cancellation)
    assert calls == 2
    assert owner._holds == set()
    owner.close()
    _assert_handle_reusable(handle)


@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_track_proof_prefers_cancellation_and_keeps_ambiguous_hold_fail_closed(
    monkeypatch: pytest.MonkeyPatch, ordering: str,
) -> None:
    proof_first, proof_second, cancellation = _ordered_failures(
        ordering, "finalizer proof")
    primary = RuntimeError("initial registration interrupted")
    proof_errors = [proof_first, proof_second]
    handle, token = _ready_handle()
    future = SetupBoundFuture.prepared(handle, token.generation)
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)
    owner.bind(handle)
    add_calls = 0
    proof_calls = 0

    def interrupted_registration(
        _self: SetupBoundFuture, _callback: object,
    ) -> None:
        nonlocal add_calls
        add_calls += 1
        if add_calls == 1:
            raise primary

    def ambiguous_proof(_self: SetupBoundFuture, _callback: object) -> bool:
        nonlocal proof_calls
        error = proof_errors[proof_calls]
        proof_calls += 1
        raise error

    monkeypatch.setattr(
        SetupBoundFuture, "add_finalizer", interrupted_registration)
    monkeypatch.setattr(SetupBoundFuture, "has_finalizer", ambiguous_proof)

    with pytest.raises(KeyboardInterrupt) as raised:
        owner.track(future)

    _assert_exact_cancellation(raised, cancellation)
    assert add_calls == 2
    assert proof_calls == 2
    assert len(owner._holds) == 1

    owner.close()
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)

    # A hold with unknown registration blocks other sessions; once an outer
    # boundary proves the work finished, finishing the hold releases the owner.
    owner._finish_hold(next(iter(owner._holds)))
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_finalizer_hold_cleanup_prefers_cancellation_and_releases_owner(
    monkeypatch: pytest.MonkeyPatch, ordering: str,
) -> None:
    first, second, cancellation = _ordered_failures(ordering, "hold cleanup")
    errors = [first, second]
    handle, token = _ready_handle()
    future = SetupBoundFuture.prepared(handle, token.generation)
    owner = RenderSession(timeout_s=0)
    contender = RenderSession(timeout_s=0)
    owner.bind(handle)
    owner.track(future)
    hold = next(iter(owner._holds))
    owner.close()
    with pytest.raises(ConcurrentRenderSessionError):
        contender.bind(handle)

    original_clear = session_mod._clear_owner
    clear_calls = 0

    def fail_twice_then_clear(state: object, session_owner: object) -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls <= 2:
            raise errors[clear_calls - 1]
        original_clear(state, session_owner)

    monkeypatch.setattr(session_mod, "_clear_owner", fail_twice_then_clear)

    with pytest.raises(KeyboardInterrupt) as raised:
        owner._finish_hold(hold)

    _assert_exact_cancellation(raised, cancellation)
    assert clear_calls == 3
    assert owner._holds == set()
    assert owner._handle_state is None
    assert session_mod._handle_state(handle).owner is None

    monkeypatch.setattr(session_mod, "_clear_owner", original_clear)
    contender.bind(handle)
    contender.close()


@pytest.mark.parametrize(
    "ordering",
    ["ordinary-cancellation", "cancellation-ordinary", "same-cancellation"],
)
def test_mutation_cleanup_prefers_exact_cancellation_and_releases_owner(
    monkeypatch: pytest.MonkeyPatch, ordering: str,
) -> None:
    body_error, cleanup_error, cancellation = _ordered_failures(
        ordering, "mutation cleanup")
    handle = _real_handle()
    original_close = session_mod._close_candidate_preserving_primary
    cleanup_calls = 0

    def fail_once_then_close(
        candidate: RenderSession, primary: BaseException | None,
    ) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        if cleanup_calls == 1:
            raise cleanup_error
        original_close(candidate, primary)

    monkeypatch.setattr(
        session_mod, "_close_candidate_preserving_primary", fail_once_then_close)

    with pytest.raises(KeyboardInterrupt) as raised:
        with session_mod.mutation_render_session(handle):
            raise body_error

    _assert_exact_cancellation(raised, cancellation)
    assert cleanup_calls == 2
    assert session_mod._handle_state(handle).owner is None

    monkeypatch.setattr(
        session_mod, "_close_candidate_preserving_primary", original_close)
    _assert_handle_reusable(handle)
