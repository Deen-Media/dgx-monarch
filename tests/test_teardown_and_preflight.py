"""Deliberate-teardown fault absorption, stop idempotence across recycle,
shutdown and supervision, memory-exhaustion classification, and the worker's
stock-load preflight."""
import io
import logging
import sys
import threading
import types
from types import SimpleNamespace

import pytest

from dgx_monarch import mesh_safety, mesh_setup

_TRANSPORT_FAULT = ("undeliverable message for dgxm_worker.status(): "
                    "broken link: channel closed with reason server closed")


def test_reason_marked_faults_always_absorb(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert mesh_safety.is_deliberate_teardown_fault("stopped: dgx-monarch recycle")
    assert mesh_safety.is_deliberate_teardown_fault(
        "ProcStopped: dgx-monarch client detach")
    # Even while a live replacement mesh exists: the reason is the proof.
    assert mesh_safety.is_deliberate_teardown_fault(
        "stopped: dgx-monarch recycle", live_mesh_exists=True)


def test_reasonless_transport_faults_absorb_only_inside_the_window(monkeypatch):
    # Measured 2026-07-10: the panel's status() poll racing a recycle faults
    # with transport detail only and no stop reason.
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert not mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT)
    mesh_safety.note_deliberate_teardown()
    assert mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT)
    # Unrelated fault text is never absorbed, window or not.
    assert not mesh_safety.is_deliberate_teardown_fault(
        "actor panicked: assertion failed")


def test_reasonless_faults_stay_visible_during_creation_or_with_live_mesh(monkeypatch):
    """An old generation's grace window must not hide a replacement's real
    faults: a live mesh or an in-progress creation keeps them visible, even
    inside the window."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    mesh_safety.note_deliberate_teardown()
    assert mesh_safety.is_deliberate_teardown_fault(_TRANSPORT_FAULT)
    assert not mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, live_mesh_exists=True)
    assert not mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, creation_in_progress=True)


def test_visibility_beats_the_in_flight_mark_when_attribution_is_ambiguous(monkeypatch):
    """An unrelated live mesh or an in-progress creation keeps reasonless
    faults visible even during a stop. The caller excludes the retiring handle
    from liveness, so live_mesh_exists=True means another mesh could own the
    fault."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, teardown_in_progress=True)
    assert not mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, teardown_in_progress=True, live_mesh_exists=True)
    assert not mesh_safety.is_deliberate_teardown_fault(
        _TRANSPORT_FAULT, teardown_in_progress=True, creation_in_progress=True)
    # Only transport-shaped faults absorb, even mid-stop.
    assert not mesh_safety.is_deliberate_teardown_fault(
        "actor panicked: assertion failed", teardown_in_progress=True)


def _bare_handle(**overrides):
    from dgx_monarch.mesh import MeshHandle

    handle = MeshHandle(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""), hosts=None,
        procs=None, workers=None, world=2, gpus_per_host=1, n_hosts=2,
        comfy_dir="", owns_hosts=True)
    for key, value in overrides.items():
        setattr(handle, key, value)
    return handle


@pytest.mark.parametrize(
    ("state", "token_present", "expected"),
    [
        ({"teardown_complete": True, "replacement_blocked": "failed", "defunct": True}, True, "completed"),
        ({"teardown_complete": False, "replacement_blocked": "failed", "defunct": True}, True, "blocked"),
        ({"teardown_complete": False, "replacement_blocked": None, "defunct": True}, False, "unresolved"),
        ({"teardown_complete": False, "replacement_blocked": None, "defunct": False}, True, "unresolved"),
        ({"teardown_complete": False, "replacement_blocked": None, "defunct": False}, False, "live"),
    ],
)
def test_relocated_lifecycle_verdict_preserves_precedence(
    monkeypatch, state, token_present, expected,
):
    from dgx_monarch import mesh as mesh_mod

    handle = SimpleNamespace(**state)
    monkeypatch.setattr(mesh_safety, "token_present", lambda _token: token_present)
    assert mesh_mod._coherent_lifecycle_verdict(handle) == expected
    assert mesh_safety.coherent_lifecycle_verdict(
        handle, mesh_mod.MeshAttachError) == expected


def test_relocated_lifecycle_verdict_preserves_typed_churn_refusal(monkeypatch):
    from dgx_monarch import mesh as mesh_mod

    versions = iter(range(16))
    monkeypatch.setattr(mesh_safety, "teardown_state_version", lambda: next(versions))
    handle = SimpleNamespace(
        teardown_complete=False, replacement_blocked=None, defunct=False)
    with pytest.raises(
        mesh_mod.MeshAttachError,
        match="mesh lifecycle state kept changing while deciding reuse; retry",
    ):
        mesh_mod._coherent_lifecycle_verdict(handle)


def test_relocated_fault_correlation_preserves_locked_facade(monkeypatch):
    from dgx_monarch import mesh as mesh_mod

    stopping = SimpleNamespace(teardown_complete=False, replacement_blocked=None)
    live = SimpleNamespace(teardown_complete=False, replacement_blocked=None)
    handles = {"stopping": stopping, "live": live}
    monkeypatch.setattr(mesh_mod, "_MESHES", handles)
    monkeypatch.setattr(mesh_mod, "_MESH_CREATING", {"new": object()})
    monkeypatch.setattr(
        mesh_safety, "token_within_authority", lambda token: token == id(stopping))
    expected = mesh_safety.fault_correlation_state(handles.values(), True)
    assert expected == (True, True, True)
    classify = mesh_safety.fault_correlation_state

    def classify_while_locked(current_handles, creation_in_progress):
        assert mesh_mod._MESH_LOCK.locked()
        return classify(current_handles, creation_in_progress)

    monkeypatch.setattr(mesh_safety, "fault_correlation_state", classify_while_locked)
    assert mesh_mod._fault_correlation_state() == expected


def _interrupted_end_recycle(monkeypatch):
    """Run a real successful recycle whose end() is interrupted before its
    atomic store, leaving the token stuck and the outcome published."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))

    class Procs:
        def stop(self, reason):
            return SimpleNamespace(get=lambda timeout: None)

    handle = _bare_handle(procs=Procs())

    def end_interrupted(token, stop_confirmed):
        raise _Interrupted()   # before the store: the old snapshot survives

    monkeypatch.setattr(mesh_safety, "end_deliberate_teardown", end_interrupted)
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    monkeypatch.undo()  # restore the real end/state for the assertions below
    return handle


def test_interrupted_end_leaves_no_authority_with_the_stuck_token(monkeypatch):
    """end() interrupted before its store after a successful stop leaves the
    token stuck and the handle not defunct, but the published outcome strips
    the token's authority: no in-flight suppression, no liveness exclusion,
    and ensure_live never returns the stopped handle."""
    import dgx_monarch.mesh as mesh_mod

    handle = _interrupted_end_recycle(monkeypatch)
    stuck = id(handle)
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE",
                        ({stuck: mesh_safety.time.monotonic()}, float("-inf"), 1))
    assert handle.teardown_complete is True
    assert handle.defunct is False           # the post-finally line was skipped
    key = ("test", "stuck-token")
    mesh_mod._MESHES[key] = handle
    try:
        live, _creating, in_flight = mesh_mod._fault_correlation_state()
    finally:
        mesh_mod._MESHES.pop(key, None)
    assert in_flight is False   # the published outcome strips its authority
    assert live is False        # completed handle is settled, not live
    # The stopped handle must never be reused: ensure_live routes to respawn.
    sentinel = object()
    monkeypatch.setattr(mesh_mod, "get_mesh", lambda **kw: sentinel)
    assert mesh_mod.ensure_live(handle) is sentinel


def test_interrupted_end_faults_stay_visible_through_the_real_hook(
        monkeypatch, installed_hook):
    """With a stuck token, a published outcome and no grace, later reasonless
    faults reach the real hook as ERROR, never as endless INFO absorption."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    handle = _interrupted_end_recycle(monkeypatch)
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE",
                        ({id(handle): mesh_safety.time.monotonic()}, float("-inf"), 1))
    key = ("test", "stuck-token-hook")
    mesh_mod._MESHES[key] = handle
    try:
        hook(_TRANSPORT_FAULT)
    finally:
        mesh_mod._MESHES.pop(key, None)
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_interrupted_end_in_shutdown_has_the_same_guarantees(monkeypatch):
    """Shutdown: the same single end() call, the same stuck-token guarantees."""
    import dgx_monarch.mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))

    class Procs:
        def stop(self, reason):
            return SimpleNamespace(get=lambda timeout: None)

    handle = _bare_handle(procs=Procs())

    def end_interrupted(token, stop_confirmed):
        raise _Interrupted()

    monkeypatch.setattr(mesh_safety, "end_deliberate_teardown", end_interrupted)
    with pytest.raises(_Interrupted):
        handle._shutdown_impl()
    monkeypatch.undo()
    assert handle.teardown_complete is True
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE",
                        ({id(handle): mesh_safety.time.monotonic()}, float("-inf"), 1))
    key = ("test", "stuck-token-shutdown")
    mesh_mod._MESHES[key] = handle
    try:
        live, _creating, in_flight = mesh_mod._fault_correlation_state()
    finally:
        mesh_mod._MESHES.pop(key, None)
    assert (live, in_flight) == (False, False)


def test_clear_stale_token_never_mints_grace(monkeypatch):
    """clear_stale_token clears the token and bumps the version but never
    opens the grace window retroactively."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE",
                        ({99: 0.0}, float("-inf"), 5))
    mesh_safety.clear_stale_token(99)
    tokens, grace, version = mesh_safety._TEARDOWN_STATE
    assert tokens == {}
    assert grace == float("-inf")
    assert version == 6


def test_active_tokens_and_authority_boundary_are_exact(monkeypatch):
    issued = 1000.0
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({7: issued, 9: issued}, 0.0, 1))
    monkeypatch.setattr(mesh_safety.time, "monotonic", lambda: issued)
    assert mesh_safety.active_teardown_tokens() == frozenset({7, 9})
    assert mesh_safety.token_within_authority(7)
    monkeypatch.setattr(
        mesh_safety.time,
        "monotonic",
        lambda: issued + mesh_safety.TOKEN_AUTHORITY_S,
    )
    assert mesh_safety.token_within_authority(7)
    monkeypatch.setattr(
        mesh_safety.time,
        "monotonic",
        lambda: issued + mesh_safety.TOKEN_AUTHORITY_S + 0.001,
    )
    assert not mesh_safety.token_within_authority(7)
    assert not mesh_safety.token_within_authority(8)


def test_blocked_recycle_gets_fresh_authority_until_cleanup_confirms(monkeypatch):
    """A failed attempt blocks reuse but not a later explicit cleanup.

    Every retry receives a fresh token. A second failure remains visible and
    retryable, while the first confirmed completion opens grace exactly once.
    """
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    calls = []
    outcomes = iter((RuntimeError("first stop failed"),
                     RuntimeError("retry stop failed"), None))

    class Procs:
        def stop(self, reason):
            calls.append(reason)

            def get(timeout):
                outcome = next(outcomes)
                if outcome is not None:
                    raise outcome

            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    assert handle._recycle_impl() is False
    assert "first stop failed" in (handle.replacement_blocked or "")
    assert handle._recycle_impl() is False
    assert "retry stop failed" in (handle.replacement_blocked or "")
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")
    assert handle._recycle_impl() is True
    assert calls == ["dgx-monarch recycle"] * 3
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert mesh_safety._TEARDOWN_STATE[1] != float("-inf")
    assert not mesh_safety.deliberate_teardown_in_progress()


def test_blocked_retry_replaces_a_non_authoritative_stale_token(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    calls = []

    class Procs:
        def stop(self, reason):
            calls.append(reason)
            return SimpleNamespace(get=lambda timeout: None)

    handle = _bare_handle(
        procs=Procs(), defunct=True,
        replacement_blocked="earlier stop failed",
        _replacement_retryable=True,
    )
    mesh_safety.begin_deliberate_teardown(id(handle))

    assert handle._recycle_impl() is True

    assert calls == ["dgx-monarch recycle"]
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert not mesh_safety.token_present(id(handle))


def test_actual_recycle_stop_marks_in_flight_then_opens_the_window(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    seen = {}

    class Procs:
        def stop(self, reason):
            assert reason == "dgx-monarch recycle"
            seen["in_flight_during_stop"] = mesh_safety.deliberate_teardown_in_progress()
            return SimpleNamespace(get=lambda timeout: None)

    handle = _bare_handle(procs=Procs())
    assert handle._recycle_impl() is True
    assert seen["in_flight_during_stop"] is True
    assert not mesh_safety.deliberate_teardown_in_progress()  # ended in finally
    assert mesh_safety._TEARDOWN_STATE[1] != float("-inf")


def test_failed_recycle_stop_never_opens_the_grace_window(monkeypatch):
    """A failed procs.stop leaves processes of unknown state; their follow-on
    faults must stay visible, so no post-stop grace opens."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))

    class Procs:
        def stop(self, reason):
            def get(timeout):
                raise RuntimeError("stop timed out")
            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    assert handle._recycle_impl() is False
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")
    assert not mesh_safety.deliberate_teardown_in_progress()  # token cleared


def test_recycle_stop_timeout_is_unknown_and_never_retried(monkeypatch):
    from dgx_monarch.mesh import RecycleStatus

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    calls = []

    class Procs:
        def stop(self, reason):
            calls.append(reason)

            def get(timeout):
                raise TimeoutError("stop acknowledgement timed out")

            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    first = handle._recycle_detailed_impl()
    second = handle._recycle_detailed_impl()

    assert first.status is RecycleStatus.PROC_STOP_TIMED_OUT
    assert second.status is RecycleStatus.PRIOR_TEARDOWN_UNKNOWN
    assert handle._replacement_retryable is False
    assert calls == ["dgx-monarch recycle"]


class _Interrupted(BaseException):
    """Non-Exception interruption (cancellation/KeyboardInterrupt class)."""


def test_interrupt_inside_begin_still_clears_the_token(monkeypatch):
    """Recycle: an interrupt after begin() publishes the token but before the
    call returns must still reach the unconditional end(). begin() sits inside
    the paired try, so an interrupt at this boundary still reaches end(),
    which clears the token."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    real_begin = mesh_safety.begin_deliberate_teardown

    def begin_then_interrupt(token):
        real_begin(token)      # the token is published...
        raise _Interrupted()   # ...and the interrupt lands before returning

    monkeypatch.setattr(mesh_safety, "begin_deliberate_teardown", begin_then_interrupt)
    handle = _bare_handle(setup_key=("nccl",))
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    assert not mesh_safety.deliberate_teardown_in_progress()   # token cleared
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")     # no grace
    assert getattr(handle, "teardown_complete", False) is False


def test_interrupt_inside_begin_in_shutdown_still_clears_the_token(monkeypatch):
    """Shutdown: same boundary, same guarantee."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    real_begin = mesh_safety.begin_deliberate_teardown

    def begin_then_interrupt(token):
        real_begin(token)
        raise _Interrupted()

    monkeypatch.setattr(mesh_safety, "begin_deliberate_teardown", begin_then_interrupt)
    handle = _bare_handle(setup_key=("nccl",))
    with pytest.raises(_Interrupted):
        handle._shutdown_impl()
    assert not mesh_safety.deliberate_teardown_in_progress()
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")
    assert getattr(handle, "teardown_complete", False) is False


def test_teardown_state_transitions_are_atomic_under_interruption(monkeypatch):
    """An interrupt at any line inside a state mutator leaves the snapshot
    untouched or fully transitioned, never partly published (for example grace
    advanced behind an unchanged version, which makes a bracket look coherent
    when it is not). The state is one name binding, so a mixed state cannot
    exist; this test interrupts every line boundary through sys.settrace and
    rejects any mix. An unchanged end() (interrupt before its store) is atomic
    but not safe on its own: the outcome-aware consumers strip the stuck
    token's authority (the interrupted-end tests above). Each boundary gets a
    fresh writer lock, because raising from a trace hook at the with-exit
    bookkeeping event leaks the lock and would deadlock the next boundary. The
    leak comes from the tracer, not the product: real async exceptions reach
    __exit__ through the exception table, and production installs no tracer."""
    import sys

    token = 12345

    def full_note(old, new):
        return new[0] == old[0] and new[1] > old[1] and new[2] == old[2] + 1

    def full_begin(old, new):
        return (set(new[0]) == set(old[0]) | {token}
                and new[1] == old[1] and new[2] == old[2] + 1)

    def full_end(old, new):
        return (set(new[0]) == set(old[0]) - {token}
                and new[1] > old[1] and new[2] == old[2] + 1)

    cases = [
        (lambda: mesh_safety.note_deliberate_teardown(), full_note, {}),
        (lambda: mesh_safety.begin_deliberate_teardown(token), full_begin, {}),
        (lambda: mesh_safety.end_deliberate_teardown(token, stop_confirmed=True),
         full_end, {token: 0.0}),
    ]
    for mutate, is_full, initial_tokens in cases:
        codes = {mesh_safety.note_deliberate_teardown.__code__,
                 mesh_safety.begin_deliberate_teardown.__code__,
                 mesh_safety.end_deliberate_teardown.__code__}
        target = 1
        while True:
            assert target < 50, "tracer sweep failed to terminate"
            monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE",
                                (initial_tokens, 0.0, 7))
            monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE_LOCK",
                                threading.Lock())
            old = mesh_safety._TEARDOWN_STATE
            seen = 0
            interrupted = False

            def tracer(frame, event, arg, codes=codes, target=target):
                nonlocal seen, interrupted
                if event == "line" and frame.f_code in codes:
                    seen += 1
                    if seen == target:
                        interrupted = True
                        raise _Interrupted()
                return tracer

            sys.settrace(tracer)
            try:
                try:
                    mutate()
                except _Interrupted:
                    pass
            finally:
                sys.settrace(None)
            state = mesh_safety._TEARDOWN_STATE
            assert state == old or is_full(old, state), (
                f"partial publication at line boundary {target}: {old} -> {state}")
            if not interrupted:
                break   # ran past the mutator's last line; every boundary covered
            target += 1


def _trapped_handle(trap_attr="setup_key", **overrides):
    """A MeshHandle whose ``trap_attr`` store raises once armed, to interrupt
    at one exact attribute store."""
    from dgx_monarch.mesh import MeshHandle

    class Trapped(MeshHandle):
        def __setattr__(self, name, value):
            if name == self.__dict__.get("_trap_attr") and self.__dict__.get("_armed"):
                raise _Interrupted()
            super().__setattr__(name, value)

    handle = Trapped(
        config=SimpleNamespace(worker_args={}, hosts=(), source=""), hosts=None,
        procs=None, workers=None, world=2, gpus_per_host=1, n_hosts=2,
        comfy_dir="", owns_hosts=True)
    for key, value in overrides.items():
        setattr(handle, key, value)
    handle.__dict__["_trap_attr"] = trap_attr
    handle.__dict__["_armed"] = True
    return handle


class _OkProcs:
    def stop(self, reason):
        return SimpleNamespace(get=lambda timeout: None)


def _age_token(monkeypatch, token):
    tokens, grace, version = mesh_safety._TEARDOWN_STATE
    aged = {**tokens, token: tokens[token] - (mesh_safety.TOKEN_AUTHORITY_S + 10)}
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", (aged, grace, version))


def test_recycle_is_idempotent_on_a_completed_handle(monkeypatch):
    """A panel retry that recycles a completed handle (its end() interrupted)
    succeeds without a second ProcMesh stop, sets no blocked state over the
    confirmed success, clears the stranded token and retires the handle."""
    import dgx_monarch.mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stops = []

    class Procs:
        def stop(self, reason):
            stops.append(reason)
            raise AssertionError("a completed handle must never be re-stopped")

    handle = _bare_handle(procs=Procs(), teardown_complete=True)
    stuck = id(handle)
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE",
                        ({stuck: mesh_safety.time.monotonic()}, float("-inf"), 1))
    key = ("test", "idempotent-recycle")
    mesh_mod._MESHES[key] = handle
    try:
        assert handle._recycle_impl() is True
    finally:
        mesh_mod._MESHES.pop(key, None)
    assert stops == []
    assert handle.replacement_blocked is None
    assert handle.defunct is True
    assert not mesh_safety.deliberate_teardown_in_progress()  # token healed
    assert key not in mesh_mod._MESHES                        # evicted
    # And ensure_live routes the retired handle to respawn, never reuse.
    sentinel = object()
    monkeypatch.setattr(mesh_mod, "get_mesh", lambda **kw: sentinel)
    assert mesh_mod.ensure_live(handle) is sentinel


def test_interrupt_at_the_success_store_is_bounded_then_visible(monkeypatch, installed_hook):
    """A BaseException at the teardown_complete store (after the real stop
    succeeded) skips the rest of the finally, leaving the token stuck with no
    outcome. Authority is time-bounded: within TOKEN_AUTHORITY_S the stop's
    own stragglers absorb; past it the token expires, faults surface through
    the real hook, and reuse is refused."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _trapped_handle(trap_attr="teardown_complete", procs=_OkProcs())
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    stuck = id(handle)
    assert stuck in mesh_safety._TEARDOWN_STATE[0]       # token stranded
    assert getattr(handle, "teardown_complete", False) is False
    assert handle.replacement_blocked is None
    key = ("test", "success-store-interrupt")
    mesh_mod._MESHES[key] = handle
    try:
        live, _creating, in_flight = mesh_mod._fault_correlation_state()
        assert (live, in_flight) == (False, True)        # bounded absorb window
        _age_token(monkeypatch, stuck)
        live, _creating, in_flight = mesh_mod._fault_correlation_state()
        assert (live, in_flight) == (True, False)        # authority expired
        hook(_TRANSPORT_FAULT)
    finally:
        mesh_mod._MESHES.pop(key, None)
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_young_outcome_less_token_refuses_reuse_and_restop(monkeypatch):
    """The authority window allows fault absorption only. A young stranded
    token (interrupt at the success store) refuses ensure_live and get_mesh
    reuse and a panel-retry recycle at once: it never returns the stopped
    ProcMesh or issues a second stop."""
    import dgx_monarch.mesh as mesh_mod
    from dgx_monarch.mesh import MeshAttachError

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stops = []

    class Procs:
        def stop(self, reason):
            stops.append(reason)
            return SimpleNamespace(get=lambda timeout: None)

    handle = _trapped_handle(trap_attr="teardown_complete", procs=Procs())
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    assert stops == ["dgx-monarch recycle"]          # the real, successful stop
    assert mesh_safety.token_present(id(handle))     # stranded and young
    assert getattr(handle, "teardown_complete", False) is False
    # Reuse is refused on presence alone; no aging is required.
    with pytest.raises(MeshAttachError, match="outcome was lost"):
        mesh_mod.ensure_live(handle)
    # A panel retry refuses to re-stop and sets no blocked state.
    handle.__dict__["_armed"] = False
    assert handle._recycle_impl() is False
    assert stops == ["dgx-monarch recycle"]          # still exactly one stop
    assert handle.replacement_blocked is None


def test_ordinary_shutdown_refuses_an_outcome_less_token(monkeypatch):
    """Ordinary shutdown never re-stops either. After an interrupted success
    store (token present, no outcome), shutdown() refuses: it issues no second
    'client detach' stop and sets no blocked state. A supervision capability
    does not prove the ProcMesh survived, so it issues no second stop either."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stops = []

    class Procs:
        def stop(self, reason):
            stops.append(reason)
            return SimpleNamespace(get=lambda timeout: None)

    handle = _trapped_handle(trap_attr="teardown_complete", procs=Procs())
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    assert stops == ["dgx-monarch recycle"]
    assert mesh_safety.token_present(id(handle))
    handle.__dict__["_armed"] = False
    with pytest.raises(RuntimeError, match="shutdown refused"):
        handle._shutdown_impl()
    assert stops == ["dgx-monarch recycle"]          # no second stop
    assert handle.replacement_blocked is None
    # A classified supervision capability bypasses session ownership, not an
    # unknown prior stop outcome.
    reconcile_token = object()
    handle.defunct = True
    handle._supervision_reconcile_token = reconcile_token
    with pytest.raises(
        mesh_safety.PriorTeardownUnknownError,
        match="prior teardown's outcome is unknown",
    ):
        handle._shutdown_impl(_reconcile_token=reconcile_token)
    assert stops == ["dgx-monarch recycle"]
    assert handle.teardown_complete is False
    assert handle.replacement_blocked is None


def test_only_verified_supervision_capability_bypasses_owned_session(monkeypatch):
    """An arbitrary object is not teardown authority, even when the render
    session holds no sample lease, so require_no_sample_leases cannot see its
    owner. Only the capability that classified supervision issues may
    reconcile that owned, defunct fleet and issue the one ProcMesh stop."""
    from dgx_monarch.mesh import mark_defunct_on_supervision_failure
    from dgx_monarch.nodes.render_session import (
        ConcurrentRenderSessionError,
        RenderSession,
    )

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stops = []

    class Procs:
        def stop(self, reason):
            stops.append(reason)
            return SimpleNamespace(get=lambda timeout: None)

    handle = _bare_handle(procs=Procs())
    owner = RenderSession(timeout_s=0)
    owner.bind(handle)
    assert sum(handle.sample_leases.values()) == 0
    try:
        with pytest.raises(
            RuntimeError,
            match="no matching supervision-reconciliation capability",
        ):
            handle._shutdown_impl(_reconcile_token=object())
        assert stops == []
        assert handle.teardown_complete is False

        with pytest.raises(
            ConcurrentRenderSessionError,
            match="another render session owns this mesh handle",
        ):
            handle._shutdown_impl()
        assert stops == []

        mark_defunct_on_supervision_failure(
            handle,
            RuntimeError("Supervision event: broken link: server closed"),
        )
        assert stops == ["dgx-monarch client detach"]
        assert handle.teardown_complete is True
        assert handle._supervision_reconcile_token is None
    finally:
        owner.close()


def test_shutdown_outer_budget_covers_the_sequential_envelope(monkeypatch):
    """The off-loop budget covers group teardown and proc stop run in sequence
    (each capped at 60 s), or a legal second-phase stop reads as a failure.
    Caller timeouts are capped per phase, so every envelope stays inside
    TOKEN_AUTHORITY_S."""
    import dgx_monarch.mesh as mesh_mod

    budgets = []

    def fake_off_loop(fn, timeout_s, name):
        budgets.append(timeout_s)

    monkeypatch.setattr(mesh_mod.mesh_helpers, "run_blocking_off_loop", fake_off_loop)
    handle = _bare_handle()
    handle.shutdown(timeout_s=60)
    handle.shutdown(timeout_s=300)
    assert budgets[0] >= 120          # covers 60 (group) + 60 (stop)
    assert budgets[1] == budgets[0]   # phases capped at 60 whatever timeout_s
    assert budgets[1] + 60 <= mesh_safety.TOKEN_AUTHORITY_S  # inside authority
    # The inner phases honor the same cap.
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    waits = []

    class Procs:
        def stop(self, reason):
            def get(timeout):
                waits.append(timeout)
            return SimpleNamespace(get=get)

    inner = _bare_handle(procs=Procs())
    inner._shutdown_impl(timeout_s=300)
    assert waits == [60]


def test_shutdown_outer_timeout_leaves_outcome_publication_to_scratch(monkeypatch):
    """A caller deadline that expires mid-stop publishes no block; the scratch
    thread that owns the lifecycle lock publishes the outcome."""
    import dgx_monarch.mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    entered = threading.Event()
    release = threading.Event()
    threads = []
    calls = []

    class Procs:
        def stop(self, reason):
            assert reason == "dgx-monarch client detach"
            calls.append(reason)

            def get(timeout):
                entered.set()
                assert release.wait(timeout=5)

            return SimpleNamespace(get=get)

    def timeout_while_scratch_owns_lifecycle(fn, timeout_s, thread_name):
        del timeout_s
        thread = threading.Thread(target=fn, name=thread_name)
        threads.append(thread)
        thread.start()
        assert entered.wait(timeout=5)
        raise TimeoutError("outer deadline")

    monkeypatch.setattr(
        mesh_mod.mesh_helpers, "run_blocking_off_loop",
        timeout_while_scratch_owns_lifecycle,
    )
    handle = _bare_handle(procs=Procs())
    try:
        with pytest.raises(
            mesh_safety.PriorTeardownUnknownError,
            match="still publishes the outcome",
        ):
            handle.shutdown(timeout_s=1)
        assert handle.replacement_blocked is None
        assert mesh_safety.token_present(id(handle))
        with pytest.raises(mesh_setup.LifecycleBusyError):
            handle._shutdown_impl(lock_timeout_s=0.01)
        assert calls == ["dgx-monarch client detach"]
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert not mesh_safety.token_present(id(handle))
    assert calls == ["dgx-monarch client detach"]


def test_shutdown_retries_a_published_block_with_fresh_authority(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    calls = []

    class Procs:
        def stop(self, reason):
            calls.append(reason)

            def get(timeout):
                if len(calls) == 1:
                    raise RuntimeError("first detach failed")

            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    with pytest.raises(RuntimeError, match="ProcMesh stop failed"):
        handle._shutdown_impl()
    assert "first detach failed" in (handle.replacement_blocked or "")

    handle._shutdown_impl()

    assert calls == ["dgx-monarch client detach"] * 2
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None


def test_shutdown_stop_timeout_is_unknown_and_never_retried(monkeypatch):
    """A stop timeout never becomes a second stop.

    The only way out of this latch is the liveness retire in mesh_liveness,
    which proves the processes are gone and issues no stop. Never widen the
    retry classifier to reach the same end.
    """
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    calls = []

    class Procs:
        def stop(self, reason):
            calls.append(reason)
            return SimpleNamespace(
                get=lambda timeout: (_ for _ in ()).throw(
                    TimeoutError("detach acknowledgement timed out")))

    handle = _bare_handle(procs=Procs())
    with pytest.raises(RuntimeError, match="ProcMesh stop timed out"):
        handle._shutdown_impl()
    with pytest.raises(
        mesh_safety.PriorTeardownUnknownError, match="second stop",
    ):
        handle._shutdown_impl()

    assert handle._replacement_retryable is False
    assert calls == ["dgx-monarch client detach"]


def test_failed_setup_outer_timeout_keeps_one_scratch_owner(monkeypatch):
    from dgx_monarch import mesh_helpers

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    entered = threading.Event()
    release = threading.Event()
    threads = []
    calls = []

    class Procs:
        def stop(self, reason):
            calls.append(reason)

            def get(timeout):
                entered.set()
                assert release.wait(timeout=5)

            return SimpleNamespace(get=get)

    def timeout_once(fn, timeout_s, thread_name):
        del timeout_s
        if threads:
            return fn()
        thread = threading.Thread(target=fn, name=thread_name)
        threads.append(thread)
        thread.start()
        assert entered.wait(timeout=5)
        raise TimeoutError("outer setup rollback deadline")

    monkeypatch.setattr(mesh_helpers, "run_blocking_off_loop", timeout_once)
    handle = _bare_handle(procs=Procs())
    try:
        first = mesh_setup.stop_failed_setup_procs(handle)
        assert isinstance(first, TimeoutError)
        assert mesh_safety.token_present(id(handle))
        assert handle.replacement_blocked is None

        second = mesh_setup.stop_failed_setup_procs(handle)
        assert isinstance(second, mesh_safety.PriorTeardownUnknownError)
        assert calls == ["dgx-monarch failed setup rollback"]
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)

    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert not mesh_safety.token_present(id(handle))
    assert mesh_setup.stop_failed_setup_procs(handle) is None
    assert calls == ["dgx-monarch failed setup rollback"]


def test_failed_setup_retry_publishes_a_fresh_in_flight_attempt(monkeypatch):
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    calls = []
    seen = {}
    handle = None

    class Procs:
        def stop(self, reason):
            calls.append(reason)
            if len(calls) == 2:
                seen["blocked"] = handle.replacement_blocked
                seen["token"] = mesh_safety.token_present(id(handle))

            def get(timeout):
                if len(calls) == 1:
                    raise RuntimeError("positive setup stop failure")

            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    first = mesh_setup.stop_failed_setup_procs(handle)
    assert isinstance(first, RuntimeError)
    assert handle._replacement_retryable is True
    assert handle.replacement_blocked is not None

    assert mesh_setup.stop_failed_setup_procs(handle) is None
    assert seen == {"blocked": None, "token": True}
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert calls == ["dgx-monarch failed setup rollback"] * 2


def test_reconcile_thread_state_uses_only_public_stdlib_evidence():
    from dgx_monarch import mesh_teardown

    assert "_started" not in mesh_teardown.reconcile_thread_started.__code__.co_consts
    pending = SimpleNamespace(ident=None, is_alive=lambda: False)
    running = SimpleNamespace(ident=None, is_alive=lambda: True)
    finished = SimpleNamespace(ident=7, is_alive=lambda: False)

    assert mesh_teardown.reconcile_thread_started(pending) is False
    assert mesh_teardown.reconcile_thread_started(running) is True
    assert mesh_teardown.reconcile_thread_started(finished) is True


def test_cache_gate_survives_a_normal_completion_mid_read(monkeypatch):
    """The gate reads defunct=False, then a concurrent recycle completes
    (publishes the outcome, clears the token, sets defunct) before the token
    read. Stale fields plus a fresh cleared token would hand back the stopped
    handle. The version-bracketed verdict retries and lands on 'completed'
    without taking handle.lock: recycle holds handle.lock while it waits on
    _MESH_LOCK to evict, and get_mesh reads the verdict under _MESH_LOCK, so
    taking handle.lock here would invert the lock order and deadlock."""
    import dgx_monarch.mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _bare_handle()
    mesh_safety.begin_deliberate_teardown(id(handle))
    real_present = mesh_safety.token_present
    fired = []

    def present_after_completion(token):
        if not fired:
            fired.append(True)
            # The concurrent recycle completes now, in publish order.
            handle.teardown_complete = True
            mesh_safety.end_deliberate_teardown(id(handle), stop_confirmed=True)
            handle.defunct = True
        return real_present(token)

    monkeypatch.setattr(mesh_safety, "token_present", present_after_completion)
    # Hold handle.lock from another thread for the whole verdict (an RLock is
    # re-entrant on the same thread, which would mask a violation): if the
    # gate ever tried to take it, the worker below would hang past the join.
    hold = threading.Event()
    release = threading.Event()

    def holder():
        with handle.lock:
            hold.set()
            release.wait(timeout=10)

    blocker = threading.Thread(target=holder)
    blocker.start()
    assert hold.wait(timeout=5)
    result = {}

    def run_verdict():
        result["verdict"] = mesh_mod._coherent_lifecycle_verdict(handle)

    worker = threading.Thread(target=run_verdict)
    worker.start()
    worker.join(timeout=5)
    try:
        assert not worker.is_alive(), "verdict blocked on handle.lock (inversion)"
    finally:
        release.set()
        blocker.join(timeout=10)
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert result["verdict"] == "completed"   # never 'live' with a stopped handle


def test_cache_gate_retries_a_torn_handle_outcome_without_token_churn(monkeypatch):
    """Blocked-to-complete publication must not yield an impossible hybrid.

    The teardown token is stable across the two handle-field stores, so a
    token-version bracket alone would accept ``complete=False, blocked=None``;
    the handle-field snapshots on both sides of the token read catch it.
    """
    import dgx_monarch.mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))

    class TornOutcome:
        def __init__(self):
            object.__setattr__(self, "teardown_complete", False)
            object.__setattr__(self, "replacement_blocked", "first stop failed")
            object.__setattr__(self, "defunct", True)
            object.__setattr__(self, "setup_cleanup_state", None)
            object.__setattr__(self, "fired", False)

        def __getattribute__(self, name):
            if name == "teardown_complete" and not object.__getattribute__(
                    self, "fired"):
                object.__setattr__(self, "fired", True)
                object.__setattr__(self, "teardown_complete", True)
                object.__setattr__(self, "replacement_blocked", None)
                return False
            return object.__getattribute__(self, name)

    handle = TornOutcome()

    assert mesh_mod._coherent_lifecycle_verdict(handle) == "completed"


def test_absorb_authority_covers_the_second_teardown_phase(monkeypatch):
    """Authority spans the whole legal envelope: the 180 s recycle deadline,
    which contains the 60 s group teardown and the 60 s proc stop. A real stop
    sampled at t=150 s, inside that deadline, still counts in flight and
    absorbs, not live and ERROR."""
    import dgx_monarch.mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _bare_handle()   # mid-stop: no outcome, not defunct
    monkeypatch.setattr(
        mesh_safety, "_TEARDOWN_STATE",
        ({id(handle): mesh_safety.time.monotonic() - 150.0}, float("-inf"), 1))
    key = ("test", "second-phase")
    mesh_mod._MESHES[key] = handle
    try:
        live, _creating, in_flight = mesh_mod._fault_correlation_state()
    finally:
        mesh_mod._MESHES.pop(key, None)
    assert (live, in_flight) == (False, True)
    assert mesh_safety.TOKEN_AUTHORITY_S >= 180.0    # envelope + margin


def test_expired_outcome_less_token_surfaces_without_a_second_stop(monkeypatch):
    """An expired token loses suppression authority, not stop idempotence.

    Handled supervision remains visible but cannot prove that a prior stop did
    not already complete before its outcome publication was interrupted.
    """
    from dgx_monarch.mesh import mark_defunct_on_supervision_failure

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stop_calls = []

    class Procs:
        def stop(self, reason):
            stop_calls.append(reason)
            raise AssertionError("an outcome-less teardown must not be re-stopped")

    handle = _bare_handle(procs=Procs())
    monkeypatch.setattr(
        mesh_safety, "_TEARDOWN_STATE",
        ({id(handle): mesh_safety.time.monotonic()
          - (mesh_safety.TOKEN_AUTHORITY_S + 10)}, float("-inf"), 1))
    mark_defunct_on_supervision_failure(
        handle, RuntimeError("Supervision event: broken link: server closed"))
    assert stop_calls == []
    assert mesh_safety.token_present(id(handle))
    assert handle.replacement_blocked is None
    assert handle.teardown_complete is False
    # A young token still absorbs: no second stop mid-teardown.
    fresh = _bare_handle()
    monkeypatch.setattr(
        mesh_safety, "_TEARDOWN_STATE",
        ({id(fresh): mesh_safety.time.monotonic()}, float("-inf"), 3))
    fresh_calls = []
    fresh.shutdown = lambda timeout_s, **kw: fresh_calls.append(timeout_s)
    mark_defunct_on_supervision_failure(
        fresh, RuntimeError("Supervision event: broken link: server closed"))
    assert fresh_calls == []


def test_later_supervision_retries_a_published_stop_failure(monkeypatch):
    """A completed reconcile thread cannot permanently consume authority."""
    from dgx_monarch.mesh import mark_defunct_on_supervision_failure

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    calls = []

    class Procs:
        def stop(self, reason):
            calls.append(reason)

            def get(timeout):
                if len(calls) == 1:
                    raise RuntimeError("first reconcile stop failed")

            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    failure = RuntimeError("Supervision event: peer closed")

    mark_defunct_on_supervision_failure(handle, failure)
    first_token = handle._supervision_reconcile_token
    assert "ProcMesh stop failed" in (handle.replacement_blocked or "")
    assert handle._replacement_retryable is True

    mark_defunct_on_supervision_failure(handle, failure)

    assert calls == ["dgx-monarch client detach"] * 2
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert handle._replacement_retryable is False
    assert handle._supervision_reconcile_token is None
    assert first_token is not None


def test_supervision_outer_timeout_leaves_settlement_to_scratch(monkeypatch):
    """The reconcile wrapper cannot publish over its live lifecycle owner."""
    import dgx_monarch.mesh as mesh_mod
    from dgx_monarch.mesh import mark_defunct_on_supervision_failure

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    entered = threading.Event()
    release = threading.Event()
    scratch_threads = []
    calls = []

    class Procs:
        def stop(self, reason):
            calls.append(reason)

            def get(timeout):
                entered.set()
                assert release.wait(timeout=5)

            return SimpleNamespace(get=get)

    def timeout_while_scratch_owns_lifecycle(fn, timeout_s, thread_name):
        del timeout_s
        thread = threading.Thread(target=fn, name=thread_name)
        scratch_threads.append(thread)
        thread.start()
        assert entered.wait(timeout=5)
        raise TimeoutError("outer reconcile deadline")

    monkeypatch.setattr(
        mesh_mod.mesh_helpers, "run_blocking_off_loop",
        timeout_while_scratch_owns_lifecycle)
    handle = _bare_handle(
        procs=Procs(), defunct=True,
        replacement_blocked="prior positive stop failure",
        _replacement_retryable=True,
    )
    failure = RuntimeError("Supervision event: peer closed")
    try:
        mark_defunct_on_supervision_failure(handle, failure)
        assert entered.is_set()
        assert calls == ["dgx-monarch client detach"]
        assert handle.replacement_blocked is None
        assert handle._replacement_retryable is False
        assert mesh_safety.token_present(id(handle))

        # A second supervision event sees the live token and must not launch
        # another reconcile stop while the scratch lifecycle owner is blocked.
        mark_defunct_on_supervision_failure(handle, failure)
        assert calls == ["dgx-monarch client detach"]
        assert handle.replacement_blocked is None
        assert mesh_safety.token_present(id(handle))
    finally:
        release.set()
        for thread in scratch_threads:
            thread.join(timeout=5)

    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert not mesh_safety.token_present(id(handle))


def test_interrupt_at_the_unknown_latch_store_expires_the_same_way(monkeypatch):
    """An interrupt at the replacement_blocked latch store (after the stop
    itself was interrupted) also strands the token with no outcome. Once the
    token expires the handle counts live and ensure_live refuses it, so it is
    never reused unnoticed."""
    import dgx_monarch.mesh as mesh_mod
    from dgx_monarch.mesh import MeshAttachError

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))

    class InterruptingProcs:
        def stop(self, reason):
            def get(timeout):
                raise _Interrupted()
            return SimpleNamespace(get=get)

    handle = _trapped_handle(trap_attr="replacement_blocked",
                             procs=InterruptingProcs())
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    stuck = id(handle)
    assert stuck in mesh_safety._TEARDOWN_STATE[0]
    assert handle.replacement_blocked is None            # latch store trapped
    _age_token(monkeypatch, stuck)
    key = ("test", "latch-store-interrupt")
    mesh_mod._MESHES[key] = handle
    try:
        live, _creating, in_flight = mesh_mod._fault_correlation_state()
    finally:
        mesh_mod._MESHES.pop(key, None)
    assert (live, in_flight) == (True, False)
    with pytest.raises(MeshAttachError, match="outcome was lost"):
        mesh_mod.ensure_live(handle)


def test_interruption_at_the_setup_key_store_still_clears_the_token(monkeypatch):
    """Recycle: a BaseException at the setup-key store, after
    begin_deliberate_teardown() and before any group teardown or stop, still
    reaches the paired finally: token cleared, nothing certified, no grace."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _trapped_handle(setup_key=("nccl",))
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    assert not mesh_safety.deliberate_teardown_in_progress()   # token cleared
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")  # no grace
    assert getattr(handle, "teardown_complete", False) is False
    assert handle.replacement_blocked is None  # nothing destructive began


def test_interruption_at_the_setup_key_store_in_shutdown_clears_the_token(monkeypatch):
    """Shutdown: same boundary, same guarantees."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _trapped_handle(setup_key=("nccl",))
    with pytest.raises(_Interrupted):
        handle._shutdown_impl()
    assert not mesh_safety.deliberate_teardown_in_progress()
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")
    assert getattr(handle, "teardown_complete", False) is False


def test_interrupted_recycle_never_certifies_an_unissued_stop(monkeypatch):
    """A BaseException in teardown_group, before procs.stop() is issued,
    certifies nothing, and the recycle stays retryable with procs untouched.
    Never initialize `stopped` to True: the finally would read this interrupt
    as a confirmed stop, publish teardown_complete, open grace and clear the
    token as a success."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stop_calls = []

    class Procs:
        def stop(self, reason):
            stop_calls.append(reason)
            return SimpleNamespace(get=lambda timeout: None)

    class Workers:
        class teardown_group:  # mimics the actor endpoint shape
            @staticmethod
            def call(release_models=True):
                def get(timeout):
                    raise _Interrupted()
                return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs(), workers=Workers(), setup_key=("nccl",))
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    assert stop_calls == []                                   # never issued
    assert getattr(handle, "teardown_complete", False) is False
    assert handle.replacement_blocked is None                 # retry is safe
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")  # no grace
    assert not mesh_safety.deliberate_teardown_in_progress()  # token cleared
    # The destructive group teardown began, so the stale setup identity is
    # retired: a reused handle must re-run setup, never trust partly torn-down
    # NCCL groups.
    assert handle.setup_key is None
    assert handle.setup_cleanup_state.outcome is mesh_setup.SetupCleanupOutcome.INTERRUPTED
    from dgx_monarch import mesh as mesh_mod

    with pytest.raises(mesh_mod.MeshAttachError, match="group cleanup is dirty"):
        mesh_mod.ensure_live(handle)
    # A recycle retry after the interruption skips the (already-begun) group
    # teardown and goes straight to the proc stop.
    handle.workers = None
    assert handle._recycle_impl() is True
    assert stop_calls == ["dgx-monarch recycle"]


def test_interrupted_shutdown_group_teardown_latches_dirty_before_retry(monkeypatch):
    from dgx_monarch import mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stop_calls = []

    class Procs:
        def stop(self, reason):
            stop_calls.append(reason)
            return SimpleNamespace(get=lambda timeout: None)

    class Workers:
        class teardown_group:
            @staticmethod
            def call(release_models=True):
                return SimpleNamespace(
                    get=lambda timeout: (_ for _ in ()).throw(_Interrupted()))

    handle = _bare_handle(procs=Procs(), workers=Workers(), setup_key=("nccl",))
    with pytest.raises(_Interrupted):
        handle._shutdown_impl()
    assert stop_calls == []
    assert handle.setup_cleanup_state.outcome is mesh_setup.SetupCleanupOutcome.INTERRUPTED
    with pytest.raises(mesh_mod.MeshAttachError, match="group cleanup is dirty"):
        mesh_mod.ensure_live(handle)

    handle.workers = None
    handle._shutdown_impl()
    assert stop_calls == ["dgx-monarch client detach"]


def test_interruption_during_the_stop_latches_unknown_state(monkeypatch):
    """An interruption while procs.stop() is in flight leaves process state
    unknown: latch replacement_blocked, certify nothing, open no grace."""
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))

    class Procs:
        def stop(self, reason):
            def get(timeout):
                raise _Interrupted()
            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    with pytest.raises(_Interrupted):
        handle._recycle_impl()
    assert getattr(handle, "teardown_complete", False) is False
    assert "process state unknown" in (handle.replacement_blocked or "")
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")
    assert not mesh_safety.deliberate_teardown_in_progress()


def test_reason_marked_fault_absorbs_despite_an_interleaving_transition(
        monkeypatch, installed_hook):
    """A fault that carries the explicit stop reason absorbs at INFO, never
    ERROR: the reason is the attribution, so no correlation or coherence check
    may reject it. The hook returns on the reason before it reads correlation
    state, so the patched token_within_authority below is never reached."""
    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    real_authority = mesh_safety.token_within_authority

    def authority_and_transition(token):
        mesh_safety.note_deliberate_teardown()  # would bump the version; the hook never calls it
        return real_authority(token)

    monkeypatch.setattr(mesh_safety, "token_within_authority",
                        authority_and_transition)
    try:
        hook("Supervision event: stopped: dgx-monarch recycle")
    finally:
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert _absorbed(records)
    assert not any(r.levelno >= logging.ERROR for r in records)


def test_hook_fails_visible_when_grace_opens_after_the_version_sample(
        monkeypatch, installed_hook):
    """The classifier reads the grace deadline itself, so the version bracket
    closes after classification. A confirmed teardown that publishes grace
    between the correlation reads and the classifier's grace read would
    otherwise absorb a fault delivered before that transition. The bracket,
    checked after classification, fails visible."""
    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    real_classifier = mesh_safety.is_deliberate_teardown_fault

    def grace_lands_then_classify(text, **kwargs):
        mesh_safety.note_deliberate_teardown()   # grace and version bump now
        return real_classifier(text, **kwargs)   # classifier sees fresh grace

    monkeypatch.setattr(mesh_safety, "is_deliberate_teardown_fault",
                        grace_lands_then_classify)
    try:
        hook(_TRANSPORT_FAULT)
    finally:
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_hook_fails_visible_when_a_teardown_begins_between_its_reads(
        monkeypatch, installed_hook):
    """A is live and B's grace is open. A begins its teardown just after its
    own authority read, so the hook still counts A live, and the version bump
    from begin() fails the bracket as well. The fault reached the hook before
    A's stop began, so that stop cannot explain it, and it stays visible."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    live_handle = _bare_handle()
    key = ("test", "begin-barrier")
    mesh_mod._MESHES[key] = live_handle
    mesh_safety.note_deliberate_teardown()   # B's confirmed stop opened grace
    real_authority = mesh_safety.token_within_authority
    fired = []

    def authority_then_begin(token):
        value = real_authority(token)        # False: A not yet retiring
        if not fired:
            fired.append(True)
            mesh_safety.begin_deliberate_teardown(id(live_handle))  # begins now
        return value

    monkeypatch.setattr(mesh_safety, "token_within_authority",
                        authority_then_begin)
    try:
        hook(_TRANSPORT_FAULT)
    finally:
        mesh_safety.end_deliberate_teardown(id(live_handle), stop_confirmed=False)
        mesh_mod._MESHES.pop(key, None)
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_hook_sees_a_failure_published_between_its_two_correlation_reads(
        monkeypatch, installed_hook):
    """Handle A is mid-teardown (token active, not blocked) when the hook
    starts, and B's grace is open. A's stop then fails inside the
    token_within_authority read, publishing replacement_blocked and clearing
    its token. The fault must surface as ERROR: the handle-field snapshot
    around that read sees the failure, because it is published before the
    token clears."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    failing = _bare_handle(defunct=True)   # teardown in flight, not yet failed
    key = ("test", "toctou-failing")
    mesh_mod._MESHES[key] = failing
    mesh_safety.begin_deliberate_teardown(id(failing))
    mesh_safety.note_deliberate_teardown()   # B's confirmed stop opened grace
    real_authority = mesh_safety.token_within_authority
    fired = []

    def authority_then_transition(token):
        value = real_authority(token)       # samples with A still retiring
        if not fired:
            fired.append(True)
            failing.replacement_blocked = "proc stop timed out"   # publish...
            mesh_safety.end_deliberate_teardown(id(failing), stop_confirmed=False)
        return value                        # ...then clear; the stale answer stands

    monkeypatch.setattr(mesh_safety, "token_within_authority",
                        authority_then_transition)
    try:
        hook(_TRANSPORT_FAULT)
    finally:
        mesh_mod._MESHES.pop(key, None)
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_no_second_stop_in_the_token_to_completion_handoff(monkeypatch):
    """Completion is published before the token clears. In the other order, a
    handled supervision failure landing in the handoff sees neither signal
    and re-stops the procs ("recycle" then "client detach")."""
    from dgx_monarch.mesh import mark_defunct_on_supervision_failure

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    stops = []

    class Procs:
        def stop(self, reason):
            stops.append(reason)
            return SimpleNamespace(get=lambda timeout: None)

    handle = _bare_handle(procs=Procs())
    real_end = mesh_safety.end_deliberate_teardown

    def end_at_the_handoff(token, stop_confirmed):
        real_end(token, stop_confirmed=stop_confirmed)
        # Token is now cleared; the completion state must already be
        # published or this handled failure re-stops the fleet.
        mark_defunct_on_supervision_failure(
            handle, RuntimeError("Supervision event: broken link: server closed"))

    monkeypatch.setattr(mesh_safety, "end_deliberate_teardown", end_at_the_handoff)
    assert handle._recycle_impl() is True
    assert stops == ["dgx-monarch recycle"]
    assert handle.replacement_blocked is None


def test_blocked_handle_is_live_even_under_a_stale_retiring_snapshot(monkeypatch):
    """A blocked handle counts live whatever the retiring snapshot says.
    Otherwise a stale token snapshot excludes a handle whose stop just failed
    (token cleared, replacement_blocked set), and another handle's grace hides
    its follow-on faults."""
    import dgx_monarch.mesh as mesh_mod

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    blocked = _bare_handle(defunct=True, teardown_complete=False,
                           replacement_blocked="proc stop timed out")
    key = ("test", "stale-snapshot-blocked")
    mesh_mod._MESHES[key] = blocked
    # Simulate the stale snapshot: the token is still registered as retiring
    # when the correlation state is sampled.
    mesh_safety.begin_deliberate_teardown(id(blocked))
    try:
        live, _creating, in_flight = mesh_mod._fault_correlation_state()
    finally:
        mesh_safety.end_deliberate_teardown(id(blocked), stop_confirmed=False)
        mesh_mod._MESHES.pop(key, None)
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert live is True
    assert in_flight is False  # the published outcome strips its authority


def test_mark_defunct_absorbs_without_a_second_stop_during_active_teardown(monkeypatch):
    """A handled supervision failure (status/telemetry future) arriving while
    this handle's deliberate stop is in flight must absorb, never initiate
    another shutdown(), and must not arm post-stop grace (the stop has not
    been confirmed yet)."""
    from dgx_monarch.mesh import mark_defunct_on_supervision_failure

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _bare_handle()
    shutdown_calls = []
    handle.shutdown = lambda timeout_s, **kw: shutdown_calls.append(timeout_s)
    mesh_safety.begin_deliberate_teardown(id(handle))
    try:
        mark_defunct_on_supervision_failure(
            handle, RuntimeError("Supervision event: broken link: server closed"))
    finally:
        mesh_safety.end_deliberate_teardown(id(handle), stop_confirmed=False)
    assert shutdown_calls == []
    assert handle.defunct is True
    assert handle.replacement_blocked is None
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")


def test_mark_defunct_reason_marker_alone_never_arms_grace(monkeypatch):
    """A reason-marked supervision failure proves a deliberate stop was
    issued, not that it succeeded: absorb the call, but leave the grace
    window closed until teardown_complete says the stop confirmed."""
    from dgx_monarch.mesh import mark_defunct_on_supervision_failure

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    handle = _bare_handle()  # teardown_complete False
    shutdown_calls = []
    handle.shutdown = lambda timeout_s, **kw: shutdown_calls.append(timeout_s)
    mark_defunct_on_supervision_failure(
        handle, RuntimeError("Supervision event: stopped: dgx-monarch recycle"))
    assert shutdown_calls == []
    assert mesh_safety._TEARDOWN_STATE[1] == float("-inf")
    # A confirmed teardown does refresh the window.
    confirmed = _bare_handle(teardown_complete=True)
    mark_defunct_on_supervision_failure(
        confirmed, RuntimeError("Supervision event: stopped: dgx-monarch recycle"))
    assert mesh_safety._TEARDOWN_STATE[1] != float("-inf")


class _CaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _FakeActorError(Exception):
    """Public Monarch endpoint-error surface for isolated hook fixtures."""


@pytest.fixture
def installed_hook(monkeypatch):
    """The real fault hook, installed through mesh.install_fault_hook against
    a fake monarch.actor module, with the mesh logger captured (it does not
    propagate to root, so caplog cannot see it)."""
    import dgx_monarch.mesh as mesh_mod

    fake_actor = types.ModuleType("monarch.actor")
    fake_actor.ActorError = _FakeActorError
    fake_actor.unhandled_fault_hook = None
    fake_pkg = types.ModuleType("monarch")
    fake_pkg.actor = fake_actor
    monkeypatch.setitem(sys.modules, "monarch", fake_pkg)
    monkeypatch.setitem(sys.modules, "monarch.actor", fake_actor)
    monkeypatch.setattr(mesh_mod, "_FAULT_HOOK_INSTALLED", False)
    mesh_mod.install_fault_hook()
    assert fake_actor.unhandled_fault_hook is not None
    capture = _CaptureHandler()
    from dgx_monarch.log import get_logger

    logger = get_logger("dgx_monarch.mesh")
    logger.addHandler(capture)
    yield fake_actor.unhandled_fault_hook, capture.records
    logger.removeHandler(capture)


def _absorbed(records):
    infos = [r for r in records if "teardown notice" in r.getMessage()]
    errors = [r for r in records if r.levelno >= logging.ERROR]
    return bool(infos) and not errors


def test_headless_get_mesh_installs_hook_before_two_fault_recycle(
        monkeypatch):
    """Hardware repro (2026-07-11): a handled status fault is followed by a
    reason-marked root-actor fault while recycle is still waiting in stop().
    Direct/headless get_mesh must install the dgx-monarch hook before any
    transport work, or Monarch's default hook injects KeyboardInterrupt into
    the main thread."""
    import dgx_monarch.mesh as mesh_mod

    def default_hook(_failure):
        raise KeyboardInterrupt("Monarch default unhandled-fault hook")

    fake_actor = types.ModuleType("monarch.actor")
    fake_actor.ActorError = _FakeActorError
    fake_actor.unhandled_fault_hook = default_hook
    fake_pkg = types.ModuleType("monarch")
    fake_pkg.actor = fake_actor
    monkeypatch.setitem(sys.modules, "monarch", fake_pkg)
    monkeypatch.setitem(sys.modules, "monarch.actor", fake_actor)
    monkeypatch.setattr(mesh_mod, "_FAULT_HOOK_INSTALLED", False)

    class StopAfterHook(Exception):
        pass

    def stop_before_config(_path):
        raise StopAfterHook

    monkeypatch.setattr(mesh_mod, "find_config_path", stop_before_config)
    with pytest.raises(StopAfterHook):
        mesh_mod.get_mesh()
    assert fake_actor.unhandled_fault_hook is not default_hook

    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    in_stop, release = threading.Event(), threading.Event()
    stops = []

    class Procs:
        def stop(self, reason):
            assert reason == "dgx-monarch recycle"
            stops.append(reason)

            def get(timeout):
                in_stop.set()
                assert release.wait(timeout=10)

            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    key = ("test", "headless-two-fault")
    mesh_mod._MESHES[key] = handle
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(handle._recycle_impl()))
    worker.start()
    try:
        assert in_stop.wait(timeout=10)
        mesh_mod.mark_defunct_on_supervision_failure(
            handle, RuntimeError("Supervision event: broken link: server closed"))
        fake_actor.unhandled_fault_hook(
            "undeliverable message for dgxm_worker.status(): stopped: "
            "dgx-monarch recycle; broken link")
    finally:
        release.set()
        worker.join(timeout=10)
        mesh_mod._MESHES.pop(key, None)
    assert not worker.is_alive()
    assert outcome == [True]
    assert stops == ["dgx-monarch recycle"]
    assert handle.teardown_complete is True
    assert handle.replacement_blocked is None
    assert not mesh_safety.token_present(id(handle))


def test_an_exact_repeat_of_a_cluster_fault_is_counted_not_reprinted(
        monkeypatch, installed_hook):
    """Hardware record 2026-10-04: a lease renewal every 30 s to a dead fleet
    came back as the same unhandled fault 792 times, each one a whole ERROR
    block and a browser toast. The first report stays whole; an exact repeat
    becomes one counted line at 2, 4, 8 and no toast."""
    hook, records = installed_hook
    toasts: list[tuple] = []
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = SimpleNamespace(instance=SimpleNamespace(
        send_sync=lambda event, payload: toasts.append((event, payload))))
    monkeypatch.setitem(sys.modules, "server", fake_server)
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    fault = ("Supervision event: actor <RootClientActor client> failed:\n"
             "  undeliverable message to cast.anon-0<abc>:castmessage<def>"
             "@anon-0<abc>.tcp://10.0.0.1:26600:\n"
             "  \terror: delivery failure: ttl expired for cast.anon-0<abc>")
    for _ in range(9):
        hook(fault)
    errors = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert errors[0] == f"cluster fault: {fault}"
    assert [line.split(" times: ")[0] for line in errors[1:]] == [
        "cluster fault repeated 2", "cluster fault repeated 4",
        "cluster fault repeated 8"]
    assert all("undeliverable message to cast.anon-0<abc>" in line for line in errors[1:])
    assert toasts == [("dgx-monarch.fault", {"message": fault})]

    # A fault on another fleet is a new fault: whole report and toast again.
    other = fault.replace("abc", "xyz")
    hook(other)
    assert [r.getMessage() for r in records if r.levelno >= logging.ERROR][-1] == (
        f"cluster fault: {other}")
    assert len(toasts) == 2


def test_an_absorbed_teardown_fault_never_reaches_the_repeat_count(
        monkeypatch, installed_hook):
    """A reason-marked stop notice is INFO every time, however often it repeats,
    and it does not make the next real fault look like a repeat."""
    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    notice = "undeliverable message: stopped: dgx-monarch recycle; broken link"
    for _ in range(3):
        hook(notice)
    assert _absorbed(records)
    assert len([r for r in records if "teardown notice" in r.getMessage()]) == 3
    hook("Supervision event: the process this actor was running on failed")
    errors = [r.getMessage() for r in records if r.levelno >= logging.ERROR]
    assert len(errors) == 1 and errors[0].startswith("cluster fault: ")


def test_hook_absorbs_the_racing_poll_fault_during_the_actual_stop(
        monkeypatch, installed_hook):
    """The 2026-07-10 poll race, mid-stop: the hook fires while
    ``procs.stop()`` is blocked mid-recycle and the retiring handle is still
    registered non-defunct in the cache."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    in_stop, release = threading.Event(), threading.Event()

    class Procs:
        def stop(self, reason):
            def get(timeout):
                in_stop.set()
                release.wait(timeout=10)
            return SimpleNamespace(get=get)

    handle = _bare_handle(procs=Procs())
    key = ("test", "hook-during-stop")
    mesh_mod._MESHES[key] = handle
    worker = threading.Thread(target=handle._recycle_impl)
    worker.start()
    try:
        assert in_stop.wait(timeout=10)
        hook(_TRANSPORT_FAULT)
    finally:
        release.set()
        worker.join(timeout=10)
        mesh_mod._MESHES.pop(key, None)
    assert _absorbed(records)


def test_hook_reports_faults_when_an_unrelated_live_mesh_exists_mid_stop(
        monkeypatch, installed_hook):
    """Two non-defunct cached handles, one tearing down: a reasonless fault
    stays visible because the other live mesh could own it. Suppression is
    per teardown, never process-wide."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    in_stop, release = threading.Event(), threading.Event()

    class Procs:
        def stop(self, reason):
            def get(timeout):
                in_stop.set()
                release.wait(timeout=10)
            return SimpleNamespace(get=get)

    retiring = _bare_handle(procs=Procs())
    bystander = _bare_handle()
    keys = (("test", "retiring"), ("test", "bystander"))
    mesh_mod._MESHES[keys[0]] = retiring
    mesh_mod._MESHES[keys[1]] = bystander
    worker = threading.Thread(target=retiring._recycle_impl)
    worker.start()
    try:
        assert in_stop.wait(timeout=10)
        hook(_TRANSPORT_FAULT)
    finally:
        release.set()
        worker.join(timeout=10)
        for key in keys:
            mesh_mod._MESHES.pop(key, None)
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_hook_reports_reasonless_faults_during_replacement_creation(
        monkeypatch, installed_hook):
    """Replacement bring-up (_MESH_CREATING set, handle not yet registered)
    inside an old generation's grace window: the fault must surface as a real
    ERROR, not teardown noise."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    mesh_safety.note_deliberate_teardown()  # grace window open
    key = ("test", "hook-during-creation")
    mesh_mod._MESH_CREATING[key] = object()
    try:
        hook(_TRANSPORT_FAULT)
    finally:
        mesh_mod._MESH_CREATING.pop(key, None)
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_grace_never_crosses_a_defunct_but_unconfirmed_handle(
        monkeypatch, installed_hook):
    """Cached handle A is defunct with a failed stop (teardown_complete False,
    replacement blocked) while B's completed stop opened the grace window.
    A's processes are in an unknown state, so its follow-on reasonless faults
    stay visible."""
    import dgx_monarch.mesh as mesh_mod

    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    unconfirmed = _bare_handle(defunct=True, teardown_complete=False,
                               replacement_blocked="proc stop timed out")
    key = ("test", "unconfirmed-teardown")
    mesh_mod._MESHES[key] = unconfirmed
    mesh_safety.note_deliberate_teardown()  # B's confirmed stop opened grace
    try:
        hook(_TRANSPORT_FAULT)
    finally:
        mesh_mod._MESHES.pop(key, None)
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert not _absorbed(records)
    assert any(r.levelno >= logging.ERROR for r in records)


def test_hook_absorbs_a_post_stop_straggler_in_the_grace_window(
        monkeypatch, installed_hook):
    """The 2026-07-10 case: the poll's fault lands just after the stop
    completes, with no mesh alive and none being created."""
    hook, records = installed_hook
    monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    mesh_safety.note_deliberate_teardown()
    try:
        hook(_TRANSPORT_FAULT)
    finally:
        monkeypatch.setattr(mesh_safety, "_TEARDOWN_STATE", ({}, float("-inf"), 0))
    assert _absorbed(records)


@pytest.mark.parametrize("text,expected", [
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20 GiB", True),
    ("RuntimeError: CUDA error: out of memory", True),
    ("cudaErrorMemoryAllocation: allocation failure", True),
    ("std::bad_alloc", True),
    ("worker died", False),
    ("ArtifactBindingError: changed after the ceremony snapshot", False),
])
def test_is_memory_exhaustion(text, expected):
    assert mesh_safety.is_memory_exhaustion(RuntimeError(text)) is expected


@pytest.mark.parametrize("explode", [False, True])
def test_is_memory_exhaustion_never_tests_cause_truthiness(explode):
    class HostileOom(RuntimeError):
        def __bool__(self):
            if explode:
                raise AssertionError("exception truthiness was evaluated")
            return False

    wrapped = RuntimeError("Monarch worker failure")
    wrapped.__cause__ = HostileOom("CUDA out of memory")

    assert mesh_safety.is_memory_exhaustion(wrapped) is True


def test_stock_load_capacity_error_is_recognized_locally_and_wrapped():
    direct = mesh_safety.StockLoadCapacityError("stock residency cannot load x")
    assert mesh_safety.is_stock_load_capacity_error(direct)
    wrapped = RuntimeError("ActorError: StockLoadCapacityError: cannot load x")
    assert mesh_safety.is_stock_load_capacity_error(wrapped)
    # A generic OOM is not a capacity refusal. Only the typed load-boundary
    # refusal reads as CAPACITY in nodes/gate_cross_mode.py; an OOM is ERROR.
    assert not mesh_safety.is_stock_load_capacity_error(
        RuntimeError("CUDA out of memory"))


def test_artifact_binding_error_is_recognized_only_by_typed_marker():
    assert mesh_safety.is_artifact_binding_error(
        mesh_safety.ArtifactBindingError("identity changed"))
    assert mesh_safety.is_artifact_binding_error(
        RuntimeError("ActorError: ArtifactBindingError: identity changed"))
    assert not mesh_safety.is_artifact_binding_error(
        RuntimeError("artifact snapshot changed"))


def test_stock_preflight_refuses_an_impossible_load_on_integrated(monkeypatch, tmp_path):
    f = tmp_path / "big.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1024)
    with pytest.raises(mesh_safety.StockLoadCapacityError, match="out of memory"):
        mesh_safety.stock_load_preflight(str(f), "big.safetensors", {})


def test_stock_preflight_never_runs_on_discrete_hardware(monkeypatch, tmp_path):
    """Host MemAvailable does not bound model capacity on a discrete GPU; the
    heuristic is restricted to positively identified integrated devices."""
    f = tmp_path / "big.safetensors"
    f.write_bytes(b"x" * 4096)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: False)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1024)
    mesh_safety.stock_load_preflight(str(f), "big.safetensors", {})  # no raise


def test_stock_preflight_allows_fitting_cast_and_unknown_loads(monkeypatch, tmp_path):
    f = tmp_path / "small.safetensors"
    f.write_bytes(b"x" * 10)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1024)
    mesh_safety.stock_load_preflight(str(f), "small.safetensors", {})  # fits
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 1)
    # A dtype cast changes the resident size: never refuse on file size.
    mesh_safety.stock_load_preflight(
        str(f), "small.safetensors", {"dtype": "fp8_e4m3fn"})
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: None)
    mesh_safety.stock_load_preflight(str(f), "small.safetensors", {})  # off-Linux


def test_stock_preflight_allows_exact_memavailable_boundary(monkeypatch, tmp_path):
    f = tmp_path / "exact.safetensors"
    f.write_bytes(b"x" * 10)
    monkeypatch.setattr(mesh_safety, "gpu_is_integrated", lambda: True)
    monkeypatch.setattr(mesh_safety, "mem_available_bytes", lambda: 10)
    mesh_safety.stock_load_preflight(str(f), "exact.safetensors", {})


def test_gpu_is_integrated_handles_true_false_missing_and_probe_error(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(
        get_device_properties=lambda _index: SimpleNamespace(is_integrated=True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert mesh_safety.gpu_is_integrated()

    fake_torch.cuda.get_device_properties = lambda _index: SimpleNamespace(is_integrated=False)
    assert not mesh_safety.gpu_is_integrated()
    fake_torch.cuda.get_device_properties = lambda _index: SimpleNamespace()
    assert not mesh_safety.gpu_is_integrated()

    def unavailable(_index):
        raise RuntimeError("no CUDA device")

    fake_torch.cuda.get_device_properties = unavailable
    assert not mesh_safety.gpu_is_integrated()


def test_mem_available_bytes_parses_and_rejects_malformed_values(monkeypatch):
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.StringIO(
        "MemTotal: 10 kB\nMemAvailable: 7 kB\n"))
    assert mesh_safety.mem_available_bytes() == 7 * 1024
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.StringIO(
        "MemAvailable: not-a-number kB\n"))
    assert mesh_safety.mem_available_bytes() is None


def test_mem_available_bytes_parses_meminfo():
    value = mesh_safety.mem_available_bytes()
    assert value is None or (isinstance(value, int) and value > 0)
