"""Driver-side exclusion for logical render sessions on one mesh handle.

Setup-generation, sample-lease, and descriptor ownership authorities are
handle-scoped. Pipeline depth and fleet waves may overlap within one logical
session, but independent sessions must not share those authorities.
"""
from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from .error_utils import (
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
)

_DEFAULT_CLAIM_TIMEOUT_S = max(
    0.0, float(os.environ.get("DGXM_RENDER_SESSION_TIMEOUT", "10")))
_HANDLE_STATE_ATTRIBUTE = "_dgxm_render_session_state"  # handle lifetime: owner state rides the MeshHandle
_HANDLE_STATE_INIT = threading.Lock()


class ConcurrentRenderSessionError(RuntimeError):
    """Another logical render session still owns this mesh handle."""


class RetiredRenderHandleError(ConcurrentRenderSessionError):
    """The handle retired between lifecycle resolution and session claim."""


class _HandleSessionState:
    def __init__(self, lock: threading.RLock):
        # Share MeshHandle.lock so setup mutation and owner publication linearize.
        self.condition = threading.Condition(lock)
        self.owner: object | None = None


def _handle_state(handle: Any) -> _HandleSessionState | None:
    """Return owner state for a real handle; a test double gets None and no lock."""
    # Keep the mesh import lazy.
    from .mesh import MeshHandle

    if not isinstance(handle, MeshHandle):
        return None
    state = getattr(handle, _HANDLE_STATE_ATTRIBUTE, None)
    if state is not None:
        return state
    with _HANDLE_STATE_INIT:
        state = getattr(handle, _HANDLE_STATE_ATTRIBUTE, None)
        if state is None:
            state = _HandleSessionState(handle.lock)
            setattr(handle, _HANDLE_STATE_ATTRIBUTE, state)
    return state


def _publish_owner(state: _HandleSessionState, owner: object) -> None:
    """Small injection seam for proving interrupted owner publication."""
    state.owner = owner


def _claim_owner(state: _HandleSessionState, owner: object,
                 timeout_s: float) -> bool:
    # Condition.wait_for() only budgets the owner wait *after* its lock has
    # been acquired. This condition shares MeshHandle.lock, so setup can hold
    # that acquisition for minutes. Budget the acquisition itself so
    # RenderSession(timeout_s=...) remains a real bound.
    deadline = time.monotonic() + timeout_s
    acquired = False
    try:
        acquired = state.condition.acquire(timeout=timeout_s)
        if not acquired:
            return False
        remaining = max(0.0, deadline - time.monotonic())
        available = state.condition.wait_for(
            lambda: state.owner is None or state.owner is owner,
            timeout=remaining,
        )
        if not available:
            return False
        _publish_owner(state, owner)
        return True
    finally:
        # Recover an interrupt after acquisition but before its return assignment.
        owned = getattr(state.condition, "_is_owned", lambda: False)
        if acquired or owned():
            state.condition.release()


def _clear_owner(state: _HandleSessionState, owner: object) -> None:
    """Idempotently clear one owner and wake bounded contenders."""
    with state.condition:
        if state.owner is owner:
            state.owner = None
            state.condition.notify_all()


class RenderSession:
    """One pipeline, fleet invocation, or direct pending render.

    Binding is idempotent for the same handle so every submit in a pipeline
    can reuse the session.  Closing releases the handle and allows this object
    to bind again after a successful drain.
    """

    def __init__(self, timeout_s: float | None = None):
        self._timeout_s = (
            _DEFAULT_CLAIM_TIMEOUT_S if timeout_s is None
            else max(0.0, float(timeout_s))
        )
        self._owner = object()
        self._state_lock = threading.Lock()
        self._handle_id: int | None = None
        self._handle_state: _HandleSessionState | None = None
        self._holds: set[object] = set()  # session lifetime: the sample leases this owner still holds
        self._closing = False

    def bind(self, handle: Any) -> None:
        state = _handle_state(handle)
        if state is None:
            return
        handle_id = id(handle)
        with self._state_lock:
            if self._handle_state is not None:
                if self._handle_id != handle_id:
                    raise ConcurrentRenderSessionError(
                        "one render session cannot span multiple mesh handles; "
                        "drain or abandon it before switching handles"
                    )
                if self._closing:
                    raise ConcurrentRenderSessionError(
                        "render session is closing while a deferred result read "
                        "still owns worker-side backing"
                    )
                from . import mesh_setup

                mesh_setup.require_not_dirty(
                    handle, "continue a render or residency session")
                with state.condition:
                    if (getattr(handle, "defunct", False)
                            or getattr(handle, "teardown_complete", False)
                            or getattr(handle, "replacement_blocked", None)):
                        raise RetiredRenderHandleError(
                            "mesh retired while claiming a render session"
                        )
                return
            claimed = False
            try:
                # Publish local cleanup authority before the shared owner.
                self._handle_id = handle_id
                self._handle_state = state
                self._closing = False
                claimed = _claim_owner(state, self._owner, self._timeout_s)
                if not claimed:
                    raise ConcurrentRenderSessionError(
                        "another render session is active on this mesh handle; "
                        f"it did not drain or abandon within {self._timeout_s:g}s"
                    )
                # A DIRTY mutation keeps new sessions refused until recycle.
                from . import mesh_setup

                mesh_setup.require_not_dirty(
                    handle, "start a render or residency session")
                # Supervision can retire a handle after ensure_live but before claim.
                with state.condition:
                    if (getattr(handle, "defunct", False)
                            or getattr(handle, "teardown_complete", False)
                            or getattr(handle, "replacement_blocked", None)):
                        raise RetiredRenderHandleError(
                            "mesh retired while claiming a render session"
                        )
                return
            except BaseException as primary:
                # Retry a possibly interrupted clear; retain pairing if both fail.
                needs_clear = claimed or state.owner is self._owner
                cleared = not needs_clear
                winner = primary
                if needs_clear:
                    for _attempt in range(2):
                        try:
                            _clear_owner(state, self._owner)
                        except BaseException as cleanup_exc:
                            winner = prefer_error(
                                winner, cleanup_exc,
                                "render-session bind compensation also failed")
                        else:
                            cleared = True
                            break
                if cleared or state.owner is not self._owner:
                    self._handle_state = None
                    self._handle_id = None
                    self._closing = False
                if winner is not primary:
                    raise_with_distinct_cause(winner, primary)
                raise

    def track(self, future: Any) -> None:
        """Hold this session until a real setup-bound future finalizes."""
        from .mesh_lease import SetupBoundFuture

        if not isinstance(future, SetupBoundFuture):
            return
        # A SetupBoundFuture on a lightweight handle is a unit-test double; no
        # real setup/sample/descriptor authority exists to arbitrate.
        if _handle_state(future.handle) is None:
            return
        hold = object()

        def finalized() -> None:
            self._finish_hold(hold)

        try:
            with self._state_lock:
                if (self._handle_state is None
                        or self._handle_id != id(future.handle)):
                    raise ConcurrentRenderSessionError(
                        "cannot track a sample future outside its bound render session")
                if self._closing:
                    raise ConcurrentRenderSessionError(
                        "cannot add a sample future to a closing render session")
                self._holds.add(hold)
            future.add_finalizer(finalized)
            return
        except BaseException as primary:
            # Registration deduplicates this exact callback identity on retry.
            with self._state_lock:
                paired = hold in self._holds
            winner = primary
            if paired:
                try:
                    future.add_finalizer(finalized)
                except BaseException as retry_exc:
                    winner = prefer_error(
                        winner, retry_exc,
                        "render-session finalizer registration retry also failed")
            # Remove a proven-unregistered hold instead of waiting forever.
            registered: bool | None = None
            for _attempt in range(2):
                try:
                    registered = future.has_finalizer(finalized)
                except BaseException as proof_exc:
                    winner = prefer_error(
                        winner, proof_exc,
                        "render-session finalizer proof also failed")
                else:
                    break
            with self._state_lock:
                if hold in self._holds and registered is False:
                    self._holds.discard(hold)
                    if self._closing and not self._holds:
                        try:
                            self._release_owner_locked()
                        except BaseException as cleanup_exc:
                            winner = prefer_error(
                                winner, cleanup_exc,
                                "render-session failed-registration hold cleanup also failed",
                            )
            if winner is not primary:
                raise_with_distinct_cause(winner, primary)
            raise

    def _finish_hold(self, hold: object) -> None:
        with self._state_lock:
            try:
                self._holds.discard(hold)
                if self._closing and not self._holds:
                    self._release_owner_locked()
            except BaseException as primary:
                # Pair an interrupt between refcount retirement and owner clear.
                try:
                    self._holds.discard(hold)
                    if self._closing and not self._holds:
                        self._release_owner_locked()
                except BaseException as retry:
                    winner, cause = reconcile_error(
                        primary, retry, "render-session hold retry failed")
                    raise_with_distinct_cause(winner, cause)
                raise

    def _release_owner_locked(self) -> None:
        state = self._handle_state
        if state is None:
            self._closing = False
            return
        first_error: BaseException | None = None
        error_cause: BaseException | None = None
        cleared = False
        for _attempt in range(2):
            try:
                _clear_owner(state, self._owner)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                else:
                    first_error, error_cause = reconcile_error(
                        first_error, exc, "render-session owner clear retry failed")
            else:
                cleared = True
                break
        if cleared or state.owner is not self._owner:
            self._handle_state = None
            self._handle_id = None
            self._closing = False
        # Retain the pairing after two failed clears so a later close can retry.
        if first_error is not None:
            raise_with_distinct_cause(first_error, error_cause)

    def close(self) -> None:
        """Release after all setup futures, including deferred RDMA, finalize."""
        try:
            self._close_once()
        except BaseException as primary:
            # Retry an interruption immediately after private-lock acquisition.
            try:
                self._close_once()
            except BaseException as retry:
                winner, cause = reconcile_error(
                    primary, retry, "render-session close retry failed")
                raise_with_distinct_cause(winner, cause)
            raise

    def _close_once(self) -> None:
        with self._state_lock:
            self._closing = True
            if not self._holds:
                self._release_owner_locked()

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this session discoverable by nested ``submit_render`` calls."""
        current = _ACTIVE_SESSION.get()
        if current is not None and current is not self:
            raise ConcurrentRenderSessionError(
                "cannot activate a render session inside a different active session")
        token = _ACTIVE_SESSION.set(self)
        try:
            yield
        finally:
            _reset_active_session(token, current)


_ACTIVE_SESSION: ContextVar[RenderSession | None] = ContextVar(
    "dgxm_active_render_session", default=None)


def _reset_active_session(token: Any, previous: RenderSession | None) -> None:
    """Restore the active context across a wrapper-boundary interruption."""
    try:
        _ACTIVE_SESSION.reset(token)
    except BaseException as primary:
        try:
            _ACTIVE_SESSION.reset(token)
        except RuntimeError:
            # ContextVar tokens are one-shot. A proxy reset may delegate and
            # then raise; a consumed token plus the old value proves success.
            if _ACTIVE_SESSION.get() is not previous:
                raise
        except BaseException as cleanup_exc:
            winner, cause = reconcile_error(
                primary, cleanup_exc,
                "active render-session context reset retry also failed")
            if winner is not primary:
                raise_with_distinct_cause(winner, cause)
        raise


def require_mutation_authority(
    handle: Any, operation: str, *, require_owner: bool = False
) -> None:
    """Authorize one setup/residency mutation at ``handle.lock``.

    An unowned handle may still perform a self-contained mutation while it
    holds ``handle.lock``. Once a logical render session is published, only
    that session, activated in the current context, may mutate setup,
    residency or sample state. Sample, session-scoped and setup-bound endpoint
    dispatch pass ``require_owner``, so an unowned handle refuses them too: a
    sample's result backing must always belong to a logical session.
    """
    state = _handle_state(handle)
    if state is None:
        return
    current = _ACTIVE_SESSION.get()
    with state.condition:
        owner = state.owner
        if owner is None:
            if require_owner:
                raise ConcurrentRenderSessionError(
                    f"cannot {operation}: no active render session owns this mesh handle"
                )
            return
        if (
            current is None
            or current._owner is not owner
            or current._handle_state is not state
            or current._handle_id != id(handle)
            or current._closing
        ):
            raise ConcurrentRenderSessionError(
                f"cannot {operation}: another render session owns this mesh handle"
            )


def claim_render_session(
    handle: Any, owned_session: RenderSession
) -> tuple[RenderSession, bool]:
    """Bind the active shared session, or the caller-owned candidate.

    Returns ``(session, owned)``. A direct ``submit_render`` owns its session
    until the returned PendingRender closes; pipeline submits inherit the
    pipeline's active session and must not release it individually.

    The caller creates and stores the candidate before binding. If an
    asynchronous exception lands after owner publication but before this tuple
    returns, the caller still holds cleanup authority and can close
    ``owned_session`` instead of leaking the opaque shared owner.
    """
    session = _ACTIVE_SESSION.get()
    owned = session is None
    if session is None:
        session = owned_session
    session.bind(handle)
    return session, owned


@contextmanager
def mutation_render_session(
    handle: Any, timeout_s: float | None = None
) -> Iterator[RenderSession]:
    """Hold one session capability for a complete long-lived mutation.

    Nested callers inherit an already-active pipeline/Gate/Fleet session.
    Standalone callers retain a pre-created candidate through bind, RPC wait,
    and cleanup, closing the async-exception handoff window.
    """
    candidate = RenderSession(timeout_s=timeout_s)
    session: RenderSession | None = None
    primary: BaseException | None = None
    try:
        try:
            session, _owned = claim_render_session(handle, candidate)
            with session.activate():
                yield session
        except BaseException as exc:
            primary = exc
            raise
        finally:
            # The candidate retains cleanup authority across claim-return loss.
            try:
                _close_candidate_preserving_primary(candidate, primary)
            except BaseException as cleanup_exc:
                if primary is None:
                    primary = cleanup_exc
                    raise
                winner, cause = reconcile_error(
                    primary, cleanup_exc,
                    "render mutation session cleanup also failed")
                if winner is not primary:
                    primary = winner
                    raise_with_distinct_cause(winner, cause)
    finally:
        # The outer finally pairs a one-shot interruption; close is idempotent.
        try:
            _close_candidate_preserving_primary(candidate, primary)
        except BaseException as cleanup_exc:
            if primary is None:
                raise
            winner, cause = reconcile_error(
                primary, cleanup_exc,
                "render mutation session retry also failed")
            if winner is not primary:
                raise_with_distinct_cause(winner, cause)


def _close_candidate_preserving_primary(
    candidate: RenderSession, primary: BaseException | None
) -> None:
    try:
        candidate.close()
    except BaseException as cleanup_exc:
        if primary is None:
            raise
        winner, cause = reconcile_error(
            primary, cleanup_exc, "render mutation session cleanup also failed")
        if winner is not primary:
            raise_with_distinct_cause(winner, cause)
