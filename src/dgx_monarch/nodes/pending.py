"""Setup-leased render result lifecycle."""
from __future__ import annotations

import threading
from typing import Any

from .. import mesh_setup
from ..log import get_logger
from ..refusal import parse_leading_refusal_tag
from ..retry_policy import retry_twice
from ..transfer_utils import (
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_call,
    safe_note,
)
from . import consent_observe
from .recycle_drain import evict_render_fleet

log = get_logger(__name__)


def typed_worker_refusal(exc: BaseException) -> bool:
    """Recognize a completed worker refusal whose sample lease can be consumed.

    Typed dgxm guards refuse before the first block-loop collective or latent
    packing. Guards using cross-rank facts finish their exchange on every rank
    before refusing, so no sample or result backing remains live. Monarch's
    ``ActorError`` identifies a completed endpoint call with a live actor.
    Read the leading tag from its inner exception; stringifying the wrapper
    exposes the remote traceback.

    Other failures abandon the lease: timeouts are ambiguous, supervision
    failures indicate a dead process, and an untagged crash may follow latent
    packing. Consuming typed refusals permits requeue after the operator
    addresses them (docs/TROUBLESHOOTING.md #55).
    """
    from monarch.actor import ActorError

    if not isinstance(exc, ActorError):
        return False
    return parse_leading_refusal_tag(consent_observe.refusal_text(exc)) is not None


class PendingRenderHandoff:
    """Caller-owned slot that closes the submit-return publication gap.

    ``submit_render`` publishes into this slot before it returns.  A caller
    interrupted before storing or queueing the return value can therefore
    still terminally abandon the submitted render.
    """

    def __init__(self) -> None:
        self._pending: PendingRender | None = None

    def publish(self, pending: PendingRender) -> None:
        if self._pending is not None:
            raise RuntimeError("pending-render handoff is already occupied")
        self._pending = pending

    def clear(self, pending: PendingRender) -> None:
        if self._pending is pending:
            self._pending = None

    def abort(self) -> None:
        """Best-effort, idempotent retirement of an unclaimed publication."""
        pending = self._pending
        if pending is None:
            return
        error: BaseException | None = None
        retired = False
        for _attempt in range(2):
            try:
                pending.abandon()
            except BaseException as exc:
                error = prefer_error(
                    error, exc, "pending-render handoff abandonment retry failed")
                safe_call(
                    log.warning, "pending-render handoff abandonment failed: %r", exc)
            else:
                break
        for _attempt in range(2):
            try:
                if pending._state != "closed":
                    break
                self.clear(pending)
                retired = self._pending is not pending
                break
            except BaseException as exc:
                error = prefer_error(
                    error, exc, "pending-render handoff clear retry failed")
        if error is not None and (
            not retired or not isinstance(error, Exception)
        ):
            raise error


def _throw_if_comfy_interrupted() -> None:
    try:
        import comfy.model_management as mm
    except ImportError:
        return
    if mm.processing_interrupted():
        mm.throw_exception_if_processing_interrupted()


class PendingRender:
    """An eagerly submitted render that owns one setup-generation lease."""

    # RenderPipeline's activation-preflight residency memo stamp
    # (docs/TROUBLESHOOTING.md #47); None outside pipelined submissions.
    _dgxm_unet_name: str | None = None

    def __init__(self, handle: Any, future: Any, progress: Any, topo: Any,
                 latent: dict, timeout: float, render_id: str, finish: Any,
                 session_close: Any = None, telemetry_token: object | None = None):
        self._handle = handle
        self._future = future
        self._progress = progress
        self._topo = topo
        self._latent = latent
        self._timeout = timeout
        self._render_id = render_id
        self._finish = finish
        self._session_close = session_close
        self._telemetry_token = telemetry_token
        self._state = "open"
        self._result_owner: int | None = None
        self._close_abandoned: bool | None = None
        self._lease_retired = False
        self._audit_retired = False
        self._progress_closed = False
        self._telemetry_finished = False
        self._session_closed = session_close is None
        self._state_lock = threading.Lock()

    def _start(self, operation: str) -> None:
        with self._state_lock:
            if self._state != "open":
                raise RuntimeError(
                    f"cannot {operation}: pending render is already {self._state}")
            if operation == "collecting":
                self._result_owner = threading.get_ident()
            self._state = operation

    def _close(self, *, abandoned: bool,
               primary: BaseException | None = None) -> None:
        """Retire every owner; cancellation may replace an ordinary primary."""
        cleanup_error: BaseException | None = None
        active = False
        close_abandoned = abandoned
        session_close = None
        begin_error: BaseException | None = None
        begin_cause: BaseException | None = None
        for _attempt in range(2):
            try:
                active, close_abandoned, session_close = self._begin_close(abandoned)
                break
            except BaseException as exc:
                begin_error, begin_cause = reconcile_error(
                    begin_error, exc, "pending render close publication retry failed")
        if begin_error is not None and (
            not active or not isinstance(begin_error, Exception)
        ):
            cleanup_error = begin_error
        if not active:
            if cleanup_error is not None:
                if primary is None:
                    raise_with_distinct_cause(cleanup_error, begin_cause)
                winner, cause = reconcile_error(
                    primary, cleanup_error, "pending close retry also failed")
                if winner is not primary:
                    raise_with_distinct_cause(winner, cause)
            return
        lease_error: BaseException | None = None
        if not self._lease_retired:
            for _attempt in range(2):
                try:
                    if close_abandoned:
                        mesh_setup.abandon_sample(self._future)
                    else:
                        mesh_setup.release_sample(self._future)
                    self._lease_retired = True
                    break
                except BaseException as exc:
                    lease_error = prefer_error(
                        lease_error, exc, "pending render lease retirement retry failed")
                    safe_call(
                        log.warning, "pending render lease retirement failed: %r", exc)
        if lease_error is not None and (
            not self._lease_retired or not isinstance(lease_error, Exception)
        ):
            cleanup_error = prefer_error(
                cleanup_error, lease_error, "pending render lease retirement failed")
        # Success stamp_result popped the audit facts; all other outcomes retire
        # them here, leaving only in-flight entries (consent_waiver.retire_audit).
        from . import consent_waiver
        audit_error: BaseException | None = None
        if not self._audit_retired:
            for _attempt in range(2):
                try:
                    consent_waiver.retire_audit(self._render_id)
                    self._audit_retired = True
                    break
                except BaseException as exc:
                    audit_error = prefer_error(
                        audit_error, exc, "render audit retirement retry failed")
        if audit_error is not None and (
            not self._audit_retired or not isinstance(audit_error, Exception)
        ):
            cleanup_error = prefer_error(
                cleanup_error, audit_error, "render audit retirement failed")
        progress_error: BaseException | None = None
        if not self._progress_closed:
            for _attempt in range(2):
                try:
                    self._progress.__exit__(None, None, None)
                    self._progress_closed = True
                    break
                except BaseException as exc:
                    progress_error = prefer_error(
                        progress_error, exc, "progress receiver cleanup retry failed")
                    safe_call(log.warning, "progress receiver cleanup failed: %r", exc)
        if progress_error is not None and (
            not self._progress_closed or not isinstance(progress_error, Exception)
        ):
            cleanup_error = prefer_error(
                cleanup_error, progress_error, "progress receiver cleanup failed")
        from ..telemetry import render_progress
        telemetry_error: BaseException | None = None
        if not self._telemetry_finished:
            for _attempt in range(2):
                try:
                    if self._telemetry_token is None:
                        render_progress.finish()
                    else:
                        render_progress.finish(self._telemetry_token)
                    self._telemetry_finished = True
                    break
                except BaseException as exc:
                    telemetry_error = prefer_error(
                        telemetry_error, exc, "render telemetry cleanup retry failed")
                    safe_call(log.warning, "render telemetry cleanup failed: %r", exc)
        if telemetry_error is not None and (
            not self._telemetry_finished or not isinstance(telemetry_error, Exception)
        ):
            cleanup_error = prefer_error(
                cleanup_error, telemetry_error, "render telemetry cleanup failed")
        session_error: BaseException | None = None
        if session_close is not None and not self._session_closed:
            for _attempt in range(2):
                try:
                    session_close()
                    self._session_closed = True
                    break
                except BaseException as exc:
                    session_error = prefer_error(
                        session_error, exc, "render session cleanup retry failed")
                    safe_call(log.warning, "render session cleanup failed: %r", exc)
        if session_error is not None and (
            not self._session_closed or not isinstance(session_error, Exception)
        ):
            cleanup_error = prefer_error(
                cleanup_error, session_error, "render session cleanup failed")
        terminal_error: BaseException | None = None
        terminal_closed = False
        if (self._lease_retired and self._audit_retired and self._progress_closed
                and self._telemetry_finished and self._session_closed):
            for _attempt in range(2):
                try:
                    self._finish_close()
                    terminal_closed = True
                    break
                except BaseException as exc:
                    terminal_error = prefer_error(
                        terminal_error, exc,
                        "pending terminal publication retry failed")
        if terminal_error is not None:
            if not terminal_closed or not isinstance(terminal_error, Exception):
                cleanup_error = prefer_error(
                    cleanup_error, terminal_error,
                    "pending terminal publication failed")
        if cleanup_error is not None:
            if primary is None:
                raise cleanup_error
            winner, cause = reconcile_error(
                primary, cleanup_error, "pending render cleanup also failed")
            if winner is not primary:
                raise_with_distinct_cause(winner, cause)

    def _begin_close(self, abandoned: bool) -> tuple[bool, bool, Any]:
        with self._state_lock:
            if self._state == "closed":
                return False, abandoned, None
            if self._close_abandoned is None:
                self._close_abandoned = abandoned
            # Publish recoverable cleanup intent before touching any resource.
            self._state = "closing"
            return True, bool(self._close_abandoned), self._session_close

    def _finish_close(self) -> None:
        with self._state_lock:
            if self._state == "closed":
                return
            self._session_close = None
            self._state = "closed"

    def _close_failed_result(self, abandoned: bool, primary: BaseException) -> None:
        with self._state_lock:
            state = self._state
            owns_collection = self._result_owner == threading.get_ident()
        if state == "closing" or (state == "collecting" and owns_collection):
            self._close(abandoned=abandoned, primary=primary)

    def _claim_abandon(self) -> bool:
        with self._state_lock:
            if self._state == "closed":
                return False
            if self._state in ("abandoning", "closing"):
                return True
            if self._state == "collecting":
                if self._result_owner != threading.get_ident():
                    return False
            elif self._state != "open":
                return False
            self._state = "abandoning"
            return True

    def cancel(self) -> None:
        self._handle.cancel_sample(self._render_id, wait=False)

    def _cancel_best_effort(self, context: str) -> BaseException | None:
        try:
            self.cancel()
        except BaseException as exc:
            safe_call(
                log.warning, "pending render cancellation %s failed: %r", context, exc)
            return exc
        return None

    def abandon(self) -> None:
        """Terminally discard this result while preserving safe recycle."""
        try:
            self._abandon_once()
        except BaseException as exc:
            try:
                self._abandon_once()
            except BaseException as retry_exc:
                winner, cause = reconcile_error(
                    exc, retry_exc, "pending render abandonment retry failed")
                safe_call(log.error,
                    "pending render abandonment failed twice: %r; retry: %r",
                    exc, retry_exc)
                raise_with_distinct_cause(winner, cause)
            raise

    def _abandon_once(self) -> None:
        """One recoverable abandon attempt; public abandon pairs interruptions."""
        owns_cleanup = False
        claim_error: BaseException | None = None
        cancel_error: BaseException | None = None
        cancel_cause: BaseException | None = None
        close_error: BaseException | None = None
        for _attempt in range(2):
            try:
                owns_cleanup = self._claim_abandon()
                break
            except BaseException as exc:
                claim_error = prefer_error(
                    claim_error, exc, "pending abandon claim retry failed")
        cancel_error = claim_error
        if owns_cleanup:
            for _attempt in range(2):
                try:
                    candidate = self._cancel_best_effort("during abandon")
                    if candidate is None:
                        if (claim_error is None
                                and isinstance(cancel_error, Exception)):
                            cancel_error = None
                        break
                except BaseException as helper_exc:
                    candidate = helper_exc
                previous = cancel_error
                cancel_error, cause = reconcile_error(
                    cancel_error, candidate,
                    "pending abandon cancellation helper retry failed")
                if cancel_error is not previous:
                    cancel_cause = cause
            for _attempt in range(2):
                try:
                    self._close(abandoned=True, primary=cancel_error)
                    break
                except BaseException as cleanup_exc:
                    close_error = prefer_error(
                        close_error, cleanup_exc,
                        "pending abandon close helper retry failed")
        if close_error is not None:
            if cancel_error is None:
                raise close_error
            winner, cause = reconcile_error(
                cancel_error, close_error, "pending abandon close failed")
            if winner is not cancel_error:
                raise_with_distinct_cause(winner, cause)
        if cancel_error is not None and (
            claim_error is not None or not isinstance(cancel_error, Exception)
        ):
            raise_with_distinct_cause(cancel_error, cancel_cause)

    def result(self, timeout_s: float | None = None) -> dict:
        phase = "start"
        try:
            self._start("collecting")
            phase = "collect"
            results = self._handle.collect_sample(
                self._future, activity_fn=getattr(self._progress, "activity", None),
                timeout_s=self._timeout if timeout_s is None else timeout_s)
            phase = "finish"
            _throw_if_comfy_interrupted()
            finished = self._finish(
                results, self._topo, self._latent, self._future)
            phase = "release"
            self._close(abandoned=False)
            return finished
        except BaseException as exc:
            surfaced = exc
            surfaced_cause: BaseException | None = None
            try:
                consent_observe.observe_refusal(exc)
            except BaseException as observe_exc:
                safe_note(exc, "worker refusal observation failed", observe_exc)
            timed_out = phase == "collect" and isinstance(exc, TimeoutError)
            if timed_out:
                cancel_retry = retry_twice(
                    lambda: self._cancel_best_effort("after timeout"),
                    "pending timeout cancellation helper retry failed")
                surfaced, surfaced_cause = cancel_retry.against(
                    exc, "pending render timeout cancellation failed")
            refused = False
            if phase == "collect":
                try:
                    refused = typed_worker_refusal(exc)
                except BaseException as refusal_exc:
                    safe_note(exc, "typed worker refusal inspection failed", refusal_exc)
            close_error: BaseException | None = None
            for _attempt in range(2):
                try:
                    self._close_failed_result(not refused, surfaced)
                    break
                except BaseException as cleanup_exc:
                    close_error = prefer_error(
                        close_error, cleanup_exc,
                        "pending render close helper retry failed")
            if close_error is not None:
                previous = surfaced
                surfaced, cause = reconcile_error(previous, close_error,
                                                   "pending render cleanup failed")
                if surfaced is not previous:
                    surfaced_cause = cause
            if (phase in ("collect", "finish") and not timed_out
                    and isinstance(exc, Exception)):
                supervision_error: BaseException | None = None
                for _attempt in range(2):
                    try:
                        evict_render_fleet(self._handle, exc)
                    except BaseException as candidate:
                        supervision_error = prefer_error(supervision_error, candidate,
                            "render supervision helper retry failed")
                    else:
                        break
                if supervision_error is not None:
                    previous = surfaced
                    surfaced, cause = reconcile_error(
                        previous, supervision_error, "render supervision helper failed")
                    if surfaced is not previous:
                        surfaced_cause = cause
                if phase == "collect":
                    try:
                        _throw_if_comfy_interrupted()
                    except BaseException as interrupt_exc:
                        previous = surfaced
                        surfaced, cause = reconcile_error(previous, interrupt_exc,
                            "post-cleanup render interruption check failed")
                        if surfaced is not previous:
                            surfaced_cause = cause
            if surfaced is not exc:
                raise_with_distinct_cause(surfaced, surfaced_cause)
            raise
