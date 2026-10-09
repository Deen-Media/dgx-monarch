"""Bounded cross-render pipeline with terminal result ownership."""
from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

from ..log import get_logger
from ..retry_policy import retry_twice
from ..transfer_utils import (
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_call,
)
from .pending import PendingRender, PendingRenderHandoff
from .render_session import RenderSession

log = get_logger(__name__)

_PIPELINE_ABORT_TIMEOUT_S = float(os.environ.get("DGXM_PIPELINE_ABORT_TIMEOUT", "30"))


def _retry_cleanup_preserving_primary(
    primary: BaseException, label: str, callback: Any
) -> None:
    def report_failure(cleanup_exc: BaseException) -> None:
        safe_call(log.error, "%s failed while preserving %r: %r",
                  label, primary, cleanup_exc)

    def attempt() -> BaseException | None:
        callback()
        return None

    outcome = retry_twice(
        attempt, f"{label} retry also failed",
        on_failure=report_failure)
    winner, cause = outcome.against(primary, f"{label} also failed")
    if winner is not None and winner is not primary:
        raise_with_distinct_cause(winner, cause)


def _close_pipeline_session_retry(session: RenderSession) -> None:
    error: BaseException | None = None
    for _attempt in range(2):
        try:
            session.close()
        except BaseException as exc:
            error = prefer_error(error, exc, "pipeline session close retry failed")
        else:
            break
    if error is not None:
        raise error


def _cancel_and_drain_inflight(inflight: deque) -> None:
    """Best-effort bounded cancellation before unconditional abandonment."""
    cancellation: BaseException | None = None
    for pending in inflight:
        try:
            cancel = getattr(pending, "cancel", None)
            if cancel is not None:
                cancel()
        except BaseException as exc:
            if not isinstance(exc, Exception):
                cancellation = prefer_error(
                    cancellation, exc, "additional pipeline cancellation failed")
    deadline = time.monotonic() + _PIPELINE_ABORT_TIMEOUT_S
    index = 0
    while index < len(inflight):
        pending = inflight[index]
        terminal = False
        try:
            if isinstance(pending, PendingRender):
                pending.result(timeout_s=max(0.05, deadline - time.monotonic()))
            else:
                pending.result()
        except BaseException as exc:
            if not isinstance(exc, Exception):
                cancellation = prefer_error(
                    cancellation, exc, "pipeline drain was interrupted")
            terminal = getattr(pending, "_state", None) == "closed"
        else:
            terminal = True
        if terminal:
            del inflight[index]
        else:
            index += 1
        if time.monotonic() >= deadline:
            safe_call(log.error,
                "pipeline abort deadline reached; stopped waiting, and cleanup abandons "
                "the cancelled renders still in flight")
            break
    if cancellation is not None:
        raise cancellation


def _abandon_all_inflight(inflight: deque) -> None:
    """Idempotently retire every reachable pending before deque publication."""
    first_error: BaseException | None = None
    index = 0
    while index < len(inflight):
        pending = inflight[index]
        abandon = getattr(pending, "abandon", None)
        if abandon is None:
            if first_error is None:
                first_error = RuntimeError(
                    "pipeline pending has no terminal abandon operation")
            index += 1
            continue
        retired = False
        attempt_error: BaseException | None = None
        for _attempt in range(2):
            try:
                abandon()
            except BaseException as exc:
                attempt_error = prefer_error(
                    attempt_error, exc, "pipeline abort abandonment retry failed")
                safe_call(log.error,
                    "pipeline abort abandonment attempt failed: %r", exc)
            else:
                retired = True
                break
        if retired:
            # Remove only a terminal entry. If deque deletion is interrupted,
            # an outer abort retry safely repeats idempotent abandonment.
            del inflight[index]
            if attempt_error is not None and not isinstance(attempt_error, Exception):
                first_error = prefer_error(
                    first_error, attempt_error,
                    "pipeline abandonment was interrupted before retry success")
            continue
        if first_error is None:
            first_error = attempt_error
        elif attempt_error is not None:
            first_error = prefer_error(
                first_error, attempt_error,
                "additional pipeline abandonment failed")
        index += 1
    if first_error is not None:
        raise first_error


def _abort_cleanup(pipeline: RenderPipeline, primary: BaseException | None) -> None:
    """Terminal cleanup phase, safe to invoke twice around async interruption."""
    cleanup_error: BaseException | None = None
    try:
        _abandon_all_inflight(pipeline._inflight)
    except BaseException as exc:
        cleanup_error = exc
    try:
        _close_pipeline_session(pipeline._session, primary)
    except BaseException as exc:
        cleanup_error = prefer_error(
            cleanup_error, exc, "pipeline session cleanup also failed")
    if cleanup_error is None:
        return
    if primary is not None:
        winner, cause = reconcile_error(
            primary, cleanup_error, "pipeline abort cleanup also failed")
        if winner is not primary:
            raise_with_distinct_cause(winner, cause)
        return
    raise cleanup_error


class RenderPipeline:
    """Submit at most ``depth`` renders and collect them in FIFO order.

    A submission or collection failure cancels and terminally abandons every
    remaining result so no setup-generation lease is silently discarded.
    """

    def __init__(self, depth: int = 1, on_step=None):
        self.depth = max(1, int(depth))
        self._on_step = on_step
        # Both instance lifetime; each render moves from `_inflight` to `_out`.
        # `_collect_oldest` appends to `_out` before it pops `_inflight`, so both
        # briefly hold the entry and there is no window where neither owns it.
        # Only `_inflight` carries an abandonment contract, and `_abort` drains
        # only `_inflight`.
        self._inflight: deque = deque()
        self._out: list = []
        self._seq = 0
        self._session = RenderSession()

    def _abort(self) -> None:
        primary: BaseException | None = None
        error_cause: BaseException | None = None
        try:
            _cancel_and_drain_inflight(self._inflight)
        except BaseException as exc:
            primary = exc
        for _attempt in range(2):
            try:
                _abort_cleanup(self, primary)
            except BaseException as cleanup_exc:
                previous = primary
                primary, cause = reconcile_error(
                    primary, cleanup_exc, "pipeline abort cleanup retry failed")
                if primary is not previous:
                    error_cause = cause
            else:
                break
        if primary is not None:
            raise_with_distinct_cause(primary, error_cause)

    def _collect_oldest(self) -> None:
        pending = None
        try:
            # Peek first: ownership stays in the deque until result publication
            # succeeds. If popleft itself is interrupted after mutation, the
            # local reference still lets this handler terminally clean it.
            pending = self._inflight[0]
            self._out.append(pending.result())
            self._inflight.popleft()
            # Mirror run_render's residency memo so pipelined warm re-submits
            # are not charged again for resident weights. Import lazily like
            # _push_bound; an entry without the attribute clears the memo.
            from . import render_preflight
            render_preflight.note_successful_render(
                getattr(pending, "_dgxm_unet_name", None))
        except BaseException as primary:
            cleanup_error: BaseException | None = None
            if pending is not None:
                abandon = getattr(pending, "abandon", None)
                abandoned = abandon is None
                if abandon is not None:
                    try:
                        abandon()
                        abandoned = True
                    except BaseException as exc:
                        cleanup_error = prefer_error(
                            cleanup_error, exc, "pipeline pending abandon failed")
                        safe_call(log.warning, "pipeline pending abandon failed: %r", exc)
                        try:
                            abandon()
                            abandoned = True
                        except BaseException as retry_exc:
                            cleanup_error = prefer_error(
                                cleanup_error, retry_exc,
                                "pipeline pending abandon retry failed")
                            safe_call(log.error,
                                "pipeline pending abandon failed twice: %r; retry: %r",
                                exc, retry_exc)
                try:
                    if ((abandoned or getattr(pending, "_state", None) == "closed")
                            and self._inflight and self._inflight[0] is pending):
                        self._inflight.popleft()
                except BaseException as exc:
                    # Local ownership already survives even if deque mutation is
                    # ambiguous; _abort idempotently handles a retained entry.
                    cleanup_error = prefer_error(
                        cleanup_error, exc, "pipeline pending dequeue cleanup failed")
                    safe_call(
                        log.warning, "pipeline pending dequeue cleanup failed: %r", exc)
            try:
                _retry_cleanup_preserving_primary(
                    primary, "pipeline abort", self._abort)
            except BaseException as abort_exc:
                cleanup_error = prefer_error(
                    cleanup_error, abort_exc, "pipeline abort also failed")
            if cleanup_error is not None:
                winner, cause = reconcile_error(
                    primary, cleanup_error, "pipeline collection cleanup failed")
                if winner is not primary:
                    raise_with_distinct_cause(winner, cause)
            raise

    def push(self, model: Any, request: dict, latent: dict,
             cfg_value: float | None, steps_hint: int) -> None:
        try:
            self._push_bound(model, request, latent, cfg_value, steps_hint)
        except BaseException as primary:
            # The try opens before the sequence number moves or the handoff
            # exists, so an interruption at either point still retires every
            # earlier in-flight render.
            _retry_cleanup_preserving_primary(primary, "pipeline abort", self._abort)
            raise

    def _push_bound(self, model: Any, request: dict, latent: dict,
                    cfg_value: float | None, steps_hint: int) -> None:
        # Import lazily so common can re-export this compatibility surface
        # without a module-import cycle.
        from .. import adoption_evidence
        from . import common, consent_waiver, gate_inconclusive, render_preflight

        consent_waiver.validate_inherited_stamps(latent)
        adoption_evidence.require_inactive_context("RenderPipeline")

        # Family-scoped capacity refusal (docs/TROUBLESHOOTING.md #47), placed
        # as in run_render: before any mesh/gate/session side effect. Without
        # it a pipelined SCAIL submission skips the activation preflight.
        render_preflight.activation_footprint_preflight_for_request(
            model, request, latent)

        model, _bound_handle = common._bind_packed_render_model(
            model, latent, cfg_value)
        # Before the risk read, not after it: a quarantine another
        # combination's abort wrote reads as no risk here, so a combination
        # the ceremony never saw would lose its ceremony too.
        common._restore_requested_levers(model)
        if model is not None and common.auto_gate_required(
                model, request.get("kind", ""), latent, cfg_value):
            while self._inflight:
                self._collect_oldest()
            # The identity ceremony runs its own direct render. Yield this
            # pipeline's handle only after every earlier result was consumed,
            # then bind it again for the user's next submission.
            _close_pipeline_session_retry(self._session)
            gate_result = common._maybe_auto_gate(
                model, request, latent, cfg_value, steps_hint)
            if gate_result not in gate_inconclusive.NO_QUARANTINE_VERDICTS:
                common._quarantine_unproven_paths(model)
        while len(self._inflight) >= self.depth:
            self._collect_oldest()
        seq = self._seq
        self._seq += 1
        handoff = PendingRenderHandoff()
        try:
            with self._session.activate():
                pending = common.submit_render(
                    common.model_for_request(model, request), request, latent,
                    cfg_value, steps_hint, seq=seq, depth=self.depth,
                    progress_on_step=self._on_step, handoff=handoff)
            # Carried to _collect_oldest for the residency memo: the memo
            # records only completed renders, so it is stamped at result
            # publication, not at submission.
            try:
                pending._dgxm_unet_name = getattr(model, "unet_name", None)
            except (AttributeError, TypeError):
                pass  # slotted/frozen test doubles: the memo stays unset
            # The in-flight memo, stamped now: a follow-up push of the same
            # checkpoint must not be charged its weight bytes again while this
            # submission is still loading or uncollected.
            render_preflight.note_submitted_render(getattr(model, "unet_name", None))
            self._inflight.append(pending)
            handoff.clear(pending)
        except BaseException as primary:
            cleanup_error: BaseException | None = None
            for label, callback in (
                ("pipeline submission handoff cleanup", handoff.abort),
                ("pipeline abort", self._abort),
            ):
                try:
                    _retry_cleanup_preserving_primary(primary, label, callback)
                except BaseException as cleanup_exc:
                    cleanup_error = prefer_error(
                        cleanup_error, cleanup_exc, f"{label} also failed")
            if cleanup_error is not None:
                winner, cause = reconcile_error(
                    primary, cleanup_error, "pipeline submission cleanup failed")
                if winner is not primary:
                    raise_with_distinct_cause(winner, cause)
            raise

    def drain(self) -> list:
        primary: BaseException | None = None
        try:
            while self._inflight:
                self._collect_oldest()
            return self._out
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            for _attempt in range(2):
                try:
                    _close_pipeline_session(self._session, primary)
                except BaseException as cleanup_exc:
                    cleanup_error = prefer_error(
                        cleanup_error, cleanup_exc,
                        "pipeline session cleanup retry failed")
                else:
                    break
            if cleanup_error is not None:
                if primary is None:
                    raise cleanup_error
                winner, cause = reconcile_error(
                    primary, cleanup_error, "pipeline session cleanup failed")
                if winner is not primary:
                    raise_with_distinct_cause(winner, cause)


def _close_pipeline_session(
    session: RenderSession, primary: BaseException | None
) -> None:
    try:
        _close_pipeline_session_retry(session)
    except BaseException as cleanup_exc:
        if primary is None:
            raise
        winner, cause = reconcile_error(
            primary, cleanup_exc, "pipeline session close failed")
        if winner is not primary:
            raise_with_distinct_cause(winner, cause)
