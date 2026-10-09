"""Interruption-safe ownership guard for eager render submission."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .. import mesh_setup
from ..log import get_logger
from ..transfer_utils import (
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_call,
    safe_note,
)
from .pending import PendingRender
from .render_session import RenderSession

log = get_logger(__name__)
T = TypeVar("T")


class SubmitRenderGuard:
    """Caller-owned authority for every pre-PendingRender resource."""

    def __init__(self) -> None:
        # The candidate exists before bind, so an exception between owner
        # publication and claim_render_session's return still leaves cleanup
        # authority here. Every other field is published before its resource
        # can acquire a remote or background side effect.
        self.candidate = RenderSession()
        self.handle: Any = None
        self.future: Any = None
        self.progress: Any = None
        self.pending: PendingRender | None = None
        # The dispatch takes its waiver-audit slot when the request is stamped,
        # before PendingRender exists. The guard is the only owner that can
        # free the slot in that window, so the id is published here.
        self.render_id: str | None = None
        self.telemetry_token = object()
        self.telemetry_started = False
        self.dispatch_started = False
        self.transferred = False
        self._pending_cleaned = False
        self._audit_cleaned = False
        self._future_cleaned = False
        self._progress_cleaned = False
        self._telemetry_cleaned = False
        self._candidate_cleaned = False

    def _retire_waiver_audit(self) -> None:
        """Free the audit slot the stamp took before PendingRender existed.

        A transferred submit never runs this: PendingRender._close owns every
        retirement from publication onward, and stamp_result pops the slot on
        the one path that writes its permanent row first.
        """
        from . import consent_waiver

        consent_waiver.retire_audit(self.render_id)

    def cleanup(self, primary: BaseException | None) -> None:
        """Idempotently retire an untransferred submit and close the candidate."""
        failures: list[tuple[str, BaseException]] = []

        def attempt(label: str, callback, completed_attr: str) -> None:
            if getattr(self, completed_attr):
                return
            try:
                callback()
            except BaseException as exc:
                failures.append((label, exc))
            else:
                setattr(self, completed_attr, True)

        if not self.transferred:
            if self.pending is not None:
                attempt(
                    "pending render abandon",
                    self.pending.abandon,
                    "_pending_cleaned",
                )
            else:
                if self.future is not None:
                    attempt(
                        "sample lease abandon",
                        lambda: mesh_setup.abandon_sample(self.future),
                        "_future_cleaned",
                    )
                if self.render_id is not None:
                    attempt(
                        "waiver audit retirement",
                        self._retire_waiver_audit,
                        "_audit_cleaned",
                    )
                if self.progress is not None:
                    attempt(
                        "progress receiver close",
                        lambda: self.progress.__exit__(None, None, None),
                        "_progress_cleaned",
                    )
                if self.telemetry_started:
                    from ..telemetry import render_progress

                    attempt(
                        "render telemetry finish",
                        lambda: render_progress.finish(self.telemetry_token),
                        "_telemetry_cleaned",
                    )
        # The submit closes the candidate itself just before it publishes the
        # PendingRender (render_submit), so this close runs only when that one
        # did not complete. Under an inherited Pipeline or Gate session the
        # candidate was never bound.
        attempt(
            "render session close",
            self.candidate.close,
            "_candidate_cleaned",
        )

        if not failures:
            return
        winner: BaseException | None = None
        for label, exc in failures:
            winner = prefer_error(winner, exc, f"{label} also failed")
            safe_call(log.error, "%s failed during submit cleanup: %r", label, exc)
        if winner is None:  # failures is nonempty; keep the type proof runtime-safe
            raise RuntimeError("submit cleanup failed without a recorded exception")
        if primary is None:
            raise winner
        surfaced, cause = reconcile_error(
            primary, winner, "render submission cleanup failed")
        if surfaced is not primary:
            raise_with_distinct_cause(surfaced, cause)


def run_guarded_submit(
    guard: SubmitRenderGuard,
    body: Callable[[], T],
    on_failure: Callable[[BaseException], None],
) -> T:
    """Run submission, retiring every owner before cancellation is surfaced."""
    primary: BaseException | None = None
    try:
        try:
            return body()
        except BaseException as exc:
            primary = exc
            try:
                on_failure(exc)
            except BaseException as failure_exc:
                safe_note(
                    primary, "render submission failure handling failed", failure_exc)
            raise
        finally:
            try:
                guard.cleanup(primary)
            except BaseException as cleanup_exc:
                if primary is None:
                    primary = cleanup_exc
                    raise
                previous = primary
                primary, cause = reconcile_error(
                    previous, cleanup_exc, "render submission cleanup failed")
                safe_call(
                    log.error,
                    "render submission cleanup failed while preserving %r: %r",
                    primary, cleanup_exc)
                if primary is not previous:
                    raise_with_distinct_cause(primary, cause)
    finally:
        try:
            guard.cleanup(primary)
        except BaseException as cleanup_exc:
            if primary is None:
                raise
            previous = primary
            primary, cause = reconcile_error(
                previous, cleanup_exc, "render submission cleanup retry failed")
            safe_call(
                log.error,
                "render submission cleanup retry failed while preserving %r: %r",
                primary, cleanup_exc)
            if primary is not previous:
                raise_with_distinct_cause(primary, cause)
