"""Retirement, finalization, and the durable owner handoff of one sample lease."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .log import get_logger
from .transfer_utils import prefer_error, raise_with_distinct_cause, reconcile_error
from .transfer_utils import safe_note as _safe_note

log = get_logger("dgx_monarch.mesh_lease")  # SetupBoundFuture's channel: these are its methods


def _safe_error(primary: BaseException, *args: Any) -> None:
    try:
        log.error(*args)
    except BaseException as exc:
        _safe_note(primary, "cleanup logging also failed", exc)


@dataclass(frozen=True)
class _RetirementPlan:
    state: str
    lease_target: int | None
    abandoned_target: int | None
    deferred: BaseException | None
    deferred_deliberate: bool
    clear_deferred: bool
    finalizers: list[Any]


def retire(self, state: str) -> None:
    first_error: BaseException | None = None
    error_cause: BaseException | None = None
    try:
        self._retire_and_drain_once(state)
    except BaseException as exc:
        first_error = exc
        try:
            self._retire_and_drain_once(state)
        except BaseException as retry_exc:
            first_error, error_cause = reconcile_error(
                first_error, retry_exc, "sample retirement retry failed")
            _safe_error(first_error,
                        "sample retirement failed twice: %r; retry: %r",
                        exc, retry_exc)
    if first_error is not None:
        raise_with_distinct_cause(first_error, error_cause)


def retire_once(self, state: str) -> None:
    with self.handle.lock:
        if self.state != "active" or self.retire_to is not None:
            return
        if self._reader_tokens:
            self.retire_to = state
            return
        self._finalize_locked(state)


def finalize_locked(self, state: str) -> None:
    applied_plan: _RetirementPlan | None = None
    try:
        if self._retirement_plan is None:
            self._retirement_plan = self._build_retirement_plan_locked(state)
        applied_plan = self._retirement_plan
        self._apply_retirement_plan_locked()
    except BaseException as primary:
        try:
            if self._retirement_plan is None:
                if applied_plan is None:
                    self._retirement_plan = self._build_retirement_plan_locked(state)
                else:
                    raise
            self._apply_retirement_plan_locked()
        except BaseException as retry:
            winner, cause = reconcile_error(primary, retry, "retirement retry failed")
            raise_with_distinct_cause(winner, cause)
        raise


def build_retirement_plan_locked(self, state: str) -> _RetirementPlan:
    leases = self.handle.sample_leases
    lease_target = None
    if self._registered:
        count = int(leases.get(self.generation, 0))
        lease_target = max(0, count - 1)
    remaining = any(
        (
            lease_target
            if generation == self.generation and lease_target is not None
            else int(count)
        ) > 0
        for generation, count in leases.items()
    )
    abandoned_target = None
    if state == "abandoned" and self._enqueue_started:
        abandoned = self.handle.abandoned_sample_leases
        abandoned_target = int(abandoned.get(self.generation, 0)) + 1
    deferred = (
        getattr(self.handle, "deferred_supervision_error", None)
        if not remaining else None
    )
    return _RetirementPlan(
        state=state,
        lease_target=lease_target,
        abandoned_target=abandoned_target,
        deferred=deferred,
        deferred_deliberate=deferred is not None and bool(
            getattr(self.handle, "_deferred_eviction_deliberate", False)),
        clear_deferred=not remaining,
        finalizers=list(self.finalizers),
    )


def apply_retirement_plan_locked(self) -> None:
    plan = self._retirement_plan
    if plan is None:
        return
    if plan.lease_target is not None:
        if plan.lease_target == 0:
            self.handle.sample_leases.pop(self.generation, None)
        else:
            self.handle.sample_leases[self.generation] = plan.lease_target
    if plan.abandoned_target is not None:
        self.handle.abandoned_sample_leases[self.generation] = plan.abandoned_target
    self.state = plan.state
    self.retire_to = None
    self._pending_completion = (plan.deferred, plan.finalizers)
    self._pending_deliberate = plan.deferred_deliberate
    self.finalizers = []
    if plan.clear_deferred:
        self.handle.deferred_supervision_error = None
        # Cleared only when it was set. The route flag is a dynamic attribute
        # (mesh_helpers), so an unconditional write would raise on a slotted
        # handle double while this holds the lock.
        if getattr(self.handle, "_deferred_eviction_deliberate", False):
            self.handle._deferred_eviction_deliberate = False
    self._registered = False
    self._retirement_plan = None


def drain_finalization(self) -> None:
    """Complete one durable terminal handoff, retrying one interruption."""
    first_error: BaseException | None = None
    for _attempt in range(2):
        with self.handle.lock:
            self._apply_retirement_plan_locked()
            completion = self._pending_completion
        if completion is None:
            if first_error is not None:
                raise first_error
            return
        try:
            self._complete_finalization(*completion)
            with self.handle.lock:
                if self._pending_completion is completion:
                    self._pending_completion = None
        except BaseException as exc:
            if first_error is None:
                first_error = exc
                continue
            prior = first_error
            first_error, cause = reconcile_error(
                first_error, exc, "sample lease completion retry failed")
            _safe_error(first_error,
                        "sample lease completion failed twice: %r; retry: %r",
                        prior, exc)
            raise_with_distinct_cause(first_error, cause)
        if first_error is not None:
            raise first_error
        return


def complete_finalization(self, deferred: BaseException | None,
                          finalizers: list[Any]) -> None:
    strongest: BaseException | None = None
    try:
        self._resume_deferred_supervision(deferred)
    except BaseException as exc:
        strongest = exc
    try:
        self._run_finalizers(finalizers)
    except BaseException as exc:
        strongest = prefer_error(strongest, exc, "sample finalizer cleanup failed")
    if strongest is not None:
        raise strongest


def run_finalizers(finalizers: list[Any]) -> None:
    strongest: BaseException | None = None
    for callback in finalizers:
        for attempt in range(2):
            try:
                callback()
            except BaseException as exc:
                strongest = prefer_error(
                    strongest, exc, "sample lease finalizer failed")
                if attempt == 0:
                    continue
                _safe_error(
                    strongest, "sample lease finalizer failed twice: %r; retry: %r",
                    strongest, exc)
            break
    if strongest is not None:
        raise strongest


def resume_deferred_supervision(self, exc: BaseException | None) -> None:
    if exc is None:
        return
    from .mesh_helpers import (
        mark_defunct_deliberate,
        mark_defunct_on_supervision_failure,
    )

    # A deliberate eviction must resume through the route that made it. The
    # supervision classifier reaches no verdict on a driver-side eviction, so
    # sending one there leaves the handle defunct with no terminal outcome:
    # every later attach and every recycle refuses until ComfyUI restarts.
    if self._pending_deliberate:
        mark_defunct_deliberate(self.handle, exc)
        return
    mark_defunct_on_supervision_failure(self.handle, exc)
