"""Typed policy and lifecycle transaction for resetting the Attached mesh."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

from . import (
    mesh_factory,
    mesh_helpers,
    mesh_liveness,
    mesh_safety,
    mesh_setup,
    mesh_teardown,
)
from .log import get_logger

log = get_logger(__name__)


class RecycleStatus(StrEnum):
    """Stable API result classes for an operator-requested Attached-mesh reset."""

    RECYCLED = "recycled"
    ACTIVE_WORK = "active_work"
    LIFECYCLE_BUSY = "lifecycle_busy"
    PRIOR_TEARDOWN_UNKNOWN = "prior_teardown_unknown"
    PROC_STOP_TIMED_OUT = "proc_stop_timed_out"
    PROC_STOP_FAILED = "proc_stop_failed"
    OVERALL_TIMED_OUT = "overall_timed_out"


@dataclass(frozen=True)
class RecycleOutcome:
    """Structured recycle result for HTTP, node and CLI callers."""

    status: RecycleStatus
    detail: str
    retryable: bool = False
    already_recycled: bool = False

    @property
    def ok(self) -> bool:
        return self.status is RecycleStatus.RECYCLED

    def __bool__(self) -> bool:
        return self.ok

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "status": self.status.value,
            "retryable": self.retryable,
            "detail": self.detail,
        }


def recycle_detailed(
    handle: Any,
    retire_completed: Callable[[], None],
    evict_completed: Callable[[], None],
) -> RecycleOutcome:
    """Run the recycle transaction, bounded at 180 s only on an event-loop thread.

    With no running loop (the HTTP route's executor, cli/cluster_smoke) it runs inline.
    """

    try:
        return cast(
            RecycleOutcome,
            mesh_helpers.run_blocking_off_loop(
                lambda: recycle_detailed_impl(
                    handle, retire_completed, evict_completed),
                180,
                "dgxm-recycle",
            ),
        )
    except TimeoutError:
        log.error("recycle did not finish within 180 s; the proc stop may have hung, "
                  "and the retained pool may still be alive")
        return RecycleOutcome(
            RecycleStatus.OVERALL_TIMED_OUT,
            "recycle did not finish within 180 s; process state is unknown "
            "and the retained pool may still be alive",
        )


def recycle_detailed_impl(
    handle: Any,
    retire_completed: Callable[[], None],
    evict_completed: Callable[[], None],
    lock_timeout_s: float = 10.0,
) -> RecycleOutcome:
    """Stop a fleet once, preserving unknown ownership and busy work."""

    # Do not use shutdown(): it raises on a failed stop instead of returning a
    # RecycleOutcome, and never retires an already-reset handle. shutdown() and
    # this function both stop local procs and leave the hosts untouched while
    # the arena and NCCL port are released.
    from .mesh_session import (
        ConcurrentRenderSessionError,
        require_mutation_authority,
    )

    # Before the lifecycle lock (mesh_liveness.try_release says why). On a
    # release the already-reset branch below answers RECYCLED and retires the
    # handle, so a reset gets the same repair a render gets.
    mesh_liveness.try_release(handle)

    stopped, stop_attempted = False, False  # stopped is set only after .get()
    stop_outcome: RecycleOutcome | None = None
    lock_scope = mesh_setup.lifecycle_lock(handle, "recycle", lock_timeout_s)
    entered = False
    try:
        lock_scope.__enter__()
        entered = True
        mesh_setup.require_no_sample_leases(
            handle, "reset the Attached mesh", allow_abandoned=True)
        require_mutation_authority(handle, "reset the Attached mesh")
        if getattr(handle, "teardown_complete", False):
            # Already stopped (a panel retry after an interrupted end(), or a
            # liveness release): never stop dead procs again; retire the handle.
            retire_completed()
            return RecycleOutcome(
                RecycleStatus.RECYCLED,
                "Attached mesh was already reset; actor processes are stopped",
                already_recycled=True,
            )
        blocked = getattr(handle, "replacement_blocked", None)
        try:
            retrying_blocked = mesh_teardown.prepare_stop_attempt(handle, "recycle")
        except mesh_safety.PriorTeardownUnknownError as exc:
            log.error("recycle remains blocked: %s", exc)
            return RecycleOutcome(
                RecycleStatus.PRIOR_TEARDOWN_UNKNOWN,
                f"{exc}. {mesh_teardown.block_cause_sentence(handle)}",
            )
        if handle.defunct and not retrying_blocked:
            # Defunct without confirmed teardown cannot safely be re-stopped.
            return RecycleOutcome(
                RecycleStatus.PRIOR_TEARDOWN_UNKNOWN,
                "recycle is blocked because this defunct fleet has no confirmed "
                f"teardown outcome. {mesh_teardown.block_cause_sentence(handle)}",
            )

        # begin() stays inside the try so every published token reaches end().
        try:
            mesh_safety.begin_deliberate_teardown(id(handle))
            if retrying_blocked:
                log.info(
                    "retrying explicit ProcMesh cleanup after prior failure: %s",
                    blocked,
                )
            if handle.setup_key is not None:
                # Keep the handle DIRTY until the ProcMesh stop is confirmed.
                handle.setup_cleanup_state = mesh_setup.cleanup_in_progress(
                    getattr(handle, "setup_generation", 0),
                    "recycle group teardown", 60.0)
                handle.setup_key = None
                # The stop below frees every weight, so the workers skip their
                # model unload: comfy copies a loaded model to host RAM before
                # dropping it, needlessly delaying process exit.
                try:
                    handle.workers.teardown_group.call(release_models=False).get(timeout=60)
                except Exception as exc:
                    log.warning("NCCL teardown on recycle did not complete; stopping the procs anyway: %r", exc)
                except BaseException as exc:
                    handle.setup_cleanup_state = mesh_setup.cleanup_failure(
                        getattr(handle, "setup_generation", 0),
                        "recycle group teardown", 60.0, exc)
                    raise
            try:
                stop_attempted = True
                mesh_teardown.enter_stop_boundary(handle, retrying_blocked)
                handle.procs.stop("dgx-monarch recycle").get(timeout=60)
                stopped = True
            except TimeoutError as exc:
                log.error("proc stop on recycle timed out: %r", exc)
                mesh_teardown.publish_stop_failure(
                    handle, mesh_factory.safe_failure_evidence(exc), exc)
                stop_outcome = RecycleOutcome(
                    RecycleStatus.PROC_STOP_TIMED_OUT,
                    "ProcMesh stop timed out; process state is unknown and the "
                    "retained pool may still be alive",
                )
            except Exception as exc:
                # A failed acknowledgement leaves process ownership unknown.
                log.error("proc stop on recycle failed: %r", exc)
                mesh_teardown.publish_stop_failure(
                    handle, mesh_factory.safe_failure_evidence(exc), exc)
                stop_outcome = RecycleOutcome(
                    RecycleStatus.PROC_STOP_FAILED,
                    "ProcMesh stop failed without a confirmed outcome; process "
                    "state is unknown and the retained pool may still be alive",
                )
        finally:
            # Publish under handle.lock before clearing the token; never certify
            # an interrupted handoff.
            mesh_teardown.publish_stop_outcome(
                handle, stopped, stop_attempted,
                "recycle interrupted during the proc stop; process state unknown")
            mesh_safety.end_deliberate_teardown(
                id(handle), stop_confirmed=stopped)
        handle.defunct = True
    except ConcurrentRenderSessionError as exc:
        log.error("recycle refused: %s", exc)
        return RecycleOutcome(
            RecycleStatus.ACTIVE_WORK,
            "recycle is blocked while a render session owns the mesh; wait "
            "for it to finish or cancel cleanly, then retry",
            retryable=True,
        )
    except mesh_setup.SampleResultBusyError as exc:
        log.error("recycle refused: %s", exc)
        return RecycleOutcome(
            RecycleStatus.ACTIVE_WORK,
            "recycle is blocked by active sample/result ownership; wait for "
            "the render or result read to settle, or cancel cleanly, then retry",
            retryable=True,
        )
    except mesh_setup.LifecycleBusyError as exc:
        log.error("recycle refused: %s", exc)
        return RecycleOutcome(
            RecycleStatus.LIFECYCLE_BUSY,
            "recycle is blocked by another mesh lifecycle transition; wait "
            "for that transition to settle, then retry",
            retryable=True,
        )
    finally:
        owned = getattr(handle.lock, "_is_owned", lambda: False)
        if entered or owned():
            lock_scope.__exit__(None, None, None)

    if stopped:
        evict_completed()
        log.info("mesh recycled: actor processes stopped; retained pool returned to the OS")
        return RecycleOutcome(
            RecycleStatus.RECYCLED,
            "Attached mesh reset; actor processes stopped; retained pool returned to the OS",
        )
    if stop_outcome is None:
        raise RuntimeError("recycle ended without a ProcMesh stop outcome")
    return stop_outcome
