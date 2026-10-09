"""Driver for one process-rooted exact-once RDMA read job."""
from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from typing import Any

from . import rdma_job_registry as _registry
from .rdma_read_job import LatentReadJob, _Claim
from .transfer_utils import raise_with_distinct_cause


def _started(runner: threading.Thread, job: LatentReadJob, claim: _Claim) -> bool:
    return runner.ident is not None or job.owns_claim(claim)


def _join_or_timeout(
    runner: threading.Thread, timeout_s: float, thread_name: str
) -> None:
    runner.join(timeout=timeout_s)
    if runner.is_alive():
        raise TimeoutError(f"{thread_name} did not complete within {timeout_s:.0f}s")


def _merge_job_error(job: LatentReadJob, primary: BaseException) -> BaseException:
    for runner_error in job.runner_errors:
        primary = _registry.prefer(
            primary, runner_error, "RDMA scratch contender also failed")
    # outcome_owner is the durable publication.  Delivery into holder may be
    # interrupted or blocked after that outcome has already gained a stronger
    # cancellation, so it must be consulted independently of the caller copy.
    outcome = (job.outcome_owner[0] if job.outcome_owner else
               job.holder[0] if job.holder else None)
    if outcome is None:
        return primary
    error = outcome[1]
    if error is None or error is primary:
        return primary
    return _registry.prefer(primary, error, "RDMA scratch settlement also failed")


def run_owned_off_loop(
    guard: Any,
    operation: Callable[
        [BaseException | None, Callable[[], None] | None,
         Callable[[], None], Callable[..., None]], Any
    ],
    timeout_s: float | Callable[[], float],
    thread_name: str,
) -> Any:
    """Root, prepare, and drive one exact-once descriptor job."""
    holder: list[tuple[Any, BaseException | None]] = []
    construction_owner: list[LatentReadJob] = []
    construction_error: BaseException | None = None
    try:
        job = LatentReadJob(guard, operation, holder, construction_owner)
    except BaseException as error:
        construction_error = error
        if construction_owner:
            job = construction_owner[0]
        else:
            job = LatentReadJob(guard, operation, holder, construction_owner)
    try:
        original, runner, attach_error = job.new_runner(
            thread_name, construction_error)
    except BaseException as runner_error:
        try:
            original, runner, attach_error = job.new_runner(
                thread_name, runner_error)
        except BaseException as retry_error:
            strongest = _registry.prefer(
                runner_error, retry_error,
                "scratch runner construction retry failed")
            if strongest is retry_error:
                raise_with_distinct_cause(retry_error, runner_error)
            raise_with_distinct_cause(runner_error, retry_error)
        construction_error = _registry.prefer(
            construction_error, runner_error, "scratch runner construction failed")
    if attach_error is not None:
        original.remember(attach_error, "scratch runner attachment failed")
        construction_error = _registry.prefer(
            construction_error, attach_error, "scratch runner attachment failed")
    outer_primary, on_loop = construction_error, True
    try:
        try:
            root_error = _registry.register(job)
        except BaseException as registration_error:
            if not _registry.registered(job):
                raise
            root_error = registration_error
        if root_error is not None:
            original.remember(root_error, "scratch job root publication failed")
            outer_primary = _registry.prefer(
                outer_primary, root_error, "scratch job root publication failed")

        try:
            timeout_value = timeout_s() if callable(timeout_s) else timeout_s
        except BaseException as timeout_error:
            timeout_value = 0.0
            original.remember(timeout_error, "scratch timeout preparation failed")
            outer_primary = _registry.prefer(
                outer_primary, timeout_error, "scratch timeout preparation failed")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop = False
        except BaseException as loop_error:
            on_loop = True
            original.remember(loop_error, "event-loop detection failed")
            outer_primary = _registry.prefer(
                outer_primary, loop_error, "event-loop detection failed")
        else:
            on_loop = True

        try:
            confirmed, preparation_error = job.prepare()
        except BaseException as preparation_boundary:
            confirmed, preparation_error = job.reconcile_prepare(
                preparation_boundary)
        if not confirmed:
            failure = preparation_error if preparation_error is not None else RuntimeError(
                "sample read token publication could not be confirmed")
            raise failure
        if preparation_error is not None:
            original.remember(preparation_error, "read-token preparation failed")
            outer_primary = _registry.prefer(
                outer_primary, preparation_error, "read-token preparation failed")

        if not on_loop and original.primary is None:
            job.run_thread(original)
        else:
            try:
                runner.start()
            except BaseException as start_error:
                applied = _started(runner, job, original)
                outer_primary = _registry.prefer(
                    outer_primary, start_error, "scratch runner start failed")
                recovery_primary = original.primary if applied else outer_primary
                recovery, recovery_runner, attach_error = job.new_runner(
                    thread_name, recovery_primary)
                if attach_error is not None:
                    outer_primary = _registry.prefer(
                        outer_primary, attach_error,
                        "scratch recovery runner attachment failed")
                try:
                    recovery_runner.start()
                except BaseException as recovery_error:
                    prior_primary = outer_primary
                    outer_primary = _registry.prefer(
                        outer_primary, recovery_error,
                        "RDMA scratch recovery start also failed")
                    if _started(recovery_runner, job, recovery):
                        runner = recovery_runner
                    elif not _started(runner, job, original):
                        if outer_primary is recovery_error:
                            raise_with_distinct_cause(
                                recovery_error, prior_primary)
                        raise_with_distinct_cause(
                            outer_primary, recovery_error)
                else:
                    runner = recovery_runner
            if runner.ident is not None:
                try:
                    _join_or_timeout(runner, float(timeout_value), thread_name)
                except BaseException as join_error:
                    if outer_primary is not None:
                        prior_primary = outer_primary
                        outer_primary = _registry.prefer(
                            outer_primary, join_error, "RDMA scratch join also failed")
                        if outer_primary is join_error:
                            raise_with_distinct_cause(join_error, prior_primary)
                        raise_with_distinct_cause(outer_primary, join_error)
                    raise

        if outer_primary is not None:
            raise _merge_job_error(job, outer_primary)
        return job.result_from_holder()
    except BaseException as caught:
        primary = caught
        if outer_primary is not None and primary is not outer_primary:
            primary = _registry.prefer(
                outer_primary, primary, "rooted driver envelope also failed")
        # A root-publication interruption can occur after the registry append
        # but before preparation begins. With no token, no tokenless prepare,
        # and no started runner, the operation cannot have native effects, so
        # unregistering this exact root is safe. Keep later PENDING states
        # rooted: tokenless preparation and failed thread starts already own
        # descriptor settlement that must never run on the event-loop caller.
        if (
            job.state == "PENDING"
            and not job.token_owner
            and not job.token_retired
            and not any(candidate.ident is not None for candidate in job.runners)
        ):
            try:
                unroot_error = _registry.unregister(job)
            except BaseException as cleanup_error:
                primary = _registry.prefer(
                    primary, cleanup_error,
                    "unprepared scratch root cleanup failed")
            else:
                if unroot_error is not None:
                    primary = _registry.prefer(
                        primary, unroot_error,
                        "unprepared scratch root cleanup was interrupted")
        # Only a proven off-loop caller may drive PENDING recovery inline.
        # On-loop failure retains the root/token instead of running native
        # settlement on the caller; the preconstructed recovery claim still
        # prevents any retry from publishing a second synchronous contender.
        for _attempt in range(2):
            try:
                if not _registry.registered(job):
                    break
                if job.state == "PENDING" and not on_loop:
                    job.sync_recovery_claim.remember(
                        primary, "rooted ownership envelope failed")
                    confirmed, reconciliation_error = job.reconcile_prepare(primary)
                    if confirmed:
                        job.run_thread(job.sync_recovery_claim)
                    elif reconciliation_error is not None:
                        primary = _registry.prefer(
                            primary, reconciliation_error, "rooted recovery failed")
                if (job.outcome_owner
                        and not any(candidate.is_alive()
                                    for candidate in job.runners)):
                    job._deliver_outcome()
                break
            except BaseException as recovery_error:
                primary = _registry.prefer(
                    primary, recovery_error, "rooted recovery failed")
        strongest = _merge_job_error(job, primary)
        if strongest is not caught:
            raise strongest from caught
        raise
