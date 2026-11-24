"""Stop-attempt authority and supervision reconciliation for mesh lifecycles."""
from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from . import driver_exit, mesh_safety, reconcile_running
from .error_utils import failure_summary, safe_note
from .log import get_logger
from .mesh_runtime import mesh_creation_timeout_s
from .mesh_setup_state import LifecycleBusyError, SampleResultBusyError
from .mesh_stop import ACTOR_STOP_S as ACTOR_STOP_S  # re-exported for the teardown callers
from .mesh_stop import PROC_STOP_DEFAULT_S as PROC_STOP_DEFAULT_S
from .mesh_stop import RECONCILE_GROUP_S, proc_stop_bound_s, stop_with_reason
from .mesh_stop import reconcile_join_s as reconcile_join_s
from .mesh_stop import shutdown_envelope_s as shutdown_envelope_s
from .mesh_stop import shutdown_phases as shutdown_phases
from .reconcile_running import caller_waits as caller_waits
from .reconcile_running import cleanup_running as cleanup_running  # re-exported for the refusals' callers
from .reconcile_running import cleanup_running_sentence as cleanup_running_sentence
from .reconcile_running import unresolved_refusal as unresolved_refusal
from .stop_causes import TAG_CREATION_CLEANUP_UNCONFIRMED as TAG_CREATION_CLEANUP_UNCONFIRMED  # re-exported causes
from .stop_causes import TAG_NO_STOP_ISSUED as TAG_NO_STOP_ISSUED
from .stop_causes import TAG_SETUP_ROLLBACK_INTERRUPTED as TAG_SETUP_ROLLBACK_INTERRUPTED
from .stop_causes import TAG_STOP_INTERRUPTED as TAG_STOP_INTERRUPTED
from .stop_causes import TAG_STOP_REPORTED_FAILURE as TAG_STOP_REPORTED_FAILURE
from .stop_causes import TAG_STOP_TIMEOUT_AFTER_DEATH as TAG_STOP_TIMEOUT_AFTER_DEATH
from .stop_causes import TAG_STOP_TIMEOUT_DELIBERATE as TAG_STOP_TIMEOUT_DELIBERATE
from .stop_causes import block_cause as block_cause
from .stop_causes import label_for_cause as label_for_cause
from .stop_causes import latch_no_stop_locked
from .stop_causes import sentence_for_cause as sentence_for_cause

log = get_logger("dgx_monarch.mesh")

# A liveness proof older than this is not a proof. The retire re-reads it under
# the handle lock, so a gone recorded while a live lease blocked the release can
# never authorize one minutes later.
EVIDENCE_MAX_AGE_S = 30.0

GET_MESH_DOOR = "get_mesh"
ENSURE_LIVE_DOOR = "ensure_live"

_LIVENESS_RELEASE_NOTE = (
    " (the fleet this one replaces was retired after the driver proved its "
    "worker processes were gone)")
# The instant of the last release, or None. The next creation block consumes
# it: that bring-up is the one the release authorized, and its poison text is
# where the operator needs to see it. The replacement's attach clears it when
# it succeeds (mesh_attach._attach_and_retry), so a later attach keeps its
# retry and a later, unrelated failure carries no release note. Time bounded as
# well, for a release whose replacement never reaches an attach.
_RELEASED_PREDECESSOR: float | None = None
# The released fleet's host addresses; the attach fail-fast applies only to them.
_RELEASED_ADDRESSES: frozenset[str] | None = None


def _release_note_window_s() -> float:
    """How long the note stays claimable: the live creation timeout.

    Read at use, not at import. An operator who raised the creation timeout
    raised how long the bring-up this release authorized may take, and the note
    must outlast that bring-up rather than the shipped default.
    """
    return mesh_creation_timeout_s()


def block_cause_sentence(handle: Any) -> str:
    return sentence_for_cause(block_cause(handle))


def blocked_refusal(handle: Any, door: str) -> str:
    """Format a lifecycle refusal with the recovery action for its cause."""
    if cleanup_running(handle):
        return f"the cached mesh cannot be used yet: {cleanup_running_sentence()}"
    sentence = block_cause_sentence(handle)
    if door == GET_MESH_DOOR:
        blocked = getattr(handle, "replacement_blocked", None)
        return ("the previous worker fleet could not be stopped safely; "
                f"replacement is blocked ({blocked}). {sentence}")
    return ("the cached mesh cannot be used or replaced: group cleanup is "
            "dirty, a prior proc stop failed, or a teardown outcome was lost "
            f"or is unknown. {sentence}")


def note_liveness_release(now: float | None = None, addresses: Any = None) -> None:
    """Arm the note the next creation block adds to its poison text."""
    global _RELEASED_PREDECESSOR, _RELEASED_ADDRESSES
    _RELEASED_PREDECESSOR = time.monotonic() if now is None else now
    _RELEASED_ADDRESSES = None if addresses is None else frozenset(addresses)


def clear_liveness_release(addresses: Any) -> None:
    """Disarm the note once the attach it authorized for ``addresses`` succeeded."""
    global _RELEASED_PREDECESSOR, _RELEASED_ADDRESSES
    if _RELEASED_ADDRESSES is None or frozenset(addresses) == _RELEASED_ADDRESSES:
        _RELEASED_PREDECESSOR = None
        _RELEASED_ADDRESSES = None


def released_predecessor_recently(now: float | None = None, addresses: Any = None) -> bool:
    """Check whether fresh liveness evidence released the predecessor fleet.

    This read does not consume the note; ``publish_creation_block`` does that
    when recording the replacement failure, and ``clear_liveness_release`` when
    the replacement attached. The attach path uses it to give
    the replacement one attempt unless a loop moved to a new id during it: a
    second dial with the same ids cannot win, and the fallback is a ComfyUI
    restart
    (``mesh_attach._attach_and_retry``, docs/TROUBLESHOOTING.md #2). Other
    attach failures retain a second attempt.
    """
    stamp = _RELEASED_PREDECESSOR
    if stamp is None:
        return False
    # Only the fleet the proof released: another cluster's attach keeps its retry.
    if addresses is not None and _RELEASED_ADDRESSES is not None \
            and frozenset(addresses) != _RELEASED_ADDRESSES:
        return False
    current = time.monotonic() if now is None else now
    return 0.0 <= current - stamp <= _release_note_window_s()


def publish_creation_block(handle: Any, blocked_text: str) -> str:
    """Poison a handle whose bring-up failed with owned processes left over.

    Return the text with the release note when a liveness release authorized
    this bring-up, for the transport poison too: every later attach prints the
    poison, with or without a handle to carry the block.
    """
    global _RELEASED_PREDECESSOR
    stamp = _RELEASED_PREDECESSOR
    released = (getattr(handle, "_liveness_released_predecessor", False)
                or (stamp is not None
                    and time.monotonic() - stamp <= _release_note_window_s()))
    _RELEASED_PREDECESSOR = None
    text = f"{blocked_text}{_LIVENESS_RELEASE_NOTE}" if released else blocked_text
    if handle is None or getattr(handle, "teardown_complete", False):
        return text
    handle._replacement_retryable = False
    handle.replacement_blocked = text
    handle._block_cause = TAG_CREATION_CLEANUP_UNCONFIRMED
    return text


def prepare_stop_attempt(handle: Any, action: str) -> bool:
    """Authorize one stop attempt without converting ambiguity into a retry."""
    blocked = getattr(handle, "replacement_blocked", None)
    retrying = bool(
        blocked and getattr(handle, "_replacement_retryable", False))
    if blocked and not retrying:
        raise mesh_safety.PriorTeardownUnknownError(
            f"{action} refused: an earlier ProcMesh stop has no confirmed "
            "failure outcome; a second stop cannot safely resolve it")
    if mesh_safety.token_present(id(handle)):
        if retrying:
            # A published failure outranks a token whose final clear was lost.
            mesh_safety.clear_stale_token(id(handle))
        else:
            raise mesh_safety.PriorTeardownUnknownError(
                f"{action} refused: a prior teardown's outcome is unknown; "
                "a second stop is unauthorized")
    return retrying


def stop_failure_is_retryable(exc: BaseException | None) -> bool:
    """Only a reported non-deadline, non-cancellation failure permits retry."""
    return (isinstance(exc, Exception)
            and not isinstance(exc, TimeoutError)
            and type(exc).__name__ != "CancelledError")


def enter_stop_boundary(handle: Any, retrying: bool) -> None:
    """Retire a previous failure only once the fresh stop is being issued."""
    if retrying:
        handle.replacement_blocked = None
        handle._replacement_retryable = False
        handle._block_cause = None


def timeout_cause(handle: Any) -> str:
    """Which of the two stop timeouts this is.

    Only a stop that times out after supervision reported a worker death can be
    released by a liveness proof. The same timeout on a Reset an operator
    pressed, on a clean detach, or on a stall-guard eviction has no death behind
    it, so no liveness proof may release it.
    """
    death = getattr(handle, "_supervision_death_report", None)
    deliberate = bool(getattr(handle, "_deferred_eviction_deliberate", False))
    return (TAG_STOP_TIMEOUT_AFTER_DEATH
            if isinstance(death, str) and death and not deliberate
            else TAG_STOP_TIMEOUT_DELIBERATE)


def publish_stop_failure(
    handle: Any, evidence: str, exc: BaseException | None
) -> None:
    """Publish a failed stop and the cause its remedy depends on."""
    handle._replacement_retryable = stop_failure_is_retryable(exc)
    handle.replacement_blocked = evidence
    handle._block_cause = (timeout_cause(handle) if isinstance(exc, TimeoutError)
                           else TAG_STOP_REPORTED_FAILURE)


def publish_stop_outcome(
    handle: Any, confirmed: bool, attempted: bool, unknown_detail: str
) -> None:
    """Publish the stop result before its matching authority token is cleared."""
    if confirmed:
        handle.teardown_complete = True
        handle.replacement_blocked = None
        handle._replacement_retryable = False
        handle._block_cause = None
    elif attempted and handle.replacement_blocked is None:
        handle._replacement_retryable = False
        handle.replacement_blocked = unknown_detail
        handle._block_cause = TAG_STOP_INTERRUPTED


def publish_setup_rollback_outcome(
    handle: Any, confirmed: bool, attempted: bool,
    stop_error: BaseException | None,
) -> None:
    if confirmed:
        publish_stop_outcome(handle, True, attempted, "")
        return
    if attempted and isinstance(stop_error, Exception):
        # The stop ran and answered, so its own cause decides the remedy, as for
        # every other stop: a timeout after a Worker died heals by itself once
        # the driver proves the processes gone, and a reported failure retries.
        verb = "timed out" if isinstance(stop_error, TimeoutError) else "failed"
        publish_stop_failure(
            handle, f"failed setup rollback proc stop {verb} ({failure_summary(stop_error)}); "
            "process state unknown", stop_error)
        return
    suffix = type(stop_error).__name__ if stop_error is not None else "interrupted"
    phase = "during" if attempted else "before"
    publish_stop_failure(
        handle, f"failed setup rollback proc stop was interrupted {phase} "
        f"execution ({suffix}); process state unknown", stop_error)
    handle._block_cause = TAG_SETUP_ROLLBACK_INTERRUPTED


def stop_failed_setup_procs(handle: Any, run_off_loop: Any) -> BaseException | None:
    """Run setup rollback under one scratch-owned outcome publication."""
    def stop_once() -> BaseException | None:
        if getattr(handle, "teardown_complete", False):
            return None
        try:
            retrying = prepare_stop_attempt(handle, "failed setup rollback")
        except mesh_safety.PriorTeardownUnknownError as exc:
            return exc
        confirmed = attempted = False
        stop_error: BaseException | None = None
        try:
            mesh_safety.begin_deliberate_teardown(id(handle))
            attempted = True
            enter_stop_boundary(handle, retrying)
            stop_with_reason(handle, "dgx-monarch failed setup rollback", proc_stop_bound_s())
            confirmed = True
        except BaseException as exc:
            stop_error = exc
        finally:
            publish_setup_rollback_outcome(
                handle, confirmed, attempted, stop_error)
            try:
                mesh_safety.end_deliberate_teardown(
                    id(handle), stop_confirmed=confirmed)
            except BaseException as end_exc:
                # Keep a stop failure primary; a lone handoff failure is returned.
                if stop_error is None:
                    stop_error = end_exc
                else:
                    safe_note(
                        stop_error,
                        "deliberate-teardown authority handoff also failed",
                        end_exc,
                    )
        return stop_error

    try:
        return run_off_loop(stop_once, proc_stop_bound_s() + ACTOR_STOP_S + 30, "dgxm-setup-rollback")
    except TimeoutError as exc:
        # The scratch owner still publishes the eventual outcome.
        return exc


def reconcile_thread_started(thread: Any) -> bool:
    is_alive = getattr(thread, "is_alive", None)
    return bool(getattr(thread, "ident", None) is not None
                or (callable(is_alive) and is_alive()))


def retire_retryable_reconcile(handle: Any) -> None:
    """Retire a settled failed reconcile so later evidence gets fresh authority."""
    current = getattr(handle, "_supervision_reconcile_thread", None)
    is_alive = getattr(current, "is_alive", None)
    if (getattr(handle, "replacement_blocked", None)
            and getattr(handle, "_replacement_retryable", False)
            and current is not None
            and callable(is_alive)
            and reconcile_thread_started(current)
            and not is_alive()):
        handle._supervision_reconcile_thread = None
        handle._supervision_reconcile_token = None


def reconcile_thread_main(handle: Any, exc: BaseException, token: object) -> None:
    reconcile_running.note_started(handle)  # bounds how long a refusal says the stop is running
    try:
        reconcile_stop(handle, exc, token)
    except BaseException as reconcile_exc:
        safe_note(exc, "supervision reconcile thread was interrupted", reconcile_exc)
        if not mesh_safety.token_present(id(handle)):
            latch_reconcile_launch_failure(
                handle, "supervision reconcile thread ended without a terminal outcome")
    finally:
        if (not getattr(handle, "teardown_complete", False)
                and not getattr(handle, "replacement_blocked", None)
                and not mesh_safety.token_present(id(handle))):
            latch_reconcile_launch_failure(
                handle, "supervision reconcile thread ended without a terminal outcome")


def latch_reconcile_launch_failure(handle: Any, detail: str) -> None:
    lock = getattr(handle, "lock", None)
    with lock if lock is not None else nullcontext():
        latch_no_stop_locked(handle, detail)


def _nothing_published(handle: Any) -> bool:
    """No outcome, no block and no stop in flight. Waits for the handle lock, without a bound."""
    lock = getattr(handle, "lock", None)
    with lock if lock is not None else nullcontext():
        return (not getattr(handle, "teardown_complete", False)
                and not getattr(handle, "replacement_blocked", None)
                and not mesh_safety.token_present(id(handle)))


def _settle_refused_reconcile(handle: Any, shut_exc: BaseException) -> str | None:
    """Publish what a reconcile shutdown that raised leaves behind; how it ended, or None.

    Read under the handle lock. A shutdown refused before its stop (a busy
    lock, a rooted read job, a live lease) leaves no block and no token: no stop
    was issued, so the fleet can still be stopped, and a Reset may stop it. A
    token with no block is a stop still in flight, which publishes its own
    outcome. A block is the outcome of a stop: this shutdown's, whose error
    text then becomes the evidence, or another's, which a refusal leaves alone.
    """
    lock = getattr(handle, "lock", None)
    with lock if lock is not None else nullcontext():
        if getattr(handle, "teardown_complete", False):
            return None
        if getattr(handle, "replacement_blocked", None):
            if not isinstance(shut_exc, LifecycleBusyError):
                handle.replacement_blocked = failure_summary(shut_exc)
            return "teardown failed and replacement remains blocked"
        if mesh_safety.token_present(id(handle)):
            return "in_flight"
        latch_no_stop_locked(handle, failure_summary(shut_exc))
        return "the cleanup was refused before it sent a stop"


def reconcile_stop(handle: Any, exc: BaseException, token: object) -> None:
    """Best-effort bounded stop outside the failing caller.

    A transition that holds the lifecycle lock past the shutdown's lock wait
    (a setup rollback, a policy apply) is waited out; if it published nothing,
    this cleanup still owns the stop and sends it then.
    """
    def stop_fleet() -> None:
        handle.shutdown(timeout_s=RECONCILE_GROUP_S, stop_timeout_s=proc_stop_bound_s(),
                        _reconcile_token=token)

    try:
        try:
            stop_fleet()
        except LifecycleBusyError as busy:
            if (isinstance(busy, SampleResultBusyError) or not _nothing_published(handle)
                    or driver_exit.exiting()):  # read after the wait: an exit sends nothing
                raise
            log.info("the defunct mesh's cleanup waited for another lifecycle transition "
                     "to end; it stops the fleet now")
            reconcile_running.note_started(handle)  # the refusals' figure counts from this stop
            stop_fleet()
    except mesh_safety.PriorTeardownUnknownError as shut_exc:
        # Fault evidence does not prove that an outcome-less stop failed.
        safe_note(exc, "supervision cleanup found a prior stop with no known outcome", shut_exc)
        log.error("the defunct mesh's shutdown found a prior stop with no known outcome; "
                  "no second stop was issued")
        return
    except BaseException as shut_exc:  # never mask the original failure
        ended = _settle_refused_reconcile(handle, shut_exc)
        if ended is None:
            return
        if ended == "in_flight":
            log.error("the defunct mesh's cleanup was refused (%r) while another stop of it "
                      "is in flight; that stop publishes the outcome", shut_exc)
            return
        log.error("the defunct mesh did not shut down; replacement is blocked: %r", shut_exc)
    else:
        lock = getattr(handle, "lock", None)
        with lock if lock is not None else nullcontext():
            handle.teardown_complete = True
            handle.replacement_blocked = None
            handle._replacement_retryable = False
            handle._block_cause = None
        return
    log.error("worker fleet failure (%s): %s. %s",
              failure_summary(exc), ended, block_cause_sentence(handle))


def _lease_total(handle: Any, attribute: str) -> int:
    total = 0
    for count in list(dict(getattr(handle, attribute, None) or {}).values()):
        try:
            total += int(count)
        except (TypeError, ValueError):
            continue
    return total


def retire_after_liveness_proof(handle: Any, evidence: Any) -> bool:
    """Retire a stop-timeout latch on positive proof the workers are gone.

    Five preconditions, all re-read under the handle lock so a race cannot
    widen any of them:

    1. supervision stored a death report through its non-deliberate route. The
       route is the test, never the wording: a classified supervision death
       often never contains the word stopped;
    2. the recorded cause is the after-death timeout, and only that one;
    3. the proof is a fleet-wide gone no older than the evidence window;
    4. no live sample lease and no pending read. An abandoned lease does not
       refuse: the stop that created this latch already ran with abandoned
       leases allowed, and a crashed render retires its leases by abandoning
       them, so refusing here would refuse the crash this release is for;
    5. no reconcile thread is pending or still running.

    It issues no stop: the processes are gone, and a second stop against a
    channel that just stalled is what ``prepare_stop_attempt`` refuses.
    """
    from .rdma_read_job import pending_job_count_for_handle
    from .telemetry import emit

    lock = getattr(handle, "lock", None)
    with lock if lock is not None else nullcontext():
        report = getattr(handle, "_supervision_death_report", None)
        if not isinstance(report, str) or not report:
            return False
        if block_cause(handle) != TAG_STOP_TIMEOUT_AFTER_DEATH:
            return False
        per_host = tuple(getattr(evidence, "per_host", ()) or ())
        if (getattr(evidence, "answer", "") != "gone"
                or getattr(evidence, "age_s", EVIDENCE_MAX_AGE_S + 1.0) > EVIDENCE_MAX_AGE_S
                or not per_host
                or any(answer != "gone" for _address, answer in per_host)):
            return False
        if _lease_total(handle, "sample_leases") or pending_job_count_for_handle(handle):
            return False
        thread = getattr(handle, "_supervision_reconcile_thread", None)
        if thread is not None and not (
                reconcile_thread_started(thread) and not thread.is_alive()):
            return False
        abandoned = _lease_total(handle, "abandoned_sample_leases")
        handle.teardown_complete = True
        handle.replacement_blocked = None
        handle._replacement_retryable = False
        handle._block_cause = None
        # Never note_deliberate_teardown here: no stop succeeded, so there is no
        # stop-success instant to date an absorption grace window from.
        mesh_safety.clear_stale_token(id(handle))
        handle._supervision_reconcile_thread = None
        handle._supervision_reconcile_token = None
        handle._liveness_evidence = evidence
        handle._liveness_released_predecessor = True
    note_liveness_release(addresses=[host.address for host in getattr(getattr(handle, 'config', None), 'hosts', None) or ()])
    hosts = ",".join(f"{address}={answer}" for address, answer in per_host)
    log.warning(
        "replacement latch released on liveness proof cause=%s report=%s "
        "hosts=%s abandoned=%d",
        TAG_STOP_TIMEOUT_AFTER_DEATH, report[:120], hosts, abandoned)
    emit("latch_release", cause=TAG_STOP_TIMEOUT_AFTER_DEATH, hosts=hosts,
         report=report[:120])
    return True
