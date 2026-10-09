"""Stop-attempt authority and supervision reconciliation for mesh lifecycles."""
from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

from . import mesh_safety
from .error_utils import failure_summary, safe_note
from .log import get_logger
from .mesh_runtime import mesh_creation_timeout_s

log = get_logger("dgx_monarch.mesh")

# Blocked-handle causes select distinct remedies. ``block_cause`` reads the
# dynamic attribute with a None default, which grants no recovery authority.
TAG_NO_STOP_ISSUED = "no_stop_issued"
TAG_STOP_REPORTED_FAILURE = "stop_reported_failure"
TAG_STOP_TIMEOUT_AFTER_DEATH = "stop_timeout_after_death"
TAG_STOP_TIMEOUT_DELIBERATE = "stop_timeout_deliberate"
TAG_SETUP_ROLLBACK_INTERRUPTED = "setup_rollback_interrupted"
TAG_STOP_INTERRUPTED = "stop_interrupted"
TAG_CREATION_CLEANUP_UNCONFIRMED = "creation_cleanup_unconfirmed"

# A liveness proof older than this is not a proof. The retire re-reads it under
# the handle lock, so a gone recorded while a live lease blocked the release can
# never authorize one minutes later.
EVIDENCE_MAX_AGE_S = 30.0

_CONFIRM_GONE = "restart ComfyUI after confirming the worker processes are gone"

# One sentence per cause, each naming the action that clears that cause. Only
# the after-death timeout promises a self-heal: it is the one cause a liveness
# proof can release (``retire_after_liveness_proof``).
_BLOCK_CAUSE_SENTENCES: dict[str | None, str] = {
    None: ("Reset the Attached mesh; restart ComfyUI only after confirming the "
           "worker processes are gone."),
    TAG_NO_STOP_ISSUED: (
        "No stop was issued, so the fleet can still be stopped: reset the "
        "Attached mesh."),
    TAG_STOP_REPORTED_FAILURE: (
        "The stop reported a failure: reset the Attached mesh, and restart "
        "ComfyUI if that reset cannot confirm the stop."),
    TAG_STOP_TIMEOUT_AFTER_DEATH: (
        "The driver checks the worker processes on the next attach; it "
        "replaces the fleet by itself if they are gone."),
    TAG_STOP_TIMEOUT_DELIBERATE: (
        f"A stop timed out with no worker death reported, so no second stop "
        f"will be issued: {_CONFIRM_GONE}."),
    TAG_SETUP_ROLLBACK_INTERRUPTED: (
        f"The setup rollback stop was interrupted: {_CONFIRM_GONE}."),
    TAG_STOP_INTERRUPTED: (
        f"The stop was interrupted before it reported an outcome: "
        f"{_CONFIRM_GONE}."),
    TAG_CREATION_CLEANUP_UNCONFIRMED: (
        f"Cleanup of the processes the failed bring-up spawned is unconfirmed: "
        f"{_CONFIRM_GONE}."),
}

# The same causes as a short phrase, for a panel row that names the cause and
# leaves the action to the remedy beside it.
_BLOCK_CAUSE_LABELS: dict[str | None, str] = {
    None: "",
    TAG_NO_STOP_ISSUED: "no stop was issued",
    TAG_STOP_REPORTED_FAILURE: "the stop reported a failure",
    TAG_STOP_TIMEOUT_AFTER_DEATH: "the stop timed out after a worker death",
    TAG_STOP_TIMEOUT_DELIBERATE: "the stop timed out with no worker death reported",
    TAG_SETUP_ROLLBACK_INTERRUPTED: "the setup rollback stop was interrupted",
    TAG_STOP_INTERRUPTED: "the stop was interrupted",
    TAG_CREATION_CLEANUP_UNCONFIRMED: "cleanup of the new processes is unconfirmed",
}

GET_MESH_DOOR = "get_mesh"
ENSURE_LIVE_DOOR = "ensure_live"

_LIVENESS_RELEASE_NOTE = (
    " (the fleet this one replaces was retired after the driver proved its "
    "worker processes were gone)")
# The instant of the last release, or None. The next creation block consumes
# it: that bring-up is the one the release authorized, and its poison text is
# where the operator needs to see it. Time bounded as well as one shot, because
# the ordinary path is a release whose replacement then succeeds and consumes
# nothing, and an unrelated failure hours later must not claim the release.
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


def sentence_for_cause(tag: str | None) -> str:
    """Return the recovery action for this cause, for CLI, panel, and log callers."""
    return _BLOCK_CAUSE_SENTENCES.get(tag, _BLOCK_CAUSE_SENTENCES[None])


def label_for_cause(tag: str | None) -> str:
    """This cause as a short phrase, or empty when the cause is unrecorded."""
    return _BLOCK_CAUSE_LABELS.get(tag, "")


def block_cause(handle: Any) -> str | None:
    """The recorded cause for a blocked handle, or None when it carries none."""
    tag = getattr(handle, "_block_cause", None)
    return tag if tag in _BLOCK_CAUSE_SENTENCES and tag is not None else None


def block_cause_sentence(handle: Any) -> str:
    return sentence_for_cause(block_cause(handle))


def blocked_refusal(handle: Any, door: str) -> str:
    """Format a lifecycle refusal with the recovery action for its cause."""
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


def released_predecessor_recently(now: float | None = None, addresses: Any = None) -> bool:
    """Check whether fresh liveness evidence released the predecessor fleet.

    This read does not consume the note; ``publish_creation_block`` does that
    when recording the replacement failure. The attach path uses it to limit
    replacement to one attempt because Monarch cannot recover that
    attach in the same driver process (docs/TROUBLESHOOTING.md #2). Other attach
    failures retain a second attempt.
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


def publish_creation_block(handle: Any, blocked_text: str) -> None:
    """Poison a handle whose bring-up failed with owned processes left over."""
    global _RELEASED_PREDECESSOR
    if handle is None or getattr(handle, "teardown_complete", False):
        return
    stamp = _RELEASED_PREDECESSOR
    released = (getattr(handle, "_liveness_released_predecessor", False)
                or (stamp is not None
                    and time.monotonic() - stamp <= _release_note_window_s()))
    _RELEASED_PREDECESSOR = None
    handle._replacement_retryable = False
    handle.replacement_blocked = (
        f"{blocked_text}{_LIVENESS_RELEASE_NOTE}" if released else blocked_text)
    handle._block_cause = TAG_CREATION_CLEANUP_UNCONFIRMED


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
            handle.procs.stop(
                "dgx-monarch failed setup rollback").get(timeout=60)
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
        return run_off_loop(stop_once, 90, "dgxm-setup-rollback")
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
        if not getattr(handle, "teardown_complete", False):
            # No ProcMesh stop was issued; the still-owned resource may be tried.
            handle._replacement_retryable = True
            handle.replacement_blocked = detail
            handle._block_cause = TAG_NO_STOP_ISSUED


def reconcile_stop(handle: Any, exc: BaseException, token: object) -> None:
    """Best-effort bounded stop outside the failing caller."""
    try:
        handle.shutdown(timeout_s=30, _reconcile_token=token)
    except mesh_safety.PriorTeardownUnknownError as shut_exc:
        # Fault evidence does not prove that an outcome-less stop failed.
        safe_note(exc, "supervision cleanup found a prior stop with no known outcome", shut_exc)
        log.error("the defunct mesh's shutdown found a prior stop with no known outcome; "
                  "no second stop was issued")
        return
    except BaseException as shut_exc:  # never mask the original failure
        lock = getattr(handle, "lock", None)
        with lock if lock is not None else nullcontext():
            if getattr(handle, "teardown_complete", False):
                return
            # The cause the stop itself published stands: this text replaces the
            # evidence, not the reason the handle is blocked.
            handle.replacement_blocked = failure_summary(shut_exc)
        log.error("the defunct mesh did not shut down; replacement is blocked: %r", shut_exc)
    else:
        lock = getattr(handle, "lock", None)
        with lock if lock is not None else nullcontext():
            handle.teardown_complete = True
            handle.replacement_blocked = None
            handle._replacement_retryable = False
            handle._block_cause = None
        return
    log.error(
        "worker fleet failure (%s): teardown failed and replacement remains blocked; "
        "restart ComfyUI after confirming the worker processes are gone", failure_summary(exc))


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
