"""Why a fleet's replacement is blocked: the causes a blocked handle records.

Each cause selects its own remedy sentence, for refusals, the panel and the
log, and a short label for a panel row that names the cause beside its remedy.
``block_cause`` reads the handle's dynamic attribute with a None default, which
grants no recovery authority. Kept apart from mesh_teardown for its line limit;
mesh_teardown re-exports every name here, and its callers read them there.
"""
from __future__ import annotations

from typing import Any

TAG_NO_STOP_ISSUED = "no_stop_issued"
TAG_STOP_REPORTED_FAILURE = "stop_reported_failure"
TAG_STOP_TIMEOUT_AFTER_DEATH = "stop_timeout_after_death"
TAG_STOP_TIMEOUT_DELIBERATE = "stop_timeout_deliberate"
TAG_SETUP_ROLLBACK_INTERRUPTED = "setup_rollback_interrupted"
TAG_STOP_INTERRUPTED = "stop_interrupted"
TAG_CREATION_CLEANUP_UNCONFIRMED = "creation_cleanup_unconfirmed"

_CONFIRM_GONE = "restart ComfyUI after confirming the worker processes are gone"

# One sentence per cause, each naming the action that clears that cause. Only
# the after-death timeout promises a self-heal: it is the one cause a liveness
# proof can release (``mesh_teardown.retire_after_liveness_proof``).
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


def latch_no_stop_locked(handle: Any, detail: str) -> bool:
    """Record that no ProcMesh stop was issued, so the still-owned fleet may be stopped.

    The caller holds the handle lock. Never over an outcome or over a block
    another stop published: that block's cause, and whether it may be retried,
    belong to the stop that wrote it.
    """
    if getattr(handle, "teardown_complete", False):
        return False
    if getattr(handle, "replacement_blocked", None) and block_cause(handle) != TAG_NO_STOP_ISSUED:
        return False
    handle._replacement_retryable = True
    handle.replacement_blocked = detail
    handle._block_cause = TAG_NO_STOP_ISSUED
    return True
