"""Creation publication and claim finalization for the driver mesh cache."""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from typing import Any

from . import mesh_factory, mesh_safety
from .error_utils import (
    prefer_error,
    raise_with_distinct_cause,
    reconcile_error,
    safe_note,
)

Runtime = dict[str, Any]
_ABANDONED_REAP_LOCK = threading.Lock()


@dataclass
class CreationAttempt:
    """Durable identity and recovery authority for one cache-key creator."""

    key: tuple
    owner_frame: Any
    owner_thread_id: int
    handoff: mesh_factory.FleetHandoff
    cleanup_claim: object
    transport_claim: tuple[int, str, object]
    expected_bind: str | None
    handle: Any | None = None
    phase: str = "building"


def attempt_is_active(attempt: CreationAttempt) -> bool:
    """Whether the exact get_mesh frame that owns this attempt is still live."""
    current = sys._current_frames().get(attempt.owner_thread_id)
    while current is not None:
        if current is attempt.owner_frame:
            return True
        current = current.f_back
    return False


def publish_pending_handle(runtime: Runtime, key: tuple, handle: Any) -> None:
    # Teardown tokens are keyed by a handle's address (mesh_safety). A live
    # handle's stop holds the handle, so a token already at a new handle's
    # address is one a freed handle stranded, never a stop of this fleet.
    if mesh_safety.token_present(id(handle)):
        mesh_safety.clear_stale_token(id(handle))
    with runtime["_MESH_LOCK"]:
        runtime["_MESH_PENDING"][key] = handle


def publish_ready_handle(
    runtime: Runtime,
    attempt: CreationAttempt,
    handle: Any,
) -> None:
    """Move one lease-armed handle into the public registry atomically."""
    with runtime["_MESH_LOCK"]:
        key, handoff = attempt.key, attempt.handoff
        meshes = runtime["_MESHES"]
        pending = runtime["_MESH_PENDING"]
        if meshes.get(key) is handle:
            handoff.ready_for_publication = True
            attempt.phase = "ready"
            pending.pop(key, None)
            return
        generation, claim_mode, _owner = attempt.transport_claim
        if runtime["_MESH_CREATING"].get(key) is not attempt:
            raise runtime["MeshAttachError"](
                "new worker fleet lost its exact creation claim before publication")
        if attempt.transport_claim not in runtime["_TRANSPORT_CLAIMS"]:
            raise runtime["MeshAttachError"](
                "new worker fleet lost its transport claim before publication")
        if (
            runtime["_TRANSPORT_POISON"] is not None
            or generation != runtime["_TRANSPORT_CLAIM_GENERATION"]
            or claim_mode != runtime["_TRANSPORT_MODE"]
            or (
                attempt.expected_bind is not None
                and runtime["_TRANSPORT_BIND"] != attempt.expected_bind
            )
        ):
            raise runtime["MeshAttachError"](
                "new worker fleet cannot publish after its transport verdict changed")
        if pending.get(key) is not handle:
            raise runtime["MeshAttachError"](
                "new worker fleet lost its pending registry entry before public publication")
        if runtime["_coherent_lifecycle_verdict"](handle) != "live":
            raise runtime["MeshAttachError"](
                "new worker fleet became non-live before public publication")
        handoff.ready_for_publication = True
        attempt.phase = "ready"
        handle._creation_phase = "ready"
        meshes[key] = handle
        pending.pop(key, None)


def publication_resolved(
    runtime: Runtime, attempt: CreationAttempt, handle: Any | None
) -> bool:
    """Whether this creator no longer owes the registry a live, unpublished fleet.

    Callers hold ``_MESH_LOCK``. A creator that stored the READY marker keeps
    its marker, its transport claim and its cleanup claim until the exact
    handle is public or is no longer live. Releasing any of the three first
    lets the next ``get_mesh`` read an empty ``_MESHES`` and spawn a second
    fleet over the transport and ports this one still holds. A handle that
    is no longer live stays in ``_MESH_PENDING``, where ``admission_occupant``
    refuses every creation over it until its stop is confirmed; keeping the
    claims as well would refuse those creations as an endless one in flight,
    with no cause and no remedy.
    """
    if handle is None or not attempt.handoff.ready_for_publication:
        return True
    if runtime["_MESHES"].get(attempt.key) is handle:
        return True
    try:
        return runtime["_coherent_lifecycle_verdict"](handle) != "live"
    except Exception:
        return False


def admission_occupant(runtime: Runtime, key: tuple) -> Any | None:
    """This key's public handle, refusing admission over a stranded fleet.

    Callers hold ``_MESH_LOCK`` and have already proved no creator is active.
    A handle left in ``_MESH_PENDING`` is a fleet whose publication was
    interrupted; ``_MESHES`` alone cannot see it. Only a confirmed stop
    (``completed``) admits a new fleet over it: a live one would share its
    transport, and a blocked, unresolved or unreadable one may still have
    processes whose stop is unknown. get_mesh runs the liveness release on
    every cached and stranded handle first, so one proven gone is admitted.
    Another key's fleet whose stop is unconfirmed refuses too
    (_refuse_beside_unknown_stop).
    """
    published = runtime["_MESHES"].get(key)
    stranded = runtime["_MESH_PENDING"].get(key)
    if stranded is not None and stranded is not published:
        verdict = _verdict(runtime, stranded)
        if verdict != "completed":
            raise runtime["MeshAttachError"](_stranded_refusal(stranded, verdict))
    _refuse_beside_unknown_stop(runtime, key)
    return published


def try_release_all(runtime: Runtime) -> None:
    """Run the liveness release on every cached and stranded handle, before any lock.

    Each returns at once unless its fleet is latched after a Worker's death, so
    admission, for this key and every other, sees a fleet proven gone as stopped.
    """
    from . import mesh_liveness

    for registry in ("_MESHES", "_MESH_PENDING"):
        for handle in list(runtime[registry].values()):
            mesh_liveness.try_release(handle)


def _verdict(runtime: Runtime, handle: Any) -> str:
    try:
        return runtime["_coherent_lifecycle_verdict"](handle)
    except Exception:
        return "unknown"


def _refuse_beside_unknown_stop(runtime: Runtime, key: tuple) -> None:
    """Refuse a creation beside another key's fleet whose stop is unconfirmed.

    The first rule: never start a second fleet beside processes whose stop is
    unknown, and every fleet of this session counts as sharing the new one's
    machines. Two local fleets share this machine. Two cluster fleets of one
    session share one transport on one fabric, and no config text tells a
    machine's several addresses and its hostname apart, so the driver cannot
    show that two configs name different machines; an exact machine identity
    would need a new probe on every host before admission. The cost: a session
    that drives two separate clusters refuses the second while the first's stop
    is unconfirmed, until that stop is confirmed (its Reset, or the liveness
    proof after a Worker's death) or ComfyUI restarts. A completed or live fleet
    of another key admits as before.
    """
    for registry in ("_MESHES", "_MESH_PENDING"):
        for other_key, handle in list(runtime[registry].items()):
            if other_key == key:
                continue
            verdict = _verdict(runtime, handle)
            if verdict in ("completed", "live"):
                continue
            raise runtime["MeshAttachError"](_beside_refusal(handle, verdict, registry == "_MESHES"))


def _beside_refusal(handle: Any, verdict: str, published: bool) -> str:
    """Why another key's fleet refuses this creation: its own cause and remedy."""
    from . import mesh_teardown

    if not published:
        inner = _stranded_refusal(handle, verdict)
    elif verdict == "blocked":
        inner = mesh_teardown.blocked_refusal(handle, mesh_teardown.GET_MESH_DOOR)
    else:
        inner = mesh_teardown.unresolved_refusal(handle)
    return ("this driver already started a Worker fleet under another configuration in this "
            "session, and its stop is unconfirmed, so a second fleet could run beside its processes. "
            "Every fleet of one session counts as sharing machines, since a machine's several "
            "addresses cannot be told apart, so a second, separate cluster waits too until that "
            f"stop is confirmed or ComfyUI restarts: {inner}")


def _stranded_refusal(stranded: Any, verdict: str) -> str:
    """Why a stranded fleet refuses admission, and the remedy that reaches it."""
    from . import mesh_teardown

    head = "a worker fleet from an interrupted creation"
    if verdict == "live":
        return (f"{head} is still live but was never published, and a second fleet "
                "would share its transport; restart ComfyUI")
    if mesh_teardown.cleanup_running(stranded):
        return f"{head} was never published, and {mesh_teardown.cleanup_running_sentence()}"
    label = (mesh_teardown.label_for_cause(mesh_teardown.block_cause(stranded))
             if verdict == "blocked" else "")
    state = {"blocked": "its stop failed", "unresolved": "its teardown has no outcome"}.get(
        verdict, "its state cannot be read")
    return (f"{head} was never published and its stop is unconfirmed ({label or state}); a "
            "second fleet could run beside its processes. The sidebar Reset cannot reach a "
            "fleet the cache never held, so restart ComfyUI: its exit stops the processes "
            "this driver started, and dgxm restart clears any that outlive it")


def recover_owned_spawned_proc(
    handoff: mesh_factory.FleetHandoff,
    primary: BaseException | None,
) -> Any | None:
    """Best-effort ownership recovery that preserves a real primary failure."""
    if handoff.proc_ownership_unconfirmed:
        return None
    try:
        return mesh_factory.owned_spawned_proc(handoff)
    except BaseException as ownership_exc:
        if handoff.proc_spawn_called:
            handoff.proc_ownership_unconfirmed = True
        if primary is None:
            raise
        safe_note(
            primary,
            "spawned ProcMesh ownership recovery also failed",
            ownership_exc,
        )
        return None


def _settle_failed_creation(
    runtime: Runtime,
    attempt: CreationAttempt,
    handle: Any | None,
    primary: BaseException | None,
) -> None:
    """Finish or durably poison ownership cleanup without trusting a latch store."""
    handoff = attempt.handoff
    attempt.phase = "rollback"
    if handle is not None:
        handle._creation_phase = "rollback"
    evidence_owner = (
        primary if primary is not None
        else RuntimeError("mesh creator exited before publishing a READY handle"))
    unsafe_attach = (
        handoff.attach_in_progress and not handoff.transport_failure_safe)
    blocked: Any = None
    if handle is not None and not getattr(handle, "teardown_complete", False):
        blocked = getattr(handle, "replacement_blocked", None)
    fleet = handoff.fleet
    owned_procs = (
        fleet[1] if fleet is not None
        else recover_owned_spawned_proc(handoff, primary)
    )
    if handoff.cleanup_started and not handoff.cleanup_confirmed:
        blocked = blocked or "worker spawn rollback outcome is unconfirmed"
    elif handoff.proc_ownership_unconfirmed:
        blocked = blocked or "spawned ProcMesh ownership is unconfirmed"
    elif owned_procs is not None and not handoff.cleanup_started and not blocked:
        if runtime["_TRANSPORT_POISON"] is None:
            runtime["_TRANSPORT_POISON"] = mesh_factory.CREATION_CLEANUP_POISON
        blocked = mesh_factory.reconcile_creation_cleanup(
            handoff, handle, owned_procs, evidence_owner)
    if unsafe_attach or (handoff.spawn_in_progress and owned_procs is None):
        blocked = blocked or (
            "cluster attach result handoff interrupted" if unsafe_attach
            else "actor spawn cleanup state is unavailable")
    if handoff.callback_cleanup_unconfirmed:
        blocked = blocked or "ProcMesh spawn callback cleanup is unconfirmed"
    if blocked:
        blocked_text = mesh_factory.safe_failure_evidence(blocked)
        if handle is not None and not getattr(handle, "teardown_complete", False):
            handle._replacement_retryable = False
            handle.replacement_blocked = blocked_text
        primary_text = mesh_factory.safe_failure_evidence(evidence_owner)
        mesh_factory.safe_note(
            evidence_owner,
            f"owned ProcMesh cleanup after mesh creation failed: {blocked_text}",
        )
        if (
            runtime["_TRANSPORT_POISON"] is None
            or runtime["_TRANSPORT_POISON"]
            is mesh_factory.CREATION_CLEANUP_POISON
        ):
            runtime["_TRANSPORT_POISON"] = (
                f"mesh creation failed ({primary_text}); owned ProcMesh cleanup "
                f"unconfirmed ({blocked_text})")


def finalize_creation_once(
    runtime: Runtime,
    attempt: CreationAttempt,
    handle: Any | None,
    transport_releasable: bool | None,
    creation_succeeded: bool,
    primary: BaseException | None,
) -> None:
    """Idempotently retire one creation claim after success or compensation."""
    key = attempt.key
    transport_claim = attempt.transport_claim
    cleanup_claim = attempt.cleanup_claim
    fleet_handoff = attempt.handoff
    error: BaseException | None = None
    if not creation_succeeded and not fleet_handoff.ready_for_publication:
        try:
            _settle_failed_creation(runtime, attempt, handle, primary)
        except BaseException as exc:
            error = prefer_error(error, exc, "creation cleanup recovery failed")
    try:
        if fleet_handoff.ready_for_publication:
            if handle is None:
                raise runtime["MeshAttachError"]("READY mesh creation lost its handle")
            publish_ready_handle(runtime, attempt, handle)
            creation_succeeded = True
    except BaseException as exc:
        error = prefer_error(error, exc, "READY publication finalization failed")
    with runtime["_MESH_LOCK"]:
        resolved = publication_resolved(runtime, attempt, handle)
    try:
        if resolved and (
            creation_succeeded
            or mesh_factory.creation_cleanup_settled(fleet_handoff)
        ):
            mesh_factory.confirm_creation_cleanup(cleanup_claim)
    except BaseException as exc:
        error = prefer_error(error, exc, "cleanup-claim finalization also failed")
    try:
        if transport_claim is not None:
            with runtime["_MESH_CONDITION"]:
                generation, claim_mode, _owner = transport_claim
                release_transport = (
                    transport_releasable is True
                    and runtime["_TRANSPORT_BIND"] is None
                    and runtime["_TRANSPORT_POISON"] is None
                )
                if not release_transport:
                    runtime["_TRANSPORT_COMMITTED_GENERATION"] = generation
                if resolved:
                    runtime["_TRANSPORT_CLAIMS"].discard(transport_claim)
                if (
                    release_transport
                    and generation == runtime["_TRANSPORT_CLAIM_GENERATION"]
                    and runtime["_TRANSPORT_COMMITTED_GENERATION"] != generation
                    and not any(
                        claim[0] == generation
                        for claim in runtime["_TRANSPORT_CLAIMS"]
                    )
                    and runtime["_TRANSPORT_MODE"] == claim_mode
                ):
                    runtime["_TRANSPORT_MODE"] = None
                if runtime["_MESH_CREATING"].get(key) is attempt and resolved:
                    runtime["_MESH_CREATING"].pop(key, None)
                if (
                    not creation_succeeded
                    and not fleet_handoff.ready_for_publication
                    and handle is not None
                    and runtime["_MESH_PENDING"].get(key) is handle
                ):
                    runtime["_MESH_PENDING"].pop(key, None)
                runtime["_MESH_CONDITION"].notify_all()
    except BaseException as exc:
        error = prefer_error(error, exc, "transport-claim finalization also failed")
    try:
        if (
            runtime["_TRANSPORT_POISON"] is mesh_factory.CREATION_CLEANUP_POISON
            and not mesh_factory.creation_cleanup_pending()
        ):
            runtime["_TRANSPORT_POISON"] = None
    except BaseException as exc:
        error = prefer_error(error, exc, "creation poison finalization also failed")
    if error is not None:
        raise error


def finalize_creation(
    runtime: Runtime,
    attempt: CreationAttempt,
    handle: Any | None,
    transport_releasable: bool | None,
    creation_succeeded: bool,
    primary: BaseException | None,
) -> None:
    """Retry interrupted finalization and preserve exact cancellation identity."""
    first: BaseException | None = None
    cause: BaseException | None = None
    for _attempt in range(2):
        try:
            finalize_creation_once(
                runtime,
                attempt,
                handle,
                transport_releasable,
                creation_succeeded,
                primary,
            )
        except BaseException as exc:
            if first is None:
                first = exc
                continue
            first, cause = reconcile_error(
                first, exc, "mesh creation finalization retry failed")
            break
        else:
            if first is None:
                return
            cause = None
            break
    if first is None:  # defensive: reaching here requires one failed pass
        return
    winner, outer_cause = reconcile_error(
        primary, first, "mesh creation finalization failed")
    raise_with_distinct_cause(
        winner, outer_cause if outer_cause is not None else cause)


def reap_abandoned_attempts(runtime: Runtime) -> None:
    """Reconcile exact creator frames that exited before clearing their marker."""
    with _ABANDONED_REAP_LOCK:
        with runtime["_MESH_LOCK"]:
            abandoned = [
                attempt for attempt in runtime["_MESH_CREATING"].values()
                if isinstance(attempt, CreationAttempt)
                and not attempt_is_active(attempt)
            ]
        for attempt in abandoned:
            with runtime["_MESH_LOCK"]:
                handle = (
                    attempt.handle
                    if attempt.handle is not None
                    else runtime["_MESH_PENDING"].get(attempt.key)
                )
            try:
                finalize_creation(
                    runtime, attempt, handle, False,
                    attempt.handoff.ready_for_publication, None)
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    raise
                evidence = mesh_factory.safe_failure_evidence(exc)
                with runtime["_MESH_LOCK"]:
                    if runtime["_TRANSPORT_POISON"] is None:
                        runtime["_TRANSPORT_POISON"] = (
                            "abandoned mesh creation could not be reconciled: "
                            + evidence)
