"""Settle jobs from the driver's queue and history snapshots.

ComfyUI keeps its queue and history in memory, so a restart, a history clear
or a dequeued cancellation can remove a prompt without any terminal event. A
known job is absent from the driver when a successful queue snapshot taken
after its state was read does not list it and an exact per-prompt history
lookup succeeds with no entry. The bulk history window is never proof of
absence: it shrinks for large workflows and drops entries past its count.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from .contracts import TERMINAL_STATES, StudioError, bounded_key, digest, identifier, timestamp
from .job_artifacts import import_artifacts
from .store import Store
from .threads import blocking

if TYPE_CHECKING:
    from .jobs import Jobs

LOOKUPS_PER_CYCLE = 32
QUARANTINE_ROWS = 8
# A row of driver data that cannot be read is skipped on its own. StudioError
# is a ValueError; storage errors are not listed and end the cycle instead.
ROW_FAULTS = (ValueError, TypeError, KeyError, RecursionError, OverflowError)
ABSENCE_STATES = frozenset({"queued", "running", "cancel_requested", "unknown"})
TERMINAL_MESSAGES = ("execution_success", "execution_error", "execution_interrupted")
_KNOWN_COLUMNS = ("id,json_extract(document,'$.prompt_id'),json_extract(document,'$.client_prompt_id'),"
                  "json_extract(document,'$.state'),json_extract(document,'$.accepted_at'),"
                  "json_type(document,'$.absence'),json_extract(document,'$.reconciliation.digest'),"
                  "json_type(document,'$.reconciliation')")
# Without table statistics the planner reads every job for an IS NULL test;
# the prompt index holds the few unbound jobs directly.
_UNBOUND = "INDEXED BY jobs_by_prompt"


def known_jobs(store: Store) -> dict[str, dict[str, Any]]:
    """Index jobs by the prompt identity the driver uses, without loading workflow payloads.

    A job whose acknowledgement was lost has no bound `prompt_id` yet; it is
    indexed by the identity the companion sent, so a queue row or an exact
    history lookup under that identity binds it.
    """
    with store.lock:
        rows = [*store.db.execute(f"SELECT {_KNOWN_COLUMNS} FROM objects WHERE kind='jobs' "  # noqa: S608
                                  "AND json_extract(document,'$.prompt_id') IS NOT NULL"),
                *store.db.execute(f"SELECT {_KNOWN_COLUMNS} FROM objects {_UNBOUND} WHERE kind='jobs' "  # noqa: S608
                                  "AND json_extract(document,'$.prompt_id') IS NULL "
                                  "AND json_extract(document,'$.client_prompt_id') IS NOT NULL")]
    known: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row[1] or row[2]
        if isinstance(key, str):
            known[key] = {"id": row[0], "prompt_id": row[1], "client_prompt_id": row[2], "state": row[3],
                          "accepted_at": row[4], "absent": row[5] == "object", "settled_digest": row[6],
                          "reconciled": row[7] == "object"}
    return known


def job_for_prompt(store: Store, prompt_id: str) -> dict[str, Any] | None:
    """One indexed lookup for a driver event, in place of reading every job."""
    with store.lock:
        row = store.db.execute("SELECT id,json_extract(document,'$.state') FROM objects WHERE kind='jobs' "
                               "AND json_extract(document,'$.prompt_id')=? LIMIT 1", (prompt_id,)).fetchone()
        if row is not None:
            return {"id": row[0], "state": row[1], "prompt_id": prompt_id}
        row = store.db.execute(f"SELECT id,json_extract(document,'$.state') FROM objects {_UNBOUND} WHERE kind='jobs' "  # noqa: S608
                               "AND json_extract(document,'$.prompt_id') IS NULL "
                               "AND json_extract(document,'$.client_prompt_id')=? LIMIT 1", (prompt_id,)).fetchone()
    return None if row is None else {"id": row[0], "state": row[1], "prompt_id": None, "client_prompt_id": prompt_id}


def entry_digest(value: dict[str, Any]) -> str:
    """Digest what can change in a history entry; its prompt is fixed by the prompt id."""
    return digest({key: item for key, item in value.items() if key != "prompt"})


def completion_time(messages: list[Any]) -> float | None:
    for message in reversed(messages):
        if (isinstance(message, (list, tuple)) and len(message) == 2 and message[0] in TERMINAL_MESSAGES
                and isinstance(message[1], dict)):
            value = message[1].get("timestamp")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    seconds = value / 1000
                except OverflowError:
                    return None
                return seconds if math.isfinite(seconds) and seconds > 0 else None
    return None


def _quarantine(rows: list[dict[str, Any]], source: str, key: Any, exc: BaseException) -> None:
    try:
        identity = bounded_key(key, "prompt id")
    except StudioError:
        identity = None
    rows.append({"source": source, "prompt_id": identity,
                 "code": exc.code if isinstance(exc, StudioError) else type(exc).__name__[:64]})


def _bind(store: Store, job: dict[str, Any], prompt_id: str, state: str) -> None:
    updates: dict[str, Any] = {"state": state, "prompt_id": prompt_id}
    if job.get("absent") or job.get("absence"):
        updates["absence"] = None
    if job.get("prompt_id") is None and not job.get("external") and not job.get("accepted_at"):
        updates["accepted_at"] = timestamp()
    store.update("jobs", job["id"], updates)


async def reconcile_cycle(owner: Jobs) -> None:
    store, client = owner.store, owner.client
    quarantined: list[dict[str, Any]] = []
    try:
        # Read identities before the queue: dispatch writes outside the
        # observation lock, so every state read here predates the snapshot.
        known = known_jobs(store)
        snapshot_at = timestamp()
        try:
            queue: dict[str, Any] | None = await client.request("GET", "/queue")
            if not isinstance(queue, dict):
                raise StudioError("driver_schema", "Driver queue snapshot is malformed.", 502)
            queue_state: dict[str, Any] = {"state": "connected"}
        except StudioError as exc:
            if exc.code != "driver_response_limit":
                raise
            # A long queue of large graphs can pass the response budget while
            # the driver is healthy. Without a queue snapshot nothing is absent.
            queue, queue_state = None, {"state": "over_budget", "code": exc.code}
        owner.driver = {"state": "connected", "timestamp": timestamp(), "queue": queue_state}
        active: set[str] = set()
        for state, key in (("running", "queue_running"), ("queued", "queue_pending")):
            rows = queue.get(key, []) if queue is not None else []
            for entry in rows if isinstance(rows, list) else []:
                try:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2 and isinstance(entry[1], str):
                        active.add(entry[1])
                    job = discover(store, entry, known)
                    if job:
                        _bind(store, job, entry[1], state)
                except ROW_FAULTS as exc:
                    _quarantine(quarantined, "queue", entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else None, exc)
        try:
            history = await client.request("GET", "/history")
            if not isinstance(history, dict):
                raise StudioError("driver_schema", "Driver history snapshot is malformed.", 502)
            owner.driver["history"] = {"state": "connected", "limit": getattr(client, "history_limit", None)}
        except StudioError as exc:
            # A large or unavailable history must not hide the queue that
            # was observed successfully. Exact known identities can still
            # reconcile through the per-prompt route below.
            history = {}
            owner.driver["history"] = {"state": "unavailable", "code": exc.code, "limit": getattr(client, "history_limit", None)}
        outcomes = await _lookups(owner, known, history, active)
        in_flight = frozenset(getattr(owner, "cancel_dispatches", ()))
        await blocking(_settle, owner, history, known, quarantined,
                       outcomes if queue is not None else {}, snapshot_at, in_flight)
        if quarantined:
            owner.driver["quarantine"] = {"count": len(quarantined), "rows": quarantined[:QUARANTINE_ROWS],
                                          "reason": "Driver rows that could not be read were skipped; other jobs reconciled."}
        events_fault = getattr(owner, "events_fault", None)
        if events_fault:
            owner.driver["events"] = dict(events_fault)
        await owner.runtime_events.poll()
        owner.telemetry["runtime_events"] = owner.runtime_events.status
        try:
            sample = await client.request("GET", "/dgxm/telemetry")
            owner.telemetry = {"sample": sample, "observed_at": timestamp(), "state": "connected",
                               "runtime_events": owner.runtime_events.status}
            owner.telemetry_clock = time.monotonic()
        except StudioError:
            owner.telemetry = {**owner.telemetry, "state": "stale" if owner.telemetry else "unavailable"}
    except StudioError as exc:
        owner.degrade(exc.code)


async def _lookups(owner: Jobs, known: dict[str, Any], history: dict[str, Any], active: set[str]) -> dict[str, str]:
    """Ask the driver for each missing known prompt; keep found, empty and failed apart."""
    lookup = getattr(owner.client, "history_prompt", None)
    if not callable(lookup):
        return {}
    # A submitting job is still in flight to the driver; its dispatch settles it.
    missing = [key for key, job in known.items() if key not in history and key not in active and not job.get("absent")
               and job["state"] != "submitting" and (job["state"] not in TERMINAL_STATES or not job.get("reconciled"))]
    offset = owner.history_cursor % len(missing) if missing else 0
    outcomes: dict[str, str] = {}
    for key in (missing[offset:] + missing[:offset])[:LOOKUPS_PER_CYCLE]:
        try:
            older = await lookup(key)
        except StudioError:
            outcomes[key] = "failed"
            continue
        if isinstance(older, dict) and isinstance(older.get(key), dict):
            history[key] = older[key]
            outcomes[key] = "found"
        else:
            # Only a successful empty answer is evidence of absence.
            outcomes[key] = "empty" if older == {} else "failed"
    owner.history_cursor = offset + LOOKUPS_PER_CYCLE
    return outcomes


def _settle(owner: Jobs, history: dict[str, Any], known: dict[str, Any], quarantined: list[dict[str, Any]],
            outcomes: dict[str, str], snapshot_at: float, in_flight: frozenset[str]) -> None:
    for prompt_id, value in history.items():
        try:
            _settle_entry(owner, prompt_id, value, known)
        except ROW_FAULTS as exc:
            _quarantine(quarantined, "history", prompt_id, exc)
    for key, outcome in outcomes.items():
        job = known.get(key)
        if outcome == "empty" and job is not None and job["id"] not in in_flight:
            _settle_absent(owner.store, key, job, snapshot_at)


def _settle_entry(owner: Jobs, prompt_id: str, value: Any, known: dict[str, Any]) -> None:
    store = owner.store
    if not isinstance(value, dict):
        return
    entry = value.get("prompt", [])
    if isinstance(entry, (list, tuple)) and len(entry) >= 2 and entry[1] != prompt_id:
        return
    job = discover(store, entry, known)
    if job is None:
        job = known.get(prompt_id)
    if not job:
        return
    status = value.get("status", {})
    if not isinstance(status, dict):
        return
    settled = entry_digest(value)
    recorded = job.get("reconciliation")
    if settled == (job.get("settled_digest") or (recorded.get("digest") if isinstance(recorded, dict) else None)):
        return  # Unchanged since this entry was last settled: no reads, writes or imports.
    messages = status.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    types = {row[0] for row in messages if isinstance(row, list) and row and isinstance(row[0], str)}
    state = "cancelled" if "execution_interrupted" in types else "failed" if status.get("status_str") == "error" else (
        "succeeded" if status.get("completed") is True else "unknown")
    for message in messages:
        if isinstance(message, (list, tuple)) and len(message) == 2 and isinstance(message[1], dict):
            owner.consume_event({"type": message[0], "data": message[1]}, known=known)
    # Import before recording the digest, so an interrupted import is retried.
    import_artifacts(store, job["id"], value.get("outputs", {}), state=state,
                     completed_at=completion_time(messages), prompt_id=prompt_id)
    updates: dict[str, Any] = {"state": state, "prompt_id": prompt_id, "backend_status": status, "progress": None,
                               "reconciliation": {"source": "driver_history", "digest": settled}}
    if job.get("absent") or job.get("absence"):
        updates["absence"] = None
    store.update("jobs", job["id"], updates)
    known[prompt_id] = {**job, "prompt_id": prompt_id, "settled_digest": settled, "reconciled": True, "absent": False}


def _dequeued_basis(store: Store, document: dict[str, Any]) -> dict[str, Any] | None:
    """A cancel the driver acknowledged for a prompt that was still queued leaves no history."""
    cancellation = document.get("cancellation")
    cancellation = cancellation if isinstance(cancellation, dict) else {}
    receipt = cancellation.get("receipt")
    if not isinstance(receipt, dict) or not (receipt.get("cancelled") is True or receipt.get("deletion_requested") is True):
        return None
    reserved = cancellation.get("reserved_state")
    if reserved is None and document.get("cancellation_scope") == "queued_only":
        reserved = "queued"  # Queue deletion is only reserved for queued jobs.
    if reserved != "queued":
        return None
    return {"basis": "cancel_receipt_and_absent_from_driver", "reserved_state": reserved, "receipt": receipt,
            "intent_id": cancellation.get("intent_id")}


def _settle_absent(store: Store, key: str, job: dict[str, Any], snapshot_at: float) -> None:
    state = job["state"]
    if job.get("absent") or state not in ABSENCE_STATES | TERMINAL_STATES:
        return  # A submitting job may still be in flight to the driver.
    accepted = job.get("accepted_at")
    if isinstance(accepted, (int, float)) and accepted >= snapshot_at:
        return
    absence = {"reason": "absent_from_driver", "prompt_id": key, "observed_at": timestamp(),
               "basis": "queue_snapshot_and_prompt_lookup",
               "message": "The driver no longer holds this prompt in its queue or history."}
    with store.lock:
        # Cancellation reservations run on the event loop; hold the store
        # lock so the state read here is still current when it is written.
        document = store.get("jobs", job["id"])
        if document.get("state") != state or document.get("absence"):
            return
        if state in TERMINAL_STATES:
            # Outputs can no longer be fetched; stop asking for them.
            store.update("jobs", job["id"], {"absence": absence})
            return
        basis = _dequeued_basis(store, document) if state == "cancel_requested" else None
        if basis is not None:
            store.update("jobs", job["id"], {
                "state": "cancelled", "progress": None, "absence": absence,
                "settlement": {**basis, "observed_at": absence["observed_at"]},
                "cancellation": {**document["cancellation"], "completion": "confirmed_absent_from_driver"}})
            if isinstance(basis.get("intent_id"), str):
                try:
                    store.update("job_cancellations", basis["intent_id"], {"completion": "confirmed_absent_from_driver"})
                except StudioError as exc:
                    if exc.code != "not_found":
                        raise
            return
        store.update("jobs", job["id"], {"state": "unknown", "progress": None, "absence": absence})


def discover(store: Store, entry: Any, known: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(entry, (list, tuple)) or len(entry) < 4 or not isinstance(entry[1], str):
        return None
    if entry[1] in known:
        return known[entry[1]]
    extra = entry[3] if isinstance(entry[3], dict) else {}
    studio = extra.get("studio")
    studio = studio if isinstance(studio, dict) else {}
    studio_id = studio.get("job_id")
    if isinstance(studio_id, str):
        try:
            job = store.get("jobs", studio_id)
            if (job.get("prompt_id") in (None, entry[1]) and studio.get("digest") == job["submission"].get("digest")
                    and entry[2] == job["submission"].get("prompt")):
                known[entry[1]] = job
                return job
        except StudioError:
            pass
    png_info = extra.get("extra_pnginfo")
    png_info = png_info if isinstance(png_info, dict) else {}
    job = {"id": identifier(), "state": "unknown", "prompt_id": entry[1], "created_at": timestamp(),
           "updated_at": timestamp(), "external": True,
           "submission": {"prompt": entry[2], "workflow": png_info.get("workflow"),
                          "requested": None, "effective": {"prompt": entry[2]},
                          "provenance_status": "imported_driver_snapshot", "versions": {}, "parent_artifact_ids": []}}
    store.create("jobs", job)
    known[entry[1]] = job
    return job
