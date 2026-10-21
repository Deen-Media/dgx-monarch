"""Durable, single-dispatch cancellation intent for one driver graph task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .contracts import TERMINAL_STATES, StudioError, bounded_key, digest, identifier, timestamp
from .store import Store

if TYPE_CHECKING:
    from .jobs import Jobs

_RESERVED_FIELDS = ("cancellation_intent_id", "cancellation_scope", "cancellation")


def reserve(store: Store, job_id: str, key: str, request: dict[str, Any], *, scoped: bool) -> tuple[dict[str, Any], bool]:
    with store.lock, store.transaction():
        existing = store.request("job_cancellations", key, request)
        if existing is not None:
            return existing, False
        job = store.get("jobs", job_id)
        if job.get("cancellation_intent_id"):
            intent = store.get("job_cancellations", job["cancellation_intent_id"])
            store.db.execute("INSERT INTO requests VALUES (?,?,?,?)", (key, digest(request), "job_cancellations", intent["id"]))
            return intent, False
        terminal = job["state"] in TERMINAL_STATES
        prior_request = job["state"] == "cancel_requested" or bool(job.get("cancellation"))
        dispatch = not terminal and not prior_request
        if dispatch:
            if not job.get("prompt_id") or job["state"] not in {"queued", "running"}:
                raise StudioError("cancellation_scope", "Only queued graph tasks can be cancelled safely; running driver interruption is global.", 409)
            if job["state"] != "queued" and not scoped:
                raise StudioError("cancellation_scope", "Only queued graph tasks are supported by this driver adapter; atomic prompt cancellation is unavailable.", 409)
        scope = "atomic_prompt" if scoped else "queued_only"
        # The state at reservation decides what a later absence proves: a
        # dequeued prompt never reaches history, an interrupted one does.
        intent = {"id": identifier(), "job_id": job_id, "prompt_id": job.get("prompt_id"),
                  "state": "unknown" if dispatch or prior_request else "not_needed", "created_at": timestamp(),
                  "scope": scope, "dispatch_reserved": dispatch, "reserved_state": job["state"]}
        value = {**job, "cancellation_intent_id": intent["id"]}
        if dispatch:
            value.update(state="cancel_requested", cancellation_scope=scope,
                         cancellation={"intent_id": intent["id"], "prompt_id": intent["prompt_id"], "completion": "unknown",
                                       "reserved_state": job["state"]})
        store.db.execute("INSERT INTO requests VALUES (?,?,?,?)", (key, digest(request), "job_cancellations", intent["id"]))
        store._put("job_cancellations", intent, 1)
        store._put("jobs", value, 1)
        store._event("jobs.updated", job_id, {"state": value["state"], "cancellation_intent_id": intent["id"]})
    return intent, dispatch


def release(store: Store, job_id: str, key: str, intent: dict[str, Any]) -> None:
    """Undo a reservation the driver never received, so the key and the job's one intent stay unspent."""
    with store.lock, store.transaction():
        job = store.get("jobs", job_id)
        store.db.execute("DELETE FROM requests WHERE key=? AND kind='job_cancellations' AND entity_id=?", (key, intent["id"]))
        store.db.execute("DELETE FROM objects WHERE kind='job_cancellations' AND id=?", (intent["id"],))
        if job.get("cancellation_intent_id") == intent["id"]:
            value = {name: item for name, item in job.items() if name not in _RESERVED_FIELDS}
            if job["state"] == "cancel_requested":
                value["state"] = intent["reserved_state"]
            store._put("jobs", {**value, "updated_at": timestamp()}, 1)
            store._event("jobs.updated", job_id, {"state": value["state"]})


async def _scoped_or_queue_delete(owner: Jobs, scoped_cancel: Any, job_id: str, key: str,
                                  intent: dict[str, Any]) -> dict[str, Any]:
    try:
        receipt: dict[str, Any] = await scoped_cancel(intent["prompt_id"])
        return receipt
    except StudioError as exc:
        if getattr(exc, "upstream_status", None) != 404:
            raise
    # An older driver has no prompt-specific route. Queue deletion is safe only
    # for a queued prompt; a running one keeps its unspent intent and key.
    owner.scoped_cancel_unavailable = True
    if intent.get("reserved_state") != "queued":
        release(owner.store, job_id, key, intent)
        raise StudioError("cancellation_scope", "This driver has no prompt-specific cancellation route; "
                          "only queued graph tasks can be cancelled.", 409)
    await owner.client.request("POST", "/queue", {"delete": [intent["prompt_id"]]})
    owner.store.update("job_cancellations", intent["id"], {"scope": "queued_only"})
    owner.store.update("jobs", job_id, {"cancellation_scope": "queued_only"})
    return {"deletion_requested": True, "scoped_route": "unavailable"}


async def cancel_job(owner: Jobs, job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    if request.get("confirmed") is not True:
        raise StudioError("confirmation", "Explicit cancellation confirmation is required.", 409)
    # Older direct Python callers had no key. Their deterministic key retains
    # exactly the same single-intent protection; HTTP callers supply their key.
    key = bounded_key(request["idempotency_key"] if "idempotency_key" in request else "cancel-" + digest(job_id))
    binding = {**request, "job_id": job_id, "action": "cancel"}
    existing = owner.store.request("job_cancellations", key, binding)
    if existing is not None:
        return owner.store.get("jobs", job_id)
    await owner.reconcile()
    scoped_cancel = getattr(owner.client, "cancel_prompt", None)
    if getattr(owner, "scoped_cancel_unavailable", False):
        scoped_cancel = None
    intent, dispatch = reserve(owner.store, job_id, key, binding, scoped=callable(scoped_cancel))
    if not dispatch:
        return owner.store.get("jobs", job_id)
    # Reconciliation must not settle the job from an absence while its
    # cancellation is still on the way to the driver.
    flights = getattr(owner, "cancel_dispatches", None)
    if isinstance(flights, set):
        flights.add(job_id)
    try:
        if callable(scoped_cancel):
            receipt = await _scoped_or_queue_delete(owner, scoped_cancel, job_id, key, intent)
        else:
            await owner.client.request("POST", "/queue", {"delete": [intent["prompt_id"]]})
            receipt = {"deletion_requested": True}
        if not isinstance(receipt, dict):
            raise StudioError("cancellation_receipt", "Driver cancellation acknowledgement is malformed; reconcile before retrying.", 502)
        completion = "not_dispatched" if receipt.get("cancelled") is False else "awaiting_driver_evidence"
        owner.store.update("job_cancellations", intent["id"], {"state": "acknowledged", "receipt": receipt,
                                                               "completion": completion, "observed_at": timestamp()})
        updates: dict[str, Any] = {"cancellation": {"intent_id": intent["id"], "prompt_id": intent["prompt_id"],
                                                    "receipt": receipt, "observed_at": timestamp(), "completion": completion,
                                                    "reserved_state": intent.get("reserved_state")}}
        if receipt.get("cancelled") is False:
            updates["state"] = "unknown"
        owner.store.update("jobs", job_id, updates)
    except BaseException as exc:
        if isinstance(exc, StudioError) and exc.code == "cancellation_scope" and owner.store.request(
                "job_cancellations", key, binding) is None:
            raise  # Released above: nothing reached the driver.
        # The durable reservation remains authoritative even when the caller
        # dies before receiving any acknowledgement from the driver.
        code = exc.code if isinstance(exc, StudioError) else "cancellation_interrupted"
        owner.store.update("job_cancellations", intent["id"], {"state": "unknown", "error": code})
        owner.store.update("jobs", job_id, {"state": "unknown", "cancellation": {
            "intent_id": intent["id"], "prompt_id": intent["prompt_id"], "code": code, "completion": "unknown",
            "reserved_state": intent.get("reserved_state")}})
        raise
    finally:
        if isinstance(flights, set):
            flights.discard(job_id)
    return owner.store.get("jobs", job_id)
