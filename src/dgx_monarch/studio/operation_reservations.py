"""Consume an operation review once, atomically with its dispatch intent."""

from typing import Any

from .contracts import StudioError, bounded_key, digest
from .store import Store


def reserve_review(store: Store, key: Any, request: dict[str, Any],
                   value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    key = bounded_key(key)
    with store.lock, store.transaction():
        existing = store.request("operations", key, request)
        if existing is not None:
            return existing, False
        review = store.get("operations", value["review_id"])
        # The query also protects reviews persisted by versions predating the
        # explicit consumed_by field; an unknown prior execution is not reusable.
        prior = store.db.execute(
            "SELECT id FROM objects WHERE kind='operations' AND json_extract(document,'$.review_id')=? LIMIT 1",
            (review["id"],),
        ).fetchone()
        if review.get("consumed_by") or prior:
            identity = review.get("consumed_by") or prior["id"]
            raise StudioError("operation_consumed", f"This review already dispatched operation {identity}; reconcile it before reviewing another action.", 409)
        if review.get("state") != "review":
            raise StudioError("operation_review", "Apply requires an unconsumed operation review.", 409)
        store._put("operations", {**review, "consumed_by": value["id"]}, 1)
        store.db.execute("INSERT INTO requests VALUES (?,?,?,?)", (key, digest(request), "operations", value["id"]))
        store._put("operations", value, 1)
        store._event("operations.accepted", value["id"], {"state": value["state"]})
    return value, True
