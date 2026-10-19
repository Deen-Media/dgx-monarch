"""A reviewed action is single-use even across racing callers and restarts."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.operation_reservations import reserve_review
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store


def test_racing_different_keys_consume_one_review(tmp_path):
    first = Store(tmp_path / "state")
    second = Store(tmp_path / "state")
    first.create("operations", {"id": "review", "state": "review"})
    barrier = threading.Barrier(2)

    def reserve(store, key):
        barrier.wait(timeout=5)
        try:
            return reserve_review(store, key, {"operation_id": "review"},
                                  {"id": key, "review_id": "review", "state": "applying"})
        except StudioError as exc:
            assert exc.code == "operation_consumed"
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(reserve, store, key) for store, key in ((first, "one"), (second, "two"))]
        results = [future.result(timeout=10) for future in futures]
    assert sum(result is not None for result in results) == 1
    execution = next(result[0] for result in results if result)
    assert first.get("operations", "review")["consumed_by"] == execution["id"]
    assert len(first.all("operations")) == 2
    first.close()
    second.close()


def test_failed_reservation_rolls_back_review_consumption(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")
    store.create("operations", {"id": "review", "state": "review"})
    event = store._event

    def fail(*args):
        raise OSError("synthetic event publication failure")

    monkeypatch.setattr(store, "_event", fail)
    request = {"operation_id": "review"}
    value = {"id": "execution", "review_id": "review", "state": "applying"}
    with pytest.raises(OSError, match="publication failure"):
        reserve_review(store, "key", request, value)
    assert "consumed_by" not in store.get("operations", "review")
    assert store.request("operations", "key", request) is None
    assert len(store.all("operations")) == 1
    monkeypatch.setattr(store, "_event", event)
    assert reserve_review(store, "key", request, value) == (value, True)
    store.close()


@pytest.mark.parametrize("legacy_review", [False, True])
def test_unknown_action_reconciles_same_key_and_rejects_new_key(tmp_path, legacy_review):
    async def scenario():
        store = Store(tmp_path / "state")
        operations = Operations(store, object(), allow_management=True)
        review = operations.preview({"action": "driver.free"})
        dispatched = []

        async def unknown(operation):
            dispatched.append(operation["id"])
            store.update("operations", operation["id"], {"state": "unknown"})

        operations.execute = unknown
        request = {"idempotency_key": "original", "digest": review["digest"], "confirmed": True}
        accepted = operations.apply(review["id"], request)
        await asyncio.gather(*operations.tasks)
        if legacy_review:
            store.update("operations", review["id"], {"consumed_by": None})
        store.close()

        store = Store(tmp_path / "state")
        operations = Operations(store, object(), allow_management=True)
        existing = operations.apply(review["id"], request)
        assert existing["id"] == accepted["id"] and existing["state"] == "unknown"
        with pytest.raises(StudioError, match="already dispatched"):
            operations.apply(review["id"], {**request, "idempotency_key": "new"})
        with pytest.raises(StudioError, match="unconsumed operation review"):
            operations.apply(accepted["id"], {**request, "idempotency_key": "clone"})
        assert len(dispatched) == 1 and not operations.tasks
        store.close()

    asyncio.run(scenario())
