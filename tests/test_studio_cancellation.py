"""Cancellation retries reconcile one durable intent and never repeat a mutator."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from dgx_monarch.studio.cancellation import cancel_job, reserve
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.store import Store


def owner(store, client):
    async def reconcile():
        pass
    return SimpleNamespace(store=store, client=client, reconcile=reconcile)


def job(store, identity="job", state="queued"):
    return store.create("jobs", {"id": identity, "state": state, "prompt_id": "prompt-" + identity,
                                  "submission": {"original": True}})


@pytest.mark.parametrize("atomic", [True, False])
def test_concurrent_cancellation_keys_share_one_driver_dispatch(tmp_path, atomic):
    async def scenario():
        store = Store(tmp_path / "state")
        job(store)
        started, release = asyncio.Event(), asyncio.Event()
        calls = []

        async def dispatch(*args):
            calls.append(args)
            value = store.get("jobs", "job")
            intent = store.get("job_cancellations", value["cancellation_intent_id"])
            assert intent["state"] == "unknown" and intent["prompt_id"] == "prompt-job"
            assert value["state"] == "cancel_requested"
            started.set()
            await release.wait()
            return {"cancelled": True}

        client = SimpleNamespace(**({"cancel_prompt": dispatch} if atomic else {"request": dispatch}))
        selected = owner(store, client)
        request = {"confirmed": True, "idempotency_key": "first"}
        task = asyncio.create_task(cancel_job(selected, "job", request))
        await started.wait()
        try:
            repeated = await cancel_job(selected, "job", request)
            fresh_key = await cancel_job(selected, "job", {**request, "idempotency_key": "second"})
            assert repeated["cancellation_intent_id"] == fresh_key["cancellation_intent_id"]
            assert len(calls) == 1
        finally:
            release.set()
            result = await task
        assert result["cancellation"]["completion"] == "awaiting_driver_evidence"
        assert result["submission"] == {"original": True}
        assert calls == ([("prompt-job",)] if atomic else [("POST", "/queue", {"delete": ["prompt-job"]})])
        store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("outcome", ["network", "cancelled", "malformed"])
def test_ambiguous_cancellation_survives_reopen_without_redispatch(tmp_path, outcome):
    async def scenario():
        store = Store(tmp_path / "state")
        job(store)
        calls = []

        async def dispatch(prompt):
            calls.append(prompt)
            if outcome == "network":
                raise StudioError("driver_timeout", "unknown", 502)
            if outcome == "cancelled":
                raise asyncio.CancelledError
            return None

        selected = owner(store, SimpleNamespace(cancel_prompt=dispatch))
        request = {"confirmed": True, "idempotency_key": "stable"}
        with pytest.raises(asyncio.CancelledError if outcome == "cancelled" else StudioError):
            await cancel_job(selected, "job", request)
        result = store.get("jobs", "job")
        assert result["state"] == "unknown" and result["cancellation"]["prompt_id"] == "prompt-job"
        store.close()
        store = Store(tmp_path / "state")
        selected = owner(store, SimpleNamespace(cancel_prompt=dispatch))
        for key in ("stable", "new"):
            recovered = await cancel_job(selected, "job", {**request, "idempotency_key": key})
            assert recovered["cancellation_intent_id"] == result["cancellation_intent_id"]
        assert calls == ["prompt-job"]
        store.close()

    asyncio.run(scenario())


def test_key_binds_job_and_payload_and_confirmation_is_required(tmp_path):
    async def scenario():
        store = Store(tmp_path / "state")
        job(store, state="succeeded")
        job(store, "other", state="succeeded")
        selected = owner(store, object())
        request = {"confirmed": True, "idempotency_key": "stable"}
        assert (await cancel_job(selected, "job", request))["state"] == "succeeded"
        for identity, payload in (("other", request), ("job", {**request, "extra": "changed"})):
            with pytest.raises(StudioError, match="different request"):
                await cancel_job(selected, identity, payload)
        with pytest.raises(StudioError, match="confirmation"):
            await cancel_job(selected, "job", {"idempotency_key": "stable"})
        store.close()

    asyncio.run(scenario())


def test_legacy_pending_cancellation_and_keyless_calls_never_redispatch(tmp_path):
    async def scenario():
        store = Store(tmp_path / "state")
        job(store, state="cancel_requested")
        selected = owner(store, object())
        first = await cancel_job(selected, "job", {"confirmed": True})
        second = await cancel_job(selected, "job", {"confirmed": True})
        assert first["cancellation_intent_id"] == second["cancellation_intent_id"]
        intent = store.get("job_cancellations", first["cancellation_intent_id"])
        assert intent["dispatch_reserved"] is False and intent["state"] == "unknown"
        store.close()

    asyncio.run(scenario())


def test_nonatomic_driver_cannot_cancel_running_job(tmp_path):
    async def scenario():
        store = Store(tmp_path / "state")
        job(store, state="running")
        with pytest.raises(StudioError, match="atomic prompt cancellation is unavailable"):
            await cancel_job(owner(store, object()), "job", {"confirmed": True})
        assert store.all("job_cancellations") == []
        store.close()

    asyncio.run(scenario())


def test_failed_intent_transaction_does_not_consume_key_or_job(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")
    job(store)

    def fail(*args):
        raise OSError("synthetic event failure")

    monkeypatch.setattr(store, "_event", fail)
    with pytest.raises(OSError, match="event failure"):
        reserve(store, "job", "key", {"confirmed": True}, scoped=True)
    assert store.request("job_cancellations", "key", {"confirmed": True}) is None
    assert "cancellation_intent_id" not in store.get("jobs", "job")
    assert store.all("job_cancellations") == []
    store.close()


def test_distinct_connections_reserve_only_one_dispatch(tmp_path):
    first, second = Store(tmp_path / "state"), Store(tmp_path / "state")
    job(first)
    barrier = threading.Barrier(2)

    def attempt(store, key):
        barrier.wait(timeout=5)
        return reserve(store, "job", key, {"key": key}, scoped=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(attempt, store, key) for store, key in ((first, "one"), (second, "two"))]
        results = [future.result(timeout=10) for future in futures]
    assert sum(result[1] for result in results) == 1
    assert results[0][0]["id"] == results[1][0]["id"]
    first.close()
    second.close()


def test_crash_after_reservation_never_grants_redispatch(tmp_path):
    store = Store(tmp_path / "state")
    job(store)
    intent, dispatch = reserve(store, "job", "original", {"original": True}, scoped=True)
    assert dispatch
    store.close()  # Simulate death before a driver result can be recorded.

    async def scenario():
        store = Store(tmp_path / "state")

        async def forbidden(*args):
            pytest.fail("orphaned reservation granted duplicate cancellation")

        value = await cancel_job(owner(store, SimpleNamespace(cancel_prompt=forbidden)), "job",
                                 {"confirmed": True, "idempotency_key": "new"})
        assert value["cancellation_intent_id"] == intent["id"]
        assert value["cancellation"]["completion"] == "unknown"
        store.close()

    asyncio.run(scenario())
