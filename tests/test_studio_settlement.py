"""A job the driver no longer holds settles once, from exact evidence; CPU fakes only."""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from copy import deepcopy

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dgx_monarch.studio import job_artifacts, reconcile
from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.contracts import MAX_BODY_BYTES, TERMINAL_STATES, StudioError
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.wire_records import snapshot_window


class Driver:
    """ComfyUI-shaped queue and history held in memory, as the real driver holds them."""

    def __init__(self):
        self.calls = []
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = {}
        self.lookups = []
        self.lookup_failures = set()

    async def request(self, method, path, data=None):
        self.calls.append((method, path, deepcopy(data)))
        if path == "/queue" and method == "GET":
            return deepcopy(self.queue)
        if path == "/queue" and method == "POST":
            for identity in data.get("delete", []):
                self.queue["queue_pending"] = [row for row in self.queue["queue_pending"] if row[1] != identity]
            return {}
        if path == "/history":
            return deepcopy(self.history)
        return {}

    async def history_prompt(self, identity):
        self.lookups.append(identity)
        if identity in self.lookup_failures:
            raise StudioError("driver_unknown", "lookup lost", 503)
        return {identity: deepcopy(self.history[identity])} if identity in self.history else {}

    def restart(self):
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = {}


class ScopedDriver(Driver):
    async def cancel_prompt(self, identity):
        self.calls.append(("CANCEL", identity, None))
        pending = [row for row in self.queue["queue_pending"] if row[1] == identity]
        self.queue["queue_pending"] = [row for row in self.queue["queue_pending"] if row[1] != identity]
        return {"cancelled": bool(pending)}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def saved(store, identity, state, prompt_id, **extra):
    return store.create("jobs", {"id": identity, "state": state, "prompt_id": prompt_id,
                                 "submission": {"prompt": {}, "workflow": {}, "digest": "d"}, **extra})


def completed(prompt_id, filename="out.png"):
    return {"prompt": [1, prompt_id, {}, {}, []], "status": {"completed": True, "status_str": "success", "messages": [
        ["execution_start", {"prompt_id": prompt_id, "timestamp": 1_700_000_000_000}],
        ["execution_success", {"prompt_id": prompt_id, "timestamp": 1_700_000_005_000}]]},
            "outputs": {"9": {"images": [{"filename": filename, "subfolder": "", "type": "output"}]}}}


def test_dequeued_scoped_cancellation_settles_cancelled_with_recorded_basis(store):
    saved(store, "job", "queued", "prompt")

    async def scenario():
        driver = ScopedDriver()
        driver.queue["queue_pending"] = [[1, "prompt", {}, {}]]
        jobs = Jobs(store, driver)
        result = await jobs.cancel("job", {"confirmed": True, "idempotency_key": "cancel"})
        assert result["state"] == "cancel_requested" and result["cancellation"]["reserved_state"] == "queued"
        await jobs.reconcile()
        settled = store.get("jobs", "job")
        assert settled["state"] == "cancelled"
        assert settled["settlement"]["basis"] == "cancel_receipt_and_absent_from_driver"
        assert settled["settlement"]["receipt"] == {"cancelled": True}
        assert settled["absence"]["reason"] == "absent_from_driver"
        intent = store.get("job_cancellations", settled["cancellation_intent_id"])
        assert intent["completion"] == "confirmed_absent_from_driver"
        # Settled jobs leave the per-prompt lookup rotation.
        driver.lookups.clear()
        await jobs.reconcile()
        assert driver.lookups == []
    asyncio.run(scenario())


def test_legacy_queue_delete_settles_cancelled_when_the_prompt_is_gone(store):
    saved(store, "job", "queued", "prompt")

    async def scenario():
        driver = Driver()
        driver.queue["queue_pending"] = [[1, "prompt", {}, {}]]
        jobs = Jobs(store, driver)
        result = await jobs.cancel("job", {"confirmed": True, "idempotency_key": "cancel"})
        assert result["cancellation"]["receipt"] == {"deletion_requested": True}
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "cancelled"
    asyncio.run(scenario())


def test_cancel_of_a_running_prompt_is_not_claimed_from_absence(store):
    saved(store, "job", "running", "prompt")

    async def scenario():
        driver = ScopedDriver()
        driver.queue["queue_running"] = [[1, "prompt", {}, {}]]

        async def interrupt(identity):
            driver.queue["queue_running"] = []
            return {"cancelled": True}
        driver.cancel_prompt = interrupt
        jobs = Jobs(store, driver)
        await jobs.cancel("job", {"confirmed": True, "idempotency_key": "cancel"})
        # History was lost (for example a restart) before the interrupt reached it.
        await jobs.reconcile()
        settled = store.get("jobs", "job")
        assert settled["state"] == "unknown" and settled["absence"]["reason"] == "absent_from_driver"
        assert "settlement" not in settled
    asyncio.run(scenario())


def test_cancellation_still_in_flight_is_never_settled_from_absence(store):
    saved(store, "job", "queued", "prompt")

    async def scenario():
        driver = ScopedDriver()
        driver.queue["queue_pending"] = [[1, "prompt", {}, {}]]
        jobs = Jobs(store, driver)
        started, release = asyncio.Event(), asyncio.Event()

        async def slow(identity):
            driver.queue["queue_pending"] = []
            started.set()
            await release.wait()
            return {"cancelled": True}
        driver.cancel_prompt = slow
        task = asyncio.create_task(jobs.cancel("job", {"confirmed": True, "idempotency_key": "cancel"}))
        await started.wait()
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "cancel_requested"
        release.set()
        await task
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "cancelled"
    asyncio.run(scenario())


def test_driver_restart_settles_inflight_jobs_unknown_once(store):
    for identity, state in (("queued-job", "queued"), ("running-job", "running"), ("recovered-job", "unknown")):
        saved(store, identity, state, "prompt-" + identity)

    async def scenario():
        driver = Driver()
        jobs = Jobs(store, driver)
        driver.restart()
        await jobs.reconcile()
        for identity in ("queued-job", "running-job", "recovered-job"):
            value = store.get("jobs", identity)
            assert value["state"] == "unknown", identity
            assert value["absence"]["reason"] == "absent_from_driver"
            assert value["absence"]["basis"] == "queue_snapshot_and_prompt_lookup"
        assert sorted(driver.lookups) == ["prompt-queued-job", "prompt-recovered-job", "prompt-running-job"]
        driver.lookups.clear()
        await jobs.reconcile()
        assert driver.lookups == []
        assert jobs.driver["state"] == "connected"
    asyncio.run(scenario())


def test_history_clear_settles_running_job_and_retires_unfetchable_outputs(store):
    saved(store, "running", "running", "prompt-running")
    saved(store, "finished", "succeeded", "prompt-finished")  # Terminal from an event, never reconciled.

    async def scenario():
        driver = Driver()
        jobs = Jobs(store, driver)
        await jobs.reconcile()
        assert store.get("jobs", "running")["state"] == "unknown"
        finished = store.get("jobs", "finished")
        assert finished["state"] == "succeeded" and finished["absence"]["reason"] == "absent_from_driver"
        driver.lookups.clear()
        await jobs.reconcile()
        assert driver.lookups == []
    asyncio.run(scenario())


@pytest.mark.parametrize("answer", ["failed", "malformed"])
def test_failed_or_malformed_lookup_is_not_evidence_of_absence(store, answer):
    saved(store, "job", "queued", "prompt")

    async def scenario():
        driver = Driver()
        if answer == "failed":
            driver.lookup_failures.add("prompt")
        else:
            async def malformed(identity):
                driver.lookups.append(identity)
                return []
            driver.history_prompt = malformed
        jobs = Jobs(store, driver)
        await jobs.reconcile()
        await jobs.reconcile()
        value = store.get("jobs", "job")
        assert value["state"] == "queued" and "absence" not in value
        assert driver.lookups == ["prompt", "prompt"]
    asyncio.run(scenario())


def test_queue_failure_never_settles_absence(store):
    saved(store, "job", "queued", "prompt")

    class Offline(Driver):
        async def request(self, method, path, data=None):
            if path == "/queue":
                raise StudioError("driver_unknown", "queue lost", 503)
            return await super().request(method, path, data)

    async def scenario():
        driver = Offline()
        jobs = Jobs(store, driver)
        await jobs.reconcile()
        assert jobs.driver["state"] == "unavailable" and jobs.driver["code"] == "driver_unknown"
        assert isinstance(jobs.driver["since"], float)
        assert store.get("jobs", "job")["state"] == "queued"
    asyncio.run(scenario())


def test_absent_job_revives_when_the_driver_lists_it_again(store):
    saved(store, "job", "queued", "prompt")

    async def scenario():
        driver = Driver()
        jobs = Jobs(store, driver)
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "unknown"
        driver.queue["queue_running"] = [[1, "prompt", {}, {}]]
        await jobs.reconcile()
        value = store.get("jobs", "job")
        assert value["state"] == "running" and value.get("absence") is None
        driver.queue["queue_running"] = []
        driver.history["prompt"] = completed("prompt")
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "succeeded"
    asyncio.run(scenario())


def test_submitting_job_is_never_settled_from_absence(store):
    saved(store, "job", "submitting", None, client_prompt_id="6f1c9a52-0d43-4a8e-9a33-3bd1c1f1a4a2")

    async def scenario():
        driver = Driver()
        jobs = Jobs(store, driver)
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "submitting"
        assert driver.lookups == []
    asyncio.run(scenario())


def test_state_written_after_the_identity_read_is_left_for_the_next_cycle(store):
    saved(store, "job", "queued", "prompt")

    class Racing(Driver):
        async def request(self, method, path, data=None):
            if path == "/queue" and method == "GET":
                # A late acknowledgement or event lands between the identity read and the snapshot.
                store.update("jobs", "job", {"state": "running"})
            return await super().request(method, path, data)

    async def scenario():
        jobs = Jobs(store, Racing())
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "running"
        assert "absence" not in store.get("jobs", "job")
    asyncio.run(scenario())


def test_oversized_queue_keeps_driver_connected_and_history_reconciling(tmp_path):
    async def scenario():
        history = {"finished": completed("finished")}

        async def queue(request):
            pending = [[index, f"pending-{index}", {"1": {"class_type": "Text", "inputs": {"text": "x" * 24000}}}, {}]
                       for index in range(200)]
            return web.json_response({"queue_running": [], "queue_pending": pending})

        async def recent(request):
            return web.json_response({})

        async def older(request):
            identity = request.match_info["identity"]
            return web.json_response({identity: history[identity]} if identity in history else {})

        app = web.Application()
        app.router.add_get("/queue", queue)
        app.router.add_get("/history", recent)
        app.router.add_get("/history/{identity}", older)
        store = Store(tmp_path / "state")
        async with TestServer(app) as upstream:
            client = ComfyClient(str(upstream.make_url("/")).rstrip("/"))
            await client.start()
            try:
                saved(store, "finished-job", "queued", "finished")
                saved(store, "pending-job", "queued", "pending-3")
                jobs = Jobs(store, client)
                await jobs.reconcile()
                assert jobs.driver["state"] == "connected"
                assert jobs.driver["queue"] == {"state": "over_budget", "code": "driver_response_limit"}
                assert store.get("jobs", "finished-job")["state"] == "succeeded"
                assert len(store.all("artifacts")) == 1
                # Without a queue snapshot nothing is absent, even after an empty lookup.
                pending = store.get("jobs", "pending-job")
                assert pending["state"] == "queued" and "absence" not in pending
            finally:
                await client.close()
                store.close()
    assert len(json.dumps([["x" * 24000]] * 200)) > MAX_BODY_BYTES
    asyncio.run(scenario())


def test_one_unreadable_row_is_quarantined_while_others_reconcile(store):
    saved(store, "job", "queued", "prompt")

    async def scenario():
        driver = Driver()
        driver.queue["queue_pending"] = [[1, "foreign-queued", {"1": {"inputs": {"cfg": float("nan")}}}, {}]]
        bad = completed("foreign-done")
        bad["prompt"][2] = {"1": {"inputs": {"cfg": float("inf")}}}
        bad["status"]["messages"][0][1]["value"] = float("nan")
        driver.history = {"foreign-done": bad, "prompt": completed("prompt")}
        jobs = Jobs(store, driver)
        await jobs.reconcile()
        assert jobs.driver["state"] == "connected"
        assert store.get("jobs", "job")["state"] == "succeeded"
        assert [row["job_id"] for row in store.all("artifacts")] == ["job"]
        quarantine = jobs.driver["quarantine"]
        assert quarantine["count"] == 2
        assert {(row["source"], row["prompt_id"], row["code"]) for row in quarantine["rows"]} == {
            ("queue", "foreign-queued", "invalid_json"), ("history", "foreign-done", "invalid_json")}
    asyncio.run(scenario())


def test_unchanged_history_is_skipped_and_events_use_the_prompt_index(store, monkeypatch):
    saved(store, "job", "queued", "prompt")
    driver = Driver()
    driver.history = {"prompt": completed("prompt")}
    jobs = Jobs(store, driver)
    asyncio.run(jobs.reconcile())
    assert store.get("jobs", "job")["state"] == "succeeded"
    calls = {"update": 0, "get": 0, "import": 0}
    for name in ("update", "get"):
        original = getattr(store, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(store, name, counted)
    imported = job_artifacts.import_artifacts

    def counted_import(*args, **kwargs):
        calls["import"] += 1
        return imported(*args, **kwargs)
    monkeypatch.setattr(reconcile, "import_artifacts", counted_import)
    for _ in range(3):
        asyncio.run(jobs.reconcile())
    assert calls == {"update": 0, "get": 0, "import": 0}
    # A changed entry is settled again, once.
    driver.history["prompt"]["outputs"]["9"]["images"].append({"filename": "second.png", "type": "output"})
    asyncio.run(jobs.reconcile())
    asyncio.run(jobs.reconcile())
    assert calls["import"] == 1 and len(store.all("artifacts")) == 2
    monkeypatch.setattr(store, "jobs_by_prompt", lambda: pytest.fail("Events must use the prompt index"))
    saved(store, "live", "queued", "live-prompt")
    jobs.consume_event({"type": "execution_start", "data": {"prompt_id": "live-prompt", "timestamp": 5}})
    assert store.get("jobs", "live")["state"] == "running"


def test_artifact_time_is_the_driver_completion_time_with_its_source(store):
    saved(store, "job", "queued", "prompt")
    driver = Driver()
    driver.history = {"prompt": completed("prompt")}
    asyncio.run(Jobs(store, driver).reconcile())
    artifact = store.all("artifacts")[0]
    assert artifact["created_at"] == 1_700_000_005.0
    assert artifact["created_at_source"] == "driver_completion"
    assert artifact["observed_at"] > artifact["created_at"]
    assert artifact["provenance"]["prompt_id"] == "prompt"
    saved(store, "other", "queued", "other-prompt")
    entry = completed("other-prompt", "other.png")
    entry["status"]["messages"] = []
    driver.history = {"other-prompt": entry}
    asyncio.run(Jobs(store, driver).reconcile())
    other = next(row for row in store.all("artifacts") if row["job_id"] == "other")
    assert other["created_at_source"] == "companion_observation" and other["created_at"] == other["observed_at"]


def test_no_snapshot_lists_a_settled_job_before_its_outputs(store, monkeypatch):
    """Settlement runs on a worker thread, and the snapshot route reads between its commits.

    A client that sees a job succeed reads its outputs from that same snapshot,
    so settlement must commit the artifacts before the terminal state.
    """
    saved(store, "job", "running", "prompt")
    driver = Driver()
    driver.history = {"prompt": completed("prompt")}
    begin, views = store.transaction, []

    @contextmanager
    def observed():
        with begin():
            yield
        rows, _omitted, _cursor = snapshot_window(store)
        views.append(({row["id"]: row["state"] for row in rows["jobs"]}, [row["job_id"] for row in rows["artifacts"]]))
    monkeypatch.setattr(store, "transaction", observed)
    asyncio.run(Jobs(store, driver).reconcile())
    early = [view for view in views if view[0].get("job") in TERMINAL_STATES and "job" not in view[1]]
    assert not early, views
    assert views[-1] == ({"job": "succeeded"}, ["job"])


def test_driver_history_window_is_reported(store):
    driver = Driver()
    driver.history_limit = 50
    jobs = Jobs(store, driver)
    asyncio.run(jobs.reconcile())
    assert jobs.driver["history"] == {"state": "connected", "limit": 50}


def test_unbound_identity_lookups_use_the_prompt_index(store):
    # Events for a prompt no job has bound yet fall back to the companion's identity;
    # that fallback and the per-cycle identity read must not scan every job.
    statements = []
    store.db.set_trace_callback(statements.append)
    try:
        reconcile.known_jobs(store)
        assert reconcile.job_for_prompt(store, "never-bound") is None
    finally:
        store.db.set_trace_callback(None)
    unbound = [sql for sql in statements if "IS NULL" in sql]
    assert len(unbound) == 2
    for sql in unbound:
        plan = " ".join(row[3] for row in store.db.execute("EXPLAIN QUERY PLAN " + sql))
        assert "USING INDEX jobs_by_prompt" in plan, plan
