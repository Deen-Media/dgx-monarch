"""Companion boundary regressions with fake drivers; never start a real workload."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch.studio import SCHEMA_VERSION
from dgx_monarch.studio.comfy import ComfyClient, safe_descriptor
from dgx_monarch.studio.contracts import MAX_BODY_BYTES, StudioError
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store


class FakeDriver:
    origin = "http://127.0.0.1:8188"

    def __init__(self):
        self.calls = []
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = {}

    async def start(self):
        pass

    async def close(self):
        pass

    async def request(self, method, path, data=None):
        self.calls.append((method, path, deepcopy(data)))
        if path == "/queue" and method == "GET":
            return self.queue
        if path == "/history":
            return self.history
        if path == "/object_info":
            return {}
        return {"ok": True}


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "state")
    yield result
    result.close()


def job_document(identity="job-one"):
    return {"id": identity, "state": "submitting", "prompt_id": None,
            "submission": {"prompt": {}, "workflow": {}, "digest": "frozen-digest"}}


def test_request_reservation_is_durable_and_payload_bound(store):
    first, created = store.reserve("jobs", "same-key", {"seed": 1}, job_document())
    repeat, repeated = store.reserve("jobs", "same-key", {"seed": 1}, job_document("unused"))
    assert created and not repeated and first == repeat
    assert len(store.list("jobs")) == 1
    with pytest.raises(StudioError, match="different request"):
        store.reserve("jobs", "same-key", {"seed": 2}, job_document("wrong"))
    restored = Store(store.directory)
    try:
        assert restored.request("jobs", "same-key", {"seed": 1}) == first
    finally:
        restored.close()


def test_failed_write_rolls_back_idempotency_reservation(store):
    with pytest.raises(StudioError, match="finite JSON"):
        store.reserve("jobs", "bad-write", {}, {**job_document(), "bad": float("nan")})
    assert store.request("jobs", "bad-write", {}) is None
    assert store.list("jobs") == []


def test_immutable_submissions_terminal_state_and_event_noops(store):
    store.create("jobs", job_document())
    with pytest.raises(StudioError, match="immutable"):
        store.update("jobs", "job-one", {"submission": {}})
    store.update("jobs", "job-one", {"state": "running"})
    assert store.update("jobs", "job-one", {"state": "queued"})["state"] == "running"
    store.update("jobs", "job-one", {"state": "succeeded"})
    watermark = store.events()["latest_cursor"]
    assert store.update("jobs", "job-one", {"state": "failed"})["state"] == "succeeded"
    store.update("jobs", "job-one", {"state": "succeeded"})
    assert store.events()["latest_cursor"] == watermark


def test_restart_does_not_invent_failure_or_redispatch(store):
    store.create("jobs", job_document())
    store.create("operations", {"id": "operation-one", "state": "applying"})
    store.recover()
    assert store.get("jobs", "job-one")["state"] == "unknown"
    assert store.get("operations", "operation-one")["state"] == "unknown"


def test_record_revision_conflict_preserves_newer_edits_and_unknown_metadata(store):
    first = store.save_record("sessions", "workspace", {"extension": {"future": True}}, 0)
    assert first["revision"] == 1
    with pytest.raises(StudioError, match="newer version"):
        store.save_record("sessions", "workspace", {"extension": {}}, 0)
    assert store.get("sessions", "workspace")["data"] == {"extension": {"future": True}}


def test_event_replay_reports_retention_gap_and_ahead_cursor(store, monkeypatch):
    import dgx_monarch.studio.store as storage
    monkeypatch.setattr(storage, "EVENT_LIMIT", 3)
    for revision in range(6):
        store.save_record("sessions", "workspace", {"n": revision}, revision)
    result = store.events(1)
    assert result["reset"] and len(result["events"]) == 3
    assert [event["cursor"] for event in result["events"]] == [4, 5, 6]
    assert store.events(100)["reset"]


def test_delayed_acceptance_cannot_regress_running_job(store):
    job = store.create("jobs", job_document())

    class DelayedDriver(FakeDriver):
        async def request(self, method, path, data=None):
            store.update("jobs", job["id"], {"state": "running", "prompt_id": "prompt-one"})
            return {"prompt_id": "prompt-one"}

    asyncio.run(Jobs(store, DelayedDriver()).dispatch(job))
    assert store.get("jobs", job["id"])["state"] == "running"


def test_lost_acceptance_is_reconciled_by_job_identity_without_duplicate_dispatch(store):
    job = store.create("jobs", job_document())

    class LostResponse(FakeDriver):
        async def request(self, method, path, data=None):
            if method == "POST" and path == "/prompt":
                self.calls.append((method, path, data))
                self.queue["queue_pending"] = [[1, "accepted", {}, {"studio": {"job_id": job["id"], "digest": "frozen-digest"}}]]
                raise StudioError("driver_unknown", "Response was lost", 503)
            return await super().request(method, path, data)

    async def scenario():
        driver = LostResponse()
        jobs = Jobs(store, driver)
        await jobs.dispatch(job)
        assert store.get("jobs", job["id"])["state"] == "unknown"
        await jobs.reconcile()
        reconciled = store.get("jobs", job["id"])
        assert reconciled["state"] == "queued" and reconciled["prompt_id"] == "accepted"
        assert len(store.list("jobs")) == 1
        assert sum(method == "POST" for method, _, _ in driver.calls) == 1
    asyncio.run(scenario())


def test_submit_replay_succeeds_after_original_review_expired(store):
    request = {"preview_id": "expired", "idempotency_key": "accepted-key"}
    accepted, _ = store.reserve("jobs", "accepted-key", request, job_document())
    jobs = Jobs(store, FakeDriver())
    assert jobs.submit(request) == accepted


@pytest.mark.parametrize("extra", [{"studio": "not-a-mapping"}, {"extra_pnginfo": []}, {"studio": None}])
def test_external_metadata_cannot_break_reconciliation(store, extra):
    driver = FakeDriver()
    driver.queue["queue_pending"] = [[1, "external", {}, extra]]
    jobs = Jobs(store, driver)
    asyncio.run(jobs.reconcile())
    assert jobs.driver["state"] == "connected"
    assert store.list("jobs")[0]["prompt_id"] == "external"


def test_running_cancellation_never_sends_global_interrupt(store):
    driver = FakeDriver()
    store.create("jobs", {**job_document(), "state": "running", "prompt_id": "active"})
    driver.queue["queue_running"] = [[1, "active", {}, {}]]

    async def scenario():
        with pytest.raises(StudioError, match="Only queued"):
            await Jobs(store, driver).cancel("job-one", {"confirmed": True})
    asyncio.run(scenario())
    assert not any(method == "POST" for method, _, _ in driver.calls)


def test_unknown_job_cannot_retry(store):
    store.create("jobs", {**job_document(), "state": "unknown"})
    with pytest.raises(StudioError, match="failed or cancelled"):
        Jobs(store, FakeDriver()).retry("job-one", {"confirmed": True, "idempotency_key": "retry"})


@pytest.mark.parametrize("kind", [[], {}, None, 1])
def test_invalid_artifact_store_cannot_break_reconciliation(store, kind):
    driver = FakeDriver()
    driver.history = {"external": {"prompt": [1, "external", {}, {}],
                                  "status": {"completed": True},
                                  "outputs": {"1": {"images": [{"filename": "result.png", "type": kind}]}}}}
    jobs = Jobs(store, driver)
    asyncio.run(jobs.reconcile())
    assert jobs.driver["state"] == "connected"
    assert store.list("jobs")[0]["state"] == "succeeded"
    assert store.list("artifacts") == []


@pytest.mark.parametrize("changed", ["permission", "service"])
def test_persisted_management_review_revalidates_current_authority(store, changed):
    old = Operations(store, FakeDriver(), allow_management=True, driver_service="comfy-first.service")
    plan = old.preview({"action": "driver.stop"})
    current = Operations(store, FakeDriver(), allow_management=changed != "permission",
                         driver_service="comfy-second.service" if changed == "service" else "comfy-first.service")
    with pytest.raises(StudioError) as error:
        current.apply(plan["id"], {"digest": plan["digest"], "confirmed": True, "idempotency_key": "apply"})
    assert error.value.code == ("operation_unavailable" if changed == "permission" else "service_changed")
    assert store.request("operations", "apply", {}) is None


def test_operation_review_detects_changed_config_and_wrong_digest(store, tmp_path):
    config = tmp_path / "cluster.toml"
    config.write_text("# reviewed\n")
    ops = Operations(store, FakeDriver(), config=config, allow_management=True)
    plan = ops.preview({"action": "workers.down"})
    with pytest.raises(StudioError) as error:
        ops.apply(plan["id"], {"digest": "wrong", "confirmed": True, "idempotency_key": "wrong"})
    assert error.value.code == "stale_operation"
    config.write_text("# changed\n")
    with pytest.raises(StudioError) as error:
        ops.apply(plan["id"], {"digest": plan["digest"], "confirmed": True, "idempotency_key": "changed"})
    assert error.value.code == "config_changed"


def test_repeated_operation_apply_executes_exactly_once(store, monkeypatch):
    ops = Operations(store, FakeDriver())
    plan = ops.preview({"action": "version"})
    calls = []

    async def execute(value):
        calls.append(value["id"])
        store.update("operations", value["id"], {"state": "succeeded"})
    monkeypatch.setattr(ops, "execute", execute)

    async def scenario():
        request = {"digest": plan["digest"], "idempotency_key": "once"}
        first = ops.apply(plan["id"], request)
        await asyncio.gather(*ops.tasks)
        store.update("operations", plan["id"], {"expires_at": 0})
        second = ops.apply(plan["id"], request)
        assert first["id"] == second["id"]
        assert calls == [first["id"]]
    asyncio.run(scenario())


@pytest.mark.parametrize("descriptor", [
    {"filename": "../escape.png"}, {"filename": "a.png", "subfolder": "../private"},
    {"filename": "a.png", "subfolder": "/etc"}, {"filename": "a\\b.png"},
    {"filename": "a.png", "subfolder": "a/./b"}, {"filename": "a.png", "type": "arbitrary"},
])
def test_artifact_paths_cannot_escape_fixed_stores(descriptor):
    with pytest.raises(StudioError):
        safe_descriptor(descriptor)


def test_comfy_adapter_reads_complete_chunked_json_and_enforces_budget():
    async def scenario():
        async def chunked(request):
            response = web.StreamResponse(headers={"Content-Type": "application/json"})
            await response.prepare(request)
            await response.write(b'{"queue_running":')
            await asyncio.sleep(0.01)
            await response.write(b'[],"queue_pending":[]}')
            await response.write_eof()
            return response

        async def oversized(request):
            return web.Response(body=b"x" * (MAX_BODY_BYTES + 1))

        app = web.Application()
        app.router.add_get("/queue", chunked)
        app.router.add_get("/history", oversized)
        async with TestServer(app) as upstream:
            client = ComfyClient(str(upstream.make_url("/")).rstrip("/"))
            await client.start()
            try:
                assert await client.request("GET", "/queue") == {"queue_running": [], "queue_pending": []}
                with pytest.raises(StudioError) as error:
                    await client.request("GET", "/history")
                assert error.value.code == "driver_response_limit"
                with pytest.raises(StudioError, match="allowlisted"):
                    await client.request("POST", "/arbitrary")
            finally:
                await client.close()
    asyncio.run(scenario())


def test_http_auth_origin_schema_and_body_boundaries(store):
    async def scenario():
        allowed = set()
        token = "owner-token-" + "a" * 32
        app = create_app(store, FakeDriver(), token, allowed_hosts=allowed, poll=False)
        async with TestClient(TestServer(app)) as browser:
            origin = str(browser.make_url("/")).rstrip("/")
            allowed.add(origin.split("://", 1)[1])
            headers = {"Authorization": "Bearer " + token, "X-DGXM-Schema": SCHEMA_VERSION}
            response = await browser.get("/api/v1/snapshot")
            assert response.status == 401
            response = await browser.get("/api/v1/snapshot", headers={**headers, "Host": "untrusted.example"})
            assert response.status == 403
            response = await browser.get("/api/v1/snapshot", headers={**headers, "Origin": "https://untrusted.example"})
            assert response.status == 403
            response = await browser.get("/api/v1/snapshot", headers={**headers, "X-DGXM-Schema": "999.0"})
            assert response.status == 409
            response = await browser.post("/api/v1/jobs", headers=headers, data="{}")
            assert response.status == 415
            response = await browser.get("/api/v1/snapshot", headers=headers)
            assert response.status == 200
            assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
            response = await browser.put("/api/v1/records/sessions/one", headers=headers,
                                         json={"revision": 0, "data": {"text": "<script>never execute</script>"}})
            assert response.status == 200
            assert (await response.json())["data"]["text"] == "<script>never execute</script>"
            response = await browser.put("/api/v1/records/sessions/one", headers=headers,
                                         json={"revision": 0, "data": {}})
            assert response.status == 409
            response = await browser.put("/api/v1/records/sessions/large", headers=headers,
                                         json={"revision": 0, "data": {"text": "x" * MAX_BODY_BYTES}})
            assert response.status == 413
    asyncio.run(scenario())


def test_cancel_race_is_reconciled_as_running_without_global_interrupt(store):
    store.create("jobs", {**job_document(), "state": "queued", "prompt_id": "race"})

    class StartsDuringDelete(FakeDriver):
        async def request(self, method, path, data=None):
            if method == "POST" and path == "/queue":
                self.calls.append((method, path, data))
                self.queue = {"queue_pending": [], "queue_running": [[1, "race", {}, {}]]}
                return {}
            return await super().request(method, path, data)

    async def scenario():
        driver = StartsDuringDelete()
        driver.queue["queue_pending"] = [[1, "race", {}, {}]]
        jobs = Jobs(store, driver)
        result = await jobs.cancel("job-one", {"confirmed": True})
        assert result["state"] == "cancel_requested"
        await jobs.reconcile()
        assert store.get("jobs", "job-one")["state"] == "running"
        assert [(path, data) for method, path, data in driver.calls if method == "POST"] == [
            ("/queue", {"delete": ["race"]}),
        ]
    asyncio.run(scenario())


def test_authority_is_rechecked_after_waiting_for_operation_lock(store):
    driver = FakeDriver()
    ops = Operations(store, driver, allow_management=True)
    plan = ops.preview({"action": "actors.recycle"})

    async def scenario():
        await ops.execution_lock.acquire()
        result = ops.apply(plan["id"], {"digest": plan["digest"], "confirmed": True, "idempotency_key": "pending"})
        ops.allow_management = False
        ops.execution_lock.release()
        await asyncio.gather(*ops.tasks)
        assert not driver.calls
        assert store.get("operations", result["id"])["state"] == "unknown"
    asyncio.run(scenario())


def test_consent_adapter_sends_fixed_accept_action_and_exact_pending_identity(store):
    driver = FakeDriver()
    ops = Operations(store, driver, allow_management=True)
    plan = ops.preview({"action": "consent.apply", "parameters": {"key": "exact-key", "id": "pending-id"}})

    async def scenario():
        ops.apply(plan["id"], {"digest": plan["digest"], "confirmed": True, "idempotency_key": "consent-once"})
        await asyncio.gather(*ops.tasks)
        assert driver.calls == [("POST", "/dgxm/consent", {"action": "accept", "key": "exact-key", "id": "pending-id"})]
    asyncio.run(scenario())
