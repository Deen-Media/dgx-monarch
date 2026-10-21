"""Large history must not make a healthy driver's queue disappear."""
from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.contracts import MAX_BODY_BYTES, StudioError
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store
from test_studio_backend import FakeDriver, job_document


def test_large_history_shrinks_window_and_reconciles_older_known_jobs(tmp_path):
    async def scenario():
        limits = []
        history = {f"old-{index}": {
            "prompt": [index, f"old-{index}", {"1": {"class_type": "Text", "inputs": {"text": "x" * 24000}}}, {}],
            "status": {"completed": True}, "outputs": {},
        } for index in range(200)}
        assert len(json.dumps(history).encode()) > MAX_BODY_BYTES

        async def recent(request):
            limit = int(request.query["max_items"])
            limits.append(limit)
            return web.json_response(dict(list(history.items())[-limit:]))

        async def older(request):
            identity = request.match_info["identity"]
            return web.json_response({identity: history[identity]})

        async def queue(request):
            return web.json_response({"queue_running": [[0, "active", {}, {}]], "queue_pending": []})

        app = web.Application()
        app.router.add_get("/queue", queue)
        app.router.add_get("/history", recent)
        app.router.add_get("/history/{identity}", older)
        with_store = Store(tmp_path / "state")
        async with TestServer(app) as upstream:
            client = ComfyClient(str(upstream.make_url("/")).rstrip("/"))
            await client.start()
            try:
                with_store.create("jobs", {**job_document("active-job"), "state": "queued", "prompt_id": "active"})
                with_store.create("jobs", {**job_document("older-job"), "state": "queued", "prompt_id": "old-0"})
                jobs = Jobs(with_store, client)
                await jobs.reconcile()
                assert limits == [200, 100]
                assert jobs.driver["state"] == "connected"
                assert jobs.driver["history"]["state"] == "connected"
                assert with_store.get("jobs", "active-job")["state"] == "running"
                assert with_store.get("jobs", "older-job")["state"] == "succeeded"
                assert len(with_store.all("jobs")) == 102
                await jobs.reconcile()
                assert limits == [200, 100, 100]
                assert len(with_store.all("jobs")) == 102
            finally:
                await client.close()
                with_store.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["oversized", "malformed", "transport"])
def test_unavailable_recent_history_preserves_queue_and_exact_prompt_recovery(tmp_path, failure):
    finished = {"status": {"completed": True}, "outputs": {"1": {"images": [{"filename": "finished.png"}]}}}

    class Driver(FakeDriver):
        async def request(self, method, path, data=None):
            if path == "/history":
                if failure == "malformed":
                    return []
                raise StudioError("driver_response_limit" if failure == "oversized" else "driver_unknown", "unavailable", 503)
            return await super().request(method, path, data)

        async def history_prompt(self, identity):
            if identity == "lost-prompt":
                raise StudioError("driver_unknown", "lookup lost", 503)
            return {identity: finished} if identity == "completed-prompt" else {}

    store = Store(tmp_path / "state")
    try:
        for identity in ("active", "completed", "restarted", "lost"):
            store.create("jobs", {**job_document(identity), "state": "queued", "prompt_id": identity + "-prompt"})
        driver = Driver()
        driver.queue["queue_running"] = [[0, "active-prompt", {}, {}]]
        jobs = Jobs(store, driver)
        asyncio.run(jobs.reconcile())
        assert jobs.driver["state"] == "connected"
        assert jobs.driver["history"]["state"] == "unavailable"
        assert store.get("jobs", "active")["state"] == "running"
        assert store.get("jobs", "completed")["state"] == "succeeded"
        assert store.all("artifacts")[0]["job_id"] == "completed"
        # An unavailable recent window is not evidence: the exact lookup is.
        restarted = store.get("jobs", "restarted")
        assert restarted["state"] == "unknown" and restarted["absence"]["reason"] == "absent_from_driver"
        lost = store.get("jobs", "lost")
        assert lost["state"] == "queued" and "absence" not in lost
    finally:
        store.close()


@pytest.mark.parametrize("identity", ["", "../unsafe", "x" * 129, None, 7, []])
@pytest.mark.parametrize("evidence", ["queue", "history"])
def test_malformed_acceptance_binds_to_the_companion_identity_and_reconciles_original_intent(tmp_path, identity, evidence):
    class Driver(FakeDriver):
        async def request(self, method, path, data=None):
            if method == "POST" and path == "/prompt":
                self.calls.append((method, path, data))
                # ComfyUI queues under the canonical identity the client sent.
                self.sent = data["prompt_id"]
                if evidence == "queue":
                    self.queue["queue_pending"] = [[0, self.sent, data["prompt"], data["extra_data"]]]
                else:
                    self.finished = {self.sent: {"prompt": [0, self.sent, data["prompt"], data["extra_data"]],
                                                 "status": {"completed": True}, "outputs": {}}}
                return {"prompt_id": identity}
            return await super().request(method, path, data)

        async def history_prompt(self, prompt_id):
            return dict(getattr(self, "finished", {})) if prompt_id == getattr(self, "sent", None) else {}

    async def scenario():
        store = Store(tmp_path / "state")
        try:
            request = {"preview_id": "reviewed", "idempotency_key": "original-request"}
            job, _ = store.reserve("jobs", "original-request", request, {**job_document(), "client_prompt_id": None})
            driver = Driver()
            jobs = Jobs(store, driver)
            await jobs.dispatch(job)
            retained = store.get("jobs", job["id"])
            assert retained["state"] == "unknown" and retained["prompt_id"] is None
            assert retained["error"]["code"] == "submission_unknown"
            minted = retained["client_prompt_id"]
            assert str(uuid.UUID(minted)) == minted and driver.sent == minted
            await jobs.reconcile()
            recovered = jobs.submit(request)
            assert recovered["id"] == job["id"] and recovered["prompt_id"] == minted
            assert recovered["state"] == ("queued" if evidence == "queue" else "succeeded")
            assert len(store.all("jobs")) == 1
            assert sum(method == "POST" for method, _, _ in driver.calls) == 1
        finally:
            store.close()
    asyncio.run(scenario())
