"""HTTP boundary regressions using only local in-process protocol fixtures."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from test_studio_integration import PNG, companion


@pytest.fixture(autouse=True)
def no_management_processes(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("HTTP quality checks must not run management processes")
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)


def preview_request(kind):
    workflow = json.loads((Path(__file__).parent / "fixtures/studio/app_mode_canonical.json").read_text())
    draft = {"workflow": workflow, "values": {"main:1:text": "quality check", "main:2:seed": 42}, "seed_policy": "fixed"}
    return draft if kind == "jobs" else {"draft": draft, "axes": [{"name": "main:2:steps", "values": [2, 3]}]}


@pytest.mark.parametrize("authorization", ["Bearer snowman-☃", "Bearer 所有者"])
def test_unicode_authorization_is_a_clean_401(tmp_path, monkeypatch, authorization):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            response = await system.browser.get("/api/v1/capabilities", headers={"Authorization": authorization})
            assert response.status == 401
            assert (await response.json())["error"]["code"] == "authentication"
            assert response.headers["Cache-Control"] == "no-store"
    asyncio.run(scenario())


@pytest.mark.parametrize("chunked", [False, True])
def test_upstream_stream_has_boundary_headers_on_wire(tmp_path, monkeypatch, chunked):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            system.protocol.chunked_media = chunked
            system.store.create("artifacts", {"id": "media", "filename": "result.png", "subfolder": "studio", "type": "output"})
            response = await system.browser.get("/api/v1/artifacts/media/file", headers=system.headers)
            assert response.status == 200 and await response.read() == PNG
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["Referrer-Policy"] == "no-referrer"
            policy = response.headers["Content-Security-Policy"]
            assert "object-src 'none'" in policy and "frame-ancestors 'none'" in policy
            assert system.protocol.media_requests == [{"filename": "result.png", "subfolder": "studio", "type": "output"}]
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["jobs", "grids"])
def test_new_submission_refreshes_schema_and_refuses_stale_review(tmp_path, monkeypatch, kind):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            preview = await system.request("POST", f"/{kind}/preview", json=preview_request(kind))
            system.protocol.schema["Sampler"]["input"]["required"]["steps"][1]["max"] = 99
            body = await system.request("POST", f"/{kind}", expected=409,
                                        json={"preview_id": preview["id"], "idempotency_key": "stale-review"})
            assert body["error"]["code"] == "schema_changed"
            assert system.protocol.submissions == []
            assert system.store.all("jobs") == []
            await system.request("GET", "/jobs/request/stale-review", expected=404)
    asyncio.run(scenario())


@pytest.mark.parametrize("kind", ["jobs", "grids"])
def test_accepted_submission_replays_while_driver_is_offline(tmp_path, monkeypatch, kind):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            preview = await system.request("POST", f"/{kind}/preview", json=preview_request(kind))
            request = {"preview_id": preview["id"], "idempotency_key": "accepted-once"}
            accepted = await system.request("POST", f"/{kind}", expected=202, json=request)
            identities = [accepted["id"]] if kind == "jobs" else [cell["job_id"] for cell in accepted["cells"]]
            for identity in identities:
                await system.store.wait_state("jobs", identity, "queued")
            assert len(system.protocol.submissions) == len(identities)
            await system.upstream.close()
            replay = await system.request("POST", f"/{kind}", expected=202, json=request)
            assert replay["id"] == accepted["id"]
            assert len(system.protocol.submissions) == len(identities)
            assert len(system.store.all("jobs")) == len(identities)
            conflict = await system.request("POST", f"/{kind}", expected=409,
                                            json={**request, "preview_id": "different"})
            assert conflict["error"]["code"] == "idempotency_conflict"
    asyncio.run(scenario())


def test_snapshot_is_summary_while_operation_endpoint_retains_receipt(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            receipt = {"source": "synthetic", "deep": {"audit": "durable detail", "values": list(range(100))}}
            operation = system.store.create("operations", {"id": "retained", "action": "version", "state": "succeeded",
                                                           "receipt": receipt, "adapter_plan": {"full": "plan evidence"}})
            system.store.create("jobs", {"id": "job", "state": "succeeded", "submission": {
                "prompt": {"large": "full prompt"}, "requested": {"operation": "image", "workflow": {"full": True}},
                "project_id": "project", "digest": "identity"}})
            snapshot = await system.request("GET", "/snapshot")
            brief = next(row for row in snapshot["operations"] if row["id"] == operation["id"])
            assert brief["summary"] is True and "receipt" not in brief and "adapter_plan" not in brief
            job = snapshot["jobs"][0]
            assert job["summary"] is True and "prompt" not in job["submission"]
            assert "workflow" not in job["submission"]["requested"]
            detail = await system.request("GET", "/operations/retained")
            assert detail["receipt"] == receipt and detail["adapter_plan"] == operation["adapter_plan"]
            full_job = await system.request("GET", "/jobs/job")
            assert full_job["submission"]["prompt"] == {"large": "full prompt"}
    asyncio.run(scenario())
