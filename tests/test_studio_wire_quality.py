"""History stays bounded without discarding full provenance or prompt identity."""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from dgx_monarch.studio.contracts import StudioError, canonical
from dgx_monarch.studio.history import page_artifacts, page_jobs
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.wire_records import PAGE_BYTES, snapshot_rows


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def job(identity="original", **changes):
    return {"id": identity, "state": "queued", "prompt_id": "accepted", "created_at": 1,
            "submission": {"digest": "reviewed", "workflow": {"extra": "x" * 100_000},
                           "prompt": {"1": {"class_type": "Test", "inputs": {}}},
                           "requested": {"operation": "fleet", "project_id": "p", "values": {"blob": "x" * 100_000}},
                           "project_id": "p", "versions": {"dgx_monarch": "test"}, "model_identities": []},
            "monarch": {"schema_version": 1, "event_count": 12, "render_count": 1,
                        "render_status_counts": {"succeeded": 1},
                        "renders": {"render-1": {"fleet_index": 2, "status": "succeeded", "large": "x" * 100_000}}},
            **changes}


def test_job_wire_summary_retains_controls_but_fetch_does_not_lose_provenance(store, monkeypatch):
    original = store.create("jobs", job())
    monkeypatch.setattr(store, "all", lambda *_: pytest.fail("List paths must not load all full jobs"))
    item = page_jobs(store, {})["items"][0]
    assert item["summary"] is True and item["submission"]["requested"]["operation"] == "fleet"
    assert item["submission"]["project_id"] == "p"
    assert item["monarch"]["renders"] == {"render-1": {"fleet_index": 2, "status": "succeeded"}}
    assert "workflow" not in item["submission"] and len(canonical(item)) < 2000
    assert store.get("jobs", "original") == original
    assert store.jobs_by_prompt()["accepted"]["id"] == "original"


def test_snapshot_uses_record_and_operation_summaries_and_prioritizes_active_jobs(store):
    store.create("jobs", job("active", state="running"))
    for number in range(205):
        store.create("jobs", job(f"ended-{number}", state="succeeded", prompt_id=f"p{number}"))
    store.save_record("workflows", "large", {"name": "Saved graph", "workflow": {"text": "x" * 100_000}}, 0)
    store.create("operations", {"id": "op", "action": "version", "state": "succeeded", "result": {"text": "x" * 100_000}})
    rows, omitted = snapshot_rows(store)
    assert rows["jobs"][0]["id"] == "active"
    assert omitted["jobs"] == 6
    assert rows["workflows"][0]["data"]["name"] == "Saved graph"
    assert "workflow" not in rows["workflows"][0]["data"]
    assert rows["operations"][0]["summary"] and "result" not in rows["operations"][0]
    assert len(canonical(rows).encode()) < PAGE_BYTES


def test_artifact_pages_bound_encoded_unicode_bytes_and_keep_cursor_complete(store):
    for number in range(10):
        store.create("artifacts", {"id": str(number), "created_at": number,
                                   "search_text": "hidden searchable source " + "z" * 100_000,
                                   "metadata": {"extension": "\u2603" * 160_000}})
    query = {"limit": 200, "q": "searchable"}
    found = []
    while True:
        page = page_artifacts(store, query)
        assert len(canonical(page).encode()) <= PAGE_BYTES
        assert all("search_text" not in item for item in page["items"])
        found.extend(item["id"] for item in page["items"])
        if page["next_cursor"] is None:
            break
        query["cursor"] = page["next_cursor"]
    assert len(found) == len(set(found)) == 10
    assert "search_text" in store.get("artifacts", "0")


def test_legacy_job_without_timestamp_has_a_usable_cursor(store):
    for identity in ("a", "b"):
        store.create("jobs", {"id": identity, "state": "succeeded", "submission": {}})
    first = page_jobs(store, {"limit": 1})
    second = page_jobs(store, {"limit": 1, "cursor": first["next_cursor"]})
    assert [item["id"] for item in first["items"] + second["items"]] == ["b", "a"]
    assert second["next_cursor"] is None


def test_replayed_driver_metadata_cannot_rebind_existing_job(store, monkeypatch):
    original = store.create("jobs", job())

    class Driver:
        async def request(self, method, path, data=None):
            if path == "/queue":
                return {"queue_running": [], "queue_pending": [[1, "duplicate", original["submission"]["prompt"],
                         {"studio": {"job_id": "original", "digest": "reviewed"}}]]}
            return {}

    monkeypatch.setattr(store, "all", lambda *_: pytest.fail("Reconciliation must use prompt projections"))
    asyncio.run(Jobs(store, Driver()).reconcile())
    assert store.get("jobs", "original")["prompt_id"] == "accepted"
    duplicate = store.jobs_by_prompt()["duplicate"]
    assert duplicate["id"] != "original" and store.get("jobs", duplicate["id"])["external"]
    with pytest.raises(StudioError, match="rebound"):
        store.update("jobs", "original", {"prompt_id": "other"})


def test_history_cannot_join_conflicting_envelope_identity(store):
    original = store.create("jobs", job())

    class Driver:
        async def request(self, method, path, data=None):
            if path == "/queue":
                return {"queue_running": [], "queue_pending": []}
            if path == "/history":
                return {"wrong-envelope": {"prompt": [0, "accepted", original["submission"]["prompt"], {}],
                                            "status": {"completed": True}, "outputs": {}}}
            return {}

    asyncio.run(Jobs(store, Driver()).reconcile())
    assert store.get("jobs", "original") == original


def test_late_dispatch_ack_cannot_overwrite_reconciled_identity(store):
    original = store.create("jobs", job(prompt_id=None, state="submitting"))

    class Driver:
        async def request(self, method, path, data=None):
            if path == "/prompt":
                store.update("jobs", "original", {"prompt_id": "accepted", "state": "running"})
                return {"prompt_id": "different"}
            return {}

    asyncio.run(Jobs(store, Driver()).dispatch(deepcopy(original)))
    value = store.get("jobs", "original")
    assert value["prompt_id"] == "accepted" and value["state"] == "running"
    assert value["error"]["code"] == "prompt_identity"
