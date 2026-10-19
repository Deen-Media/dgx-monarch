"""Durable grid and driver observation regressions; all drivers are CPU fakes."""
from __future__ import annotations

import asyncio
import io
import json
import struct
from copy import deepcopy

import pytest
from PIL import Image

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.experiments import GridExperiments
from dgx_monarch.studio.history import page_artifacts, page_jobs
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.live_preview import accept_preview, parse_frame, preview_frame
from dgx_monarch.studio.store import Store


class Driver:
    def __init__(self):
        self.calls = []
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = {}
        self.schema = {"Render": {"input": {"required": {
            "seed": ["INT", {"min": 0, "max": 100}], "steps": ["INT", {"min": 1, "max": 20}],
            "text": ["STRING", {}]}}, "output": [], "output_node": True}}

    async def request(self, method, path, data=None):
        self.calls.append((method, path, deepcopy(data)))
        if path == "/object_info":
            return self.schema
        if path == "/system_stats":
            return {"system": {"comfyui_version": "test-version", "argv": ["secret"]}}
        if path == "/prompt":
            identity = "prompt-" + str(len(self.calls))
            self.queue["queue_pending"].append([1, identity, data["prompt"], data["extra_data"]])
            return {"prompt_id": identity}
        if path == "/queue":
            return self.queue
        if path == "/history":
            return self.history
        return {}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def draft():
    return {"workflow": {"1": {"class_type": "Render", "inputs": {"seed": 5, "steps": 2, "text": "cat"}}},
            "values": {}, "seed_policy": "fixed", "operation": "image"}


def grid_request():
    return {"draft": draft(), "axes": [{"name": "root:1:steps", "values": [2, 3]},
                                        {"name": "root:1:text", "values": ["cat", "dog"]}],
            "seed_policy": "increment", "seed_binding": "root:1:seed", "base_seed": 7}


async def drained(jobs):
    await asyncio.gather(*jobs.tasks)


def test_grid_atomic_durable_dispatch_and_idempotent_selective_retry(store):
    async def scenario():
        driver = Driver()
        jobs = Jobs(store, driver)
        grids = GridExperiments(jobs)
        preview = await grids.preview(grid_request())
        assert preview["ready"] and preview["job_count"] == 4
        assert [store.get("previews", row["preview_id"])["effective"]["seeds"]["1.seed"] for row in preview["cells"]] == [7, 8, 9, 10]
        request = {"preview_id": preview["id"], "idempotency_key": "grid-once"}
        accepted = grids.submit(request)
        assert len(store.all("jobs")) == 4
        assert not any(method == "POST" for method, _, _ in driver.calls)
        await drained(jobs)
        repeated = grids.submit(request)
        assert repeated["id"] == accepted["id"] and repeated["state"] == "queued"
        assert sum(path == "/prompt" for _, path, _ in driver.calls) == 4
        first = repeated["cells"][0]["job_id"]
        store.update("jobs", first, {"state": "failed"})
        frozen = store.get("jobs", first)["submission"]
        retry_request = {"indices": [0], "confirmed": True, "idempotency_key": "retry-once"}
        retry = grids.retry(accepted["id"], retry_request)
        await drained(jobs)
        assert len(store.all("jobs")) == 5
        assert store.get("jobs", retry["cells"][0]["job_id"])["submission"] == frozen
        grids.retry(accepted["id"], retry_request)
        assert len(store.all("jobs")) == 5
        with pytest.raises(StudioError, match="failed or cancelled"):
            grids.retry(accepted["id"], {"indices": [1], "confirmed": True, "idempotency_key": "retry-running"})
    asyncio.run(scenario())


def test_grid_invalid_cells_are_visible_and_block_until_explicitly_excluded(store):
    async def scenario():
        grids = GridExperiments(Jobs(store, Driver()))
        request = grid_request()
        request["axes"] = [{"name": "root:1:steps", "values": [2, 999]}]
        preview = await grids.preview(request)
        assert not preview["ready"] and len(preview["cells"]) == 2
        assert preview["cells"][1]["errors"]
        with pytest.raises(StudioError, match="fully validated"):
            grids.submit({"preview_id": preview["id"], "idempotency_key": "invalid"})
        assert not store.all("jobs")
        request["excluded"] = [1]
        corrected = await grids.preview(request)
        assert corrected["ready"] and corrected["job_count"] == 1
        assert corrected["cells"][1]["errors"]
    asyncio.run(scenario())


def test_grid_limits_unknown_outcomes_and_transaction_rollback(store):
    async def scenario():
        jobs = Jobs(store, Driver())
        grids = GridExperiments(jobs)
        request = grid_request()
        request["axes"] = [{"name": "root:1:text", "values": ["x"] * 1025}]
        with pytest.raises(StudioError, match="1024"):
            await grids.preview(request)
        preview = await grids.preview(grid_request())
        accepted = grids.submit({"preview_id": preview["id"], "idempotency_key": "grid-crash"})
        for task in jobs.tasks:
            task.cancel()
        await asyncio.gather(*jobs.tasks, return_exceptions=True)
        restored = Store(store.directory)
        try:
            restored.recover()
            state = GridExperiments(Jobs(restored, Driver())).get(accepted["id"])
            assert state["state"] == "unknown"
            with pytest.raises(StudioError, match="failed or cancelled"):
                grids.retry(accepted["id"], {"indices": [0], "confirmed": True, "idempotency_key": "unknown-retry"})
        finally:
            restored.close()
    asyncio.run(scenario())
    with pytest.raises(StudioError, match="finite JSON"):
        store.reserve_group("grid_runs", "rollback", {}, {"id": "bad-grid"},
                            [{"id": "bad-child", "state": "submitting", "bad": float("nan")}])
    assert store.request("grid_runs", "rollback", {}) is None
    with pytest.raises(StudioError, match="not found"):
        store.get("grid_runs", "bad-grid")


def saved_job(store, identity="job", prompt_id="prompt"):
    return store.create("jobs", {"id": identity, "state": "queued", "prompt_id": prompt_id,
                                 "submission": {"prompt": {}, "workflow": {}, "digest": "d"}})


def test_events_require_identity_are_durable_ordered_and_terminal_monotonic(store):
    saved_job(store)
    jobs = Jobs(store, Driver())
    jobs.consume_event({"type": "progress", "data": {"value": 1, "max": 10}})
    jobs.consume_event({"type": "progress", "data": {"prompt_id": "elsewhere", "value": 1, "max": 10}})
    assert store.get("jobs", "job")["state"] == "queued"
    event = {"type": "progress", "data": {"prompt_id": "prompt", "value": 4, "max": 10, "timestamp": 20}}
    jobs.consume_event(event)
    watermark = store.events()["latest_cursor"]
    jobs.consume_event(event)
    assert store.events()["latest_cursor"] == watermark
    jobs.consume_event({"type": "progress", "data": {"prompt_id": "prompt", "value": 1, "max": 10, "timestamp": 10}})
    assert store.get("jobs", "job")["progress"]["value"] == 4
    jobs.consume_event({"type": "execution_success", "data": {"prompt_id": "prompt", "timestamp": 30}})
    jobs.consume_event({"type": "execution_start", "data": {"prompt_id": "prompt", "timestamp": 40}})
    restored = Store(store.directory)
    try:
        assert restored.get("jobs", "job")["state"] == "succeeded"
        assert restored.get("jobs", "job")["phase"] == "execution_success"
        assert len(restored.get("jobs", "job")["backend_events"]) == 4
        assert restored.client_id == store.client_id
    finally:
        restored.close()


def test_history_beyond_recent_page_reconciles_outputs_and_cancellation_receipts(store):
    saved_job(store)

    class ScopedDriver(Driver):
        async def history_prompt(self, identity):
            return {identity: {"status": {"completed": True}, "outputs": {"9": {"images": [{"filename": "ok.png"}]}}}}

        async def cancel_prompt(self, identity):
            self.calls.append(("CANCEL", identity, None))
            return {"cancelled": True}

    async def scenario():
        driver = ScopedDriver()
        jobs = Jobs(store, driver)
        driver.queue["queue_running"] = [[0, "prompt", {}, {}]]
        result = await jobs.cancel("job", {"confirmed": True})
        assert result["state"] == "cancel_requested"
        assert result["cancellation"]["receipt"] == {"cancelled": True}
        assert not any(path == "/interrupt" for _, path, _ in driver.calls)
        driver.queue["queue_running"] = []
        await jobs.reconcile()
        assert store.get("jobs", "job")["state"] == "succeeded"
        assert len(store.all("artifacts")) == 1  # Cancellation never erases completed outputs.
    asyncio.run(scenario())


def test_history_keyset_filtering_retains_all_pages_and_literal_search(store):
    for index in range(240):
        store.create("artifacts", {"id": str(index), "created_at": index, "kind": "image", "operation": "image",
                                   "status": "succeeded", "project_id": "project", "search_text": "cat%_" if index % 2 else "dog",
                                   "model_ids": ["model-a"], "metadata": {"width": 512}})
    query = {"limit": 17, "q": "cat%_", "model": "model-a", "width": "512", "project_id": "project"}
    identities = []
    while True:
        page = page_artifacts(store, query)
        identities.extend(row["id"] for row in page["items"])
        if page["next_cursor"] is None:
            break
        query["cursor"] = page["next_cursor"]
    assert len(identities) == len(set(identities)) == 120
    assert not page_artifacts(store, {"q": "' OR 1=1 --"})["items"]
    cursor = page_artifacts(store, {"limit": 1})["next_cursor"]
    with pytest.raises(StudioError, match="changed filters"):
        page_artifacts(store, {"cursor": cursor, "q": "changed"})
    saved_job(store)
    assert page_jobs(store, {"status": "queued"})["items"][0]["id"] == "job"


def test_live_previews_require_prompt_metadata_and_do_not_become_final_outputs(store, monkeypatch):
    import dgx_monarch.studio.live_preview as preview
    monkeypatch.setattr(preview, "FRAME_COUNT", 1)
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, format="PNG")
    metadata = json.dumps({"prompt_id": "prompt", "node_id": "1", "image_type": "image/png"}).encode()
    frame = struct.pack(">II", 4, len(metadata)) + metadata + image.getvalue()
    assert parse_frame(struct.pack(">II", 1, len(metadata)) + metadata + image.getvalue()) is None
    message = parse_frame(frame)
    accept_preview(store, message)
    assert not store.all("artifacts")
    saved_job(store)
    accept_preview(store, message)
    artifact = store.all("artifacts")[0]
    data, mime = preview_frame(store, artifact["id"])
    assert mime == "image/jpeg" and data.startswith(b"\xff\xd8")
    assert artifact["intermediate"] and artifact["provenance"]["status"] == "approximate_live_preview"
    saved_job(store, "job-two", "prompt-two")
    message["data"]["prompt_id"] = "prompt-two"
    accept_preview(store, message)
    with pytest.raises(StudioError, match="bounded cache"):
        preview_frame(store, artifact["id"])


def test_runtime_capture_does_not_claim_global_telemetry_as_job_execution(store):
    async def scenario():
        jobs = Jobs(store, Driver())
        value = await jobs.preview(draft())
        assert value["versions"]["driver"] == "test-version"
        assert "argv" not in value["runtime_observation"]["reported"]
        assert value["effective_topology"] == "pending_backend_admission"
        assert value["model_fingerprints"] == "unavailable"
    asyncio.run(scenario())


def test_style_grid_freezes_ordered_saved_revisions_through_dispatch_and_retry(store):
    async def scenario():
        store.save_record("styles", "paint", {"prompt": "painting of {prompt}"}, 0)
        store.save_record("styles", "frame", {"prompt": "framed {prompt}"}, 0)
        jobs = Jobs(store, Driver())
        grids = GridExperiments(jobs)
        request = grid_request()
        request["draft"].update(style_bindings={"prompt": "root:1:text"}, styles=[{"id": "old", "prompt": "old"}])
        request["axes"] = [{"name": "styles", "values": [[], ["paint", "frame"], ["frame", "paint"]]},
                           {"name": "root:1:steps", "values": [2, 4]}]
        preview = await grids.preview(request)
        assert preview["ready"] and preview["job_count"] == 6
        texts = [store.get("previews", row["preview_id"])["prompt"]["1"]["inputs"]["text"] for row in preview["cells"]]
        assert texts == ["cat", "cat", "framed painting of cat", "framed painting of cat", "painting of framed cat", "painting of framed cat"]
        assert [row["id"] for row in preview["cells"][2]["style_records"]] == ["paint", "frame"]
        store.save_record("styles", "paint", {"prompt": "changed {prompt}"}, 1)
        submit = {"preview_id": preview["id"], "idempotency_key": "style-grid"}
        result = grids.submit(submit)
        await drained(jobs)
        assert grids.submit(submit)["id"] == result["id"]
        first = result["cells"][2]["job_id"]
        frozen = store.get("jobs", first)["submission"]
        assert frozen["prompt"]["1"]["inputs"]["text"] == "framed painting of cat"
        assert frozen["requested"]["style_records"][0]["revision"] == 1
        store.update("jobs", first, {"state": "failed"})
        retry_request = {"indices": [2], "confirmed": True, "idempotency_key": "style-retry"}
        retried = grids.retry(result["id"], retry_request)
        await drained(jobs)
        assert store.get("jobs", retried["cells"][2]["job_id"])["submission"] == frozen
        grids.retry(result["id"], retry_request)
        assert len(store.all("jobs")) == 7
    asyncio.run(scenario())


def test_style_grid_missing_records_are_visible_and_explicitly_excludable(store):
    async def scenario():
        request = grid_request()
        request["draft"]["style_bindings"] = {"prompt": "root:1:text"}
        request["axes"] = [{"name": "styles", "values": [[], ["missing"], "malformed"]}]
        grids = GridExperiments(Jobs(store, Driver()))
        preview = await grids.preview(request)
        assert not preview["ready"]
        assert not preview["cells"][0]["errors"]
        assert all(row["errors"] for row in preview["cells"][1:])
        request["excluded"] = [1, 2]
        preview = await grids.preview(request)
        assert preview["ready"] and preview["job_count"] == 1
    asyncio.run(scenario())


def test_fixed_grid_preserves_draft_seeds_despite_stale_increment_controls(store):
    async def scenario():
        request = grid_request()
        request.update(seed_policy="fixed", base_seed=42, seed_binding="root:1:seed")
        grids = GridExperiments(Jobs(store, Driver()))
        preview = await grids.preview(request)
        assert preview["ready"]
        assert all(store.get("previews", row["preview_id"])["prompt"]["1"]["inputs"]["seed"] == 5
                   for row in preview["cells"])
        request["draft"]["values"] = {"root:1:seed": 8}
        request.update(seed_binding="deleted:node:seed", base_seed=-1)
        preview = await grids.preview(request)
        assert all(store.get("previews", row["preview_id"])["prompt"]["1"]["inputs"]["seed"] == 8
                   for row in preview["cells"])
    asyncio.run(scenario())


def test_experiment_record_freezes_grid_definition_and_live_outcome_without_dispatch(store):
    from dgx_monarch.studio.experiment_records import bind_grid_record

    async def scenario():
        jobs = Jobs(store, Driver())
        grids = GridExperiments(jobs)
        preview = await grids.preview(grid_request())
        grid = grids.submit({"preview_id": preview["id"], "idempotency_key": "linked-grid"})
        await drained(jobs)
        failed = grid["cells"][0]["job_id"]
        store.update("jobs", failed, {"state": "failed"})
        calls = len(jobs.client.calls)
        source = {"name": "Linked experiment", "grid_id": grid["id"], "extension": {"retained": True}}
        linked = bind_grid_record(store, source)
        assert linked["grid_request"] == grid_request()
        assert linked["grid_snapshot"]["cells"][0]["state"] == "failed"
        assert linked["grid_source"]["state"] == "available"
        assert source == {"name": "Linked experiment", "grid_id": grid["id"], "extension": {"retained": True}}
        assert len(jobs.client.calls) == calls and len(store.all("jobs")) == 4
        linked["grid_id"] = "unavailable-imported-grid"
        restored = bind_grid_record(store, linked)
        assert restored["grid_request"] == grid_request()
        assert restored["grid_source"]["state"] == "unavailable"
        assert len(jobs.client.calls) == calls and len(store.all("jobs")) == 4
    asyncio.run(scenario())
