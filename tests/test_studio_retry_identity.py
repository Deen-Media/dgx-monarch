"""Retry idempotency is bound to the endpoint's job, never a body override."""

import asyncio
from copy import deepcopy

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store


def test_retry_body_cannot_override_job_identity(tmp_path, monkeypatch):
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, None)
        monkeypatch.setattr(jobs, "launch", lambda coroutine: coroutine.close())
        for identity in ("first", "second"):
            store.create("jobs", {"id": identity, "state": "failed", "submission": {"prompt": {}}})
        request = {"confirmed": True, "idempotency_key": "shared", "retry_of": "spoofed"}
        retried = jobs.retry("first", request)
        assert jobs.retry("first", request)["id"] == retried["id"]
        with pytest.raises(StudioError) as caught:
            jobs.retry("second", request)
        assert caught.value.code == "idempotency_conflict"
        assert retried["retry_of"] == "first"
        assert len(store.all("jobs")) == 3
    finally:
        store.close()


class VersionedDriver:
    """A driver whose version and node definitions can change between a run and its retry."""

    def __init__(self):
        self.version, self.calls = "driver-1", []
        self.schema = {"Render": {"input": {"required": {
            "steps": ["INT", {"min": 1, "max": 20}], "sampler": [["euler", "heun"]]}}, "output": [], "output_node": True}}

    async def request(self, method, path, data=None):
        self.calls.append((method, path, data))
        if path == "/object_info":
            return deepcopy(self.schema)
        if path == "/system_stats":
            return {"system": {"comfyui_version": self.version}}
        if path == "/prompt":
            return {"prompt_id": data.get("prompt_id") or "prompt-" + str(len(self.calls))}
        return {"queue_running": [], "queue_pending": []} if path == "/queue" else {}


def render_draft():
    return {"workflow": {"1": {"class_type": "Render", "inputs": {"steps": 2, "sampler": "euler"}}},
            "values": {}, "seed_policy": "fixed", "operation": "image"}


async def failed_job(store, driver):
    jobs = Jobs(store, driver)
    preview = await jobs.preview(render_draft())
    job = jobs.submit({"preview_id": preview["id"], "idempotency_key": "first"})
    await asyncio.gather(*jobs.tasks)
    store.update("jobs", job["id"], {"state": "failed"})
    store.update("previews", preview["id"], {"expires_at": 0})
    return jobs, store.get("jobs", job["id"])


def test_retry_records_its_own_runtime_and_keeps_the_frozen_request(tmp_path):
    async def scenario():
        store = Store(tmp_path / "state")
        try:
            driver = VersionedDriver()
            jobs, original = await failed_job(store, driver)
            driver.version = "driver-2"
            driver.schema["Render"]["input"]["required"]["sampler"][0].append("new-sampler")
            await jobs.catalog()
            # The review expired; a confirmed retry re-dispatches against the live schema instead.
            retried = jobs.retry(original["id"], {"confirmed": True, "idempotency_key": "retry"})
            await asyncio.gather(*jobs.tasks)
            value = store.get("jobs", retried["id"])
            assert value["submission"] == original["submission"]
            assert value["submission"]["versions"]["driver"] == "driver-1"
            assert value["retry_runtime"]["versions"]["driver"] == "driver-2"
            assert value["retry_runtime"]["runtime_observation"]["reported"]["comfyui_version"] == "driver-2"
            assert value["retry_runtime"]["schema_identity"] == original["submission"]["schema_identity"]
            assert value["retry_runtime"]["schema_digest"] != original["submission"]["schema_digest"]
            assert "retry_runtime" not in original
            assert value["state"] == "queued"
        finally:
            store.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["definition", "choice"])
def test_retry_is_refused_when_the_live_schema_no_longer_fits(tmp_path, change):
    async def scenario():
        store = Store(tmp_path / "state")
        try:
            driver = VersionedDriver()
            jobs, original = await failed_job(store, driver)
            if change == "definition":
                driver.schema["Render"]["output"] = ["IMAGE"]
            else:
                driver.schema["Render"]["input"]["required"]["sampler"][0].remove("euler")
            await jobs.catalog()
            with pytest.raises(StudioError) as caught:
                jobs.retry(original["id"], {"confirmed": True, "idempotency_key": "retry"})
            assert caught.value.code == ("schema_changed" if change == "definition" else "choice_unavailable")
            assert caught.value.status == 409
            assert len(store.all("jobs")) == 1
            assert store.request("jobs", "retry", {"confirmed": True, "idempotency_key": "retry",
                                                   "retry_of": original["id"]}) is None
        finally:
            store.close()
    asyncio.run(scenario())


def test_grid_retry_records_runtime_per_cell_in_the_full_manifest(tmp_path):
    from dgx_monarch.studio.experiments import GridExperiments

    async def scenario():
        store = Store(tmp_path / "state")
        try:
            driver = VersionedDriver()
            jobs = Jobs(store, driver)
            grids = GridExperiments(jobs)
            preview = await grids.preview({"draft": render_draft(), "axes": [{"name": "root:1:steps", "values": [2, 3]}]})
            grid = grids.submit({"preview_id": preview["id"], "idempotency_key": "grid"})
            await asyncio.gather(*jobs.tasks)
            store.update("jobs", grid["cells"][0]["job_id"], {"state": "failed"})
            driver.version = "driver-2"
            await jobs.catalog()
            grids.retry(grid["id"], {"indices": [0], "confirmed": True, "idempotency_key": "cells"})
            await asyncio.gather(*jobs.tasks)
            full = grids.get(grid["id"])
            assert full["cells"][0]["retry_runtime"]["versions"]["driver"] == "driver-2"
            assert "retry_runtime" not in full["cells"][1]
            assert "retry_runtime" not in grids.get(grid["id"], compact=True)["cells"][0]
            assert sum(path == "/system_stats" for _, path, _ in driver.calls) == 2  # Preview, then one retry.
        finally:
            store.close()
    asyncio.run(scenario())
