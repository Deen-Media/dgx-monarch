"""Store retention: one workflow copy per preview, expiry sweeps, restart recovery and real SQLite faults."""
from __future__ import annotations

import asyncio
import sqlite3
from copy import deepcopy

import pytest

import dgx_monarch.studio.experiments as experiments_module
import dgx_monarch.studio.jobs as jobs_module
import dgx_monarch.studio.store as store_module
from dgx_monarch.studio import preview_storage
from dgx_monarch.studio.contracts import StudioError, canonical, timestamp
from dgx_monarch.studio.experiments import GridExperiments
from dgx_monarch.studio.grid_views import read as read_grid
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store
from test_studio_execution import Driver, draft, grid_request

MARKER = "retention-marker-" * 64


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def stored_text(store, *kinds):
    with store.lock:
        rows = store.db.execute(f"SELECT document FROM objects WHERE kind IN ({','.join('?' for _ in kinds)})",  # noqa: S608
                                kinds).fetchall()
    return "".join(row[0] for row in rows)


def blob_text(store):
    with store.lock:
        return "".join(row[0] for row in store.db.execute("SELECT document FROM preview_workflows"))


def marked_draft():
    value = draft()
    value["workflow"]["1"]["inputs"]["text"] = MARKER
    return value


def marked_grid(excluded=()):
    request = grid_request()
    request["draft"]["workflow"]["1"]["inputs"]["text"] = MARKER
    request["excluded"] = list(excluded)
    return request


def test_job_preview_stores_its_workflow_once_and_reads_back_unchanged(store):
    async def scenario():
        jobs = Jobs(store, Driver())
        preview = await jobs.preview(marked_draft())
        # The wire shape keeps all three workflow slots.
        assert preview["requested"]["workflow"] == preview["effective"]["workflow"] == preview["workflow"]
        assert store.get("previews", preview["id"]) == preview
        assert store.list("previews") == [preview] and store.all("previews") == [preview]
        # Before single-copy storage the row held the graph three times plus
        # the compiled prompt; now only the compiled prompt holds the text.
        assert stored_text(store, "previews").count(MARKER) == 1
        assert blob_text(store).count(MARKER) == 1
        with store.lock:
            keys = [row[0] for row in store.db.execute("SELECT digest FROM preview_workflows")]
        assert keys == [preview["workflow_digest"]]
        job = jobs.submit({"preview_id": preview["id"], "idempotency_key": "submit-once"})
        assert job["submission"]["workflow"] == preview["workflow"]
        assert store.get("jobs", job["id"])["submission"]["requested"] == preview["requested"]
        await asyncio.gather(*jobs.tasks)
    asyncio.run(scenario())


def test_a_job_keeps_one_workflow_copy_after_its_preview_is_swept(tmp_path, monkeypatch):
    state = tmp_path / "state"
    store = Store(state)

    async def scenario():
        jobs = Jobs(store, Driver())
        preview = await jobs.preview(marked_draft())
        job = jobs.submit({"preview_id": preview["id"], "idempotency_key": "kept"})
        await asyncio.gather(*jobs.tasks)
        return preview, job

    try:
        preview, job = asyncio.run(scenario())
        # Only the compiled prompt holds the text; the job's workflow,
        # requested and effective slots name the one stored copy.
        assert stored_text(store, "jobs").count(MARKER) == 1
        assert blob_text(store).count(MARKER) == 1
    finally:
        store.close()
    later = timestamp() + 3600
    monkeypatch.setattr(store_module, "timestamp", lambda: later)
    restored = Store(state)
    try:
        restored.recover()
        assert restored.all("previews") == []
        kept = restored.get("jobs", job["id"])
        assert kept["submission"] == job["submission"]
        assert kept["submission"]["requested"]["workflow"] == kept["submission"]["effective"]["workflow"] == preview["workflow"]
        # Neither the startup sweep nor a later preview write's sweep removes a workflow a job names.
        restored.create("previews", {"id": "later", "workflow": {"5": {}}, "expires_at": 0})
        monkeypatch.setattr(store_module, "timestamp", lambda: later + 2 * preview_storage.SWEEP_SECONDS)
        restored.create("previews", {"id": "sweeps", "workflow": {"6": {}}, "expires_at": later + 10_000})
        assert {row["id"] for row in restored.all("previews")} == {"sweeps"}
        assert restored.get("jobs", job["id"])["submission"] == job["submission"]
        assert blob_text(restored).count(MARKER) == 1
    finally:
        restored.close()


def test_a_slot_that_differs_from_the_source_graph_stays_inline(store):
    workflow = {"1": {"class_type": "Render", "inputs": {"text": MARKER}}}
    other = {"2": {"class_type": "Render", "inputs": {"text": "edited"}}}
    preview = {"id": "preview", "workflow": workflow, "requested": {"workflow": deepcopy(workflow)},
               "effective": {"workflow": other}, "expires_at": timestamp() + 600}
    assert store.create("previews", preview) == preview
    assert store.get("previews", "preview") == preview
    assert "edited" in stored_text(store, "previews") and MARKER not in stored_text(store, "previews")


def test_grid_previews_share_one_workflow_copy_across_cells(store):
    async def scenario():
        jobs = Jobs(store, Driver())
        grids = GridExperiments(jobs)
        preview = await grids.preview(marked_grid())
        assert preview["ready"] and preview["job_count"] == 4
        # Text is a grid axis, so no cell prompt repeats the source graph's text.
        assert stored_text(store, "previews", "grid_previews").count(MARKER) == 0
        assert blob_text(store).count(MARKER) == 1
        assert store.get("grid_previews", preview["id"]) == preview
        assert read_grid(store, "grid_previews", preview["id"], compact=False) == preview
        cell = store.get("previews", preview["cells"][0]["preview_id"])
        assert cell["workflow"]["1"]["inputs"]["text"] == MARKER
        run = grids.submit({"preview_id": preview["id"], "idempotency_key": "grid-once"})
        await asyncio.gather(*jobs.tasks)
        assert run["submission"] == preview
    asyncio.run(scenario())


def test_startup_sweeps_expired_previews_except_those_a_grid_run_reads(tmp_path, monkeypatch):
    state = tmp_path / "state"
    store = Store(state)

    async def scenario():
        jobs = Jobs(store, Driver())
        grids = GridExperiments(jobs)
        dispatched = await grids.preview(marked_grid(excluded=[0]))
        run = grids.submit({"preview_id": dispatched["id"], "idempotency_key": "dispatched"})
        await asyncio.gather(*jobs.tasks)
        abandoned = await grids.preview(marked_grid())
        single = await jobs.preview(draft())
        return dispatched, run, abandoned, single

    try:
        dispatched, run, abandoned, single = asyncio.run(scenario())
        excluded, sent = dispatched["cells"][0], dispatched["cells"][1]
        assert excluded["excluded"] and not sent["excluded"]
        # An excluded cell is never built, so it names no preview to hold.
        assert "preview_id" not in excluded
        # A grid run written before the hold table existed is held by recovery.
        legacy_cell = store.create("previews", {"id": "legacy-cell", "workflow": {"9": {}}, "expires_at": 0})
        store.create("grid_runs", {"id": "legacy-run", "preview_id": None, "cells": [
            {"index": 0, "excluded": True, "preview_id": legacy_cell["id"]}]})
    finally:
        store.close()
    later = timestamp() + 3600
    monkeypatch.setattr(store_module, "timestamp", lambda: later)
    restored = Store(state)
    try:
        fresh = restored.create("previews", {"id": "fresh", "workflow": {"8": {}}, "expires_at": later + 600})
        restored.recover()
        kept = {row["id"] for row in restored.all("previews")} | {row["id"] for row in restored.all("grid_previews")}
        assert kept == {dispatched["id"], legacy_cell["id"], fresh["id"]}
        assert {sent["preview_id"], abandoned["id"], single["id"]}.isdisjoint(kept)
        # What stays still reads back whole; the run keeps its own copy.
        assert restored.get("grid_previews", dispatched["id"]) == dispatched
        assert restored.get("grid_previews", dispatched["id"])["request"]["draft"]["workflow"]["1"]["inputs"]["text"] == MARKER
        assert restored.get("previews", legacy_cell["id"])["workflow"] == {"9": {}}
        assert restored.get("grid_runs", run["id"])["submission"] == dispatched
        # Only workflows a remaining preview or job names stay stored; this run's jobs share the dispatched grid's.
        with restored.lock:
            digests = {row[0] for row in restored.db.execute("SELECT digest FROM preview_workflows")}
        named = {preview_storage.pack(restored.db, kind, restored.get(kind, identity))["workflow_storage"]["digest"]
                 for kind, identity in (("grid_previews", dispatched["id"]), ("previews", fresh["id"]),
                                        ("previews", legacy_cell["id"]))}
        assert digests == named
    finally:
        restored.close()


def test_preview_writes_sweep_expired_previews_at_most_once_per_interval(store, monkeypatch):
    clock = [timestamp()]
    monkeypatch.setattr(store_module, "timestamp", lambda: clock[0])
    # Before recovery derives the grid-run holds, no write sweeps anything.
    store.create("previews", {"id": "stale", "workflow": {"0": {}}, "expires_at": 0})
    clock[0] += preview_storage.SWEEP_SECONDS
    store.create("previews", {"id": "unswept", "workflow": {"0": {}}, "expires_at": clock[0] + 10_000})
    assert [row["id"] for row in store.all("previews")] == ["stale", "unswept"]
    store.recover()
    assert [row["id"] for row in store.all("previews")] == ["unswept"]
    store.create("previews", {"id": "first", "workflow": {"1": {}}, "expires_at": clock[0] + 1})
    clock[0] += preview_storage.SWEEP_SECONDS
    store.create("previews", {"id": "second", "workflow": {"2": {}}, "expires_at": clock[0] + 1})
    assert [row["id"] for row in store.all("previews")] == ["second", "unswept"]
    clock[0] += 60
    store.create("previews", {"id": "third", "workflow": {"3": {}}, "expires_at": clock[0] + 600})
    assert [row["id"] for row in store.all("previews")] == ["second", "third", "unswept"]
    clock[0] += preview_storage.SWEEP_SECONDS
    store.create("grid_previews", {"id": "grid", "request": {"draft": {"workflow": {"4": {}}}},
                                   "expires_at": clock[0] + 600})
    assert [row["id"] for row in store.all("previews")] == ["third", "unswept"]
    with store.lock:
        digests = {row[0] for row in store.db.execute("SELECT digest FROM preview_workflows")}
    # The swept previews' graphs left with them; stale shared its graph with unswept.
    assert len(digests) == 3


def test_rollback_keeps_the_real_sqlite_error(store):
    store.create("jobs", {"id": "small", "state": "queued"})
    pages = store.db.execute("PRAGMA page_count").fetchone()[0]
    store.db.execute(f"PRAGMA max_page_count={pages}")
    with pytest.raises(sqlite3.OperationalError, match="database or disk is full"):
        store.create("jobs", {"id": "large", "state": "queued", "padding": "x" * 2_000_000})
    assert not store.db.in_transaction
    store.db.execute(f"PRAGMA max_page_count={pages * 1000}")
    store.create("jobs", {"id": "after", "state": "queued", "padding": "x" * 2_000_000})
    assert {row["id"] for row in store.all("jobs")} == {"small", "after"}


def test_restart_marks_queued_jobs_unknown_like_running_ones(store):
    for identity, state in (("queued", "queued"), ("running", "running"), ("submitting", "submitting"),
                            ("succeeded", "succeeded"), ("cancelled", "cancelled")):
        store.create("jobs", {"id": identity, "state": state, "prompt_id": f"prompt-{identity}"})
    store.create("operations", {"id": "operation", "state": "queued"})
    store.recover()
    states = {row["id"]: row["state"] for row in store.all("jobs")}
    assert states == {"queued": "unknown", "running": "unknown", "submitting": "unknown",
                      "succeeded": "succeeded", "cancelled": "cancelled"}
    assert store.get("jobs", "queued")["recovery"] == "Reconcile before any retry."
    # Operations have no driver queue; their states are unchanged by this rule.
    assert store.get("operations", "operation")["state"] == "queued"


def test_progress_state_receipts_keep_identity_without_node_maps(store):
    store.create("jobs", {"id": "job", "state": "queued", "prompt_id": "prompt"})
    jobs = Jobs(store, Driver())
    nodes = {str(index): {"value": index, "max": 100, "state": "running", "node_id": str(index),
                          "display_node_id": str(index), "prompt_id": "prompt"} for index in range(40)}
    for step in range(128):
        message = {"type": "progress_state", "data": {"prompt_id": "prompt", "timestamp": step,
                                                      "nodes": {**nodes, "step": {"value": step}}}}
        jobs.consume_event(message)
    jobs.consume_event(message)
    job = store.get("jobs", "job")
    assert len(job["backend_events"]) == 128
    assert all("nodes" not in receipt["data"] for receipt in job["backend_events"])
    assert job["node_progress"]["step"] == {"value": 127}
    # Before receipts dropped the node map, this document held about 0.8 MB.
    assert len(canonical(job)) < 64 * 1024


def test_a_swept_preview_still_refuses_submission_as_expired(store, monkeypatch):
    # The sweep deletes expired previews; a later submit must stay a 409 refusal
    # the browser recognizes, never a 404 that leaves its outcome unknown.
    async def build():
        jobs = Jobs(store, Driver())
        grids = GridExperiments(jobs)
        return jobs, grids, await jobs.preview(draft()), await grids.preview(grid_request())
    jobs, grids, preview, grid = asyncio.run(build())
    later = timestamp() + 3600
    for module in (store_module, jobs_module, experiments_module):
        monkeypatch.setattr(module, "timestamp", lambda: later)
    store.recover()
    assert store.all("previews") == [] and store.all("grid_previews") == []
    with pytest.raises(StudioError) as refused:
        jobs.submit({"preview_id": preview["id"], "idempotency_key": "after-sweep"})
    assert (refused.value.code, refused.value.status) == ("preview_expired", 409)
    with pytest.raises(StudioError) as refused:
        grids.submit({"preview_id": grid["id"], "idempotency_key": "grid-after-sweep"})
    assert (refused.value.code, refused.value.status) == ("grid_preview", 409)
    assert store.request("jobs", "after-sweep", {}) is None
