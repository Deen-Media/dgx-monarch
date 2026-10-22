"""Grid history and expansion stay bounded without losing full review evidence."""
from __future__ import annotations

import asyncio

import pytest

from dgx_monarch.studio import drafts, experiments
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.drafts import DraftError, expand_grid
from dgx_monarch.studio.experiment_records import bind_grid_record
from dgx_monarch.studio.experiments import GridExperiments
from dgx_monarch.studio.history import page_grids
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.library import Library
from dgx_monarch.studio.store import Store
from test_studio_execution import Driver, grid_request
from test_studio_integration import companion


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def test_grid_parameter_budget_refuses_before_repeated_deepcopy(monkeypatch):
    monkeypatch.setattr(drafts, "MAX_GRID_PARAMETER_BYTES", 400)
    copies = []
    original = drafts.copy.deepcopy

    def copy(value):
        copies.append(value)
        return original(value)

    monkeypatch.setattr(drafts.copy, "deepcopy", copy)
    with pytest.raises(DraftError, match="expanded parameter budget"):
        expand_grid({"nested": {"x": list(range(100))}}, [{"name": "x", "values": list(range(1024))}],
                    {"x": {"type": "INT"}}, seed_policy="fixed", base_seed=0)
    assert sum(isinstance(value, dict) and "nested" in value for value in copies) <= 1


def test_over_budget_grid_does_not_persist_partial_previews(store, monkeypatch):
    monkeypatch.setattr(experiments, "GRID_BYTES", 3000)
    with pytest.raises(StudioError, match=r"grid budget|storage"):
        asyncio.run(GridExperiments(Jobs(store, Driver())).preview(grid_request()))
    assert store.list("previews") == [] and store.list("grid_previews") == []


def test_grid_preview_commit_is_atomic(store, monkeypatch):
    original = store._put
    count = 0

    def fail(kind, value, revision):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("injected storage interruption")
        original(kind, value, revision)

    monkeypatch.setattr(store, "_put", fail)
    with pytest.raises(OSError, match="injected"):
        asyncio.run(GridExperiments(Jobs(store, Driver())).preview(grid_request()))
    assert store.list("previews") == [] and store.list("grid_previews") == []


def saved_grid(store, identity, stamp):
    job_id = "job-" + identity
    store.create("jobs", {"id": job_id, "state": "failed", "prompt_id": "p-" + identity,
                          "submission": {"digest": identity, "workflow": {}, "prompt": {}}})
    store.create("grid_runs", {"id": identity, "state": "submitting", "created_at": stamp,
                              "submission": {"large": "x" * 100_000, "request": {"draft": {"workflow": {}}}},
                              "cells": [{"index": 0, "coordinates": {"x": 1}, "job_id": job_id,
                                         "attempts": [job_id], "excluded": False, "style_records": ["x" * 100_000]}]})


def test_grid_history_is_paginated_live_and_does_not_load_full_jobs(store, monkeypatch):
    for index in range(5):
        saved_grid(store, str(index), index)
    monkeypatch.setattr(store, "get", lambda *_: pytest.fail("Grid summaries must not load full records"))
    page = page_grids(store, {"limit": 2})
    assert [row["id"] for row in page["items"]] == ["4", "3"]
    assert all(row["state"] == "completed_with_failures" and "cells" not in row for row in page["items"])
    next_page = page_grids(store, {"limit": 2, "cursor": page["next_cursor"]})
    assert [row["id"] for row in next_page["items"]] == ["2", "1"]
    compact = GridExperiments(Jobs(store, Driver())).get("4", compact=True)
    assert compact["cells"][0]["state"] == "failed"
    assert "submission" not in compact and "style_records" not in compact["cells"][0]


def test_grid_retry_key_cannot_be_spoofed_across_parent_identities(store):
    for identity in ("a", "b"):
        saved_grid(store, identity, 1)
    jobs = Jobs(store, Driver())
    jobs.launch = lambda coroutine: coroutine.close()
    grids = GridExperiments(jobs)
    request = {"idempotency_key": "one-key", "grid_id": "spoof", "confirmed": True, "indices": [0]}
    grids.retry("a", request)
    with pytest.raises(StudioError, match="different request"):
        grids.retry("b", request)


def test_experiment_freezes_live_outputs_including_previous_attempts(store):
    saved_grid(store, "grid", 1)
    value = store.get("grid_runs", "grid")
    value["cells"][0]["attempts"].append("earlier-job")
    store.update("grid_runs", "grid", {"cells": value["cells"]})
    for identity, job_id in (("current-output", "job-grid"), ("prior-output", "earlier-job")):
        store.create("artifacts", {"id": identity, "job_id": job_id})
    live = GridExperiments(Jobs(store, Driver())).get("grid", compact=True)
    frozen = bind_grid_record(store, {"grid_id": "grid"})
    assert set(frozen["grid_snapshot"]["cells"][0]["artifact_ids"]) == {"current-output", "prior-output"}
    assert frozen["grid_snapshot"]["cells"][0]["artifact_ids"] == live["cells"][0]["artifact_ids"]


def test_missing_library_review_is_a_definitive_preacceptance_refusal(store):
    request = {"preview_id": "expired", "digest": "old", "confirmed": True, "idempotency_key": "import"}
    with pytest.raises(StudioError) as exc:
        Library(store).apply(request)
    assert exc.value.code == "library_preview"
    assert store.request("library_imports", "import", request) is None


def test_http_compact_grid_roundtrip_retains_full_export(tmp_path, monkeypatch):
    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            from test_studio_http_quality import preview_request

            preview = await system.request("POST", "/grids/preview?summary=true", json=preview_request("grids"))
            assert preview["summary"] and "request" not in preview
            full = await system.request("GET", f"/grids/previews/{preview['id']}")
            assert "request" in full
            accepted = await system.request("POST", "/grids?summary=true", expected=202,
                                            json={"preview_id": preview["id"], "idempotency_key": "grid-once"})
            assert accepted["summary"] and "submission" not in accepted
            for cell in accepted["cells"]:
                await system.store.wait_state("jobs", cell["job_id"], "queued")
            compact = await system.request("GET", f"/grids/{accepted['id']}?summary=true")
            assert compact["state"] == "queued" and "submission" not in compact
            exported = await system.request("GET", f"/grids/{accepted['id']}")
            assert exported["submission"]["request"] == full["request"]
            listing = await system.request("GET", "/grids")
            assert listing["items"][0]["id"] == accepted["id"] and "cells" not in listing["items"][0]
    asyncio.run(scenario())


def large_workflow_grid(excluded=()):
    request = grid_request()
    # About 24 KB of graph metadata that the compiled prompt does not carry, as in a UI workflow.
    request["draft"]["workflow"]["extra"] = {"layout": "x" * 24 * 1024}
    request.update(seed_policy="fixed", seed_binding=None, excluded=list(excluded),
                   axes=[{"name": "root:1:steps", "values": list(range(1, 17))},
                         {"name": "root:1:text", "values": [f"subject {index}" for index in range(64)]}])
    return request


def test_full_grid_fits_when_each_distinct_workflow_is_charged_once(store):
    preview = asyncio.run(GridExperiments(Jobs(store, Driver())).preview(large_workflow_grid()))
    assert preview["ready"] and preview["total"] == preview["job_count"] == 1024
    assert len(store.list("previews", limit=200)) == 200


def test_excluded_cells_are_neither_built_nor_charged(store, monkeypatch):
    jobs = Jobs(store, Driver())
    built = []
    original = jobs.build_preview

    def counted(request, **kwargs):
        built.append(request["values"])
        return original(request, **kwargs)
    monkeypatch.setattr(jobs, "build_preview", counted)
    preview = asyncio.run(GridExperiments(jobs).preview(large_workflow_grid(excluded=range(0, 1024, 2))))
    assert preview["job_count"] == 512 and len(built) == 512
    assert all("preview_id" not in cell for cell in preview["cells"] if cell["excluded"])
    assert all("preview_id" in cell for cell in preview["cells"] if not cell["excluded"])


def written_bytes(store, *kinds):
    """Bytes of the stored rows of these kinds plus the shared workflow copies."""
    with store.lock:
        rows = store.db.execute(f"SELECT COALESCE(SUM(length(CAST(document AS BLOB))),0) FROM objects "  # noqa: S608
                                f"WHERE kind IN ({','.join('?' for _ in kinds)})", kinds).fetchone()[0]
        shared = store.db.execute("SELECT COALESCE(SUM(length(CAST(document AS BLOB))),0) FROM preview_workflows").fetchone()[0]
    return rows, shared


def preview_and_submit(store, request, monkeypatch):
    """Preview a grid, then submit it without dispatching; return the bytes each step wrote."""
    jobs = Jobs(store, Driver())
    monkeypatch.setattr(jobs, "launch", lambda coroutine: coroutine.close())
    grids = GridExperiments(jobs)
    preview = asyncio.run(grids.preview(request))
    rows, shared = written_bytes(store, "previews", "grid_previews")
    reviewed = rows + shared
    run = grids.submit({"preview_id": preview["id"], "idempotency_key": "grid-bytes"})
    rows, after = written_bytes(store, "jobs", "grid_runs")
    return preview, run, reviewed, rows + after - shared


def test_a_full_grid_of_a_large_workflow_submits_within_the_grid_budget(store, monkeypatch):
    # Each child job keeps the workflow under workflow, requested and effective;
    # stored three times per cell, 1,024 cells of this graph would write about 75 MB.
    preview, run, reviewed, submitted = preview_and_submit(store, large_workflow_grid(), monkeypatch)
    assert preview["job_count"] == 1024 and len(run["cells"]) == 1024
    assert reviewed <= experiments.GRID_BYTES and submitted <= experiments.GRID_BYTES
    job = store.get("jobs", run["cells"][0]["job_id"])
    layout = preview["request"]["draft"]["workflow"]["extra"]["layout"]
    assert job["submission"]["workflow"]["extra"]["layout"] == layout
    assert job["submission"]["requested"]["workflow"] == job["submission"]["effective"]["workflow"] == job["submission"]["workflow"]
    with store.lock:
        stored = "".join(row[0] for row in store.db.execute("SELECT document FROM objects WHERE kind='jobs'"))
    assert layout not in stored


def test_the_grid_budget_bounds_what_preview_and_submit_write_at_its_edge(tmp_path, monkeypatch):
    request = large_workflow_grid()
    request["axes"][1]["values"] = request["axes"][1]["values"][:4]

    admitted = {}

    def attempt(limit):
        monkeypatch.setattr(experiments, "GRID_BYTES", limit)
        store = Store(tmp_path / f"budget-{len(admitted)}-{limit}")
        try:
            admitted[limit] = preview_and_submit(store, request, monkeypatch)[2:]
            return True
        except StudioError as exc:
            assert exc.code == "grid_size"
            return False
        finally:
            store.close()

    low, high = 1, 64 * 1024 * 1024
    assert attempt(high)
    while low < high:
        middle = (low + high) // 2
        low, high = (low, middle) if attempt(middle) else (middle + 1, high)
    # The smallest budget that admitted the grid still bounds both writes.
    reviewed, submitted = admitted[high]
    assert reviewed <= high and submitted <= high


def test_a_grid_axis_can_name_a_child_input_the_draft_choice_reveals(store):
    # The saved "plain" mode has no strength input; the draft picks "tuned", so an
    # axis over its strength child is a typed axis and each cell compiles with it.
    driver = Driver()
    driver.schema = {"Render": {"input": {"required": {
        "mode": ["COMFY_DYNAMICCOMBO_V3", {"options": [
            {"key": "plain", "inputs": {"required": {}}},
            {"key": "tuned", "inputs": {"required": {"strength": ["INT", {"default": 1, "min": 0, "max": 9}]}}}]}],
        "text": ["STRING", {}]}}, "output": [], "output_node": True}}
    request = {"draft": {"workflow": {"1": {"class_type": "Render", "inputs": {"mode": "plain", "text": "cat"}}},
                         "values": {"root:1:mode": "tuned"}, "seed_policy": "fixed", "operation": "image"},
               "axes": [{"name": "root:1:mode.strength", "values": [2, 5]}]}
    preview = asyncio.run(GridExperiments(Jobs(store, driver)).preview(request))
    assert preview["ready"] and [cell["errors"] for cell in preview["cells"]] == [[], []]
    prompts = [store.get("previews", cell["preview_id"])["prompt"]["1"]["inputs"] for cell in preview["cells"]]
    assert [(inputs["mode"], inputs["mode.strength"]) for inputs in prompts] == [("tuned", 2), ("tuned", 5)]
