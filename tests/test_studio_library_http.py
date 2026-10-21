"""Authenticated library and real-journal paging through companion HTTP routes."""
from __future__ import annotations

import asyncio
import subprocess
from contextlib import asynccontextmanager

import pytest
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.runtime_observations import record_event
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store


@pytest.fixture(autouse=True)
def no_external_commands(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("Library and history tests must not execute commands")
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", denied)


@asynccontextmanager
async def companion(tmp_path):
    store = Store(tmp_path / "state")
    allowed = set()
    token = "owner-" + "x" * 32
    app = create_app(store, ComfyClient("http://127.0.0.1:8188"), token, allowed_hosts=allowed, poll=False)
    try:
        async with TestClient(TestServer(app)) as browser:
            origin = str(browser.make_url("/")).rstrip("/")
            allowed.add(origin.split("://", 1)[1])
            headers = {"Authorization": "Bearer " + token}
            async def request(method, path, expected=200, **kwargs):
                response = await browser.request(method, "/api/v1" + path, headers=headers, **kwargs)
                assert response.status == expected, await response.text()
                return await response.json()
            yield store, browser, headers, request
    finally:
        store.close()


def runtime_event(sequence, render_id):
    return {"schema_version": 1, "producer_id": "producer", "sequence": sequence, "timestamp": float(sequence),
            "prompt_id": "prompt", "node_id": "node", "render_id": render_id, "list_index": 0,
            "fleet_index": None, "kind": "dispatch_submitted", "facts": {}}


def test_runtime_http_pages_real_journal_and_legacy_rows_without_skipping_updates(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (store, browser, _headers, request):
            store.create("jobs", {"id": "job", "prompt_id": "prompt", "state": "succeeded"})
            store.create("jobs", {"id": "other", "state": "succeeded"})
            for index in range(45):
                record_event(store, runtime_event(index + 1, f"render-{index}"), "job")
            for index in range(2):
                store.create("runtime_renders", {"id": f"legacy-{index}", "job_id": "job", "render_id": f"legacy-{index}"})
            response = await browser.get("/api/v1/jobs/job/runtime")
            assert response.status == 401
            # Legacy rows have no timestamp. A one-row page must still produce
            # a usable cursor rather than indexing missing created_at.
            first = await request("GET", "/jobs/job/runtime?limit=1")
            assert first["next_cursor"] and first["items"][0]["render_id"] == "legacy-0"
            await request("GET", "/jobs/other/runtime", expected=400, params={"cursor": first["next_cursor"]})
            items, cursor = first["items"], first["next_cursor"]
            while cursor:
                page = await request("GET", "/jobs/job/runtime", params={"limit": 10, "cursor": cursor})
                items.extend(page["items"])
                cursor = page["next_cursor"]
                # Enrich previously paged render evidence while paging. Its
                # immutable sort position must not move into a later page.
                record_event(store, runtime_event(46, "render-0"), "job")
            assert len(items) == 47 and len({row["id"] for row in items}) == 47
            assert {row["render_id"] for row in items} == {f"render-{index}" for index in range(45)} | {"legacy-0", "legacy-1"}
            event_page = await request("GET", "/jobs/job/runtime?kind=events&limit=100")
            assert len(event_page["items"]) == 46 and event_page["next_cursor"] is None
            await request("GET", "/jobs/job/runtime?limit=101", expected=400)
            await request("GET", "/jobs/missing/runtime", expected=404)
    asyncio.run(scenario())


def library_document():
    return {"format": "dgxm.studio.library", "schema_version": "1.0", "records": {
        "projects": [{"id": "project", "revision": 1, "data": {"name": "Project"}}],
        "styles": [{"id": "style", "revision": 1, "vendor": {"keep": True},
                    "data": {"name": "Style", "project_ids": ["project"], "prompt": "prefix {prompt}"}}]}}


def test_library_http_review_revision_idempotency_membership_and_project_filter(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (store, browser, headers, request):
            denied = await browser.get("/api/v1/library/export")
            assert denied.status == 401
            foreign = await browser.post("/api/v1/library/import/preview", json={"library": library_document()},
                                         headers={**headers, "Origin": "https://foreign.invalid"})
            assert foreign.status == 403
            plan = await request("POST", "/library/import/preview", json={"library": library_document()})
            body = {"preview_id": plan["id"], "digest": plan["digest"], "confirmed": True, "idempotency_key": "library-import"}
            await request("POST", "/library/import", expected=409, json={**body, "confirmed": False})
            applied = await request("POST", "/library/import", json=body)
            assert applied["imported"] == 2
            assert await request("POST", "/library/import", json=body) == applied
            page = await request("GET", "/records/styles?project_id=project&limit=1")
            assert [row["id"] for row in page["records"]] == ["style"]
            assert page["next_offset"] is None
            saved = await request("PUT", "/records/styles/style", json={"revision": 1, "data": {"name": "Changed", "project_ids": ["project"]}})
            assert saved["vendor"] == {"keep": True} and saved["revision"] == 2
            await request("PUT", "/records/styles/style", expected=409, json={"revision": 1, "data": {}})
            store.create("artifacts", {"id": "artifact", "created_at": 1, "project_id": "old"})
            member = await request("PUT", "/artifacts/artifact/projects", json={"project_ids": ["project"], "revision": 0})
            assert member["membership_revision"] == 1
            assert [row["id"] for row in (await request("GET", "/artifacts?project_id=project"))["items"]] == ["artifact"]
            assert (await request("GET", "/artifacts?project_id=old"))["items"] == []
            await request("PUT", "/artifacts/artifact/projects", expected=409, json={"project_ids": [], "revision": 0})
            exported = await request("GET", "/library/export?project_id=project")
            assert exported["records"]["styles"][0]["revision"] == 2
            assert exported["artifact_references"][0]["id"] == "artifact"
    asyncio.run(scenario())


@pytest.mark.parametrize("data", [
    {"draft": {"styles": None}}, {"draft": {"parent_artifact_ids": None}},
    {"draft": {"parent_artifact_ids": "not-an-array"}},
])
def test_library_http_rejects_malformed_reference_arrays(tmp_path, data):
    async def scenario():
        async with companion(tmp_path) as (store, _browser, _headers, request):
            await request("POST", "/library/import/preview", expected=400, json={"library": {
                "schema_version": "1.0", "records": {"sessions": [{"id": "bad", "data": data}]}}})
            assert store.all("sessions") == [] and store.all("library_previews") == []
    asyncio.run(scenario())


def test_library_http_rejects_nested_projects(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (store, _browser, _headers, request):
            await request("POST", "/library/import/preview", expected=400, json={"library": {
                "schema_version": "1.0", "records": {"projects": [{"id": "nested", "data": {"project_ids": ["nested"]}}]}}})
            assert store.all("projects") == []
    asyncio.run(scenario())


def test_project_export_does_not_include_other_imports_metadata_or_references(tmp_path):
    import json

    async def scenario():
        async with companion(tmp_path) as (_store, _browser, _headers, request):
            for project in ("first", "second-private"):
                document = {"schema_version": "1.0", "private_vendor": project, "records": {
                    "projects": [{"id": project, "data": {"name": project}}],
                    "sessions": [{"id": project + "-session", "data": {"project_ids": [project],
                                  "selected_artifact_id": project + "-missing-artifact"}}]}}
                plan = await request("POST", "/library/import/preview", json={"library": document})
                await request("POST", "/library/import", json={"preview_id": plan["id"], "digest": plan["digest"],
                              "confirmed": True, "idempotency_key": project})
            exported = await request("GET", "/library/export?project_id=first")
            assert "second-private" not in json.dumps(exported)
            assert "first-missing-artifact" in json.dumps(exported)
    asyncio.run(scenario())


def test_library_copy_does_not_remap_external_style_as_same_named_project(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (_store, _browser, _headers, request):
            await request("PUT", "/records/styles/shared", json={"revision": 0, "data": {"name": "External style"}})
            document = {"schema_version": "1.0", "records": {
                "projects": [{"id": "shared", "data": {"name": "Copied project"}}],
                "sessions": [{"id": "session", "data": {"draft": {"styles": [{"id": "shared"}]}}}]}}
            plan = await request("POST", "/library/import/preview", json={"library": document, "mode": "copy"})
            session = next(change for change in plan["changes"] if change["kind"] == "sessions")
            assert session["data"]["draft"]["styles"][0]["id"] == "shared"
    asyncio.run(scenario())


def test_unchanged_library_roundtrip_does_not_nest_export_metadata(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (_store, _browser, _headers, request):
            document = {**library_document(), "vendor": {"preserve": "meaningful"}}
            expected_metadata_count = None
            for index in range(3):
                plan = await request("POST", "/library/import/preview", json={"library": document})
                await request("POST", "/library/import", json={"preview_id": plan["id"], "digest": plan["digest"],
                              "confirmed": True, "idempotency_key": f"roundtrip-{index}"})
                document = await request("GET", "/library/export")
                if expected_metadata_count is None:
                    expected_metadata_count = len(document["import_metadata"])
                assert len(document["import_metadata"]) == expected_metadata_count
                assert all("import_metadata" not in row["data"] for row in document["import_metadata"])
    asyncio.run(scenario())


def test_library_copy_retains_artifact_reference_without_mutating_membership(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (store, _browser, _headers, request):
            artifact = store.create("artifacts", {"id": "artifact", "project_ids": ["project"], "created_at": 1})
            document = {**library_document(), "artifact_references": [artifact]}
            plan = await request("POST", "/library/import/preview", json={"library": document, "mode": "copy"})
            copied_project = plan["typed_id_mapping"]["projects"]["project"]
            repair = next(row for row in plan["repair_list"] if row["reason"] == "source_membership_review")
            assert repair["project_ids"] == [copied_project]
            assert repair["reference"]["project_ids"] == [copied_project]
            for index in range(3):
                await request("POST", "/library/import", json={"preview_id": plan["id"], "digest": plan["digest"],
                              "confirmed": True, "idempotency_key": f"copy-reference-{index}"})
                assert store.get("artifacts", "artifact") == artifact
                document = await request("GET", f"/library/export?project_id={copied_project}")
                assert document["artifact_references"] == []
                # One retained row per reference: the review reason travels with it.
                assert [row["reason"] for row in document["unresolved_references"]] == ["source_membership_review"]
                assert all(row["reference"]["project_ids"] == [copied_project] for row in document["unresolved_references"])
                plan = await request("POST", "/library/import/preview", json={"library": document})
    asyncio.run(scenario())


@pytest.mark.parametrize("collection", ["library_metadata", "library_references"])
def test_library_auxiliary_capacity_is_atomic_and_deduplicates(tmp_path, monkeypatch, collection):
    import dgx_monarch.studio.library as module

    monkeypatch.setattr(module, "AUXILIARY_LIMIT", 1)
    async def scenario():
        async with companion(tmp_path) as (store, _browser, _headers, request):
            def document(name):
                result = {"schema_version": "1.0", "records": {"projects": [{"id": name, "data": {"name": name}}]}}
                if collection == "library_metadata":
                    result["vendor"] = name
                else:
                    result["unresolved_references"] = [{"source_id": name, "project_ids": [name]}]
                return result
            plans = [await request("POST", "/library/import/preview", json={"library": document(name)})
                     for name in ("first", "second")]
            def body(plan, key):
                return {"preview_id": plan["id"], "digest": plan["digest"], "confirmed": True, "idempotency_key": key}
            await request("POST", "/library/import", json=body(plans[0], "first"))
            await request("POST", "/library/import", expected=409, json=body(plans[1], "second"))
            assert [row["id"] for row in store.all("projects")] == ["first"]
            assert len(store.all(collection)) == 1 and len(store.all("library_imports")) == 1
            assert store.db.execute("SELECT count(*) FROM requests WHERE key='second'").fetchone()[0] == 0
            await request("POST", "/library/import/preview", expected=409, json={"library": document("third")})
            # Exported canonical rows must remain importable at the cap.
            exported = await request("GET", "/library/export")
            plan = await request("POST", "/library/import/preview", json={"library": exported})
            await request("POST", "/library/import", json=body(plan, "same-content"))
            assert len(store.all(collection)) == 1
    asyncio.run(scenario())


def test_library_capacity_bytes_preview_retention_and_receipt_reconciliation(tmp_path, monkeypatch):
    import dgx_monarch.studio.library as module

    monkeypatch.setattr(module, "PREVIEW_LIMIT", 1)
    monkeypatch.setattr(module, "IMPORT_LIMIT", 1)
    async def scenario():
        async with companion(tmp_path) as (store, _browser, _headers, request):
            monkeypatch.setattr(module, "AUXILIARY_BYTES", 100)
            large = {"schema_version": "1.0", "records": {}, "vendor": "x" * 200}
            await request("POST", "/library/import/preview", expected=413, json={"library": large})
            assert store.all("library_previews") == [] and store.all("library_metadata") == []
            monkeypatch.setattr(module, "AUXILIARY_BYTES", 10000)
            plan = await request("POST", "/library/import/preview", json={"library": library_document()})
            await request("POST", "/library/import/preview", expected=409, json={"library": library_document()})
            body = {"preview_id": plan["id"], "digest": plan["digest"], "confirmed": True, "idempotency_key": "receipt"}
            receipt = await request("POST", "/library/import", json=body)
            # Expired previews can be removed, but successful receipt replay
            # does not depend on retaining the preview or spare receipt space.
            with store.lock, store.transaction():
                store._put("library_previews", {**plan, "expires_at": 0}, 1)
            document = {"schema_version": "1.0", "records": {"projects": [{"id": "second", "data": {}}]}}
            second = await request("POST", "/library/import/preview", json={"library": document})
            assert [row["id"] for row in store.all("library_previews")] == [second["id"]]
            await request("POST", "/library/import", expected=409, json={"preview_id": second["id"],
                          "digest": second["digest"], "confirmed": True, "idempotency_key": "new-receipt"})
            assert {row["id"] for row in store.all("projects")} == {"project"}
            assert await request("POST", "/library/import", json=body) == receipt
            assert len(store.all("library_imports")) == 1
    asyncio.run(scenario())


def test_library_import_preserves_frozen_grid_and_reports_missing_source(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (store, _browser, _headers, request):
            frozen = {"draft": {"prompt": "preserved"}, "axes": [], "vendor": "keep"}
            document = {"schema_version": "1.0", "records": {"experiments": [{"id": "experiment",
                        "data": {"grid_id": "remote-grid", "grid_request": frozen}}]}}
            plan = await request("POST", "/library/import/preview", json={"library": document})
            assert any(row["kind"] == "grid_runs" and row["id"] == "remote-grid" for row in plan["repair_list"])
            await request("POST", "/library/import", json={"preview_id": plan["id"], "digest": plan["digest"],
                          "confirmed": True, "idempotency_key": "grid-reference"})
            data = store.get("experiments", "experiment")["data"]
            assert data["grid_request"] == frozen
            assert data["grid_source"] == {"state": "unavailable", "grid_id": "remote-grid", "reason": "source_grid_missing"}
            assert store.all("grid_runs") == [] and store.all("jobs") == []
    asyncio.run(scenario())


def test_library_copy_uses_typed_maps_and_preserves_metadata_extensions(tmp_path):
    async def scenario():
        async with companion(tmp_path) as (_store, _browser, _headers, request):
            document = {"schema_version": "1.0", "records": {
                "projects": [{"id": "shared", "data": {"name": "Project"}}],
                "styles": [{"id": "shared", "data": {"project_ids": ["shared"]}}]},
                "import_metadata": [{"id": "old", "data": {"vendor": 1}, "envelope_vendor": 2, "project_ids": ["shared"]}]}
            plan = await request("POST", "/library/import/preview", json={"library": document, "mode": "copy"})
            assert "shared" not in plan["id_mapping"]
            project_id = plan["typed_id_mapping"]["projects"]["shared"]
            style_id = plan["typed_id_mapping"]["styles"]["shared"]
            assert project_id != style_id
            await request("POST", "/library/import", json={"preview_id": plan["id"], "digest": plan["digest"],
                          "confirmed": True, "idempotency_key": "typed-copy"})
            result = await request("GET", f"/library/export?project_id={project_id}")
            assert result["records"]["styles"][0]["id"] == style_id
            assert result["import_metadata"][0]["envelope_vendor"] == 2
            assert result["import_metadata"][0]["data"] == {"vendor": 1}
    asyncio.run(scenario())
