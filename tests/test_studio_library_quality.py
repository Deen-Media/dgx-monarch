"""Regression checks for durable membership, portable references and bounded pages."""
import io
import json
from copy import deepcopy

import pytest
from PIL import Image

from dgx_monarch.studio import library as library_module
from dgx_monarch.studio import live_preview
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.library import Library
from dgx_monarch.studio.store import Store


@pytest.fixture
def library(tmp_path):
    store = Store(tmp_path / "state")
    yield Library(store)
    store.close()


def test_new_preview_frame_preserves_membership_extensions_and_sort_identity(library, monkeypatch):
    store = library.store
    store.create("jobs", {"id": "job", "prompt_id": "prompt", "state": "running"})
    library.save("projects", "project", {"name": "Project"}, 0)
    clock = [10.0]
    monkeypatch.setattr(live_preview, "timestamp", lambda: clock[0])

    def frame(color):
        output = io.BytesIO()
        Image.new("RGB", (4, 4), color).save(output, format="PNG")
        live_preview.accept_preview(store, {"data": {"prompt_id": "prompt", "node_id": "node"},
                                            "image": output.getvalue()})

    frame("red")
    original = store.list("artifacts")[0]
    library.membership(original["id"], {"project_ids": ["project"], "revision": 0})
    store.update("artifacts", original["id"], {"extension": {"note": "retained"},
                                              "metadata": {**original["metadata"], "extra": 42}})
    clock[0] = 20.0
    frame("blue")
    updated = store.get("artifacts", original["id"])
    assert updated["project_ids"] == ["project"] and updated["membership_revision"] == 1
    assert updated["extension"] == {"note": "retained"} and updated["metadata"]["extra"] == 42
    assert updated["created_at"] == original["created_at"] == 10.0
    assert updated["updated_at"] == 20.0
    assert updated["metadata"]["sha256"] != original["metadata"]["sha256"]
    library.membership(original["id"], {"project_ids": [], "revision": 1})


def test_copy_import_remaps_grid_definition_but_preserves_historic_evidence(library):
    definition = {"draft": {"project_id": "project", "styles": [{"id": "style", "prompt": "paint"}]},
                  "axes": [{"name": "styles", "values": [["style"], ["external", "style"]]},
                           {"name": "root:1:text", "values": ["style"]}]}
    historical = {"request": deepcopy(definition), "cells": [{"submission_digest": "frozen", "style_id": "style"}]}
    source = {"schema_version": "1.0", "records": {
        "projects": [{"id": "project", "data": {"name": "Project"}}],
        "styles": [{"id": "style", "data": {"prompt": "paint"}}],
        "experiments": [{"id": "experiment", "data": {"project_ids": ["project"],
                        "grid_request": definition, "grid_snapshot": historical}}]}}
    original = deepcopy(source)
    plan = library.preview({"mode": "copy", "library": source})
    result = library.apply({"preview_id": plan["id"], "digest": plan["digest"],
                            "confirmed": True, "idempotency_key": "copy"})
    mapping = result["typed_id_mapping"]
    copied = library.store.get("experiments", mapping["experiments"]["experiment"])["data"]
    assert copied["grid_request"]["draft"]["project_id"] == mapping["projects"]["project"]
    assert copied["grid_request"]["draft"]["styles"][0]["id"] == mapping["styles"]["style"]
    assert copied["grid_request"]["axes"][0]["values"] == [[mapping["styles"]["style"]], ["external", mapping["styles"]["style"]]]
    assert copied["grid_request"]["axes"][1]["values"] == ["style"]
    assert copied["grid_snapshot"] == historical and source == original


def test_record_byte_pages_advance_exactly_without_collecting_all_rows(library, monkeypatch):
    monkeypatch.setattr(library_module, "PAGE_BYTES", 2400)
    for index in range(12):
        library.save("workflows", str(index), {"name": str(index), "extension": "é" * 120}, 0)

    class Cursor:
        def __init__(self, cursor):
            self.cursor = cursor
        def __iter__(self):
            return iter(self.cursor)
        def close(self):
            self.cursor.close()
        def fetchall(self):
            pytest.fail("Library pagination materialized the complete requested page")

    class Connection:
        def execute(self, *args):
            return Cursor(database.execute(*args))

    database = library.store.db
    with monkeypatch.context() as patch:
        patch.setattr(library.store, "db", Connection())
        identities, offset = [], 0
        while offset is not None:
            page = library.page("workflows", {"limit": "200", "offset": str(offset)})
            assert len(json.dumps(page).encode()) <= library_module.PAGE_BYTES
            assert page["records"]
            identities.extend(row["id"] for row in page["records"])
            next_offset = page["next_offset"]
            assert next_offset is None or next_offset == offset + len(page["records"])
            offset = next_offset
    assert len(identities) == len(set(identities)) == 12
    assert set(identities) == {str(index) for index in range(12)}


def test_export_refuses_at_budget_without_loading_full_collections(library, monkeypatch):
    for index in range(8):
        library.save("workflows", str(index), {"name": str(index), "extension": "x" * 500}, 0)
    monkeypatch.setattr(library_module, "MAX_LIBRARY_BYTES", 1500)
    monkeypatch.setattr(library.store, "all", lambda kind: pytest.fail("Export loaded the entire collection"))
    with pytest.raises(StudioError, match="Export exceeds"):
        library.export()


def test_oversized_individual_record_refuses_instead_of_nonadvancing_page(library, monkeypatch):
    library.save("workflows", "large", {"extension": "x" * 2000}, 0)
    monkeypatch.setattr(library_module, "PAGE_BYTES", 1000)
    with pytest.raises(StudioError, match="record exceeds"):
        library.page("workflows", {})


def test_page_offset_is_bounded_before_sqlite_integer_conversion(library):
    with pytest.raises(StudioError, match="Offset"):
        library.page("styles", {"offset": str(2**100)})


def test_live_preview_uses_indexed_exact_prompt_match_and_refuses_ambiguity(library, monkeypatch):
    store = library.store
    store.create("jobs", {"id": "target", "prompt_id": "prompt", "state": "running"})
    for index in range(24):
        store.create("jobs", {"id": f"other-{index}", "prompt_id": f"other-prompt-{index}",
                              "state": "succeeded", "submission": {"workflow": "x" * 10000}})
    monkeypatch.setattr(store, "all", lambda kind: pytest.fail("Preview loaded the complete job history"))
    original_get = store.get

    def get(kind, identity):
        assert kind != "jobs" or identity == "target", "Preview loaded an unrelated submission"
        return original_get(kind, identity)

    monkeypatch.setattr(store, "get", get)
    output = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(output, format="PNG")
    message = {"data": {"prompt_id": "prompt", "node_id": "node"}, "image": output.getvalue()}
    live_preview.accept_preview(store, message)
    artifacts = store.list("artifacts")
    assert len(artifacts) == 1 and artifacts[0]["job_id"] == "target"
    store.create("jobs", {"id": "duplicate", "prompt_id": "prompt", "state": "running"})
    cursor = store.events()["latest_cursor"]
    live_preview.accept_preview(store, message)
    assert store.events()["latest_cursor"] == cursor
    assert store.list("artifacts") == artifacts
