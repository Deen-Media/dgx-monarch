"""Portable library transactions preserve identities without importing execution."""
from copy import deepcopy

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.library import Library
from dgx_monarch.studio.store import Store


@pytest.fixture
def library(tmp_path):
    store = Store(tmp_path / "state")
    yield Library(store)
    store.close()


def exported():
    return {"format": "dgxm.studio.library", "schema_version": "1.0", "vendor_extra": {"keep": [1, 2]}, "records": {
        "projects": [{"id": "p1", "revision": 7, "data": {"name": "Project"}}],
        "styles": [{"id": "s1", "revision": 9, "vendor": "retained", "data": {"name": "Style", "project_ids": ["p1"], "prompt": "{prompt}", "unknown": 42}}],
        "sessions": [{"id": "session1", "revision": 2, "data": {"name": "Session", "project_ids": ["p1"], "active_job_ids": ["missing-job"], "draft": {"project_id": "p1", "parent_artifact_ids": ["missing-image"]}}}],
    }}


def apply(library, preview, key="import-once"):
    return library.apply({"preview_id": preview["id"], "digest": preview["digest"], "confirmed": True, "idempotency_key": key})


def test_roundtrip_preserves_metadata_and_references_without_dispatch(library):
    preview = library.preview({"library": exported()})
    assert preview["ready"] and len(preview["repair_list"]) == 2
    result = apply(library, preview)
    assert result["imported"] == 3
    assert apply(library, preview) == result
    assert library.store.all("jobs") == []
    document = library.export("p1")
    assert document["records"]["styles"][0]["vendor"] == "retained"
    assert document["import_metadata"][0]["data"]["vendor_extra"] == {"keep": [1, 2]}
    assert document["records"]["styles"][0]["data"]["unknown"] == 42
    style = document["records"]["styles"][0]
    library.save("styles", style["id"], {**style["data"], "name": "Edited"}, style["revision"])
    assert library.store.get("styles", "s1")["vendor"] == "retained"


def test_conflicts_require_review_and_replacement_is_atomic(library):
    apply(library, library.preview({"library": exported()}))
    incoming = exported()
    incoming["records"]["styles"][0]["data"]["prompt"] = "different"
    blocked = library.preview({"library": incoming})
    assert not blocked["ready"] and len(blocked["conflicts"]) == 1
    with pytest.raises(StudioError, match="Resolve conflicts"):
        apply(library, blocked, "blocked")
    approved = library.preview({"library": incoming, "choices": {"styles:s1": {"action": "replace", "revision": 1}}})
    library.save("projects", "p1", {"name": "Concurrent"}, 1)
    with pytest.raises(StudioError, match="no records"):
        apply(library, approved, "stale")
    assert library.store.get("styles", "s1")["data"]["prompt"] == "{prompt}"


def test_copy_remaps_membership_and_known_embedded_references(library):
    source = exported()
    source["records"]["sessions"][0]["data"]["draft"]["styles"] = [{"id": "s1", "prompt": "a"}]
    before = deepcopy(source)
    preview = library.preview({"library": source, "mode": "copy"})
    mapping = apply(library, preview)["id_mapping"]
    assert mapping["p1"] != "p1"
    style = library.store.get("styles", mapping["s1"])
    session = library.store.get("sessions", mapping["session1"])
    assert style["data"]["project_ids"] == [mapping["p1"]]
    assert session["data"]["draft"]["project_id"] == mapping["p1"]
    assert session["data"]["draft"]["styles"][0]["id"] == mapping["s1"]
    assert source == before


def test_project_filter_and_artifact_membership_do_not_edit_submission(library):
    apply(library, library.preview({"library": exported()}))
    library.save("styles", "global", {"name": "Shared"}, 0)
    assert [row["id"] for row in library.page("styles", {"project_id": "p1"})["records"]] == ["s1"]
    library.store.create("jobs", {"id": "job", "submission": {"project_id": "old"}})
    library.store.create("artifacts", {"id": "image", "job_id": "job", "project_id": "old"})
    saved = library.membership("image", {"project_ids": ["p1"], "revision": 0})
    assert saved["membership_revision"] == 1 and saved["project_ids"] == ["p1"]
    assert library.store.get("jobs", "job")["submission"]["project_id"] == "old"
    with pytest.raises(StudioError, match="membership changed"):
        library.membership("image", {"project_ids": [], "revision": 0})


@pytest.mark.parametrize("change", [
    {"mode": []}, {"library": {"schema_version": "2.0", "records": {}}},
    {"library": {"schema_version": "1.0", "records": {"jobs": []}}},
])
def test_invalid_imports_refuse_without_partial_changes(library, change):
    request = {"library": deepcopy(exported()), **change}
    with pytest.raises(StudioError):
        library.preview(request)
    assert library.store.all("projects") == []
