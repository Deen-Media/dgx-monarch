"""Portable library transactions preserve identities without importing execution."""
import json
from copy import deepcopy

import pytest

import dgx_monarch.studio.library as library_module
import dgx_monarch.studio.library_export as export_module
from dgx_monarch.studio.contracts import RECORD_LIMIT, StudioError
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


def size(document):
    return len(json.dumps(document).encode())


def add_artifacts(store, count, project="p1", text=20_000):
    with store.lock, store.transaction():
        for index in range(count):
            # Each observed output carries a repr of the full request as search text.
            store._put("artifacts", {"id": f"artifact-{index:05d}", "job_id": "job", "kind": "image",
                                     "filename": f"{index}.png", "project_ids": [project], "created_at": index,
                                     "search_text": "{'workflow': " + "x" * text + "}"}, 1)


def test_export_and_reimport_with_artifacts_does_not_grow(library):
    store = library.store
    library.save("projects", "p1", {"name": "Project"}, 0)
    add_artifacts(store, 40)
    first = library.export("p1")
    assert len(first["artifact_references"]) == 40
    assert all("search_text" not in row for row in first["artifact_references"])
    assert size(first) < 40 * 1000
    # The gallery still searches the stored text in place.
    assert "search_text" in store.get("artifacts", "artifact-00000")
    plan = library.preview({"library": first})
    assert plan["repair_list"] == []
    apply(library, plan, "merge-round-trip")
    second = library.export("p1")
    assert second == first and size(second) <= size(first)
    # A copy keeps each reference once, for review, and stops growing there.
    result = apply(library, library.preview({"library": first, "mode": "copy"}), "copy-1")
    copied = result["typed_id_mapping"]["projects"]["p1"]
    third = library.export(copied)
    assert len(third["unresolved_references"]) == 40 and len(store.all("library_references")) == 40
    assert {row["reason"] for row in third["unresolved_references"]} == {"source_membership_review"}
    assert all("search_text" not in row["reference"] for row in third["unresolved_references"])
    result = apply(library, library.preview({"library": third, "mode": "copy"}), "copy-2")
    fourth = library.export(result["typed_id_mapping"]["projects"][copied])
    assert len(fourth["unresolved_references"]) == 40 and size(fourth) <= size(third)
    # Two stored copies of each reference with its search text came to about 1.6 MB.
    assert size(third) < 40 * 1000


def test_an_older_export_with_search_text_and_doubled_references_imports_once(library):
    library.save("projects", "p1", {"name": "Project"}, 0)
    add_artifacts(library.store, 1)
    reference = {**library.store.get("artifacts", "artifact-00000"), "project_ids": ["p1"]}
    retained = {"artifact_id": reference["id"], "reference": reference, "project_ids": ["p1"]}
    document = {"schema_version": "1.0", "records": {"projects": [{"id": "p2", "data": {"name": "Other"}}]},
                "unresolved_references": [retained, {**retained, "reason": "source_artifact_missing"}]}
    apply(library, library.preview({"library": document}), "older")
    rows = library.store.all("library_references")
    assert len(rows) == 1 and rows[0]["reason"] == "source_artifact_missing"
    assert "search_text" not in rows[0]["reference"]


def test_export_row_limit_matches_what_import_accepts(library):
    assert export_module.EXPORT_ROWS == RECORD_LIMIT
    library.save("projects", "p1", {"name": "Project"}, 0)
    add_artifacts(library.store, RECORD_LIMIT, text=10)
    document = library.export()
    assert len(document["artifact_references"]) == RECORD_LIMIT
    assert library.preview({"library": document})["ready"]
    add_artifacts(library.store, RECORD_LIMIT + 1, text=10)
    with pytest.raises(StudioError, match="more than 2000 artifact references, the most one import accepts"):
        library.export()
    with pytest.raises(StudioError, match="Artifact references must be a bounded array"):
        library.preview({"library": {**document, "artifact_references": document["artifact_references"] * 2}})


def test_export_refusals_do_not_send_a_project_caller_in_a_circle(library, monkeypatch):
    library.save("projects", "p1", {"name": "Project"}, 0)
    add_artifacts(library.store, 4, text=10)
    monkeypatch.setattr(export_module, "EXPORT_ROWS", 3)
    with pytest.raises(StudioError, match="export one project at a time"):
        library.export()
    with pytest.raises(StudioError) as caught:
        library.export("p1")
    assert str(caught.value).startswith("This project would export more than 3 artifact references")
    assert "project at a time" not in str(caught.value) and "single project" not in str(caught.value)
    monkeypatch.setattr(export_module, "EXPORT_ROWS", RECORD_LIMIT)
    monkeypatch.setattr(library_module, "MAX_LIBRARY_BYTES", 700)
    with pytest.raises(StudioError, match=r"^Export exceeds .*export one project at a time"):
        library.export()
    with pytest.raises(StudioError) as caught:
        library.export("p1")
    assert caught.value.code == "library_limit" and caught.value.status == 413
    assert str(caught.value).startswith("This project's export passed")
    assert "while adding artifact references" in str(caught.value)
    assert "project at a time" not in str(caught.value) and "single project" not in str(caught.value)


def browser_catalog_bytes(store, kind):
    # The browser sums JSON.stringify of every record plus one, and two for the array.
    return 2 + sum(len(json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode()) + 1
                   for row in store.all(kind))


def test_record_saves_stay_inside_the_browser_catalog_window(library, monkeypatch):
    monkeypatch.setattr(library_module, "RECORD_WINDOW_BYTES", 4000)
    store = library.store
    library.save("presets", "a", {"name": "A", "notes": "é" * 600}, 0)
    library.save("presets", "b", {"name": "B", "notes": "n" * 1500}, 0)
    cursor = store.events()["latest_cursor"]
    with pytest.raises(StudioError) as caught:
        library.save("presets", "c", {"name": "C", "notes": "n" * 1500}, 0)
    assert (caught.value.code, caught.value.status) == ("record_window_limit", 413)
    assert "8 MiB browser window" in str(caught.value)
    assert [row["id"] for row in store.all("presets")] == ["a", "b"] and store.events()["latest_cursor"] == cursor
    assert browser_catalog_bytes(store, "presets") <= 4000
    # Growing an existing record past the window is refused; shrinking is not.
    with pytest.raises(StudioError, match="browser window"):
        library.save("presets", "b", {"name": "B", "notes": "n" * 3000}, 1)
    library.save("presets", "b", {"name": "B", "notes": "n" * 100}, 1)
    library.save("presets", "c", {"name": "C", "notes": "n" * 1500}, 0)
    # A catalog already past a lowered window still takes edits that do not grow it.
    monkeypatch.setattr(library_module, "RECORD_WINDOW_BYTES", 1000)
    library.save("presets", "c", {"name": "D", "notes": "n" * 1500}, 1)
    # A stale revision keeps its own refusal.
    with pytest.raises(StudioError, match="newer version"):
        library.save("presets", "c", {"name": "E", "notes": "n" * 9000}, 1)


def presets(*names, notes=1500):
    return {"schema_version": "1.0",
            "records": {"presets": [{"id": name, "data": {"name": name, "notes": "n" * notes}} for name in names]}}


def test_imports_stay_inside_the_browser_catalog_window(library, monkeypatch):
    # Each import passes the import size limit, but together they could take a
    # kind past the catalog window the browser loads whole.
    monkeypatch.setattr(library_module, "RECORD_WINDOW_BYTES", 4000)
    store = library.store
    apply(library, library.preview({"library": presets("a", "b")}), "first")
    assert browser_catalog_bytes(store, "presets") <= 4000
    cursor = store.events()["latest_cursor"]
    # The review refuses an import that would pass the window.
    with pytest.raises(StudioError) as caught:
        library.preview({"library": presets("c")})
    assert (caught.value.code, caught.value.status) == ("record_window_limit", 413)
    assert "8 MiB browser window" in str(caught.value) and "presets" in str(caught.value)
    # Apply checks again inside its transaction: a save since the review grew the catalog.
    monkeypatch.setattr(library_module, "RECORD_WINDOW_BYTES", 4500)
    review = library.preview({"library": presets("c", notes=800)})
    library.save("presets", "d", {"name": "d", "notes": "n" * 800}, 0)
    cursor = store.events()["latest_cursor"]
    with pytest.raises(StudioError) as caught:
        apply(library, review, "late")
    assert (caught.value.code, caught.value.status) == ("record_window_limit", 413)
    assert "no records were imported" in str(caught.value)
    assert sorted(row["id"] for row in store.all("presets")) == ["a", "b", "d"]
    assert store.events()["latest_cursor"] == cursor and len(store.all("library_imports")) == 1
    assert browser_catalog_bytes(store, "presets") <= 4500
    # A catalog already past a lowered window still takes an import that shrinks it.
    monkeypatch.setattr(library_module, "RECORD_WINDOW_BYTES", 1000)
    smaller = presets("a", notes=10)
    review = library.preview({"library": smaller, "choices": {"presets:a": {"action": "replace", "revision": 1}}})
    assert apply(library, review, "shrink")["imported"] == 1
    assert store.get("presets", "a")["data"]["notes"] == "n" * 10
