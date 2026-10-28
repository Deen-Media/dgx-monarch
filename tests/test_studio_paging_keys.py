"""Keyset cursors carry the exact key SQLite sorted by, never a Python reading of a row.

SQLite orders Studio pages by its own parse of the stored created_at text. A
cursor built from Python's parse, or from a list projection that re-renders the
number, can name a key one ulp or more away from that sort key, so the next page
serves the boundary row again or skips the row after it. Artifact pages order
equal creation times as the snapshot does, by the store's write time, and their
cursors carry that key too.
"""
from __future__ import annotations

import math
from itertools import count

import pytest

from dgx_monarch.studio import store as store_module
from dgx_monarch.studio.history import page_artifacts, page_jobs
from dgx_monarch.studio.job_artifacts import import_artifacts
from dgx_monarch.studio.runtime_history import page_runtime
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.wire_records import snapshot_rows

# SQLite keeps 19 significand digits and drops the rest, so it reads this text
# one ulp below the value Python's correctly rounded parser returns.
STORED_TIME = "1761151234.567891716957092285156251"
SQLITE_TIME = 1761151234.5678916
# Shortest-repr timestamps that SQLite 3.45's x87 long-double parse reads one
# ulp high on x86-64, found by replaying its parse at a 64-bit significand. On
# x86 they reproduce the hosted CI duplicate; other hosts parse them exactly.
X86_HIGH_TIMES = (1761868913.087623, 1761243671.442644, 1761210631.298859)
# The driver's completion time, which every output of one history entry shares.
COMPLETED_AT = 1761151234.5
# What the output viewer asks for when the snapshot lacks a finished job's output.
LOOKUP = {"job_id": "job", "intermediate": "false", "sort": "newest", "limit": "200"}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    value.create("jobs", {"id": "job"})
    yield value
    value.close()


def walk(fetch, query):
    """Every row a client reads by following cursors; a repeated cursor fails at once."""
    rows, seen = [], set()
    while True:
        page = fetch(query)
        rows.extend(page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            return rows
        repeated = cursor in seen
        assert not repeated, "a page served its boundary row again and the cursor stopped advancing"
        seen.add(cursor)
        query = {**query, "cursor": cursor}


def artifact_ids(store, query):
    return [row["id"] for row in walk(lambda value: page_artifacts(store, value), query)]


def snapshot_ids(store):
    return [row["id"] for row in snapshot_rows(store)[0]["artifacts"]]


def store_outputs(store, identities):
    for identity in identities:
        store.create("artifacts", {"id": identity, "job_id": "job", "kind": "image", "intermediate": False,
                                   "created_at": COMPLETED_AT})


def runtime_ids(store, limit):
    return [row["id"] for row in walk(lambda query: page_runtime(store, "job", query), {"limit": str(limit)})]


def test_runtime_cursor_names_the_sort_key_sqlite_parsed_not_pythons(store):
    parsed = store.db.execute("SELECT json_extract(?,'$')", (STORED_TIME,)).fetchone()[0]
    # The split is the condition under test; a SQLite that parses the text
    # exactly would let this pass without exercising the cursor.
    assert parsed == SQLITE_TIME and float(STORED_TIME) == math.nextafter(SQLITE_TIME, math.inf)
    with store.lock, store.transaction():
        store.db.execute("INSERT INTO objects VALUES ('runtime_renders','r1',1,?,0)",
                         ('{"created_at":' + STORED_TIME + ',"id":"r1","job_id":"job","render_id":"r1"}',))
    # Ties r1 on SQLite's key; only the id orders it after r1.
    store.create("runtime_renders", {"id": "r2", "job_id": "job", "render_id": "r2", "created_at": SQLITE_TIME})
    assert runtime_ids(store, 1) == ["r1", "r2"]


@pytest.mark.parametrize("created_at", X86_HIGH_TIMES)
def test_runtime_pages_serve_a_boundary_render_once(store, created_at):
    for index, offset in enumerate((0.0, 1.0)):
        store.create("runtime_renders", {"id": f"r{index}", "job_id": "job", "render_id": f"r{index}",
                                         "created_at": created_at + offset})
    assert runtime_ids(store, 1) == ["r0", "r1"]


@pytest.mark.parametrize("sort", ["newest", "oldest"])
def test_job_history_cursor_survives_the_projection_rounding_created_at(store, sort):
    times = {f"job-{index:02d}": 1761151234.5678911 + index * 0.0000173 for index in range(12)}
    for identity, created_at in times.items():
        store.create("jobs", {"id": identity, "state": "succeeded", "created_at": created_at})
    rows = walk(lambda query: page_jobs(store, query), {"limit": 1, "sort": sort})
    assert sorted(row["id"] for row in rows) == ["job", *times]
    # The job list projection re-renders created_at at 15 significant digits,
    # so the rows a page returns do not carry the key SQLite sorted by.
    assert any(row["created_at"] != times[row["id"]] for row in rows if row["id"] in times)


# Identity order alone happens to match the first order and reverses the second.
@pytest.mark.parametrize("stored", [("a", "b", "c"), ("c", "b", "a")])
def test_artifact_pages_list_equal_times_last_stored_first_as_the_snapshot_does(store, monkeypatch, stored):
    clock = count(COMPLETED_AT)
    monkeypatch.setattr(store_module, "timestamp", lambda: next(clock))
    store_outputs(store, stored)
    newest = list(reversed(stored))
    assert [row["id"] for row in page_artifacts(store, LOOKUP)["items"]] == newest
    assert snapshot_ids(store) == newest
    assert artifact_ids(store, {**LOOKUP, "limit": "1"}) == newest
    assert artifact_ids(store, {**LOOKUP, "sort": "oldest", "limit": "1"}) == list(stored)


def test_artifact_pages_break_equal_write_times_by_identity_as_the_snapshot_does(store, monkeypatch):
    monkeypatch.setattr(store_module, "timestamp", lambda: COMPLETED_AT)
    store_outputs(store, ("b", "c", "a"))
    assert snapshot_ids(store) == ["a", "b", "c"]
    assert artifact_ids(store, {**LOOKUP, "limit": "1"}) == ["a", "b", "c"]
    assert artifact_ids(store, {**LOOKUP, "sort": "oldest", "limit": "1"}) == ["c", "b", "a"]


def test_the_artifacts_route_lists_an_imported_entry_last_output_first(store, monkeypatch):
    clock = count(COMPLETED_AT)
    monkeypatch.setattr(store_module, "timestamp", lambda: next(clock))
    store.create("jobs", {"id": "render", "state": "succeeded", "submission": {}})
    images = [{"filename": name, "subfolder": "", "type": "output"} for name in ("base.png", "detail.png")]
    outputs = {"9": {"images": images}, "12": {"images": [{"filename": "upscaled.png", "type": "output"}]}}
    import_artifacts(store, "render", outputs, completed_at=COMPLETED_AT)
    rows = page_artifacts(store, {**LOOKUP, "job_id": "render"})["items"]
    assert {row["created_at"] for row in rows} == {COMPLETED_AT}
    assert [row["filename"] for row in rows] == ["upscaled.png", "detail.png", "base.png"]
    assert [row["id"] for row in rows] == snapshot_ids(store)
