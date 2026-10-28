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
                           "effective": {"seeds": {"1.seed": 2**53 - 1, "2.noise_seed": 0},
                                         "values": {"blob": "x" * 100_000}},
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


def test_job_summary_carries_the_driver_start_and_end_times(store):
    """Studio shows elapsed and total time from these driver-clock fields; a missing one is null."""
    store.create("jobs", job("timed", state="succeeded", started_at=1000.25, ended_at=1073.5))
    store.create("jobs", job("unstarted", state="queued", prompt_id="other"))
    listed = {item["id"]: item for item in page_jobs(store, {})["items"]}
    rows, _ = snapshot_rows(store)
    snapshot = {item["id"]: item for item in rows["jobs"]}
    for view in (listed, snapshot):
        assert (view["timed"]["started_at"], view["timed"]["ended_at"]) == (1000.25, 1073.5)
        assert view["timed"]["ended_at"] - view["timed"]["started_at"] == 73.25
        assert (view["unstarted"]["started_at"], view["unstarted"]["ended_at"]) == (None, None)
        assert "backend_events" not in view["timed"] and "workflow" not in view["timed"]["submission"]


def test_the_companion_notes_on_its_own_clock_when_it_first_saw_a_job_run(store, monkeypatch):
    """Live elapsed time compares this with server_time, so a remote driver's clock never enters."""
    from dgx_monarch.studio import store as store_module
    from dgx_monarch.studio.job_observations import observation

    now = {"time": 100.0}
    monkeypatch.setattr(store_module, "timestamp", lambda: now["time"])

    def event(kind, prompt_id, **data):
        parsed = observation({"type": kind, "data": {"prompt_id": prompt_id, **data}})
        assert parsed is not None
        return parsed
    # The queue poll sees the prompt running first.
    store.create("jobs", job("polled", state="queued", prompt_id="p1"))
    store.update("jobs", "polled", {"state": "running", "prompt_id": "p1"})
    now["time"] = 105.0
    store.update("jobs", "polled", {"state": "running", "prompt_id": "p1"})
    store.observe("polled", *event("progress", "p1", value=1, max=4, node="3", timestamp=900_000))
    assert store.get("jobs", "polled")["running_observed_at"] == 100.0
    # A driver event sees it first: the field matches that event's receipt, not the driver's time.
    store.create("jobs", job("evented", state="queued", prompt_id="p2"))
    now["time"] = 200.0
    store.observe("evented", *event("execution_start", "p2", timestamp=900_000))
    evented = store.get("jobs", "evented")
    assert evented["running_observed_at"] == evented["backend_events"][-1]["observed_at"] == 200.0
    assert evented["started_at"] == 900.0
    # A job that was already running, or has ended, gets no companion start.
    store.create("jobs", job("upgraded", state="running", prompt_id="p3"))
    store.observe("upgraded", *event("progress", "p3", value=2, max=4, timestamp=901_000))
    store.create("jobs", job("ended", state="succeeded", prompt_id="p4"))
    store.observe("ended", *event("execution_start", "p4", timestamp=902_000))
    # Settlement replays a finished prompt's history messages; the companion never saw that one run.
    store.create("jobs", job("replayed", state="queued", prompt_id="p5"))
    Jobs(store, None).consume_event({"type": "execution_start", "data": {"prompt_id": "p5", "timestamp": 903_000}})
    assert store.get("jobs", "replayed")["state"] == "running"
    store.create("jobs", job("watched", state="queued", prompt_id="p6"))
    now["time"] = 300.0
    Jobs(store, None).consume_event({"type": "execution_start", "data": {"prompt_id": "p6", "timestamp": 904_000}},
                                    live=True)
    rows, _ = snapshot_rows(store)
    summary = {item["id"]: item["running_observed_at"] for item in rows["jobs"]}
    assert summary == {"polled": 100.0, "evented": 200.0, "upgraded": None, "ended": None, "replayed": None,
                       "watched": 300.0}


def test_job_summary_carries_the_recorded_seeds_alone(store):
    """Queue rows and the viewer read the seeds the driver ran with without the full record."""
    store.create("jobs", job())
    store.create("jobs", job("external", prompt_id="other", external=True,
                             submission={"prompt": {"1": {"class_type": "Test", "inputs": {"seed": 4}}}}))
    summaries = {item["id"]: item["submission"] for item in page_jobs(store, {})["items"]}
    # Safe integers come back as JSON numbers, and no other effective field rides along.
    assert summaries["original"]["effective"] == {"seeds": {"1.seed": 2**53 - 1, "2.noise_seed": 0}}
    # An imported job has no frozen seed list; its prompt's integer seed inputs stand in.
    assert summaries["external"]["effective"] == {"seeds": {"1.seed": 4}}
    rows, _ = snapshot_rows(store)
    assert {row["id"]: row["submission"]["effective"] for row in rows["jobs"]} == {
        identity: submission["effective"] for identity, submission in summaries.items()}


def test_a_seed_a_browser_cannot_hold_exactly_travels_as_its_digits(store):
    """ComfyUI seeds reach 2**64 - 1, but a browser reads JSON numbers as doubles, exact to 2**53 - 1."""
    seeds = {"1.seed": 2**64 - 1, "2.seed": 2**53, "3.seed": -(2**53), "4.seed": 2**53 - 1, "5.noise_seed": 0}
    recorded = job("recorded")
    recorded["submission"]["effective"]["seeds"] = seeds
    store.create("jobs", recorded)
    store.create("jobs", job("imported", prompt_id="other", external=True, submission={"prompt": {
        "3": {"class_type": "KSampler", "inputs": {"seed": 2**64 - 1, "steps": 20}},
        "4": {"class_type": "RandomNoise", "inputs": {"noise_seed": 2**53}},
        "5": {"class_type": "KSampler", "inputs": {"seed": ["4", 0]}},
        "6": "a node that is plain text"}}))
    wire = {"1.seed": "18446744073709551615", "2.seed": "9007199254740992", "3.seed": "-9007199254740992",
            "4.seed": 9007199254740991, "5.noise_seed": 0}
    imported = {"3.seed": "18446744073709551615", "4.noise_seed": "9007199254740992"}
    page = page_jobs(store, {})
    rows, _ = snapshot_rows(store)
    for items in (page["items"], rows["jobs"]):
        listed = {item["id"]: item["submission"]["effective"]["seeds"] for item in items}
        assert listed == {"recorded": wire, "imported": imported}
    # The text the browser parses holds the digits, quoted above 2**53 - 1 and bare below.
    text = canonical(page)
    assert '"1.seed":"18446744073709551615"' in text and '"4.seed":9007199254740991' in text
    # The stored records keep exact integers: the frozen submission and its digest do not change.
    assert store.get("jobs", "recorded")["submission"]["effective"]["seeds"] == seeds
    assert store.get("jobs", "imported")["submission"]["prompt"]["3"]["inputs"]["seed"] == 2**64 - 1
    # One job's summary reads by its identity, for a job the snapshot leaves out.
    single = page_jobs(store, {"job_id": "imported", "limit": 1})
    assert [item["id"] for item in single["items"]] == ["imported"] and single["next_cursor"] is None
    assert single["items"][0]["submission"]["effective"]["seeds"] == imported


def test_a_failed_job_summary_carries_its_whole_recorded_error(store):
    """The viewer status line and Queue rows read a failure from the summary row, never the full record."""
    from dgx_monarch.studio.job_observations import bounded_refusal

    message = bounded_refusal("A remote actor call has failed.\n" + "  File worker.py, line 1\n" * 400
                              + "RuntimeError: the stock load does not fit. See docs/TROUBLESHOOTING.md #12.")
    error = {"code": "ActorError", "message": message}
    store.create("jobs", job("failed", state="failed", error=error))
    store.create("jobs", job("unresolved", state="unknown", prompt_id="other",
                             error={"code": "driver_unavailable", "message": "Dispatch stopped."}))
    listed = {item["id"]: item["error"] for item in page_jobs(store, {})["items"]}
    rows, _ = snapshot_rows(store)
    assert listed == {row["id"]: row["error"] for row in rows["jobs"]}
    assert listed["failed"] == error and listed["unresolved"]["code"] == "driver_unavailable"
    assert listed["failed"]["message"].endswith(" See docs/TROUBLESHOOTING.md #12.")


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
