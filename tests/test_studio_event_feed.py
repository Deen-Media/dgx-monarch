"""The event feed names every created and changed entity, within the per-event cap."""
from __future__ import annotations

import io

import pytest
from PIL import Image

import dgx_monarch.studio.media_assets as media_assets
from dgx_monarch.studio.contracts import EVENT_BYTE_LIMIT, EVENT_LIMIT, MAX_EVENT_BYTES, StudioError, schema_document
from dgx_monarch.studio.live_preview import accept_preview
from dgx_monarch.studio.store import Store


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def events_after(store, cursor):
    return list(store.events(cursor)["events"])


def frame(prompt_id="prompt"):
    image = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(image, format="PNG")
    return {"type": "preview_frame", "data": {"prompt_id": prompt_id, "node_id": "9", "image_type": "image/png"},
            "image": image.getvalue()}


def test_created_operations_artifacts_and_driver_jobs_emit_small_events(store):
    store.create("operations", {"id": "operation", "action": "doctor", "state": "reviewed",
                                "parameters": {"note": "x" * 100_000}})
    store.create("artifacts", {"id": "artifact", "job_id": "job", "kind": "image", "source": "driver_history",
                               "intermediate": False, "search_text": "x" * 100_000})
    store.create("jobs", {"id": "external", "state": "unknown", "prompt_id": "driver-prompt", "external": True,
                          "submission": {"workflow": {"padding": "x" * 100_000}}})
    store.create("previews", {"id": "preview", "workflow": {}, "expires_at": 0})
    rows = {row["kind"]: row for row in events_after(store, 0)}
    assert set(rows) == {"operations.created", "artifacts.created", "jobs.created"}
    assert rows["operations.created"]["data"] == {"action": "doctor", "state": "reviewed"}
    assert rows["artifacts.created"]["data"] == {"job_id": "job", "kind": "image", "source": "driver_history",
                                                 "intermediate": False}
    assert rows["jobs.created"]["data"] == {"state": "unknown", "prompt_id": "driver-prompt", "external": True}
    assert rows["jobs.created"]["job_id"] == "external"


def test_an_oversized_field_is_named_and_never_aborts_the_write(store):
    # The driver chooses foreign prompt identities; one that cannot fit the
    # event must not stop the job from being recorded.
    store.create("jobs", {"id": "foreign", "state": "unknown", "prompt_id": "p" * 100_000, "external": True})
    created = events_after(store, 0)[-1]
    assert created["data"] == {"state": "unknown", "external": True, "omitted_fields": ["prompt_id"]}
    cursor = store.events()["latest_cursor"]
    store.update("jobs", "foreign", {"backend_status": {"messages": ["m" * 100_000]}, "phase": "execution_success"})
    updated = events_after(store, cursor)[-1]
    assert updated["kind"] == "jobs.updated" and updated["data"]["changed"] == ["backend_status", "phase"]
    assert updated["data"]["phase"] == "execution_success"
    assert set(updated["data"]["omitted_fields"]) == {"prompt_id", "backend_status"}
    with store.lock:
        assert store.db.execute("SELECT MAX(size) FROM events").fetchone()[0] < MAX_EVENT_BYTES // 2
    assert store.get("jobs", "foreign")["backend_status"]["messages"][0] == "m" * 100_000


def test_job_updates_name_the_fields_that_changed(store):
    store.create("jobs", {"id": "job", "state": "running", "prompt_id": "prompt"})
    cursor = store.events()["latest_cursor"]
    store.update("jobs", "job", {"preview_artifact_id": "artifact", "state": "running"})
    row = events_after(store, cursor)[-1]
    assert row["data"] == {"state": "running", "prompt_id": "prompt", "preview_artifact_id": "artifact",
                           "changed": ["preview_artifact_id"]}


def test_live_preview_frames_name_their_artifact_in_the_job_update(store):
    store.create("jobs", {"id": "job", "state": "running", "prompt_id": "prompt"})
    cursor = store.events()["latest_cursor"]
    accept_preview(store, frame())
    kinds = [(row["kind"], row["data"].get("changed")) for row in events_after(store, cursor)]
    assert kinds == [("artifacts.preview", None), ("jobs.updated", ["preview_artifact_id"])]
    artifact = store.get("jobs", "job")["preview_artifact_id"]
    assert events_after(store, cursor)[-1]["data"]["preview_artifact_id"] == artifact


@pytest.mark.parametrize("outcome", ["cancelled", "succeeded", "failed"])
def test_a_frame_decoded_during_a_settlement_never_attaches(store, monkeypatch, outcome):
    store.create("jobs", {"id": "job", "state": "cancel_requested", "prompt_id": "prompt"})
    original = media_assets.thumbnail

    def settle_while_decoding(data):
        # The cancel or the history reconcile lands while the frame decodes
        # outside the store lock.
        store.update("jobs", "job", {"state": outcome})
        return original(data)

    monkeypatch.setattr(media_assets, "thumbnail", settle_while_decoding)
    cursor = store.events()["latest_cursor"]
    accept_preview(store, frame())
    job = store.get("jobs", "job")
    assert job["state"] == outcome and "preview_artifact_id" not in job
    assert store.all("artifacts") == []
    assert [row["kind"] for row in events_after(store, cursor)] == ["jobs.updated"]


def test_capabilities_publish_every_event_feed_limit_the_store_enforces(store):
    limits = schema_document()["limits"]
    assert (limits["event_count"], limits["event_bytes"], limits["event_max_bytes"]) == (
        EVENT_LIMIT, EVENT_BYTE_LIMIT, MAX_EVENT_BYTES) == (4096, 8 * 1024 * 1024, 64 * 1024)
    # The store refuses one event past the published per-event cap.
    with pytest.raises(StudioError, match="retention budget"), store.lock, store.transaction():
        store._event("jobs.updated", "job", {"padding": "x" * limits["event_max_bytes"]})
