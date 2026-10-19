"""Retry idempotency is bound to the endpoint's job, never a body override."""

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
