"""Recording append remains valid JSONL across external file boundaries."""
from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import closing

import pytest

from dgx_monarch.studio import management
from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.management import Management, recording_bytes
from dgx_monarch.studio.store import Store


class Driver:
    origin = "http://127.0.0.1:8188"

    async def request(self, method, path):
        assert (method, path) == ("GET", "/dgxm/telemetry")
        return {"render": {"active": False}, "workers": [], "driver_host": {"host": "fake"}}


@pytest.mark.parametrize("ending", ["", "\n", "\r\n"])
def test_append_preserves_replayable_jsonl_and_exact_digest(tmp_path, ending):
    path = tmp_path / "external.jsonl"
    prior = ('{"t":1}' + ending).encode()
    path.write_bytes(prior)
    path.chmod(0o600)
    with closing(Store(tmp_path / "state")) as store:
        adapter = Management(store, Driver(), None, None)
        parameters, plan = adapter.preview("top.record", {"path": str(path), "samples": 1, "interval_s": 0.1})
        store.create("operations", {"id": "operation", "state": "applying"})
        state, receipt = asyncio.run(adapter.record(parameters, "operation", plan))
        raw = recording_bytes(path)
        assert state == "succeeded" and raw.startswith(prior)
        assert len(raw.splitlines()) == receipt["samples"] == 2
        assert receipt["bytes"] == len(raw)
        assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
        _, replay_plan = adapter.preview("top.replay", {"recording_id": receipt["recording_id"]})
        state, replay = adapter.replay(replay_plan)
        assert state == "succeeded" and len(replay["samples"]) == 2
        assert json.loads(raw.splitlines()[0]) == {"t": 1}


def test_missing_newline_separator_counts_against_budget_without_partial_write(tmp_path, monkeypatch):
    path = tmp_path / "external.jsonl"
    prior = b'{"t":1}'
    path.write_bytes(prior)
    path.chmod(0o600)
    monkeypatch.setattr(management.RecordedPoller, "tick", lambda self: {"t": 2})
    class SmallDriver(Driver):
        async def request(self, method, path):
            return {"ok": True}
    # Without accounting for the required separator this sample just fits.
    sample = b'{"source_telemetry":{"ok":true},"t":2}\n'
    monkeypatch.setattr(management, "RECORD_BYTES", len(prior) + len(sample))
    with closing(Store(tmp_path / "state")) as store:
        adapter = Management(store, SmallDriver(), None, None)
        parameters, plan = adapter.preview("top.record", {"path": str(path), "samples": 1, "interval_s": 0.1})
        store.create("operations", {"id": "operation", "state": "applying"})
        state, receipt = asyncio.run(adapter.record(parameters, "operation", plan))
        assert state == "unknown" and receipt["samples"] == 1
        assert path.read_bytes() == prior
        assert receipt["bytes"] == len(prior)
        assert receipt["sha256"] == hashlib.sha256(prior).hexdigest()


def test_append_refuses_changed_review_before_inserting_separator(tmp_path):
    path = tmp_path / "external.jsonl"
    path.write_bytes(b'{"t":1}')
    path.chmod(0o600)
    with closing(Store(tmp_path / "state")) as store:
        adapter = Management(store, Driver(), None, None)
        parameters, plan = adapter.preview("top.record", {"path": str(path), "samples": 1, "interval_s": 0.1})
        path.write_bytes(b'{"t":3}')
        store.create("operations", {"id": "operation", "state": "applying"})
        with pytest.raises(StudioError, match="changed after review"):
            asyncio.run(adapter.record(parameters, "operation", plan))
        assert path.read_bytes() == b'{"t":3}'
