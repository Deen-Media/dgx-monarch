"""A long driver refusal keeps its class tag, remedy sentence and consent block in the job record.

This module builds a real refusal, so the protected test inventory covers it;
it stays separate so that the general execution tests are not protected too.
"""
from __future__ import annotations

import json

import pytest

from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store


class Driver:
    async def request(self, method, path, data=None):
        return {}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def long_consent_refusal():
    from dgx_monarch import consent_descriptor
    from dgx_monarch.refusal import PanelAction, RefusalClass, refusal

    descriptor = consent_descriptor.DESCRIPTOR_BEGIN + json.dumps({"v": 1, "kind": "rescue-slab", "consent_id": "c-1"}) \
        + consent_descriptor.DESCRIPTOR_END
    body = "The stock load does not fit this host. " + "Measured detail. " * 180 + descriptor
    text = refusal(RefusalClass.CAPACITY, body, guard="stock_load_preflight", waivable=True,
                   panel_action=PanelAction("Load with slab residency", "DGXM_ALLOW_SLAB_RESCUE"), troubleshooting=12)
    return "RuntimeError: worker 0: " + text, descriptor


def test_long_refusal_keeps_its_class_tag_remedy_and_consent_descriptor(store):
    from dgx_monarch import consent_descriptor
    from dgx_monarch.refusal import parse_refusal_tag
    from dgx_monarch.studio import job_observations

    assert job_observations.DESCRIPTOR_BEGIN == consent_descriptor.DESCRIPTOR_BEGIN
    assert job_observations.DESCRIPTOR_END == consent_descriptor.DESCRIPTOR_END
    text, descriptor = long_consent_refusal()
    assert len(text) > 2000
    store.create("jobs", {"id": "job", "state": "queued", "prompt_id": "prompt",
                          "submission": {"prompt": {}, "workflow": {}, "digest": "d"}})
    Jobs(store, Driver()).consume_event({"type": "execution_error", "data": {
        "prompt_id": "prompt", "exception_type": "StockLoadCapacityError", "exception_message": text}})
    message = store.get("jobs", "job")["error"]["message"]
    assert parse_refusal_tag(message).refusal_class.value == "C"
    assert 'Open the DGX Monarch panel and click "Load with slab residency"' in message
    assert "Headless: set DGXM_ALLOW_SLAB_RESCUE=1 on the driver and run again. See docs/TROUBLESHOOTING.md #12." in message
    assert descriptor in message and "characters omitted]" in message
    assert len(message) < 2000 + 400 + len(descriptor) + 40
    short = "[dgxm:P] Out of memory."
    assert job_observations.bounded_refusal(short) == short
