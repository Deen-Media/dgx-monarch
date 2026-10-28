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


def framed_refusal(prefix_chars, body):
    """A tagged refusal after ``prefix_chars`` of remote traceback, as Monarch delivers a worker raise."""
    from dgx_monarch.refusal import PanelAction, RefusalClass, refusal

    text = refusal(RefusalClass.CAPACITY, body, guard="stock_load_preflight", waivable=True,
                   panel_action=PanelAction("Load with slab residency", "DGXM_ALLOW_SLAB_RESCUE"), troubleshooting=12)
    lead = "A remote actor call has failed.\n Traceback of where the remote call failed (most recent call last):\n"
    frames = '  File "/srv/dgx-monarch/src/dgx_monarch/actor/worker_calls.py", line 7, in frame\n'
    prefix = (lead + frames * (prefix_chars // len(frames) + 1))[:prefix_chars]
    return prefix + text


def stored_message(store, text):
    store.create("jobs", {"id": "job", "state": "queued", "prompt_id": "prompt",
                          "submission": {"prompt": {}, "workflow": {}, "digest": "d"}})
    Jobs(store, Driver()).consume_event({"type": "execution_error", "data": {
        "prompt_id": "prompt", "exception_type": "ActorError", "exception_message": text}})
    return store.get("jobs", "job")["error"]["message"]


def test_a_class_tag_past_the_head_is_kept_with_the_text_after_it(store):
    from dgx_monarch.refusal import parse_refusal_tag
    from dgx_monarch.studio import job_observations

    head = job_observations.REFUSAL_HEAD_CHARS
    body = "The stock load does not fit this host." + " Measured detail." * 60
    text = framed_refusal(head + 3000, body)
    message = stored_message(store, text)
    tag = parse_refusal_tag(message)
    assert tag is not None and (tag.refusal_class.value, tag.guard, tag.waivable) == ("C", "stock_load_preflight", True)
    assert message.startswith("A remote actor call has failed.")
    excerpt = "[dgxm:C guard=stock_load_preflight waivable=1] The stock load does not fit this host."
    assert f" [3000 characters omitted] {excerpt}" in message
    # The excerpt is bounded too, and the second marker counts what follows it.
    omitted_after = len(body) - (job_observations.TAGGED_CHARS - len("[dgxm:C guard=stock_load_preflight waivable=1] "))
    assert f" [{omitted_after} characters omitted]" in message
    assert message.endswith("Headless: set DGXM_ALLOW_SLAB_RESCUE=1 on the driver and run again. "
                            "See docs/TROUBLESHOOTING.md #12.")
    assert len(message) < head + job_observations.TAGGED_CHARS + 400 + 80


def test_a_class_tag_across_the_head_boundary_stays_whole(store):
    from dgx_monarch.refusal import parse_refusal_tag
    from dgx_monarch.studio import job_observations

    head = job_observations.REFUSAL_HEAD_CHARS
    message = stored_message(store, framed_refusal(head - 10, "The stock load does not fit this host."))
    assert parse_refusal_tag(message).refusal_class.value == "C"
    assert "[0 characters omitted]" not in message and "characters omitted]" not in message.split("[dgxm:")[0]
    assert message.index("[dgxm:C guard=stock_load_preflight waivable=1] The stock load") == head - 10


def test_two_class_tags_keep_the_plain_head():
    from dgx_monarch.studio import job_observations

    head = job_observations.REFUSAL_HEAD_CHARS
    text = "x" * head + " [dgxm:P] one. [dgxm:K guard=ring_pad waivable=1] two."
    assert job_observations.bounded_refusal(text) == "x" * head + f" [{len(text) - head} characters omitted]"


def test_the_kept_tag_grammar_is_the_refusal_modules_own():
    from dgx_monarch import refusal
    from dgx_monarch.studio import job_observations

    samples = ["[dgxm:P]", "[dgxm:C guard=stock_load_preflight waivable=1]", "[dgxm:K guard=ring_pad:minimax_h3 waivable=0]",
               "[dgxm:C waivable=1]", "[dgxm:C guard=stock_load_preflight]", "[dgxm:X]", "[dgxm;P]",
               '[dgxm:measured {"checkpoint_bytes":1}]', "[dgxm:U guard=" + "g" * 65 + " waivable=1]"]
    for sample in samples:
        assert bool(job_observations._TAG.fullmatch(sample)) == bool(refusal._TAG_RE.fullmatch(sample)), sample
