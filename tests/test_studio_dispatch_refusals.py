"""POST /prompt outcomes: a definite refusal fails, a partial acceptance warns, anything else stays unknown."""
from __future__ import annotations

import asyncio
import uuid

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.submission_guard import node_error_summary

NODE_ERRORS = {"7": {"class_type": "KSampler", "dependent_outputs": ["9"], "errors": [{
    "type": "value_not_in_list", "message": "Value not in list: /home/user/private.safetensors",
    "details": "ckpt_name: 'private.safetensors' not in [...]", "extra_info": {"input_name": "ckpt_name",
                                                                            "received_value": "private.safetensors"}}]}}


def refusal(status, detail=None):
    exc = StudioError("driver_rejected", f"Driver returned HTTP {status}; inspect its local diagnostics.", 502)
    if status is not None:
        exc.upstream_status = status  # Contract C1: the transport carries the upstream status.
        exc.upstream_detail = detail
    return exc


class Driver:
    def __init__(self, answer):
        self.answer, self.posts = answer, []

    async def request(self, method, path, data=None):
        if method == "POST" and path == "/prompt":
            self.posts.append(data)
            if isinstance(self.answer, BaseException):
                raise self.answer
            return self.answer(data) if callable(self.answer) else self.answer
        return {"queue_running": [], "queue_pending": []} if path == "/queue" else {}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def submitted(store):
    job, _ = store.reserve("jobs", "key", {"preview_id": "p"}, {
        "id": "job", "state": "submitting", "prompt_id": None, "client_prompt_id": str(uuid.uuid4()),
        "submission": {"prompt": {"1": {"class_type": "KSampler", "inputs": {}}}, "workflow": {}, "digest": "d"}})
    return job


@pytest.mark.parametrize("status", [400, 404, 413, 499])
def test_client_error_from_prompt_is_a_definite_failure_with_clean_node_errors(store, status):
    job = submitted(store)
    rows = [{"node_id": "7", "class_type": "KSampler", "input_name": "ckpt_name", "type": "value_not_in_list"}]

    class Lookups(Driver):
        lookups: list[str] = []

        async def history_prompt(self, identity):
            self.lookups.append(identity)
            return {}
    driver = Lookups(refusal(status, rows))
    jobs = Jobs(store, driver)
    asyncio.run(jobs.dispatch(job))
    value = store.get("jobs", "job")
    assert value["state"] == "failed"
    assert value["error"]["code"] == "prompt_rejected" and value["error"]["status"] == status
    assert value["error"]["node_errors"] == rows
    # A failed job can be retried; an unknown one could not.
    assert value["prompt_id"] is None
    # The refusal settles the job: it never enters the lookup rotation or reads as absent.
    asyncio.run(jobs.reconcile())
    assert driver.lookups == [] and "absence" not in store.get("jobs", "job")


@pytest.mark.parametrize("failure", [refusal(500), refusal(502), refusal(None),
                                     StudioError("driver_unknown", "lost", 503)])
def test_server_and_transport_failures_stay_unknown(store, failure):
    job = submitted(store)
    asyncio.run(Jobs(store, Driver(failure)).dispatch(job))
    value = store.get("jobs", "job")
    assert value["state"] == "unknown" and value["error"]["code"] == failure.code


def test_accepted_prompt_with_node_errors_is_queued_with_a_cleaned_warning(store):
    job = submitted(store)
    driver = Driver(lambda data: {"prompt_id": data["prompt_id"], "number": 3, "node_errors": NODE_ERRORS})
    asyncio.run(Jobs(store, driver).dispatch(job))
    value = store.get("jobs", "job")
    assert value["state"] == "queued" and value["prompt_id"] == job["client_prompt_id"]
    warning = value["driver_warnings"][0]
    assert warning["code"] == "node_errors"
    assert warning["node_errors"] == [{"node_id": "7", "class_type": "KSampler", "input_name": "ckpt_name",
                                       "type": "value_not_in_list"}]
    assert "private" not in str(value["driver_warnings"])


def test_accepted_prompt_without_node_errors_carries_no_warning(store):
    job = submitted(store)
    asyncio.run(Jobs(store, Driver(lambda data: {"prompt_id": data["prompt_id"], "node_errors": {}})).dispatch(job))
    assert "driver_warnings" not in store.get("jobs", "job")


@pytest.mark.parametrize("answer", [[], {"prompt_id": None}, {"prompt_id": "../x"}, "accepted"])
def test_malformed_acknowledgement_stays_unknown_and_keeps_its_companion_identity(store, answer):
    job = submitted(store)
    driver = Driver(answer)
    asyncio.run(Jobs(store, driver).dispatch(job))
    value = store.get("jobs", "job")
    assert value["state"] == "unknown" and value["error"]["code"] == "submission_unknown"
    assert value["client_prompt_id"] == job["client_prompt_id"]
    assert driver.posts[0]["prompt_id"] == job["client_prompt_id"]


def test_node_error_summary_is_bounded_and_drops_values():
    many = {str(index): {"class_type": "X" * 500, "errors": [{"type": "t\x00ype", "message": "secret",
                                                              "extra_info": {"input_name": "in", "received_value": "v"}}]}
            for index in range(100)}
    rows = node_error_summary(many)
    assert len(rows) == 64
    assert rows[0] == {"node_id": "0", "class_type": "X" * 128, "input_name": "in", "type": "type"}
    assert "secret" not in str(rows) and "received_value" not in str(rows)
    assert node_error_summary([{"node_id": 7, "class_type": "A", "input_name": None, "type": "t", "extra": "x"}]) == [
        {"node_id": "7", "class_type": "A", "input_name": None, "type": "t"}]
    assert node_error_summary(None) == [] and node_error_summary({"1": "bad"}) == []


def test_unexpected_dispatch_fault_settles_unknown_and_reports_the_driver_state(store, monkeypatch):
    job = submitted(store)
    jobs = Jobs(store, Driver(lambda data: {"prompt_id": data["prompt_id"]}))

    async def broken():
        raise RuntimeError("synthetic fault")
    monkeypatch.setattr(jobs.runtime_events, "poll", broken)
    asyncio.run(jobs.dispatch(job))
    value = store.get("jobs", "job")
    assert value["state"] == "unknown" and value["error"]["code"] == "studio_internal"
    assert jobs.driver["state"] == "unavailable" and jobs.driver["code"] == "studio_internal"
