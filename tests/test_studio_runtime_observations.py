"""CPU-only runtime journal replay with exact prompt joins and durable dedupe."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from dgx_monarch import execution_events
from dgx_monarch.nodes import execution_event_routes
from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.runtime_observations import RuntimeObservations, record_event, valid_event
from dgx_monarch.studio.store import Store


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def event(sequence=1, kind="dispatch_submitted", prompt_id="p", **facts):
    return {"schema_version": 1, "producer_id": "producer", "sequence": sequence, "timestamp": float(sequence),
            "prompt_id": prompt_id, "node_id": "node", "render_id": "render", "list_index": 0,
            "fleet_index": None, "kind": kind, "facts": facts}


def job(store, identity="job", prompt="p", state="running"):
    return store.create("jobs", {"id": identity, "prompt_id": prompt, "state": state,
                                 "submission": {"immutable": True}})


def render(store, identity="job"):
    return next(iter(store.get("jobs", identity)["monarch"]["renders"].values()))


def test_durable_exact_join_and_dedupe_survive_terminal_state(store):
    job(store, state="succeeded")
    job(store, "another", "other")
    row = event(effective_topology={"world": 2}, worker_acceptance="not_yet_observed")
    record_event(store, row, None)
    assert "monarch" not in store.get("jobs", "job")
    RuntimeObservations(store, object())._join_pending()
    record_event(store, row, "job")
    assert store.get("jobs", "job")["monarch"]["event_count"] == 1
    assert "monarch" not in store.get("jobs", "another")
    record_event(store, event(4, "render_completed", outcome="validated_output_and_retired_lease"), "job")
    record_event(store, event(2, "native_ack", owner_token_sha256="a" * 64,
                              authority="validated_full_world_owner_ack", setup_generation=1, statuses=["released"]), "job")
    record_event(store, event(3, "worker_results", workers=[{"rank": 0}], authority="validated_render_results"), "job")
    record_event(store, event(5, "render_failed", worker_refusal_verified=True), "job")
    final = store.get("jobs", "job")
    assert final["state"] == "succeeded" and final["submission"] == {"immutable": True}
    assert final["monarch"]["event_count"] == 5
    assert final["monarch"]["watermarks"] == {"producer": 5}
    assert render(store)["status"] == "completed" and len(render(store)["acks"]) == 1
    assert render(store)["workers"] == [{"rank": 0}]
    assert render(store)["worker_acceptance"] == "validated_result_observed"


def test_wrong_join_is_atomic_and_unbound_event_remains_pending(store):
    job(store, prompt="different")
    row = event()
    record_event(store, row, None)
    with pytest.raises(ValueError, match="identity mismatch"):
        record_event(store, row, "job")
    assert "monarch" not in store.get("jobs", "job")
    assert store.all("runtime_events")[0]["job_id"] is None


def test_unrelated_pending_events_do_not_starve_later_exact_join(store):
    job(store)
    with store.lock, store.transaction():
        for number in range(2050):
            store._put("runtime_events", {"id": str(number), "job_id": None,
                                          "event": event(number + 1, prompt_id="unrelated")}, 1)
    record_event(store, event(), None)
    RuntimeObservations(store, object())._join_pending()
    assert store.get("jobs", "job")["monarch"]["event_count"] == 1


@pytest.mark.parametrize("field,value", [("timestamp", float("nan")), ("timestamp", True), ("sequence", True),
                                           ("prompt_id", ""), ("list_index", -1), ("fleet_index", "0"), ("kind", [])])
def test_malformed_events_refused(field, value):
    row = event()
    row[field] = value
    assert not valid_event(row)


def test_runtime_http_route_leases_replays_and_retains_restart_gaps(store, monkeypatch):
    async def scenario():
        journal = execution_events.Journal()
        monkeypatch.setattr(execution_event_routes, "journal", journal)
        monkeypatch.setattr(execution_event_routes, "context_available", lambda: True)
        app = web.Application()
        routes = web.RouteTableDef()
        execution_event_routes.register(SimpleNamespace(routes=routes), web)
        app.add_routes(routes)
        async with TestServer(app) as server:
            client = ComfyClient(str(server.make_url("/")))
            await client.start()
            try:
                consumer = RuntimeObservations(store, client)
                await consumer.poll()
                assert journal.active() and consumer.status["context_available"]
                job(store)
                scope = execution_events.Scope("p", "node", 0, "r", 0)
                journal.publish(scope, "dispatch_submitted", {"effective_topology": {"world": 1}})
                journal.publish(scope, "result_validated", {"outcome": "validated_output"})
                await asyncio.gather(consumer.poll(), consumer.poll())
                assert store.get("jobs", "job")["monarch"]["event_count"] == 2
                assert render(store)["status"] == "output_validated"
                restarted = RuntimeObservations(store, client)
                await restarted.poll()
                assert store.get("jobs", "job")["monarch"]["event_count"] == 2
                new_journal = execution_events.Journal()
                monkeypatch.setattr(execution_event_routes, "journal", new_journal)
                await restarted.poll()
                assert restarted.status["has_gaps"]
                await restarted.poll()
                assert restarted.status["has_gaps"] and not restarted.status["gap"]
                assert len(store.all("runtime_gaps")) == 1
            finally:
                await client.close()
    asyncio.run(scenario())


def test_regressing_cursor_and_unordered_replay_leave_evidence_unchanged(store):
    class Client:
        async def execution_events(self, after, producer):
            return deepcopy(self.page)
    async def scenario():
        job(store)
        client = Client()
        client.page = {"schema_version": 1, "producer_id": "producer", "next_sequence": 2,
                       "events": [event(2)]}
        consumer = RuntimeObservations(store, client)
        await consumer.poll()
        assert store.get("jobs", "job")["monarch"]["event_count"] == 1
        client.page["next_sequence"] = 1
        client.page["events"] = [event(1)]
        await consumer.poll()
        assert consumer.status["state"] == "unavailable"
        assert store.get("jobs", "job")["monarch"]["event_count"] == 1
    asyncio.run(scenario())


def test_bounded_job_projection_keeps_all_durable_render_outcomes(store):
    from dgx_monarch.studio.contracts import canonical
    from dgx_monarch.studio.runtime_observations import RENDER_SUMMARY_BYTES, RENDER_SUMMARY_LIMIT
    job(store)
    for index in range(RENDER_SUMMARY_LIMIT + 5):
        row = event(index + 1, "worker_results", workers=[{"rank": 0, "runtime_identity": {"description": "x" * 12000}}],
                    authority="validated_render_results")
        row["render_id"] = f"render-{index}"
        record_event(store, row, "job")
    state = store.get("jobs", "job")["monarch"]
    assert state["render_count"] == 37 and state["omitted_render_count"] == 5
    assert len(state["renders"]) == RENDER_SUMMARY_LIMIT
    assert all(len(canonical(row).encode()) <= RENDER_SUMMARY_BYTES for row in state["renders"].values())
    assert state["render_status_counts"] == {"unknown": 37}
    assert len(store.all("runtime_renders")) == 37
    assert all(len(row["workers"][0]["runtime_identity"]["description"]) == 12000 for row in store.all("runtime_renders"))
    # A late event for an evicted summary updates its durable outcome and counts.
    row = event(38, "render_completed", outcome="validated_output_and_retired_lease")
    row["render_id"] = "render-0"
    record_event(store, row, "job")
    state = store.get("jobs", "job")["monarch"]
    assert state["render_status_counts"] == {"unknown": 36, "completed": 1}
    assert state["render_count"] == 37


def test_same_event_identity_cannot_replace_existing_payload(store):
    job(store)
    row = event()
    record_event(store, row, None)
    changed = deepcopy(row)
    changed["prompt_id"] = "different"
    with pytest.raises(ValueError, match="conflicting payloads"):
        record_event(store, changed, None)
    assert store.all("runtime_events")[0]["event"] == row


def test_observer_lease_gap_remains_visible_after_reconnection(store, monkeypatch):
    async def scenario():
        now = [1.0]
        monkeypatch.setattr(execution_events.time, "monotonic", lambda: now[0])
        journal = execution_events.Journal()
        class Client:
            async def execution_events(self, after, producer):
                return journal.read(after, producer)
        consumer = RuntimeObservations(store, Client())
        await consumer.poll()
        assert not consumer.status["has_gaps"]
        now[0] += execution_events.LEASE_SECONDS + 1
        await consumer.poll()
        assert consumer.status["has_gaps"] and consumer.status["gap"]
        await consumer.poll()
        assert consumer.status["has_gaps"] and not consumer.status["gap"]
        assert len(store.all("runtime_gaps")) == 1
    asyncio.run(scenario())


def test_replay_joins_only_page_prompts_without_materializing_job_history(store, monkeypatch):
    class Client:
        async def execution_events(self, after, producer):
            return {"schema_version": 1, "producer_id": "producer", "next_sequence": 1,
                    "events": [event()] if after == 0 else []}

    job(store)
    for index in range(20):
        store.create("jobs", {"id": f"other-{index}", "prompt_id": f"other-prompt-{index}",
                              "state": "succeeded", "submission": {"workflow": "x" * 10000}})
    original_get = store.get

    def get(kind, identity):
        assert kind != "jobs" or identity == "job", "Replay loaded an unrelated submission"
        return original_get(kind, identity)

    def all_records(kind):
        pytest.fail(f"Replay materialized the complete {kind} history")

    monkeypatch.setattr(store, "get", get)
    monkeypatch.setattr(store, "all", all_records)
    consumer = RuntimeObservations(store, Client())
    asyncio.run(consumer.poll())
    assert consumer.status["state"] == "connected"
    assert original_get("jobs", "job")["monarch"]["event_count"] == 1
    asyncio.run(consumer.poll())
    assert original_get("jobs", "job")["monarch"]["event_count"] == 1
