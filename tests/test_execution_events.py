"""Opt-in observations preserve exact identities and existing runtime authority."""
from __future__ import annotations

import hashlib
import sys
import threading
from types import SimpleNamespace

import pytest

from dgx_monarch import execution_events as events
from dgx_monarch import runtime_observer as observer
from dgx_monarch.rdma_receiver import validate_ack_responses


@pytest.fixture
def journal(monkeypatch):
    value = events.Journal()
    monkeypatch.setattr(events, "journal", value)
    return value


@pytest.fixture
def context(monkeypatch):
    current = SimpleNamespace(prompt_id="prompt-a", node_id="node-7", list_index=3)
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", SimpleNamespace(get_executing_context=lambda: current))
    return current


def test_disabled_observation_performs_no_context_lookup_or_fingerprinting(journal, monkeypatch):
    def forbidden():
        pytest.fail("Inactive observation touched the Comfy context")
    monkeypatch.setitem(sys.modules, "comfy_execution.utils", SimpleNamespace(get_executing_context=forbidden))
    request = {"_dgxm_render_id": "render"}
    assert observer.prepare(request) is None
    assert "_dgxm_observe_runtime" not in request
    result = {}
    observer.worker_identity(request, object(), result)
    assert result == {}
    journal.publish(None, "worker_results", {"unserializable": object()})
    assert journal._sequence == 0


def test_exact_context_and_lease_expiry(journal, context, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(events.time, "monotonic", lambda: clock[0])
    journal.read()
    request = {"_dgxm_render_id": "render-1"}
    scope = observer.prepare(request, 4)
    assert scope == events.Scope("prompt-a", "node-7", 3, "render-1", 4)
    assert request["_dgxm_observe_runtime"] is True
    events.emit(scope, "result_validated", outcome="validated_output")
    clock[0] = 41.0
    events.emit(scope, "render_completed", outcome="validated_output_and_retired_lease")
    assert events.capture("another") is None
    assert [row["kind"] for row in journal.read()["events"]] == ["result_validated"]
    context.prompt_id = ""
    assert events.capture("render") is None


def test_replay_bounds_gap_restart_and_detached_rows(journal, monkeypatch):
    monkeypatch.setattr(events, "MAX_EVENTS", 2)
    journal.read()
    scope = events.Scope("p", "n", None, "r")
    for value in range(3):
        events.emit(scope, "worker_results", value=value)
    page = journal.read()
    assert page["gap"] and page["oldest_sequence"] == 2
    assert [row["sequence"] for row in page["events"]] == [2, 3]
    page["events"][0]["facts"]["value"] = 999
    assert journal.read()["events"][0]["facts"]["value"] == 1
    assert journal.read(3, "old-producer")["producer_changed"]
    assert not journal.read(3, journal.producer_id)["events"]
    events.emit(scope, "worker_results", value="x" * events.MAX_EVENT_BYTES)
    events.emit(scope, "worker_results", value=float("nan"))
    assert journal.read()["omitted_events"] == 2


def test_verified_ack_only_and_scratch_thread_identity(journal):
    journal.read()
    token = "a" * 32
    rows = [{"setup_generation": 7, "token": token, "status": status} for status in ("released", "unknown")]
    scope = events.Scope("p", "n", 0, "r")
    with events.activate(scope):
        with pytest.raises(RuntimeError, match="partial"):
            validate_ack_responses(rows[:1], 7, token, 2)
        assert not journal.read()["events"]
        operation = events.bind_operation(lambda: validate_ack_responses(rows, 7, token, 2))
    thread = threading.Thread(target=operation)
    thread.start()
    thread.join()
    page = journal.read()
    assert len(page["events"]) == 1
    ack = page["events"][0]
    assert ack["prompt_id"] == "p" and ack["render_id"] == "r"
    assert ack["facts"]["owner_token_sha256"] == hashlib.sha256(token.encode()).hexdigest()
    assert token not in str(page)
    validate_ack_responses(rows, 7, token, 2)
    assert len(journal.read()["events"]) == 1


def test_fleet_requires_validated_output_and_complete_retirement_once(journal):
    journal.read()
    item = SimpleNamespace(_execution_scope=events.Scope("p", "n", None, "r", 2),
                           consumed=True, retired=True, audit_retired=True, progress_closed=True)
    observer.fleet_settled(item)
    assert not journal.read()["events"]
    observer.fleet_result(item, {"rank": 1, "latent": object(), "resident_adoption_evidence": {"private": True}})
    item.retired = False
    observer.fleet_settled(item)
    assert [row["kind"] for row in journal.read()["events"]] == ["worker_results", "result_validated"]
    item.retired = True
    observer.fleet_settled(item)
    observer.fleet_settled(item)
    page = journal.read()
    assert len(page["events"]) == 3
    assert page["events"][-1]["kind"] == "fleet_item_completed"
    assert "private" not in str(page) and "latent'" not in str(page)


def test_failed_output_never_publishes_worker_result_or_completion(journal):
    journal.read()
    scope = events.Scope("p", "n", None, "r")
    def refused(*args):
        raise ValueError("invalid output")
    with pytest.raises(ValueError, match="invalid output"):
        observer.finish(scope, refused, [{"rank": 0}])
    assert not journal.read()["events"]
    observer.failed(scope, TimeoutError(), "collect", False)
    assert journal.read()["events"][0]["facts"]["outcome"] == "unknown"


def test_worker_identity_is_requested_and_reports_real_generation(journal, monkeypatch):
    from dgx_monarch import runtime_identity, runtime_provenance
    monkeypatch.setattr(runtime_identity, "runtime_versions", lambda: {"python": "test"})
    monkeypatch.setattr(runtime_provenance, "cached_dgx_source_manifest_sha256", lambda: "b" * 64)
    request = {"_dgxm_observe_runtime": True, "_dgxm_render_id": "render-1"}
    worker = SimpleNamespace(rank=1, _setup_generation=8, topology={"dp": 2})
    result = {}
    observer.worker_identity(request, worker, result)
    assert result["runtime_identity"] == {"source": "worker_result", "render_id": "render-1", "rank": 1,
                                           "setup_generation": 8, "topology": {"dp": 2}, "versions": {"python": "test"},
                                           "selected_attention": None, "effective_attention": None,
                                           "source_manifest_sha256": "b" * 64}


def test_actual_read_job_binds_operation_scope(journal):
    from dgx_monarch.rdma_read_job import LatentReadJob
    journal.read()
    scope = events.Scope("p", "n", 0, "r")
    owners = []
    def operation(*_args):
        events.native_ack(3, "b" * 32, ["released"])
    with events.activate(scope):
        job = LatentReadJob(object(), operation, [], owners)
    thread = threading.Thread(target=job.operation)
    thread.start()
    thread.join()
    assert journal.read()["events"][0]["prompt_id"] == "p"


@pytest.mark.parametrize("failure", [None, "finish", "release"])
def test_pending_publishes_completion_only_after_finish_and_release(journal, monkeypatch, failure):
    from functools import partial

    from dgx_monarch.nodes import pending as pending_module
    journal.read()
    scope = events.Scope("prompt", "node", 0, "render")
    observed = []
    def finish(*_args):
        observed.append("finish")
        if failure == "finish":
            raise ValueError("finish failed")
        return {"samples": "validated"}
    def close(**_kwargs):
        observed.append("release")
        assert all(row["kind"] != "render_completed" for row in journal.read()["events"])
        if failure == "release":
            raise ValueError("release failed")
    pending = pending_module.PendingRender.__new__(pending_module.PendingRender)
    pending._execution_scope = scope
    pending._start = lambda *_args: None
    pending._handle = SimpleNamespace(collect_sample=lambda *_args, **_kwargs: [{"rank": 0}])
    pending._future = object()
    pending._progress = object()
    pending._timeout = 1
    pending._topo = object()
    pending._latent = {}
    pending._finish = partial(observer.finish, scope, finish)
    pending._close = close
    pending._close_failed_result = lambda *_args: None
    monkeypatch.setattr(pending_module, "_throw_if_comfy_interrupted", lambda: None)
    monkeypatch.setattr(pending_module, "evict_render_fleet", lambda *_args: None)
    if failure:
        with pytest.raises(ValueError, match=failure):
            pending.result()
    else:
        assert pending.result() == {"samples": "validated"}
    kinds = [row["kind"] for row in journal.read()["events"]]
    assert ("render_completed" in kinds) == (failure is None)
    if failure == "finish":
        assert kinds == ["render_failed"]
    else:
        assert observed == ["finish", "release"]
        assert kinds[:2] == ["worker_results", "result_validated"]


def test_real_fleet_collection_emits_each_settled_item_once(journal, monkeypatch):
    from dgx_monarch.nodes import fleet
    from dgx_monarch.nodes.fleet_cleanup import _FleetPending
    journal.read()
    items = [_FleetPending(index, object(), object(), render_id=f"r{index}", rank=index,
                           _execution_scope=events.Scope("p", "n", 0, f"r{index}", index)) for index in range(2)]
    monkeypatch.setattr(fleet, "read_latent_result", lambda *_args: "validated-latent")
    monkeypatch.setattr(fleet, "_retire_fleet_audit", lambda *_args: None)
    monkeypatch.setattr(fleet, "_retire_fleet_future", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fleet, "_close_fleet_progress", lambda *_args: None)
    monkeypatch.setattr(fleet.telemetry_fleet.fleet_jobs, "record_job", lambda *_args: None)
    handle = SimpleNamespace(collect_one=lambda *_args, **_kwargs: {"latent": "wire", "rank": 0})
    results = [None, None]
    fleet._collect_fleet_wave(handle, items, results, 1, 2)
    assert results == ["validated-latent", "validated-latent"]
    completed = [row for row in journal.read()["events"] if row["kind"] == "fleet_item_completed"]
    assert [(row["render_id"], row["fleet_index"]) for row in completed] == [("r0", 0), ("r1", 1)]
