"""consent.apply is driven from the driver's pending list, and driver refusals settle.

GET /dgxm/consents is the only source of the exact pending key and id an accept
needs, so the review reads it. A 4xx from POST /dgxm/consent or /dgxm/recycle
means the driver refused and changed nothing, so the receipt says failed and
keeps the driver's short reason; a 5xx or a lost response stays unknown.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from dgx_monarch.studio.contracts import StudioError
from dgx_monarch.studio.management_consent import PENDING_LIMIT, consent_plan
from dgx_monarch.studio.operations import Operations
from dgx_monarch.studio.store import Store

CARD = {"id": "p-0a1b2c3d4e5f", "key": "capacity:3f2a", "consent_id": "c-1", "kind": "capacity", "class": "C",
        "title": "Run big_model.safetensors past the measured capacity", "artifact": "big_model.safetensors",
        "risk": "The render may run out of memory.", "occurrences": 2, "seen_at": 1760000000.0,
        "primary": {"label": "Accept", "action": "accept"}}
OTHER = {**CARD, "id": "p-ffeeddccbbaa", "key": "capacity:9c1d", "title": "Another pending request"}


def refused(status, detail=None):
    """The C1 contract: the transport carries the driver's status and a bounded reason."""
    error = StudioError("driver_rejected", f"Driver returned HTTP {status}; inspect its local diagnostics.", 502)
    error.upstream_status, error.upstream_detail = status, detail
    return error


class Driver:
    origin = "http://127.0.0.1:8188"

    def __init__(self, pending=(), failure=None, consents=None):
        self.calls = []
        self.pending = [deepcopy(card) for card in pending]
        self.failure = failure
        self.consents = consents

    async def request(self, method, path, data=None):
        self.calls.append((method, path, deepcopy(data)))
        if (method, path) == ("GET", "/dgxm/consents"):
            return self.consents if self.consents is not None else {"pending": self.pending, "active": []}
        if self.failure is not None:
            raise self.failure
        return {"ok": True}


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state")
    yield value
    value.close()


def review(store, driver, parameters=None, *, allow=True):
    ops = Operations(store, driver, allow_management=allow)
    return ops, asyncio.run(ops.prepare({"action": "consent.apply", "parameters": parameters or {}}))


def apply(ops, plan, key="once"):
    async def scenario():
        record = ops.apply(plan["id"], {"digest": plan["digest"], "confirmed": True, "idempotency_key": key})
        await asyncio.gather(*ops.tasks)
        return ops.store.get("operations", record["id"])
    return asyncio.run(scenario())


def test_empty_fields_select_the_one_pending_request_and_accept_it_exactly(store):
    driver = Driver([CARD])
    ops, plan = review(store, driver, {"key": "", "id": ""})
    assert plan["available"] and plan["requires_confirmation"]
    assert plan["parameters"] == {"key": CARD["key"], "id": CARD["id"]}
    assert plan["adapter_plan"]["card"]["title"] == CARD["title"] and plan["adapter_plan"]["pending_count"] == 1
    result = apply(ops, plan)
    assert result["state"] == "succeeded"
    assert driver.calls == [("GET", "/dgxm/consents", None),
                            ("POST", "/dgxm/consent", {"key": CARD["key"], "id": CARD["id"], "action": "accept"})]


def test_typed_identity_must_match_a_pending_request(store):
    driver = Driver([CARD, OTHER])
    ops, plan = review(store, driver, {"key": OTHER["key"], "id": OTHER["id"]})
    assert plan["available"] and plan["adapter_plan"]["card"]["id"] == OTHER["id"]
    _, stale = review(store, driver, {"key": OTHER["key"], "id": CARD["id"]})
    assert stale["available"] is False and "not a pending consent request" in stale["reason"]
    assert {row["id"] for row in stale["adapter_plan"]["pending"]} == {CARD["id"], OTHER["id"]}
    with pytest.raises(StudioError) as caught:
        ops.apply(stale["id"], {"digest": stale["digest"], "confirmed": True, "idempotency_key": "stale"})
    assert caught.value.code == "operation_unavailable"
    assert [call for call in driver.calls if call[0] == "POST"] == []


@pytest.mark.parametrize(("pending", "phrase"), [([CARD, OTHER], "Several"), ([], "No consent request is pending")])
def test_empty_fields_list_pending_requests_when_none_or_several_wait(store, pending, phrase):
    _, plan = review(store, Driver(pending))
    assert plan["available"] is False and phrase in plan["reason"]
    assert [row["key"] for row in plan["adapter_plan"]["pending"]] == [card["key"] for card in pending]
    assert plan["parameters"] == {"key": "", "id": ""}


@pytest.mark.parametrize("parameters", [{"key": CARD["key"], "id": ""}, {"key": "", "id": CARD["id"]},
                                        {"key": "bad\nkey", "id": "p"}, {"key": 3, "id": "p"}])
def test_half_or_malformed_identity_is_refused(store, parameters):
    with pytest.raises(StudioError) as caught:
        review(store, Driver([CARD]), parameters)
    assert (caught.value.code, caught.value.status) == ("operation_fields", 400)


@pytest.mark.parametrize("consents", [{"pending": "not a list"}, ["pending"], {}])
def test_malformed_pending_list_is_a_driver_refusal(store, consents):
    with pytest.raises(StudioError) as caught:
        review(store, Driver(consents=consents))
    assert (caught.value.code, caught.value.status) == ("consent_state", 502)


def test_pending_cards_keep_exact_identities_and_bound_display_text():
    hostile = {**CARD, "title": "line\nbreak" + "x" * 400, "artifact": {"nested": True}}
    noise = [{"key": "", "id": "p"}, {"key": "k\x00", "id": "p"}, "not a card"]
    parameters, plan = consent_plan({"key": "", "id": ""}, {"pending": [*noise, hostile]})
    assert parameters == {"key": CARD["key"], "id": CARD["id"]}
    assert "\n" not in plan["card"]["title"] and len(plan["card"]["title"]) == 240
    assert plan["card"]["artifact"] is None and "primary" not in plan["card"]
    many = [{**CARD, "id": f"p-{index}"} for index in range(PENDING_LIMIT + 5)]
    _, listing = consent_plan({"key": "", "id": ""}, {"pending": many})
    assert len(listing["pending"]) == PENDING_LIMIT


def test_review_without_the_driver_list_is_refused(store):
    ops = Operations(store, Driver([CARD]), allow_management=True)
    with pytest.raises(StudioError) as caught:
        ops.preview({"action": "consent.apply", "parameters": {"key": CARD["key"], "id": CARD["id"]}})
    assert (caught.value.code, caught.value.status) == ("operation_plan", 409)


def test_without_management_authority_the_list_is_not_read(store):
    driver = Driver([CARD])
    _, plan = review(store, driver, allow=False)
    assert plan["available"] is False and "--allow-management" in plan["reason"]
    assert driver.calls == []


def test_consent_fields_are_optional_and_explained(store):
    row = next(row for row in Operations(store, Driver()).capabilities() if row["id"] == "consent.apply")
    assert row["parameters"] == [{"name": "key", "type": "STRING", "default": ""},
                                 {"name": "id", "type": "STRING", "default": ""}]
    assert "pending" in row["description"]


@pytest.mark.parametrize(("status", "detail"), [
    (409, "no pending consent for that request; queue the render again"), (400, "consent action needs the pending id"),
    (429, "too many consent actions; try again in a moment"),
])
def test_consent_refusal_settles_failed_with_the_driver_reason(store, status, detail):
    driver = Driver([CARD], failure=refused(status, detail))
    ops, plan = review(store, driver)
    result = apply(ops, plan)
    assert result["state"] == "failed"
    assert result["receipt"] == {"source": "driver", "error": "driver_rejected", "upstream_status": status, "detail": detail}


@pytest.mark.parametrize("failure", [refused(503, "consent action timed out server-side; see driver log"),
                                     refused(500), StudioError("driver_unknown", "lost response", 503)])
def test_consent_server_error_or_lost_response_stays_unknown(store, failure):
    ops, plan = review(store, Driver([CARD], failure=failure))
    result = apply(ops, plan)
    assert result["state"] == "unknown" and result["receipt"]["error"] == failure.code


def test_recycle_refusal_settles_failed_and_free_refusal_stays_unknown(store):
    for action, state in (("actors.recycle", "failed"), ("driver.free", "unknown")):
        ops = Operations(store, Driver(failure=refused(409, "a render is running")), allow_management=True)
        plan = ops.preview({"action": action})
        assert apply(ops, plan, key=action)["state"] == state


def test_driver_detail_is_bounded_and_never_raw_structure(store):
    for index, detail in enumerate(("bad\x07reason " + "y" * 400, {"node_errors": []})):
        ops, plan = review(store, Driver([CARD], failure=refused(409, detail)))
        receipt = apply(ops, plan, key=f"detail-{index}")["receipt"]
        if isinstance(detail, str):
            assert "\x07" not in receipt["detail"] and len(receipt["detail"]) == 300
        else:
            assert receipt["detail"] is None
