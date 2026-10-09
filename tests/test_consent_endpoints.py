"""Consent endpoints, the kind registry, the card model, and the pending registry.

The server computes the whole card, so most of what a browser test would have
clicked is asserted here in Python.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import types

import pytest

pytest.importorskip("aiohttp")

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch import consent_pending, consent_store
from dgx_monarch.nodes import consent_routes

CONTEXT = {"combo_key": "9a3f2c81d4e6f0a1b2c3d4e5", "weight_dtype": "default"}
COMBO = "9a3f2c81d4e6f0a1b2c3d4e5"
ARTIFACTS = "b" * 64
LEGACY_ARTIFACTS = "legacy-checkpoint-digest"
CAPABILITY_CONTEXT = {"worker_args": {"slab_weights": "auto"}, "world": 1}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(consent_store, "MEMO_PATH",
                        str(tmp_path / "cache" / "dgx-monarch" / "consent_memo.json"))
    fake_folder = types.ModuleType("folder_paths")
    fake_folder.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder)
    monkeypatch.setattr(consent_routes, "_RATE",
                        {"tokens": float(consent_routes.RATE_LIMIT_N), "t": 0.0})
    for spec in consent_pending.KIND_SPECS.values():
        monkeypatch.delenv(spec.env_var, raising=False)
    monkeypatch.delenv(consent_pending.AUTO_RESCUE_ENV, raising=False)
    consent_pending.clear_all()
    yield
    consent_pending.clear_all()


def _checkpoint(tmp_path, name="minimax_h3_fl2va_bf16.safetensors"):
    path = tmp_path / name
    path.write_bytes(b"weights")
    return str(path)


def _descriptor(path, **overrides):
    fields = {
        "kind": "rescue-slab",
        "path": path,
        "memo_context": dict(CONTEXT),
        "combo_key": COMBO,
        "artifacts": ARTIFACTS,
        "artifacts_legacy": LEGACY_ARTIFACTS,
        "artifacts_legacy_complete": True,
        "capability_context": CAPABILITY_CONTEXT,
        "file_identity": consent_store.file_identity(path),
        "unet_name": os.path.basename(path),
        "loras": 0,
        "numbers": "needs 61.7 GiB, 44.2 GiB available",
    }
    fields.update(overrides)
    return consent_pending.ConsentDescriptor(**fields)


def _fake_recorder(recorded):
    def record(**kwargs):
        recorded.append(kwargs)
        return f"audit:waiver:{kwargs['kind']}:{kwargs['combo_key']}"

    return record


def _ledger(tmp_path, monkeypatch):
    from dgx_monarch.gate_ledger import GateLedger

    fake_folder = types.ModuleType("folder_paths")
    fake_folder.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder)
    return GateLedger(str(tmp_path))


def test_every_kind_is_completely_specified():
    from dgx_monarch.gate_audit_vocab import KIND_CLASS, WAIVER_KINDS, check_guard

    assert set(consent_pending.KIND_SPECS) == {
        "rescue-slab", "waive-unvouched-slab", "waive-first-load-stock",
        "waive-known-wrong:ring-pad", "waive-known-wrong:pixeldit-sp",
        "waive-known-wrong:sol-attn", "waive-known-wrong:shard-quant",
    }
    # Every card kind is a member of the protocol-bound audit-row vocabulary
    # (gate_audit_vocab), and its class and guard come from there, so a card
    # can never present a bypass as a class it is not.
    assert set(consent_pending.KIND_SPECS) <= WAIVER_KINDS
    for kind, spec in consent_pending.KIND_SPECS.items():
        assert spec.kind == kind
        assert spec.refusal_class == KIND_CLASS[kind]
        assert spec.style in {"capacity", "accuracy"}
        assert spec.title and spec.risk and spec.primary_label and spec.env_var
        assert spec.context_fields
        assert check_guard(kind, spec.default_guard) == spec.default_guard
        assert (spec.measured is not None) == (spec.refusal_class == "K")


def test_known_wrong_math_is_never_auto_eligible():
    """Known-wrong math has no auto-rescue analog (DESIGN section 5.9, class K).
    A known-wrong kind that turns auto-eligible fails here."""
    for kind, spec in consent_pending.KIND_SPECS.items():
        if kind.startswith("waive-known-wrong:"):
            assert spec.auto_eligible is False
            assert spec.style == "accuracy"


def test_every_kind_can_build_the_shipped_refusal_sentence():
    """The contract is checked against the one sentence generator that ships.

    `refusal.refusal(..., panel_action=...)` is what every refusal site calls,
    so the panel button, the wording and the headless value are asserted there
    and nowhere else. A second generator in a test drifts from the shipped text.
    """
    from dgx_monarch import accuracy_waiver
    from dgx_monarch.refusal import GUARDS, PanelAction, RefusalClass, refusal

    for spec in consent_pending.KIND_SPECS.values():
        if not spec.wired:
            continue
        # A class-K kind's `default_guard` is the unscoped prefix the ledger
        # vocabulary validates and the headless value names. Most raise sites
        # add a family scope (`ring_pad:minimax_h3`), but `next` takes the first
        # GUARDS entry for the kind, which is the unscoped spelling.
        guard = spec.default_guard
        env_value = "1"
        if spec.refusal_class == "K":
            guard = next(name for name, entry in GUARDS.items()
                         if entry.consent_kind == spec.kind)
            env_value = accuracy_waiver.panel_action(guard).env_value
            assert env_value == spec.default_guard
        text = refusal(
            RefusalClass(spec.refusal_class),
            f"model.safetensors: {spec.risk} Measured: 61.7 GiB.",
            guard=guard, waivable=True,
            panel_action=PanelAction(spec.primary_label, spec.env_var, env_value))
        assert f'"{spec.primary_label}"' in text
        assert "DGX Monarch" in text and "panel" in text
        assert f"{spec.env_var}={env_value}" in text
        assert text.index("DGX Monarch") < text.index(spec.env_var)


def test_an_unknown_kind_has_no_context_and_no_card():
    with pytest.raises(ValueError):
        consent_pending.consent_context("no-such-kind")


def test_the_card_offers_exactly_one_primary_action(tmp_path):
    card = consent_pending.register_pending(_descriptor(_checkpoint(tmp_path)))
    assert card is not None
    assert card["primary"] == {"label": "Load with slab residency", "action": "accept"}
    assert "secondary" not in card and "actions" not in card
    serialized = json.dumps(card)
    assert serialized.count('"label"') == 1
    assert serialized.count('"action"') == 1
    assert card["class"] == "C" and card["style"] == "capacity"
    assert card["numbers"] == "needs 61.7 GiB, 44.2 GiB available"


def test_a_card_carries_no_absolute_path(tmp_path):
    consent_pending.register_pending(_descriptor(_checkpoint(tmp_path)))
    for card in consent_pending.pending_cards():
        for value in card.values():
            assert os.sep not in str(value)


def test_the_same_failing_graph_queued_five_times_is_one_card(tmp_path):
    descriptor = _descriptor(_checkpoint(tmp_path))
    for _ in range(5):
        consent_pending.register_pending(descriptor)
    cards = consent_pending.pending_cards()
    assert len(cards) == 1
    assert cards[0]["occurrences"] == 5


def test_a_re_ask_refreshes_the_numbers_and_keeps_the_id(tmp_path):
    """The id is the identity of the question, not of the measurement.

    The numbers carry a live MemAvailable reading and move between queues.
    Rotating the id on that would 409 a click on a freshly rendered card, so
    the same memo key keeps the same id and the numbers are refreshed in place.
    """
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    second = consent_pending.register_pending(_descriptor(path, numbers="needs 61.7 GiB, 2 GiB available"))
    assert first is not None and second is not None
    assert first["id"] == second["id"]
    assert second["numbers"] == "needs 61.7 GiB, 2 GiB available"
    assert second["occurrences"] == 2
    # The id still has to answer a live question: an id from a question that
    # was taken, or any other id, answers nothing.
    assert consent_pending.take_pending(second["key"], "p-000000000000") is None
    assert consent_pending.take_pending(second["key"], str(first["id"])) is not None


def test_the_pending_registry_is_bounded_and_expires(tmp_path, monkeypatch):
    monkeypatch.setattr(consent_pending, "PENDING_LIMIT", 3)
    for index in range(5):
        consent_pending.register_pending(
            _descriptor(_checkpoint(tmp_path, name=f"m{index}.safetensors")))
    assert consent_pending.pending_count() == 3

    monkeypatch.setattr(consent_pending, "PENDING_TTL_S", -1.0)
    assert consent_pending.pending_count() == 0


def test_registration_never_masks_a_refusal(tmp_path, monkeypatch):
    """A bookkeeping failure returns None; it must not raise into the refusal
    the user needs to read."""
    monkeypatch.setattr(consent_store, "memo_key",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    assert consent_pending.register_pending(_descriptor(_checkpoint(tmp_path))) is None
    assert consent_pending.register_pending({"kind": "nope"}) is None
    assert consent_pending.register_pending(_descriptor(_checkpoint(tmp_path), combo_key="")) is None


def test_kinds_that_are_not_wired_raise_no_card(tmp_path):
    """An explicit slab_weights=on loads slab for any family, and a rescue grant
    subsumes the first-load-stock bypass, so neither raises its own card."""
    path = _checkpoint(tmp_path)
    for kind in ("waive-unvouched-slab", "waive-first-load-stock"):
        assert consent_pending.register_pending(
            _descriptor(path, kind=kind, target_guard="",
                        memo_context=dict(CONTEXT))) is None
    assert consent_pending.pending_count() == 0


def test_a_descriptor_may_be_a_plain_mapping(tmp_path):
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending({
        "kind": "rescue-slab", "path": path, "memo_context": dict(CONTEXT),
        "combo_key": COMBO, "artifacts": ARTIFACTS, "numbers": "needs 61.7 GiB",
    })
    assert card is not None and card["artifact"] == os.path.basename(path)


def test_a_worker_side_refusal_becomes_a_card_with_the_driver_local_half(tmp_path):
    """The refusal descriptor crosses a process boundary and carries no host
    path; the driver holds the path, the combination and the trust context."""
    path = _checkpoint(tmp_path)
    refusal = {
        "version": 1,
        "kind": "rescue-slab",
        "consent_id": "f" * 32,
        "unet_name": os.path.basename(path),
        "file_identity": consent_store.file_identity(path),
        "measured": {"probe": "stock_load_preflight",
                     "checkpoint_bytes": 66266529792,
                     "mem_available_bytes": 47458156544},
    }
    desc = consent_pending.descriptor_from_refusal(
        refusal, path=path, combo_key=COMBO, artifacts=ARTIFACTS,
        memo_context=dict(CONTEXT), capability_context={"world": 2})
    assert desc.numbers == "needs 61.7 GiB, 44.2 GiB available"
    assert desc.target_guard == "stock_load_preflight"
    assert desc.artifact == os.path.basename(path)

    card = consent_pending.register_pending(desc)
    assert card is not None and card["consent_id"] == "f" * 32
    assert "61.7 GiB" in str(card["numbers"])


def test_an_id_the_permanent_row_would_refuse_is_not_carried(tmp_path):
    """A grant that fails row validation at click time is worse than two ids in
    a log, so an id of the wrong shape is replaced, never carried."""
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path, consent_id="c-3f9a12c4b7e0"))
    assert card is not None
    assert card["consent_id"] != "c-3f9a12c4b7e0"
    assert len(str(card["consent_id"])) == consent_pending.CONSENT_ID_HEX


def test_measured_numbers_are_derived_only_from_a_complete_block():
    assert consent_pending.numbers_from_measured(None) is None
    assert consent_pending.numbers_from_measured({"checkpoint_bytes": 1}) is None
    assert consent_pending.numbers_from_measured(
        {"checkpoint_bytes": 1 << 30, "mem_available_bytes": 2 << 30}
    ) == "needs 1.0 GiB, 2.0 GiB available"


def test_resolution_order_memo_then_toggle_then_environment(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    assert consent_pending.resolve(kind="rescue-slab", path=path, context=CONTEXT) is None

    monkeypatch.setenv("DGXM_ALLOW_SLAB_RESCUE", "1")
    grant = consent_pending.resolve(kind="rescue-slab", path=path, context=CONTEXT)
    assert grant is not None and grant.consent_source == "env"
    assert grant.key == ""
    assert consent_store.read()["consents"] == {}, "an env grant must never write a memo"

    consent_store.set_auto_rescue(True)
    grant = consent_pending.resolve(kind="rescue-slab", path=path, context=CONTEXT)
    assert grant is not None and grant.consent_source == "auto_rescue"

    consent_store.set_auto_rescue(False)
    monkeypatch.delenv("DGXM_ALLOW_SLAB_RESCUE")
    assert consent_pending.resolve(kind="rescue-slab", path=path, context=CONTEXT) is None


def test_the_standing_toggle_never_grants_a_known_wrong_waiver(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    consent_store.set_auto_rescue(True)
    monkeypatch.setenv(consent_pending.AUTO_RESCUE_ENV, "1")
    context = consent_pending.consent_context(
        "waive-known-wrong:ring-pad", combo_key=COMBO, topology="ring2", world=2)
    assert consent_pending.resolve(
        kind="waive-known-wrong:ring-pad", path=path, context=context) is None


def test_a_known_wrong_environment_grant_must_name_its_guard(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    context = consent_pending.consent_context(
        "waive-known-wrong:ring-pad", combo_key=COMBO, topology="ring2", world=2)
    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", "1")
    assert consent_pending.resolve(
        kind="waive-known-wrong:ring-pad", path=path, context=context) is None

    monkeypatch.setenv("DGXM_WAIVE_KNOWN_WRONG", "ring_pad")
    grant = consent_pending.resolve(
        kind="waive-known-wrong:ring-pad", path=path, context=context)
    assert grant is not None and grant.consent_source == "env"


def test_resolution_degrades_to_ask_when_the_store_is_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(consent_store, "lookup",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert consent_pending.resolve(
        kind="rescue-slab", path=_checkpoint(tmp_path), context=CONTEXT) is None
    assert consent_pending.resolve(kind="unknown-kind", path="x", context={}) is None


def test_a_grant_stamp_degrades_safely_in_both_skew_directions():
    grant = consent_pending.ConsentGrant(
        id="a" * 32, kind="rescue-slab", consent_source="panel", key="", reason="r")
    wire = consent_pending.grant_wire(grant)
    assert wire == {"id": "a" * 32, "kind": "rescue-slab", "consent_source": "panel"}

    read_back = consent_pending.grant_from_wire(wire)
    assert read_back is not None and read_back.kind == "rescue-slab"
    # a newer driver's extra fields stay readable by an older reader
    assert consent_pending.grant_from_wire({**wire, "future": 1}) is not None
    # and every degraded shape reads as "no grant", never as "the bypass happened"
    assert consent_pending.grant_from_wire({"kind": "rescue-slab"}) is None
    assert consent_pending.grant_from_wire({**wire, "kind": "kind-from-the-future"}) is None
    assert consent_pending.grant_from_wire(None) is None
    assert consent_pending.grant_from_wire("rescue-slab") is None


async def _build_client(monkeypatch):
    from dgx_monarch.nodes import routes as routes_mod

    fake_mesh = types.ModuleType("dgx_monarch.mesh")
    fake_mesh._MESHES = {}
    fake_mesh._MESH_LOCK = threading.Lock()
    monkeypatch.setitem(sys.modules, "dgx_monarch.mesh", fake_mesh)

    route_table = web.RouteTableDef()
    fake_app = types.SimpleNamespace(routes=route_table)
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(instance=fake_app)
    monkeypatch.setitem(sys.modules, "server", fake_server)

    routes_mod.register()
    application = web.Application()
    application.add_routes(route_table)
    client = TestClient(TestServer(application))
    await client.start_server()
    return client


def _headers(client, action="consent"):
    url = client.make_url("/")
    return {"X-DGXM-Action": action,
            "Origin": f"{url.scheme}://{url.host}:{url.port}",
            "Sec-Fetch-Site": "same-origin",
            "Content-Type": "application/json"}


def _post(client, body, action="consent"):
    return client.post("/dgxm/consent", headers=_headers(client, action),
                       data=json.dumps(body))


def test_get_serves_cards_active_rows_and_the_toggle(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    consent_pending.register_pending(_descriptor(path))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.get("/dgxm/consents")
            assert resp.status == 200
            payload = await resp.json()
            assert payload["schema"] == consent_store.SCHEMA
            assert payload["auto_rescue"] is False
            assert len(payload["pending"]) == 1
            assert payload["active"] == []
            assert payload["kinds"]["rescue-slab"]["auto_eligible"] is True
            assert payload["pending"][0]["style"] == "capacity"
            body = json.dumps(payload)
            assert str(tmp_path) not in body
            assert consent_store.file_identity(path) not in body
        finally:
            await client.close()

    asyncio.run(scenario())


def test_post_requires_the_action_header_and_a_same_site_origin(monkeypatch):
    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.post("/dgxm/consent", data="{}")
            assert resp.status == 403
            assert "X-DGXM-Action" in await resp.text()

            headers = {**_headers(client), "Sec-Fetch-Site": "cross-site"}
            resp = await client.post("/dgxm/consent", headers=headers, data="{}")
            assert resp.status == 403
            assert "cross-site consent" in await resp.text()

            headers = {**_headers(client), "Origin": "http://attacker.example"}
            resp = await client.post("/dgxm/consent", headers=headers, data="{}")
            assert resp.status == 403
            assert "consent Origin does not match" in await resp.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_accept_without_a_live_pending_entry_is_refused(monkeypatch):
    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await _post(client, {"action": "accept", "key": "k" * 32, "id": "p-1"})
            assert resp.status == 409
            assert (await resp.json())["ok"] is False
        finally:
            await client.close()

    asyncio.run(scenario())


def test_accept_refuses_a_card_after_persisted_gate_fail(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    _ledger(tmp_path, monkeypatch).record(
        COMBO, ARTIFACTS, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]},
        CAPABILITY_CONTEXT)

    status, payload = consent_routes.handle_action(
        {"action": "accept", "key": card["key"], "id": card["id"]})

    assert status == 409
    assert payload["ok"] is False and "FAILED" in payload["detail"]
    assert recorded == []
    assert consent_pending.pending_count() == 0
    assert consent_store.read()["consents"] == {}


def test_consent_state_omits_quarantined_rows_without_mutating_them(
        tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    assert first is not None
    assert consent_routes._accept(first["key"], first["id"])[0] == 200
    second = consent_pending.register_pending(_descriptor(path))
    assert second is not None
    _ledger(tmp_path, monkeypatch).record(
        COMBO, ARTIFACTS, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]},
        CAPABILITY_CONTEXT)

    state = consent_routes.consent_state()

    assert state["pending"] == []
    assert state["active"] == []
    assert consent_pending.pending_count() == 1
    assert first["key"] in consent_store.read()["consents"]
    assert [row["action"] for row in recorded] == ["grant"]


def test_consent_state_uses_one_memo_snapshot_for_rows_and_quarantine(
        tmp_path, monkeypatch):
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder([]))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None and consent_routes._accept(card["key"], card["id"])[0] == 200
    first_snapshot = consent_store.read()
    second_snapshot = json.loads(json.dumps(first_snapshot))
    failed_context = {"worker_args": {"slab_weights": False}, "world": 2}
    second_snapshot["consents"][card["key"]]["capability_context"] = failed_context
    _ledger(tmp_path, monkeypatch).record(
        COMBO, ARTIFACTS, "deadbeef", "FAIL", context=failed_context)
    snapshots = iter((first_snapshot, second_snapshot, second_snapshot))
    reads: list[dict] = []

    def swapped_read():
        memo = next(snapshots)
        reads.append(memo)
        return memo

    monkeypatch.setattr(consent_store, "read", swapped_read)
    state = consent_routes.consent_state()

    assert len(reads) == 1
    assert [row["key"] for row in state["active"]] == [card["key"]]


def test_fail_then_pass_allows_the_fresh_consent(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    ledger = _ledger(tmp_path, monkeypatch)
    context = CAPABILITY_CONTEXT
    ledger.record(
        COMBO, LEGACY_ARTIFACTS, "deadbeef", "FAIL",
        {"quarantine_levers": ["slab_weights", "lora_low_rss"]}, context)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "PASS", context=context)

    status, payload = consent_routes._accept(card["key"], card["id"])

    assert status == 200 and payload["ok"] is True
    assert [row["action"] for row in recorded] == ["grant"]
    assert consent_pending.resolve(
        kind="rescue-slab", path=path, context=CONTEXT) is not None


def test_pass_in_another_capability_context_does_not_clear_fail(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    failed_context = {"worker_args": {"slab_weights": False}, "world": 2}
    card = consent_pending.register_pending(
        _descriptor(path, capability_context=failed_context))
    assert card is not None
    ledger = _ledger(tmp_path, monkeypatch)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "FAIL", context=failed_context)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "PASS", context=CAPABILITY_CONTEXT)

    status, payload = consent_routes._accept(card["key"], card["id"])

    assert status == 409 and "FAILED" in payload["detail"]
    assert recorded == []


@pytest.mark.parametrize("superseder", ["legacy", "wrong_source"])
def test_unscoped_pass_cannot_clear_unscoped_fail_for_contextual_consent(
        tmp_path, monkeypatch, superseder):
    from dgx_monarch import __version__
    from dgx_monarch.gate_ledger import GATE_PROTOCOL_VERSION

    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    assert first is not None and consent_routes._accept(first["key"], first["id"])[0] == 200
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    ledger = _ledger(tmp_path, monkeypatch)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "FAIL")
    detail = ({} if superseder == "legacy" else {
        "gate_protocol": GATE_PROTOCOL_VERSION,
        "dgx_monarch": __version__,
        "dgx_source": "foreign-source",
    })
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "PASS", detail=detail)

    state = consent_routes.consent_state()
    assert state["pending"] == [] and state["active"] == []
    assert consent_pending.pending_count() == 1
    assert first["key"] in consent_store.read()["consents"]
    status, payload = consent_routes._accept(card["key"], card["id"])

    assert status == 409 and "FAILED" in payload["detail"]
    assert consent_pending.pending_count() == 0
    assert first["key"] not in consent_store.read()["consents"]
    assert [row["action"] for row in recorded] == ["grant", "revoke"]


@pytest.mark.parametrize("stale_dimension", ["protocol", "package"])
def test_stale_contextual_pass_cannot_clear_compatible_historical_fail(
        tmp_path, monkeypatch, stale_dimension):
    from dgx_monarch import gate_ledger as ledger_mod

    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    assert first is not None and consent_routes._accept(first["key"], first["id"])[0] == 200
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    ledger = _ledger(tmp_path, monkeypatch)
    # The FAIL comes from the running release, which is what makes the stale
    # PASS the subject. A scoped FAIL that is itself superseded retires for the
    # gate and for consent alike, covered in test_consent_wiring.py.
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "FAIL", context=CAPABILITY_CONTEXT)
    with monkeypatch.context() as stale:
        if stale_dimension == "protocol":
            stale.setattr(ledger_mod, "GATE_PROTOCOL_VERSION",
                          ledger_mod.GATE_PROTOCOL_VERSION - 1)
        else:
            stale.setattr(ledger_mod, "__version__", "stale-package")
        ledger.record(COMBO, ARTIFACTS, "deadbeef", "PASS", context=CAPABILITY_CONTEXT)

    state = consent_routes.consent_state()
    assert state["pending"] == [] and state["active"] == []
    assert first["key"] in consent_store.read()["consents"]
    status, payload = consent_routes._accept(card["key"], card["id"])

    assert status == 409 and "FAILED" in payload["detail"]
    assert first["key"] not in consent_store.read()["consents"]
    assert [row["action"] for row in recorded] == ["grant", "revoke"]


def test_retesting_after_fail_preserves_quarantine_and_consumes_card(tmp_path, monkeypatch):
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    ledger = _ledger(tmp_path, monkeypatch)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "FAIL", context=CAPABILITY_CONTEXT)
    ledger.begin_retest_required(COMBO, ARTIFACTS, "deadbeef", [CAPABILITY_CONTEXT])

    state = consent_routes.consent_state()
    status, payload = consent_routes._accept(card["key"], card["id"])

    assert state["pending"] == []
    assert status == 409 and "FAILED" in payload["detail"]
    assert consent_pending.pending_count() == 0


def test_inconclusive_after_fail_preserves_quarantine_and_revokes(
        tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    assert first is not None and consent_routes._accept(first["key"], first["id"])[0] == 200
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    ledger = _ledger(tmp_path, monkeypatch)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "FAIL", context=CAPABILITY_CONTEXT)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "INCONCLUSIVE",
                  context=CAPABILITY_CONTEXT)

    state = consent_routes.consent_state()
    status, payload = consent_routes._accept(card["key"], card["id"])

    assert state["pending"] == [] and state["active"] == []
    assert status == 409 and "FAILED" in payload["detail"]
    assert consent_pending.pending_count() == 0
    assert first["key"] not in consent_store.read()["consents"]
    assert [row["action"] for row in recorded] == ["grant", "revoke"]


@pytest.mark.parametrize("terminal", ["INCONCLUSIVE", "RETESTING"])
def test_foreign_source_terminal_cannot_mask_older_fail(
        tmp_path, monkeypatch, terminal):
    from dgx_monarch import runtime_provenance

    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    assert first is not None and consent_routes._accept(first["key"], first["id"])[0] == 200
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    ledger = _ledger(tmp_path, monkeypatch)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "FAIL", context=CAPABILITY_CONTEXT)
    if terminal == "RETESTING":
        ledger.begin_retest_required(COMBO, ARTIFACTS, "deadbeef", [CAPABILITY_CONTEXT])
    else:
        ledger.record(COMBO, ARTIFACTS, "deadbeef", terminal,
                      context=CAPABILITY_CONTEXT)
    monkeypatch.setattr(
        runtime_provenance, "cached_dgx_source_manifest_sha256",
        lambda: "foreign-runtime-source")

    state = consent_routes.consent_state()
    assert state["pending"] == [] and state["active"] == []
    assert first["key"] in consent_store.read()["consents"]
    status, payload = consent_routes._accept(card["key"], card["id"])

    assert status == 409 and "FAILED" in payload["detail"]
    assert first["key"] not in consent_store.read()["consents"]
    assert [row["action"] for row in recorded] == ["grant", "revoke"]


@pytest.mark.parametrize("failure", ["ledger_scan", "output_directory"])
def test_quarantine_read_failure_hides_and_refuses_without_mutation(
        tmp_path, monkeypatch, failure):
    from dgx_monarch.gate_ledger import GateLedger

    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    _ledger(tmp_path, monkeypatch)
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    assert first is not None and consent_routes._accept(first["key"], first["id"])[0] == 200
    second = consent_pending.register_pending(_descriptor(path))
    assert second is not None

    def unreadable(*_args, **_kwargs):
        raise OSError("ledger unavailable")

    if failure == "ledger_scan":
        monkeypatch.setattr(GateLedger, "entries_with_integrity", unreadable)
    else:
        sys.modules["folder_paths"].get_output_directory = unreadable

    state = consent_routes.consent_state()
    status, payload = consent_routes._accept(second["key"], second["id"])

    assert state["pending"] == [] and state["active"] == []
    assert status == 409 and "could not be read" in payload["detail"]
    assert consent_pending.pending_count() == 1
    assert first["key"] in consent_store.read()["consents"]
    assert [row["action"] for row in recorded] == ["grant"]


def test_torn_row_after_fail_then_pass_hides_and_refuses_without_mutation(
        tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    first = consent_pending.register_pending(_descriptor(path))
    assert first is not None and consent_routes._accept(first["key"], first["id"])[0] == 200
    second = consent_pending.register_pending(_descriptor(path))
    assert second is not None
    ledger = _ledger(tmp_path, monkeypatch)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "FAIL", context=CAPABILITY_CONTEXT)
    ledger.record(COMBO, ARTIFACTS, "deadbeef", "PASS", context=CAPABILITY_CONTEXT)
    with open(ledger.path, "ab") as handle:
        handle.write(b'{"key":"concealed-newer-row"')

    state = consent_routes.consent_state()
    status, payload = consent_routes._accept(second["key"], second["id"])

    assert state["pending"] == [] and state["active"] == []
    assert status == 409 and "unambiguously" in payload["detail"]
    assert consent_pending.pending_count() == 1
    assert first["key"] in consent_store.read()["consents"]
    assert [row["action"] for row in recorded] == ["grant"]


@pytest.mark.parametrize("epoch", [10 ** 1000, float("inf")], ids=["overflow", "infinity"])
def test_hostile_grant_epoch_cannot_poison_active_list_or_state(
        tmp_path, monkeypatch, epoch):
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder([]))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None and consent_routes._accept(card["key"], card["id"])[0] == 200
    with open(consent_store.MEMO_PATH) as handle:
        memo = json.load(handle)
    memo["consents"][card["key"]]["granted_epoch"] = epoch
    with open(consent_store.MEMO_PATH, "w") as handle:
        json.dump(memo, handle)

    assert consent_store.list_active() == []
    assert consent_routes.consent_state()["active"] == []
    assert consent_store.lookup("rescue-slab", path, CONTEXT) is None


def test_one_click_writes_one_ledger_row_then_the_memo(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            assert resp.status == 200
            body = await resp.json()
            assert body["ok"] is True
            assert body["ledger"] == {"recorded": True, "key": f"audit:waiver:rescue-slab:{COMBO}"}
            assert body["consent"]["consent_source"] == "panel"

            # a second click on a stale tab grants nothing new
            resp = await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            assert resp.status == 409
        finally:
            await client.close()

    asyncio.run(scenario())

    assert len(recorded) == 1
    row = recorded[0]
    assert row["action"] == "grant" and row["kind"] == "rescue-slab"
    assert row["target_guard"] == "stock_load_preflight"
    assert row["consent_source"] == "panel"
    assert row["artifacts"] == ARTIFACTS and row["combo_key"] == COMBO
    assert row["memo_context"] == CONTEXT, "the row fingerprints the NARROW memo context"
    assert row["capability_context"]["world"] == 1, "and records the full trust context"
    assert row["file_identity"] == consent_store.file_identity(path)
    assert row["consent_id"] == card["consent_id"]
    assert "61.7 GiB" in row["reason"] and len(row["reason"]) <= consent_routes.REASON_MAX_CHARS
    assert row["stamp"] is None, "a capacity rescue stamps no output"

    grant = consent_pending.resolve(kind="rescue-slab", path=path, context=CONTEXT)
    assert grant is not None and grant.consent_source == "panel"
    assert grant.id == row["consent_id"]
    assert consent_pending.pending_count() == 0


def test_a_checkpoint_that_changed_since_the_refusal_is_refused_not_laundered(
        tmp_path, monkeypatch):
    """The card was minted for the bytes that were on disk when the render
    refused. If the file changed in between, granting under the old key would
    file the memo where no lookup can find it while the permanent row named a
    different file: refuse, and ask again."""
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))
    assert card is not None
    with open(path, "ab") as handle:            # a re-stage, an rsync, a touch
        handle.write(b"more bytes")

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            assert resp.status == 409
            body = await resp.json()
            assert body["ok"] is False
            assert "checkpoint changed" in body["detail"]
        finally:
            await client.close()

    asyncio.run(scenario())
    assert recorded == []
    assert consent_store.read()["consents"] == {}


def test_accepting_an_already_granted_consent_writes_no_second_row(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            first = await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            assert first.status == 200
            again = consent_pending.register_pending(_descriptor(path))
            resp = await _post(client, {"action": "accept", "key": again["key"], "id": again["id"]})
            assert resp.status == 200
            assert (await resp.json())["already"] is True
        finally:
            await client.close()

    asyncio.run(scenario())
    assert len(recorded) == 1


def test_a_waiver_that_cannot_be_audited_is_not_granted(tmp_path, monkeypatch):
    """Ledger first, memo second, fail closed: an unrecorded bypass must not
    exist, and the card stays so the user can click again."""
    monkeypatch.setattr(consent_routes, "record_waiver", lambda **_kwargs: None)
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            assert resp.status == 503
            assert "could not be recorded" in (await resp.json())["detail"]
        finally:
            await client.close()

    asyncio.run(scenario())
    assert consent_store.read()["consents"] == {}
    assert consent_pending.pending_count() == 1


def test_a_memo_that_cannot_be_saved_is_a_503(tmp_path, monkeypatch):
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder([]))
    monkeypatch.setattr(consent_store, "grant", lambda *a, **k: (_ for _ in ()).throw(
        consent_store.ConsentStoreError("read-only file system")))
    card = consent_pending.register_pending(_descriptor(_checkpoint(tmp_path)))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            assert resp.status == 503
            assert "could not be saved" in (await resp.json())["detail"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_dismiss_is_not_consent(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    card = consent_pending.register_pending(_descriptor(_checkpoint(tmp_path)))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await _post(client, {"action": "dismiss", "key": card["key"], "id": card["id"]})
            assert resp.status == 200
            assert (await resp.json())["dismissed"] is True
        finally:
            await client.close()

    asyncio.run(scenario())
    assert recorded == []
    assert consent_store.read()["consents"] == {}
    assert consent_pending.pending_count() == 0


def test_revoke_removes_the_authorization_and_audits_it(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            listing = await (await client.get("/dgxm/consents")).json()
            assert [row["style"] for row in listing["active"]] == ["capacity"]
            assert listing["active"][0]["consent_source"] == "panel"

            resp = await _post(client, {"action": "revoke", "key": card["key"], "id": "c-x"})
            assert resp.status == 200
            assert (await resp.json()) == {"ok": True, "already": False}
            resp = await _post(client, {"action": "revoke", "key": card["key"], "id": "c-x"})
            assert (await resp.json())["already"] is True
        finally:
            await client.close()

    asyncio.run(scenario())
    assert [row["action"] for row in recorded] == ["grant", "revoke"]
    assert recorded[1]["consent_id"] == recorded[0]["consent_id"]
    assert consent_pending.resolve(kind="rescue-slab", path=path, context=CONTEXT) is None


def test_the_auto_rescue_toggle_clears_only_auto_eligible_cards(tmp_path, monkeypatch):
    recorded: list = []
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder(recorded))
    path = _checkpoint(tmp_path)
    consent_pending.register_pending(_descriptor(path))
    accuracy = consent_pending.ConsentDescriptor(
        kind="waive-known-wrong:ring-pad", path=path,
        memo_context=consent_pending.consent_context(
            "waive-known-wrong:ring-pad", combo_key=COMBO, topology="ring2", world=2),
        combo_key=COMBO, artifacts=ARTIFACTS)
    with consent_pending._PENDING_LOCK:
        consent_pending._PENDING["k-card"] = consent_pending.PendingConsent(
            id="p-accuracy", key="k-card", descriptor=consent_pending.describe(accuracy))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await _post(client, {"action": "auto_rescue", "value": True})
            assert resp.status == 200
            body = await resp.json()
            assert body == {"ok": True, "auto_rescue": True, "cleared": 1}

            resp = await client.get("/dgxm/consents")
            payload = await resp.json()
            assert payload["auto_rescue"] is True
            assert [card["kind"] for card in payload["pending"]] == ["waive-known-wrong:ring-pad"]

            resp = await _post(client, {"action": "auto_rescue", "value": "yes"})
            assert resp.status == 400
        finally:
            await client.close()

    asyncio.run(scenario())
    assert consent_store.auto_rescue() is True
    # A standing authorization is itself a bypass decision, so it is audited
    # once; every rescue it later covers still writes its own row.
    assert [row["kind"] for row in recorded] == [consent_routes.STANDING_KIND]
    assert recorded[0]["action"] == "grant" and recorded[0]["target_guard"] is None


def test_disabling_the_toggle_keeps_existing_memos(tmp_path, monkeypatch):
    monkeypatch.setattr(consent_routes, "record_waiver", _fake_recorder([]))
    path = _checkpoint(tmp_path)
    card = consent_pending.register_pending(_descriptor(path))

    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            await _post(client, {"action": "accept", "key": card["key"], "id": card["id"]})
            await _post(client, {"action": "auto_rescue", "value": True})
            resp = await _post(client, {"action": "auto_rescue", "value": False})
            assert (await resp.json())["auto_rescue"] is False
        finally:
            await client.close()

    asyncio.run(scenario())
    assert consent_pending.resolve(kind="rescue-slab", path=path, context=CONTEXT) is not None


def test_malformed_oversized_and_flooding_requests_are_refused(monkeypatch):
    async def scenario():
        client = await _build_client(monkeypatch)
        try:
            resp = await client.post("/dgxm/consent", headers=_headers(client), data="not json")
            assert resp.status == 400

            resp = await client.post("/dgxm/consent", headers=_headers(client), data="[1,2]")
            assert resp.status == 400

            resp = await _post(client, {"action": "fly-away", "key": "k", "id": "p"})
            assert resp.status == 400

            resp = await _post(client, {"action": "accept", "id": "p-1"})
            assert resp.status == 400

            big = "x" * (consent_routes.MAX_BODY_BYTES + 64)
            resp = await client.post("/dgxm/consent", headers=_headers(client),
                                     data=json.dumps({"action": "dismiss", "key": big, "id": "p"}))
            assert resp.status == 413

            statuses = []
            for _ in range(12):
                resp = await _post(client, {"action": "revoke", "key": "k" * 32})
                statuses.append(resp.status)
            assert 429 in statuses
            assert statuses.index(429) >= consent_routes.RATE_LIMIT_N - 5
        finally:
            await client.close()

    asyncio.run(scenario())


def test_the_recycle_boundary_is_unchanged_by_the_generalized_helper():
    from dgx_monarch.nodes.routes import _action_request_allowed, _recycle_request_allowed

    headers = {"Host": "127.0.0.1:8188", "Origin": "http://127.0.0.1:8188",
               "Sec-Fetch-Site": "same-origin", "X-DGXM-Action": "recycle"}
    assert _recycle_request_allowed(headers, "http") == (True, "")
    assert _action_request_allowed(headers, "http", "consent")[0] is False
    consent = {**headers, "X-DGXM-Action": "consent"}
    assert _recycle_request_allowed(consent, "http")[0] is False
    assert _action_request_allowed(consent, "http", "consent") == (True, "")


def test_the_panel_gate_chips_never_count_a_waiver_as_a_verdict(tmp_path, monkeypatch):
    """The summary feeds the panel chips and `dgxm top`. A waiver counted there
    reads as a lost gate."""
    from dgx_monarch.gate_ledger import GateLedger
    from dgx_monarch.nodes import routes as routes_mod

    fake_folder = types.ModuleType("folder_paths")
    fake_folder.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder)

    ledger = GateLedger(str(tmp_path))
    ledger.record(COMBO, "a" * 64, "commit0", "PASS", {"model": "m.safetensors"}, {"world": 1})
    assert consent_routes.record_waiver(**_seam_fields()) is not None

    summary = routes_mod._ledger_summary()
    assert summary["combinations"] == {"PASS": 1}
    assert summary["waivers"] == 1
    assert len(summary["latest"]) == 1


def test_status_node_projects_real_public_ledger_rows(tmp_path, monkeypatch):
    from dgx_monarch.gate_ledger import GateLedger
    from dgx_monarch.nodes import ops as ops_mod

    ledger = GateLedger(str(tmp_path))
    ledger.record(
        COMBO,
        "a" * 64,
        "commit0",
        "PASS",
        {"model": "m.safetensors"},
        {"world": 1},
    )

    class Handle:
        @staticmethod
        def call_all(endpoint, timeout_s):
            assert (endpoint, timeout_s) == ("status", 60)
            return [{"rank": 0, "healthy": True}]

    handle = Handle()
    monkeypatch.setattr(ops_mod, "ensure_live", lambda candidate: candidate)
    result = ops_mod.DGXMonarchStatus().status(
        types.SimpleNamespace(handle=handle)
    )
    payload = json.loads(result["result"][0])

    assert payload["workers"] == [{"rank": 0, "healthy": True}]
    assert payload["identity_gates"] == {
        "combinations": {"PASS": 1},
        "ledger": "output/dgxm_gate_ledger.jsonl",
    }


def _seam_fields(**overrides):
    fields = {
        "action": "grant", "kind": "rescue-slab", "target_guard": "stock_load_preflight",
        "consent_id": "a" * 32, "consent_source": "panel",
        "reason": "because it does not fit", "combo_key": COMBO, "artifacts": ARTIFACTS,
        "memo_context": dict(CONTEXT), "capability_context": {"world": 2},
        "unet_name": "m.safetensors", "file_identity": "1:2:3:4:5", "loras": 0,
    }
    fields.update(overrides)
    return fields


def test_the_ledger_seam_writes_a_real_v9_waiver_row(tmp_path, monkeypatch):
    """End to end against the ledger's own audit-row builder: a signature or
    vocabulary drift fails here rather than at a user's click."""
    fake_folder = types.ModuleType("folder_paths")
    fake_folder.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder)

    key = consent_routes.record_waiver(**_seam_fields())
    assert key == f"audit:waiver:rescue-slab:{COMBO}"

    rows = [json.loads(line) for line in
            open(os.path.join(str(tmp_path), "dgxm_gate_ledger.jsonl"))]
    assert len(rows) == 1
    row = rows[0]
    assert row["key"] == key and row["verdict"] == "WAIVER"
    assert row["artifacts"] == ARTIFACTS
    assert row["gate_protocol"] >= 9
    assert row["waiver_kind"] == "rescue-slab" and row["waiver_class"] == "C"
    assert row["action"] == "grant" and row["consent_source"] == "panel"
    assert row["consent_id"] == "a" * 32
    assert json.loads(row["capability_context"])["record"] == "waiver"

    # and the audit row is invisible to the verdict counters
    from dgx_monarch.gate_audit import trust_rows

    assert trust_rows(rows) == []


def test_the_ledger_seam_fails_closed(tmp_path, monkeypatch):
    fake_folder = types.ModuleType("folder_paths")
    fake_folder.get_output_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fake_folder)

    # a malformed field (an unknown guard for this kind) records nothing
    assert consent_routes.record_waiver(**_seam_fields(target_guard="not_a_guard")) is None
    assert not os.path.exists(os.path.join(str(tmp_path), "dgxm_gate_ledger.jsonl"))

    # an unwritable ledger refuses the grant instead of granting it silently
    from dgx_monarch import gate_audit

    def refuse(*_a, **_k):
        from dgx_monarch.gate_audit_vocab import WaiverNotAuditedError

        raise WaiverNotAuditedError("output directory is not writable")

    monkeypatch.setattr(gate_audit, "record_waiver_grant", refuse)
    assert consent_routes.record_waiver(**_seam_fields()) is None

    monkeypatch.setattr(gate_audit, "record_waiver", lambda *a, **k: False)
    assert consent_routes.record_waiver(**_seam_fields(action="revoke", revoked_by="user")) is None

    # and a driver that cannot import a module the ledger write needs (folder_paths here) refuses too
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    assert consent_routes.record_waiver(**_seam_fields()) is None
