"""Real-JS unit check for the consent panel's row model.

web/js/dgx_monarch_consent_model.js decides which consent rows are shown; the
server computes every card in full. This runs that module under `node` when it
is available and skips otherwise.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODEL = Path(__file__).parents[1] / "web" / "js" / "dgx_monarch_consent_model.js"

CARD = {
    "id": "p-9c2a71f0b4d3",
    "key": "8b1f4c" + "0" * 26,
    "kind": "rescue-slab",
    "class": "C",
    "style": "capacity",
    "title": "Load a_checkpoint.safetensors with slab residency",
    "artifact": "a_checkpoint.safetensors",
    "risk": "Stock residency cannot fit this checkpoint on this box.",
    "evidence": "Every byte is verified against the checkpoint while it loads.",
    "numbers": "needs 61.7 GiB, 44.2 GiB available",
    "measured": None,
    "primary": {"label": "Load with slab residency", "action": "accept"},
    "dismissible": True,
    "auto_eligible": True,
    "occurrences": 2,
    "seen_at": 1754305327.4,
}


def _run_model(fixtures: list[dict]) -> list[dict]:
    node = shutil.which("node")
    if node is None:  # pragma: no cover - environment dependent
        pytest.skip("node not installed; JS model check skipped")
    driver = (
        f"import {{ consentPanelModel }} from {json.dumps(MODEL.as_uri())};\n"
        f"const fixtures = JSON.parse({json.dumps(json.dumps(fixtures))});\n"
        "const out = fixtures.map((f) => consentPanelModel(f.state, f.ui));\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", driver],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode != 0:  # pragma: no cover - surfaced only on a real break
        raise AssertionError(f"node driver failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def test_rows_badge_and_active_ordering():
    empty, one_card, accepted, active = _run_model([
        {"state": {}, "ui": {"now": 1000}},
        {"state": {"payload": {"pending": [CARD], "active": [], "auto_rescue": False}},
         "ui": {"now": 1000}},
        # The pending entry is gone from the payload (the server dropped it on
        # the grant) and the confirmation must still be on screen.
        {"state": {"payload": {"pending": [], "active": [], "auto_rescue": True}},
         "ui": {"now": 5000,
                "justAccepted": {CARD["key"]: {"card": CARD, "at": 4000, "message": "Allowed"}}}},
        {"state": {"payload": {"pending": [], "auto_rescue": False, "active": [
            {"key": "k-stale", "id": "c-1", "kind": "rescue-slab", "artifact": "old.safetensors",
             "granted_at": "2026-08-04 13:22:07", "granted_by": "panel", "stale": True},
            {"key": "k-fresh", "id": "c-2", "kind": "rescue-slab", "artifact": "new.safetensors",
             "granted_at": "2026-08-04 13:24:07", "granted_by": "auto", "stale": False},
        ]}}, "ui": {"now": 1000}},
    ])

    assert [row["type"] for row in empty["rows"]] == ["empty"]
    assert empty["badge"] == 0
    assert empty["activeCount"] == 0

    assert [row["type"] for row in one_card["rows"]] == ["card"]
    assert one_card["badge"] == 1
    assert one_card["rows"][0]["card"]["primary"]["label"] == "Load with slab residency"

    assert [row["type"] for row in accepted["rows"]] == ["accepted"]
    assert accepted["badge"] == 0
    assert accepted["autoRescue"] is True

    assert [row["key"] for row in active["active"]] == ["k-fresh", "k-stale"]
    assert [row["stale"] for row in active["active"]] == [False, True]
    assert active["active"][1]["grantedBy"] == "panel"


def test_confirmation_expires_and_a_failed_poll_keeps_the_card_clickable():
    expired, errored = _run_model([
        {"state": {"payload": {"pending": [], "active": []}},
         "ui": {"now": 999_000,
                "justAccepted": {CARD["key"]: {"card": CARD, "at": 1000, "message": "Allowed"}}}},
        {"state": {"payload": {"pending": [CARD], "active": []}, "error": "HTTP 503"},
         "ui": {"now": 1000}},
    ])
    assert [row["type"] for row in expired["rows"]] == ["empty"]
    assert [row["type"] for row in errored["rows"]] == ["card", "error"]
    assert errored["badge"] == 1


def test_a_malformed_card_or_active_row_is_dropped_not_rendered():
    (only_good,) = _run_model([
        {"state": {"payload": {"pending": [
            {"id": "p-1", "key": "k-1"},                  # no primary
            {"key": "k-2", "primary": {"label": "x"}},    # no id
            CARD,
        ], "active": [{"artifact": "no-key.safetensors"}]}}, "ui": {"now": 1000}},
    ])
    assert only_good["badge"] == 1
    assert only_good["rows"][0]["card"]["id"] == CARD["id"]
    assert only_good["active"] == []


def test_an_active_class_k_row_keeps_the_accuracy_style_without_a_style_field():
    # consent_store.list_active() emits `class`, not `style`: an accuracy
    # waiver must still read as one in the active list.
    (rows,) = _run_model([
        {"state": {"payload": {"pending": [], "active": [
            {"key": "k-k", "id": "c-9", "kind": "waive-known-wrong:ring-pad",
             "class": "K", "artifact": "h3.safetensors", "granted_at": "2026-08-04 13:22:07",
             "consent_source": "panel", "stale": False},
        ]}}, "ui": {"now": 1000}},
    ])
    assert rows["active"][0]["style"] == "accuracy"
    assert rows["active"][0]["grantedBy"] == "panel"


def test_the_env_fallback_reports_the_effective_toggle_state():
    (forced,) = _run_model([
        {"state": {"payload": {"pending": [], "active": [],
                               "auto_rescue": False, "auto_rescue_env": True}},
         "ui": {"now": 1000}},
    ])
    assert forced["autoRescue"] is True
    assert forced["autoRescueEnv"] is True


def test_a_busy_card_reports_its_local_progress_label():
    (busy,) = _run_model([
        {"state": {"payload": {"pending": [CARD], "active": []}},
         "ui": {"now": 1000, "busy": {CARD["key"]: {"busy": True, "label": "allowing"}}}},
    ])
    assert busy["rows"][0]["busy"] is True
    assert busy["rows"][0]["label"] == "allowing"


def test_proto_named_card_is_an_ordinary_own_busy_key():
    proto_card = {**CARD, "id": "p-proto", "key": "__proto__"}
    (busy,) = _run_model([
        {"state": {"payload": {"pending": [proto_card], "active": []}},
         "ui": {"now": 1000,
                "busy": {"__proto__": {"busy": True, "label": "allowing"}}}},
    ])
    assert busy["rows"][0]["key"] == "__proto__"
    assert busy["rows"][0]["busy"] is True
    assert busy["rows"][0]["label"] == "allowing"


def test_two_accuracy_waivers_on_one_checkpoint_are_told_apart_by_their_context():
    """The Revoke button needs a row an operator can identify.

    Both rows read the same kind, class and artifact; only the narrow context
    the memo is keyed on (resolved topology and world) differs.
    """
    (rows,) = _run_model([
        {"state": {"payload": {"pending": [], "active": [
            {"key": "k-1", "id": "c-1", "kind": "waive-known-wrong:ring-pad",
             "class": "K", "artifact": "h3.safetensors", "consent_source": "panel",
             "context": {"topology": "ring2", "world": "2"}, "stale": False},
            {"key": "k-2", "id": "c-2", "kind": "waive-known-wrong:ring-pad",
             "class": "K", "artifact": "h3.safetensors", "consent_source": "panel",
             "context": {"topology": "uly2+ring2", "world": "4"}, "stale": False},
        ]}}, "ui": {"now": 1000}},
    ])
    assert [row["context"] for row in rows["active"]] == [
        "topology ring2, world 2", "topology uly2+ring2, world 4"]
    assert rows["accuracyActive"] == 2


def test_a_live_accuracy_waiver_is_counted_so_the_panel_can_stop_whispering():
    """The panel counts live authorizations to render wrong math so it can show them.

    A stale row (checkpoint replaced since the grant) authorizes nothing, so it
    does not count: the memo is keyed on file identity and misses.
    """
    (none, one, gone) = _run_model([
        {"state": {"payload": {"pending": [], "active": [
            {"key": "k-c", "id": "c-3", "kind": "rescue-slab", "class": "C",
             "artifact": "h3.safetensors", "stale": False}]}}, "ui": {"now": 1000}},
        {"state": {"payload": {"pending": [], "active": [
            {"key": "k-k", "id": "c-4", "kind": "waive-known-wrong:pixeldit-sp",
             "class": "K", "artifact": "pid.safetensors", "stale": False}]}},
         "ui": {"now": 1000}},
        {"state": {"payload": {"pending": [], "active": [
            {"key": "k-k", "id": "c-4", "kind": "waive-known-wrong:pixeldit-sp",
             "class": "K", "artifact": "pid.safetensors", "stale": True}]}},
         "ui": {"now": 1000}},
    ])
    assert none["accuracyActive"] == 0
    assert one["accuracyActive"] == 1
    assert gone["accuracyActive"] == 0
