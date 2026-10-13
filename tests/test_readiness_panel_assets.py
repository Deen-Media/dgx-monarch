"""Source contract for passive readiness and the fixed mesh reset action."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).parents[1]
PANEL = REPO / "web/js/dgx_monarch_panel.js"
MODEL = REPO / "web/js/dgx_monarch_readiness_model.js"


def test_readiness_model_is_pure_and_panel_uses_text_nodes():
    model = MODEL.read_text()
    panel = PANEL.read_text()
    assert re.search(r"^\s*import\s", model, re.M) is None
    for capability in ("document", "window", "fetchApi", "/dgxm/"):
        assert capability not in model
    assert "innerHTML" not in model
    assert "innerHTML" not in panel
    assert 'from "./dgx_monarch_readiness_model.js"' in panel
    assert "readinessPanelModel(telemetry)" in panel
    assert 'el("span", "opacity:.75;", row.detail)' in panel
    assert "readinessCard(null)" in panel
    assert "await refreshOnce(root, zones)" in panel
    assert "replaceWithUnknown(root, zones)" in panel
    assert "current readiness is unknown" in panel


def test_readiness_never_turns_server_actions_into_browser_controls():
    model = MODEL.read_text()
    assert "report.actions" not in model
    assert "readiness.actions" not in model
    assert "button(" not in model
    assert "onclick" not in model
    assert "fetch" not in model.lower()


def test_mesh_reset_names_the_lifecycle_boundary_and_keeps_the_fixed_post():
    panel = PANEL.read_text()
    assert "Reset attached mesh" in panel
    assert "Recycle workers" not in panel
    assert "Persistent DGX Monarch worker services stay running and " in panel
    assert "Stops only the client-owned attached mesh" in panel
    assert 'api.fetchApi("/dgxm/recycle"' in panel
    assert 'headers: { "X-DGXM-Action": "recycle" }' in panel
    assert "window.confirm" in panel
    assert "refresh(root)" in panel
    assert "mesh outcome unknown; inspect" in panel
    assert 'body.status === "no_live_mesh"' in panel
    assert "mesh response unknown; inspect" in panel
    assert "body.detail" not in panel


def test_browser_actions_are_bounded_and_recycle_outcome_remains_visible():
    panel = PANEL.read_text()
    free_request = panel[panel.index('api.fetchApi("/free"'):
                         panel.index('api.fetchApi("/free"') + 500]
    assert "signal: AbortSignal.timeout(30000)" in free_request
    assert 'body.status === "rate_limited"' in panel
    assert "reset already active; wait" in panel
    recycle_handler = panel[panel.index("recycleBtn.onclick = async () =>"):
                            panel.index("actions.appendChild(recycleBtn)")]
    assert "actionState.recycle.busy = false" in recycle_handler
    assert "recycleBtn.disabled = false" in recycle_handler
    assert "setTimeout" not in recycle_handler


def test_panel_labels_comfyui_as_telemetry_not_a_lifecycle_service():
    panel = PANEL.read_text()
    assert "ComfyUI telemetry" in panel


FABRIC = REPO / "web/js/dgx_monarch_fabric_model.js"
UI = REPO / "web/js/dgx_monarch_ui.js"


def test_fabric_model_is_pure_and_the_panel_feeds_it_one_monotonic_clock():
    model = FABRIC.read_text()
    code = re.sub(r"//.*", "", model)
    panel = PANEL.read_text()
    assert re.search(r"^\s*import\s", model, re.M) is None
    for capability in ("document", "window", "fetch", "/dgxm/", "performance",
                       "Date.now", "innerHTML", "setInterval"):
        assert capability not in code
    assert "\u2014" not in model and "\u2013" not in model  # no em or en dash
    assert 'from "./dgx_monarch_fabric_model.js"' in panel
    # One tracker for the page's lifetime, so rates span polls. Each poll
    # reads every box once on the monotonic clock, never the wall clock, and
    # hands each card its box's rates: two cards for one box must not time
    # one reading against the other.
    assert panel.count("createRailRateTracker()") == 1
    refresh = panel[panel.index("async function refreshOnce"):panel.index("let decorateTimer")]
    card = panel[panel.index("function hostCard("):panel.index("function timelineRow(")]
    assert "createRailRateTracker" not in refresh + card
    assert "railRates.update(" not in panel
    assert refresh.count("performance.now()") == 1
    assert "performance.now()" not in card
    assert "Date.now()" not in refresh + card
    assert "hostRailRates(railRates, cards, performance.now())" in refresh
    assert "hostCard(name, stats, meta, rates.get(stats.host))" in refresh
    assert "railRows(rates, stats.rail_state)" in card
    assert refresh.count("hostCard(") == 1
    assert "activeTopology(t.workers)" in refresh
    assert "lastRenderFacts(last)" in refresh


def test_the_host_card_draws_rail_text_through_text_nodes():
    panel = PANEL.read_text()
    card = panel[panel.index("function hostCard("):panel.index("function timelineRow(")]
    assert "for (const part of railParts)" in card
    assert 'el("span", "color:#f87171;", part.text)' in card
    assert "document.createTextNode(part.text)" in card


def test_the_shared_ui_helper_stays_free_of_panel_logic():
    # Its header forbids module-level DOM work; the new lines live in the
    # model file and the panel.
    ui = UI.read_text()
    assert "rail" not in ui.lower()
    assert "topology" not in ui.lower()
