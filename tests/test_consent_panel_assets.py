"""Source-level contract for the consent panel's browser assets.

The repo ships no JS test runner, so the panel is pinned the way the rest of
web/js already is: by reading the source from Python. These assertions hold the
one-click rule: a card with a second primary button, a confirm dialog in front
of a consent, or a poller that only runs while the sidebar tab is open fails here.
"""
from __future__ import annotations

import re
import socket
from pathlib import Path

REPO = Path(__file__).parents[1]
WEB = REPO / "web" / "js"
CONSENT = WEB / "dgx_monarch_consent.js"
MODEL = WEB / "dgx_monarch_consent_model.js"
PANEL = WEB / "dgx_monarch_panel.js"


def test_consent_assets_exist_and_are_declared_to_the_entrypoint_canary():
    assert CONSENT.is_file()
    assert MODEL.is_file()
    canary = (REPO / "tests/canary/comfy_entrypoint_canary.py").read_text()
    assert '"dgx_monarch_consent.js"' in canary
    assert '"dgx_monarch_consent_model.js"' in canary


def test_consent_zone_avoids_html_interpolation_and_reuses_the_panel_vocabulary():
    source = CONSENT.read_text()
    assert "innerHTML" not in source
    assert 'from "./dgx_monarch_consent_model.js"' in source
    assert '"X-DGXM-Action": "consent"' in source
    assert "AbortSignal.timeout(" in source
    assert "consentBusy" in source
    assert "typeof app.queuePrompt" in source


def test_consent_browser_maps_do_not_treat_proto_as_an_inherited_entry():
    source = CONSENT.read_text()
    model = MODEL.read_text()
    assert "const justAccepted = Object.create(null)" in source
    assert "const busyState = Object.create(null)" in source
    assert "const served = Object.create(null)" in source
    assert "Object.hasOwn(busyState, key)" in source
    assert "Object.hasOwn(object, key)" in model


def test_consent_zone_never_puts_a_dialog_in_front_of_one_click():
    # Recycle keeps its window.confirm: it is destructive and expensive.
    # A consent must be one click, and the card text is the confirmation.
    assert "window.confirm" not in CONSENT.read_text()


def test_card_has_exactly_one_primary_action():
    source = CONSENT.read_text()
    assert "card.primary.label" in source
    assert source.count('"accept"') == 1
    assert "secondary" not in source


def test_card_styling_is_server_driven_so_pr_d_adds_zero_javascript():
    source = CONSENT.read_text()
    assert 'card.style === "accuracy"' in source
    assert "card.measured" in source
    # No consent kind is named in the browser: kinds live in the server registry.
    for kind in ("rescue-slab", "waive-known-wrong", "waive-first-load-stock"):
        assert kind not in source


def test_consent_poller_is_app_lifetime_and_the_badge_is_written_by_the_decorator():
    consent = CONSENT.read_text()
    panel = PANEL.read_text()
    assert "app.registerExtension" in consent
    assert "setInterval(refreshConsents, POLL_MS)" in consent
    # The panel's destroy() clears only the telemetry timer, so the consent
    # poll must never be created inside render(container).
    render_body = panel[panel.index("render: (container)"):panel.index("destroy: ()")]
    assert "startConsentPolling" not in render_body
    assert "startConsentPolling();" in panel
    assert 'from "./dgx_monarch_consent.js"' in panel
    assert "consentPendingCount()" in panel
    assert "data-dgxm-badge" in panel
    assert "dgxmBadge" in panel


def test_panel_keeps_the_consent_zone_when_telemetry_fails():
    panel = PANEL.read_text()
    tail = panel[panel.index("telemetry unreachable"):][:400]
    assert "zones.pending" in tail
    assert "zones.active" in tail


def test_model_module_is_dom_free_and_import_free():
    source = MODEL.read_text()
    assert re.search(r"^\s*import\s", source, re.M) is None
    assert 'from "' not in source
    assert "document" not in source
    assert "window" not in source
    assert "export function consentPanelModel" in source


def test_new_browser_assets_carry_no_em_dashes_and_no_rig_identifiers():
    # Match shapes, never this rig's literals: a guard that names the box it
    # was written on leaks the identifier it looks for.
    pattern = re.compile(
        r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"          # RFC1918 / loopback
        r"|\b192\.168\.\d{1,3}\.\d{1,3}\b"
        r"|\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b"
        r"|\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"  # CGNAT
        r"|" + re.escape(socket.gethostname().split(".")[0]),
        re.IGNORECASE)
    for path in (CONSENT, MODEL):
        source = path.read_text()
        assert "\u2014" not in source  # em dash
        assert "\u2013" not in source  # en dash
        assert not pattern.search(source), f"rig identifier in {path.name}"


def test_the_pre_card_surfaces_tell_an_accuracy_waiver_from_a_capacity_rescue():
    """The toast and the section header are what a user meets before the card.

    "needs one click" fits a capacity rescue and misleads for known-wrong math,
    which refuses by default. The server already computes `style` on every card
    and every active row, so both surfaces branch on data it already sends.
    """
    source = CONSENT.read_text()
    toast = source[source.index("function announce()"):]
    assert 'card.style === "accuracy"' in toast
    assert "refused known-wrong math" in toast
    assert "card.measured" in toast
    assert 'severity: accuracy ? "error" : "warn"' in toast
    # The permanent authorization must not be quieter than the transient card:
    # a live accuracy waiver opens the active list and tints it.
    assert "model.accuracyActive" in source
    assert "activeZone.open = true" in source


def test_an_accepted_accuracy_waiver_stays_a_red_warning_not_a_green_allowance():
    source = CONSENT.read_text()
    accepted = source[source.index("function acceptedNode(row)"):
                      source.index("function activeRowNode(row)")]
    assert "Accuracy waiver recorded:" in source
    assert 'row.card?.style === "accuracy"' in accepted
    assert "const accent = accuracy ? ACCURACY : GRANTED" in accepted
    assert "accuracy waiver; output remains rendered under known-wrong math" in accepted


def test_an_active_row_names_the_narrow_context_it_was_granted_for():
    """Two waivers on one checkpoint differ only by topology and world.

    docs/TROUBLESHOOTING.md #55 tells the operator to press Revoke on "the
    row", so a row that cannot be told from its neighbour may revoke the wrong
    grant.
    """
    assert "row.context" in CONSENT.read_text()
    model = MODEL.read_text()
    assert "function contextLabel(" in model
    assert "context: contextLabel(row.context)" in model
