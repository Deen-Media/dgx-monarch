"""Every panel action and TUI binding has a reviewed Studio view or a waiver.

A probe once added a panel handler and a TUI binding and both entered the
inventory with no Studio surface while the same drift on a CLI row was
refused. These tests pin the closed registries that now refuse both.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dgx_monarch.studio import inventory_registry
from dgx_monarch.studio.inventory import build_inventory, load_inventory
from dgx_monarch.studio.inventory_registry import (
    CONFIGURATION,
    KIND_VIEWS,
    PANEL,
    STATUSES,
    TUI,
    apply_studio_views,
    studio_operations,
    studio_routes,
)

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = load_inventory()
# Written out, not read from management_parity, so a drift in the waiver list
# fails here instead of moving with it.
STUDIO_TREE = {
    "cli.command.studio", "cli.command.studio.up", "cli.command.studio.down", "cli.command.studio.open",
    "cli.command.studio.token", "cli.command.studio.tunnel",
    "cli.option.studio.help", "cli.option.studio.config", "cli.option.studio.driver", "cli.option.studio.state_dir",
    "cli.option.studio.allow_management", "cli.option.studio.no_open",
    "cli.option.studio.up.help", "cli.option.studio.up.config", "cli.option.studio.up.allow_management",
    "cli.option.studio.down.help", "cli.option.studio.open.help",
    "cli.option.studio.token.help", "cli.option.studio.token.value", "cli.option.studio.token.rotate",
    "cli.option.studio.tunnel.help", "cli.option.studio.tunnel.config", "cli.option.studio.tunnel.host",
    "cli.option.studio.tunnel.port", "cli.option.studio.tunnel.run",
}
PANEL_JS = ROOT / "web/js/dgx_monarch_panel.js"
TUI_APP = ROOT / "src/dgx_monarch/tui/app.py"


def _patched_source(monkeypatch, target: Path, edit):
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        return edit(text) if Path(path) == target else text

    monkeypatch.setattr(Path, "read_text", read_text)


def test_a_new_panel_handler_without_a_studio_view_fails(monkeypatch):
    _patched_source(monkeypatch, PANEL_JS, lambda text: text + "\npurgeBtn.onclick = () => {};\n")
    with pytest.raises(ValueError, match=r"Panel coverage changed: missing=\['panel\.action\.dgx_monarch_panel\.purgeBtn"):
        build_inventory(ROOT)


def test_a_new_tui_binding_without_a_studio_view_fails(monkeypatch):
    _patched_source(monkeypatch, TUI_APP, lambda text: text.replace(
        'Binding("q", "quit", "quit"),', 'Binding("q", "quit", "quit"), Binding("p", "purge_cache", "purge"),'))
    with pytest.raises(ValueError, match=r"TUI coverage changed: missing=\['tui\.action\.purge_cache'\]"):
        build_inventory(ROOT)


def test_a_removed_binding_leaves_a_stale_entry_that_fails(monkeypatch):
    _patched_source(monkeypatch, TUI_APP, lambda text: text.replace('Binding("w", "workers", "workers"),', ""))
    with pytest.raises(ValueError, match=r"TUI coverage changed: missing=\[\]; stale=\['tui\.action\.workers'\]"):
        build_inventory(ROOT)


@pytest.mark.parametrize(("view", "message"), [
    (inventory_registry.operation("purge.everything", "Not a Studio operation."), "Missing Studio operation"),
    (inventory_registry.route("GET", "/api/v1/purge", "Not a Studio route."), "Missing Studio route"),
    (inventory_registry.waiver(" "), "needs a reason"),
    (inventory_registry.operation("driver.free", ""), "needs an equivalence"),
    ({"status": "typed advanced inventory"}, "Unknown Studio view status"),
])
def test_each_mapping_must_name_a_real_operation_route_or_reason(monkeypatch, view, message):
    monkeypatch.setitem(PANEL, "panel.action.dgx_monarch_panel.freeBtn.onclick", view)
    with pytest.raises(ValueError, match=message):
        build_inventory(ROOT)


def test_a_new_row_kind_needs_its_own_rule():
    rows = load_inventory()["rows"]
    rows.append({"id": "future.row", "kind": "future", "interfaces": {"studio": {}}})
    with pytest.raises(ValueError, match="No Studio view rule for inventory kind: future"):
        apply_studio_views(rows, ROOT)


def test_registries_name_only_operations_and_routes_studio_serves():
    operations, routes = studio_operations(ROOT), studio_routes(ROOT)
    assert {"driver.free", "actors.recycle", "consent.apply", "top.record", "gate.inspect"} <= operations
    for view in [*PANEL.values(), *TUI.values(), *KIND_VIEWS.values(), *CONFIGURATION.values()]:
        assert view["status"] in STATUSES
        if view["status"] == "operation":
            assert view["operation"] in operations
        if view["status"] == "route":
            assert (view["method"], view["path"]) in routes


def test_consent_actions_studio_cannot_drive_are_waived_with_reasons():
    rows = {row["id"]: row for row in INVENTORY["rows"]}
    for handler, words in (("dismiss", "dismiss"), ("revoke", "revoke"), ("box", "auto_rescue")):
        view = rows[f"panel.action.dgx_monarch_consent.{handler}.onchange" if handler == "box"
                    else f"panel.action.dgx_monarch_consent.{handler}.onclick"]["interfaces"]["studio"]
        assert view["status"] == "not_represented"
        assert words in view["reason"] and "ComfyUI panel" in view["reason"]
    accept = rows["panel.action.dgx_monarch_consent.primary.onclick"]["interfaces"]["studio"]
    assert accept == PANEL["panel.action.dgx_monarch_consent.primary.onclick"]


def test_terminal_scrub_keys_map_to_the_chart_slider_and_pause_stays_waived():
    """Studio's metric chart has an Inspect sample slider, so scrubbing is represented; pausing is not."""
    chart = (ROOT / "studio/src/components/TelemetryCharts.svelte").read_text()
    assert "Inspect sample" in chart
    assert 'type="range"' in chart
    for key in ("back", "fwd"):
        view = TUI[f"tui.action.{key}"]
        assert view["status"] == "equivalent" and "Inspect sample slider" in view["equivalence"]
    pause = TUI["tui.action.pause"]
    assert pause["status"] == "not_represented" and "no pause" in pause["reason"]
    assert "scrub" not in pause["reason"]


def test_every_checked_in_row_carries_a_reviewed_view_and_no_unserved_label():
    assert "typed advanced inventory" not in json.dumps(INVENTORY)
    by_kind: dict[str, set[str]] = {}
    for row in INVENTORY["rows"]:
        by_kind.setdefault(row["kind"], set()).add(row["interfaces"]["studio"]["status"])
    assert by_kind["panel_action"] <= {"operation", "route", "not_represented"}
    assert by_kind["tui_action"] <= {"operation", "route", "equivalent", "not_represented"}
    assert by_kind["cli_command"] <= {"operation", "not_represented"}
    assert by_kind["cli_option"] <= {"operation", "equivalent", "not_represented"}
    waived = {row["id"] for row in INVENTORY["rows"] if row["kind"] in {"cli_command", "cli_option"}
              and row["interfaces"]["studio"]["status"] == "not_represented"}
    assert len(STUDIO_TREE) == 25
    assert waived == STUDIO_TREE
    assert {row["id"] for row in INVENTORY["rows"] if row["kind"] == "panel_action"} == PANEL.keys()
    assert {row["id"] for row in INVENTORY["rows"] if row["kind"] == "tui_action"} == TUI.keys()


def test_cli_rows_take_their_view_from_the_validated_binding():
    for row in INVENTORY["rows"]:
        if row["kind"] not in {"cli_command", "cli_option"}:
            continue
        view, binding = row["interfaces"]["studio"], row["studio_binding"]
        if binding["kind"] in {"adapter", "parameter", "selector"}:
            assert view["operations"] == binding["actions"]
        elif binding["kind"] == "companion":
            assert view == {"status": "not_represented", "reason": binding["reason"]}
        else:
            assert view == {"status": "equivalent", "equivalence": binding["equivalence"]}


def test_a_configuration_view_without_its_row_fails(monkeypatch):
    monkeypatch.setitem(CONFIGURATION, "config.studio.host", inventory_registry.waiver("No such key."))
    with pytest.raises(ValueError, match=r"Configuration view names no row: stale=\['config\.studio\.host'\]"):
        build_inventory(ROOT)


def test_context_texts_name_both_ways_to_launch():
    rows = {row["id"]: row["interfaces"]["studio"] for row in INVENTORY["rows"]}
    assert rows["cli.option.global.config"] == {"status": "equivalent", "equivalence": (
        "Owner selects --config at companion launch, or dgxm studio passes the file dgxm itself reads; "
        "each review binds the selected file identity. The companion never discovers another config.")}
    for command in ("diagnostics", "top", "gate"):
        assert rows[f"cli.option.{command}.host"] == {"status": "equivalent", "equivalence": (
            "Owner selects --driver at companion launch, or dgxm studio passes [studio] driver from cluster.toml; "
            "all authenticated requests use that fixed origin. Studio never probes for drivers.")}
    assert rows["config.cluster.client_bind"] == {"status": "equivalent", "equivalence": (
        "The owner selects the cluster configuration with --config at companion launch, or dgxm studio passes it; "
        "Studio does not edit it.")}


def test_building_the_inventory_never_imports_the_http_layer():
    program = ("import sys\nfrom pathlib import Path\nfrom dgx_monarch.studio.inventory import build_inventory\n"
               "build_inventory(Path(sys.argv[1]))\nprint('aiohttp' in sys.modules)\n")
    result = subprocess.run([sys.executable, "-c", program, str(ROOT)], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=120, check=True)
    assert result.stdout.strip() == "False"
