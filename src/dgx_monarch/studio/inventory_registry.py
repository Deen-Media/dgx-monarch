"""Closed registries that give every inventory row its Studio view.

Discovery finds every panel handler and TUI binding in source. Each one must
appear here, mapped to the Studio operation or route that does the same job, or
waived with the reason Studio does not represent it. ``apply_studio_views``
refuses a discovered surface with no entry and an entry with no surface, as
``management_parity.validate_bindings`` does for CLI rows. CLI rows take their
view from that validated binding. Every other row kind has one rule here, so a
new kind also fails until someone reviews it; the only per-row exceptions are
the ``[studio]`` launch keys in ``CONFIGURATION``.

Operation names and route paths are read from source without importing the
HTTP layer, so building the inventory never needs aiohttp.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

STATUSES = frozenset({"operation", "route", "equivalent", "not_represented", "source_only"})


def operation(action: str, equivalence: str) -> dict[str, Any]:
    return {"status": "operation", "operation": action, "equivalence": equivalence}


def route(method: str, path: str, equivalence: str) -> dict[str, Any]:
    return {"status": "route", "method": method, "path": path, "equivalence": equivalence}


def equivalent(equivalence: str) -> dict[str, Any]:
    return {"status": "equivalent", "equivalence": equivalence}


def waiver(reason: str) -> dict[str, Any]:
    return {"status": "not_represented", "reason": reason}


# Studio's metric chart (studio/src/components/TelemetryCharts.svelte) keeps
# taking samples, and its Inspect sample slider steps through the ones it holds.
_SCRUB = "The metric chart's Inspect sample slider steps through one host and metric's retained samples."

PANEL = {
    "panel.action.dgx_monarch_panel.freeBtn.onclick": operation(
        "driver.free", "Free driver memory sends the same ComfyUI /free request after review and confirmation."),
    "panel.action.dgx_monarch_panel.recycleBtn.onclick": operation(
        "actors.recycle", "Reset attached mesh sends the same /dgxm/recycle request after review and confirmation."),
    "panel.action.dgx_monarch_consent.primary.onclick": operation(
        "consent.apply", "Accepts one pending consent card by its exact key and id."),
    "panel.action.dgx_monarch_consent.again.onclick": route(
        "POST", "/api/v1/jobs/{id}/{action}",
        "The retry action resubmits a Studio job that settled failed or cancelled; the panel queues the graph open in its own tab."),
    "panel.action.dgx_monarch_consent.dismiss.onclick": waiver(
        "Studio has no control to dismiss a pending consent card; dismiss it in the ComfyUI panel."),
    "panel.action.dgx_monarch_consent.revoke.onclick": waiver(
        "Studio has no control to revoke an accepted consent; revoke it in the ComfyUI panel."),
    "panel.action.dgx_monarch_consent.box.onchange": waiver(
        "Studio has no control for the automatic rescue setting (auto_rescue); change it in the ComfyUI panel."),
}

TUI = {
    "tui.action.quit": waiver("Ends the terminal session; a Studio view ends when its browser tab closes."),
    "tui.action.pause": waiver(
        "Pauses the terminal's sample ring; Studio metric charts keep taking samples and have no pause."),
    "tui.action.back": equivalent("Scrubs back through the terminal's sample ring. " + _SCRUB),
    "tui.action.fwd": equivalent("Scrubs forward through the terminal's sample ring. " + _SCRUB),
    "tui.action.anim": waiver("Turns the terminal header animation on or off; Studio follows the browser's reduced-motion setting."),
    "tui.action.theme": waiver("Terminal palette selection is presentation only; Studio uses its shared semantic palette."),
    "tui.action.snapshot": operation(
        "top.record", "Records the next telemetry samples to a reviewed file; the terminal key saves the samples it already holds."),
    "tui.action.ledger": operation("gate.inspect", "Reads the trusted gate verdicts in the selected ledger."),
    "tui.action.workers": route(
        "GET", "/api/v1/snapshot", "The Fleet view shows the Worker and topology sample from the snapshot's telemetry."),
}

REGISTRIES = {"panel_action": ("Panel", PANEL), "tui_action": ("TUI", TUI)}

# Kinds without a per-row registry. A row kind missing from here and from
# REGISTRIES, other than the CLI kinds, fails validation.
KIND_VIEWS = {
    "workflow": route("GET", "/api/v1/workflows", "The bundled workflow catalog serves every shipped example."),
    "telemetry": route("GET", "/api/v1/snapshot", "Fleet view metric charts draw every presentation contract metric from the snapshot."),
    "configuration": {"status": "equivalent", "equivalence":
                      "The owner selects the cluster configuration with --config at companion launch, "
                      "or dgxm studio passes it; Studio does not edit it."},
    "node": {"status": "source_only", "reason":
             "Studio reads node schemas from the connected driver's object_info at run time; no closed registry checks this row."},
    "node_input": {"status": "source_only", "reason":
                   "Studio reads node inputs from the connected driver's object_info at run time; no closed registry checks this row."},
}

# dgxm studio turns these [studio] keys into companion flags before the
# companion starts, so the companion never reads them. They take this view
# instead of the configuration kind's. driver_command keeps the kind's view,
# because the companion reads it from its own --config.
CONFIGURATION = {f"config.studio.{key}": waiver(
    "Read by dgxm studio before the companion starts; the page cannot change how its own companion was launched.")
    for key in ("driver", "port", "driver_service")}

_CLI_KINDS = {"adapter": "operation", "parameter": "operation", "selector": "operation",
              "presentation": "equivalent", "context": "equivalent", "companion": "not_represented"}


def studio_operations(root: Path) -> frozenset[str]:
    """Every operation id Studio's capability list can advertise."""
    from .inventory import _static_symbols
    from .management_fields import FIELDS, FIXED_COMMANDS

    symbols = _static_symbols(root / "src/dgx_monarch/studio/operations.py")
    return frozenset({*FIXED_COMMANDS, *FIELDS, *symbols["DRIVER_ACTIONS"], *symbols["HTTP_ACTIONS"]})


def studio_routes(root: Path) -> frozenset[tuple[str, str]]:
    """(method, path) for every route the companion registers."""
    found = set()
    for node in ast.walk(ast.parse((root / "src/dgx_monarch/studio/server.py").read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr.startswith("add_") and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            found.add((node.func.attr.removeprefix("add_").upper(), node.args[0].value))
    return frozenset(found)


def _cli_view(binding: dict[str, Any]) -> dict[str, Any]:
    status = _CLI_KINDS[binding["kind"]]
    if status == "operation":
        return {"status": status, "operations": list(binding["actions"]), "binding": binding["kind"]}
    if status == "not_represented":
        return waiver(binding["reason"])
    return {"status": status, "equivalence": binding["equivalence"]}


def _check(identity: str, view: dict[str, Any], operations: frozenset[str],
           routes: frozenset[tuple[str, str]]) -> None:
    status = view.get("status")
    if status not in STATUSES:
        raise ValueError(f"Unknown Studio view status: {identity}")
    if status == "operation" and set(view.get("operations") or [view.get("operation")]) - operations:
        raise ValueError(f"Missing Studio operation: {identity}")
    if status == "route" and (view.get("method"), view.get("path")) not in routes:
        raise ValueError(f"Missing Studio route: {identity}")
    if status in {"not_represented", "source_only"} and not str(view.get("reason") or "").strip():
        raise ValueError(f"Studio waiver needs a reason: {identity}")
    if status in {"operation", "route", "equivalent"} and not str(view.get("equivalence") or "").strip() \
            and "binding" not in view:
        raise ValueError(f"Studio mapping needs an equivalence: {identity}")


def apply_studio_views(rows: list[dict[str, Any]], root: Path) -> None:
    """Give every row its reviewed Studio view, or refuse the inventory."""
    for kind, (label, registry) in REGISTRIES.items():
        found = {row["id"] for row in rows if row["kind"] == kind}
        if found != registry.keys():
            raise ValueError(f"{label} coverage changed: missing={sorted(found - registry.keys())}; "
                             f"stale={sorted(registry.keys() - found)}")
    stale = CONFIGURATION.keys() - {row["id"] for row in rows if row["kind"] == "configuration"}
    if stale:
        raise ValueError(f"Configuration view names no row: stale={sorted(stale)}")
    operations, routes = studio_operations(root), studio_routes(root)
    for row in rows:
        kind = row["kind"]
        if kind in REGISTRIES:
            view = dict(REGISTRIES[kind][1][row["id"]])
        elif kind in {"cli_command", "cli_option"}:
            view = _cli_view(row["studio_binding"])
        elif kind == "configuration" and row["id"] in CONFIGURATION:
            view = dict(CONFIGURATION[row["id"]])
        elif kind in KIND_VIEWS:
            view = dict(KIND_VIEWS[kind])
        else:
            raise ValueError(f"No Studio view rule for inventory kind: {kind}")
        _check(row["id"], view, operations, routes)
        row["interfaces"]["studio"] = view
