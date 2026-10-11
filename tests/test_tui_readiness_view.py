"""The terminal readiness block is strict, bounded, and read-only."""
from __future__ import annotations

import pytest

pytest.importorskip("rich")

from dgx_monarch.tui.readiness_view import readiness_text

PALETTE = {"good": "green", "warn": "yellow", "bad": "red"}


def _tick(**changes):
    report = {
        "schema_version": 1,
        "overall": "ready",
        "lifecycle": {
            "worker_service": {"state": "ready", "detail": "Two services are ready."},
            "attached_mesh": {"state": "idle", "detail": "No mesh is attached."},
            "render_session": {"state": "idle", "detail": "No render is active."},
        },
        "actions": [],
    }
    report.update(changes)
    return {"readiness": report}


def test_view_names_the_three_distinct_lifecycle_layers():
    plain = readiness_text(_tick(), PALETTE).plain
    assert "Readiness READY" in plain
    assert "Worker service READY" in plain
    assert "Attached mesh IDLE" in plain
    assert "Render session IDLE" in plain


@pytest.mark.parametrize(
    "readiness",
    [None, {}, {"schema_version": 2}, {
        "schema_version": 1,
        "overall": "ready",
        "lifecycle": {"worker_service": {"state": "ready", "detail": "ok"}},
    }],
)
def test_missing_or_malformed_payload_fails_closed(readiness):
    plain = readiness_text({"readiness": readiness}, PALETTE).plain
    assert "Readiness UNKNOWN" in plain
    assert plain.count("UNKNOWN") == 5
    assert plain.count("missing or malformed") == 3


def test_detail_is_single_line_bounded_and_actions_are_inert():
    tick = _tick(actions=[{
        "title": "Run supplied operation",
        "command": "dangerous-command --token private",
    }])
    tick["readiness"]["lifecycle"]["attached_mesh"]["detail"] = " x\n" + "y" * 500
    plain = readiness_text(tick, PALETTE).plain
    assert "dangerous-command" not in plain
    assert "private" not in plain
    assert "\n x\n" not in plain
    attached = next(line for line in plain.splitlines() if line.startswith(" Attached mesh"))
    assert len(attached) < 450


def test_overall_cannot_claim_healthier_state_than_a_layer():
    tick = _tick()
    tick["readiness"]["lifecycle"]["attached_mesh"] = {
        "state": "blocked", "detail": "Unsafe",
    }
    plain = readiness_text(tick, PALETTE).plain
    assert "Readiness UNKNOWN" in plain
    assert "Unsafe" not in plain


def test_app_labels_the_comfyui_value_as_telemetry():
    from pathlib import Path

    source = (Path(__file__).parents[1] / "src/dgx_monarch/tui/app.py").read_text()
    assert "ComfyUI telemetry" in source
    assert "readiness_text(tick, pal)" in source


def _live_tick(*, mesh_active=False, render_active=False, error=None):
    from dgx_monarch.operator_readiness import readiness_from_telemetry

    return {"readiness": readiness_from_telemetry(
        [{"healthy": True}],
        {"state": "ok", "verdict": "live", "active_leases": int(mesh_active),
         "abandoned_samples": 0},
        {"active": render_active}, error,
    )}


@pytest.mark.parametrize("mesh_active,render_active,activity", [
    (False, False, "IDLE"), (True, False, "BUSY"), (True, True, "RENDERING"),
])
def test_live_view_separates_health_activity_and_maintenance(mesh_active, render_active, activity):
    tick = _live_tick(mesh_active=mesh_active, render_active=render_active)
    plain = readiness_text(tick, PALETTE).plain
    assert "Health READY" in plain
    assert f"Activity {activity}" in plain
    assert "Maintenance " in plain
    assert "DEGRADED" not in plain
    if activity != "IDLE":
        assert "Wait for" in plain


def test_live_unknown_health_does_not_hide_observed_activity():
    plain = readiness_text(_live_tick(render_active=True, error="timeout"), PALETTE).plain
    assert "Health UNKNOWN" in plain
    assert "Activity RENDERING" in plain
    assert "Maintenance Inspect health; wait for the active render to finish." in plain


@pytest.mark.parametrize("changes", [
    {"health": "ready"}, {"activity": "idle"}, {"maintenance": "none"},
    {"schema_version": True}, {"schema_version": 2}, {"command": "unsafe-operation"},
])
def test_inconsistent_presentation_fails_closed(changes):
    tick = _live_tick(render_active=True, error="timeout")
    tick["readiness"]["presentation"].update(changes)
    plain = readiness_text(tick, PALETTE).plain
    assert "Readiness UNKNOWN" in plain
    assert "Health READY" not in plain
    assert "unsafe-operation" not in plain


def test_legacy_active_payload_retains_aggregate_and_explains_it():
    tick = _live_tick(render_active=True)
    del tick["readiness"]["presentation"]
    plain = readiness_text(tick, PALETTE).plain
    assert "Readiness DEGRADED" in plain
    assert "Activity RENDERING" in plain
    assert "Legacy readiness includes active work" in plain
    assert "Maintenance Inspect health; wait for the active render to finish." in plain


def test_invalid_worker_activity_cannot_be_presented_as_healthy_occupancy():
    tick = _live_tick()
    tick["readiness"]["overall"] = "degraded"
    tick["readiness"]["lifecycle"]["worker_service"]["state"] = "active"
    plain = readiness_text(tick, PALETTE).plain
    assert "Readiness UNKNOWN" in plain
    assert "Health READY" not in plain
