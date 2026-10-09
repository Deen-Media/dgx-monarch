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
    assert plain.count("UNKNOWN") == 4
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
