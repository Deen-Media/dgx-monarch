"""Real-JS contract for the fail-closed browser readiness projection."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODEL = Path(__file__).parents[1] / "web/js/dgx_monarch_readiness_model.js"


def _canonical(**changes):
    report = {
        "schema_version": 1,
        "overall": "ready",
        "lifecycle": {
            "worker_service": {"state": "ready", "detail": "2 worker services ready."},
            "attached_mesh": {"state": "idle", "detail": "No mesh is attached."},
            "render_session": {"state": "idle", "detail": "No render is active."},
        },
        "actions": [],
    }
    report.update(changes)
    return {"readiness": report}


def _run_model(fixtures):
    node = shutil.which("node")
    if node is None:  # pragma: no cover - environment dependent
        pytest.skip("node not installed; JS model check skipped")
    driver = (
        f"import {{ readinessPanelModel }} from {json.dumps(MODEL.as_uri())};\n"
        f"const fixtures = {json.dumps(fixtures)};\n"
        "process.stdout.write(JSON.stringify(fixtures.map(readinessPanelModel)));\n"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", driver],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode != 0:  # pragma: no cover - only a real JS break
        raise AssertionError(f"node driver failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def test_canonical_payload_projects_only_fixed_lifecycle_rows():
    payload = _canonical()
    payload["readiness"]["actions"] = [{
        "id": "injected", "title": "Do something", "command": "private-operation",
        "endpoint": "/arbitrary",
    }]
    (model,) = _run_model([payload])
    assert model["source"] == "canonical"
    assert model["overall"] == "ready"
    assert [row["label"] for row in model["rows"]] == [
        "Worker service", "Attached mesh", "Render session"]
    assert set(model) == {"schemaVersion", "source", "overall", "notice", "rows"}
    encoded = json.dumps(model)
    assert "private-operation" not in encoded
    assert "/arbitrary" not in encoded
    assert "actions" not in encoded


def test_malformed_or_healthier_than_layers_payload_fails_closed():
    blocked = _canonical(overall="ready")
    blocked["readiness"]["lifecycle"]["attached_mesh"] = {
        "state": "blocked", "detail": "Unsafe",
    }
    missing_detail = _canonical()
    del missing_detail["readiness"]["lifecycle"]["render_session"]["detail"]
    models = _run_model([
        {"readiness": None},
        _canonical(schema_version=2),
        blocked,
        missing_detail,
    ])
    for model in models:
        assert model["source"] == "invalid"
        assert model["overall"] == "unknown"
        assert {row["state"] for row in model["rows"]} == {"unknown"}


def test_canonical_telemetry_error_may_raise_otherwise_ready_layers_to_unknown():
    idle = _canonical(overall="unknown")
    active = _canonical(overall="unknown")
    active["readiness"]["lifecycle"]["render_session"] = {
        "state": "active", "detail": "One render is active.",
    }
    idle_model, active_model = _run_model([idle, active])
    for model in (idle_model, active_model):
        assert model["source"] == "canonical"
        assert model["overall"] == "unknown"
        assert model["notice"] == "Telemetry freshness could not be established."
    assert {row["state"] for row in idle_model["rows"]} == {"ready", "idle"}
    assert active_model["rows"][2]["state"] == "active"


def test_old_server_fallback_is_explicit_and_conservative():
    ready, actor_only, missing, errored = _run_model([
        {
            "workers": [{"healthy": True}, {"loop": "running", "port": "open"}],
            "mesh": {"state": "idle", "verdict": "none", "active_leases": 0,
                     "abandoned_samples": 0},
            "render": {"active": False},
        },
        {
            "workers": [{"rank": 0, "world": 2}, {"rank": 1, "world": 2}],
            "mesh": {"state": "ok", "verdict": "none", "active_leases": 0,
                     "abandoned_samples": 0},
            "render": {"active": False},
        },
        {},
        {
            "workers": [{"healthy": True}],
            "mesh": {"state": "idle", "verdict": "none"},
            "render": {"active": False},
            "telemetry_error": "TimeoutError",
        },
    ])
    assert ready["source"] == "compatibility"
    assert ready["overall"] == "ready"
    assert "older telemetry" in ready["notice"]
    assert actor_only["overall"] == "unknown"
    assert actor_only["rows"][0]["detail"] == "Worker service health could not be confirmed."
    assert missing["overall"] == "unknown"
    assert errored["overall"] == "unknown"


def test_blocked_state_outranks_unrelated_unknown_in_compatibility_view():
    (model,) = _run_model([{
        "workers": [],
        "mesh": {"state": "dirty", "verdict": "blocked"},
        "render": None,
    }])
    assert model["overall"] == "blocked"
    assert model["rows"][1]["state"] == "blocked"


def test_a_poisoned_mesh_reads_blocked_and_not_malformed():
    """`poisoned`, the state that proves no attach can succeed, reads blocked.

    The malformed check runs first, so a state list without `poisoned` would
    report it as malformed.
    """
    (model,) = _run_model([{
        "workers": [{"healthy": True}],
        "mesh": {"state": "poisoned", "verdict": "none", "poisoned": True},
        "render": {"active": False},
    }])
    assert model["rows"][1]["state"] == "blocked"
    assert model["rows"][1]["detail"] == "The attached mesh is not safe to reuse."
    assert model["overall"] == "blocked"


def test_malformed_legacy_safety_fields_never_default_to_ready():
    malformed_mesh, contradictory_render, unknown_verdict, idle_leased, busy_mismatch = _run_model([
        {
            "workers": [{"healthy": True}],
            "mesh": {"state": "ok", "verdict": "none", "replacement_blocked": 7},
            "render": {"active": False},
        },
        {
            "workers": [{"healthy": True}],
            "mesh": {"state": "idle", "verdict": "none"},
            "render": {"active": False, "active_renders": 2},
        },
        {"workers": [{"healthy": True}],
         "mesh": {"state": "ok", "verdict": "nonsense"},
         "render": {"active": False}},
        {"workers": [{"healthy": True}],
         "mesh": {"state": "idle", "verdict": "none", "active_leases": 1},
         "render": {"active": False}},
        {"workers": [{"healthy": True}],
         "mesh": {"state": "ok", "verdict": "unresolved", "busy_phase": "load"},
         "render": {"active": False}},
    ])
    assert malformed_mesh["rows"][1]["state"] == "unknown"
    assert contradictory_render["rows"][2]["state"] == "unknown"
    for model in (malformed_mesh, contradictory_render, unknown_verdict,
                  idle_leased, busy_mismatch):
        assert model["overall"] == "unknown"


def test_coherent_legacy_mesh_states_keep_their_conservative_meaning():
    base = {"workers": [{"healthy": True}], "render": {"active": False}}
    ready, leased, busy, completed = _run_model([
        {**base, "mesh": {"state": "ok", "verdict": "live"}},
        {**base, "mesh": {"state": "ok", "verdict": "live", "active_leases": 1}},
        {**base, "mesh": {"state": "busy", "verdict": "unresolved",
                           "busy_phase": "load"}},
        {**base, "mesh": {"state": "idle", "verdict": "completed",
                           "active_leases": 1}},
    ])
    assert ready["rows"][1]["state"] == "ready"
    assert leased["rows"][1]["state"] == "active"
    assert busy["rows"][1]["state"] == "active"
    assert completed["rows"][1]["state"] == "idle"
    assert [model["overall"] for model in (ready, leased, busy, completed)] == [
        "ready", "degraded", "degraded", "ready"]
