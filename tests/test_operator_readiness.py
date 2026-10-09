"""Canonical readiness shared by the operator surfaces."""
from __future__ import annotations

import copy
import json

import pytest

from dgx_monarch import operator_readiness as readiness

HEALTHY_WORKERS = [
    {"loop": "running", "port": "open"},
    {"healthy": True, "up": True},
]
IDLE_MESH = {
    "state": "idle",
    "verdict": "none",
    "active_leases": 0,
    "abandoned_samples": 0,
}
IDLE_RENDER = {"active": False, "active_renders": 0}


def _telemetry(**changes):
    values = {
        "workers": copy.deepcopy(HEALTHY_WORKERS),
        "mesh": copy.deepcopy(IDLE_MESH),
        "render": copy.deepcopy(IDLE_RENDER),
        "telemetry_error": None,
    }
    values.update(changes)
    return readiness.readiness_from_telemetry(**values)


def _action_ids(report):
    return [action["id"] for action in report["actions"]]


def test_schema_and_healthy_idle_lifecycle_are_ready():
    report = _telemetry()
    assert report == {
        "schema_version": 1,
        "overall": "ready",
        "lifecycle": {
            "worker_service": {"state": "ready", "detail": "2 worker services ready."},
            "attached_mesh": {
                "state": "idle",
                "detail": "No mesh is attached; the next render can attach one.",
            },
            "render_session": {"state": "idle", "detail": "No render session is active."},
        },
        "actions": [],
    }
    assert tuple(report["lifecycle"]) == readiness.LAYER_NAMES


@pytest.mark.parametrize(
    "workers",
    [None, {}, [], "running", [None], [{"healthy": "yes"}], [{"rank": True, "world": 2}]],
)
def test_missing_or_malformed_workers_are_unknown(workers):
    report = _telemetry(workers=workers)
    assert report["overall"] == "unknown"
    assert report["lifecycle"]["worker_service"]["state"] == "unknown"
    assert "inspect-worker-service" in _action_ids(report)


@pytest.mark.parametrize(
    "worker",
    [
        {"healthy": False},
        {"up": False},
        {"running": False, "listening": True},
        {"loop": "stopped", "port": "open"},
        {"loop": "running", "port": "closed"},
    ],
)
def test_an_explicitly_unhealthy_worker_blocks(worker):
    report = _telemetry(workers=[worker])
    assert report["overall"] == "blocked"
    assert report["lifecycle"]["worker_service"]["state"] == "blocked"


def test_dirty_setup_cleanup_verdict_blocks_without_copying_detail():
    report = _telemetry(workers=[{
        "healthy": True,
        "setup_cleanup_failed": True,
        "setup_cleanup_detail": "/private/path/token-value",
    }])
    assert report["overall"] == "blocked"
    assert report["lifecycle"]["worker_service"] == {
        "state": "blocked",
        "detail": "At least one worker service is not ready.",
    }
    assert "private/path" not in json.dumps(report)


def test_malformed_setup_cleanup_verdict_fails_unknown():
    report = _telemetry(workers=[{
        "healthy": True,
        "setup_cleanup_failed": "false",
    }])
    assert report["overall"] == "unknown"
    assert report["lifecycle"]["worker_service"]["state"] == "unknown"


def test_an_actor_rank_world_response_does_not_claim_worker_service_health():
    report = _telemetry(workers=[{"rank": 0, "world": 2}, {"rank": 1, "world": 2}])
    assert report["overall"] == "unknown"
    assert report["lifecycle"]["worker_service"]["state"] == "unknown"


def test_doctor_config_permission_warning_is_allowlisted_and_reversible():
    report = readiness.readiness_from_doctor_rows([
        {"status": "WARN", "name": "config permissions", "detail": "not copied"}
    ])
    action = next(
        item for item in report["actions"] if item["id"] == "tighten-config-permissions"
    )
    assert action["repairable"] is True
    assert action["reversible"] is True
    assert action["requires_confirmation"] is True
    assert action["command"] == "dgxm doctor --repair"


@pytest.mark.parametrize(
    "mesh",
    [
        {"state": "dirty"},
        {"state": "ok", "verdict": "blocked"},
        {"state": "ok", "replacement_blocked": "stop failed"},
        {"state": "ok", "poisoned": True},
        {"state": "ok", "abandoned_samples": 1},
        {"state": "ok", "abandoned_leases": 2},
        {"state": "ok", "verdict": "unresolved", "busy_phase": ""},
    ],
)
def test_dirty_poisoned_or_abandoned_mesh_is_blocked_and_manual(mesh):
    report = _telemetry(mesh=mesh)
    assert report["overall"] == "blocked"
    assert report["lifecycle"]["attached_mesh"]["state"] == "blocked"
    action = next(a for a in report["actions"] if a["id"] == "recover-attached-mesh")
    assert action["safety"] == "manual"
    assert action["requires_confirmation"] is True
    assert action["repairable"] is False
    assert "command" not in action


def test_blocked_known_state_outranks_unrelated_unknown_state():
    report = readiness.readiness_from_telemetry(None, {"state": "dirty"}, None)
    assert report["overall"] == "blocked"
    assert _action_ids(report)[0] == "recover-attached-mesh"


@pytest.mark.parametrize(
    ("mesh", "render"),
    [
        ({"state": "busy", "verdict": "unresolved", "busy_phase": "load",
          "active_leases": 0}, IDLE_RENDER),
        ({"state": "ok", "verdict": "live", "active_leases": 1}, IDLE_RENDER),
        (IDLE_MESH, {"active": True, "active_renders": 2}),
    ],
)
def test_active_work_is_degraded_not_broken(mesh, render):
    report = _telemetry(mesh=mesh, render=render)
    assert report["overall"] == "degraded"
    assert _action_ids(report) == ["wait-for-active-work"]
    action = report["actions"][0]
    assert action["safety"] == "automatic"
    assert action["requires_confirmation"] is False
    assert "not broken" in action["detail"]


@pytest.mark.parametrize("render", [
    {"active": False, "active_renders": 1},
    {"active": True, "active_renders": 0},
])
def test_contradictory_render_evidence_is_unknown(render):
    report = _telemetry(render=render)
    assert report["overall"] == "unknown"
    assert report["lifecycle"]["render_session"]["state"] == "unknown"


def test_active_render_with_missing_legacy_count_remains_active():
    report = _telemetry(render={"active": True})
    assert report["overall"] == "degraded"
    assert report["lifecycle"]["render_session"]["state"] == "active"


@pytest.mark.parametrize(
    ("field", "value", "layer"),
    [
    ("mesh", None, "attached_mesh"),
    ("mesh", {"state": "surprising"}, "attached_mesh"),
    ("mesh", {"state": "ok", "active_leases": True}, "attached_mesh"),
    ("mesh", {"state": "ok", "verdict": "none", "active_leases": 0}, "attached_mesh"),
    ("mesh", {"state": "idle", "verdict": "none", "active_leases": 1}, "attached_mesh"),
    ("mesh", {"state": "busy", "verdict": "live", "busy_phase": "load"}, "attached_mesh"),
        ("render", None, "render_session"),
        ("render", {"active": "false"}, "render_session"),
        ("render", {"active": False, "active_renders": -1}, "render_session"),
    ],
)
def test_malformed_mesh_or_render_is_unknown(field, value, layer):
    report = _telemetry(**{field: value})
    assert report["overall"] == "unknown"
    assert report["lifecycle"][layer]["state"] == "unknown"


def test_completed_mesh_may_be_idle_with_inert_lease_bookkeeping():
    report = _telemetry(mesh={
        "state": "idle", "verdict": "completed", "active_leases": 2,
        "abandoned_samples": 0,
    })
    assert report["overall"] == "ready"
    assert report["lifecycle"]["attached_mesh"]["state"] == "idle"


def test_telemetry_error_is_not_echoed_and_forces_unknown():
    private_detail = "/home/operator?credential=private-value"
    report = _telemetry(telemetry_error=private_detail)
    encoded = json.dumps(report)
    assert report["overall"] == "unknown"
    assert "restore-telemetry" in _action_ids(report)
    assert private_detail not in encoded
    assert "private-value" not in encoded


def test_foreign_exception_like_values_are_never_stringified():
    class Trap:
        def __str__(self):
            raise AssertionError("must not call str")

        def __repr__(self):
            raise AssertionError("must not call repr")

        def __bool__(self):
            raise AssertionError("must not call bool")

    report = readiness.readiness_from_telemetry(
        [{"status_error": Trap()}],
        {"state": Trap(), "replacement_blocked": Trap()},
        {"active": Trap()},
        Trap(),
    )
    assert report["overall"] == "unknown"
    assert "Trap" not in json.dumps(report)


def test_sanitized_detail_overrides_are_bounded_and_single_line():
    report = _telemetry(sanitized_details={
        "attached_mesh": "  cached\nmesh  ",
        "render_session": "x" * 500,
    })
    assert report["lifecycle"]["attached_mesh"]["detail"] == "cached mesh"
    assert report["lifecycle"]["render_session"]["detail"] == "x" * 400


def test_commands_are_plain_inert_json_data_and_actions_have_the_contract():
    report = readiness.readiness_from_telemetry(
        [{"healthy": False}], {"state": "dirty"}, {"active": None}, "timeout")
    required = {
        "id", "title", "detail", "safety", "repairable", "reversible",
        "requires_confirmation",
    }
    for action in report["actions"]:
        assert required <= set(action)
        assert action["safety"] in readiness.ACTION_SAFETY
        assert type(action["repairable"]) is bool
        assert type(action["reversible"]) is bool
        assert type(action["requires_confirmation"]) is bool
        if "command" in action:
            assert isinstance(action["command"], str)
    json.loads(json.dumps(report))


def test_payload_wrapper_fails_closed_for_non_object_and_missing_blocks():
    for payload in (None, [], {}, {"workers": HEALTHY_WORKERS}):
        report = readiness.readiness_from_telemetry_payload(payload)
        assert report["overall"] == "unknown"
        json.dumps(report)


def test_doctor_rows_are_not_mutated_and_all_green_is_ready():
    rows = [
        {"status": "ok", "name": "left worker loop", "detail": "private details"},
        {"status": "ok", "name": "right worker loop", "detail": "private details"},
        {"status": "ok", "name": "mesh health", "detail":
         "attached mesh ready, 0 active leases"},
        {"status": "ok", "name": "torch CUDA", "detail": "available"},
    ]
    before = copy.deepcopy(rows)
    report = readiness.readiness_from_doctor_rows(rows)
    assert rows == before
    assert report["overall"] == "ready"
    assert report["lifecycle"]["worker_service"]["state"] == "ready"
    assert report["lifecycle"]["attached_mesh"]["state"] == "ready"
    assert report["actions"] == []


def test_doctor_no_driver_compatibility_ok_does_not_claim_mesh_readiness():
    report = readiness.readiness_from_doctor_rows([
        {"status": "ok", "name": "mesh health", "detail":
         "no driver reachable; start ComfyUI and re-run doctor for the mesh view"},
    ])

    # The row stays OK, so doctor's exit status does not change, but a missing
    # observation is not mesh evidence, and the overall must not claim more
    # than its layers show.
    assert report["overall"] == "unknown"
    assert report["lifecycle"]["attached_mesh"] == {
        "state": "unknown",
        "detail": "The attached mesh health check was inconclusive.",
    }
    assert _action_ids(report) == ["inspect-attached-mesh"]


@pytest.mark.parametrize(
    ("detail", "state", "actions"),
    [
        ("no attached mesh; the next render session creates one", "idle", []),
        ("loading in flight; an attached mesh its own driver is working on is not a dirty one",
         "active", ["wait-for-active-work"]),
        ("usable", "unknown", ["inspect-attached-mesh"]),
    ],
)
def test_doctor_mesh_readiness_requires_canonical_evidence(detail, state, actions):
    report = readiness.readiness_from_doctor_rows([
        {"status": "ok", "name": "mesh health", "detail": detail},
    ])
    assert report["lifecycle"]["attached_mesh"]["state"] == state
    assert _action_ids(report) == actions


def test_doctor_failure_blocks_and_warning_only_degrades():
    failed = readiness.readiness_from_doctor_rows([
        {"status": "FAIL", "name": "node worker loop", "detail": "stopped"},
        {"status": "WARN", "name": "memory headroom", "detail": "low"},
    ])
    assert failed["overall"] == "blocked"
    assert failed["lifecycle"]["worker_service"]["state"] == "blocked"
    assert _action_ids(failed) == [
        "inspect-worker-service", "resolve-doctor-failures", "review-doctor-warnings"]

    warned = readiness.readiness_from_doctor_rows([
        {"status": "WARN", "name": "memory headroom", "detail": "low"},
    ])
    assert warned["overall"] == "degraded"
    assert _action_ids(warned) == ["review-doctor-warnings"]


def test_dirty_doctor_mesh_gets_the_same_recovery_action():
    report = readiness.readiness_from_doctor_rows([
        {"status": "FAIL", "name": "mesh health", "detail": "secret host/path"},
    ])
    assert report["overall"] == "blocked"
    assert report["lifecycle"]["attached_mesh"]["state"] == "blocked"
    assert _action_ids(report)[0] == "recover-attached-mesh"
    assert "secret host/path" not in json.dumps(report)


@pytest.mark.parametrize("rows", [None, [], "FAIL", [{}], [{"status": object(), "name": "x"}]])
def test_missing_or_malformed_doctor_rows_are_unknown(rows):
    report = readiness.readiness_from_doctor_rows(rows)
    assert report["overall"] == "unknown"
    assert "rerun-doctor" in _action_ids(report)
    json.dumps(report)


def test_a_known_doctor_failure_outranks_an_unreadable_row():
    report = readiness.readiness_from_doctor_rows([
        {}, {"status": "FAIL", "name": "torch CUDA", "detail": "unavailable"},
    ])
    assert report["overall"] == "blocked"
    assert "resolve-doctor-failures" in _action_ids(report)
    assert "rerun-doctor" in _action_ids(report)


def test_action_order_does_not_depend_on_doctor_row_order():
    rows = [
        {"status": "WARN", "name": "memory headroom", "detail": "low"},
        {"status": "FAIL", "name": "mesh health", "detail": "dirty"},
        {"status": "FAIL", "name": "node worker loop", "detail": "stopped"},
    ]
    forward = readiness.readiness_from_doctor_rows(rows)
    backward = readiness.readiness_from_doctor_rows(list(reversed(rows)))
    assert _action_ids(forward) == _action_ids(backward)


def test_public_state_spaces_and_every_report_value_are_json_serializable():
    assert readiness.OVERALL_STATES == ("ready", "degraded", "blocked", "unknown")
    assert set(readiness.LAYER_NAMES) == {
        "worker_service", "attached_mesh", "render_session"}
    for report in (
        _telemetry(),
        _telemetry(render={"active": True}),
        readiness.readiness_from_doctor_rows([
            {"status": "WARN", "name": "x", "detail": "y"}]),
    ):
        assert report["overall"] in readiness.OVERALL_STATES
        for layer in report["lifecycle"].values():
            assert layer["state"] in readiness.LAYER_STATES
        json.loads(json.dumps(report))


def test_a_doctor_overall_never_outranks_the_layers_it_publishes():
    """Row counts alone would publish ready over an unknown mesh, so the doctor
    overall also folds in its worker and mesh layers, as every projection here does."""
    for detail, expected in (
        ("attached mesh ready, 0 active leases", "ready"),
        ("no driver reachable; start ComfyUI and re-run doctor for the mesh view",
         "unknown"),
        ("this dgxm does not know the mesh state 'quiescing'", "unknown"),
    ):
        report = readiness.readiness_from_doctor_rows([
            {"status": "ok", "name": "mesh health", "detail": detail},
        ])
        assert report["overall"] == expected, detail
        # The row stays OK, so doctor's failure count and exit are untouched.
        layers = report["lifecycle"]
        required = readiness._overall([
            readiness._LAYER_OVERALL.get(layers[name]["state"], "ready")
            for name in ("worker_service", "attached_mesh")
        ])
        assert readiness.OVERALL_STATES.index(report["overall"]) >= (
            readiness.OVERALL_STATES.index(required)) or report["overall"] == required


def test_an_unknown_worker_layer_also_refuses_a_ready_doctor_overall():
    report = readiness.readiness_from_doctor_rows([
        {"status": "warn", "name": "w1 worker service",
         "detail": "10.0.0.2:29500 process=unknown listener=unknown",
         "reason": "unobserved"},
        {"status": "ok", "name": "mesh health",
         "detail": "attached mesh ready, 0 active leases"},
    ])

    assert report["lifecycle"]["worker_service"]["state"] == "unknown"
    assert report["overall"] == "unknown"


def test_an_omitted_lease_count_is_not_an_observed_zero():
    """A partial mesh block from an older or foreign driver must not earn the
    healthiest verdict by leaving the counts out."""
    healthy = {"state": "ok", "verdict": "live"}
    report = readiness.readiness_from_telemetry([], healthy, None)
    assert report["lifecycle"]["attached_mesh"]["state"] == "unknown"
    assert "inspect-attached-mesh" in _action_ids(report)

    # With both counts reported, zero active leases read ready and one reads active.
    observed = {**healthy, "active_leases": 0, "abandoned_samples": 0}
    assert readiness.readiness_from_telemetry(
        [], observed, None)["lifecycle"]["attached_mesh"]["state"] == "ready"
    busy = {**healthy, "active_leases": 1, "abandoned_samples": 0}
    assert readiness.readiness_from_telemetry(
        [], busy, None)["lifecycle"]["attached_mesh"]["state"] == "active"

    # An idle mesh with no counts is unknown too, and a dirty one stays dirty:
    # known-blocked must keep outranking unknown.
    idle = {"state": "idle", "verdict": "none"}
    assert readiness.readiness_from_telemetry(
        [], idle, None)["lifecycle"]["attached_mesh"]["state"] == "unknown"
    assert readiness.readiness_from_telemetry(
        [], {**idle, "active_leases": 0, "abandoned_samples": 0},
        None)["lifecycle"]["attached_mesh"]["state"] == "idle"
    assert readiness.readiness_from_telemetry(
        [], {**healthy, "poisoned": True},
        None)["lifecycle"]["attached_mesh"]["state"] == "blocked"


def test_abandoned_leases_stays_an_optional_alias():
    """Nothing in the repository emits it, so requiring it would make every
    healthy real block read unknown."""
    block = {"state": "ok", "verdict": "live",
             "active_leases": 0, "abandoned_samples": 0}
    assert readiness.readiness_from_telemetry(
        [], block, None)["lifecycle"]["attached_mesh"]["state"] == "ready"
    assert readiness.readiness_from_telemetry(
        [], {**block, "abandoned_leases": 2},
        None)["lifecycle"]["attached_mesh"]["state"] == "blocked"
