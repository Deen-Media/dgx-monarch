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
    assert set(model) == {"schemaVersion", "source", "overall", "notice", "rows", "presentation"}
    encoded = json.dumps(model)
    assert "private-operation" not in encoded
    assert "/arbitrary" not in encoded
    assert "actions" not in encoded


def _live_report(*, mesh_active=False, render_active=False, error=None, workers=None):
    from dgx_monarch.operator_readiness import readiness_from_telemetry

    return {"readiness": readiness_from_telemetry(
        workers if workers is not None else [{"healthy": True}],
        {"state": "ok", "verdict": "live", "active_leases": int(mesh_active),
         "abandoned_samples": 0},
        {"active": render_active}, error,
    )}


def test_python_and_browser_presentation_agree_without_changing_legacy_overall():
    fixtures = [
        _live_report(), _live_report(mesh_active=True), _live_report(render_active=True),
        _live_report(render_active=True, error="timeout"),
        _live_report(render_active=True, workers=[{"healthy": False}]),
    ]
    for fixture, model in zip(fixtures, _run_model(fixtures), strict=True):
        assert model["source"] == "canonical"
        assert model["presentation"] == fixture["readiness"]["presentation"]
        assert model["overall"] == fixture["readiness"]["overall"]
    assert fixtures[2]["readiness"]["overall"] == "degraded"
    assert fixtures[2]["readiness"]["presentation"]["health"] == "ready"


@pytest.mark.parametrize("changes", [
    {"health": "ready"}, {"activity": "idle"}, {"maintenance": "none"},
    {"schema_version": True}, {"schema_version": 2}, {"command": "unsafe-operation"},
])
def test_inconsistent_presentation_fails_closed(changes):
    fixture = _live_report(render_active=True, error="timeout")
    fixture["readiness"]["presentation"].update(changes)
    (model,) = _run_model([fixture])
    assert model["source"] == "invalid"
    assert model["overall"] == "unknown"
    assert model["presentation"]["health"] is None
    assert "unsafe-operation" not in json.dumps(model)


def test_legacy_active_payload_does_not_claim_separate_health():
    fixture = _live_report(render_active=True)
    del fixture["readiness"]["presentation"]
    (model,) = _run_model([fixture])
    assert model["source"] == "canonical"
    assert model["overall"] == "degraded"
    assert model["presentation"]["health"] is None
    assert model["presentation"]["activity"] == "rendering"
    assert model["presentation"]["maintenance"] == "inspect_and_wait"
    assert "Legacy readiness includes active work" in model["notice"]


def test_invalid_worker_activity_cannot_be_presented_as_healthy_occupancy():
    fixture = _live_report()
    fixture["readiness"]["overall"] = "degraded"
    fixture["readiness"]["lifecycle"]["worker_service"]["state"] = "active"
    (model,) = _run_model([fixture])
    assert model["source"] == "invalid"
    assert model["overall"] == "unknown"


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


# --- topology and rail lines (web/js/dgx_monarch_fabric_model.js) ----------

FABRIC = Path(__file__).parents[1] / "web/js/dgx_monarch_fabric_model.js"


def _run_fabric(script: str):
    """Run ``script`` against the real fabric model under node; it sets ``out``."""
    node = shutil.which("node")
    if node is None:  # pragma: no cover - environment dependent
        pytest.skip("node not installed; JS model check skipped")
    driver = (
        "import * as fabric from " f"{json.dumps(FABRIC.as_uri())};\n"
        "let out;\n" + script + "\nprocess.stdout.write(JSON.stringify(out));\n"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", driver],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode != 0:  # pragma: no cover - only a real JS break
        raise AssertionError(f"node driver failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def test_the_topology_label_matches_the_drivers_for_every_preset():
    # A worker publishes dict(topo) from its setup request; the label must be
    # the one the driver's log and the render record use.
    from dataclasses import asdict

    from dgx_monarch.topology import PRESETS, topology_from_preset

    cases = []
    for preset in PRESETS:
        for world in (1, 2, 4):
            try:
                topology = topology_from_preset(preset, world)
            except ValueError:
                continue
            cases.append((asdict(topology), topology.describe()))
    labels = _run_fabric(
        f"out = {json.dumps([case for case, _ in cases])}.map(fabric.topologyLabel);")
    assert labels == [label for _, label in cases]
    assert {"uly2", "cfg2", "uly2+fsdp", "single"} <= set(labels)


def test_a_worker_before_setup_has_no_topology_and_junk_reads_none():
    out = _run_fabric(
        "out = [{}, null, [], 'uly2', {ulysses: 0}, {ulysses: 1.5}, {ring: 'two'},"
        " {ulysses: 2, fsdp: 'yes'}, {cfg: 2}].map(fabric.topologyLabel);")
    assert out == [None, None, None, None, None, None, None, "uly2", "cfg2"]


def test_the_active_topology_line_names_one_shape_or_every_disagreeing_one():
    out = _run_fabric("""
        const uly = {ulysses: 2, ring: 1, cfg: 1, dp: 1, fsdp: false, world: 2};
        out = [
            fabric.activeTopology([{rank: 0, world: 2, topology: uly},
                                   {rank: 1, world: 2, topology: uly}]),
            fabric.activeTopology([{rank: 0, world: 2, topology: uly},
                                   {rank: 1, world: 2, topology: {cfg: 2, world: 2}}]),
            fabric.activeTopology([{topology: {}}, {status_error: "TimeoutError"}]),
            fabric.activeTopology(undefined),
            fabric.activeTopology([null, 7, {topology: {ulysses: 2, fsdp: true}}]),
        ];
    """)
    assert out == [
        "Topology uly2 · world 2",
        "Topology mixed: cfg2, uly2",
        None,
        None,
        "Topology uly2+fsdp",
    ]


def test_rail_rates_use_the_tui_math_on_a_monotonic_clock():
    # 2.5 s apart in milliseconds: 5e9 bytes in reads 2 GB/s.
    out = _run_fabric("""
        const tracker = fabric.createRailRateTracker();
        const first = tracker.update("a", {r0: {rx_bytes: 1e9, tx_bytes: 4e9}}, 100, 1000);
        const second = tracker.update("a", {r0: {rx_bytes: 6e9, tx_bytes: 2e9}}, 102.5, 3500);
        out = {first, second};
    """)
    # A first sample has no interval yet; a counter that went backwards
    # (a reset) reads 0, never a negative rate.
    assert out["first"] == {"r0": {"rx_gbs": None, "tx_gbs": None}}
    assert out["second"] == {"r0": {"rx_gbs": 2.0, "tx_gbs": 0}}


def test_rail_rates_floor_the_interval_and_keep_hosts_and_rails_apart():
    out = _run_fabric("""
        const tracker = fabric.createRailRateTracker();
        tracker.update("a", {r0: {rx_bytes: 0, tx_bytes: 0}}, 1, 1000);
        tracker.update("b", {r0: {rx_bytes: 5e9, tx_bytes: 5e9}}, 1, 1000);
        const sameInstant = tracker.update("a", {r0: {rx_bytes: 1e6, tx_bytes: 0},
                                                 r1: {rx_bytes: 9, tx_bytes: 9}}, 2, 1000);
        const other = tracker.update("b", {r0: {rx_bytes: 6e9, tx_bytes: 5e9}}, 2, 2000);
        out = {sameInstant, other};
    """)
    # dt floors at 1 ms: 1e6 bytes over 1e-3 s is 1 GB/s. A new rail starts
    # without a rate, and host b's counters never mix with host a's.
    assert out["sameInstant"] == {
        "r0": {"rx_gbs": 1.0, "tx_gbs": 0}, "r1": {"rx_gbs": None, "tx_gbs": None}}
    assert out["other"] == {"r0": {"rx_gbs": 1.0, "tx_gbs": 0}}


def test_a_repeated_reading_keeps_its_rates_instead_of_reading_idle():
    # While a render keeps a worker busy the driver serves its last good row:
    # the same stamp and counters on several polls. That gap must not read 0,
    # and the next fresh row is measured from the last fresh one.
    out = _run_fabric("""
        const tracker = fabric.createRailRateTracker();
        tracker.update("a", {r0: {rx_bytes: 0, tx_bytes: 0}}, 10, 0);
        const fresh = tracker.update("a", {r0: {rx_bytes: 2e9, tx_bytes: 2e9}}, 11, 1000);
        const cached = tracker.update("a", {r0: {rx_bytes: 2e9, tx_bytes: 2e9}}, 11, 3500);
        const next = tracker.update("a", {r0: {rx_bytes: 8e9, tx_bytes: 2e9}}, 14, 4000);
        const unstamped = tracker.update("a", {r0: {rx_bytes: 8e9, tx_bytes: 2e9}}, undefined, 5000);
        out = {fresh, cached, next, unstamped};
    """)
    assert out["fresh"] == {"r0": {"rx_gbs": 2.0, "tx_gbs": 2.0}}
    assert out["cached"] == out["fresh"]
    assert out["next"] == {"r0": {"rx_gbs": 2.0, "tx_gbs": 0}}
    # Without a stamp the TUI rule applies: equal counters read 0.
    assert out["unstamped"] == {"r0": {"rx_gbs": 0, "tx_gbs": 0}}


def test_rail_rates_use_the_boxs_read_stamps_when_both_readings_carry_one():
    # The panel's clock says when a poll arrived; rails_m says when the box
    # read its counters, by its monotonic clock. A reading that waited in the
    # driver's cache arrives late, so only the box's stamps give the true
    # interval. Without a stamp on both readings, or when the stamp did not
    # move forward, the panel's clock applies, as for an older driver. The
    # wall-clock stamps (the fifth argument) jump about and never time a rate.
    out = _run_fabric("""
        const tracker = fabric.createRailRateTracker();
        const r = (rx) => ({r0: {rx_bytes: rx, tx_bytes: 0}});
        tracker.update("a", r(0), 1, 0, 1000, 50);
        const stamped = tracker.update("a", r(5e9), 2, 4000, 4602.5, 52.5);
        const unstamped = tracker.update("a", r(7e9), 3, 6000);
        const mixed = tracker.update("a", r(8e9), 4, 7000, 4610, 60);
        const backwards = tracker.update("a", r(10e9), 5, 9000, 4612, 59);
        const resumed = tracker.update("a", r(14e9), 6, 9500, 900, 61);
        const repeat = tracker.update("a", r(14e9), 7, 12000, 900, 61);
        out = {stamped, unstamped, mixed, backwards, resumed, repeat};
    """)
    rx = {name: rates["r0"]["rx_gbs"] for name, rates in out.items()}
    # 5e9 over the box's 2.5 s, not the panel's 4 s.
    assert rx["stamped"] == 2.0
    # Unstamped, then stamped after an unstamped one: the panel's 2 s and 1 s.
    assert rx["unstamped"] == 1.0
    assert rx["mixed"] == 1.0
    # A stamp that went back (the box rebooted) falls back to the panel's 2 s.
    assert rx["backwards"] == 1.0
    # Both stamped again: 4e9 over the box's 2 s, not the panel's 0.5 s.
    assert rx["resumed"] == 2.0
    # The same read stamp is the same reading: it keeps its rates.
    assert out["repeat"] == out["resumed"]


def test_each_poll_passes_the_boxs_read_stamp_to_the_tracker():
    out = _run_fabric("""
        const tracker = fabric.createRailRateTracker();
        const cards = (rx, t, readAt, readMono) => [{name: "boxA", stats: {
            host: "boxA", t, rails_t: readAt, rails_m: readMono,
            rails: {r0: {rx_bytes: rx, tx_bytes: 0}}}}];
        fabric.hostRailRates(tracker, cards(0, 10, 100, 40), 1000);
        out = Object.fromEntries(fabric.hostRailRates(tracker, cards(3e9, 14, 101, 41), 5000));
    """)
    # 3e9 over the box's 1 s, not the panel's 4 s.
    assert out == {"boxA": {"r0": {"rx_gbs": 3.0, "tx_gbs": 0}}}


def test_a_wall_clock_step_on_the_box_never_times_a_rail_rate():
    # The box's wall clock jumps forward an hour between two readings one
    # second apart, and the second one waits 3 s in the driver's cache. With
    # the monotonic stamp the rate is the true 10 GB/s, where the wall-clock
    # stamps would give about 0.003 GB/s. A driver that sends only the
    # wall-clock stamp is timed by the panel's clock, as an older driver is.
    out = _run_fabric("""
        const card = (host, rx, t, readAt, readMono) => ({name: host, stats: {
            host, t, rails_t: readAt, rails_m: readMono,
            rails: {r0: {rx_bytes: rx, tx_bytes: 0}}}});
        const run = (readMono) => {
            const tracker = fabric.createRailRateTracker();
            fabric.hostRailRates(tracker, [card("boxA", 0, 10, 1000, readMono(0))], 1000);
            return Object.fromEntries(fabric.hostRailRates(
                tracker, [card("boxA", 10e9, 11, 4601, readMono(1))], 5000));
        };
        out = {monotonic: run((s) => 40 + s), wallOnly: run(() => undefined)};
    """)
    # 10e9 over the box's monotonic 1 s.
    assert out["monotonic"] == {"boxA": {"r0": {"rx_gbs": 10.0, "tx_gbs": 0}}}
    # 10e9 over the panel's 4 s, not over the box's 3601 wall-clock seconds.
    assert out["wallOnly"] == {"boxA": {"r0": {"rx_gbs": 2.5, "tx_gbs": 0}}}


def test_two_cards_for_one_box_read_its_rails_once_per_poll():
    # Two ranks on one box send two worker rows with the same host name, read
    # 0.3 s apart by their own processes, drawn 0.2 ms apart. The true rate is
    # 2 GB/s. Timing the second row against the first would print 600 GB/s.
    out = _run_fabric("""
        const tracker = fabric.createRailRateTracker();
        const poll = (base, at, nowMs) => [
            {name: "boxA", stats: {host: "boxA", t: at,
                                   rails: {r0: {rx_bytes: base, tx_bytes: 0}}}},
            {name: "boxA", stats: {host: "boxA", t: at + 0.3,
                                   rails: {r0: {rx_bytes: base + 6e8, tx_bytes: 0}}}},
            {name: "boxB (driver)", stats: {host: "boxB", t: at,
                                            rails: {r0: {rx_bytes: base, tx_bytes: 0}}}},
            {name: "junk", stats: null},
        ];
        let calls = 0;
        const counted = {update(...args) { calls += 1; return tracker.update(...args); }};
        const first = fabric.hostRailRates(counted, poll(1e9, 10, 1000), 1000);
        const second = fabric.hostRailRates(counted, poll(6e9, 12.5, 3500), 3500);
        out = {calls, hosts: [...second.keys()],
               first: Object.fromEntries(first), second: Object.fromEntries(second),
               none: Object.fromEntries(fabric.hostRailRates(tracker, undefined, 0))};
    """)
    assert out["calls"] == 4
    assert out["hosts"] == ["boxA", "boxB"]
    assert out["first"]["boxA"] == {"r0": {"rx_gbs": None, "tx_gbs": None}}
    assert out["second"] == {"boxA": {"r0": {"rx_gbs": 2.0, "tx_gbs": 0}},
                             "boxB": {"r0": {"rx_gbs": 2.0, "tx_gbs": 0}}}
    assert out["none"] == {}


def test_rail_rates_skip_malformed_rows_like_the_tui():
    out = _run_fabric("""
        const tracker = fabric.createRailRateTracker();
        out = [
            tracker.update("a", {bad_row: "oops", bad_number: {rx_bytes: "one", tx_bytes: 2},
                                 nan: {rx_bytes: NaN, tx_bytes: 1}, good: {rx_bytes: 3, tx_bytes: 4}},
                           1, 0),
            tracker.update("b", null, 1, 0),
            tracker.update("c", [1, 2], 1, 0),
        ];
    """)
    assert out == [{"good": {"rx_gbs": None, "tx_gbs": None}}, {}, {}]


def test_the_rail_line_marks_a_down_rail_and_skips_unused_ports():
    out = _run_fabric("""
        const rates = {
            rocep1s0f0: {rx_gbs: 1.234, tx_gbs: 1.2},
            roceP2p1s0f0: {rx_gbs: 0, tx_gbs: 0},
        };
        const state = {
            rocep1s0f0: {state: "ACTIVE", phys_state: "5: LinkUp"},
            roceP2p1s0f0: {state: "1: DOWN", phys_state: "3: Disabled"},
            rocep1s0f1: {state: "ACTIVE", phys_state: "5: LinkUp"},
            uncabled: {state: "1: DOWN", phys_state: "2: Polling"},
            trained_down: {state: "1: DOWN", phys_state: "5: LinkUp"},
        };
        const rows = fabric.railRows(rates, state);
        out = {rows, parts: fabric.railLineParts(rows),
               older: fabric.railLineParts(fabric.railRows(rates, undefined)),
               none: fabric.railLineParts(fabric.railRows({}, {}))};
    """)
    # Shown: both rails that carry traffic, and the cabled port that is down.
    # Not shown: the ACTIVE port with no counters and the uncabled one.
    assert [row["name"] for row in out["rows"]] == ["P2p", "f0", "trained_down"]
    assert [row["active"] for row in out["rows"]] == [False, True, False]
    assert out["parts"] == [
        {"text": "rails P2p rx 0.00 tx 0.00 GB/s", "alert": False},
        {"text": " DOWN (Disabled)", "alert": True},
        {"text": " · f0 rx 1.23 tx 1.20 GB/s", "alert": False},
        {"text": " · trained_down rx - tx - GB/s", "alert": False},
        {"text": " DOWN (LinkUp)", "alert": True},
    ]
    # An older payload with no link state draws rates and no marker.
    assert out["older"] == [
        {"text": "rails P2p rx 0.00 tx 0.00 GB/s", "alert": False},
        {"text": " · f0 rx 1.23 tx 1.20 GB/s", "alert": False},
    ]
    assert out["none"] == []


def test_the_rail_line_names_each_port_of_a_device_with_two_rails():
    # The driver keys a device with two cabled ports as device/port, the same
    # in the counters and the link state, so each port keeps its own marker.
    out = _run_fabric("""
        const rates = {"mlx5_0/1": {rx_gbs: 1, tx_gbs: 2}, "mlx5_0/2": {rx_gbs: 0, tx_gbs: 0}};
        const state = {"mlx5_0/1": {state: "ACTIVE", phys_state: "5: LinkUp"},
                       "mlx5_0/2": {state: "1: DOWN", phys_state: "3: Disabled"}};
        out = fabric.railLineParts(fabric.railRows(rates, state));
    """)
    assert out == [
        {"text": "rails mlx5_0/1 rx 1.00 tx 2.00 GB/s", "alert": False},
        {"text": " · mlx5_0/2 rx 0.00 tx 0.00 GB/s", "alert": False},
        {"text": " DOWN (Disabled)", "alert": True},
    ]


def test_the_last_render_line_says_where_it_ran_and_its_step_rate():
    out = _run_fabric("""
        out = [
            fabric.lastRenderFacts({topology: "uly2", auto_row: 11, sec_per_step: 1.92}),
            fabric.lastRenderFacts({topology: "uly2+fsdp", auto_row: null, sec_per_step: 0.5}),
            fabric.lastRenderFacts({model: "m", wall_s: 3}),
            fabric.lastRenderFacts(null),
            fabric.lastRenderFacts({topology: 7, auto_row: "11", sec_per_step: "x"}),
        ];
    """)
    assert out == ["uly2 · auto row 11 · 1.92s/it", "uly2+fsdp · 0.5s/it", "", "", ""]


def test_the_panel_rates_equal_dgxm_tops_for_the_same_two_readings():
    from dgx_monarch.tui.data import Poller

    samples = [
        (10.0, {"rocep1s0f0": {"rx_bytes": 4_000_000_000, "tx_bytes": 1_000},
                "roceP2p1s0f0": {"rx_bytes": 7, "tx_bytes": 7}}),
        (12.5, {"rocep1s0f0": {"rx_bytes": 9_123_456_789, "tx_bytes": 500},
                "roceP2p1s0f0": {"rx_bytes": 2_500_000_007, "tx_bytes": 7}}),
    ]
    poller = Poller(loops=())
    tui = [poller._rail_rates("box", rails, at) for at, rails in samples][-1]
    panel = _run_fabric(
        "const tracker = fabric.createRailRateTracker();\n"
        f"const samples = {json.dumps(samples)};\n"
        "out = samples.map(([at, rails]) => tracker.update('box', rails, at, at * 1000)).at(-1);")
    aliases = {"rocep1s0f0": "f0", "roceP2p1s0f0": "P2p"}
    assert set(tui) == {aliases[dev] for dev in panel}
    for dev, rate in panel.items():
        assert rate == pytest.approx(tui[aliases[dev]])


def test_the_panel_and_dgxm_top_agree_on_stamped_readings():
    # Polls 4 s and 1 s apart carry readings the boxes took 2.5 s and 2 s
    # apart by their monotonic clocks. Both readers time the rates by those
    # stamps, so they agree.
    from dgx_monarch.tui.data import Poller

    samples = [
        (10.0, 500.0, {"rocep1s0f0": {"rx_bytes": 1_000, "tx_bytes": 0}}),
        (14.0, 502.5, {"rocep1s0f0": {"rx_bytes": 5_000_001_000, "tx_bytes": 2_500_000_000}}),
        (15.0, 504.5, {"rocep1s0f0": {"rx_bytes": 9_000_001_000, "tx_bytes": 2_500_000_000}}),
    ]
    poller = Poller(loops=())
    tui = [poller._rail_rates("box", rails, at, read_mono=read) for at, read, rails in samples][1:]
    panel = _run_fabric(
        "const tracker = fabric.createRailRateTracker();\n"
        f"const samples = {json.dumps(samples)};\n"
        "out = samples.map(([at, read, rails]) => "
        "tracker.update('box', rails, at, at * 1000, undefined, read)).slice(1);")
    assert [rate["rocep1s0f0"] for rate in panel] == [
        {"rx_gbs": 2.0, "tx_gbs": 1.0}, {"rx_gbs": 2.0, "tx_gbs": 0}]
    for tui_rate, panel_rate in zip(tui, panel, strict=True):
        assert panel_rate["rocep1s0f0"] == pytest.approx(tui_rate["f0"])
