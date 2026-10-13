"""Projection uses bounded labels and does not report stale lifecycle facts."""
from __future__ import annotations

from dgx_monarch.nodes.metrics_route import _metrics_text
from dgx_monarch.runtime_diagnostics import PhaseMeasurements


def _diagnostics():
    store = PhaseMeasurements()
    store.record("rdma_read", 0.02, return_mode="rdma")
    store.record("rdma_read", 2, return_mode="rdma")
    return store.snapshot()


def test_phase_metrics_are_cumulative_histograms():
    result = _metrics_text({"diagnostics": _diagnostics()})
    labels = 'scope="driver",phase="rdma_read",outcome="ok",return_mode="rdma"'
    assert f'dgxm_phase_duration_seconds_count{{{labels}}} 2' in result
    assert f'dgxm_phase_duration_seconds_sum{{{labels}}} 2.02' in result
    assert f'dgxm_phase_duration_seconds_bucket{{le="0.1",{labels}}} 1' in result
    assert f'dgxm_phase_duration_seconds_bucket{{le="+Inf",{labels}}} 2' in result


def test_worker_diagnostics_do_not_require_host_gpu_poll():
    result = _metrics_text({"workers": [{"diagnostics": _diagnostics()}]})
    assert 'scope="worker",worker_index="0"' in result
    assert "rdma_read" in result


def test_mesh_current_counts_come_from_existing_fresh_snapshot():
    result = _metrics_text({"mesh": {
        "state": "dirty", "active_leases": 2, "abandoned_samples": 1,
        "pending": False, "poisoned": False,
    }})
    assert 'dgxm_mesh_state{state="dirty"} 1' in result
    assert "dgxm_mesh_active_leases 2" in result
    assert "dgxm_mesh_abandoned_samples 1" in result


def test_stale_payload_does_not_claim_current_mesh_or_phase_state():
    result = _metrics_text({"telemetry_error": "RefreshInFlight",
                           "mesh": {"state": "idle", "active_leases": 0},
                           "diagnostics": _diagnostics()})
    assert "dgxm_telemetry_stale 1" in result
    assert "dgxm_mesh_" not in result
    assert 'dgxm_diagnostics_available{scope="driver"} 0' in result
    assert "dgxm_phase_" not in result


def test_unknown_state_does_not_export_zero_counts():
    result = _metrics_text({"mesh": {"state": "unknown", "active_leases": 0}})
    assert 'dgxm_mesh_state{state="unknown"} 1' in result
    assert "dgxm_mesh_active_leases" not in result


def test_untrusted_phase_labels_and_inconsistent_buckets_are_omitted():
    block = _diagnostics()
    good = block["phases"][0]
    block["phases"] = [
        {**good, "phase": "secret\nrequest"},
        {**good, "phase": ["hostile"]},
        {**good, "count": 100},
        {**good, "count": 10**10000},
    ]
    result = _metrics_text({"diagnostics": block})
    assert "dgxm_phase_" not in result
    assert "secret" not in result


def test_duplicate_series_is_emitted_once():
    block = _diagnostics()
    block["phases"] *= 2
    result = _metrics_text({"diagnostics": block})
    assert result.count("dgxm_phase_duration_seconds_count{") == 1


def test_return_ownership_omits_missing_fields_and_counts_only_fixed_reasons():
    result = _metrics_text({"return_ownership": {
        "schema": 1, "available": True, "pending_read_jobs": 2,
    }, "workers": [{"return_ownership": {
        "schema": 1, "available": True, "live_handoffs": 1, "tombstones": 4,
        "return_counts": [
            {"kind": "message", "reason": "registry_capacity", "count": 3},
            {"kind": "rdma", "reason": "native", "count": 8},
            {"kind": "rdma", "reason": "native", "count": 9},
            {"kind": "message", "reason": "private detail", "count": 1},
            {"kind": ["invalid"], "reason": "native", "count": 1}],
    }}]})
    assert 'dgxm_return_pending_read_jobs{scope="driver"} 2' in result
    assert 'dgxm_return_live_handoffs{scope="driver"}' not in result
    assert 'dgxm_return_live_handoffs{scope="worker",worker_index="0"} 1' in result
    assert result.count('reason="native"') == 1
    assert 'reason="registry_capacity"' in result
    assert "private detail" not in result
    assert "return_poison_owners" not in result


def test_stale_or_unavailable_return_ownership_does_not_report_zero_state():
    block = {"schema": 1, "available": True, "live_handoffs": 0}
    for payload in ({"return_ownership": block, "telemetry_error": "stale"},
                    {"workers": [{"return_ownership": block, "telemetry_error": "stale"}]},
                    {"return_ownership": {**block, "available": False}}):
        result = _metrics_text(payload)
        assert "dgxm_return_live_handoffs" not in result
        assert "dgxm_return_ownership_available" in result


def test_failed_or_inflight_worker_poll_hides_cached_phase_and_ownership_counts():
    worker = {"diagnostics": _diagnostics(), "return_ownership": {
        "schema": 1, "available": True, "live_handoffs": 0}}
    for payload in (
        {"workers": [worker], "status_poll": {"fresh": False, "state": "busy"}},
        {"workers": [worker], "status_poll": {"state": "unknown"}},
        {"workers": [{**worker, "status_error": ""}]},
        {"workers": [{**worker, "status_poll": {"fresh": False}}]},
    ):
        result = _metrics_text(payload)
        assert "dgxm_phase_duration_seconds_count" not in result
        assert "dgxm_return_live_handoffs" not in result
    result = _metrics_text({"workers": [worker], "status_poll": {"fresh": True}})
    assert "dgxm_phase_duration_seconds_count" in result


def test_a_worker_on_older_source_reports_unavailable_rather_than_no_series():
    current = {"diagnostics": _diagnostics(),
               "return_ownership": {"schema": 1, "available": True, "live_handoffs": 0}}
    for older in ({}, {"diagnostics": None, "return_ownership": "absent"}):
        result = _metrics_text({"diagnostics": _diagnostics(), "workers": [current, older]})
        # An `== 0` alert fires for the worker that lacks the blocks after a partial deploy.
        assert 'dgxm_diagnostics_available{scope="worker",worker_index="0"} 1' in result
        assert 'dgxm_diagnostics_available{scope="worker",worker_index="1"} 0' in result
        assert 'dgxm_return_ownership_available{scope="worker",worker_index="0"} 1' in result
        assert 'dgxm_return_ownership_available{scope="worker",worker_index="1"} 0' in result
        assert 'worker_index="1",phase=' not in result
        assert 'dgxm_return_live_handoffs{scope="worker",worker_index="1"}' not in result
    driverless = _metrics_text({})
    assert 'dgxm_diagnostics_available{scope="driver"} 0' in driverless
    assert 'dgxm_return_ownership_available{scope="driver"} 0' in driverless
