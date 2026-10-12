"""Public 0.7 instrumentation only; no runtime actors or external backend."""
from __future__ import annotations

import sys
import threading
import types

import pytest

from dgx_monarch import monarch_diagnostics as adapter
from dgx_monarch import runtime_diagnostics as diagnostics


@pytest.fixture(autouse=True)
def isolated_policy_status(monkeypatch):
    monkeypatch.setattr(adapter, "_POLICY_STATE", {})


@pytest.fixture
def fake_meter(monkeypatch):
    adapter.configure()
    records = []
    delivered = threading.Event()

    class Instrument:
        def add(self, value, labels):
            records.append(("count", value, labels))

        def record(self, value, labels):
            records.append(("duration", value, labels))
            delivered.set()

    class Meter:
        def create_counter(self, name):
            assert name == "dgxm.phase.completed"
            return Instrument()

        def create_histogram(self, name):
            assert name == "dgxm.phase.duration.seconds"
            return Instrument()

    class Span:
        def __init__(self, name):
            records.append(("span", name))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            records.append(("span_closed",))

    monkeypatch.setitem(sys.modules, "monarch.actor", types.SimpleNamespace(
        get_meter=lambda: Meter(), span=Span))
    monkeypatch.setattr(diagnostics, "_STORE", diagnostics.PhaseMeasurements())
    yield records, delivered
    adapter.configure()


def test_opt_in_uses_only_supported_instruments_and_fixed_labels(fake_meter):
    records, delivered = fake_meter
    assert adapter.configure(metrics=True)["available"] is True
    diagnostics.record_phase("collect", 0.25, return_mode="message")
    assert delivered.wait(2)
    assert records == [
        ("count", 1, {"phase": "collect", "outcome": "ok", "return_mode": "message"}),
        ("duration", 0.25, {"phase": "collect", "outcome": "ok", "return_mode": "message"}),
    ]


def test_trace_opt_in_has_no_payload_or_private_job(fake_meter):
    records, _ = fake_meter
    assert adapter.configure(traces=True)["available"] is True
    with diagnostics.phase("setup"):
        pass
    assert records == [("span", "dgxm.setup"), ("span_closed",)]


def test_disabled_adapter_never_imports_monarch(monkeypatch):
    adapter.configure()
    monkeypatch.setitem(sys.modules, "monarch.actor", None)
    assert adapter.configure() == {"metrics": False, "traces": False, "available": True}
    diagnostics.record_phase("compute", 1)


def test_setup_failure_is_advisory(monkeypatch):
    adapter.configure()
    monkeypatch.setitem(sys.modules, "monarch.actor", None)
    state = adapter.configure(metrics=True)
    assert state["available"] is False
    assert state["reason"] == "instrument_setup_failed"
    with diagnostics.phase("compute"):
        pass
    # A failed attempt can be explicitly disabled without an import.
    adapter.configure()


def test_queue_is_bounded_without_blocking_caller():
    instrument = types.SimpleNamespace(add=lambda *_: None, record=lambda *_: None)
    meter = types.SimpleNamespace(create_counter=lambda _: instrument,
                                  create_histogram=lambda _: instrument)
    exporter = adapter._MetricExporter(meter)
    event = {"phase": "ack", "outcome": "ok", "return_mode": "rdma", "duration_s": 1}
    for _ in range(adapter._QUEUE_CAPACITY + 5):
        exporter.submit(event)
    assert exporter._queue.qsize() == adapter._QUEUE_CAPACITY
    assert exporter.dropped == 5


def test_otlp_logs_require_separate_explicit_acknowledgment(monkeypatch):
    from dgx_monarch.config_schema import ClusterConfigError
    from dgx_monarch.monarch_policy import DiagnosticsPolicy

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    with pytest.raises(ClusterConfigError, match="also exports upstream logs"):
        adapter.configure_policy(DiagnosticsPolicy(otlp_endpoint="http://127.0.0.1:4318"))
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in __import__("os").environ


def test_new_otlp_destination_requires_fresh_native_runtime(monkeypatch):
    from dgx_monarch.monarch_policy import DiagnosticsPolicy

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setitem(sys.modules, "monarch._rust_bindings.test_marker", types.ModuleType("marker"))
    state = adapter.configure_policy(DiagnosticsPolicy(
        metrics=True, otlp_endpoint="http://127.0.0.1:4318", otlp_logs=True))
    assert state["available"] is False
    assert state["reason"] == "otlp_restart_required"
    assert adapter.status()["reason"] == "otlp_restart_required"
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in __import__("os").environ


def test_apply_then_configure_preserves_late_native_refusal(monkeypatch, isolated_environ):
    import os

    from dgx_monarch import monarch_policy as policy

    monkeypatch.setattr(policy, '_ACTIVE', None)
    for name in policy.policy_environment(policy.MonarchPolicy()):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    monkeypatch.setitem(sys.modules, 'monarch._rust_bindings', types.ModuleType('native'))
    selected = policy.MonarchPolicy(diagnostics=policy.DiagnosticsPolicy(
        otlp_endpoint='http://127.0.0.1:4318', otlp_logs=True))
    policy.apply_policy(selected)
    adapter.configure_policy(selected.diagnostics)
    assert 'OTEL_EXPORTER_OTLP_ENDPOINT' not in os.environ
    assert adapter.status()['available'] is False
    assert adapter.status()['reason'] == 'otlp_restart_required'


def test_otlp_endpoint_is_installed_only_before_native_import(monkeypatch, isolated_environ):
    import os

    from dgx_monarch.monarch_policy import DiagnosticsPolicy

    for name in tuple(sys.modules):
        if name.startswith('monarch._rust_bindings'):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    adapter.configure_policy(DiagnosticsPolicy(
        otlp_endpoint='http://127.0.0.1:4318', otlp_logs=True))
    assert os.environ['OTEL_EXPORTER_OTLP_ENDPOINT'] == 'http://127.0.0.1:4318'
    assert adapter.status()['available'] is True
    assert adapter.status()['export_scope'] == 'upstream_metrics_and_logs'


def test_inherited_matching_child_endpoint_can_attach_instruments(monkeypatch):
    from dgx_monarch.monarch_policy import DiagnosticsPolicy

    monkeypatch.setenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://127.0.0.1:4318')
    monkeypatch.setitem(sys.modules, 'monarch._rust_bindings', types.ModuleType('native'))
    adapter.configure_policy(DiagnosticsPolicy(
        otlp_endpoint='http://127.0.0.1:4318', otlp_logs=True))
    assert adapter.status()['available'] is True
    assert adapter.status()['upstream_log_export_acknowledged'] is True
