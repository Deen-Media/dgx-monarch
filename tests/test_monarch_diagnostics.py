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

    monkeypatch.setitem(sys.modules, "monarch.actor", types.SimpleNamespace(
        get_meter=lambda: Meter()))
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


def test_disabled_adapter_never_imports_monarch(monkeypatch):
    adapter.configure()
    monkeypatch.setitem(sys.modules, "monarch.actor", None)
    assert adapter.configure() == {"metrics": False, "available": True}
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
    """The refusal state survives, and the driver now turns it into a typed error."""
    import os

    from dgx_monarch import monarch_capabilities, monarch_launch
    from dgx_monarch import monarch_policy as policy
    from dgx_monarch.config import ClusterConfig, ClusterConfigError

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
    # Driver semantics: a requested export that cannot start refuses before any
    # runtime admission or transport work, instead of silently dropping metrics.
    monkeypatch.setattr(monarch_capabilities, 'assert_supported_runtime',
                        lambda: pytest.fail('runtime admitted after a diagnostics refusal'))
    with pytest.raises(ClusterConfigError, match=r'otlp_restart_required.*restart ComfyUI'):
        monarch_launch.prepare_driver(ClusterConfig(monarch=selected))
    assert adapter.status()['reason'] == 'otlp_restart_required'
    assert 'OTEL_EXPORTER_OTLP_ENDPOINT' not in os.environ


def test_actor_and_worker_refusals_log_an_error_and_continue(monkeypatch):
    import logging

    from dgx_monarch.log import get_logger
    from dgx_monarch.monarch_policy import DiagnosticsPolicy

    monkeypatch.delenv('OTEL_EXPORTER_OTLP_ENDPOINT', raising=False)
    monkeypatch.setitem(sys.modules, 'monarch._rust_bindings', types.ModuleType('native'))
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = get_logger('dgx_monarch.monarch_diagnostics')
    logger.addHandler(handler)
    try:
        requested = DiagnosticsPolicy(metrics=True, otlp_endpoint='http://127.0.0.1:4318', otlp_logs=True)
        state = adapter.configure_policy_logged(requested, 'actor')
        adapter.configure_policy_logged(DiagnosticsPolicy(), 'worker')
    finally:
        logger.removeHandler(handler)
    assert state['reason'] == 'otlp_restart_required'
    assert [(r.levelno, r.getMessage()) for r in records] == [(
        logging.ERROR, 'Monarch diagnostics requested for this actor are unavailable '
        '(otlp_restart_required); it continues without them')]


@pytest.mark.parametrize(('requested', 'state', 'reason'), [
    ({'metrics': True}, {'available': False, 'reason': 'instrument_setup_failed'}, 'instrument_setup_failed'),
    ({'otlp_endpoint': 'http://x:1', 'otlp_logs': True}, {'available': False}, 'unavailable'),
    ({'metrics': True}, {'available': True, 'metrics': True}, None),
    ({}, {'available': False, 'reason': 'otlp_restart_required'}, None),
    ({'metrics': True}, None, None),
])
def test_requested_refusal_reports_only_requested_export(requested, state, reason):
    from dgx_monarch.monarch_policy import DiagnosticsPolicy

    assert adapter.requested_refusal(state, DiagnosticsPolicy(**requested)) == reason


_IMPORT_ORDER_PROBE = r'''
import importlib.util, os, sys
from pathlib import Path

def native():
    return sorted(m for m in sys.modules if m.startswith("monarch._rust_bindings"))

import dgx_monarch.nodes  # what ComfyUI registration reaches
assert native() == [], ("nodes", native())
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location(
    "dgxm_entry_probe", root / "__init__.py", submodule_search_locations=[str(root)])
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)  # the custom_nodes entrypoint, fault hook call included
assert native() == [], ("entrypoint", native())
import dgx_monarch.mesh as mesh
assert mesh._FAULT_HOOK_INSTALLED is False  # deferred until prepare_driver loads Monarch

from dgx_monarch.monarch_diagnostics import configure_policy
from dgx_monarch.monarch_policy import apply_policy, parse_policy
policy = parse_policy({"diagnostics": {"otlp_endpoint": "http://127.0.0.1:4318", "otlp_logs": True}})
apply_policy(policy, role="driver")
state = configure_policy(policy.diagnostics)
assert state.get("export_scope") == "upstream_metrics_and_logs", state
assert os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://127.0.0.1:4318"
print("ok")
'''


def test_node_registration_leaves_monarch_unloaded_so_driver_otlp_can_start():
    """Node and entrypoint import must not load Monarch's native bindings.

    Their endpoint is fixed when the bindings load, so a load at registration
    made every driver OTLP endpoint refuse with otlp_restart_required.
    """
    import os
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if key != 'OTEL_EXPORTER_OTLP_ENDPOINT' and not key.startswith('MONARCH_RDMA_')
           and key not in ('DGXM_MONARCH_POLICY', 'DGXM_RDMA_QP_SPLIT')}
    env.update(PYTHONPATH=str(repo / 'src'), PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run([sys.executable, '-c', _IMPORT_ORDER_PROBE, str(repo)], env=env,
                            capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith('ok')


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
