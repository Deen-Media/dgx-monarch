"""Per-signal OTLP endpoints cannot start Monarch export behind the policy.

torchmonarch 0.7.0 exported over OTLP only when the base
``OTEL_EXPORTER_OTLP_ENDPOINT`` was set. 0.8.0 starts each signal from its own
``OTEL_EXPORTER_OTLP_{LOGS,METRICS,TRACES}_ENDPOINT`` even with the base empty,
and adds spans; every post carries the user and host names, and the logs carry
process command lines. The policy owns the base variable only, so every role
refuses the per-signal ones and actors start without them.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import monarch_launch
from dgx_monarch import monarch_policy as policy
from dgx_monarch.config import ClusterConfigError
from dgx_monarch.monarch_diagnostics import start_refusal

PROBE = Path(__file__).with_name("otlp_signal_probe.py")
SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture
def isolated_policy(monkeypatch, isolated_environ):
    monkeypatch.setattr(policy, "_ACTIVE", None)
    for key in (*policy.policy_environment(policy.MonarchPolicy()), "OTEL_EXPORTER_OTLP_ENDPOINT",
                *policy.SIGNAL_OTLP_ENV):
        monkeypatch.delenv(key, raising=False)


def test_the_three_per_signal_variables_are_the_ones_guarded():
    assert set(policy.SIGNAL_OTLP_ENV) == {
        f"OTEL_EXPORTER_OTLP_{signal}_ENDPOINT" for signal in ("LOGS", "METRICS", "TRACES")}


@pytest.mark.parametrize("name", policy.SIGNAL_OTLP_ENV)
@pytest.mark.parametrize("endpoint", ["", "http://127.0.0.1:4318"])
def test_every_role_refuses_a_per_signal_endpoint_whatever_the_policy(isolated_policy, monkeypatch,
                                                                      name, endpoint):
    monkeypatch.setenv(name, "http://127.0.0.1:4318/v1/x")
    diagnostics = policy.DiagnosticsPolicy(otlp_endpoint=endpoint, otlp_logs=bool(endpoint))
    configured = policy.MonarchPolicy(diagnostics=diagnostics)
    with pytest.raises(ClusterConfigError, match=name):
        policy.apply_policy(configured, role="worker")
    assert policy._ACTIVE is None
    assert start_refusal(diagnostics) == "ambient_otlp_signal_endpoint"


def test_the_drivers_refusal_points_at_its_troubleshooting_entry():
    """prepare_driver runs start_refusal first, so this is the message a user sees."""
    with pytest.raises(ClusterConfigError) as refused:
        monarch_launch._refuse_export("ambient_otlp_signal_endpoint")
    assert "(ambient_otlp_signal_endpoint)" in str(refused.value)
    assert str(refused.value).endswith(f"({policy.AMBIENT_TROUBLESHOOTING})")


@pytest.mark.parametrize("name", policy.SIGNAL_OTLP_ENV)
def test_an_empty_per_signal_variable_is_no_destination(isolated_policy, monkeypatch, name):
    monkeypatch.setenv(name, "")
    policy.apply_policy(policy.MonarchPolicy())
    assert start_refusal(policy.DiagnosticsPolicy()) is None


@pytest.mark.parametrize("endpoint", ["", "http://127.0.0.1:4318"])
def test_actors_start_without_the_per_signal_variables(endpoint):
    diagnostics = policy.DiagnosticsPolicy(otlp_endpoint=endpoint, otlp_logs=bool(endpoint))
    config = SimpleNamespace(monarch=policy.MonarchPolicy(diagnostics=diagnostics),
                             rdma_latent_return=False)
    environment = monarch_launch.actor_environment(config, "")
    assert all(name in environment and environment[name] is None for name in policy.SIGNAL_OTLP_ENV)


class _Sink(BaseHTTPRequestHandler):
    posts: list[str] = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        _Sink.posts.append(self.path)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args):
        return None


def test_nothing_is_posted_with_export_off_and_a_per_signal_endpoint_set(tmp_path):
    pytest.importorskip("monarch.actor")
    _Sink.posts = []
    server = HTTPServer(("127.0.0.1", 0), _Sink)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("MONARCH_", "HYPERACTOR_", "OTEL_", policy.POLICY_ENV))}
    environment.update(
        HOME=str(tmp_path), XDG_RUNTIME_DIR=str(runtime_dir), CUDA_VISIBLE_DEVICES="",
        PYTHONPATH=os.pathsep.join(filter(None, (str(SRC), os.environ.get("PYTHONPATH")))),
        OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=f"http://127.0.0.1:{server.server_port}/v1/traces")
    try:
        proc = subprocess.run([sys.executable, str(PROBE)], capture_output=True, text=True,
                              timeout=240, env=environment)
    finally:
        server.shutdown()
        server.server_close()
    assert proc.returncode == 0, f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert _Sink.posts == [], (_Sink.posts, result)
    assert result["admitted"] is False, result
    assert "ambient_otlp_signal_endpoint" in result["refusal"], result
    assert result["monarch_loaded"] is False, result
