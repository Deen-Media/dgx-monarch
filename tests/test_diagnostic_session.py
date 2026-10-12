"""Finite captures remain advisory, private, sanitized and explicit about gaps."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from dgx_monarch import diagnostic_session as sessions
from dgx_monarch import runtime_diagnostics
from dgx_monarch.cli import diagnostics_command


def _payload(capture, *, sequence=1, worker=False, private_text="private prompt"):
    identity = {"source_manifest_sha256": "a" * 64, "transport_sha256": "b" * 64,
                "torchmonarch": "0.7.0", "torch": "2.12.0", "dgx_monarch": "1.2.0",
                "path": private_text}
    block = {"schema": 1, "available": True, "sequence": sequence, "process_nonce": "c" * 32, "recent": [{
        "sequence": sequence, "phase": "collect", "outcome": "ok", "return_mode": "message",
        "duration_s": 0.02, "finished_at": capture.started_at + 0.01, "token": private_text}]}
    payload = {"diagnostics": block, "diagnostic_identity": identity, "prompt": private_text}
    if worker:
        payload["workers"] = [{"diagnostics": block, "diagnostic_identity": identity,
                               "rank": 0, "world": 1, "nccl": True, "models": [private_text]}]
    return payload


def test_capture_sanitizes_all_streams_and_keeps_runtime_identity(tmp_path):
    capture = sessions.Capture()
    capture.ingest(_payload(capture, worker=True))
    path = capture.export(tmp_path / "capture.json")
    content = path.read_text()
    assert "private prompt" not in content
    assert os.stat(path).st_mode & 0o777 == 0o600
    report = json.loads(content)
    assert len(report["traceEvents"]) == 2
    assert report["identities"]["worker-0"]["torchmonarch"] == "0.7.0"
    assert report["topology"] == [{"participant": "worker-0", "rank": 0, "world": 1, "nccl": True}]
    assert report["capture"]["partial"] is False


def test_partial_start_worker_loss_and_stale_snapshots_are_preserved():
    capture = sessions.Capture()
    capture.ingest({"telemetry_error": "private network error"})
    capture.ingest(_payload(capture, worker=True))
    capture.ingest(_payload(capture, sequence=3))
    report = capture.report()
    assert report["capture"]["partial"] is True
    assert report["capture"]["gaps"] == {"stale_snapshot": 1, "ring_evicted": 1, "collector_missing": 1}
    assert len(report["traceEvents"]) == 3


@pytest.mark.parametrize("marker", ["status_poll", "status_error", "telemetry_error"])
def test_stale_workers_leave_valid_driver_events_and_explicit_coverage_gap(marker):
    capture = sessions.Capture()
    payload = _payload(capture, worker=True)
    if marker == "status_poll":
        payload["status_poll"] = {"fresh": False}
    else:
        payload["workers"][0][marker] = "failed"
    capture.ingest(payload)
    report = capture.report()
    assert [event["pid"] for event in report["traceEvents"]] == ["driver"]
    assert report["topology"] == []
    assert report["capture"]["gaps"]["stale_worker"] == 1
    assert report["capture"]["partial"] is True


def test_event_limit_retains_partial_capture(tmp_path):
    capture = sessions.Capture(max_events=1)
    capture.ingest(_payload(capture, worker=True))
    assert capture.done
    assert capture.report()["capture"]["stop_reason"] == "event_limit"
    assert len(capture.report()["traceEvents"]) == 1
    assert capture.export(tmp_path / "limited.json").stat().st_size <= capture.max_bytes


def test_missing_identity_is_partial_not_a_runtime_attestation():
    capture = sessions.Capture()
    payload = _payload(capture)
    payload.pop("diagnostic_identity")
    capture.ingest(payload)
    assert capture.report()["capture"]["identity_coverage"] == "partial"
    assert capture.report()["capture"]["partial"] is True


def test_old_events_and_repeated_snapshots_are_not_replayed():
    capture = sessions.Capture()
    payload = _payload(capture)
    payload["diagnostics"]["recent"][0]["finished_at"] = capture.started_at - 1
    capture.ingest(payload)
    capture.ingest(_payload(capture, sequence=2))
    capture.ingest(_payload(capture, sequence=2))
    assert len(capture.report()["traceEvents"]) == 1


def test_same_version_new_process_does_not_reuse_old_cursor():
    capture = sessions.Capture()
    capture.ingest(_payload(capture, sequence=10))
    replacement = _payload(capture, sequence=10)
    replacement["diagnostics"]["process_nonce"] = "d" * 32
    capture.ingest(replacement)
    assert len(capture.report()["traceEvents"]) == 2
    assert capture.report()["capture"]["gaps"]["identity_changed"] == 1


@pytest.mark.parametrize("options", [
    {"duration_s": float("nan")}, {"duration_s": 301}, {"max_events": True},
    {"max_events": 20001}, {"max_bytes": 100}, {"max_bytes": 9_000_000},
])
def test_invalid_budgets_reject_before_collection(options):
    with pytest.raises(ValueError):
        sessions.Capture(**options)


def test_deadline_and_fetch_failure_leave_no_collectors(monkeypatch):
    monkeypatch.setitem(sys.modules, "monarch.actor", None)
    timeouts = []

    def failed(timeout):
        timeouts.append(timeout)
        raise ConnectionError("private endpoint")

    before = time.monotonic()
    capture = sessions.collect(failed, duration_s=0.1)
    assert time.monotonic() - before < 1
    assert capture.done and capture.report()["capture"]["gaps"] == {"fetch_failed": 1}
    assert 0 < timeouts[0] <= 0.1


def test_export_refuses_overwrite_and_symlink(tmp_path):
    capture = sessions.Capture()
    existing = tmp_path / "existing.json"
    existing.write_text("keep")
    with pytest.raises(FileExistsError):
        capture.export(existing)
    assert existing.read_text() == "keep"
    link = tmp_path / "link.json"
    link.symlink_to(existing)
    with pytest.raises(FileExistsError):
        capture.export(link)
    assert existing.read_text() == "keep"


def test_export_failure_cleans_owned_artifact_without_affecting_instrumentation(monkeypatch, tmp_path):
    capture = sessions.Capture()
    capture.ingest(_payload(capture))

    def fail(_fd):
        raise OSError("disk full")

    monkeypatch.setattr(sessions.os, "fsync", fail)
    output = tmp_path / "failed.json"
    with pytest.raises(OSError):
        capture.export(output)
    assert not output.exists()
    with runtime_diagnostics.phase("compute"):
        completed = True
    assert completed


def test_capabilities_do_not_claim_clean_admin_ownership():
    result = sessions.capabilities()
    for key in ("attached_admin", "distributed_sql", "owned_process_stacks"):
        assert result[key]["available"] is False
        assert "no_stop_or_partial_start_recovery" in result[key]["reason"]


def test_capabilities_cli_never_resolves_or_connects(capsys):
    args = SimpleNamespace(capabilities=True)
    assert diagnostics_command.run(args, resolve_host=lambda _: pytest.fail("must not resolve")) == 0
    assert json.loads(capsys.readouterr().out)["phase_capture"]["available"] is True


def test_large_capture_never_exceeds_artifact_byte_budget(tmp_path):
    capture = sessions.Capture(max_bytes=65536)
    for sequence in range(1, 500):
        capture.ingest(_payload(capture, sequence=sequence))
        if capture.done:
            break
    artifact = capture.export(tmp_path / "bounded.json")
    assert artifact.stat().st_size <= 65536
    result = json.loads(artifact.read_text())
    assert result["capture"]["partial"] is True
    assert result["capture"]["stop_reason"] == "byte_limit"


def test_http_fetch_reads_only_fixed_endpoint_and_rejects_oversize(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            content = json.dumps({"private": "x" * 100}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = f"127.0.0.1:{server.server_port}"
    try:
        assert diagnostics_command.fetch_telemetry(host, 1) == {"private": "x" * 100}
        monkeypatch.setattr(diagnostics_command, "_MAX_RESPONSE_BYTES", 32)
        with pytest.raises(ValueError, match="input limit"):
            diagnostics_command.fetch_telemetry(host, 1)
        assert requests == ["/dgxm/telemetry", "/dgxm/telemetry"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
