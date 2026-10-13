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


class _Clock:
    """Deterministic stand-in for the module's time: fetches and sleeps advance it."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return 1_000_000.0 + self.now

    def sleep(self, seconds):
        self.now += max(0.0, seconds) + 1e-9


def _healthy(clock, costs, payload, timeouts):
    """An endpoint that honors its deadline: a fetch longer than it times out."""
    calls = iter(costs)

    def fetch(timeout):
        timeouts.append(timeout)
        cost = next(calls)
        if cost > timeout:
            clock.now += timeout
            raise TimeoutError("fetch exceeded its deadline")
        clock.now += cost
        return payload
    return fetch


def _steady_payload():
    identity = {"source_manifest_sha256": "a" * 64, "transport_sha256": "b" * 64,
                "torchmonarch": "0.7.0", "dgx_monarch": "1.2.0"}
    return {"diagnostics": {"schema": 1, "available": True, "sequence": 0,
                            "process_nonce": "c" * 32, "recent": []},
            "diagnostic_identity": identity}


@pytest.mark.parametrize("costs", [
    pytest.param(lambda: iter(lambda: 0.2, None), id="steady-0.2s"),
    pytest.param(lambda: (0.2 if index % 2 == 0 else 0.01 for index in range(10_000)),
                 id="alternating-cache"),
])
def test_partial_reflects_coverage_not_where_the_deadline_falls(monkeypatch, costs):
    clock = _Clock()
    monkeypatch.setattr(sessions, "time", clock)
    outcomes = []
    for step in range(20):
        clock.now = 0.0
        timeouts = []
        duration = round(1.0 + 0.07 * step, 2)
        capture = sessions.collect(
            _healthy(clock, costs(), _steady_payload(), timeouts), duration_s=duration)
        report = capture.report()["capture"]
        outcomes.append((duration, report["partial"], report["gaps"], report["stop_reason"]))
        # Only the first fetch, with no cost observed yet, may run in a short window.
        assert all(timeout >= 0.2 for timeout in timeouts[1:]), (duration, timeouts)
    assert all(partial is False and gaps == {} and stop == "deadline"
               for _duration, partial, gaps, stop in outcomes), outcomes


def test_a_failing_endpoint_still_records_fetch_failures(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(sessions, "time", clock)

    def refused(_timeout):
        clock.now += 0.001
        raise ConnectionRefusedError("private endpoint")

    capture = sessions.collect(refused, duration_s=1.3)
    report = capture.report()["capture"]
    assert report["gaps"]["fetch_failed"] >= 2
    assert report["partial"] is True and report["stop_reason"] == "deadline"


def test_a_slow_endpoint_keeps_full_windows_and_skips_only_the_short_tail(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(sessions, "time", clock)
    timeouts = []
    capture = sessions.collect(
        _healthy(clock, iter(lambda: 1.0, None), _steady_payload(), timeouts), duration_s=3.8)
    # Fetches run 0 to 1 s and 1.5 to 2.5 s; the 0.8 s window left at 3.0 s
    # cannot hold a 1 s fetch, so it is slept rather than recorded as a failure.
    assert timeouts == [2.0, 2.0]
    report = capture.report()["capture"]
    assert report["gaps"] == {} and report["stop_reason"] == "deadline"
    clock.now, timeouts[:] = 0.0, []
    capture = sessions.collect(
        _healthy(clock, iter(lambda: 1.0, None), _steady_payload(), timeouts), duration_s=4.4)
    # A final window that can hold the slowest fetch is still polled.
    assert timeouts == [2.0, 2.0, pytest.approx(1.4, abs=1e-6)]
    assert capture.report()["capture"]["gaps"] == {}


def test_a_tail_shorter_than_the_slowest_fetch_goes_unpolled_and_unflagged(monkeypatch):
    """rev-pr-c nit 6 records the current trade: no gap, but up to fetch + poll unpolled."""
    clock = _Clock()
    monkeypatch.setattr(sessions, "time", clock)
    timeouts, ended = [], []

    def fetch(timeout):
        result = _healthy(clock, iter([1.9]), _steady_payload(), timeouts)(timeout)
        ended.append(clock.now)
        return result

    report = sessions.collect(fetch, duration_s=4.2).report()["capture"]
    # One fetch from 0 to 1.9 s, a 0.5 s poll sleep, then the 1.8 s window is slept.
    assert timeouts == [2.0] and report["samples"] == 1
    assert report["partial"] is False and report["gaps"] == {} and report["stop_reason"] == "deadline"
    assert 4.2 - ended[-1] == pytest.approx(2.3)


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


def _capture_args(output, **overrides):
    values = {"capabilities": False, "output": str(output), "host": "127.0.0.1:9",
              "duration_s": 0.1, "max_events": 4096, "max_bytes": 1_048_576}
    values.update(overrides)
    return SimpleNamespace(**values)


def _never_resolve(_host):
    pytest.fail("the destination must be refused before any capture starts")


def test_cli_refuses_an_existing_output_before_capturing(tmp_path, capsys):
    existing = tmp_path / "existing.json"
    existing.write_text("keep")
    assert diagnostics_command.run(_capture_args(existing), resolve_host=_never_resolve) == 2
    err = capsys.readouterr().err
    assert "--output already exists" in err and str(tmp_path) not in err
    assert existing.read_text() == "keep"
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent.json")
    assert diagnostics_command.run(_capture_args(dangling), resolve_host=_never_resolve) == 2
    assert "--output already exists" in capsys.readouterr().err


@pytest.mark.parametrize("layout", ["missing_directory", "symlink_parent", "file_parent",
                                    "symlink_loop", "unknown_user"])
def test_cli_refuses_an_unusable_directory_with_the_fixed_reason(tmp_path, capsys, layout):
    real = tmp_path / "real"
    real.mkdir()
    if layout == "missing_directory":
        output = tmp_path / "absent" / "capture.json"
    elif layout == "symlink_parent":
        (tmp_path / "link").symlink_to(real)
        output = tmp_path / "link" / "capture.json"
    elif layout == "symlink_loop":
        # rev-pr-c 4: resolve() raises RuntimeError naming the loop's path.
        (tmp_path / "loopa").symlink_to(tmp_path / "loopb")
        (tmp_path / "loopb").symlink_to(tmp_path / "loopa")
        output = tmp_path / "loopa" / "capture.json"
    elif layout == "unknown_user":
        output = "~dgxm-no-such-user-for-tests/capture.json"
    else:
        (tmp_path / "file").write_text("")
        output = tmp_path / "file" / "capture.json"
    with pytest.raises(ValueError, match=r"^capture output requires an existing directory"):
        sessions.validate_destination(output)
    assert diagnostics_command.run(_capture_args(output), resolve_host=_never_resolve) == 2
    err = capsys.readouterr().err
    assert "capture output requires an existing directory without symbolic links" in err
    assert str(tmp_path) not in err and "no-such-user" not in err
    assert list(real.iterdir()) == []


def test_cli_prints_fixed_budget_refusals_and_hides_other_error_text(tmp_path, capsys):
    output = tmp_path / "capture.json"
    args = _capture_args(output, duration_s=301)
    assert diagnostics_command.run(args, resolve_host=lambda host: host) == 2
    assert "(capture duration must be between 0.1 and 300 seconds)" in capsys.readouterr().err

    def hostile(_host):
        raise ValueError("private host detail")

    assert diagnostics_command.run(_capture_args(output), resolve_host=hostile) == 2
    err = capsys.readouterr().err
    assert "(ValueError)" in err and "private" not in err
    assert not output.exists()


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
