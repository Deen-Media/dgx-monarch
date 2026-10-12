"""Doctor diagnostics expose fixed categories without affecting updater outcomes."""
from __future__ import annotations

import json
import subprocess
from io import StringIO
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import update_doctor_diagnostics as diagnostics
from dgx_monarch.cli.json_report import doctor_payload
from dgx_monarch.cli.update_command import SystemUpdateOps
from dgx_monarch.cli.update_worker_lifecycle import operation_from_exit


def report(rows):
    return json.dumps({"checks": rows, "summary": {"failures": 1}})


def test_private_host_names_and_probe_details_never_reach_diagnostics(capsys):
    private = "private-node /private/path credential=secret 192.0.2.4"
    diagnostics.report_doctor_result(1, report([
        {"check": private + " env", "state": "fail", "detail": private},
        {"check": private + " worker service", "state": "warn", "detail": private},
        {"check": private, "state": "fail", "detail": private},
        {"check": "torchmonarch", "state": "ok", "detail": private},
    ]), (private,))
    output = capsys.readouterr()
    assert not output.out and private not in output.err
    assert set(output.err.splitlines()) == {
        "doctor diagnostics: worker_environment fail (1)",
        "doctor diagnostics: worker_service warn (1)",
        "doctor diagnostics: other fail (1)",
    }


@pytest.mark.parametrize("payload", [None, "secret raw traceback", "x" * (128 * 1024 + 1),
    "[", "[]", '{"checks":{}}', '{"checks":[null]}',
    '{"checks":[{"check":"torch CUDA","state":"secret"}]}',
    json.dumps({"checks": [{}] * 257}), "[" * 10000])
def test_invalid_or_excessive_report_is_fixed_unavailable(payload, capsys):
    diagnostics.report_doctor_result(75, payload, ())
    assert capsys.readouterr().err == "doctor diagnostics: unavailable\n"


def test_many_rows_aggregate_and_output_is_bounded(capsys):
    checks = [{"check": "torch CUDA", "state": "fail", "detail": "private"}] * 256
    diagnostics.report_doctor_result(1, report(checks), ())
    assert capsys.readouterr().err == "doctor diagnostics: torch_cuda fail (256)\n"
    checks = [{"check": name, "state": state} for name in diagnostics._CHECKS
              for state in ("fail", "warn", "unknown")]
    diagnostics.report_doctor_result(1, report(checks), ())
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 21 and lines[-1].endswith("additional categories omitted")
    assert all(" fail " in line for line in lines[:16])


@pytest.mark.parametrize("returncode", [0, 1, 2, 75])
def test_actual_doctor_call_reports_categories_and_preserves_certainty(tmp_path, monkeypatch, capsys, returncode):
    ops = object.__new__(SystemUpdateOps)
    ops.config = SimpleNamespace(hosts=[SimpleNamespace(name="secret-host")])
    ops.repo = tmp_path
    ops._config_path = tmp_path / "private.toml"
    monkeypatch.setattr(ops, "_config_current", lambda: True)
    monkeypatch.setattr(ops, "_target_site", lambda: tmp_path)
    calls = []
    payload = json.dumps(doctor_payload([
        {"name": "secret-host env", "status": "FAIL", "detail": "secret path"},
    ], 1))

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, returncode, payload, "secret traceback")

    ops._run = runner
    assert ops.doctor() == operation_from_exit(returncode)
    output = capsys.readouterr()
    assert output.err == ("" if returncode == 0 else "doctor diagnostics: worker_environment fail (1)\n")
    assert len(calls) == 1 and calls[0][1]["env"]["PYTHONPATH"] == str(tmp_path)


@pytest.mark.parametrize("error", [BrokenPipeError, ValueError])
def test_diagnostic_stream_failure_does_not_escape(monkeypatch, error):
    class BrokenStream(StringIO):
        def write(self, text):
            raise error("private pipe")

    monkeypatch.setattr(diagnostics.sys, "stderr", BrokenStream())
    diagnostics.report_doctor_result(75, "bad", ())
