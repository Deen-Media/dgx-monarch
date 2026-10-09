"""The `NCCL library` row: the loaded library per host, compared across hosts.

torch reports the NCCL it was built against, so the row reads the library the
host process loaded. Every case is built from the probe's own spellings
(`23007`, a 12 character hash prefix, `unknown`). Unknown stays distinct from
equal and from different: a host that did not report never reads as a match.
"""
from __future__ import annotations

import subprocess
import sys

from dgx_monarch.cli import doctor, doctor_nccl
from dgx_monarch.cli.probe_certainty import FAIL, OK, WARN, probe_row
from dgx_monarch.config import ClusterConfig, HostConfig
from test_doctor_hostname_row import _REMOTE_OK, _stub_driver

HASH_A = "a1b2c3d4e5f6"
HASH_B = "90013adbc408"
NAMES = ["head", "worker"]


def _host(code: str = "23007", sha: str = HASH_A) -> dict[str, str]:
    return {"nccl_lib": code, "nccl_sha": sha}


def _row(per_host, names=NAMES) -> dict:
    return probe_row(*doctor_nccl.doctor_row(per_host, names))


def test_equal_versions_and_files_are_ok_and_show_version_and_hash():
    row = _row({"head": _host(), "worker": _host()})
    assert row["status"] == OK and row["name"] == "NCCL library"
    assert row["detail"].count(f"2.30.7 sha256 {HASH_A}") == 2
    assert "reason" not in row


def test_a_single_host_reports_ok_with_its_own_value():
    row = _row({"head": _host("23203", HASH_B)}, ["head"])
    assert row["status"] == OK
    assert f"head 2.32.3 sha256 {HASH_B}" in row["detail"]
    assert _row({"head": _host(sha="unknown")}, ["head"])["status"] == OK


def test_different_versions_fail_and_name_both():
    row = _row({"head": _host("23007"), "worker": _host("23203", HASH_B)})
    assert row["status"] == FAIL
    assert "head 2.30.7" in row["detail"] and "worker 2.32.3" in row["detail"]


def test_a_version_mismatch_fails_even_when_a_third_host_is_silent():
    names = ["head", "worker", "spare"]
    row = _row({"head": _host("23007"), "worker": _host("23203", HASH_B)}, names)
    assert row["status"] == FAIL


def test_equal_versions_from_different_files_warn():
    row = _row({"head": _host(), "worker": _host(sha=HASH_B)})
    assert row["status"] == WARN
    assert HASH_A in row["detail"] and HASH_B in row["detail"]
    assert "reason" not in row


def test_a_host_that_did_not_report_warns_as_unobserved():
    for per_host in ({"head": _host()},
                     {"head": _host(), "worker": {}},
                     {"head": _host(), "worker": _host("unknown", "unknown")},
                     {"head": _host(), "worker": _host("0", "unknown")},
                     {}):
        row = _row(per_host)
        assert row["status"] == WARN, per_host
        assert row["reason"] == "unobserved", per_host
        assert "critical" not in row, per_host


def test_an_unhashed_host_cannot_confirm_a_match_but_a_lone_host_needs_none():
    row = _row({"head": _host(), "worker": _host(sha="unknown")})
    assert row["status"] == WARN and row["reason"] == "unobserved"
    assert "worker" in row["detail"]
    assert _row({"head": _host(sha="unknown")}, ["head"])["status"] == OK


def test_a_silent_host_does_not_hide_a_hash_difference():
    names = ["head", "worker", "spare"]
    row = _row({"head": _host(), "worker": _host(sha=HASH_B)}, names)
    assert row["status"] == WARN
    assert "no report from spare" in row["detail"]


def test_an_odd_version_code_formats_as_major_minor_patch():
    assert doctor_nccl._version("22809") == "2.28.9"
    assert doctor_nccl._version("30010") == "3.0.10"
    assert doctor_nccl._version("unknown") == ""
    assert doctor_nccl._version("-5") == ""


def test_no_input_marks_the_row_critical():
    """No reading marks the row critical, so an unobserved NCCL row never sets the unknown exit code."""
    for per_host in ({}, {"head": _host()}, {"head": _host(), "worker": _host(sha=HASH_B)}):
        assert "critical" not in _row(per_host)


def test_fields_reads_only_this_probes_line():
    lines = ["torchmonarch=1 torch=2 nccl_proto=unset", f"nccl_lib=23007 nccl_sha={HASH_A}",
             "sol=True"]
    assert doctor_nccl.fields(lines) == _host()
    assert doctor_nccl.fields(lines[:1]) == {}


def test_the_probe_prints_the_line_doctor_selects_and_the_row_reads_it():
    """Run the payload as doctor does: the mechanics, not the answers."""
    result = subprocess.run([sys.executable, "-"], input=doctor_nccl.PROBE_SOURCE,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    printed = result.stdout.strip().splitlines()
    assert len(printed) == 1 and printed[0].startswith(doctor_nccl.LINE_PREFIX)
    found = doctor_nccl.fields(printed)
    assert set(found) == {"nccl_lib", "nccl_sha"}
    assert _row({"head": found}, ["head"])["status"] in (OK, WARN)
    assert "\nEOF" not in doctor_nccl.PROBE_SOURCE


def test_proto_helpers_keep_their_spelling():
    assert doctor_nccl.normalize_proto(" simple, ll128 ") == "SIMPLE,LL128"
    assert doctor_nccl.uses_ll(" simple, Ll ")
    assert not doctor_nccl.uses_ll("LL128,SIMPLE")


def _cluster() -> ClusterConfig:
    return ClusterConfig(
        hosts=(HostConfig(name="head", address="tcp://10.0.0.2:26600"),
               HostConfig(name="worker", address="tcp://10.0.0.3:26600")),
        client_bind="tcp://10.0.0.1:0", fabric_profile="generic-roce",
        transport_security="trusted_fabric", source="test.toml")


def _run_doctor(monkeypatch, capsys, remote_by_host) -> tuple[bool, str, list[str]]:
    _stub_driver(monkeypatch, "10.0.0.1")
    monkeypatch.setattr(doctor.shutil, "which", lambda _name: "/usr/bin/rsync")
    monkeypatch.setattr(doctor, "_tcp_probe", lambda *_a, **_kw: False)
    monkeypatch.setattr(doctor, "passive_worker_health",
                        lambda *_a, **_kw: {"running": True, "listening": True, "healthy": True})
    scripts: list[str] = []

    def fake(_config, host, script, **_kw):
        scripts.append(script)
        return subprocess.CompletedProcess([], 0, remote_by_host[host.name], "")

    monkeypatch.setattr(doctor, "run_on_host", fake)
    ok = doctor.run_doctor(_cluster())
    return ok, capsys.readouterr().out, scripts


def test_doctor_sends_the_probe_and_fails_on_a_version_split(monkeypatch, capsys):
    remote = {"head": _REMOTE_OK + f"nccl_lib=23007 nccl_sha={HASH_A}\n",
              "worker": _REMOTE_OK + f"nccl_lib=23203 nccl_sha={HASH_B}\n"}
    ok, output, scripts = _run_doctor(monkeypatch, capsys, remote)
    assert ok is False
    assert "[FAIL] NCCL library" in output
    assert all(doctor_nccl.PROBE_SOURCE in script for script in scripts)


def test_doctor_is_ok_when_both_hosts_load_one_library(monkeypatch, capsys):
    line = f"nccl_lib=23007 nccl_sha={HASH_A}\n"
    ok, output, _ = _run_doctor(monkeypatch, capsys, {"head": _REMOTE_OK + line,
                                                      "worker": _REMOTE_OK + line})
    assert ok is True
    assert f"[ ok ] NCCL library: head 2.30.7 sha256 {HASH_A}; worker 2.30.7 sha256 {HASH_A}" in output


def test_doctor_warns_when_a_host_prints_no_nccl_line(monkeypatch, capsys):
    remote = {"head": _REMOTE_OK + f"nccl_lib=23007 nccl_sha={HASH_A}\n", "worker": _REMOTE_OK}
    ok, output, _ = _run_doctor(monkeypatch, capsys, remote)
    assert ok is True
    assert "[WARN] NCCL library: unobserved" in output
