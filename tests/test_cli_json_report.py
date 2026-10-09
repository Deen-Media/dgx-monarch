"""Check doctor/status JSON schemas, exit codes, and unchanged plain-text output.

CI smoke checks consume doctor's prose, while ``dgxm update --verify`` reads
its JSON and exit code. Both interfaces must remain consistent.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from dgx_monarch.cli import doctor, json_report, mesh_health_row
from dgx_monarch.cli import main as cli
from dgx_monarch.config import ClusterConfig

ROWS = [
    {"status": doctor._OK, "name": "torch CUDA", "detail": "2.test, 1 device(s)"},
    {"status": doctor._WARN, "name": "memory headroom", "detail": "40/120 GiB available"},
    {"status": doctor._FAIL, "name": "client_bind", "detail": "missing"},
]

HOSTS = [
    {"host": "worker", "loop": "running", "port": "open",
     "address": "10.0.0.2:29500", "mode": "systemd"},
]


def _rows_with_prose(rows):
    """A stand-in for the real checks: same return, same printing habit."""

    def collect(_config):
        for row in rows:
            print(f"  [{row['status']:^4}] {row['name']}: {row['detail']}")
        print(f"doctor: {len(rows)} checks, "
              f"{len(doctor._failures(rows))} failures")
        return rows

    return collect


def test_doctor_json_is_one_object_per_row_plus_the_summary(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(ROWS))
    assert doctor.run_doctor(None, as_json=True) is False
    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"] == [
        {"check": "torch CUDA", "state": "ok", "detail": "2.test, 1 device(s)"},
        {"check": "memory headroom", "state": "warn",
         "detail": "40/120 GiB available"},
        {"check": "client_bind", "state": "fail", "detail": "missing"},
    ]
    assert payload["summary"] == {"checks": 3, "failures": 1}
    assert payload["readiness"]["overall"] == "blocked"


def test_doctor_json_counts_match_the_rows_a_caller_would_have_parsed(monkeypatch, capsys):
    """The acceptance a script asserts on: N checks, 0 failures."""
    green = [row for row in ROWS if row["status"] != doctor._FAIL]
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(green))
    assert doctor.run_doctor(None, as_json=True) is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"] == {"checks": len(green), "failures": 0}
    assert len(payload["checks"]) == payload["summary"]["checks"]


def test_doctor_json_mode_prints_no_prose(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(ROWS))
    doctor.run_doctor(None, as_json=True)
    out = capsys.readouterr().out
    assert "doctor: 3 checks" not in out
    assert json.loads(out)["summary"]["checks"] == 3


def test_doctor_prose_is_byte_identical_without_the_flag(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(ROWS))
    doctor._doctor_rows(None)
    unwrapped = capsys.readouterr().out
    assert doctor.run_doctor(None) is False
    assert capsys.readouterr().out == unwrapped
    assert unwrapped.endswith("doctor: 3 checks, 1 failures\n")


def test_doctor_exit_code_is_the_same_in_both_modes(monkeypatch, capsys):
    for rows, expected in ((ROWS, False), (ROWS[:2], True)):
        monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(rows))
        assert doctor.run_doctor(None) is expected
        assert doctor.run_doctor(None, as_json=True) is expected
    capsys.readouterr()


def test_the_doctor_subcommand_wires_the_flag_and_defaults_it_off(monkeypatch, capsys):
    seen: list[bool] = []

    def fake_doctor_exit(_config, as_json=False):
        seen.append(as_json)
        return 0

    monkeypatch.setattr(cli, "_load_config", lambda _args: None)
    monkeypatch.setattr(cli.doctor_mod, "doctor_exit", fake_doctor_exit)
    assert cli.main(["doctor"]) == 0
    assert cli.main(["doctor", "--json"]) == 0
    assert seen == [False, True]
    capsys.readouterr()


def test_status_json_carries_the_host_rows_and_the_mesh_row(monkeypatch, capsys):
    monkeypatch.setattr(mesh_health_row.lifecycle, "status", lambda _config: HOSTS)
    monkeypatch.setattr(
        mesh_health_row, "fetch_mesh_block",
        lambda: {"state": "ok", "verdict": "live", "active_leases": 2},
    )
    rows = mesh_health_row.status_with_mesh(ClusterConfig(), as_json=True)
    assert rows == HOSTS  # the exit code still reads the host rows
    payload = json.loads(capsys.readouterr().out)
    assert payload["hosts"] == HOSTS
    assert payload["mesh"] == {"state": "ok",
                               "detail": "ok (cached, 2 active leases)"}
    assert payload["readiness"]["lifecycle"]["worker_service"]["state"] == "ready"
    assert payload["readiness"]["lifecycle"]["attached_mesh"]["state"] == "active"


def test_status_json_names_a_dirty_mesh_without_a_regex(monkeypatch, capsys):
    monkeypatch.setattr(mesh_health_row.lifecycle, "status", lambda _config: [])
    monkeypatch.setattr(
        mesh_health_row, "fetch_mesh_block",
        lambda: {"state": "dirty", "reason": "1 abandoned sample",
                 "remedy": "Reset attached mesh"},
    )
    mesh_health_row.status_with_mesh(ClusterConfig(), as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["hosts"] == []
    assert payload["mesh"]["state"] == "dirty"
    assert "1 abandoned sample" in payload["mesh"]["detail"]
    assert payload["readiness"]["overall"] == "blocked"


def test_status_json_fills_every_host_field_a_probe_left_blank(monkeypatch, capsys):
    monkeypatch.setattr(
        mesh_health_row.lifecycle, "status",
        lambda _config: [{"host": "worker", "loop": "running", "port": "open"}],
    )
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: None)
    mesh_health_row.status_with_mesh(ClusterConfig(), as_json=True)
    payload = json.loads(capsys.readouterr().out)
    assert payload["hosts"] == [{"host": "worker", "loop": "running",
                                 "port": "open", "address": "", "mode": ""}]
    assert payload["mesh"]["state"] == "no-driver"


def test_status_prose_is_byte_identical_without_the_flag(monkeypatch, capsys):
    printed: list[str] = []

    def fake_status(_config):
        print("  worker: worker_service=running listener=open (10.0.0.2:29500, systemd)")
        return HOSTS

    monkeypatch.setattr(mesh_health_row.lifecycle, "status", fake_status)
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: None)
    mesh_health_row.status_with_mesh(ClusterConfig())
    printed.append(capsys.readouterr().out)
    assert printed == [
        "  worker: worker_service=running listener=open (10.0.0.2:29500, systemd)\n"
        "  mesh: no driver reachable (no mesh view)\n"
    ]


def test_status_json_mode_prints_no_prose(monkeypatch, capsys):
    def fake_status(_config):
        print("  worker: worker_service=running listener=open (10.0.0.2:29500, systemd)")
        return HOSTS

    monkeypatch.setattr(mesh_health_row.lifecycle, "status", fake_status)
    monkeypatch.setattr(mesh_health_row, "fetch_mesh_block", lambda: None)
    mesh_health_row.status_with_mesh(ClusterConfig(), as_json=True)
    out = capsys.readouterr().out
    assert "worker_service=running" not in out
    assert json.loads(out)["hosts"] == HOSTS


def test_the_status_subcommand_wires_the_flag_and_keeps_its_exit_code(monkeypatch, capsys):
    seen: list[bool] = []

    def fake_status_report(_config, as_json=False):
        seen.append(as_json)
        return ([{"host": "worker", "loop": "stopped", "port": "closed"}], None)

    monkeypatch.setattr(cli, "_require_config", lambda _args: ClusterConfig())
    monkeypatch.setattr(cli.mesh_health_row, "status_report", fake_status_report)
    assert cli.main(["status"]) == 1
    assert cli.main(["status", "--json"]) == 1
    assert seen == [False, True]
    capsys.readouterr()


def test_status_subcommand_fails_when_the_single_mesh_readback_is_dirty(monkeypatch):
    rows = [{"host": "worker", "loop": "running", "port": "open"}]
    dirty = {"state": "dirty", "reason": "teardown failed"}
    monkeypatch.setattr(cli, "_require_config", lambda _args: ClusterConfig())
    monkeypatch.setattr(
        cli.mesh_health_row,
        "status_report",
        lambda _config, as_json=False: (rows, dirty),
    )

    assert cli.main(["status"]) == 1
    assert cli.main(["status", "--json"]) == 1


def test_the_doctor_state_is_a_token_and_not_the_prose_badge():
    """The badge fills `[{status:^4}]`; the token must not depend on that format."""
    badges = (doctor._OK, doctor._WARN, doctor._FAIL)
    assert [json_report.doctor_state(badge) for badge in badges] == [
        "ok", "warn", "fail"]
    # This pins the token for each badge doctor prints, so a rename that changes a
    # token fails here before it reaches callers.
    assert {json_report.doctor_state(badge) for badge in badges} == {
        "ok", "warn", "fail"}
    # Padding and capitals belong to the badge, never to the token.
    assert json_report.doctor_state(f"{doctor._WARN:^4}") == "warn"
    assert json_report.doctor_state("SKIP") == "skip"


def test_the_doctor_payload_publishes_the_token(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(ROWS))
    doctor.run_doctor(None, as_json=True)
    payload = json.loads(capsys.readouterr().out)
    states = [check["state"] for check in payload["checks"]]
    assert states == ["ok", "warn", "fail"]
    assert states == [state.lower() for state in states]


def test_every_mesh_state_gets_a_token_of_its_own():
    assert mesh_health_row.state_name(None) == "no-driver"
    assert mesh_health_row.state_name({}) == "no-mesh-block"
    for state in ("ok", "idle", "busy", "dirty", "unknown"):
        assert mesh_health_row.state_name({"state": state}) == state
    # A driver newer than this CLI: the token is the driver's, not a guess.
    assert mesh_health_row.state_name({"state": "quarantined"}) == "quarantined"
    assert mesh_health_row.state_name({"active_leases": 1}) == "unknown"


def test_the_json_payloads_survive_a_round_trip():
    payload = json_report.doctor_payload(ROWS, 1)
    assert json.loads(json.dumps(payload)) == payload
    status = json_report.status_payload(HOSTS, "ok", "ok (cached, 0 active leases)")
    assert json.loads(json.dumps(status)) == status


BLIND = [
    {"status": doctor._OK, "name": "torch CUDA", "detail": "2.test, 1 device(s)"},
    {"status": doctor._WARN, "name": "worker worker service",
     "detail": "10.0.0.2:29500 process=unknown listener=unknown",
     "reason": "unobserved", "critical": True},
]

# A local probe that could not run. It warns and says so, but the cluster
# checks it sits beside were all observed, so the run still reached a verdict.
ADVISORY = [
    {"status": doctor._OK, "name": "torch CUDA", "detail": "2.test, 1 device(s)"},
    {"status": doctor._WARN, "name": "GPU clock under load",
     "detail": "unobserved: the torch CUDA probe did not complete",
     "reason": "unobserved"},
]


def test_doctor_exits_75_when_its_only_non_ok_rows_are_blind_probes(monkeypatch, capsys):
    """A run that proved nothing is not a run that found something wrong.

    `SystemUpdateOps.doctor` maps this exit to UNKNOWN, and the update state
    machine suppresses rollback on UNKNOWN. Exit 1 for a flaky SSH hop would
    let the update restore the prior release over a healthy target.
    """
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(BLIND))
    assert doctor.doctor_exit(None) == 75
    assert doctor.doctor_exit(None, as_json=True) == 75
    capsys.readouterr()

    # A definite failure alongside the blind row keeps the definite exit.
    monkeypatch.setattr(
        doctor, "_doctor_rows", _rows_with_prose([*BLIND, ROWS[2]]))
    assert doctor.doctor_exit(None) == 1
    assert doctor.doctor_exit(None, as_json=True) == 1
    capsys.readouterr()

    # Observed and clean stays 0; an ordinary warning is not an unknown.
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(ROWS[:2]))
    assert doctor.doctor_exit(None) == 0
    capsys.readouterr()


def test_a_blind_probe_refuses_the_run_doctor_gate_without_claiming_failure(
    monkeypatch, capsys
):
    """Setup gates a mutation on `run_doctor`, so unobserved must refuse there."""
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(BLIND))
    assert doctor.run_doctor(None) is False
    capsys.readouterr()

    payload = json_report.doctor_payload(BLIND, 0)
    assert payload["summary"] == {"checks": 2, "failures": 0}
    assert payload["checks"][1]["state"] == "warn"


def test_the_doctor_command_returns_the_unknown_exit_to_the_shell(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(BLIND))
    monkeypatch.setattr(cli, "_load_config", lambda _args: None)
    args = SimpleNamespace(json=False, repair=False, yes=False, receipt=None)
    assert cli.cmd_doctor(args) == 75
    capsys.readouterr()


def test_an_advisory_probe_that_could_not_run_warns_without_moving_the_exit(
    monkeypatch, capsys
):
    """Both kinds of blind row share one reason token; only one gates the run.

    Doctor's local probes, among them boot records, GPU clock, co-resident
    processes and the orphan scan, say `unobserved:` when their own tool did
    not run. They warn and the exit stays 0; `probe_certainty.exit_code` gives
    the reason.
    """
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose(ADVISORY))
    assert doctor.doctor_exit(None) == 0
    assert doctor.run_doctor(None) is True
    capsys.readouterr()

    # The same row beside a blind cluster probe still exits 75, and a definite
    # failure anywhere still exits 1.
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose([*ADVISORY, *BLIND]))
    assert doctor.doctor_exit(None) == 75
    capsys.readouterr()
    monkeypatch.setattr(doctor, "_doctor_rows", _rows_with_prose([*ADVISORY, ROWS[2]]))
    assert doctor.doctor_exit(None) == 1
    capsys.readouterr()


def test_the_unobserved_prose_convention_stamps_the_machine_field_itself():
    """The `reason` field derives from the `unobserved:` prose (`probe_certainty.probe_row` says why)."""
    row = doctor._row(doctor._WARN, "GPU clock under load",
                      "unobserved: the torch CUDA probe did not complete")
    assert row["reason"] == "unobserved"
    assert "critical" not in row

    gating = doctor._row(doctor._WARN, "w1 env", "unobserved: ssh did not answer",
                         critical=True)
    assert gating["reason"] == "unobserved" and gating["critical"] is True

    plain = doctor._row(doctor._OK, "torch CUDA", "2.test, 1 device(s)")
    assert "reason" not in plain and "critical" not in plain
    # A row that names an observed failure is never tagged, critical or not.
    assert "critical" not in doctor._row(
        doctor._FAIL, "w1 env", "ssh failed", critical=True)
