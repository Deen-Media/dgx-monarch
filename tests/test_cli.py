"""dgxm CLI surface: help, version, config plumbing (no cluster needed)."""
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import TORCHMONARCH_PIN, __version__
from dgx_monarch.cli import main as cli
from dgx_monarch.cli.main import _find_gate_report, _prompt_input, main
from dgx_monarch.config import ClusterConfig, HostConfig


def test_version(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_doctor_runs_without_config(capsys, monkeypatch, tmp_path):
    # No cluster.toml anywhere: driver-only checks must run and not crash.
    monkeypatch.delenv("DGXM_CLUSTER_TOML", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert "driver checks" in out
    assert rc in (0, 1)  # environment-dependent, but it must complete


def test_doctor_repair_is_allowlisted_and_reruns_doctor(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n")
    config_path.chmod(0o644)
    monkeypatch.setattr(
        cli,
        "_load_config",
        lambda _args: ClusterConfig(source=str(config_path)),
    )
    ran: list[bool] = []
    monkeypatch.setattr(
        cli.doctor_mod,
        "doctor_exit",
        lambda _config, as_json=False: ran.append(as_json) or 0,
    )
    receipt = tmp_path / "repair.json"
    assert main(["doctor", "--repair", "--yes", "--receipt", str(receipt)]) == 0
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert receipt.is_file()
    assert ran == [False]
    assert "no services, processes, network" in capsys.readouterr().out


def test_doctor_repair_refuses_json_and_configless_modes(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_load_config", lambda _args: None)
    assert main(["doctor", "--repair"]) == 2
    assert main(["doctor", "--repair", "--json"]) == 2
    assert "cannot be combined" in capsys.readouterr().err


def test_explicit_missing_config_is_a_hard_error(capsys):
    # A typo'd --config must not silently degrade to driver-only checks.
    try:
        rc = main(["--config", "/nonexistent/cluster.toml", "doctor"])
    except SystemExit as e:
        rc = e.code
    assert rc == 2
    assert "does not exist" in capsys.readouterr().err


def test_lifecycle_commands_require_config(capsys, monkeypatch, tmp_path):
    monkeypatch.delenv("DGXM_CLUSTER_TOML", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    for cmd in ("up", "down", "status", "restart"):
        try:
            rc = main([cmd])
        except SystemExit as e:
            rc = e.code
        assert rc == 2


def test_module_entrypoint():
    for module in ("dgx_monarch.cli.main", "dgx_monarch"):
        result = subprocess.run(
            [sys.executable, "-m", module, "version"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, f"-m {module}: {result.stderr}"
        assert __version__ in result.stdout


def test_gate_report_is_correlated_by_run_id(tmp_path):
    report = tmp_path / "dgxm_gate_reports.jsonl"
    report.write_text(
        '{"run_id":"wanted","verdict":"PASS"}\n'
        'not-json\n'
        '{"run_id":"other","verdict":"FAIL"}\n'
    )
    assert _find_gate_report(str(tmp_path), "wanted")["verdict"] == "PASS"
    assert _find_gate_report(str(tmp_path), "missing") is None


def test_gate_prompt_values_preserve_links_and_zero_literals():
    link = ["node-id", 0]
    assert _prompt_input(link, 42, int) is link
    assert _prompt_input(0, 42, int) == 0
    assert _prompt_input(0.0, 1.0, float) == 0.0


def test_gate_actions_are_explicit_defaulted_and_mutually_exclusive(
    monkeypatch, capsys
):
    seen: list[str] = []
    monkeypatch.setattr(
        cli,
        "cmd_gate",
        lambda args: seen.append(args.gate_action) or 0,
    )

    assert main(["gate"]) == 0
    assert main(["gate", "--last"]) == 0
    assert main(["gate", "--list"]) == 0
    with pytest.raises(SystemExit) as raised:
        main(["gate", "--last", "--list"])

    assert raised.value.code == 2
    assert seen == ["last", "last", "list"]
    capsys.readouterr()


def test_gate_list_uses_the_public_ledger_diagnostic_api(monkeypatch, tmp_path):
    from dgx_monarch import gate_ledger

    calls: list[str] = []
    monkeypatch.setattr(
        gate_ledger.GateLedger,
        "entries",
        lambda self: calls.append(self.path) or [],
    )
    monkeypatch.setattr(
        gate_ledger.GateLedger,
        "_entries",
        lambda _self: pytest.fail("CLI must not call the private compatibility alias"),
    )

    args = SimpleNamespace(
        gate_action="list",
        report_dir=str(tmp_path),
        timeout=1,
    )
    assert cli._cmd_gate_impl(args) == 0
    assert calls and calls[0].endswith("dgxm_gate_ledger.jsonl")


def test_verified_update_default_tracks_the_remote_default_branch(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(
        cli,
        "cmd_update",
        lambda args: seen.append(args.target) or 0,
    )

    assert main(["update"]) == 0
    assert seen == ["origin/HEAD"]


def test_init_propagates_worker_setup_failure(monkeypatch, tmp_path):
    answers = iter([
        "worker", "10.0.0.2", "1", "",  # one host
        "", "", "", "",                 # profile, client, python, ssh key
        "trusted_fabric", "y",            # security acknowledgement, install
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setattr(cli.lifecycle, "install_systemd", lambda _config: False)
    dest = tmp_path / "cluster.toml"
    assert cli.cmd_init(SimpleNamespace(config=str(dest))) == 1
    assert dest.is_file()
    assert dest.stat().st_mode & 0o777 == 0o600


def test_init_does_not_self_attest_an_unconfirmed_trusted_fabric(monkeypatch, tmp_path):
    answers = iter([
        "worker", "10.0.0.2", "1", "",  # one host
        "", "", "", "",                 # profile, client, python, ssh key
        "",                                # no security acknowledgement
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    dest = tmp_path / "cluster.toml"
    assert cli.cmd_init(SimpleNamespace(config=str(dest))) == 2
    assert not dest.exists()


def test_legacy_init_refuses_a_contended_config_destination(monkeypatch, tmp_path):
    answers = iter([
        "worker", "10.0.0.2", "1", "",
        "", "", "", "",
        "trusted_fabric",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setenv("HOME", str(tmp_path))
    dest = tmp_path / "cluster.toml"
    from dgx_monarch.cli.setup_config_lock import SetupConfigLock

    with SetupConfigLock(dest):
        assert cli.cmd_init(SimpleNamespace(config=str(dest))) == 1

    assert not dest.exists()


def test_legacy_init_holds_config_lock_through_optional_lifecycle(monkeypatch, tmp_path):
    answers = iter([
        "worker", "10.0.0.2", "1", "",
        "", "", "", "",
        "trusted_fabric", "n", "",
    ])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    monkeypatch.setenv("HOME", str(tmp_path))
    dest = tmp_path / "cluster.toml"
    from dgx_monarch.cli.setup_config_lock import (
        SetupConfigLock,
        SetupConfigLockUnavailable,
    )

    def assert_locked(_config):
        with pytest.raises(SetupConfigLockUnavailable):
            with SetupConfigLock(dest):
                pass
        return True

    monkeypatch.setattr(cli.lifecycle, "up", assert_locked)
    assert cli.cmd_init(SimpleNamespace(config=str(dest))) == 0


def test_cmd_up_warns_but_proceeds_on_torchmonarch_pin_mismatch(monkeypatch, capsys):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    reported_pin = "unexpected-test-pin"
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    monkeypatch.setattr(cli, "_load_config", lambda _args: config)
    monkeypatch.setattr(cli.lifecycle, "up", lambda _config, sync=True: True)
    monkeypatch.setattr(
        cli.pincheck, "run_on_host",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, f"{reported_pin}\n", ""))

    rc = main(["up"])

    assert rc == 0  # a pin mismatch must never block the recovery command
    err = capsys.readouterr().err
    assert "worker" in err
    assert reported_pin in err
    assert TORCHMONARCH_PIN in err


def test_cmd_restart_silent_when_every_host_matches_the_pin(monkeypatch, capsys):
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    config = ClusterConfig(hosts=(host,), transport_security="trusted_fabric")
    monkeypatch.setattr(cli, "_load_config", lambda _args: config)
    monkeypatch.setattr(cli.lifecycle, "restart", lambda _config, sync=True: True)
    monkeypatch.setattr(
        cli.pincheck, "run_on_host",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, f"{TORCHMONARCH_PIN}\n", ""))

    rc = main(["restart"])

    assert rc == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("command", ["up", "restart"])
def test_recovery_commands_refuse_an_unacknowledged_transport_before_the_pin_probe(
    monkeypatch, capsys, command,
):
    """A start the config refuses must refuse before the pin probe, which costs
    each host a locality lookup and each remote host an SSH command."""
    host = HostConfig(name="worker", address="tcp://10.0.0.2:26600")
    monkeypatch.setattr(cli, "_load_config", lambda _args: ClusterConfig(hosts=(host,)))
    touched: list[str] = []
    monkeypatch.setattr(
        cli.pincheck, "run_on_host",
        lambda *_a, **_k: touched.append("pin probe") or subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(cli.lifecycle, command, lambda *_a, **_k: touched.append(command) or True)

    rc = main([command])

    assert rc == 1
    assert touched == []
    captured = capsys.readouterr()
    assert captured.out.startswith("refusing to start the worker service: peer authentication is unavailable")
    assert captured.err == ""


def test_package_version_has_one_source_of_truth():
    repo = Path(__file__).resolve().parents[1]
    data = tomllib.loads((repo / "pyproject.toml").read_text())
    assert __version__ == "1.0.0"
    assert "version" not in data["project"]
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "dgx_monarch.__version__"
    }
