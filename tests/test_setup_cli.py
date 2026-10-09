"""User-facing contracts for the guided setup CLI adapter."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from dgx_monarch import TORCHMONARCH_PIN
from dgx_monarch.cli import arguments, setup_cli, setup_command
from dgx_monarch.operator_profiles import ProfileRefusal


class FakePlan:
    def __init__(self, diff: str = "", blockers=()) -> None:
        self.diff = diff
        self.blockers = list(blockers)

    def as_dict(self):
        return {
            "host_count": 2,
            "profile": {"name": "safe"},
            "ready": not self.blockers,
            "blockers": self.blockers,
            "warnings": [],
        }


class FakeOutcome:
    def __init__(
        self,
        plan,
        *,
        status="planned",
        failure=None,
        receipt_written=False,
    ) -> None:
        self.plan = plan
        self.status = status
        self.failure = failure
        self.receipt_written = receipt_written

    def as_dict(self):
        return {
            "status": self.status,
            "plan": self.plan.as_dict(),
            "applied": self.status != "planned",
            "config_changed": self.status != "planned",
            "services_changed": False,
            "verified": False,
            "failure": self.failure,
            "receipt": {"schema_version": 1},
            "receipt_written": self.receipt_written,
        }


def _args(**changes) -> Namespace:
    values = {
        "host": ["spark-a,10.0.0.2,1", "spark-b,10.0.0.3,1,alice,/srv/ComfyUI"],
        "client_ip": "10.0.0.1",
        "setup_config_path": "/operator/config/cluster.toml",
        "profile": "safe",
        "fabric_profile": "generic-roce",
        "python_bin": "python3",
        "ssh_key": "",
        "comfy_dir": "/srv/ComfyUI",
        "acknowledge_trusted_fabric": True,
        "artifact": [],
        "apply": False,
        "yes": False,
        "install_service": False,
        "start_worker_service": False,
        "verify": False,
        "privileged_process_inspection": False,
        "receipt": None,
        "json": False,
    }
    values.update(changes)
    return Namespace(**values)


def _wire(monkeypatch, *, plan=None, outcome=None, build_error=None):
    plan = plan or FakePlan()
    outcome = outcome or FakeOutcome(plan)
    calls = []

    def build(request):
        calls.append(("build", request))
        print("backend planning prose")
        if build_error is not None:
            raise build_error
        return plan

    def execute(request, *, plan):
        calls.append(("run", request, plan))
        print("backend execution prose")
        return outcome

    monkeypatch.setattr(setup_cli.setup_command, "build_setup_plan", build)
    monkeypatch.setattr(setup_cli.setup_command, "run_setup", execute)
    return plan, outcome, calls


def test_full_host_spec_is_parsed_without_discovery():
    host = setup_cli.parse_host_spec("spark-a,2001:db8::2,4,alice,/srv/ComfyUI")
    assert host == setup_command.CandidateHost(
        "spark-a", "2001:db8::2", 4, "alice", "/srv/ComfyUI")


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "only-a-name",
        "name,not-an-ip",
        "name,0.0.0.0",
        "name,224.0.0.1",
        # IPv4-mapped literals fail even when the plain form passes (127.0.0.1); see setup_cli._literal_ip.
        "name,::ffff:0.0.0.0",
        "name,::ffff:127.0.0.1",
        "name,::ffff:224.0.0.1",
        "name,10.0.0.2,0",
        "name,10.0.0.2,65",
        "name,10.0.0.2,one",
        "-oProxy,10.0.0.2",
        "name,10.0.0.2,1,-root",
        "name,10.0.0.2,1,user,path,extra",
    ],
)
def test_invalid_host_specs_fail_at_the_input_boundary(spec):
    with pytest.raises(setup_cli.SetupCliInputError):
        setup_cli.parse_host_spec(spec)


def test_hosts_are_collected_from_an_explicit_count_and_fields():
    responses = iter([
        "2",
        "spark-a", "10.0.0.2", "", "", "/srv/a",
        "spark-b", "10.0.0.3", "2", "alice", "",
    ])
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        return next(responses)

    hosts = setup_cli.collect_hosts(None, ask=ask)
    assert hosts == (
        setup_command.CandidateHost("spark-a", "10.0.0.2", 1, "", "/srv/a"),
        setup_command.CandidateHost("spark-b", "10.0.0.3", 2, "alice", ""),
    )
    assert prompts[0].startswith("Host count")
    assert len(prompts) == 11


def test_interactive_inventory_never_calls_a_discovery_helper(monkeypatch):
    # Setup never scans the LAN for hosts: every host is operator input (docs/INSTALL.md).
    assert not hasattr(setup_cli, "discover_hosts")
    responses = iter(["1", "spark", "10.0.0.2", "1", "", ""])
    hosts = setup_cli.collect_hosts([], ask=lambda _prompt: next(responses))
    assert len(hosts) == 1


def test_request_maps_every_namespace_field_and_boolean_acknowledgement(tmp_path: Path):
    receipt = tmp_path / "receipt.json"
    request = setup_cli.request_from_args(_args(
        host=["spark-a,10.0.0.2,2,alice,/comfy/a"],
        setup_config_path=tmp_path / "cluster.toml",
        profile="advanced",
        fabric_profile="generic-ib",
        python_bin="/venv/bin/python",
        ssh_key="/keys/id_ed25519",
        comfy_dir="/comfy/default",
        acknowledge_trusted_fabric=True,
        artifact=["models/a.safetensors", "text/b.safetensors"],
        apply=True,
        yes=True,
        install_service=True,
        start_worker_service=True,
        verify=True,
        privileged_process_inspection=True,
        receipt=receipt,
    ))
    assert request.hosts == (
        setup_command.CandidateHost("spark-a", "10.0.0.2", 2, "alice", "/comfy/a"),)
    assert request.client_ip == "10.0.0.1"
    assert request.config_path == tmp_path / "cluster.toml"
    assert request.profile == "advanced"
    assert request.fabric_profile == "generic-ib"
    assert request.python_bin == "/venv/bin/python"
    assert request.ssh_key == "/keys/id_ed25519"
    assert request.comfy_dir == "/comfy/default"
    assert request.transport_security == "trusted_fabric"
    assert request.artifacts == ("models/a.safetensors", "text/b.safetensors")
    assert request.apply and request.assume_yes and request.install_service
    assert request.start_workers and request.verify
    assert request.privileged_process_inspection is True
    assert request.receipt_path == receipt


def test_output_path_precedes_global_config_and_global_precedes_default():
    output = setup_cli.request_from_args(_args(
        setup_config_path="/operator/output.toml", config="/operator/global.toml"))
    assert output.config_path == Path("/operator/output.toml")

    global_config = setup_cli.request_from_args(_args(
        setup_config_path=None, config="/operator/global.toml"))
    assert global_config.config_path == Path("/operator/global.toml")

    default = setup_cli.request_from_args(_args(setup_config_path=None, config=None))
    assert default.config_path == Path("~/.config/dgx-monarch/cluster.toml")


def test_real_parser_namespace_maps_opt_in_setup_selections():
    names = (
        "version", "doctor", "init", "setup", "up", "down", "reap", "top",
        "gate", "restart", "status", "install_service", "update", "uninstall",
    )
    parser = arguments.build_parser({name: lambda _args: 0 for name in names})
    namespace = parser.parse_args([
        "--config", "/operator/global.toml",
        "setup",
        "--host", "spark-a,10.0.0.2",
        "--client-ip", "10.0.0.1",
        "--acknowledge-trusted-fabric",
        "--privileged-process-inspection",
    ])
    request = setup_cli.request_from_args(namespace)
    assert request.transport_security == "trusted_fabric"
    assert request.privileged_process_inspection is True
    assert request.config_path == Path("/operator/global.toml")

    namespace = parser.parse_args([
        "setup", "--host", "spark-a,10.0.0.2", "--client-ip", "10.0.0.1"])
    default_request = setup_cli.request_from_args(namespace)
    assert default_request.transport_security == ""
    assert default_request.privileged_process_inspection is False


def test_an_unwired_transport_string_cannot_forge_the_acknowledgement():
    request = setup_cli.request_from_args(_args(
        acknowledge_trusted_fabric=False,
        transport_security="trusted_fabric",
    ))
    assert request.transport_security == ""


def test_missing_explicit_client_ip_is_input_error_not_an_interface_scan():
    with pytest.raises(setup_cli.SetupCliInputError):
        setup_cli.request_from_args(_args(client_ip=None))


def test_interactive_inventory_also_asks_for_explicit_client_ip():
    responses = iter(["1", "spark", "10.0.0.2", "1", "", "", "10.0.0.1"])
    request = setup_cli.request_from_args(
        _args(host=None, client_ip=None), ask=lambda _prompt: next(responses))
    assert request.client_ip == "10.0.0.1"


@pytest.mark.parametrize(
    "changes",
    [
        {"profile": "fast"},
        {"acknowledge_trusted_fabric": "yes"},
        {"apply": "false"},
        {"artifact": [object()]},
        {"receipt": "relative.json"},
        {"host": ["same,10.0.0.2", "same,10.0.0.3"]},
    ],
)
def test_namespace_values_are_strict(changes):
    with pytest.raises((setup_cli.SetupCliInputError, ValueError)):
        setup_cli.request_from_args(_args(**changes))


def test_terminal_builds_once_prints_exact_diff_then_applies_same_plan(monkeypatch, capsys):
    diff = "--- current/cluster.toml\n+++ planned/cluster.toml\n+host = secret-spark\n"
    plan = FakePlan(diff)
    outcome = FakeOutcome(plan, status="succeeded", receipt_written=True)
    plan, _outcome, calls = _wire(monkeypatch, plan=plan, outcome=outcome)

    assert setup_cli.run(_args(apply=True, yes=True)) == 0

    assert [call[0] for call in calls] == ["build", "run"]
    assert calls[1][2] is plan
    output = capsys.readouterr().out
    assert output.count("setup plan (sanitized)") == 1
    assert diff in output
    assert output.index(diff) < output.index("backend execution prose")
    assert output.endswith("  receipt: written\n")


def test_json_is_one_object_and_never_contains_diff_paths_hosts_or_backend_prose(
    monkeypatch, capsys,
):
    private = "/private/path secret-spark 10.0.0.2"
    plan = FakePlan(f"--- {private}\n+++ planned\n")
    outcome = FakeOutcome(plan, status="succeeded", receipt_written=True)
    _wire(monkeypatch, plan=plan, outcome=outcome)

    assert setup_cli.run(_args(apply=True, yes=True, json=True)) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["operation"] == "setup"
    assert payload["status"] == "succeeded"
    assert "diff" not in captured.out
    assert private not in captured.out
    assert "backend planning prose" not in captured.out
    assert "backend execution prose" not in captured.out


def test_blocked_plan_includes_bounded_actions_in_terminal_and_json(monkeypatch, capsys):
    blockers = [
        "host_1_comfy_missing",
        "host_2_python_version_unsupported",
        "host_2_cuda_torch_unavailable",
        "host_2_torchmonarch_pin_mismatch",
        "unmapped_internal_blocker",
    ]
    plan = FakePlan(blockers=blockers)
    outcome = FakeOutcome(plan)
    _wire(monkeypatch, plan=plan, outcome=outcome)

    assert setup_cli.run(_args()) == 0
    terminal = capsys.readouterr().out
    assert "Install ComfyUI separately" in terminal
    assert "--python-bin" in terminal
    assert "CUDA-enabled PyTorch" in terminal
    assert f"torchmonarch=={TORCHMONARCH_PIN}" in terminal

    _wire(monkeypatch, plan=plan, outcome=outcome)
    assert setup_cli.run(_args(json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    actions = payload["plan"]["remediations"]
    assert [item["blocker"] for item in actions] == blockers[:-1]
    assert all(set(item) == {"blocker", "action"} for item in actions)


def test_json_apply_without_yes_refuses_before_planning(monkeypatch, capsys):
    _plan, _outcome, calls = _wire(monkeypatch)
    assert setup_cli.run(_args(apply=True, yes=False, json=True)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "input_invalid"
    assert calls == []


@pytest.mark.parametrize(
    ("status", "failure", "receipt", "requested_receipt", "expected"),
    [
        ("planned", None, False, False, 0),
        ("planned", "confirmation_declined", False, False, 2),
        ("failed", "readiness_blocked", True, False, 2),
        ("failed", "verification_failed", True, False, 1),
        ("partial", "worker_start_failed", True, False, 1),
        ("succeeded", None, True, False, 0),
        ("planned", None, False, True, 1),
    ],
)
def test_exit_codes_distinguish_success_refusal_failure_and_receipt_loss(
    monkeypatch, tmp_path, status, failure, receipt, requested_receipt, expected,
):
    plan = FakePlan()
    outcome = FakeOutcome(plan, status=status, failure=failure, receipt_written=receipt)
    _wire(monkeypatch, plan=plan, outcome=outcome)
    receipt_path = str(tmp_path / "receipt.json") if requested_receipt else None
    assert setup_cli.run(_args(receipt=receipt_path)) == expected


@pytest.mark.parametrize(
    ("error", "expected", "code"),
    [
        (ValueError("/private/path token=secret"), 2, "input_invalid"),
        (OSError("/private/path token=secret"), 1, "setup_failed"),
        (RuntimeError("/private/path token=secret"), 1, "setup_failed"),
        (
            ProfileRefusal("advanced", "unknown", "/private/path token=secret"),
            2,
            "profile_refused",
        ),
    ],
)
def test_planning_failures_have_stable_non_disclosing_json(
    monkeypatch, capsys, error, expected, code,
):
    _wire(monkeypatch, build_error=error)
    assert setup_cli.run(_args(json=True)) == expected
    output = capsys.readouterr().out
    assert json.loads(output)["error"]["code"] == code
    assert "/private/path" not in output
    assert "token=secret" not in output


def test_cancelled_interactive_input_is_a_refusal(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(EOFError()))
    assert setup_cli.run(_args(host=None, client_ip=None, json=True)) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "cancelled"


def test_keyboard_interrupt_keeps_cancellation_identity(monkeypatch, capsys):
    monkeypatch.setattr(
        "builtins.input", lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    with pytest.raises(KeyboardInterrupt):
        setup_cli.run(_args(host=None, client_ip=None, json=True))
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "cancelled"


def test_requested_receipt_failure_is_called_out_in_terminal(monkeypatch, tmp_path, capsys):
    plan = FakePlan()
    outcome = FakeOutcome(plan, status="planned", receipt_written=False)
    _wire(monkeypatch, plan=plan, outcome=outcome)
    assert setup_cli.run(_args(receipt=tmp_path / "receipt.json")) == 1
    output = capsys.readouterr().out
    assert "setup: failed" in output
    assert "receipt: FAILED" in output


def test_requested_receipt_failure_is_failed_in_json(monkeypatch, tmp_path, capsys):
    plan = FakePlan()
    outcome = FakeOutcome(plan, status="planned", receipt_written=False)
    _wire(monkeypatch, plan=plan, outcome=outcome)
    assert setup_cli.run(_args(json=True, receipt=tmp_path / "receipt.json")) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["failure"] == "receipt_write_failed"
