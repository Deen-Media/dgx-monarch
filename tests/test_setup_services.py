from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from dgx_monarch.cli import setup_services_platform as platform
from dgx_monarch.cli import setup_services_scripts as scripts
from dgx_monarch.cli.lifecycle_generation import systemd_generation_invalidator
from dgx_monarch.cli.setup_services import (
    ActivitySnapshot,
    FleetPhaseResult,
    HostOwnership,
    HostPhaseResult,
    HostServiceResult,
    ServiceCertainty,
    SetupServiceRequest,
    SetupServiceResult,
    compensate_setup_services,
    execute_setup_services,
)
from dgx_monarch.config import ClusterConfig, HostConfig

DIGEST = "d" * 64
TOKEN = "s-" + "a" * 16


def _request(tmp_path: Path, hosts: int = 2, *, start: bool = False) -> SetupServiceRequest:
    source = tmp_path / "dgx_monarch"
    source.mkdir(exist_ok=True)
    config = ClusterConfig(
        hosts=tuple(HostConfig(f"node-{index}", f"tcp://10.0.0.{index}:26600") for index in range(1, hosts + 1)),
        python_bin=sys.executable,
        transport_security="trusted_fabric",
    )
    return SetupServiceRequest(config, source, DIGEST, True, start)


def _phase(*values: ServiceCertainty | None, changed: bool = True) -> FleetPhaseResult:
    return FleetPhaseResult(
        tuple(
            HostPhaseResult(index, value is not None, value, changed and value is not None)
            for index, value in enumerate(values, 1)
        )
    )


def _clear(count: int) -> tuple[HostOwnership, ...]:
    return tuple(HostOwnership(index, True, True, True, True, True, True, True, True) for index in range(1, count + 1))


class FakeOps:
    def __init__(self, count: int) -> None:
        self.count = count
        self.events: list[str] = []
        self.sources = [ServiceCertainty.SUCCEEDED, ServiceCertainty.SUCCEEDED]
        self.activities = [ActivitySnapshot(False, False, 0), ActivitySnapshot(False, False, 0)]
        self.ownership = [_clear(count), _clear(count)]
        self.stage_result = _phase(*([ServiceCertainty.SUCCEEDED] * count))
        self.activate_result = _phase(*([ServiceCertainty.SUCCEEDED] * count))
        self.compensate_result = _phase(*([ServiceCertainty.SUCCEEDED] * count))
        self.cleanup_result = _phase(*([ServiceCertainty.SUCCEEDED] * count))
        self.stage_error: BaseException | None = None
        self.compensate_error: BaseException | None = None
        self.cleanup_error: BaseException | None = None

    def source_current(self) -> ServiceCertainty:
        self.events.append("source")
        return self.sources.pop(0)

    def activity_snapshot(self) -> ActivitySnapshot:
        self.events.append("activity")
        return self.activities.pop(0)

    def ownership_snapshot(self) -> tuple[HostOwnership, ...]:
        self.events.append("ownership")
        return self.ownership.pop(0)

    def stage(self) -> FleetPhaseResult:
        self.events.append("stage")
        if self.stage_error is not None:
            raise self.stage_error
        return self.stage_result

    def activate(self) -> FleetPhaseResult:
        self.events.append("activate")
        return self.activate_result

    def compensate(self) -> FleetPhaseResult:
        self.events.append("compensate")
        if self.compensate_error is not None:
            raise self.compensate_error
        return self.compensate_result

    def cleanup(self) -> FleetPhaseResult:
        self.events.append("cleanup")
        if self.cleanup_error is not None:
            raise self.cleanup_error
        return self.cleanup_result


def test_request_and_disclosure_models_reject_unsafe_shapes(tmp_path: Path) -> None:
    request = _request(tmp_path, 1)
    assert request.privileged_process_inspection is False
    with pytest.raises(ValueError, match="absolute"):
        SetupServiceRequest(request.config, Path("dgx_monarch"), DIGEST, True, False)
    with pytest.raises(ValueError, match="starting"):
        SetupServiceRequest(request.config, request.source_root, DIGEST, False, True)
    with pytest.raises(ValueError, match="booleans"):
        SetupServiceRequest(request.config, request.source_root, DIGEST, True, False, "yes")
    with pytest.raises(ValueError, match="sanitized"):
        HostServiceResult(1, stage="/secret/path")
    with pytest.raises(ValueError, match="sequential"):
        SetupServiceResult(
            ServiceCertainty.FAILED,
            "stage_failed",
            False,
            True,
            DIGEST,
            (HostServiceResult(2),),
        )


@pytest.mark.parametrize(
    ("activity", "code", "certainty"),
    [
        (ActivitySnapshot(True, None, None), "activity_busy", ServiceCertainty.FAILED),
        (ActivitySnapshot(False, None, 0), "activity_unknown", ServiceCertainty.UNKNOWN),
    ],
)
def test_activity_guard_blocks_before_mutation(
    tmp_path: Path,
    activity: ActivitySnapshot,
    code: str,
    certainty: ServiceCertainty,
) -> None:
    ops = FakeOps(2)
    ops.activities = [activity]
    result = execute_setup_services(_request(tmp_path), ops=ops)
    assert (result.code, result.certainty, result.compensated) == (code, certainty, True)
    assert ops.events == ["source", "activity"]


def test_preexisting_inactive_unit_is_a_refusal(tmp_path: Path) -> None:
    ops = FakeOps(1)
    ops.ownership = [(HostOwnership(1, False, True, True, True, True, True, True, True),)]
    result = execute_setup_services(_request(tmp_path, 1), ops=ops)
    assert result.code == "ownership_conflict"
    assert "stage" not in ops.events


def test_stage_recheck_activate_order_and_sanitized_receipt(tmp_path: Path) -> None:
    request = _request(tmp_path)
    ops = FakeOps(2)
    result = execute_setup_services(request, ops=ops)
    assert result.certainty is ServiceCertainty.SUCCEEDED
    assert ops.events == [
        "source",
        "activity",
        "ownership",
        "stage",
        "source",
        "activity",
        "ownership",
        "activate",
    ]
    encoded = json.dumps(result.as_dict(), sort_keys=True)
    assert "node-" not in encoded and str(tmp_path) not in encoded
    assert result.receipt_counts() == {
        "hosts": 2,
        "staged": 2,
        "activated": 2,
        "compensated": 0,
        "cleaned": 0,
        "failed": 0,
        "unknown": 0,
    }


def test_poststage_drift_settles_everything(tmp_path: Path) -> None:
    ops = FakeOps(3)
    ops.activities[1] = ActivitySnapshot(False, True, 0)
    result = execute_setup_services(_request(tmp_path, 3), ops=ops)
    assert result.code == "activity_busy" and result.compensated
    assert ops.events[-2:] == ["compensate", "cleanup"]
    assert all(row.compensate == row.cleanup == "succeeded" for row in result.hosts)


def test_unknown_cleanup_keeps_failure_unknown(tmp_path: Path) -> None:
    ops = FakeOps(2)
    ops.stage_result = _phase(ServiceCertainty.FAILED, ServiceCertainty.SUCCEEDED)
    ops.cleanup_result = _phase(ServiceCertainty.UNKNOWN, ServiceCertainty.SUCCEEDED)
    result = execute_setup_services(_request(tmp_path), ops=ops)
    assert result.certainty is ServiceCertainty.UNKNOWN
    assert result.code == "stage_failed" and not result.compensated


def test_cancellation_identity_survives_attempt_all_settlement(tmp_path: Path) -> None:
    primary = KeyboardInterrupt()
    cleanup = SystemExit(9)
    ops = FakeOps(2)
    ops.stage_error = primary
    ops.cleanup_error = cleanup
    result = execute_setup_services(_request(tmp_path), ops=ops)
    assert result.interruption is primary
    assert ops.events[-2:] == ["compensate", "cleanup"]
    assert not result.compensated


def test_successful_install_only_can_be_compensated_later(tmp_path: Path) -> None:
    request = _request(tmp_path, 2, start=False)
    ops = FakeOps(2)
    installed = execute_setup_services(request, ops=ops)
    compensated = compensate_setup_services(request, installed)
    assert compensated.certainty is ServiceCertainty.SUCCEEDED
    assert compensated.code == "compensated" and compensated.compensated
    with pytest.raises(ValueError, match="already settled"):
        compensate_setup_services(request, compensated)
    with pytest.raises(ValueError, match="transaction handle"):
        compensate_setup_services(request, installed, ops=FakeOps(2))


def test_terminal_cleanup_readback_settles_after_compensation_cancellation(tmp_path: Path) -> None:
    interrupt = KeyboardInterrupt()

    class InterruptedAfterCompensation(FakeOps):
        def compensate(self) -> FleetPhaseResult:
            self.events.append("compensate")
            raise interrupt

    ops = InterruptedAfterCompensation(2)
    ops.stage_result = _phase(ServiceCertainty.FAILED, ServiceCertainty.SUCCEEDED)

    result = execute_setup_services(_request(tmp_path), ops=ops)

    assert result.interruption is interrupt
    assert result.certainty is ServiceCertainty.FAILED
    assert result.compensated is True
    assert result.code == "interrupted"
    assert ops.events[-2:] == ["compensate", "cleanup"]


def test_non_success_result_retains_authority_for_bounded_resettlement(tmp_path: Path) -> None:
    request = _request(tmp_path)
    ops = FakeOps(2)
    ops.stage_result = _phase(ServiceCertainty.FAILED, ServiceCertainty.SUCCEEDED)
    ops.compensate_result = _phase(ServiceCertainty.UNKNOWN, ServiceCertainty.SUCCEEDED)
    ops.cleanup_result = _phase(ServiceCertainty.UNKNOWN, ServiceCertainty.SUCCEEDED)
    unsettled = execute_setup_services(request, ops=ops)
    assert unsettled.certainty is ServiceCertainty.UNKNOWN
    assert unsettled.compensated is False

    ops.compensate_result = _phase(ServiceCertainty.SUCCEEDED, ServiceCertainty.SUCCEEDED)
    ops.cleanup_result = _phase(ServiceCertainty.SUCCEEDED, ServiceCertainty.SUCCEEDED)
    settled = compensate_setup_services(request, unsettled)

    assert settled.certainty is ServiceCertainty.SUCCEEDED
    assert settled.code == "compensated"
    assert settled.compensated is True
    assert ops.events[-2:] == ["compensate", "cleanup"]


def _completed(stdout: str = "", code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(["fake"], code, stdout, "")


class ScriptFleet:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.cleaned: set[str] = set()
        self.reserve_foreign: set[str] = set()
        self.reserve_absent: set[str] = set()
        self.activation_state: dict[str, str] = {}
        self.compensation_state: dict[str, str] = {}

    def __call__(
        self, _config: ClusterConfig, host: HostConfig, script: str, _timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        name = host.name
        if scripts.STATE_MARKER in script:
            self.events.append(("ownership", name))
            payload = {
                "ordinal": int(name.rsplit("-", 1)[1]),
                "unit_absent": True,
                "systemd_inactive": True,
                "worker_absent": True,
                "listener_absent": True,
                "actors_absent": True,
                "source_absent": True,
                "systemd_available": True,
                "enablement_absent": True,
            }
            return _completed(scripts.STATE_MARKER + json.dumps(payload) + "\n")
        if "state='CLEANED'" in script:
            self.events.append(("cleanup", name))
            self.cleaned.add(name)
            return _completed("CLEANED\n")
        if "run_ok=" in script:
            self.events.append(("probe_activation", name))
            return _completed(self.activation_state.get(name, "NEW") + "\n")
        if "print('COMPENSATED')" in script:
            self.events.append(("compensate", name))
            return _completed(self.compensation_state.get(name, "COMPENSATED") + "\n")
        if "print('PUBLISHED')" in script:
            self.events.append(("activate", name))
            return _completed("PUBLISHED\n")
        if "actual!=expected" in script:
            self.events.append(("verify", name))
            return _completed("MATCH\n")
        if "print('OWNED' if exact" in script:
            self.events.append(("probe_stage", name))
            state = (
                "ABSENT"
                if name in self.cleaned or name in self.reserve_absent
                else "FOREIGN"
                if name in self.reserve_foreign
                else "OWNED"
            )
            return _completed(state + "\n")
        if "print('RESERVED')" in script:
            self.events.append(("reserve", name))
            failed = name in self.reserve_foreign or name in self.reserve_absent
            return _completed("", 1) if failed else _completed("RESERVED\n")
        raise AssertionError("unexpected setup service script")


class Commands:
    def __init__(self) -> None:
        self.argv: list[list[str]] = []

    def __call__(self, argv: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert isinstance(argv, list)
        self.argv.append([str(value) for value in argv])
        return _completed()


def _system_ops(
    request: SetupServiceRequest,
    fleet: ScriptFleet,
    commands: Commands,
    *,
    local: str = "",
) -> platform.SystemSetupServiceOps:
    return platform.SystemSetupServiceOps(
        request,
        command_runner=commands,
        host_runner=fleet,
        local_detector=lambda host: host.name == local,
        token_factory=lambda: TOKEN,
        activity_probe=lambda: ActivitySnapshot(False, False, 0),
    )


def test_platform_activity_requires_complete_comfy_process_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path, 1)
    ops = platform.SystemSetupServiceOps(request, local_detector=lambda _host: False)
    monkeypatch.setattr(platform.comfy_ports, "find_comfy_ports", lambda: {})
    monkeypatch.setattr(
        platform.comfy_ports, "driver_probe", lambda _host, _ports: (None, True))

    monkeypatch.setattr(platform.comfy_ports, "comfy_candidate_count", lambda: None)
    assert ops.activity_snapshot() == ActivitySnapshot(None, None, None)

    monkeypatch.setattr(platform.comfy_ports, "comfy_candidate_count", lambda: 1)
    assert ops.activity_snapshot() == ActivitySnapshot(True, None, None)

    monkeypatch.setattr(platform.comfy_ports, "comfy_candidate_count", lambda: 0)
    assert ops.activity_snapshot() == ActivitySnapshot(False, False, 0)


def test_platform_activity_is_unknown_when_a_driver_port_never_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unobserved port is not an idle box.

    When the process scan misses a Comfy and its object-info probe times out,
    the snapshot must leave all three fields unobserved. A false idle reading,
    `ActivitySnapshot(False, False, 0)`, would let setup install and start
    services under a live render.
    """
    request = _request(tmp_path, 1)
    ops = platform.SystemSetupServiceOps(request, local_detector=lambda _host: False)
    monkeypatch.setattr(platform.comfy_ports, "find_comfy_ports", lambda: {8188: None})
    monkeypatch.setattr(platform.comfy_ports, "comfy_candidate_count", lambda: 0)
    monkeypatch.setattr(
        platform.comfy_ports, "driver_probe", lambda _host, _ports: (None, False))

    assert ops.activity_snapshot() == ActivitySnapshot(None, None, None)
    assert ops.activity_snapshot().verdict() is ServiceCertainty.UNKNOWN


def test_platform_refuses_unresolved_local_identity(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unable to tell whether a setup target is local"):
        platform.SystemSetupServiceOps(_request(tmp_path, 1), local_detector=lambda _host: None)


@pytest.mark.parametrize("locality", [False, None])
def test_platform_refuses_an_invalid_token_before_resolving_any_target(
    tmp_path: Path, locality: bool | None,
) -> None:
    """The token refusal needs no host, so no target is resolved before it.

    With ``None`` both refusals apply, and the token one wins.
    """
    resolved: list[str] = []

    def detector(host: HostConfig) -> bool | None:
        resolved.append(host.name)
        return locality

    with pytest.raises(ValueError, match=r"^invalid setup service token$"):
        platform.SystemSetupServiceOps(
            _request(tmp_path, 2), local_detector=detector, token_factory=lambda: "s-short")
    assert resolved == []


def test_comfy_candidate_scan_refuses_incomplete_process_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    class Process:
        def __init__(self, command: object) -> None:
            self.info = {"cmdline": command}

    class Psutil:
        @staticmethod
        def process_iter(_attrs, *, ad_value):
            return (Process(ad_value),)

    monkeypatch.setitem(sys.modules, "psutil", Psutil())
    assert platform.comfy_ports.comfy_candidate_count() is None

    Psutil.process_iter = staticmethod(lambda _attrs, *, ad_value: ())
    assert platform.comfy_ports.comfy_candidate_count() == 0

    Psutil.process_iter = staticmethod(
        lambda _attrs, *, ad_value: (Process(["python", "main.py", "--port", "8188"]),)
    )
    assert platform.comfy_ports.comfy_candidate_count() == 1


def test_platform_stages_all_hosts_into_exact_slots_and_compensates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "dgx_source_manifest_sha256", lambda _root: DIGEST)
    request = _request(tmp_path, 2)
    fleet, commands = ScriptFleet(), Commands()
    ops = _system_ops(request, fleet, commands, local="node-1")
    installed = execute_setup_services(request, ops=ops)
    assert installed.certainty is ServiceCertainty.SUCCEEDED
    rsync = [argv for argv in commands.argv if "rsync" in argv]
    assert len(rsync) == 1 and TOKEN + "-2" in rsync[0][-1] and "-e" in rsync[0]
    assert str(request.source_root.parent) in ops._unit(request.config.hosts[0], 1)
    assert TOKEN + "-1" not in ops._unit(request.config.hosts[0], 1)
    assert systemd_generation_invalidator() in ops._unit(request.config.hosts[0], 1)
    assert max(i for i, row in enumerate(fleet.events) if row[0] == "verify") < min(
        i for i, row in enumerate(fleet.events) if row[0] == "activate"
    )
    undone = compensate_setup_services(request, installed)
    assert undone.compensated and undone.certainty is ServiceCertainty.SUCCEEDED
    assert [name for event, name in fleet.events if event == "cleanup"] == ["node-1", "node-2"]


def test_platform_classifies_foreign_reservation_as_failed_and_absent_as_unknown(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, 4)
    fleet, commands = ScriptFleet(), Commands()
    fleet.reserve_foreign.add("node-2")
    fleet.reserve_absent.add("node-3")
    result = _system_ops(request, fleet, commands).stage()
    assert [row.certainty for row in result.hosts] == [
        ServiceCertainty.SUCCEEDED,
        ServiceCertainty.FAILED,
        ServiceCertainty.UNKNOWN,
        ServiceCertainty.SUCCEEDED,
    ]
    assert result.hosts[1].changed is False and result.hosts[2].changed is True
    assert [name for event, name in fleet.events if event == "reserve"] == [
        "node-1",
        "node-2",
        "node-3",
        "node-4",
    ]


def test_platform_foreign_reservation_stays_definite_without_foreign_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "dgx_source_manifest_sha256", lambda _root: DIGEST)
    request = _request(tmp_path, 3)
    fleet, commands = ScriptFleet(), Commands()
    fleet.reserve_foreign.add("node-2")

    result = execute_setup_services(request, ops=_system_ops(request, fleet, commands))

    assert result.certainty is ServiceCertainty.FAILED
    assert result.code == "stage_failed" and result.compensated is True
    assert ("cleanup", "node-2") not in fleet.events
    assert [name for event, name in fleet.events if event == "cleanup"] == ["node-1", "node-3"]


def test_platform_cleanup_rechecks_after_ambiguous_compensation(tmp_path: Path) -> None:
    request = _request(tmp_path, 3)
    fleet, commands = ScriptFleet(), Commands()
    ops = _system_ops(request, fleet, commands)
    assert ops.stage().verdict() is ServiceCertainty.SUCCEEDED
    fleet.activation_state["node-2"] = "UNKNOWN"
    activated = ops.activate()
    assert activated.hosts[2].attempted is False
    fleet.compensation_state["node-2"] = "UNKNOWN"
    compensation = ops.compensate()
    assert compensation.hosts[1].certainty is ServiceCertainty.UNKNOWN
    cleanup = ops.cleanup()
    assert cleanup.hosts[1].certainty is ServiceCertainty.SUCCEEDED
    assert ("cleanup", "node-2") in fleet.events


def test_platform_terminal_cleanup_settles_post_effect_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    interrupt = KeyboardInterrupt()
    request = _request(tmp_path, 1)
    fleet, commands = ScriptFleet(), Commands()
    fleet.activation_state["node-1"] = "UNKNOWN"
    ops = _system_ops(request, fleet, commands)
    monkeypatch.setattr(platform, "dgx_source_manifest_sha256", lambda _root: DIGEST)
    original = fleet.__call__

    def interrupt_after_effect(
        config: ClusterConfig, host: HostConfig, script: str, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        if "print('COMPENSATED')" in script:
            fleet.events.append(("compensate", host.name))
            raise interrupt
        return original(config, host, script, timeout)

    ops._host_run = interrupt_after_effect
    result = execute_setup_services(request, ops=ops)

    assert result.interruption is interrupt
    assert result.code == "interrupted"
    assert result.certainty is ServiceCertainty.FAILED
    assert result.compensated is True
    assert fleet.events[-3:] == [
        ("compensate", "node-1"),
        ("cleanup", "node-1"),
        ("probe_stage", "node-1"),
    ]


def test_platform_resettlement_retries_only_unconfirmed_compensation(tmp_path: Path) -> None:
    request = _request(tmp_path, 3)
    fleet, commands = ScriptFleet(), Commands()
    ops = _system_ops(request, fleet, commands)
    ops._activation = dict.fromkeys(range(1, 4), ServiceCertainty.SUCCEEDED)
    fleet.compensation_state["node-2"] = "UNKNOWN"

    first = ops.compensate()
    assert first.verdict() is ServiceCertainty.UNKNOWN
    fleet.compensation_state["node-2"] = "COMPENSATED"
    second = ops.compensate()

    assert second.verdict() is ServiceCertainty.SUCCEEDED
    compensated_hosts = [name for event, name in fleet.events if event == "compensate"]
    assert compensated_hosts == ["node-1", "node-2", "node-3", "node-2"]


def test_platform_finalization_attempts_all_and_prefers_cancellation(tmp_path: Path) -> None:
    request = _request(tmp_path, 3)
    fleet, commands = ScriptFleet(), Commands()
    ops = _system_ops(request, fleet, commands)
    ops._activation = dict.fromkeys(range(1, 4), ServiceCertainty.UNKNOWN)
    original = fleet.__call__

    def interrupting(
        config: ClusterConfig, host: HostConfig, script: str, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        if "print('COMPENSATED')" in script:
            fleet.events.append(("compensate", host.name))
            if host.name == "node-1":
                raise RuntimeError("ordinary")
            if host.name == "node-2":
                raise KeyboardInterrupt
            return _completed("COMPENSATED\n")
        return original(config, host, script, timeout)

    ops._host_run = interrupting
    with pytest.raises(KeyboardInterrupt):
        ops.compensate()
    assert [name for event, name in fleet.events if event == "compensate"] == [
        "node-1",
        "node-2",
        "node-3",
    ]


def test_generated_scripts_are_fail_closed_and_compile() -> None:
    built = [
        scripts.build_ownership_script(sys.executable, "tcp://127.0.0.1:26600", 1, local_source=True),
        scripts.build_reserve_script(sys.executable, TOKEN, 1, DIGEST),
        scripts.build_stage_probe_script(sys.executable, TOKEN, 1, DIGEST),
        scripts.build_verify_stage_script(sys.executable, TOKEN, 1, DIGEST),
        scripts.build_activate_script(
            sys.executable,
            TOKEN,
            1,
            DIGEST,
            "[Unit]\n",
            "tcp://127.0.0.1:26600",
            start_service=False,
        ),
        scripts.build_probe_activation_script(sys.executable, TOKEN, 1, DIGEST),
        scripts.build_compensate_script(sys.executable, TOKEN, 1, DIGEST),
        scripts.build_cleanup_script(sys.executable, TOKEN, 1, DIGEST),
    ]
    joined = "\n".join(built)
    assert "pkill" not in joined and "nohup" not in joined
    assert "from dgx_monarch.runtime_provenance" not in joined
    assert "trusted_manifest" in joined
    assert "O_NOFOLLOW" in joined and "O_EXCL" in joined and "slot_dev" in joined
    assert "/proc/net/tcp6" in joined and "FragmentPath" in joined and "mutation.lock" in joined
    for number, script in enumerate(built):
        assert "\0" not in script
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, re.S)
        assert blocks
        for index, block in enumerate(blocks):
            compile(block, f"script-{number}-{index}", "exec")


def test_reservation_is_private_and_does_not_chmod_existing_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    (home / ".config").mkdir(parents=True, mode=0o755)
    (home / ".local").mkdir(mode=0o755)
    os.chmod(home, 0o700)
    os.chmod(home / ".config", 0o755)  # noqa: S103 - prove setup preserves mode
    os.chmod(home / ".local", 0o755)  # noqa: S103 - prove setup preserves mode
    env = dict(os.environ, HOME=str(home))
    result = subprocess.run(
        ["bash", "-s"],
        input=scripts.build_reserve_script(sys.executable, TOKEN, 1, DIGEST),
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert result.returncode == 0 and "RESERVED" in result.stdout
    slot = home / scripts.release_rel(TOKEN, 1)
    assert stat.S_IMODE(slot.stat().st_mode) == 0o700
    assert stat.S_IMODE((slot / "setup.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((home / ".config").stat().st_mode) == 0o755
    assert stat.S_IMODE((home / ".local").stat().st_mode) == 0o755
    again = subprocess.run(
        ["bash", "-s"],
        input=scripts.build_reserve_script(sys.executable, TOKEN, 1, DIGEST),
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert again.returncode != 0
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    systemctl = fake_bin / "systemctl"
    systemctl.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        " *LoadState*) echo LoadState=not-found;;\n"
        " *ActiveState*) echo ActiveState=inactive;;\n"
        " *SubState*) echo SubState=dead;;\n"
        " *MainPID*) echo MainPID=0;;\n"
        " *Job*) echo Job=0;;\n"
        " *is-enabled*) echo not-found; exit 1;;\n"
        "esac\n",
        encoding="utf-8",
    )
    systemctl.chmod(0o700)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    foreign = slot / "site/foreign.txt"
    foreign.write_text("not transaction source", encoding="utf-8")
    refused = subprocess.run(
        ["bash", "-s"],
        input=scripts.build_cleanup_script(sys.executable, TOKEN, 1, DIGEST),
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert "UNKNOWN" in refused.stdout and slot.exists()
    foreign.unlink()
    cleaned = subprocess.run(
        ["bash", "-s"],
        input=scripts.build_cleanup_script(sys.executable, TOKEN, 1, DIGEST),
        text=True,
        capture_output=True,
        env=env,
        timeout=10,
    )
    assert "CLEANED" in cleaned.stdout and not slot.exists()
    assert not (home / f".local/state/dgx-monarch/setup/{TOKEN}-1").exists()
