"""Guided setup request, plan and outcome records, with injectable SetupOps.

SetupPlan excludes the config path, current and planned config, diff and source
root from repr and as_dict. SetupRequest's repr includes host names, IPs, SSH
key path and other paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import ClusterConfig
from ..operator_profiles import ProfileResolution
from . import operator_receipt
from .doctor import run_doctor
from .lifecycle import run_on_host
from .setup_config_io import (
    ConfigMutation,
    ConfigSnapshot,
    apply_config,
    confirm_setup,
    private_roundtrip,
    read_snapshot,
    request_binding,
    rollback_config,
    run_setup_smoke,
)
from .setup_config_lock import SetupConfigLock
from .setup_probe import ArtifactComparison, HostProbe, validate_artifact_paths
from .setup_receipt_timing import PrestartedReceipt
from .setup_services import (
    SetupServiceRequest,
    SetupServiceResult,
    compensate_setup_services,
    execute_setup_services,
)
from .setup_source import SetupSource, read_setup_source


@dataclass(frozen=True)
class CandidateHost:
    name: str
    fabric_ip: str
    gpus: int = 1
    ssh_user: str = ""
    comfy_dir: str = ""

    def __post_init__(self) -> None:
        if (
            not all(isinstance(value, str) for value in (self.name, self.fabric_ip, self.ssh_user, self.comfy_dir))
            or not self.name
            or not self.fabric_ip
            or type(self.gpus) is not int
            or self.gpus < 1
        ):
            raise ValueError("invalid setup candidate host")


@dataclass(frozen=True)
class SetupRequest:
    hosts: tuple[CandidateHost, ...]
    client_ip: str
    config_path: Path
    profile: str = "balanced"
    fabric_profile: str | None = None
    python_bin: str = "python3"
    ssh_key: str = ""
    comfy_dir: str = ""
    transport_security: str = ""
    artifacts: tuple[str, ...] = ()
    apply: bool = False
    assume_yes: bool = False
    install_service: bool = False
    start_workers: bool = False
    verify: bool = False
    receipt_path: Path | None = None
    privileged_process_inspection: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.hosts, tuple)
            or not self.hosts
            or any(not isinstance(host, CandidateHost) for host in self.hosts)
        ):
            raise ValueError("setup requires at least one explicit candidate host")
        if len({host.name for host in self.hosts}) != len(self.hosts):
            raise ValueError("setup candidate host names must be unique")
        flags = (
            self.apply,
            self.assume_yes,
            self.install_service,
            self.start_workers,
            self.verify,
            self.privileged_process_inspection,
        )
        if any(type(value) is not bool for value in flags):
            raise ValueError("setup choices must be booleans")
        if not isinstance(self.config_path, Path):
            raise ValueError("setup config path must be a Path")
        if self.receipt_path is not None and (
            not isinstance(self.receipt_path, Path) or not self.receipt_path.expanduser().is_absolute()
        ):
            raise ValueError("setup receipt path must be absolute")
        text = (
            self.client_ip,
            self.profile,
            self.python_bin,
            self.ssh_key,
            self.comfy_dir,
            self.transport_security,
        )
        if not all(isinstance(value, str) for value in text) or not isinstance(self.artifacts, tuple):
            raise ValueError("invalid setup request fields")
        if self.fabric_profile is not None and not isinstance(self.fabric_profile, str):
            raise ValueError("invalid setup fabric profile")
        if self.start_workers and not self.install_service:
            raise ValueError("worker start requires a service installed by the same setup transaction")
        validate_artifact_paths(self.artifacts)

    def binding(self) -> str:
        return request_binding(
            {
                "hosts": [
                    [host.name, host.fabric_ip, host.gpus, host.ssh_user, host.comfy_dir] for host in self.hosts
                ],
                "client_ip": self.client_ip,
                "config_path": str(self.config_path.expanduser().absolute()),
                "profile": self.profile,
                "fabric_profile": self.fabric_profile,
                "python_bin": self.python_bin,
                "ssh_key": self.ssh_key,
                "comfy_dir": self.comfy_dir,
                "transport_security": self.transport_security,
                "artifacts": list(self.artifacts),
                "install_service": self.install_service,
                "start_workers": self.start_workers,
                "verify": self.verify,
                "privileged_process_inspection": self.privileged_process_inspection,
            }
        )


@dataclass(frozen=True)
class SetupPlan:
    config_path: Path = field(repr=False)
    config_text: str = field(repr=False)
    config: ClusterConfig = field(repr=False)
    snapshot: ConfigSnapshot = field(repr=False)
    receipt_start: PrestartedReceipt = field(repr=False, compare=False)
    source: SetupSource = field(repr=False)
    request_binding: str
    probes: tuple[HostProbe, ...]
    artifacts: tuple[ArtifactComparison, ...]
    resolution: ProfileResolution
    recommended_profile: str
    fabric_profile: str
    topology: str
    config_digest: str
    current_digest: str | None
    diff: str = field(repr=False)
    gate_context_changes: bool
    gate_warning: str
    privileged_process_inspection: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "host_count": len(self.probes),
            "probes": [probe.as_dict() for probe in self.probes],
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
            "profile": self.resolution.as_dict(),
            "recommended_profile": self.recommended_profile,
            "fabric_profile": self.fabric_profile,
            "topology": self.topology,
            "config_digest": self.config_digest,
            "source_manifest": self.source.manifest,
            "current_digest": self.current_digest,
            "config_changed": self.gate_context_changes,
            "gate_warning": self.gate_warning,
            "privileged_process_inspection": self.privileged_process_inspection,
            "ready": not self.blockers,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class SetupOutcome:
    status: operator_receipt.ReceiptStatus
    plan: SetupPlan
    applied: bool
    config_changed: bool
    services_changed: bool
    verified: bool
    failure: str | None
    receipt: Mapping[str, object]
    receipt_written: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "plan": self.plan.as_dict(),
            "applied": self.applied,
            "config_changed": self.config_changed,
            "services_changed": self.services_changed,
            "verified": self.verified,
            "failure": self.failure,
            "receipt": dict(self.receipt),
            "receipt_written": self.receipt_written,
        }


@dataclass(frozen=True)
class SetupOps:
    run_host: Callable[..., Any] = run_on_host
    roundtrip: Callable[[str], ClusterConfig] = private_roundtrip
    snapshot: Callable[[Path], ConfigSnapshot] = read_snapshot
    config_lock: Callable[[Path], SetupConfigLock] = SetupConfigLock
    apply_config: Callable[..., ConfigMutation] = apply_config
    rollback_config: Callable[[ConfigMutation], bool] = rollback_config
    execute_services: Callable[[SetupServiceRequest], SetupServiceResult] = execute_setup_services
    compensate_services: Callable[[SetupServiceRequest, SetupServiceResult], SetupServiceResult] = (
        compensate_setup_services
    )
    run_smoke: Callable[[Path, str], object] = run_setup_smoke
    run_doctor: Callable[[ClusterConfig], bool] = field(default=lambda config: run_doctor(config, as_json=False))
    confirm: Callable[[str], bool] = confirm_setup
    begin_receipt: Callable[[], PrestartedReceipt] = PrestartedReceipt.start
    source_snapshot: Callable[[], SetupSource] = read_setup_source
    receipt_builder: Callable[..., Any] = operator_receipt.OperatorReceiptBuilder
    write_receipt: Callable[..., Path] = operator_receipt.write_receipt
