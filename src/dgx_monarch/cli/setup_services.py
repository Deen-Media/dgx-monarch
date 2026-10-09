"""Manage guided setup Worker-service changes; unconfirmed outcomes remain UNKNOWN."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from ..config import ClusterConfig

_DIGEST = re.compile(r"[0-9a-f]{64}")
_HOST_STATUSES = frozenset("not_run not_attempted succeeded failed unknown".split())
_CODES = frozenset(
    "succeeded activity_busy activity_unknown ownership_conflict ownership_unknown "
    "source_changed source_unknown stage_failed stage_unknown activation_failed "
    "activation_unknown compensated compensation_failed compensation_unknown "
    "operation_unknown interrupted".split()
)


class ServiceCertainty(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _ordinal(value: object) -> bool:
    return type(value) is int and value >= 1


def _tri_bool(value: object) -> bool:
    return value is None or type(value) is bool


@dataclass(frozen=True)
class SetupServiceRequest:
    config: ClusterConfig = field(repr=False)
    source_root: Path = field(repr=False)
    source_manifest: str
    install_service: bool
    start_service: bool
    privileged_process_inspection: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.config, ClusterConfig) or not self.config.hosts:
            raise ValueError("setup service transaction requires worker hosts")
        flags = (self.install_service, self.start_service, self.privileged_process_inspection)
        if any(type(flag) is not bool for flag in flags):
            raise ValueError("service choices must be booleans")
        if self.start_service and not self.install_service:
            raise ValueError("starting requires a unit installed by this transaction")
        if not self.install_service:
            raise ValueError("setup service transaction requires service installation")
        if not isinstance(self.source_root, Path) or not self.source_root.is_absolute():
            raise ValueError("source_root must be an absolute Path")
        if not isinstance(self.source_manifest, str) or _DIGEST.fullmatch(self.source_manifest) is None:
            raise ValueError("source_manifest must be a SHA-256 digest")

    def binding(self) -> str:
        payload = {
            "hosts": [[host.name, host.address, host.ssh_user] for host in self.config.hosts],
            "python_bin": self.config.python_bin,
            "ssh_key": self.config.ssh_key,
            "source_root": str(self.source_root.expanduser().absolute()),
            "source_manifest": self.source_manifest,
            "install_service": self.install_service,
            "start_service": self.start_service,
            "privileged_process_inspection": self.privileged_process_inspection,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(b"dgxm-setup-services-v1\0" + encoded.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class ActivitySnapshot:
    comfy_running: bool | None
    render_active: bool | None
    active_leases: int | None

    def __post_init__(self) -> None:
        lease = self.active_leases
        if (
            not _tri_bool(self.comfy_running)
            or not _tri_bool(self.render_active)
            or (lease is not None and (type(lease) is not int or lease < 0))
        ):
            raise ValueError("invalid setup activity snapshot")

    def verdict(self) -> ServiceCertainty:
        if self.comfy_running is True or self.render_active is True or (self.active_leases or 0) > 0:
            return ServiceCertainty.FAILED
        if None in (self.comfy_running, self.render_active, self.active_leases):
            return ServiceCertainty.UNKNOWN
        return ServiceCertainty.SUCCEEDED


@dataclass(frozen=True)
class HostOwnership:
    ordinal: int
    unit_absent: bool | None
    systemd_inactive: bool | None
    worker_absent: bool | None
    listener_absent: bool | None
    actors_absent: bool | None
    source_absent: bool | None
    systemd_available: bool | None
    enablement_absent: bool | None

    def __post_init__(self) -> None:
        values = tuple(self.__dict__.values())
        if not _ordinal(self.ordinal) or any(not _tri_bool(value) for value in values[1:]):
            raise ValueError("invalid setup host ownership")

    def verdict(self, *, install_service: bool) -> ServiceCertainty:
        required = (
            self.unit_absent,
            self.systemd_inactive,
            self.worker_absent,
            self.listener_absent,
            self.actors_absent,
            self.source_absent,
            self.enablement_absent,
            *((self.systemd_available,) if install_service else ()),
        )
        if any(value is False for value in required):
            return ServiceCertainty.FAILED
        if any(value is not True for value in required):
            return ServiceCertainty.UNKNOWN
        return ServiceCertainty.SUCCEEDED


@dataclass(frozen=True)
class HostPhaseResult:
    ordinal: int
    attempted: bool
    certainty: ServiceCertainty | None
    changed: bool = False

    def __post_init__(self) -> None:
        valid_certainty = self.certainty is None or isinstance(self.certainty, ServiceCertainty)
        if (
            not _ordinal(self.ordinal)
            or type(self.attempted) is not bool
            or type(self.changed) is not bool
            or not valid_certainty
            or self.attempted != (self.certainty is not None)
            or (self.changed and not self.attempted)
        ):
            raise ValueError("invalid setup host phase")

    @property
    def status(self) -> str:
        return str(self.certainty) if self.certainty is not None else "not_attempted"


@dataclass(frozen=True)
class FleetPhaseResult:
    hosts: tuple[HostPhaseResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.hosts, tuple) or any(not isinstance(row, HostPhaseResult) for row in self.hosts):
            raise ValueError("invalid setup fleet phase")
        if tuple(row.ordinal for row in self.hosts) != tuple(range(1, len(self.hosts) + 1)):
            raise ValueError("setup fleet phase host ordinals must be sequential")

    def verdict(self) -> ServiceCertainty:
        if not self.hosts or any(row.certainty is ServiceCertainty.UNKNOWN for row in self.hosts):
            return ServiceCertainty.UNKNOWN
        if any(row.certainty is not ServiceCertainty.SUCCEEDED for row in self.hosts):
            return ServiceCertainty.FAILED
        return ServiceCertainty.SUCCEEDED


@dataclass(frozen=True)
class HostServiceResult:
    ordinal: int
    stage: str = "not_run"
    activate: str = "not_run"
    compensate: str = "not_run"
    cleanup: str = "not_run"

    def __post_init__(self) -> None:
        if not _ordinal(self.ordinal) or any(
            type(value) is not str or value not in _HOST_STATUSES
            for value in (self.stage, self.activate, self.compensate, self.cleanup)
        ):
            raise ValueError("invalid sanitized host service result")

    def as_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "stage": self.stage,
            "activate": self.activate,
            "compensate": self.compensate,
            "cleanup": self.cleanup,
        }


@dataclass(frozen=True)
class SetupServiceResult:
    certainty: ServiceCertainty
    code: str
    changed: bool
    compensated: bool
    source_manifest: str
    hosts: tuple[HostServiceResult, ...]
    interruption: BaseException | None = field(default=None, repr=False, compare=False)
    _request_binding: str = field(default="", repr=False, compare=False)
    _ops: SetupServiceOps | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        valid_hosts = isinstance(self.hosts, tuple) and all(isinstance(row, HostServiceResult) for row in self.hosts)
        ordinals = tuple(row.ordinal for row in self.hosts) if valid_hosts else ()
        if (
            not isinstance(self.certainty, ServiceCertainty)
            or type(self.code) is not str
            or self.code not in _CODES
            or type(self.changed) is not bool
            or type(self.compensated) is not bool
            or type(self.source_manifest) is not str
            or _DIGEST.fullmatch(self.source_manifest) is None
            or not valid_hosts
            or (self.interruption is not None and not isinstance(self.interruption, BaseException))
        ):
            raise ValueError("invalid setup service result")
        if ordinals != tuple(range(1, len(self.hosts) + 1)):
            raise ValueError("setup service result host ordinals must be sequential")

    def receipt_counts(self) -> dict[str, int]:
        values = {
            "hosts": len(self.hosts),
            "staged": sum(row.stage == "succeeded" for row in self.hosts),
            "activated": sum(row.activate == "succeeded" for row in self.hosts),
            "compensated": sum(row.compensate == "succeeded" for row in self.hosts),
            "cleaned": sum(row.cleanup == "succeeded" for row in self.hosts),
            "failed": sum("failed" in (row.stage, row.activate, row.compensate, row.cleanup) for row in self.hosts),
            "unknown": sum("unknown" in (row.stage, row.activate, row.compensate, row.cleanup) for row in self.hosts),
        }
        return {name: int(value) for name, value in values.items()}

    def as_dict(self) -> dict[str, object]:
        return {
            "certainty": str(self.certainty),
            "code": self.code,
            "changed": self.changed,
            "compensated": self.compensated,
            "source_manifest": self.source_manifest,
            "counts": self.receipt_counts(),
            "hosts": [row.as_dict() for row in self.hosts],
        }


class SetupServiceOps(Protocol):
    def source_current(self) -> ServiceCertainty: ...
    def activity_snapshot(self) -> ActivitySnapshot: ...
    def ownership_snapshot(self) -> tuple[HostOwnership, ...]: ...
    def stage(self) -> FleetPhaseResult: ...
    def activate(self) -> FleetPhaseResult: ...
    def compensate(self) -> FleetPhaseResult: ...
    def cleanup(self) -> FleetPhaseResult: ...


def _phase(value: object, *, hosts: int) -> FleetPhaseResult:
    if not isinstance(value, FleetPhaseResult) or len(value.hosts) != hosts:
        raise TypeError("setup service operation returned an invalid result")
    return value


def _unknown_phase(hosts: int) -> FleetPhaseResult:
    rows = tuple(HostPhaseResult(index, True, ServiceCertainty.UNKNOWN, True) for index in range(1, hosts + 1))
    return FleetPhaseResult(rows)


def _guard(request: SetupServiceRequest, ops: SetupServiceOps) -> tuple[ServiceCertainty, str]:
    activity = ops.activity_snapshot()
    if not isinstance(activity, ActivitySnapshot):
        return ServiceCertainty.UNKNOWN, "activity_unknown"
    verdict = activity.verdict()
    if verdict is not ServiceCertainty.SUCCEEDED:
        return verdict, "activity_busy" if verdict is ServiceCertainty.FAILED else "activity_unknown"
    ownership = ops.ownership_snapshot()
    if tuple(row.ordinal for row in ownership) != tuple(range(1, len(request.config.hosts) + 1)):
        return ServiceCertainty.UNKNOWN, "ownership_unknown"
    verdicts = tuple(row.verdict(install_service=request.install_service) for row in ownership)
    if ServiceCertainty.FAILED in verdicts:
        return ServiceCertainty.FAILED, "ownership_conflict"
    if any(value is not ServiceCertainty.SUCCEEDED for value in verdicts):
        return ServiceCertainty.UNKNOWN, "ownership_unknown"
    return ServiceCertainty.SUCCEEDED, "succeeded"


def _rows(count: int, phases: dict[str, FleetPhaseResult]) -> tuple[HostServiceResult, ...]:
    values: list[HostServiceResult] = []
    for ordinal in range(1, count + 1):
        statuses: dict[str, str] = {}
        for name in ("stage", "activate", "compensate", "cleanup"):
            phase = phases.get(name)
            statuses[name] = phase.hosts[ordinal - 1].status if phase is not None else "not_run"
        values.append(HostServiceResult(ordinal, **statuses))
    return tuple(values)


def _result(
    request: SetupServiceRequest,
    ops: SetupServiceOps,
    phases: dict[str, FleetPhaseResult],
    certainty: ServiceCertainty,
    code: str,
    *,
    compensated: bool,
    interruption: BaseException | None = None,
) -> SetupServiceResult:
    changed = any(row.changed for phase in phases.values() for row in phase.hosts)
    return SetupServiceResult(
        certainty,
        code,
        changed,
        compensated,
        request.source_manifest,
        _rows(len(request.config.hosts), phases),
        interruption,
        request.binding(),
        ops,
    )


def _settle(
    request: SetupServiceRequest,
    ops: SetupServiceOps,
    phases: dict[str, FleetPhaseResult],
) -> tuple[bool, BaseException | None]:
    count = len(request.config.hosts)
    interruption: BaseException | None = None
    try:
        phases["compensate"] = _phase(ops.compensate(), hosts=count)
    except BaseException as exc:
        phases["compensate"] = _unknown_phase(count)
        if not isinstance(exc, Exception):
            interruption = exc
    try:
        phases["cleanup"] = _phase(ops.cleanup(), hosts=count)
    except BaseException as exc:
        phases["cleanup"] = _unknown_phase(count)
        if interruption is None and not isinstance(exc, Exception):
            interruption = exc
    # A confirmed cleanup proves this transaction left no service or source change, even when an interruption hid the
    # compensation result.
    compensated = phases["cleanup"].verdict() is ServiceCertainty.SUCCEEDED
    return compensated, interruption


def _failure_certainty(
    primary: ServiceCertainty, compensated: bool, phases: dict[str, FleetPhaseResult]
) -> ServiceCertainty:
    if primary is ServiceCertainty.UNKNOWN:
        return ServiceCertainty.UNKNOWN
    if not compensated and any(phase.verdict() is ServiceCertainty.UNKNOWN for phase in phases.values()):
        return ServiceCertainty.UNKNOWN
    return ServiceCertainty.FAILED


def _settled_failure(
    request: SetupServiceRequest,
    ops: SetupServiceOps,
    phases: dict[str, FleetPhaseResult],
    primary: ServiceCertainty,
    code: str,
) -> SetupServiceResult:
    compensated, interruption = _settle(request, ops, phases)
    if interruption is not None:
        certainty = ServiceCertainty.FAILED if compensated else ServiceCertainty.UNKNOWN
        return _result(
            request,
            ops,
            phases,
            certainty,
            "interrupted",
            compensated=compensated,
            interruption=interruption,
        )
    return _result(
        request,
        ops,
        phases,
        _failure_certainty(primary, compensated, phases),
        code,
        compensated=compensated,
    )


def execute_setup_services(request: SetupServiceRequest, *, ops: SetupServiceOps | None = None) -> SetupServiceResult:
    if ops is None:
        from .setup_services_platform import SystemSetupServiceOps

        ops = SystemSetupServiceOps(request)
    count = len(request.config.hosts)
    phases: dict[str, FleetPhaseResult] = {}
    mutation_started = False
    try:
        source = ops.source_current()
        source = source if isinstance(source, ServiceCertainty) else ServiceCertainty.UNKNOWN
        if source is not ServiceCertainty.SUCCEEDED:
            code = "source_changed" if source is ServiceCertainty.FAILED else "source_unknown"
            return _result(request, ops, phases, source, code, compensated=True)
        guard, code = _guard(request, ops)
        if guard is not ServiceCertainty.SUCCEEDED:
            return _result(request, ops, phases, guard, code, compensated=True)
        mutation_started = True
        phases["stage"] = _phase(ops.stage(), hosts=count)
        stage = phases["stage"].verdict()
        if stage is not ServiceCertainty.SUCCEEDED:
            code = "stage_unknown" if stage is ServiceCertainty.UNKNOWN else "stage_failed"
            return _settled_failure(request, ops, phases, stage, code)
        source = ops.source_current()
        source = source if isinstance(source, ServiceCertainty) else ServiceCertainty.UNKNOWN
        if source is not ServiceCertainty.SUCCEEDED:
            code = "source_changed" if source is ServiceCertainty.FAILED else "source_unknown"
            return _settled_failure(request, ops, phases, source, code)
        guard, code = _guard(request, ops)
        if guard is not ServiceCertainty.SUCCEEDED:
            return _settled_failure(request, ops, phases, guard, code)
        phases["activate"] = _phase(ops.activate(), hosts=count)
        activation = phases["activate"].verdict()
        if activation is ServiceCertainty.SUCCEEDED:
            return _result(request, ops, phases, ServiceCertainty.SUCCEEDED, "succeeded", compensated=False)
        code = "activation_unknown" if activation is ServiceCertainty.UNKNOWN else "activation_failed"
        return _settled_failure(request, ops, phases, activation, code)
    except BaseException as exc:
        compensated = True
        cleanup_interruption: BaseException | None = None
        if mutation_started:
            compensated, cleanup_interruption = _settle(request, ops, phases)
        interruption = exc if not isinstance(exc, Exception) else cleanup_interruption
        certainty = ServiceCertainty.FAILED if interruption is not None and compensated else ServiceCertainty.UNKNOWN
        code = "interrupted" if interruption is not None else "operation_unknown"
        return _result(
            request,
            ops,
            phases,
            certainty,
            code,
            compensated=compensated,
            interruption=interruption,
        )


def compensate_setup_services(
    request: SetupServiceRequest,
    result: SetupServiceResult,
    *,
    ops: SetupServiceOps | None = None,
) -> SetupServiceResult:
    if result._request_binding != request.binding() or len(result.hosts) != len(request.config.hosts):
        raise ValueError("setup service result does not match this request")
    if result.compensated:
        raise ValueError("setup service transaction is already settled")
    selected = result._ops
    if selected is None or (ops is not None and ops is not selected):
        raise ValueError("setup service compensation requires its transaction handle")
    phases = {name: _restored_phase(result.hosts, name) for name in ("stage", "activate")}
    compensated, interruption = _settle(request, selected, phases)
    unknown = any(phase.verdict() is ServiceCertainty.UNKNOWN for phase in phases.values())
    certainty = (
        ServiceCertainty.FAILED
        if interruption is not None and compensated
        else ServiceCertainty.SUCCEEDED
        if compensated
        else ServiceCertainty.UNKNOWN
        if unknown
        else ServiceCertainty.FAILED
    )
    code: str = (
        "interrupted"
        if interruption is not None
        else "compensated"
        if compensated
        else "compensation_unknown"
        if unknown
        else "compensation_failed"
    )
    return _result(
        request, selected, phases, certainty, code,
        compensated=compensated,
        interruption=interruption,
    )


def _restored_phase(hosts: tuple[HostServiceResult, ...], name: str) -> FleetPhaseResult:
    rows: list[HostPhaseResult] = []
    for host in hosts:
        status = getattr(host, name)
        certainty = None if status in {"not_run", "not_attempted"} else ServiceCertainty(status)
        rows.append(HostPhaseResult(host.ordinal, certainty is not None, certainty, certainty is not None))
    return FleetPhaseResult(tuple(rows))
