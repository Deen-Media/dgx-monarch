"""System/SSH adapter for the guided setup service transaction."""

from __future__ import annotations

import json
import os
import pwd
import secrets
import shlex
import subprocess
from collections.abc import Callable, Sequence
from typing import Protocol

from ..config import ClusterConfig, HostConfig
from ..runtime_provenance import SOURCE_ONLY_PYCACHE_PREFIX, dgx_source_manifest_sha256
from . import comfy_ports, lifecycle
from . import setup_services_scripts as scripts
from .lifecycle_generation import systemd_generation_invalidator
from .setup_services import (
    ActivitySnapshot,
    FleetPhaseResult,
    HostOwnership,
    HostPhaseResult,
    ServiceCertainty,
    SetupServiceRequest,
)
from .setup_services_rsync_lock import locked_exec_source
from .systemd_unit import quote as systemd_quote
from .systemd_unit import resolve_python


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]: ...


class HostRunner(Protocol):
    def __call__(
        self,
        config: ClusterConfig,
        host: HostConfig,
        script: str,
        timeout: int = 60,
    ) -> subprocess.CompletedProcess[str]: ...


def _completed(
    argv: Sequence[str],
    *,
    capture_output: bool,
    text: bool,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=capture_output, text=text, timeout=timeout)


def _unknown_ownership(ordinal: int) -> HostOwnership:
    return HostOwnership(ordinal, None, None, None, None, None, None, None, None)


def _word(result: subprocess.CompletedProcess[str] | None, allowed: frozenset[str]) -> str | None:
    if result is None or result.returncode != 0 or len(result.stdout) > 16_384:
        return None
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if len(lines) == 1 and lines[0] in allowed else None


class SystemSetupServiceOps:
    """Concrete operations whose public results contain no host/path detail."""

    def __init__(
        self,
        request: SetupServiceRequest,
        *,
        command_runner: CommandRunner = _completed,
        host_runner: HostRunner = lifecycle.run_on_host,
        local_detector: Callable[[HostConfig], bool | None] = lifecycle._is_local,
        token_factory: Callable[[], str] | None = None,
        activity_probe: Callable[[], ActivitySnapshot] | None = None,
    ) -> None:
        self.request = request
        self._run = command_runner
        self._host_run = host_runner
        # The token refusal needs no host, so it precedes target resolution.
        self._token = token_factory() if token_factory else f"s-{secrets.token_hex(8)}"
        scripts.validate_identity(self._token, 1, request.source_manifest)
        try:
            current_user = pwd.getpwuid(os.geteuid()).pw_name
        except KeyError as exc:
            raise ValueError("unable to identify the current setup user") from exc
        local_flags: list[bool] = []
        for host in request.config.hosts:
            local = local_detector(host)
            if type(local) is not bool:
                raise ValueError("unable to tell whether a setup target is local")
            local_flags.append(local)
        detected = tuple(local_flags)
        if any(
            local and host.ssh_user not in ("", current_user)
            for local, host in zip(detected, request.config.hosts, strict=True)
        ):
            raise ValueError("a local setup target cannot select a different SSH user")
        self._local = detected
        self._activity_probe = activity_probe
        self._stage: dict[int, ServiceCertainty] = {}
        self._reservation: dict[int, bool | None] = {}
        self._activation: dict[int, ServiceCertainty] = {}
        self._compensation: dict[int, ServiceCertainty] = {}
        self._cleanup: dict[int, ServiceCertainty] = {}

    def source_current(self) -> ServiceCertainty:
        try:
            root = self.request.source_root.resolve(strict=True)
            if not root.is_dir() or root.name != "dgx_monarch":
                return ServiceCertainty.FAILED
            actual = dgx_source_manifest_sha256(root)
        except (OSError, RuntimeError, ValueError):
            return ServiceCertainty.UNKNOWN
        return ServiceCertainty.SUCCEEDED if actual == self.request.source_manifest else ServiceCertainty.FAILED

    def activity_snapshot(self) -> ActivitySnapshot:
        if self._activity_probe is not None:
            return self._activity_probe()
        try:
            ports = comfy_ports.find_comfy_ports()
            running, observed = comfy_ports.driver_probe(None, ports)
            candidates = comfy_ports.comfy_candidate_count()
        except Exception:
            return ActivitySnapshot(None, None, None)
        if running is not None:
            return ActivitySnapshot(True, None, None)
        # Confirming idle requires a complete process scan and answers from
        # every driver port. Otherwise render, lease and activity state remain
        # unobserved.
        if candidates is None or not observed:
            return ActivitySnapshot(None, None, None)
        if candidates > 0:
            return ActivitySnapshot(True, None, None)
        return ActivitySnapshot(False, False, 0)

    def ownership_snapshot(self) -> tuple[HostOwnership, ...]:
        return tuple(
            self._ownership(host, ordinal, local=self._local[ordinal - 1])
            for ordinal, host in enumerate(self.request.config.hosts, 1)
        )

    def _ownership(self, host: HostConfig, ordinal: int, *, local: bool) -> HostOwnership:
        script = scripts.build_ownership_script(
            self.request.config.python_bin, host.address, ordinal, local_source=local,
            privileged_process_inspection=self.request.privileged_process_inspection,
        )
        result = self._host_result(host, script, 30)
        if result is None or result.returncode != 0 or len(result.stdout) > 16_384:
            return _unknown_ownership(ordinal)
        lines = [
            line.removeprefix(scripts.STATE_MARKER)
            for line in result.stdout.splitlines()
            if line.startswith(scripts.STATE_MARKER)
        ]
        if len(lines) != 1:
            return _unknown_ownership(ordinal)
        try:
            payload = json.loads(lines[0])
        except (TypeError, ValueError):
            return _unknown_ownership(ordinal)
        names = (
            "unit_absent",
            "systemd_inactive",
            "worker_absent",
            "listener_absent",
            "actors_absent",
            "source_absent",
            "systemd_available",
            "enablement_absent",
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != {"ordinal", *names}
            or payload.get("ordinal") != ordinal
            or any(value is not None and not isinstance(value, bool) for value in map(payload.get, names))
        ):
            return _unknown_ownership(ordinal)
        return HostOwnership(ordinal, *(payload[name] for name in names))

    def stage(self) -> FleetPhaseResult:
        rows: list[HostPhaseResult] = []
        for ordinal, host in enumerate(self.request.config.hosts, 1):
            self._stage[ordinal] = ServiceCertainty.UNKNOWN
            self._reservation[ordinal] = None
            certainty, changed = self._stage_host(host, ordinal)
            self._stage[ordinal] = certainty
            rows.append(HostPhaseResult(ordinal, True, certainty, changed))
        return FleetPhaseResult(tuple(rows))

    def _stage_host(self, host: HostConfig, ordinal: int) -> tuple[ServiceCertainty, bool]:
        reserve = self._host_result(
            host,
            scripts.build_reserve_script(
                self.request.config.python_bin,
                self._token,
                ordinal,
                self.request.source_manifest,
            ),
            30,
        )
        if _word(reserve, frozenset({"RESERVED"})) != "RESERVED":
            state = self._probe_stage(host, ordinal)
            if state == "FOREIGN":
                # The slot exists but this transaction does not own it: a
                # definite refusal, not missing transport evidence.
                self._reservation[ordinal] = False
                return ServiceCertainty.FAILED, False
            if state == "ABSENT":
                # An absent slot does not prove the reserve script created no parent directory.
                return ServiceCertainty.UNKNOWN, True
            if state != "OWNED":
                return ServiceCertainty.UNKNOWN, True
        self._reservation[ordinal] = True
        self._copy(host, ordinal)
        verified = self._verify_stage(host, ordinal)
        if verified == "MATCH":
            return ServiceCertainty.SUCCEEDED, True
        if verified == "MISMATCH":
            return ServiceCertainty.FAILED, True
        return ServiceCertainty.UNKNOWN, True

    def _copy(self, host: HostConfig, ordinal: int) -> None:
        if self._local[ordinal - 1]:
            return
        locked_exec = locked_exec_source()
        txn_rel = f".local/state/dgx-monarch/setup/{self._token}-{ordinal}"
        slot_rel = scripts.release_rel(self._token, ordinal)
        rsync = [
            "rsync",
            "-a",
            "--delete",
            "--delete-excluded",
            "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=",
            "--exclude=__pycache__/",
            "--exclude=*.pyc",
            "--exclude=*.pyo",
        ]
        remote_shell = lifecycle._rsync_remote_shell(self.request.config, host)
        target = lifecycle._rsync_target(host)
        destination = f"{target}:~/{scripts.release_rel(self._token, ordinal)}/site/dgx_monarch/"
        python = shlex.quote(self.request.config.python_bin)
        expand = 'case "$PYBIN" in "~/"*) PYBIN="$HOME/${PYBIN#\\~/}";; "~") PYBIN="$HOME";; esac'
        fixed = " ".join(
            shlex.quote(value)
            for value in (txn_rel, slot_rel, self._token, str(ordinal), self.request.source_manifest, "rsync")
        )
        remote_lock = f'PYBIN={python}; {expand}; exec "$PYBIN" -I -S -B -c {shlex.quote(locked_exec)} {fixed}'
        argv = [*rsync, "-e", remote_shell, "--rsync-path", remote_lock]
        try:
            self._run(
                [*argv, "--", f"{self.request.source_root}/", destination],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _probe_stage(self, host: HostConfig, ordinal: int) -> str | None:
        result = self._host_result(
            host,
            scripts.build_stage_probe_script(
                self.request.config.python_bin,
                self._token,
                ordinal,
                self.request.source_manifest,
            ),
            30,
        )
        return _word(result, frozenset({"ABSENT", "OWNED", "FOREIGN", "UNKNOWN"}))

    def _verify_stage(self, host: HostConfig, ordinal: int) -> str | None:
        result = self._host_result(
            host,
            scripts.build_verify_stage_script(
                self.request.config.python_bin,
                self._token,
                ordinal,
                self.request.source_manifest,
                source_root=str(self.request.source_root) if self._local[ordinal - 1] else None,
            ),
            60,
        )
        return _word(result, frozenset({"MATCH", "MISMATCH", "UNKNOWN"}))

    def activate(self) -> FleetPhaseResult:
        rows: list[HostPhaseResult] = []
        halted = False
        for ordinal, host in enumerate(self.request.config.hosts, 1):
            if halted:
                rows.append(HostPhaseResult(ordinal, False, None))
                continue
            if self._stage.get(ordinal) is not ServiceCertainty.SUCCEEDED:
                certainty = ServiceCertainty.UNKNOWN
            else:
                self._activation[ordinal] = ServiceCertainty.UNKNOWN
                self._host_result(
                    host,
                    scripts.build_activate_script(
                        self.request.config.python_bin,
                        self._token,
                        ordinal,
                        self.request.source_manifest,
                        self._unit(host, ordinal),
                        host.address,
                        start_service=self.request.start_service,
                        privileged_process_inspection=self.request.privileged_process_inspection,
                        source_site=str(self.request.source_root.parent) if self._local[ordinal - 1] else None,
                    ),
                    90,
                )
                state = self._probe_activation(host, ordinal)
                certainty = (
                    ServiceCertainty.SUCCEEDED
                    if state == "NEW"
                    else ServiceCertainty.FAILED
                    if state == "PRIOR"
                    else ServiceCertainty.UNKNOWN
                )
            self._activation[ordinal] = certainty
            rows.append(HostPhaseResult(ordinal, True, certainty, certainty is not ServiceCertainty.FAILED))
            halted = certainty is not ServiceCertainty.SUCCEEDED
        return FleetPhaseResult(tuple(rows))

    def _probe_activation(self, host: HostConfig, ordinal: int) -> str | None:
        result = self._host_result(
            host,
            scripts.build_probe_activation_script(
                self.request.config.python_bin,
                self._token,
                ordinal,
                self.request.source_manifest,
                privileged_process_inspection=self.request.privileged_process_inspection,
            ),
            30,
        )
        return _word(result, frozenset({"NEW", "PRIOR", "UNKNOWN"}))

    def compensate(self) -> FleetPhaseResult:
        rows: list[HostPhaseResult] = []
        interruption: BaseException | None = None
        for ordinal, host in enumerate(self.request.config.hosts, 1):
            certainty: ServiceCertainty
            try:
                prior = self._compensation.get(ordinal)
                activated = self._activation.get(ordinal)
                if prior is ServiceCertainty.SUCCEEDED:
                    certainty = prior
                    changed = activated is not None and activated is not ServiceCertainty.FAILED
                elif activated is None or activated is ServiceCertainty.FAILED:
                    certainty = ServiceCertainty.SUCCEEDED
                    changed = False
                else:
                    result = self._host_result(
                        host,
                        scripts.build_compensate_script(
                            self.request.config.python_bin,
                            self._token,
                            ordinal,
                            self.request.source_manifest,
                            privileged_process_inspection=self.request.privileged_process_inspection,
                        ),
                        160,
                    )
                    if _word(result, frozenset({"COMPENSATED", "UNKNOWN"})) != "COMPENSATED":
                        certainty = ServiceCertainty.UNKNOWN
                    else:
                        observed = self._ownership(host, ordinal, local=self._local[ordinal - 1]).verdict(
                            install_service=True
                        )
                        certainty = observed
                    changed = True
            except BaseException as exc:
                certainty, changed = ServiceCertainty.UNKNOWN, True
                if interruption is None or (isinstance(interruption, Exception) and not isinstance(exc, Exception)):
                    interruption = exc
            self._compensation[ordinal] = certainty
            rows.append(HostPhaseResult(ordinal, True, certainty, changed))
        if interruption is not None:
            raise interruption
        return FleetPhaseResult(tuple(rows))

    def cleanup(self) -> FleetPhaseResult:
        rows: list[HostPhaseResult] = []
        interruption: BaseException | None = None
        for ordinal, host in enumerate(self.request.config.hosts, 1):
            try:
                completed = self._cleanup.get(ordinal)
                if completed is ServiceCertainty.SUCCEEDED:
                    rows.append(HostPhaseResult(ordinal, True, completed, True))
                    continue
                if self._reservation.get(ordinal, False) is False:
                    rows.append(HostPhaseResult(ordinal, True, ServiceCertainty.SUCCEEDED, False))
                    continue
                result = self._host_result(
                    host,
                    scripts.build_cleanup_script(
                        self.request.config.python_bin,
                        self._token,
                        ordinal,
                        self.request.source_manifest,
                    ),
                    60,
                )
                state = _word(result, frozenset({"CLEANED", "RETAINED", "UNKNOWN"}))
                certainty = (
                    ServiceCertainty.SUCCEEDED
                    if state == "CLEANED"
                    else ServiceCertainty.FAILED
                    if state == "RETAINED"
                    else ServiceCertainty.UNKNOWN
                )
                if certainty is ServiceCertainty.SUCCEEDED:
                    probe = self._probe_stage(host, ordinal)
                    certainty = (
                        ServiceCertainty.SUCCEEDED
                        if probe == "ABSENT"
                        else ServiceCertainty.FAILED
                        if probe == "OWNED"
                        else ServiceCertainty.UNKNOWN
                    )
                self._cleanup[ordinal] = certainty
                rows.append(HostPhaseResult(ordinal, True, certainty, True))
            except BaseException as exc:
                self._cleanup[ordinal] = ServiceCertainty.UNKNOWN
                rows.append(HostPhaseResult(ordinal, True, ServiceCertainty.UNKNOWN, True))
                if interruption is None or (isinstance(interruption, Exception) and not isinstance(exc, Exception)):
                    interruption = exc
        if interruption is not None:
            raise interruption
        return FleetPhaseResult(tuple(rows))

    def _unit(self, host: HostConfig, ordinal: int) -> str:
        python = resolve_python(self.request.config.python_bin)
        python_uses_home = python.startswith("%h")
        site = (
            str(self.request.source_root.parent)
            if self._local[ordinal - 1]
            else scripts.systemd_site(self._token, ordinal)
        )
        exec_start = " ".join(
            (
                systemd_quote(python, preserve_home_specifier=python_uses_home),
                systemd_quote("-m"),
                systemd_quote("dgx_monarch.cli.worker_loop"),
                systemd_quote("--address"),
                systemd_quote(host.address),
            )
        )
        environment = systemd_quote(f"PYTHONPATH={site}", preserve_home_specifier=not self._local[ordinal - 1])
        bytecode = " ".join(
            (
                systemd_quote("PYTHONDONTWRITEBYTECODE=1"),
                systemd_quote(f"PYTHONPYCACHEPREFIX={SOURCE_ONLY_PYCACHE_PREFIX}"),
            )
        )
        return f"""# dgxm-setup-token={self._token} ordinal={ordinal} source={self.request.source_manifest}
[Unit]
Description=dgx-monarch worker service
Wants=network-online.target
After=network-online.target

[Service]
Environment={environment}
Environment={bytecode}
{systemd_generation_invalidator()}
ExecStart={exec_start}
UMask=0077
Restart=on-failure
RestartSec=3
TimeoutStopSec=20

[Install]
WantedBy=default.target
"""

    def _host_result(self, host: HostConfig, script: str, timeout: int) -> subprocess.CompletedProcess[str] | None:
        try:
            return self._host_run(self.request.config, host, script, timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
