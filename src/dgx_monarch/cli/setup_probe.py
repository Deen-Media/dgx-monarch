"""Probe only the hosts setup was given, parse each size-bounded reply, and compare artifacts across hosts."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal, TypeAlias, cast

from ..config import ClusterConfig
from ..operator_profiles import HardwareObservation
from .setup_probe_script import MARKER as _MARKER
from .setup_probe_script import build_probe_script as _build_probe_script

ProbeFailure: TypeAlias = Literal["transport_unavailable", "probe_failed", "invalid_response", "python_unavailable"]
ArtifactState: TypeAlias = Literal["match", "missing", "mismatch", "unknown"]
RunHost = Callable[..., subprocess.CompletedProcess[str]]

_HASH = re.compile(r"[0-9a-f]{64}")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,63}")
_LINK_LAYERS = frozenset({"Ethernet", "InfiniBand", "Unknown"})
_MAX_ARTIFACTS = 64
_MAX_RESPONSE_BYTES = 1_000_000


@dataclass(frozen=True)
class ArtifactProbe:
    ordinal: int
    exists: bool
    regular: bool
    size: int | None
    sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "exists": self.exists,
            "regular": self.regular,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HostProbe:
    ordinal: int
    reachable: bool
    failure: ProbeFailure | None
    python_available: bool
    python_version: str | None
    torch_version: str | None
    cuda_available: bool | None
    gpu_count: int | None
    integrated: bool | None
    model_fingerprint: str | None
    torchmonarch_version: str | None
    comfy_exists: bool | None
    comfy_git: bool | None
    comfy_dirty: bool | None
    comfy_commit: str | None
    fabric_interface_count: int | None
    link_layers: tuple[str, ...]
    rsync_available: bool | None
    systemd_user_available: bool | None
    linger_enabled: bool | None
    service_installed: bool | None
    systemd_service_active: bool | None
    worker_process_active: bool | None
    worker_listener_active: bool | None
    service_active: bool | None
    artifacts: tuple[ArtifactProbe, ...]
    comfy_runtime_marker: bool | None = None
    comfy_only_missing_examples: bool | None = None

    def hardware_observation(self) -> HardwareObservation:
        """Return the GPU-model fingerprint and UMA flag; operator_profiles owns the policy that reads them."""
        return HardwareObservation(model=self.model_fingerprint, uma=self.integrated)

    def as_dict(self) -> dict[str, object]:
        """Name the host by ordinal only: never add a host name, address, path or error text."""
        return {
            "ordinal": self.ordinal,
            "reachable": self.reachable,
            "failure": self.failure,
            "python_available": self.python_available,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "gpu_count": self.gpu_count,
            "integrated": self.integrated,
            "model_fingerprint": self.model_fingerprint,
            "torchmonarch_version": self.torchmonarch_version,
            "comfy_exists": self.comfy_exists,
            "comfy_runtime_marker": self.comfy_runtime_marker,
            "comfy_git": self.comfy_git,
            "comfy_dirty": self.comfy_dirty,
            "comfy_only_missing_examples": self.comfy_only_missing_examples,
            "comfy_commit": self.comfy_commit,
            "fabric_interface_count": self.fabric_interface_count,
            "link_layers": list(self.link_layers),
            "rsync_available": self.rsync_available,
            "systemd_user_available": self.systemd_user_available,
            "linger_enabled": self.linger_enabled,
            "service_installed": self.service_installed,
            "systemd_service_active": self.systemd_service_active,
            "worker_process_active": self.worker_process_active,
            "worker_listener_active": self.worker_listener_active,
            "service_active": self.service_active,
            "artifacts": [artifact.as_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True)
class ArtifactComparison:
    ordinal: int
    state: ArtifactState
    expected_hosts: int
    present_hosts: int
    distinct_hashes: int
    sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "state": self.state,
            "expected_hosts": self.expected_hosts,
            "present_hosts": self.present_hosts,
            "distinct_hashes": self.distinct_hashes,
            "sha256": self.sha256,
        }


def validate_artifact_paths(values: Sequence[str]) -> tuple[str, ...]:
    """Deduplicate the paths in order; raise ValueError unless each is a relative POSIX path under ComfyUI."""
    if isinstance(values, (str, bytes)) or len(values) > _MAX_ARTIFACTS:
        raise ValueError(f"artifacts must be a sequence of at most {_MAX_ARTIFACTS} paths")
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise ValueError("artifact paths must be non-empty strings no longer than 512 characters")
        if "\\" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("artifact paths must not contain backslashes or control characters")
        path = PurePosixPath(value)
        if path.is_absolute() or value.startswith("~") or path == PurePosixPath("."):
            raise ValueError("artifact paths must be relative to the configured ComfyUI directory")
        if path.as_posix() != value or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("artifact paths must not contain empty, dot, or parent components")
        normalized = path.as_posix()
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return tuple(result)


def probe_hosts(
    config: ClusterConfig,
    *,
    artifacts: Sequence[str] = (),
    runner: RunHost | None = None,
    timeout: int = 300,
) -> tuple[HostProbe, ...]:
    """Probe each host in ``config.hosts`` in rank order, one at a time; never look for other hosts."""
    selected = validate_artifact_paths(artifacts)
    if runner is None:
        from .lifecycle import run_on_host

        runner = run_on_host
    results: list[HostProbe] = []
    for ordinal, host in enumerate(config.hosts, 1):
        comfy_dir = host.comfy_dir or config.comfy_dir or "~/ComfyUI"
        script = build_probe_script(config.python_bin, comfy_dir, selected, worker_address=host.address)
        try:
            completed = runner(config, host, script, timeout=timeout)
        except (OSError, subprocess.SubprocessError, UnicodeError):
            results.append(_failed(ordinal, "transport_unavailable", len(selected)))
            continue
        if completed.returncode != 0:
            results.append(_failed(ordinal, "probe_failed", len(selected)))
            continue
        results.append(parse_probe_output(completed.stdout, ordinal, len(selected)))
    return tuple(results)


def compare_artifacts(probes: Sequence[HostProbe], artifact_count: int) -> tuple[ArtifactComparison, ...]:
    comparisons: list[ArtifactComparison] = []
    expected = len(probes)
    for ordinal in range(1, artifact_count + 1):
        rows = [probe.artifacts[ordinal - 1] for probe in probes if len(probe.artifacts) >= ordinal]
        present = [row for row in rows if row.exists and row.regular and row.sha256 is not None]
        hashes = {cast(str, row.sha256) for row in present}
        sizes = {row.size for row in present}
        if len(rows) != expected or any(probe.failure is not None for probe in probes):
            state: ArtifactState = "unknown"
        elif len(present) != expected:
            state = "missing"
        elif len(hashes) != 1 or len(sizes) != 1:
            state = "mismatch"
        else:
            state = "match"
        comparisons.append(
            ArtifactComparison(
                ordinal=ordinal,
                state=state,
                expected_hosts=expected,
                present_hosts=len(present),
                distinct_hashes=len(hashes),
                sha256=next(iter(hashes)) if state == "match" else None,
            )
        )
    return tuple(comparisons)


def build_probe_script(
    python_bin: str,
    comfy_dir: str,
    artifacts: Sequence[str],
    *,
    worker_address: str = "",
) -> str:
    return _build_probe_script(python_bin, comfy_dir, artifacts, worker_address=worker_address)


def parse_probe_output(stdout: str, ordinal: int, artifact_count: int) -> HostProbe:
    if not isinstance(stdout, str) or len(stdout.encode("utf-8", errors="ignore")) > _MAX_RESPONSE_BYTES:
        return _failed(ordinal, "invalid_response", artifact_count)
    marked = [line[len(_MARKER) :] for line in stdout.splitlines() if line.startswith(_MARKER)]
    if len(marked) != 1:
        return _failed(ordinal, "invalid_response", artifact_count)
    try:
        raw = json.loads(marked[0])
        return _parse_payload(raw, ordinal, artifact_count)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError):
        return _failed(ordinal, "invalid_response", artifact_count)


def _parse_payload(raw: object, ordinal: int, artifact_count: int) -> HostProbe:
    if not isinstance(raw, dict) or raw.get("schema") != 1:
        raise ValueError("unsupported probe response")
    python_available = _bool(raw, "python_available", required=True)
    if not python_available:
        return _failed(ordinal, "python_unavailable", artifact_count)
    python_version = _version(raw.get("python_version"))
    if python_version is None:
        raise ValueError("invalid Python version")
    models = raw.get("gpu_models")
    if not isinstance(models, list) or not all(isinstance(model, str) for model in models):
        raise ValueError("invalid GPU models")
    model_fingerprint = None
    if models:
        model_fingerprint = hashlib.sha256(
            json.dumps(models, sort_keys=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    artifacts = _parse_artifacts(raw.get("artifacts"), artifact_count)
    layers = raw.get("link_layers", [])
    if not isinstance(layers, list) or not all(layer in _LINK_LAYERS for layer in layers):
        raise ValueError("invalid link layers")
    commit = raw.get("comfy_commit")
    commit = commit if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) else None
    systemd_active = _bool(raw, "systemd_service_active")
    worker_active = _bool(raw, "worker_process_active")
    listener_active = _bool(raw, "worker_listener_active")
    activity = (systemd_active, worker_active, listener_active)
    service_active = True if True in activity else (False if all(item is False for item in activity) else None)
    if _bool(raw, "service_active") is not service_active:
        raise ValueError("incoherent worker activity")
    return HostProbe(
        ordinal=ordinal,
        reachable=True,
        failure=None,
        python_available=True,
        python_version=python_version,
        torch_version=_version(raw.get("torch_version")),
        cuda_available=_bool(raw, "cuda_available"),
        gpu_count=_integer(raw.get("gpu_count"), 0, 1024),
        integrated=_bool(raw, "integrated"),
        model_fingerprint=model_fingerprint,
        torchmonarch_version=_version(raw.get("torchmonarch_version")),
        comfy_exists=_bool(raw, "comfy_exists"),
        comfy_runtime_marker=_bool(raw, "comfy_runtime_marker"),
        comfy_git=_bool(raw, "comfy_git"),
        comfy_dirty=_bool(raw, "comfy_dirty"),
        comfy_only_missing_examples=_bool(raw, "comfy_only_missing_examples"),
        comfy_commit=commit,
        fabric_interface_count=_integer(raw.get("fabric_interface_count"), 0, 4096),
        link_layers=tuple(sorted(set(cast(list[str], layers)))),
        rsync_available=_bool(raw, "rsync_available"),
        systemd_user_available=_bool(raw, "systemd_user_available"),
        linger_enabled=_bool(raw, "linger_enabled"),
        service_installed=_bool(raw, "service_installed"),
        systemd_service_active=systemd_active,
        worker_process_active=worker_active,
        worker_listener_active=listener_active,
        service_active=service_active,
        artifacts=artifacts,
    )


def _parse_artifacts(raw: object, count: int) -> tuple[ArtifactProbe, ...]:
    if not isinstance(raw, list) or len(raw) != count:
        raise ValueError("invalid artifact results")
    parsed: list[ArtifactProbe] = []
    for ordinal, row in enumerate(raw, 1):
        if not isinstance(row, dict) or row.get("ordinal") != ordinal:
            raise ValueError("invalid artifact result")
        exists = _bool(row, "exists", required=True)
        regular = _bool(row, "regular", required=True)
        if exists is None or regular is None:
            raise ValueError("missing artifact booleans")
        size = _integer(row.get("size"), 0, 2**63 - 1)
        digest = row.get("sha256")
        digest = digest if isinstance(digest, str) and _HASH.fullmatch(digest) else None
        if (not exists or not regular) and (size is not None or digest is not None):
            raise ValueError("incoherent artifact result")
        if regular and (size is None or digest is None):
            raise ValueError("incomplete artifact result")
        parsed.append(ArtifactProbe(ordinal, exists, regular, size, digest))
    return tuple(parsed)


def _failed(ordinal: int, failure: ProbeFailure, artifact_count: int) -> HostProbe:
    return HostProbe(
        ordinal=ordinal,
        reachable=failure == "python_unavailable",
        failure=failure,
        python_available=False,
        python_version=None,
        torch_version=None,
        cuda_available=None,
        gpu_count=None,
        integrated=None,
        model_fingerprint=None,
        torchmonarch_version=None,
        comfy_exists=None,
        comfy_runtime_marker=None,
        comfy_git=None,
        comfy_dirty=None,
        comfy_commit=None,
        fabric_interface_count=None,
        link_layers=(),
        rsync_available=None,
        systemd_user_available=None,
        linger_enabled=None,
        service_installed=None,
        systemd_service_active=None,
        worker_process_active=None,
        worker_listener_active=None,
        service_active=None,
        artifacts=tuple(
            ArtifactProbe(index, False, False, None, None)
            for index in range(1, artifact_count + 1)
        ),
    )


def _version(value: object) -> str | None:
    return value if isinstance(value, str) and _VERSION.fullmatch(value) else None


def _bool(raw: dict[str, object], key: str, *, required: bool = False) -> bool | None:
    value = raw.get(key)
    if isinstance(value, bool):
        return value
    if value is None and not required:
        return None
    raise ValueError(f"invalid {key}")


def _integer(value: object, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError("invalid integer")
    return value
