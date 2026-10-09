"""Release-slot types, patterns and default runners for update_release.ReleaseManager.

ReleaseManager.prepare runs the private-directory and metadata checks here
before it reserves any remote path.
"""
from __future__ import annotations

import os
import re
import secrets
import stat
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import ClusterConfig, HostConfig
from . import lifecycle
from .update_transaction import Certainty, OperationResult

_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_TOKEN = re.compile(r"u-[0-9a-f]{12}-[0-9a-f]{16}")
_REMOTE_BASE = ".local/share/dgx-monarch"


class ReleaseLayoutError(RuntimeError):
    """The host layout or local private root cannot take a release slot, or the
    slot failed to build, copy or verify, is not ready, or its cleanup target
    escaped its private root."""


class CommandRunner(Protocol):
    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: str | os.PathLike[str] | None = None,
        capture_output: bool = True,
        text: bool = True,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]: ...


HostRunner = Callable[
    [ClusterConfig, HostConfig, str, int], subprocess.CompletedProcess[str]
]


@dataclass(frozen=True)
class ReleaseMetadata:
    target_sha: str
    version: str
    torchmonarch_pin: str
    source_manifest: str
    dependency_manifest: str


@dataclass
class ReleaseSlot:
    token: str
    root: Path
    site: Path
    metadata: ReleaseMetadata
    pin_payload_digest: str = ""
    site_manifest: str = ""
    verifier_digest: str = ""
    prior_metadata: ReleaseMetadata | None = None
    prior_pin_payload_digest: str = ""
    remote_staged: int = 0
    remote_activated: bool = False
    activation_ambiguous: bool = False
    finalized: bool = False
    installations: dict[str, dict[str, Any]] = field(default_factory=dict)
    reservation: str = field(default_factory=lambda: secrets.token_hex(16))


def _default_host_runner(
    config: ClusterConfig, host: HostConfig, script: str, timeout: int
) -> subprocess.CompletedProcess[str]:
    return lifecycle.run_on_host(config, host, script, timeout=timeout, require_known_locality=True)


def _default_command_runner(
    argv: Sequence[str],
    *,
    cwd: str | os.PathLike[str] | None = None,
    capture_output: bool = True,
    text: bool = True,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv), cwd=cwd, capture_output=capture_output, text=text,
        timeout=timeout, env=env,
    )


def _result(certainty: Certainty) -> OperationResult:
    return OperationResult(certainty)


def _private_directory(path: Path) -> Path:
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ReleaseLayoutError("private release root is not a real directory")
    if hasattr(os, "geteuid") and metadata.st_uid != os.geteuid():
        raise ReleaseLayoutError("private release root has the wrong owner")
    path.chmod(0o700)
    return path


def _validate_metadata(metadata: ReleaseMetadata) -> None:
    if _SHA.fullmatch(metadata.target_sha) is None:
        raise ValueError("release target must be a full commit")
    if _DIGEST.fullmatch(metadata.source_manifest) is None:
        raise ValueError("release source manifest must be a SHA-256 digest")
    if _DIGEST.fullmatch(metadata.dependency_manifest) is None:
        raise ValueError("release dependency manifest must be a SHA-256 digest")
    for label, value in (
        ("version", metadata.version),
        ("torchmonarch pin", metadata.torchmonarch_pin),
    ):
        if not value or len(value) > 128 or any(ch.isspace() or ord(ch) < 32 for ch in value):
            raise ValueError(f"release {label} is invalid")
