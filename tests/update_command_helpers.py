"""Checkout, release and driver-pin doubles shared by the update command suites."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dgx_monarch import __version__
from dgx_monarch.cli import update_command, update_entrypoint
from dgx_monarch.cli.update_lock import UpdateLock
from dgx_monarch.cli.update_release import ReleaseMetadata
from dgx_monarch.cli.update_transaction import (
    Certainty,
    OperationResult,
    StagedRelease,
)
from dgx_monarch.config import ClusterConfig, HostConfig

RELEASE_ID = "u-000000000000-1111111111111111"
SOURCE = "a" * 64
DEPS = "b" * 64


@pytest.fixture(autouse=True)
def _isolated_update_locks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        update_entrypoint, "UpdateLock",
        lambda repo: UpdateLock(repo, state_root=tmp_path / "update-locks"),
    )


def _run(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, check=True)


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    package = repo / "src" / "dgx_monarch"
    package.mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        """[project]
dependencies = ["torchmonarch==0.6.0", "xfuser==0.7.0+dgxm.npuimport1", "yunchang>=0.6.4"]
""",
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text(
        "torchmonarch==0.6.0\nxfuser==0.7.0+dgxm.npuimport1\nyunchang>=0.6.4\n",
        encoding="utf-8",
    )
    (package / "__init__.py").write_text(
        f'__version__ = "{__version__}"\nTORCHMONARCH_PIN = "0.6.0"\n',
        encoding="utf-8",
    )
    (package / "worker.py").write_text("VALUE = 1\n", encoding="utf-8")
    _run("git", "init", "-q", cwd=repo)
    _run("git", "config", "user.email", "test" + "@example.invalid", cwd=repo)
    _run("git", "config", "user.name", "test", cwd=repo)
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "base", cwd=repo)
    return repo, _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()


def _config() -> ClusterConfig:
    return ClusterConfig(
        hosts=(HostConfig("remote", "tcp://192.0.2.10:26600", gpus=2),),
        transport_security="trusted_fabric",
    )


def _metadata(repo: Path, sha: str) -> ReleaseMetadata:
    metadata, _ = update_command._inspect_checkout(repo, sha)
    return metadata


def _target(repo: Path, *, pin: str | None = None) -> str:
    (repo / "src" / "dgx_monarch" / "worker.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )
    if pin is not None:
        for relative in ("pyproject.toml", "requirements.txt", "src/dgx_monarch/__init__.py"):
            path = repo / relative
            path.write_text(
                path.read_text(encoding="utf-8").replace("0.6.0", pin),
                encoding="utf-8",
            )
    _run("git", "add", ".", cwd=repo)
    _run("git", "commit", "-qm", "target", cwd=repo)
    return _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()


def _stage(metadata: ReleaseMetadata, *, ranks: int = 2) -> StagedRelease:
    return StagedRelease(
        metadata.target_sha, metadata.version, metadata.torchmonarch_pin,
        metadata.source_manifest, metadata.dependency_manifest,
        True, True, True, True, 1, 1, ranks,
    )


class _PinHarness:
    def __init__(self) -> None:
        self.live = "prior"
        self.target_site_ok = True
        self.prepared: tuple[Path, str, str] | None = None
        self.promote_result = OperationResult(Certainty.SUCCEEDED)
        self.restore_result = OperationResult(Certainty.SUCCEEDED)

    def prepare(self, root: Path, prior: str, target: str) -> None:
        self.prepared = (root, prior, target)

    def matches_prior(self) -> bool:
        return self.live == "prior"

    def matches_target(self) -> bool:
        return self.live == "target"

    def target_matches_site(self, _site: Path) -> bool:
        return self.target_site_ok

    @property
    def prior_payload_digest(self) -> str:
        return "d" * 64

    def promote(self) -> OperationResult:
        if self.promote_result.certainty == Certainty.SUCCEEDED:
            self.live = "target"
        return self.promote_result

    def restore_prior(self) -> OperationResult:
        if self.restore_result.certainty == Certainty.SUCCEEDED:
            self.live = "prior"
        return self.restore_result
