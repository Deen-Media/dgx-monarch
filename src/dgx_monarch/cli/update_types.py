"""Verified-update results that distinguish success, failure and unknown outcomes."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

DEFAULT_TARGET = "origin/HEAD"


class Certainty(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OperationResult:
    certainty: Certainty


@dataclass(frozen=True)
class RepoSnapshot:
    current_sha: str
    clean: bool
    fast_forward: bool


@dataclass(frozen=True)
class ActivitySnapshot:
    comfy_running: bool | None
    render_active: bool | None
    active_leases: int | None
    marked_actors: int | None

    def blockers(self) -> tuple[str, ...]:
        active_leases = self.active_leases
        marked_actors = self.marked_actors
        blockers: list[str] = []
        if self.comfy_running is True:
            blockers.append("comfy_running")
        if self.render_active is True:
            blockers.append("render_active")
        if type(active_leases) is int and active_leases > 0:
            blockers.append("active_leases")
        if type(marked_actors) is int and marked_actors > 0:
            blockers.append("marked_actors")
        if blockers:
            return tuple(blockers)
        if (
            not isinstance(self.comfy_running, bool)
            or not isinstance(self.render_active, bool)
            or type(active_leases) is not int
            or type(marked_actors) is not int
            or active_leases < 0
            or marked_actors < 0
        ):
            return ("activity_unknown",)
        return ()


@dataclass(frozen=True)
class StagedRelease:
    target_sha: str
    version: str
    torchmonarch_pin: str
    source_manifest: str
    dependency_manifest: str
    detached_head: bool
    pin_staged_no_deps: bool
    dependencies_validated: bool
    driver_release_ready: bool
    worker_releases_expected: int
    worker_releases_staged: int
    ranks_expected: int


@dataclass(frozen=True)
class ExactReadback:
    certainty: Certainty
    ranks_expected: int
    ranks_matched: int
    target_sha: str = ""
    version: str = ""
    torchmonarch_pin: str = ""
    source_manifest: str = ""


@dataclass(frozen=True)
class UpdateRequest:
    target_ref: str = DEFAULT_TARGET
    assume_yes: bool = False


@dataclass(frozen=True)
class UpdateResult:
    exit_code: int
    code: str
    message: str
    receipt: Mapping[str, object]

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


class UpdatePreconditionError(RuntimeError):
    """A refusal with a stable code: ops raise it before mutation; update_transaction's aborts subclass it."""

    def __init__(self, code: str, message: str, exit_code: int = 2) -> None:
        self.code, self.message, self.exit_code = code, message, exit_code
        super().__init__(message)


class UpdateOps(Protocol):
    """All effects required by the verified-update state machine."""

    def resolve_target(self, target_ref: str) -> str: ...
    def repo_snapshot(self, target_sha: str) -> RepoSnapshot: ...
    def activity_snapshot(self) -> ActivitySnapshot: ...
    def stage(self, target_sha: str) -> StagedRelease: ...
    def confirm(self, staged: StagedRelease) -> bool: ...
    def stop_workers(self) -> OperationResult: ...
    def stop_started_workers(self, staged: StagedRelease) -> OperationResult: ...
    def activate(self, staged: StagedRelease) -> OperationResult: ...
    def compensate(self, staged: StagedRelease) -> OperationResult: ...
    def start_target_workers(self, staged: StagedRelease) -> OperationResult: ...
    def start_prior_workers(self, staged: StagedRelease) -> OperationResult: ...
    def start_workers_without_sync(self) -> OperationResult: ...
    def doctor(self) -> OperationResult: ...
    def cluster_smoke(self, staged: StagedRelease) -> OperationResult: ...
    def exact_readback(self, staged: StagedRelease) -> ExactReadback: ...
    def commit_driver(self, staged: StagedRelease) -> OperationResult: ...
    def prior_driver_readback(self, staged: StagedRelease) -> OperationResult: ...
    def finalize(self, staged: StagedRelease) -> OperationResult: ...
    def cleanup_stage(self, staged: StagedRelease) -> OperationResult: ...
