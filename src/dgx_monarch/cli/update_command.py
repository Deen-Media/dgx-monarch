from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import cast

from ..config import ClusterConfig
from . import comfy_ports
from .cluster_smoke import exact_pass_evidence
from .update_cancellation import run_finalizer_pair
from .update_driver_pin import DriverPinError, DriverPinTransition
from .update_entrypoint import run_serialized_update
from .update_inspect import inspect_checkout as _inspect_checkout
from .update_release import CommandRunner, ReleaseLayoutError, ReleaseManager, ReleaseMetadata
from .update_site_probe import site_matches
from .update_stage_cleanup import settle_stage_failure
from .update_support import comfy_candidate_count as _comfy_candidate_count
from .update_support import dependencies_available as _dependencies_available
from .update_support import marked_actor_count as _count_marked_actors
from .update_support import private_root as _private_root
from .update_support import runtime_env as _runtime_env
from .update_target import resolve_target as _resolve_target
from .update_transaction import (
    DEFAULT_TARGET,
    ActivitySnapshot,
    Certainty,
    ExactReadback,
    OperationResult,
    RepoSnapshot,
    StagedRelease,
    UpdateOps,
    UpdatePreconditionError,
    UpdateRequest,
)
from .update_worker_lifecycle import (
    LIFECYCLE_UNKNOWN_EXIT,
    active_generation,
    operation_from_exit,
    prior_generation,
    start_script,
    stop_active_generation,
)
from .update_worktree import remove_worktree

_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


_OK = OperationResult(Certainty.SUCCEEDED)
_FAILED = OperationResult(Certainty.FAILED)
_UNKNOWN = OperationResult(Certainty.UNKNOWN)


class SystemUpdateOps(UpdateOps):
    def __init__(
        self,
        repo: Path,
        config: ClusterConfig,
        *,
        driver_host: str | None = None,
        command_runner: CommandRunner = cast(CommandRunner, subprocess.run),
        release_manager: ReleaseManager | None = None,
        pin_transition: DriverPinTransition | None = None,
        worktree_root: Path | None = None,
    ) -> None:
        self.repo = repo.resolve()
        self._config_path = Path(config.source).expanduser().resolve()
        self.config = replace(config, source=str(self._config_path))
        self.driver_host = driver_host
        self._run = command_runner
        self._releases = release_manager or ReleaseManager(self.config, command_runner=command_runner)
        self._pins = pin_transition or DriverPinTransition(command_runner=command_runner)
        self._worktree_root = worktree_root or Path.home() / ".local" / "state" / "dgx-monarch" / "update-worktrees"
        self._config_identity = hashlib.sha256(self._config_path.read_bytes()).digest() if self._config_path.is_file() else b""
        self._checkout: Path | None = None
        self._prior_sha = ""
        self._resolved = False

    def _git(self, *args: str, cwd: Path | None = None, timeout: float = 60) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
             "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false",
             "-C", str(cwd or self.repo), *args],
            capture_output=True, text=True, timeout=timeout,
            env={
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.defpath,
            },
        )

    def resolve_target(self, target_ref: str) -> str:
        if self._resolved:
            raise RuntimeError("update target resolution was repeated")
        self._resolved = True
        return _resolve_target(target_ref, self._git)

    def repo_snapshot(self, target_sha: str) -> RepoSnapshot:
        if _SHA.fullmatch(target_sha) is None:
            raise RuntimeError("update target is not a full commit")
        if not self._config_current():
            raise RuntimeError("cluster config changed during verified update")
        slot = getattr(self._releases, "slot", None)
        if slot is not None and slot.prior_metadata is not None and not slot.remote_activated and not self._releases.prior_hosts_match():
            raise UpdatePreconditionError("prior_worker_changed", "worker release changed after staging")
        head = self._git("rev-parse", "HEAD")
        status = self._git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
        assume = self._git("ls-files", "-v")
        sparse = self._git("ls-files", "-t")
        if any(result.returncode != 0 for result in (head, status, assume, sparse)):
            raise RuntimeError("repository state query failed")
        hidden = any(line[:1].islower() for line in assume.stdout.splitlines()) or any(
            line.startswith("S ") for line in sparse.stdout.splitlines()
        )
        ff = self._git("merge-base", "--is-ancestor", head.stdout.strip(), target_sha)
        if ff.returncode not in (0, 1):
            raise RuntimeError("fast-forward query failed")
        snapshot = RepoSnapshot(
            current_sha=head.stdout.strip(),
            clean=not status.stdout.strip() and not hidden,
            fast_forward=ff.returncode == 0,
        )
        if not self._prior_sha:
            self._prior_sha = snapshot.current_sha
        return snapshot

    def activity_snapshot(self) -> ActivitySnapshot:
        ports = comfy_ports.find_comfy_ports()
        running, observed = comfy_ports.driver_probe(self.driver_host, ports)
        actors = self._marked_actor_count()
        candidates = _comfy_candidate_count()
        candidate_running = type(candidates) is int and candidates > 0
        ambiguous_driver = running is None and (
            not observed
            or self.driver_host is not None
            or candidates is None
        )
        return ActivitySnapshot(
            comfy_running=running is not None or candidate_running,
            render_active=None if ambiguous_driver or running is not None or candidate_running else False,
            active_leases=None if ambiguous_driver or running is not None or candidate_running else 0,
            marked_actors=actors,
        )

    def _marked_actor_count(self) -> int | None:
        return _count_marked_actors(self.config)

    def stage(self, target_sha: str) -> StagedRelease:
        checkout: Path | None = None
        release_started = False
        try:
            try:
                self._releases.ensure_supported()
            except ReleaseLayoutError:
                raise UpdatePreconditionError(
                    "release_layout_unsupported",
                    "verified update could not prove a supported Worker installation",
                ) from None
            _private_root(self._worktree_root)
            checkout = Path(tempfile.mkdtemp(prefix="update-", dir=self._worktree_root))
            added = self._git("worktree", "add", "--detach", str(checkout), target_sha, timeout=120)
            if added.returncode != 0:
                raise RuntimeError("detached update worktree creation failed")
            self._checkout = checkout
            metadata, dependencies = _inspect_checkout(checkout, target_sha)
            if _SHA.fullmatch(self._prior_sha) is None:
                raise RuntimeError("prior driver commit was not captured")
            prior_metadata, _prior_dependencies = _inspect_checkout(self.repo, self._prior_sha)
            if not _dependencies_available(self.config, dependencies, self._run):
                raise RuntimeError("target dependency contract is not satisfied")
            release_started = True
            slot = self._releases.prepare(checkout, metadata)
            try:
                self._pins.prepare(
                    slot.root, prior_metadata.torchmonarch_pin, metadata.torchmonarch_pin
                )
            except DriverPinError as exc:
                raise RuntimeError("driver pin artifacts could not be staged") from exc
            if not self._pins.matches_prior():
                raise UpdatePreconditionError(
                    "driver_pin_drift", "the driver's torchmonarch does not match the current checkout's exact pin",
                )
            prior_payload = self._pins.prior_payload_digest
            if prior_payload is None or not self._releases.attest_prior(prior_metadata, prior_payload):
                raise UpdatePreconditionError("prior_worker_drift", "worker hosts do not exactly match the current driver release")
            if not self._pins.target_matches_site(slot.site):
                raise RuntimeError("worker and driver target pin artifacts do not match")
            status = self._git("status", "--porcelain=v1", "--untracked-files=all", cwd=checkout)
            if status.returncode != 0 or status.stdout.strip():
                raise RuntimeError("staged checkout changed during release construction")
            staged = StagedRelease(
                target_sha=metadata.target_sha,
                version=metadata.version,
                torchmonarch_pin=metadata.torchmonarch_pin,
                source_manifest=metadata.source_manifest,
                dependency_manifest=metadata.dependency_manifest,
                detached_head=True,
                pin_staged_no_deps=True,
                dependencies_validated=True,
                driver_release_ready=slot.site.is_dir(),
                worker_releases_expected=len(self.config.hosts),
                worker_releases_staged=slot.remote_staged,
                ranks_expected=self.config.world_size,
            )
            return staged
        except BaseException as error:
            settle_stage_failure(
                error, release_started=release_started, cleanup_release=self._releases.cleanup,
                cleanup_worktree=lambda: self._remove_worktree(checkout),
            )

    def confirm(self, staged: StagedRelease) -> bool:
        print(
            f"activate {staged.target_sha} (dgx-monarch {staged.version}, "
            f"torchmonarch {staged.torchmonarch_pin}) on "
            f"{staged.worker_releases_staged} worker services?"
        )
        print("This invalidates source-bound Gate grants; sticky FAIL evidence remains.")
        return input("continue? [y/N] ").strip().lower() == "y"

    def stop_workers(self) -> OperationResult:
        try:
            return self._releases.stop_prior_workers()
        except (OSError, ReleaseLayoutError, subprocess.TimeoutExpired):
            return _UNKNOWN

    def stop_started_workers(self, staged: StagedRelease) -> OperationResult:
        return stop_active_generation(self.config, self._releases, staged)

    def activate(self, staged: StagedRelease) -> OperationResult:
        try:
            snapshot = self.repo_snapshot(staged.target_sha)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return _UNKNOWN
        if snapshot.current_sha != self._prior_sha or not snapshot.clean or not snapshot.fast_forward:
            return _UNKNOWN
        return self._releases.activate_workers()

    def compensate(self, staged: StagedRelease) -> OperationResult:
        try:
            head = self._git("rev-parse", "HEAD")
            status = self._git("status", "--porcelain=v1", "--untracked-files=all")
        except (OSError, subprocess.TimeoutExpired):
            return _UNKNOWN
        if (
            head.returncode != 0 or status.returncode != 0
            or head.stdout.strip() != self._prior_sha or status.stdout.strip()
            or not self._pins.matches_prior()
        ):
            return _UNKNOWN
        return self._releases.compensate_workers()

    def start_target_workers(self, staged: StagedRelease) -> OperationResult:
        generation = active_generation(self._releases, staged)
        if generation is None:
            return _UNKNOWN
        return self._start_workers_without_sync(generation)

    def start_workers_without_sync(self) -> OperationResult:
        return self._start_workers_without_sync(None)

    def start_prior_workers(self, staged: StagedRelease) -> OperationResult:
        generation = prior_generation(self._releases, staged)
        if generation is None:
            return _UNKNOWN
        try:
            if not self._config_current():
                return _FAILED
            return self._releases.start_prior_workers(generation)
        except (OSError, ReleaseLayoutError, subprocess.TimeoutExpired):
            return _UNKNOWN

    def _start_workers_without_sync(self, generation: str | None) -> OperationResult:
        script = start_script()
        try:
            if not self._config_current():
                return _FAILED
            self._releases.ensure_supported()
            site = self._runtime_site()
            result = self._run([sys.executable, "-c", script, str(self._config_path), generation or ""], cwd=self.repo,
                               capture_output=True, text=True, timeout=180, env=_runtime_env(site))
        except (OSError, ReleaseLayoutError, subprocess.TimeoutExpired):
            return _UNKNOWN
        return operation_from_exit(result.returncode)

    def doctor(self) -> OperationResult:
        if not self._config_current():
            return _FAILED
        args = [sys.executable, "-m", "dgx_monarch", "--config", str(self._config_path), "doctor", "--json"]
        try:
            site = self._target_site()
            result = self._run(
                args, cwd=self.repo, capture_output=True, text=True, timeout=180,
                env=_runtime_env(site),
            )
        except (OSError, ReleaseLayoutError, subprocess.TimeoutExpired):
            return _UNKNOWN
        # Doctor exits 75 when no row failed but a critical probe never answered.
        # That is no reason to restore the prior release over a started target.
        return operation_from_exit(result.returncode)

    def cluster_smoke(self, staged: StagedRelease) -> OperationResult:
        if not self._config_current():
            return _FAILED
        script = (
            "import sys; from dgx_monarch.cli.cluster_smoke import smoke_main; "
            "raise SystemExit(smoke_main(sys.argv[1]))"
        )
        try:
            site = self._target_site()
            result = self._run(
                [sys.executable, "-c", script, str(self._config_path)], cwd=self.repo,
                capture_output=True, text=True, timeout=600, env=_runtime_env(site),
            )
            # Transport loss or RPC timeout exits 75: the smoke result is unknown.
            if result.returncode == LIFECYCLE_UNKNOWN_EXIT:
                return _UNKNOWN
            if result.returncode != 0:
                return _FAILED
            payload = json.loads(result.stdout.splitlines()[-1])
        except (IndexError, json.JSONDecodeError, OSError, ReleaseLayoutError,
                subprocess.TimeoutExpired):
            return _UNKNOWN
        if not isinstance(payload, dict) or payload.get("result") != "PASS":
            return _FAILED
        self._smoke = payload
        return _OK

    def exact_readback(self, staged: StagedRelease) -> ExactReadback:
        smoke = getattr(self, "_smoke", None) or {}
        try:
            if self._checkout is None or not self._config_current():
                raise RuntimeError("staging state changed")
            live = self.repo_snapshot(staged.target_sha)
            metadata, _dependencies = _inspect_checkout(self._checkout, staged.target_sha)
            hosts_match = self._releases.readback_hosts()
            target_site = self._target_site()
            site_match = self._site_matches(target_site, staged)
            pin_site_match = self._pins.target_matches_site(target_site)
        except Exception:
            return ExactReadback(Certainty.UNKNOWN, staged.ranks_expected, 0)
        matched = staged.ranks_expected if (
            hosts_match
            and site_match
            and pin_site_match
            and live.current_sha == self._prior_sha
            and live.clean
            and metadata == ReleaseMetadata(
                staged.target_sha, staged.version, staged.torchmonarch_pin,
                staged.source_manifest, staged.dependency_manifest,
            )
            and exact_pass_evidence(
                smoke, staged.ranks_expected, staged.source_manifest)
        ) else 0
        certainty = Certainty.SUCCEEDED if matched else Certainty.FAILED
        return ExactReadback(
            certainty, staged.ranks_expected, matched, staged.target_sha, metadata.version,
            metadata.torchmonarch_pin, metadata.source_manifest,
        )

    def commit_driver(self, staged: StagedRelease) -> OperationResult:
        try:
            before = self.repo_snapshot(staged.target_sha)
        except (OSError, RuntimeError):
            return _UNKNOWN
        try:
            target_exact = self._pins.target_matches_site(self._target_site())
            workers_exact = self._releases.readback_hosts()
        except Exception:
            return _UNKNOWN
        if before.current_sha != self._prior_sha or not before.clean or not self._pins.matches_prior() or not target_exact or not workers_exact:
            return _UNKNOWN
        promoted = self._pins.promote()
        if promoted.certainty != Certainty.SUCCEEDED:
            return promoted
        try:
            merged = self._git("merge", "--ff-only", staged.target_sha, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return _UNKNOWN
        if merged.returncode == 0 and self._driver_target_matches(staged):
            return _OK
        try:
            unchanged = self.repo_snapshot(staged.target_sha)
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return _UNKNOWN
        if merged.returncode != 0 and unchanged.current_sha == self._prior_sha and unchanged.clean:
            restored = self._pins.restore_prior()
            if restored.certainty == Certainty.SUCCEEDED and self._driver_prior_matches(staged):
                return _FAILED
        return _UNKNOWN

    def finalize(self, staged: StagedRelease) -> OperationResult:
        if not self._driver_target_matches(staged):
            return _UNKNOWN
        return self._releases.finalize()

    def prior_driver_readback(self, staged: StagedRelease) -> OperationResult:
        return _OK if self._driver_prior_matches(staged) else _UNKNOWN

    def _driver_target_matches(self, staged: StagedRelease) -> bool:
        try:
            metadata, _dependencies = _inspect_checkout(self.repo, staged.target_sha)
            snapshot = self.repo_snapshot(staged.target_sha)
            return snapshot.clean and metadata == ReleaseMetadata(
                staged.target_sha, staged.version, staged.torchmonarch_pin,
                staged.source_manifest, staged.dependency_manifest,
            ) and self._pins.matches_target() and self._site_matches(self.repo / "src", staged)
        except Exception:
            return False

    def _driver_prior_matches(self, staged: StagedRelease) -> bool:
        try:
            snapshot = self.repo_snapshot(staged.target_sha)
            return (
                snapshot.current_sha == self._prior_sha and snapshot.clean
                and self._pins.matches_prior()
            )
        except Exception:
            return False

    def _site_matches(self, site: Path, staged: StagedRelease) -> bool:
        return site_matches(site, staged.version, staged.torchmonarch_pin, staged.source_manifest, self._run)

    def _runtime_site(self) -> Path:
        slot = self._releases.slot
        if slot is None:
            raise ReleaseLayoutError("release slot was not prepared")
        return slot.site if slot.remote_activated else self.repo / "src"

    def _target_site(self) -> Path:
        slot = self._releases.slot
        if slot is None or not slot.remote_activated or not self._releases.readback_hosts():
            raise ReleaseLayoutError("target release is not active")
        return slot.site

    def _config_current(self) -> bool:
        try:
            return bool(
                self._config_identity and self._config_path.is_file()
                and hashlib.sha256(self._config_path.read_bytes()).digest() == self._config_identity
            )
        except OSError:
            return False

    def cleanup_stage(self, staged: StagedRelease) -> OperationResult:
        release, worktree = run_finalizer_pair(
            self._releases.cleanup, lambda: self._remove_worktree(self._checkout)
        )
        if release.certainty == Certainty.UNKNOWN or worktree.certainty == Certainty.UNKNOWN:
            return _UNKNOWN
        if release.certainty == Certainty.FAILED or worktree.certainty == Certainty.FAILED:
            return _FAILED
        return _OK

    def _remove_worktree(self, checkout: Path | None) -> OperationResult:
        result = remove_worktree(checkout, self._worktree_root, self._git)
        if result.certainty == Certainty.SUCCEEDED:
            self._checkout = None
        return result


def run_verified_update(
    config: ClusterConfig,
    *,
    repo: Path,
    target_ref: str = DEFAULT_TARGET,
    assume_yes: bool = False,
    driver_host: str | None = None,
    receipt_path: str | os.PathLike[str] | None = None,
    ops: UpdateOps | None = None,
) -> int:
    if ops is None:
        from .update_bootstrap import launch_verified_update

        return launch_verified_update(
            config, repo=repo, target_ref=target_ref, assume_yes=assume_yes,
            driver_host=driver_host, receipt_path=receipt_path,
        )
    return run_serialized_update(repo=repo, request=UpdateRequest(target_ref, assume_yes), ops=ops, receipt_path=receipt_path)
