"""Driver-pin staging, promotion and compensation regressions."""
from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch.cli import update_command
from dgx_monarch.cli.update_release import ReleaseMetadata
from dgx_monarch.cli.update_transaction import Certainty, OperationResult
from dgx_monarch.config import ClusterConfig, HostConfig
from update_command_helpers import (  # noqa: F401  # autouse fixture import.
    DEPS,
    SOURCE,
    _checkout,
    _isolated_update_locks,
    _metadata,
    _PinHarness,
    _run,
    _stage,
    _target,
)


def test_relative_config_is_bound_once_for_lifecycle_checks_and_readback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "configs" / "cluster.toml"
    config_path.parent.mkdir()
    config_path.write_text("[cluster]\n", encoding="utf-8")
    decoy = repo / "configs" / "cluster.toml"
    decoy.parent.mkdir()
    decoy.write_text("[cluster]\nauto_heal = false\n", encoding="utf-8")
    target_site = tmp_path / "target-site"
    target_site.mkdir()
    release = SimpleNamespace(
        ensure_supported=lambda: None,
        slot=SimpleNamespace(site=target_site, remote_activated=True),
    )
    calls: list[list[str]] = []

    def runner(argv, **_kwargs):
        calls.append(list(argv))
        output = '{"result":"PASS"}\n' if "smoke_main" in " ".join(argv) else ""
        return subprocess.CompletedProcess(list(argv), 0, output, "")

    monkeypatch.chdir(tmp_path)
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source="configs/cluster.toml"), command_runner=runner,
        release_manager=release,
    )
    stopped_with: list[str] = []
    release.stop_prior_workers = lambda: stopped_with.append(ops.config.source) or OperationResult(Certainty.SUCCEEDED)
    monkeypatch.setattr(ops, "_target_site", lambda: target_site)

    assert ops.stop_workers().certainty == Certainty.SUCCEEDED
    assert ops.start_workers_without_sync().certainty == Certainty.SUCCEEDED
    assert ops.doctor().certainty == Certainty.SUCCEEDED
    assert ops.cluster_smoke(_stage(ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS))).certainty == Certainty.SUCCEEDED
    canonical = str(config_path.resolve())
    assert ops.config.source == canonical
    assert stopped_with == [canonical]
    assert all(canonical in argv for argv in calls)
    decoy.write_text("[cluster]\n", encoding="utf-8")
    assert ops._config_current()
    config_path.write_text("[cluster]\nauto_heal = false\n", encoding="utf-8")
    assert not ops._config_current()


def test_target_pin_transition_is_fully_staged_before_live_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, prior = _checkout(tmp_path)
    target = _target(repo, pin="0.7.0")
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    root = tmp_path / "release"
    site = root / "site"
    site.mkdir(parents=True)
    cleaned: list[bool] = []
    release = SimpleNamespace(
        ensure_supported=lambda: None,
        prepare=lambda *_args: SimpleNamespace(root=root, site=site, remote_staged=1),
        attest_prior=lambda *_args: True,
        cleanup=lambda: cleaned.append(True) or OperationResult(Certainty.SUCCEEDED),
    )
    pins = _PinHarness()
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(
            hosts=(HostConfig("remote", "tcp://192.0.2.10:26600", gpus=2),),
            transport_security="trusted_fabric", source=str(config_path),
        ),
        release_manager=release, pin_transition=pins,  # type: ignore[arg-type]
        worktree_root=tmp_path / "worktrees",
    )
    ops.repo_snapshot(target)
    monkeypatch.setattr(update_command, "_dependencies_available", lambda *_args: True)

    staged = ops.stage(target)

    assert staged.torchmonarch_pin == "0.7.0"
    assert pins.prepared == (root, "0.6.0", "0.7.0")
    assert _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip() == prior
    assert ops.cleanup_stage(staged).certainty == Certainty.SUCCEEDED
    assert cleaned == [True]


def test_post_build_git_status_query_failure_cleans_owned_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    root, site = tmp_path / "release", tmp_path / "release" / "site"
    site.mkdir(parents=True)
    cleaned: list[bool] = []
    release = SimpleNamespace(
        ensure_supported=lambda: None,
        prepare=lambda *_args: SimpleNamespace(root=root, site=site, remote_staged=1),
        attest_prior=lambda *_args: True,
        cleanup=lambda: cleaned.append(True) or OperationResult(Certainty.SUCCEEDED),
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
        pin_transition=_PinHarness(),  # type: ignore[arg-type]
        worktree_root=tmp_path / "worktrees",
    )
    ops.repo_snapshot(sha)
    monkeypatch.setattr(update_command, "_dependencies_available", lambda *_args: True)
    real_git = ops._git
    def broken_status(*args, cwd=None, timeout=60):
        result = real_git(*args, cwd=cwd, timeout=timeout)
        if args[:1] == ("status",) and cwd is not None:
            return subprocess.CompletedProcess(result.args, 1, "", "query failed")
        return result
    monkeypatch.setattr(ops, "_git", broken_status)

    with pytest.raises(RuntimeError, match="release staging failed"):
        ops.stage(sha)

    assert cleaned == [True]
    assert not list((tmp_path / "worktrees").glob("update-*"))


def test_activation_switches_only_remote_slots_before_proof(tmp_path: Path):
    repo, prior = _checkout(tmp_path)
    target = _target(repo)
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    activated: list[bool] = []
    release = SimpleNamespace(
        activate_workers=lambda: activated.append(True)
        or OperationResult(Certainty.SUCCEEDED)
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release
    )
    ops.repo_snapshot(target)

    result = ops.activate(_stage(metadata))

    assert result.certainty == Certainty.SUCCEEDED
    assert activated == [True]
    assert _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip() == prior


def test_target_checks_run_from_private_staged_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    target_site = tmp_path / "target-site"
    target_site.mkdir()
    release = SimpleNamespace(
        slot=SimpleNamespace(site=target_site, remote_activated=True, prior_metadata=None)
    )
    environments: list[dict[str, str]] = []

    def runner(argv, **kwargs):
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), command_runner=runner,
        release_manager=release,
    )
    monkeypatch.setattr(ops, "_target_site", lambda: target_site)
    assert ops.doctor().certainty == Certainty.SUCCEEDED
    # Give the smoke parser its final JSON line without importing target code.
    def smoke_runner(argv, **kwargs):
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(list(argv), 0, '{"result":"PASS"}\n', "")
    ops._run = smoke_runner
    assert ops.cluster_smoke(_stage(ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS))).certainty == Certainty.SUCCEEDED
    assert [env["PYTHONPATH"] for env in environments] == [str(target_site)] * 2


def test_exact_readback_proves_staged_target_while_checkout_remains_prior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, prior = _checkout(tmp_path)
    target = _target(repo)
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    checkout = tmp_path / "detached-target"
    _run("git", "worktree", "add", "--detach", str(checkout), target, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    site = tmp_path / "target-site"
    site.mkdir()
    release = SimpleNamespace(
        slot=SimpleNamespace(site=site, remote_activated=True, prior_metadata=None),
        readback_hosts=lambda: True,
    )
    pins = _PinHarness()
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(
            hosts=(HostConfig("remote", "tcp://192.0.2.10:26600", gpus=2),),
            transport_security="trusted_fabric", source=str(config_path),
        ),
        release_manager=release, pin_transition=pins,  # type: ignore[arg-type]
    )
    ops.repo_snapshot(target)
    ops._checkout = checkout
    ops._smoke = {
        "result": "PASS",
        "world": 2,
        "data_parallel": 2,
        "nccl": True,
        "status_rank_coverage": 2,
        "source_rank_coverage": 2,
        "source_manifest_sha256": metadata.source_manifest,
        "source_cohort_boundaries": 2,
        "owned_procmeshes": 1,
        "teardown_confirmed": True,
    }
    monkeypatch.setattr(ops, "_site_matches", lambda *_args: True)

    readback = ops.exact_readback(_stage(metadata))

    assert readback.certainty == Certainty.SUCCEEDED
    assert readback.target_sha == target
    assert _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip() == prior
    pins.target_site_ok = False
    assert ops.exact_readback(_stage(metadata)).certainty == Certainty.FAILED


@pytest.mark.parametrize("merge_mode", ["success", "failed", "false_success"])
def test_driver_commit_classifies_only_exact_clean_outcomes(
    merge_mode: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, prior = _checkout(tmp_path)
    target = _target(repo)
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")

    def runner(argv, **kwargs):
        if "merge" not in argv or merge_mode == "success":
            return subprocess.run(list(argv), **kwargs)
        if merge_mode == "failed":
            return subprocess.CompletedProcess(list(argv), 1, "", "refused")
        completed = subprocess.run(list(argv), **kwargs)
        return subprocess.CompletedProcess(list(argv), 1, completed.stdout, completed.stderr)

    pins = _PinHarness()
    site = tmp_path / "target-site"
    site.mkdir()
    release = SimpleNamespace(
        slot=SimpleNamespace(site=site, remote_activated=True, prior_metadata=None),
        readback_hosts=lambda: True,
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), command_runner=runner,
        release_manager=release, pin_transition=pins,  # type: ignore[arg-type]
    )
    ops.repo_snapshot(target)
    monkeypatch.setattr(ops, "_site_matches", lambda *_args: True)

    result = ops.commit_driver(_stage(metadata))

    expected = {
        "success": Certainty.SUCCEEDED,
        "failed": Certainty.FAILED,
        "false_success": Certainty.UNKNOWN,
    }
    assert result.certainty == expected[merge_mode]
    head = _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip()
    assert head == (prior if merge_mode == "failed" else target)
    assert pins.live == ("prior" if merge_mode == "failed" else "target")


@pytest.mark.parametrize("certainty", [Certainty.FAILED, Certainty.UNKNOWN])
def test_pin_promotion_failure_never_touches_driver_source(
    certainty: Certainty, tmp_path: Path
):
    repo, prior = _checkout(tmp_path)
    target = _target(repo, pin="0.7.0")
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    pins = _PinHarness()
    pins.promote_result = OperationResult(certainty)
    site = tmp_path / "target-site"
    site.mkdir()
    release = SimpleNamespace(
        slot=SimpleNamespace(site=site, remote_activated=True, prior_metadata=None),
        readback_hosts=lambda: True,
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
        pin_transition=pins,  # type: ignore[arg-type]
    )
    ops.repo_snapshot(target)

    result = ops.commit_driver(_stage(metadata))

    assert result.certainty == certainty
    assert _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip() == prior


def test_failed_source_commit_with_unconfirmed_prior_pin_is_ambiguous(tmp_path: Path):
    repo, prior = _checkout(tmp_path)
    target = _target(repo, pin="0.7.0")
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    pins = _PinHarness()
    pins.restore_result = OperationResult(Certainty.UNKNOWN)
    site = tmp_path / "target-site"
    site.mkdir()
    release = SimpleNamespace(
        slot=SimpleNamespace(site=site, remote_activated=True, prior_metadata=None),
        readback_hosts=lambda: True,
    )

    def runner(argv, **kwargs):
        if "merge" in argv:
            return subprocess.CompletedProcess(list(argv), 1, "", "refused")
        return subprocess.run(list(argv), **kwargs)

    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), command_runner=runner,
        release_manager=release, pin_transition=pins,  # type: ignore[arg-type]
    )
    ops.repo_snapshot(target)

    assert ops.commit_driver(_stage(metadata)).certainty == Certainty.UNKNOWN
    assert pins.live == "target"
    assert _run("git", "rev-parse", "HEAD", cwd=repo).stdout.strip() == prior


def test_compensation_is_suppressed_after_live_driver_moves(
    tmp_path: Path,
):
    repo, prior = _checkout(tmp_path)
    target = _target(repo)
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    release = SimpleNamespace(
        compensate_workers=lambda: pytest.fail("unsafe remote rollback")
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release
    )
    ops.repo_snapshot(target)
    _run("git", "merge", "--ff-only", target, cwd=repo)

    assert ops.compensate(_stage(metadata)).certainty == Certainty.UNKNOWN


def test_compensation_is_suppressed_when_driver_pin_is_not_prior(tmp_path: Path):
    repo, prior = _checkout(tmp_path)
    target = _target(repo, pin="0.7.0")
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    pins = _PinHarness()
    pins.live = "target"
    release = SimpleNamespace(
        compensate_workers=lambda: pytest.fail("unsafe remote rollback")
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
        pin_transition=pins,  # type: ignore[arg-type]
    )
    ops.repo_snapshot(target)

    assert ops.compensate(_stage(metadata)).certainty == Certainty.UNKNOWN


def test_config_drift_blocks_promotion_but_not_fail_closed_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, prior = _checkout(tmp_path)
    target = _target(repo)
    metadata = _metadata(repo, target)
    _run("git", "checkout", "-q", prior, cwd=repo)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    site = tmp_path / "target-site"
    site.mkdir()
    release = SimpleNamespace(
        slot=SimpleNamespace(site=site, remote_activated=True, prior_metadata=None),
        ensure_supported=lambda: None,
        activate_workers=lambda: pytest.fail("activation after config drift"),
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release
    )
    ops.repo_snapshot(target)
    config_path.write_text("[cluster]\nauto_heal = false\n", encoding="utf-8")
    stops: list[bool] = []
    release.stop_prior_workers = lambda: stops.append(True) or OperationResult(Certainty.SUCCEEDED)

    assert ops.stop_workers().certainty == Certainty.SUCCEEDED
    assert stops == [True]
    assert ops.activate(_stage(metadata)).certainty == Certainty.UNKNOWN
    assert ops.start_workers_without_sync().certainty == Certainty.FAILED
    assert ops.commit_driver(_stage(metadata)).certainty == Certainty.UNKNOWN


def test_release_cleanup_exception_cannot_skip_worktree_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    release = SimpleNamespace(
        cleanup=lambda: (_ for _ in ()).throw(RuntimeError("cleanup failed"))
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release
    )
    removed: list[Path | None] = []
    monkeypatch.setattr(
        ops, "_remove_worktree",
        lambda path: removed.append(path) or OperationResult(Certainty.SUCCEEDED),
    )
    checkout = tmp_path / "update-owned"
    ops._checkout = checkout

    with pytest.raises(RuntimeError, match="cleanup failed"):
        ops.cleanup_stage(_stage(ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)))

    assert removed == [checkout]
