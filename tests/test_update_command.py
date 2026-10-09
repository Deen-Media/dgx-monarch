from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import __version__
from dgx_monarch.cli import lifecycle, update_command, update_entrypoint
from dgx_monarch.cli.update_lock import UpdateLock
from dgx_monarch.cli.update_release import ReleaseMetadata, ReleaseSlot
from dgx_monarch.cli.update_transaction import (
    ActivitySnapshot,
    Certainty,
    ExactReadback,
    OperationResult,
    RepoSnapshot,
    StagedRelease,
    UpdatePreconditionError,
)
from dgx_monarch.config import ClusterConfig
from update_command_helpers import (  # noqa: F401  # autouse fixture import.
    DEPS,
    RELEASE_ID,
    SOURCE,
    _checkout,
    _config,
    _isolated_update_locks,
    _run,
    _stage,
)


def test_checkout_inspection_binds_version_pin_dependencies_and_tracked_source(tmp_path: Path):
    repo, sha = _checkout(tmp_path)

    metadata, dependencies = update_command._inspect_checkout(repo, sha)

    assert metadata.target_sha == sha
    assert metadata.version == __version__
    assert metadata.torchmonarch_pin == "0.6.0"
    assert len(metadata.source_manifest) == 64
    assert len(metadata.dependency_manifest) == 64
    assert dependencies[0] == "torchmonarch==0.6.0"


def test_checkout_inspection_rejects_dependency_mirror_and_source_drift(tmp_path: Path):
    repo, sha = _checkout(tmp_path)
    (repo / "requirements.txt").write_text("torchmonarch==0.6.0\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="requirements"):
        update_command._inspect_checkout(repo, sha)

    _run("git", "checkout", "--", "requirements.txt", cwd=repo)
    (repo / "src" / "dgx_monarch" / "untracked.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked commit"):
        update_command._inspect_checkout(repo, sha)


def test_repo_snapshot_detects_dirty_and_hidden_index_state(tmp_path: Path):
    repo, sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    config = ClusterConfig(source=str(config_path))
    ops = update_command.SystemUpdateOps(repo, config, release_manager=SimpleNamespace())

    assert ops.repo_snapshot(sha) == RepoSnapshot(sha, True, True)
    (repo / "untracked.txt").write_text("dirty", encoding="utf-8")
    assert ops.repo_snapshot(sha).clean is False
    (repo / "untracked.txt").unlink()
    _run("git", "update-index", "--assume-unchanged", "requirements.txt", cwd=repo)
    assert ops.repo_snapshot(sha).clean is False


def test_repo_snapshot_rechecks_exact_cluster_config_bytes(tmp_path: Path):
    repo, sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=SimpleNamespace()
    )
    config_path.write_text("[cluster]\nauto_heal = false\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="config changed"):
        ops.repo_snapshot(sha)


def test_repo_snapshot_refuses_a_partial_target_before_probing_the_prior_workers(tmp_path: Path):
    """The full-commit check reads only the string, so a partial target fails before the remote prior-release probe."""
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    probed: list[str] = []
    release = SimpleNamespace(
        slot=SimpleNamespace(prior_metadata=object(), remote_activated=False),
        prior_hosts_match=lambda: probed.append("prior_hosts_match") or False,
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release
    )

    with pytest.raises(RuntimeError, match=r"^update target is not a full commit$"):
        ops.repo_snapshot("not-a-sha")
    assert probed == []


def test_repo_snapshot_rechecks_exact_prior_workers_before_stop(tmp_path: Path):
    repo, sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    release = SimpleNamespace(
        slot=SimpleNamespace(prior_metadata=object(), remote_activated=False),
        prior_hosts_match=lambda: False,
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release
    )

    with pytest.raises(UpdatePreconditionError, match="worker release changed") as raised:
        ops.repo_snapshot(sha)

    assert raised.value.code == "prior_worker_changed"


@pytest.mark.parametrize(
    ("status_coverage", "source_coverage"),
    [(1, 2), (True, 2), (2, True), ("2", 2), (2, None)],
)
def test_exact_readback_requires_complete_typed_rank_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status_coverage: object, source_coverage: object,
):
    repo, sha = _checkout(tmp_path)
    metadata = update_command._inspect_checkout(repo, sha)[0]
    staged = _stage(metadata, ranks=2)
    ops = object.__new__(update_command.SystemUpdateOps)
    ops._checkout = repo
    ops._prior_sha = sha
    ops._config_current = lambda: True
    ops.repo_snapshot = lambda _target: RepoSnapshot(sha, True, True)
    ops._releases = SimpleNamespace(readback_hosts=lambda: True)
    ops._target_site = lambda: repo / "src"
    ops._site_matches = lambda _site, _stage: True
    ops._pins = SimpleNamespace(target_matches_site=lambda _site: True)
    ops._smoke = {
        "result": "PASS",
        "world": 2,
        "data_parallel": 2,
        "nccl": True,
        "status_rank_coverage": status_coverage,
        "source_rank_coverage": source_coverage,
        "source_manifest_sha256": metadata.source_manifest,
        "source_cohort_boundaries": 2,
        "owned_procmeshes": 1,
        "teardown_confirmed": True,
    }
    monkeypatch.setattr(update_command, "_inspect_checkout", lambda *_args: (metadata, ()))

    readback = ops.exact_readback(staged)

    assert readback.certainty is Certainty.FAILED
    assert readback.ranks_matched == 0


def test_activity_is_ambiguous_for_unconfirmed_comfy_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    config = _config()
    ops = update_command.SystemUpdateOps(repo, config, release_manager=SimpleNamespace())
    monkeypatch.setattr(update_command.comfy_ports, "find_comfy_ports", lambda: {9000: 123})
    monkeypatch.setattr(
        update_command.comfy_ports, "driver_probe", lambda *_args: (None, True))
    monkeypatch.setattr(update_command, "_comfy_candidate_count", lambda: 1)
    monkeypatch.setattr(ops, "_marked_actor_count", lambda: 0)

    snapshot = ops.activity_snapshot()

    assert snapshot == ActivitySnapshot(True, None, None, 0)
    assert snapshot.blockers() == ("comfy_running",)


def test_activity_is_ambiguous_when_comfy_process_enumeration_is_denied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    ops = update_command.SystemUpdateOps(repo, _config(), release_manager=SimpleNamespace())
    monkeypatch.setattr(update_command.comfy_ports, "find_comfy_ports", lambda: {8188: None})
    monkeypatch.setattr(
        update_command.comfy_ports, "driver_probe", lambda *_args: (None, True))
    monkeypatch.setattr(update_command, "_comfy_candidate_count", lambda: None)
    monkeypatch.setattr(ops, "_marked_actor_count", lambda: 0)

    assert ops.activity_snapshot().blockers() == ("activity_unknown",)


def test_marked_actor_report_is_read_only_and_counts_every_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    ops = update_command.SystemUpdateOps(repo, _config(), release_manager=SimpleNamespace())
    payload = json.dumps({"candidates": [{"pid": 1}, {"pid": 2}]}) + "\n"
    scripts: list[str] = []

    def run_host(_config, _host, script, timeout=60):
        scripts.append(script)
        return subprocess.CompletedProcess([], 0, payload, "")

    monkeypatch.setattr(lifecycle, "run_on_host", run_host)

    assert ops._marked_actor_count() == 2
    assert "--report" in scripts[0]
    assert "--sweep" not in scripts[0]


def test_start_uses_target_subprocess_without_resyncing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    target_site = tmp_path / "target-site"
    target_site.mkdir()
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    release = SimpleNamespace(
        ensure_supported=lambda: None,
        slot=ReleaseSlot(
            RELEASE_ID, tmp_path / "slot", target_site, metadata,
            remote_activated=True,
        ),
    )
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []

    def runner(argv, **kwargs):
        calls.append(list(argv))
        environments.append(kwargs["env"])
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    monkeypatch.setattr(
        lifecycle,
        "up",
        lambda *_args, **_kwargs: pytest.fail("loaded lifecycle.up must not start target code"),
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), command_runner=runner,
        release_manager=release,
    )

    assert ops.start_target_workers(_stage(metadata)).certainty == Certainty.SUCCEEDED
    # The child runs the target site's own entry point; what it passes to
    # `up` is that function's contract, pinned by the start_target_main test below.
    assert "start_target_main" in calls[0][2]
    assert str(config_path) == calls[0][3]
    assert calls[0][4] == RELEASE_ID
    assert environments[0]["PYTHONPATH"] == str(target_site)


@pytest.mark.parametrize(
    ("returncode", "certainty"),
    # A signalled child never observed its own outcome: -2 is SIGINT, -15
    # SIGTERM, -9 SIGKILL (what the OOM killer sends); none is a definite failed start.
    [(1, Certainty.FAILED), (75, Certainty.UNKNOWN), (-2, Certainty.UNKNOWN),
     (-15, Certainty.UNKNOWN), (-9, Certainty.UNKNOWN)],
)
def test_target_start_subprocess_preserves_lifecycle_certainty(
    tmp_path: Path, returncode: int, certainty: Certainty
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    target_site = tmp_path / "target-site"
    target_site.mkdir()
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    release = SimpleNamespace(
        slot=ReleaseSlot(
            RELEASE_ID, tmp_path / "slot", target_site, metadata,
            remote_activated=True,
        ),
        ensure_supported=lambda: None,
    )
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(source=str(config_path)),
        command_runner=lambda argv, **_kwargs: subprocess.CompletedProcess(
            list(argv), returncode, "", ""
        ),
        release_manager=release,
    )

    assert ops.start_target_workers(_stage(metadata)).certainty == certainty


def test_target_start_without_exact_active_generation_makes_no_mutation(tmp_path: Path):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    calls: list[list[str]] = []
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    invalid_generation = "not-a-generation"
    release = SimpleNamespace(
        slot=SimpleNamespace(
            token=invalid_generation,
            site=tmp_path / "target-site",
            metadata=metadata,
            remote_activated=True,
        ),
        ensure_supported=lambda: None,
    )
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(source=str(config_path)),
        command_runner=lambda argv, **_kwargs: calls.append(list(argv))
        or subprocess.CompletedProcess(list(argv), 0, "", ""),
        release_manager=release,
    )

    assert ops.start_target_workers(_stage(metadata)).certainty == Certainty.UNKNOWN
    assert calls == []


def test_prior_restart_uses_the_exact_compensated_transaction_generation(tmp_path: Path):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    slot = ReleaseSlot(
        RELEASE_ID,
        tmp_path / "slot",
        tmp_path / "site",
        metadata,
        prior_metadata=metadata,
        remote_activated=False,
        activation_ambiguous=False,
    )
    calls: list[str] = []
    release = SimpleNamespace(
        slot=slot, ensure_supported=lambda: None,
        start_prior_workers=lambda generation: calls.append(generation) or OperationResult(Certainty.SUCCEEDED),
    )
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(source=str(config_path)),
        release_manager=release,
    )

    assert ops.start_prior_workers(_stage(metadata)).certainty == Certainty.SUCCEEDED
    assert calls == [RELEASE_ID]


@pytest.mark.parametrize(
    ("remote_activated", "activation_ambiguous", "has_prior"),
    [(True, False, True), (False, True, True), (False, False, False)],
)
def test_prior_restart_without_exact_compensated_authority_makes_no_mutation(
    tmp_path: Path,
    remote_activated: bool,
    activation_ambiguous: bool,
    has_prior: bool,
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    slot = ReleaseSlot(
        RELEASE_ID,
        tmp_path / "slot",
        tmp_path / "site",
        metadata,
        prior_metadata=metadata if has_prior else None,
        remote_activated=remote_activated,
        activation_ambiguous=activation_ambiguous,
    )
    calls: list[list[str]] = []
    release = SimpleNamespace(slot=slot, ensure_supported=lambda: None)
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(source=str(config_path)),
        command_runner=lambda argv, **_kwargs: calls.append(list(argv))
        or subprocess.CompletedProcess(list(argv), 0, "", ""),
        release_manager=release,
    )

    assert ops.start_prior_workers(_stage(metadata)).certainty == Certainty.UNKNOWN
    assert calls == []


@pytest.mark.parametrize(
    "certainty",
    [Certainty.SUCCEEDED, Certainty.FAILED, Certainty.UNKNOWN],
)
def test_worker_stop_preserves_release_manager_certainty(
    tmp_path: Path, certainty: Certainty,
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(source=str(config_path)),
        release_manager=SimpleNamespace(stop_prior_workers=lambda: OperationResult(certainty)),
    )

    assert ops.stop_workers().certainty == certainty


def test_worker_stop_adapter_has_no_release_switch_or_restart_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    release = SimpleNamespace(
        compensate_workers=lambda: pytest.fail("stop must not switch releases"),
        activate_workers=lambda: pytest.fail("stop must not activate a release"),
    )
    ops = update_command.SystemUpdateOps(
        repo,
        ClusterConfig(source=str(config_path)),
        release_manager=release,
    )
    events: list[str] = []
    release.stop_prior_workers = lambda: events.append("stop_prior_workers") or OperationResult(Certainty.SUCCEEDED)

    assert ops.stop_workers().certainty == Certainty.SUCCEEDED
    assert events == ["stop_prior_workers"]


def test_started_worker_stop_is_bound_to_exact_active_release_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    release = SimpleNamespace(
        slot=ReleaseSlot(
            RELEASE_ID, tmp_path / "slot", tmp_path / "site", metadata,
            remote_activated=True,
        )
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
    )
    generations: list[str] = []
    monkeypatch.setattr(
        lifecycle,
        "down_if_generation",
        lambda _config, generation: generations.append(generation) or True,
    )
    monkeypatch.setattr(
        lifecycle,
        "down",
        lambda _config: pytest.fail("generation settlement must not use generic down"),
    )

    assert ops.stop_started_workers(_stage(metadata)).certainty == Certainty.SUCCEEDED
    assert generations == [RELEASE_ID]


def test_started_worker_stop_refuses_staged_identity_drift_without_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    release = SimpleNamespace(
        slot=ReleaseSlot(
            RELEASE_ID, tmp_path / "slot", tmp_path / "site", metadata,
            remote_activated=True,
        )
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
    )
    monkeypatch.setattr(
        lifecycle,
        "down_if_generation",
        lambda *_args: pytest.fail("identity drift grants no stop authority"),
    )

    drifted = _stage(replace(metadata, source_manifest="c" * 64))
    assert ops.stop_started_workers(drifted).certainty == Certainty.UNKNOWN


class _HappyOps:
    def resolve_target(self, _target): return "2" * 40
    def repo_snapshot(self, _target): return RepoSnapshot("1" * 40, True, True)
    def activity_snapshot(self): return ActivitySnapshot(False, False, 0, 0)
    def stage(self, _target):
        return StagedRelease("2" * 40, "0.5.0", "0.7.0", SOURCE, DEPS, True, True, True, True, 1, 1, 1)
    def confirm(self, _stage): return True
    def stop_workers(self): return OperationResult(Certainty.SUCCEEDED)
    def stop_started_workers(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def activate(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def compensate(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def start_target_workers(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def start_prior_workers(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def start_workers_without_sync(self): return OperationResult(Certainty.SUCCEEDED)
    def doctor(self): return OperationResult(Certainty.SUCCEEDED)
    def cluster_smoke(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def exact_readback(self, stage):
        return ExactReadback(Certainty.SUCCEEDED, 1, 1, stage.target_sha, stage.version, stage.torchmonarch_pin, stage.source_manifest)
    def commit_driver(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def prior_driver_readback(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def finalize(self, _stage): return OperationResult(Certainty.SUCCEEDED)
    def cleanup_stage(self, _stage): return OperationResult(Certainty.SUCCEEDED)


def test_command_publishes_sanitized_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    destination = tmp_path / "receipts" / "update.json"

    rc = update_command.run_verified_update(
        ClusterConfig(), repo=tmp_path, assume_yes=True,
        receipt_path=destination, ops=_HappyOps(),
    )

    assert rc == 0
    receipt = json.loads(destination.read_text(encoding="ascii"))
    assert receipt["operation"] == "update"
    assert receipt["status"] == "succeeded"
    assert str(destination) in capsys.readouterr().out


def test_command_returns_failure_when_receipt_cannot_be_published(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    occupied = tmp_path / "receipt.json"
    occupied.write_text("keep", encoding="utf-8")

    rc = update_command.run_verified_update(
        ClusterConfig(), repo=tmp_path, assume_yes=True,
        receipt_path=occupied, ops=_HappyOps(),
    )

    assert rc == 1
    assert occupied.read_text(encoding="utf-8") == "keep"
    assert "receipt publication failed" in capsys.readouterr().err


class _RecordingOps(_HappyOps):
    """Every operation the transaction reaches, in order."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __getattribute__(self, name: str):
        value = super().__getattribute__(name)
        if name.startswith("_") or name == "calls" or not callable(value):
            return value

        def record(*args, **kwargs):
            self.calls.append(name)
            return value(*args, **kwargs)

        return record


@pytest.mark.parametrize(
    ("receipt_path", "refusal"),
    [("receipt.json", "operator receipt path must be absolute"),
     ("/", "operator receipt path needs a filename")],
    ids=["relative", "nameless"],
)
def test_command_refuses_a_bad_receipt_path_before_any_operation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    receipt_path: str, refusal: str,
):
    """A bad --receipt path is refused before the update lock and any operation.

    The check needs only the argument. If it runs only where the receipt is
    written, `dgxm update --verify --yes --receipt receipt.json` fetches, stages
    on every host, stops and restarts the workers and commits the driver, then
    exits 1 with no receipt (fixed 2026-10-05).
    """
    monkeypatch.chdir(tmp_path)
    ops = _RecordingOps()
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(
        update_entrypoint, "UpdateLock",
        lambda repo: UpdateLock(repo, state_root=lock_root),
    )

    rc = update_command.run_verified_update(
        ClusterConfig(), repo=tmp_path, assume_yes=True, receipt_path=receipt_path, ops=ops,
    )

    assert rc == 2
    assert ops.calls == []
    assert not (tmp_path / "receipt.json").exists()
    assert not lock_root.exists()
    captured = capsys.readouterr()
    assert captured.err == f"error: --receipt: {refusal}\n"
    assert captured.out == ""


def test_command_refuses_concurrent_verified_update_before_any_operation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    ops = _HappyOps()
    ops.resolve_target = lambda _target: pytest.fail("operation ran without lock")  # type: ignore[method-assign]
    lock_root = tmp_path / "locks"
    monkeypatch.setattr(
        update_entrypoint, "UpdateLock",
        lambda repo: UpdateLock(repo, state_root=lock_root),
    )

    with UpdateLock(tmp_path, state_root=lock_root):
        rc = update_command.run_verified_update(
            ClusterConfig(), repo=tmp_path, assume_yes=True, ops=ops,
        )

    assert rc == 2
    assert "already active" in capsys.readouterr().err


def test_command_publishes_partial_receipt_and_preserves_interrupt_identity(
    tmp_path: Path,
):
    destination = tmp_path / "receipts" / "interrupted-update.json"
    interrupt = KeyboardInterrupt()
    ops = _HappyOps()
    def interrupted_doctor():
        raise interrupt
    ops.doctor = interrupted_doctor  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt) as raised:
        update_command.run_verified_update(
            ClusterConfig(), repo=tmp_path, assume_yes=True,
            receipt_path=destination, ops=ops,
        )

    assert raised.value is interrupt
    receipt = json.loads(destination.read_text(encoding="ascii"))
    assert receipt["status"] == "partial"
    assert receipt["notes"][0].startswith("verified update was interrupted")


def test_stage_cleanup_runs_both_finalizers_and_keeps_first_cancellation(
    tmp_path: Path,
):
    ops = object.__new__(update_command.SystemUpdateOps)
    first, second = KeyboardInterrupt(), SystemExit(130)
    calls: list[str] = []
    def release_cleanup():
        calls.append("release")
        raise first
    def worktree_cleanup(_checkout):
        calls.append("worktree")
        raise second
    ops._releases = SimpleNamespace(cleanup=release_cleanup)
    ops._checkout = tmp_path / "checkout"
    ops._remove_worktree = worktree_cleanup  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt) as raised:
        ops.cleanup_stage(_HappyOps().stage(""))

    assert raised.value is first
    assert calls == ["release", "worktree"]


def test_start_target_main_never_exits_1_on_a_fault_it_did_not_observe():
    """Exit 1 must mean `up` returned False.

    Python exits 1 when a traceback escapes the host loop uncaught, and exit 1
    is the update transaction's authority to restore the prior release over
    hosts that may already run the target.
    """
    from dgx_monarch.cli import update_worker_lifecycle as lifecycle_ops

    seen: list[dict] = []

    def up(config, **kwargs):
        seen.append({"source": config, **kwargs})
        return outcome() if callable(outcome) else outcome

    outcome: object = True
    lifecycle_ops.lifecycle.up  # noqa: B018 - the name the child rebinds
    import dgx_monarch.cli.lifecycle as real_lifecycle

    original_up, original_load = real_lifecycle.up, None
    import dgx_monarch.config as config_mod

    original_load = config_mod.load_cluster_config
    real_lifecycle.up = up
    config_mod.load_cluster_config = lambda path: f"config:{path}"
    try:
        assert lifecycle_ops.start_target_main("/c.toml", "gen-1") == 0
        # The child must never resync: the release it starts is already staged.
        assert seen[0] == {
            "source": "config:/c.toml", "sync": False, "generation": "gen-1"}

        outcome = None
        assert lifecycle_ops.start_target_main("/c.toml", None) == 75
        outcome = False
        assert lifecycle_ops.start_target_main("/c.toml", None) == 1

        def explode():
            raise RuntimeError("half the hosts are already up")

        outcome = explode
        assert lifecycle_ops.start_target_main("/c.toml", None) == 75

        def cancelled():
            raise KeyboardInterrupt

        outcome = cancelled
        assert lifecycle_ops.start_target_main("/c.toml", None) == 75
    finally:
        real_lifecycle.up = original_up
        config_mod.load_cluster_config = original_load


def test_a_smoke_that_lost_its_transport_is_unknown_not_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A smoke child that lost its transport reads UNKNOWN; only a definite refusal reads FAILED.

    The child's 60 s status timeout and the parent's 600 s timeout are one class of
    evidence; if they read differently, rollback depends on which clock fires first.
    """
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    target_site = tmp_path / "target-site"
    target_site.mkdir()
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    release = SimpleNamespace(
        ensure_supported=lambda: None,
        slot=SimpleNamespace(site=target_site, remote_activated=True),
    )

    def ops_exiting(returncode: int, stdout: str):
        ops = update_command.SystemUpdateOps(
            repo,
            ClusterConfig(source=str(config_path)),
            command_runner=lambda argv, **_kwargs: subprocess.CompletedProcess(
                list(argv), returncode, stdout, ""),
            release_manager=release,
        )
        monkeypatch.setattr(ops, "_target_site", lambda: target_site)
        return ops

    blind = '{"error": "status_unknown", "result": "FAIL"}\n'
    assert ops_exiting(75, blind).cluster_smoke(
        _stage(metadata)).certainty == Certainty.UNKNOWN
    refused = '{"error": "status_topology_invalid", "result": "FAIL"}\n'
    assert ops_exiting(1, refused).cluster_smoke(
        _stage(metadata)).certainty == Certainty.FAILED


def test_doctor_carries_the_unknown_exit_through_to_certainty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Doctor exit 75 (a critical probe never answered) reads UNKNOWN, so it cannot authorize a restore."""
    repo, _sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    target_site = tmp_path / "target-site"
    target_site.mkdir()
    release = SimpleNamespace(
        ensure_supported=lambda: None,
        slot=SimpleNamespace(site=target_site, remote_activated=True),
    )
    for returncode, certainty in (
        (0, Certainty.SUCCEEDED),
        (75, Certainty.UNKNOWN),
        (1, Certainty.FAILED),
    ):
        ops = update_command.SystemUpdateOps(
            repo,
            ClusterConfig(source=str(config_path)),
            command_runner=lambda argv, _rc=returncode, **_kwargs:
                subprocess.CompletedProcess(list(argv), _rc, "", ""),
            release_manager=release,
        )
        monkeypatch.setattr(ops, "_target_site", lambda: target_site)
        assert ops.doctor().certainty == certainty


def test_verified_update_default_uses_isolated_bootstrap(monkeypatch, tmp_path):
    from dgx_monarch.cli import update_bootstrap
    calls = []
    config = ClusterConfig(source=str(tmp_path / "cluster.toml"))
    receipt = tmp_path / "receipt.json"

    def launch(supplied, **kwargs):
        calls.append((supplied, kwargs))
        return 17

    monkeypatch.setattr(update_bootstrap, "launch_verified_update", launch)
    monkeypatch.setattr(update_command, "SystemUpdateOps", lambda *_a, **_k: pytest.fail("parent must not create update operations"))
    assert update_command.run_verified_update(
        config, repo=tmp_path, target_ref="origin/master", assume_yes=True,
        driver_host="controller", receipt_path=receipt,
    ) == 17
    assert calls == [(config, {
        "repo": tmp_path, "target_ref": "origin/master", "assume_yes": True,
        "driver_host": "controller", "receipt_path": receipt,
    })]
