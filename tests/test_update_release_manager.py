"""ReleaseManager host activation, compensation, finalize and lock regressions."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from dgx_monarch import __version__
from dgx_monarch.cli import lifecycle, update_command
from dgx_monarch.cli.lifecycle_systemd import systemd_worker_unit
from dgx_monarch.cli.update_driver_pin import site_payload_digest
from dgx_monarch.cli.update_release import (
    ReleaseLayoutError,
    ReleaseManager,
    ReleaseMetadata,
    ReleaseSlot,
)
from dgx_monarch.cli.update_transaction import (
    ActivitySnapshot,
    Certainty,
    OperationResult,
    UpdateRequest,
    execute_verified_update,
)
from dgx_monarch.cli.update_worker_attestation import site_tree_digest, verifier_tree_digest
from dgx_monarch.config import ClusterConfig, HostConfig
from dgx_monarch.runtime_provenance import dgx_source_manifest_sha256
from update_command_helpers import (  # noqa: F401  # autouse fixture import.
    DEPS,
    RELEASE_ID,
    SOURCE,
    _checkout,
    _config,
    _isolated_update_locks,
    _metadata,
    _PinHarness,
)


def _operation(script: str, operation: str) -> bool:
    markers = {"activate": "echo ACTIVATED", "compensate": "echo COMPENSATED", "cleanup": "echo CLEANED", "finalize": "echo FINALIZED"}
    return f"# DGXM_INSTALLATION_OPERATION={operation}\n" in script or markers.get(operation, "never-match-operation") in script


class _HostHarness:
    def __init__(self) -> None:
        self.scripts: list[str] = []
        self.activation = "success"
        self.rsync_ok = True

    def host(self, _config, _host, script: str, _timeout: int):
        self.scripts.append(script)
        if _operation(script, "snapshot"):
            unit = systemd_worker_unit(_config, _host, managed_source=True)
            row = {"unit": base64.b64encode(unit.encode()).decode(), "sha256": hashlib.sha256(unit.encode()).hexdigest(), "site": "/home/user/.local/share/dgx-monarch/src", "home": "/home/user", "user": "user", "site_packages": "", "live": {"kind": "directory", "target": ""}}
            return subprocess.CompletedProcess([], 0, json.dumps(row), "")
        if "echo RELEASE_MATCH" in script and not _operation(script, "activate"):
            return subprocess.CompletedProcess([], 0, "RELEASE_MATCH\n", "")
        if _operation(script, "activate"):
            if self.activation == "timeout":
                raise subprocess.TimeoutExpired([], 60)
            if self.activation == "failed":
                return subprocess.CompletedProcess(
                    [], 0, "ACTIVATION_PROBE_PRIOR\n", ""
                )
            return subprocess.CompletedProcess([], 0, "ACTIVATED\n", "")
        if "echo NEW" in script:
            return subprocess.CompletedProcess([], 0, "PRIOR\n", "")
        if _operation(script, "compensate"):
            return subprocess.CompletedProcess([], 0, "COMPENSATED\n", "")
        if _operation(script, "cleanup"):
            return subprocess.CompletedProcess([], 0, "CLEANED\n", "")
        if _operation(script, "finalize"):
            return subprocess.CompletedProcess([], 0, "FINALIZED\n", "")
        return subprocess.CompletedProcess([], 0, "", "")

    def command(self, argv, **_kwargs):
        status = 0 if self.rsync_ok or argv[0] != "rsync" else 1
        return subprocess.CompletedProcess(list(argv), status, "", "")


def _manager(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, harness: _HostHarness
) -> ReleaseManager:
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: False)
    monkeypatch.setattr(
        "dgx_monarch.cli.update_release.site_payload_digest", lambda _site, _pin: "c" * 64
    )
    manager = ReleaseManager(
        _config(), command_runner=harness.command, host_runner=harness.host,
        state_root=tmp_path / "state", token_factory=lambda: RELEASE_ID,
    )

    def fake_pip(site: Path, checkout: Path, _pin: str) -> None:
        shutil.copytree(checkout / "src" / "dgx_monarch", site / "dgx_monarch")

    monkeypatch.setattr(manager, "_pip_stage", fake_pip)
    return manager


def _attest_prior(manager: ReleaseManager, repo: Path, sha: str) -> None:
    assert manager.attest_prior(_metadata(repo, sha), "d" * 64)


def _hold_lifecycle_lock(home: Path) -> int:
    parent = home
    for name in (".local", "state", "dgx-monarch"):
        parent = parent / name
        parent.mkdir(mode=0o700, exist_ok=True)
        parent.chmod(0o700)
    lock = parent / "lifecycle.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    os.fchmod(fd, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def test_activation_and_compensation_wait_for_shared_lifecycle_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    base = home / ".local" / "share" / "dgx-monarch"
    prior_site = base / "releases" / "u-111111111111-2222222222222222" / "site"
    target_site = base / "releases" / RELEASE_ID / "site"
    prior_site.mkdir(parents=True)
    (target_site / "dgx_monarch").mkdir(parents=True)
    live = base / "src"
    live.symlink_to(prior_site, target_is_directory=True)
    config = replace(_config(), python_bin=sys.executable)
    entered = threading.Event()

    def host_runner(_config, _host, script, timeout):
        entered.set()
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
            env={"HOME": str(home), "PATH": os.defpath},
        )

    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    manager = ReleaseManager(config, host_runner=host_runner)
    manager.slot = ReleaseSlot(
        RELEASE_ID, tmp_path / "slot", tmp_path / "site", metadata,
        prior_metadata=metadata, prior_pin_payload_digest="d" * 64,
    )
    results: list[OperationResult] = []
    errors: list[BaseException] = []

    def activate() -> None:
        try:
            results.append(manager.activate_workers())
        except BaseException as exc:
            errors.append(exc)

    fd = _hold_lifecycle_lock(home)
    thread = threading.Thread(target=activate)
    thread.start()
    assert entered.wait(2)
    try:
        thread.join(0.1)
        assert thread.is_alive()
        assert live.resolve() == prior_site
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    thread.join(5)
    assert not thread.is_alive() and not errors
    assert results.pop().certainty == Certainty.SUCCEEDED
    assert live.resolve() == target_site

    monkeypatch.setattr(manager, "_prior_hosts_match", lambda _slot: True)
    entered.clear()
    fd = _hold_lifecycle_lock(home)
    thread = threading.Thread(target=lambda: results.append(manager.compensate_workers()))
    thread.start()
    assert entered.wait(2)
    try:
        thread.join(0.1)
        assert thread.is_alive()
        assert live.resolve() == target_site
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    thread.join(5)
    assert not thread.is_alive()
    assert results.pop().certainty == Certainty.SUCCEEDED
    assert live.resolve() == prior_site


def test_source_finalization_waits_for_shared_lifecycle_lock(tmp_path: Path):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    base = home / ".local" / "share" / "dgx-monarch"
    prior_root = base / "releases" / "u-111111111111-2222222222222222"
    target_site = base / "releases" / RELEASE_ID / "site"
    (prior_root / "site").mkdir(parents=True)
    target_site.mkdir(parents=True)
    (base / "src").symlink_to(target_site, target_is_directory=True)
    txn = base / "transactions" / RELEASE_ID
    txn.mkdir(parents=True)
    (txn / "kind").write_text("symlink", encoding="ascii")
    (txn / "prior").write_text(str(prior_root / "site"), encoding="ascii")
    entered = threading.Event()

    def host_runner(_config, _host, script, timeout):
        entered.set()
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, timeout=timeout,
            env={"HOME": str(home), "PATH": os.defpath},
        )

    manager = ReleaseManager(replace(_config(), python_bin=sys.executable), host_runner=host_runner)
    manager.slot = ReleaseSlot(
        RELEASE_ID, tmp_path / "slot", tmp_path / "site",
        ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS),
        remote_activated=True,
    )
    results: list[OperationResult] = []
    fd = _hold_lifecycle_lock(home)
    thread = threading.Thread(target=lambda: results.append(manager.finalize()))
    thread.start()
    assert entered.wait(2)
    try:
        thread.join(0.1)
        assert thread.is_alive()
        assert prior_root.is_dir()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    thread.join(5)
    assert not thread.is_alive()
    assert results.pop().certainty == Certainty.SUCCEEDED
    assert not prior_root.exists()


def test_release_manager_supports_one_colocated_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: True)
    manager = ReleaseManager(_config())
    manager.ensure_supported()


def test_release_manager_refuses_unknown_worker_colocation(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: None)
    manager = ReleaseManager(_config())
    with pytest.raises(ReleaseLayoutError, match=r"definite.*colocation"):
        manager.ensure_supported()


@pytest.mark.parametrize(
    ("target_sha", "token", "refusal"),
    [
        ("not-a-commit", RELEASE_ID, "release target must be a full commit"),
        ("0" * 40, "u-short", "release token is invalid"),
    ],
)
def test_prepare_refuses_bad_metadata_and_tokens_before_resolving_any_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target_sha: str, token: str, refusal: str,
):
    """Neither refusal needs a host, so `prepare` raises both before the layout check.

    The layout check resolves every worker host. The detector answers "remote"
    here, so a layout check that ran first would pass, and only the recorded
    host lookups would show it.
    """
    resolved: list[str] = []

    def locality(host: HostConfig) -> bool:
        resolved.append(host.name)
        return False

    monkeypatch.setattr(lifecycle, "_is_local", locality)
    harness = _HostHarness()
    commands: list[list[str]] = []

    def command(argv, **_kwargs):
        commands.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, "", "")

    manager = ReleaseManager(
        _config(), command_runner=command, host_runner=harness.host,
        state_root=tmp_path / "state", token_factory=lambda: token,
    )
    metadata = ReleaseMetadata(target_sha, __version__, "0.6.0", SOURCE, DEPS)

    with pytest.raises(ValueError, match=rf"^{refusal}$"):
        manager.prepare(tmp_path, metadata)
    assert resolved == [] and harness.scripts == [] and commands == []
    assert manager.slot is None and not (tmp_path / "state").exists()


def test_unknown_layout_is_an_actionable_end_to_end_precondition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    config = ClusterConfig(
        hosts=(HostConfig("local", "tcp://127.0.0.1:26600"),),
        transport_security="trusted_fabric", source=str(config_path),
    )
    ops = update_command.SystemUpdateOps(repo, config, worktree_root=tmp_path / "worktrees")
    monkeypatch.setattr(lifecycle, "_is_local", lambda _host: None)
    monkeypatch.setattr(ops, "resolve_target", lambda _ref: sha)
    monkeypatch.setattr(
        ops, "activity_snapshot", lambda: ActivitySnapshot(False, False, 0, 0)
    )

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "release_layout_unsupported"
    assert result.exit_code == 2
    assert not (tmp_path / "worktrees").exists()


def test_live_driver_pin_drift_is_an_actionable_end_to_end_precondition(
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
        cleanup=lambda: cleaned.append(True) or OperationResult(Certainty.SUCCEEDED),
    )
    pins = _PinHarness()
    pins.live = "target"
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
        pin_transition=pins, worktree_root=tmp_path / "worktrees",  # type: ignore[arg-type]
    )
    monkeypatch.setattr(ops, "resolve_target", lambda _ref: sha)
    monkeypatch.setattr(
        ops, "activity_snapshot", lambda: ActivitySnapshot(False, False, 0, 0)
    )
    monkeypatch.setattr(update_command, "_dependencies_available", lambda *_args: True)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "driver_pin_drift"
    assert result.exit_code == 2
    assert cleaned == [True]
    assert not list((tmp_path / "worktrees").glob("update-*"))


def test_prior_worker_drift_is_an_actionable_pre_mutation_refusal(
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
        attest_prior=lambda *_args: False,
        cleanup=lambda: cleaned.append(True) or OperationResult(Certainty.SUCCEEDED),
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
        pin_transition=_PinHarness(), worktree_root=tmp_path / "worktrees",  # type: ignore[arg-type]
    )
    monkeypatch.setattr(ops, "resolve_target", lambda _ref: sha)
    monkeypatch.setattr(ops, "activity_snapshot", lambda: ActivitySnapshot(False, False, 0, 0))
    monkeypatch.setattr(update_command, "_dependencies_available", lambda *_args: True)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    assert result.code == "prior_worker_drift"
    assert result.exit_code == 2
    assert cleaned == [True]
    assert not list((tmp_path / "worktrees").glob("update-*"))


def test_failed_stage_reports_unconfirmed_cleanup_in_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    repo, sha = _checkout(tmp_path)
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("[cluster]\n", encoding="utf-8")
    cleanup_calls: list[bool] = []
    release = SimpleNamespace(
        slot=None,
        ensure_supported=lambda: None,
        prepare=lambda *_args: (_ for _ in ()).throw(RuntimeError("rsync failed")),
        cleanup=lambda: cleanup_calls.append(True) or OperationResult(Certainty.UNKNOWN),
    )
    ops = update_command.SystemUpdateOps(
        repo, ClusterConfig(source=str(config_path)), release_manager=release,
        pin_transition=_PinHarness(), worktree_root=tmp_path / "worktrees",  # type: ignore[arg-type]
    )
    monkeypatch.setattr(ops, "resolve_target", lambda _ref: sha)
    monkeypatch.setattr(ops, "activity_snapshot", lambda: ActivitySnapshot(False, False, 0, 0))
    monkeypatch.setattr(update_command, "_dependencies_available", lambda *_args: True)

    result = execute_verified_update(UpdateRequest(assume_yes=True), ops)

    cleanup_steps = [step for step in result.receipt["steps"] if step["name"] == "stage_cleanup"]
    assert result.code == "stage_failed"
    assert result.receipt["status"] == "partial"
    assert cleanup_steps == [{"name": "stage_cleanup", "status": "unknown", "counts": {}, "notes": []}]
    assert cleanup_calls == [True]
    assert not list((tmp_path / "worktrees").glob("update-*"))


def test_release_stage_failure_cleans_owned_remote_and_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    harness.rsync_ok = False
    manager = _manager(tmp_path, monkeypatch, harness)

    with pytest.raises(ReleaseLayoutError, match="validation"):
        manager.prepare(repo, _metadata(repo, sha))

    assert not (tmp_path / "state" / RELEASE_ID).exists()
    assert any(_operation(script, "cleanup") for script in harness.scripts)
    assert not any(_operation(script, "activate") for script in harness.scripts)


def test_remote_readback_binds_exact_torchmonarch_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    site = home / ".local" / "share" / "dgx-monarch" / "releases" / RELEASE_ID / "site"
    package_root = Path(update_command.__file__).parents[1]
    shutil.copytree(package_root, site / "dgx_monarch")
    verifier = site.parent / "verifier" / "dgx_monarch"
    shutil.copytree(package_root, verifier)
    candidate_verifier = site / "dgx_monarch" / "cli" / "update_driver_pin.py"
    candidate_verifier.write_text(
        candidate_verifier.read_text(encoding="utf-8")
        + "\ndef site_payload_digest(*_args):\n    raise RuntimeError('candidate verifier ran')\n",
        encoding="utf-8",
    )
    torch_package = site / "torchmonarch"
    torch_package.mkdir()
    (torch_package / "__init__.py").write_text('VERSION = "0.6.0"\n', encoding="utf-8")
    (torch_package / "runtime.py").write_text('PIN = "0.6.0"\n', encoding="utf-8")
    dist_info = site / "torchmonarch-0.6.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torchmonarch\nVersion: 0.6.0\n",
        encoding="utf-8",
    )
    (dist_info / "RECORD").write_text(
        "torchmonarch/__init__.py,,\ntorchmonarch/runtime.py,,\n"
        "torchmonarch-0.6.0.dist-info/METADATA,,\n"
        "torchmonarch-0.6.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )
    metadata = ReleaseMetadata(
        "2" * 40, __version__, "0.6.0",
        dgx_source_manifest_sha256(site / "dgx_monarch"), DEPS,
    )
    local_site = tmp_path / "slot" / "site"
    shutil.copytree(site, local_site)
    shadow = home / "dgx_monarch"
    (shadow / "cli").mkdir(parents=True)
    for path in (shadow / "__init__.py", shadow / "cli" / "__init__.py"):
        path.write_text("", encoding="utf-8")
    (shadow / "cli" / "update_driver_pin.py").write_text(
        "raise RuntimeError('cwd shadow imported')\n", encoding="utf-8"
    )
    (shadow / "runtime_provenance.py").write_text(
        "raise RuntimeError('cwd shadow imported')\n", encoding="utf-8"
    )

    def host_runner(_config, _host, script, _timeout):
        assert "\0" not in script
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            cwd=home, env={"HOME": str(home), "PATH": os.environ["PATH"]},
        )

    config = ClusterConfig(
        hosts=_config().hosts, python_bin=sys.executable,
        transport_security="trusted_fabric",
    )
    manager = ReleaseManager(config, host_runner=host_runner)
    manager.slot = ReleaseSlot(
        RELEASE_ID, tmp_path / "slot", local_site, metadata,
        pin_payload_digest=site_payload_digest(site, "0.6.0"),
        site_manifest=site_tree_digest(local_site),
        verifier_digest=verifier_tree_digest(verifier),
    )
    live = home / ".local" / "share" / "dgx-monarch" / "src"
    live.symlink_to(site, target_is_directory=True)
    assert manager.readback_hosts()
    (site / "sitecustomize.py").write_text("raise RuntimeError('drift')\n", encoding="utf-8")
    assert not manager.readback_hosts()
    (site / "sitecustomize.py").unlink()
    assert manager.readback_hosts()
    (site / "sitecustomize.pyc").write_bytes(b"sourceless drift")
    assert not manager.readback_hosts()
    (site / "sitecustomize.pyc").unlink()
    assert manager.readback_hosts()
    other = home / ".local" / "share" / "dgx-monarch" / "releases" / "u-333333333333-4444444444444444" / "site"
    shutil.copytree(site, other)
    live.unlink()
    live.symlink_to(other, target_is_directory=True)
    assert not manager.readback_hosts()
    live.unlink()
    live.symlink_to(site, target_is_directory=True)
    live.unlink()
    shutil.copytree(site, live)
    digest = site_payload_digest(live, "0.6.0")
    monkeypatch.setattr("dgx_monarch.cli.update_release.capture_installation", lambda *_args: {"site": str(live)})
    assert manager.attest_prior(metadata, digest)
    (live / "torchmonarch" / "runtime.py").write_text("PIN = 'skewed'\n", encoding="utf-8")
    assert not manager.attest_prior(metadata, digest)
    shutil.rmtree(live)
    live.symlink_to(site, target_is_directory=True)
    (torch_package / "runtime.py").write_text("PIN = 'tampered'\n", encoding="utf-8")
    assert not manager.readback_hosts()


def test_release_activation_unknown_suppresses_compensation_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)
    harness.activation = "timeout"

    assert manager.activate_workers().certainty == Certainty.UNKNOWN
    assert manager.compensate_workers().certainty == Certainty.UNKNOWN
    assert manager.cleanup().certainty == Certainty.UNKNOWN
    assert (tmp_path / "state" / RELEASE_ID).exists()
    assert not any(_operation(script, "compensate") for script in harness.scripts)


def test_keyboard_interrupt_during_activation_latches_ambiguity_and_retains_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    slot = manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)
    interrupt = KeyboardInterrupt()

    def interrupted_host(_config, _host, script, _timeout):
        if _operation(script, "activate"):
            raise interrupt
        return harness.host(_config, _host, script, _timeout)

    manager._host_run = interrupted_host
    with pytest.raises(KeyboardInterrupt) as raised:
        manager.activate_workers()

    assert raised.value is interrupt
    assert slot.activation_ambiguous
    assert manager.cleanup().certainty == Certainty.UNKNOWN
    assert slot.root.exists()
    assert not any(_operation(script, "cleanup") for script in harness.scripts)


def test_activation_failure_probe_is_in_the_same_lock_held_host_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    slot = manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)
    calls: list[str] = []

    def definite_prior(_config, _host, script, _timeout):
        calls.append(script)
        return subprocess.CompletedProcess([], 0, "ACTIVATION_PROBE_PRIOR\n", "")

    manager._host_run = definite_prior
    assert manager.activate_workers().certainty == Certainty.FAILED
    assert len(calls) == 1
    assert _operation(calls[0], "activate")
    assert "ACTIVATION_PROBE_PRIOR" in calls[0]
    assert "flock" in calls[0]
    assert not slot.activation_ambiguous


def test_definite_partial_promotion_can_be_compensated_and_cleaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)
    harness.activation = "failed"

    assert manager.activate_workers().certainty == Certainty.FAILED
    assert manager.compensate_workers().certainty == Certainty.SUCCEEDED
    assert manager.cleanup().certainty == Certainty.SUCCEEDED
    assert any(_operation(script, "compensate") for script in harness.scripts)
    assert not (tmp_path / "state" / RELEASE_ID).exists()


def test_interrupted_directory_activation_is_ambiguous_and_preserves_backup(
    tmp_path: Path,
):
    home = tmp_path / "home"
    base = home / ".local" / "share" / "dgx-monarch"
    (base / "transactions" / RELEASE_ID).mkdir(parents=True)
    backup = base / "backups" / RELEASE_ID / "src"
    backup.mkdir(parents=True)
    site = tmp_path / "site"
    site.mkdir()
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)

    def host_runner(_config, _host, script, _timeout):
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        )

    manager = ReleaseManager(_config(), host_runner=host_runner)
    slot = ReleaseSlot(RELEASE_ID, tmp_path / "slot", site, metadata)

    assert manager._probe_activation(_config().hosts[0], slot) == Certainty.UNKNOWN
    slot.activation_ambiguous = True
    manager.slot = slot
    assert manager.cleanup().certainty == Certainty.UNKNOWN
    assert backup.is_dir()


def test_directory_switch_durably_journals_before_moving_live_tree(tmp_path: Path):
    manager = ReleaseManager(_config())
    slot = ReleaseSlot(
        RELEASE_ID, tmp_path / "slot", tmp_path / "site",
        ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS),
    )
    script = manager._activation_script(slot)
    journal = script.index('printf %s directory > "$TXN/kind"')
    durable = script.index('sync "$TXN/kind" "$TXN/prior"')
    moved = script.index('mv -- "$LIVE" "$BACKUP"')
    assert journal < durable < moved


def test_unconfirmed_remote_cleanup_retains_local_recovery_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    slot = manager.prepare(repo, _metadata(repo, sha))

    def refused_cleanup(config, host, script, timeout):
        if _operation(script, "cleanup"):
            return subprocess.CompletedProcess([], 60, "", "live release")
        return harness.host(config, host, script, timeout)

    manager._host_run = refused_cleanup
    assert manager.cleanup().certainty == Certainty.UNKNOWN
    assert slot.root.exists()


def test_cleanup_attempts_every_host_and_preserves_cancellation(tmp_path: Path):
    hosts = (
        HostConfig("one", "tcp://192.0.2.10:26600"),
        HostConfig("two", "tcp://192.0.2.11:26600"),
    )
    config = ClusterConfig(hosts=hosts, transport_security="trusted_fabric")
    calls: list[str] = []
    interrupt = KeyboardInterrupt()

    def host_runner(_config, host, _script, _timeout):
        calls.append(host.name)
        if host.name == "one":
            raise interrupt
        return subprocess.CompletedProcess([], 0, "CLEANED\n", "")

    state = tmp_path / "state"
    root = state / RELEASE_ID
    site = root / "site"
    site.mkdir(parents=True)
    manager = ReleaseManager(config, host_runner=host_runner, state_root=state)
    manager.slot = ReleaseSlot(
        RELEASE_ID, root, site,
        ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS),
        remote_staged=2,
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        manager.cleanup()
    assert raised.value is interrupt
    assert calls == ["one", "two"]
    assert root.exists()


def test_failed_compensation_retains_all_recovery_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)
    harness.activation = "failed"
    assert manager.activate_workers().certainty == Certainty.FAILED

    original = harness.host
    def failed_compensation(config, host, script, timeout):
        if _operation(script, "compensate"):
            return subprocess.CompletedProcess([], 1, "", "")
        return original(config, host, script, timeout)
    manager._host_run = failed_compensation

    assert manager.compensate_workers().certainty == Certainty.UNKNOWN
    assert manager.cleanup().certainty == Certainty.UNKNOWN
    assert manager.slot is not None and manager.slot.root.exists()


def test_keyboard_interrupt_during_compensation_retains_recovery_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    slot = manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)
    harness.activation = "failed"
    assert manager.activate_workers().certainty == Certainty.FAILED
    interrupt = KeyboardInterrupt()

    def interrupted_compensation(config, host, script, timeout):
        if _operation(script, "compensate"):
            raise interrupt
        return harness.host(config, host, script, timeout)

    manager._host_run = interrupted_compensation
    with pytest.raises(KeyboardInterrupt) as raised:
        manager.compensate_workers()
    assert raised.value is interrupt
    assert slot.activation_ambiguous
    assert manager.cleanup().certainty == Certainty.UNKNOWN
    assert slot.root.exists()


def test_compensation_requires_exact_prior_worker_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    slot = manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)
    harness.activation = "failed"
    assert manager.activate_workers().certainty == Certainty.FAILED

    def skewed_prior(config, host, script, timeout):
        if "echo RELEASE_MATCH" in script and not _operation(script, "activate"):
            return subprocess.CompletedProcess([], 1, "", "skewed prior")
        return harness.host(config, host, script, timeout)

    manager._host_run = skewed_prior
    assert manager.compensate_workers().certainty == Certainty.UNKNOWN
    assert slot.activation_ambiguous
    assert manager.cleanup().certainty == Certainty.UNKNOWN


def test_active_unfinalized_release_retains_local_recovery_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, sha = _checkout(tmp_path)
    harness = _HostHarness()
    manager = _manager(tmp_path, monkeypatch, harness)
    slot = manager.prepare(repo, _metadata(repo, sha))
    _attest_prior(manager, repo, sha)

    assert manager.activate_workers().certainty == Certainty.SUCCEEDED
    assert manager.cleanup().certainty == Certainty.UNKNOWN
    assert slot.root.exists()


def test_finalize_attempts_every_host_before_aggregating_failure(tmp_path: Path):
    hosts = (
        HostConfig("one", "tcp://192.0.2.10:26600"),
        HostConfig("two", "tcp://192.0.2.11:26600"),
    )
    config = ClusterConfig(hosts=hosts, transport_security="trusted_fabric")
    calls: list[str] = []
    def host_runner(_config, host, _script, _timeout):
        calls.append(host.name)
        return subprocess.CompletedProcess([], 1 if host.name == "one" else 0,
                                           "FINALIZED\n", "")
    manager = ReleaseManager(config, host_runner=host_runner)
    metadata = ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS)
    manager.slot = ReleaseSlot(
        RELEASE_ID, tmp_path / "slot", tmp_path / "site", metadata,
        remote_activated=True,
    )

    assert manager.finalize().certainty == Certainty.FAILED
    assert calls == ["one", "two"]


def test_finalize_attempts_every_host_and_preserves_cancellation(tmp_path: Path):
    hosts = (
        HostConfig("one", "tcp://192.0.2.10:26600"),
        HostConfig("two", "tcp://192.0.2.11:26600"),
    )
    calls: list[str] = []
    interrupt = KeyboardInterrupt()

    def host_runner(_config, host, _script, _timeout):
        calls.append(host.name)
        if host.name == "one":
            raise interrupt
        return subprocess.CompletedProcess([], 0, "FINALIZED\n", "")

    manager = ReleaseManager(
        ClusterConfig(hosts=hosts, transport_security="trusted_fabric"),
        host_runner=host_runner,
    )
    manager.slot = ReleaseSlot(
        RELEASE_ID, tmp_path / "slot", tmp_path / "site",
        ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS),
        remote_activated=True,
    )
    with pytest.raises(KeyboardInterrupt) as raised:
        manager.finalize()
    assert raised.value is interrupt
    assert calls == ["one", "two"]


def test_finalize_retires_previous_symlink_release(tmp_path: Path):
    home = tmp_path / "home"
    base = home / ".local" / "share" / "dgx-monarch"
    old_release_id = "u-111111111111-2222222222222222"
    old_site = base / "releases" / old_release_id / "site"
    new_site = base / "releases" / RELEASE_ID / "site"
    old_site.mkdir(parents=True)
    new_site.mkdir(parents=True)
    (home / ".local").chmod(0o700)
    (base / "src").symlink_to(new_site, target_is_directory=True)
    transaction = base / "transactions" / RELEASE_ID
    transaction.mkdir(parents=True)
    (transaction / "kind").write_text("symlink", encoding="ascii")
    (transaction / "prior").write_text(str(old_site), encoding="ascii")

    def host_runner(_config, _host, script, _timeout):
        return subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True,
            env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
        )

    manager = ReleaseManager(_config(), host_runner=host_runner)
    manager.slot = ReleaseSlot(
        RELEASE_ID, tmp_path / "slot", tmp_path / "site",
        ReleaseMetadata("2" * 40, "0.5.0", "0.6.0", SOURCE, DEPS),
        remote_activated=True,
    )
    assert manager.finalize().certainty == Certainty.SUCCEEDED
    assert not (base / "releases" / old_release_id).exists()
    assert not transaction.exists()
    assert new_site.is_dir()
