from __future__ import annotations

import hashlib
import json
import signal
import subprocess
import sys
from unittest.mock import Mock

import pytest

from dgx_monarch.cli import update_bootstrap as bootstrap


def source_tree(tmp_path):
    source = tmp_path / "source"
    (source / "cli").mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (source / "cli" / "__init__.py").write_text("")
    (source / "cli" / "update_bootstrap.py").write_text(
        'def _child(root, manifest):\n'
        '    from .late import VALUE\n'
        '    print(VALUE)\n'
        '    print(manifest["request"]["repo"])\n'
        '    return 0\n'
    )
    (source / "cli" / "late.py").write_text('VALUE = "original"\n')
    return source


def run_snapshot(snapshot, digest):
    return subprocess.run(
        [sys.executable, "-I", "-B", "-c", bootstrap._LOADER, str(snapshot), digest],
        capture_output=True, text=True, timeout=10,
    )


def test_snapshot_late_import_survives_original_checkout_change(tmp_path):
    source = source_tree(tmp_path)
    original_repo = str(tmp_path / "original-checkout")
    snapshot, digest = bootstrap._snapshot(source, tmp_path / "private", {"repo": original_repo})
    (source / "cli" / "late.py").write_text('raise RuntimeError("new checkout")\n')
    result = run_snapshot(snapshot, digest)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["original", original_repo]
    assert snapshot.stat().st_mode & 0o777 == 0o700
    assert not list(snapshot.rglob("*.pyc"))


@pytest.mark.parametrize("mutation", ["source", "manifest", "extra", "symlink"])
def test_loader_refuses_changed_inventory_before_import(tmp_path, mutation):
    source = source_tree(tmp_path)
    snapshot, digest = bootstrap._snapshot(source, tmp_path / "private", {"repo": "/original"})
    if mutation == "manifest":
        (snapshot / "manifest.json").write_text("{}")
    elif mutation == "extra":
        (snapshot / "dgx_monarch" / "extra.py").write_text("")
    else:
        target = snapshot / "dgx_monarch" / "cli" / "late.py"
        target.unlink()
        if mutation == "symlink":
            target.symlink_to(source / "cli" / "late.py")
        else:
            target.write_text('VALUE="tampered"')
    assert run_snapshot(snapshot, digest).returncode != 0


@pytest.mark.parametrize("kind", ["symlink", "public"])
def test_snapshot_refuses_unsafe_root(tmp_path, kind):
    source = source_tree(tmp_path)
    root = tmp_path / "private"
    if kind == "symlink":
        root.symlink_to(tmp_path, target_is_directory=True)
    else:
        root.mkdir(mode=0o755)
    with pytest.raises(ValueError):
        bootstrap._snapshot(source, root, {})


def test_snapshot_refuses_source_symlink(tmp_path):
    source = source_tree(tmp_path)
    (source / "alias.py").symlink_to(source / "__init__.py")
    with pytest.raises((RuntimeError, ValueError)):
        bootstrap._snapshot(source, tmp_path / "private", {})


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(19)])
def test_cancellation_forwards_once_and_retains_recovery(tmp_path, error):
    source = source_tree(tmp_path)
    snapshot, digest = bootstrap._snapshot(source, tmp_path / "private", {})
    process = Mock(pid=123)
    process.wait.side_effect = [error, subprocess.TimeoutExpired("controller", 1)]
    with pytest.raises(type(error)) as caught:
        bootstrap._wait_child(process, snapshot, timeout=1)
    assert caught.value is error
    process.send_signal.assert_called_once_with(signal.SIGINT)
    assert hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest() == digest
    assert json.loads((snapshot / "manifest.json").read_text())["files"]
    process.kill.assert_not_called()


def test_invalid_receipt_refused_before_source_or_subprocess(tmp_path, monkeypatch):
    spawn = Mock()
    monkeypatch.setattr(bootstrap.subprocess, "Popen", spawn)
    from dgx_monarch.config import ClusterConfig
    assert bootstrap.launch_verified_update(
        ClusterConfig(), repo=tmp_path, target_ref="origin/master", assume_yes=True,
        driver_host=None, receipt_path="relative.json",
    ) == 2
    spawn.assert_not_called()


def test_snapshot_rejects_mid_copy_source_change(tmp_path, monkeypatch):
    source = source_tree(tmp_path)
    read = bootstrap._regular_bytes

    def changing_read(path):
        result = read(path)
        if path.name == "late.py":
            path.write_text('VALUE="changed"\n')
        return result

    monkeypatch.setattr(bootstrap, "_regular_bytes", changing_read)
    root = tmp_path / "private"
    with pytest.raises(ValueError, match="changed during snapshot"):
        bootstrap._snapshot(source, root, {})
    assert not list(root.iterdir())


@pytest.mark.parametrize("changed", ["none", "source", "identity"])
def test_child_locks_original_repo_and_publishes_settlement(tmp_path, monkeypatch, changed):
    from dgx_monarch.cli import update_command, update_entrypoint
    from dgx_monarch.config import ClusterConfig
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("")
    config = ClusterConfig(source=str(config_path))
    runtime = Mock()
    create_ops = Mock(return_value=runtime)
    serialized = Mock(return_value=0)
    monkeypatch.setattr(bootstrap, "load_cluster_config", lambda path: config)
    monkeypatch.setattr(update_command, "SystemUpdateOps", create_ops)
    monkeypatch.setattr(update_entrypoint, "run_serialized_update", serialized)
    monkeypatch.setattr(bootstrap.signal, "signal", Mock())
    repo = tmp_path / "original-repo"
    repo.mkdir()
    snapshot = tmp_path / "controller"
    snapshot.mkdir()
    monkeypatch.setattr(bootstrap, "dgx_source_manifest_sha256", lambda path: "a" * 64)
    manifest = {"source_manifest": "a" * 64, "request": {
        "config": str(config_path), "config_sha256": hashlib.sha256(b"").hexdigest(),
        "repo": str(repo), "repo_identity": [repo.stat().st_dev, repo.stat().st_ino], "target_ref": "origin/master", "assume_yes": True,
        "driver_host": None, "receipt_path": str(tmp_path / "receipt.json"),
    }}
    if changed == "source":
        manifest["source_manifest"] = "b" * 64
    elif changed == "identity":
        manifest["request"]["repo_identity"][1] += 1
    if changed != "none":
        with pytest.raises(ValueError, match="original"):
            bootstrap._child(snapshot, manifest)
        serialized.assert_not_called()
        create_ops.assert_not_called()
        return
    assert bootstrap._child(snapshot, manifest) == 0
    assert serialized.call_args.kwargs["repo"] == repo
    assert serialized.call_args.kwargs["receipt_path"] == str(tmp_path / "receipt.json")
    assert create_ops.call_args.args == (repo, config)
    assert json.loads((snapshot / "settled.json").read_text()) == {"exit_code": 0}


@pytest.mark.parametrize("result", [0, 1, 130])
def test_launcher_cleans_only_confirmed_success(tmp_path, monkeypatch, result):
    from dgx_monarch.config import ClusterConfig
    config_path = tmp_path / "cluster.toml"
    config_path.write_text("")
    config = ClusterConfig(source=str(config_path))
    monkeypatch.setattr(bootstrap, "load_cluster_config", lambda path: config)
    monkeypatch.setattr(bootstrap, "dgx_source_manifest_sha256", lambda path: "a" * 64)
    snapshot = tmp_path / "controller"
    snapshot.mkdir()
    captured = {}

    def snapshot_source(source, root, request):
        captured.update(request)
        return snapshot, "b" * 64

    process = Mock(pid=42)
    process.wait.return_value = result
    if result == 0:
        (snapshot / "settled.json").write_text('{"exit_code": 0}')
    spawn = Mock(return_value=process)
    monkeypatch.setattr(bootstrap, "_snapshot", snapshot_source)
    monkeypatch.setattr(bootstrap.subprocess, "Popen", spawn)
    assert bootstrap.launch_verified_update(
        config, repo=tmp_path, target_ref="origin/master", assume_yes=True,
        driver_host=None, receipt_path=None,
    ) == result
    assert captured["repo"] == str(tmp_path.resolve())
    assert spawn.call_args.args[0][:3] == [sys.executable, "-I", "-B"]
    assert spawn.call_args.kwargs["start_new_session"] is True
    assert snapshot.exists() == (result != 0)
