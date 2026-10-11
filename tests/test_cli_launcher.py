"""Local launcher ownership and argument contracts; no cluster is contacted."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

import pytest

from dgx_monarch.cli import launcher
from dgx_monarch.cli.main import main


@pytest.fixture
def paths(tmp_path):
    target = tmp_path / "venv space ' dollar $/bin/dgxm"
    target.parent.mkdir(parents=True)
    target.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\nsys.exit(7)\n")
    target.chmod(0o755)
    config = tmp_path / "cluster ' $ space.toml"
    config.write_text("[cluster]\n")
    output = tmp_path / "local/bin/dgxm"
    return target, config, output


def apply(paths):
    return launcher.install(*(str(p) for p in paths), apply=True)


def test_plan_creates_nothing_and_apply_is_idempotent(paths):
    assert launcher.install(*(str(p) for p in paths)) == "would create"
    assert not paths[2].parent.exists()
    assert apply(paths) == "installed"
    before = paths[2].stat()
    assert apply(paths) == "unchanged"
    assert paths[2].stat().st_ino == before.st_ino
    assert paths[2].stat().st_mtime_ns == before.st_mtime_ns


def test_exec_preserves_exact_arguments_exit_and_config_override(paths, tmp_path):
    apply(paths)
    result = subprocess.run([str(paths[2]), "top", "--host", "a ' $ b"], capture_output=True, text=True)
    assert result.returncode == 7
    assert json.loads(result.stdout) == ["--config", str(paths[1]), "top", "--host", "a ' $ b"]
    override = tmp_path / "override.toml"
    override.write_text("[cluster]\n")
    paths[1].unlink()
    for args in (["--config", str(override)], [f"--config={override}"]):
        result = subprocess.run([str(paths[2]), *args, "version"], capture_output=True, text=True)
        assert result.returncode == 7
        assert json.loads(result.stdout) == [*args, "version"]
    result = subprocess.run([str(paths[2]), "top"], capture_output=True, text=True)
    assert result.returncode == 2
    assert not result.stdout
    assert "config is missing" in result.stderr


def test_exec_preserves_signal_identity(paths):
    paths[0].write_text(f"#!{sys.executable}\nimport os,signal\nos.kill(os.getpid(),signal.SIGTERM)\n")
    apply(paths)
    assert subprocess.run([str(paths[2]), "version"]).returncode == -signal.SIGTERM


@pytest.mark.parametrize("config", ["~/cluster.toml", "~another-user/cluster.toml"])
def test_explicit_home_overrides_reach_native_cli_unchanged(paths, config):
    apply(paths)
    paths[1].unlink()
    for option in (["--config", config], [f"--config={config}"]):
        result = subprocess.run([str(paths[2]), *option, "top"], capture_output=True, text=True)
        assert result.returncode == 7
        assert json.loads(result.stdout) == [*option, "top"]


def test_rerun_restores_owner_execution_even_when_other_execute_bit_is_set(paths):
    apply(paths)
    paths[2].chmod(0o645)
    assert apply(paths) == "installed"
    assert paths[2].stat().st_mode & 0o777 == 0o755
    assert subprocess.run([str(paths[2]), "version"], capture_output=True).returncode == 7


@pytest.mark.parametrize("apply_change", [False, True])
@pytest.mark.parametrize("target_kind", ["managed", "symlink", "edited"])
def test_managed_targets_refused_without_changing_either_launcher(paths, tmp_path, apply_change, target_kind):
    apply(paths)
    first = paths[2]
    second = tmp_path / "other-bin/dgxm"
    # Represent an existing chain from the previous installer without running it.
    second.parent.mkdir()
    second.write_bytes(launcher._render(first, paths[1]))
    second.chmod(0o755)
    if target_kind == "edited":
        second.write_bytes(second.read_bytes().replace(
            b"# dgx-monarch-launcher-v1 ", b"#  dgx-monarch-launcher-v1 broken ",
        ))
    target = second
    if target_kind == "symlink":
        target = tmp_path / "alias"
        target.symlink_to(second)
    before = {path: (path.read_bytes(), path.stat().st_ino) for path in (first, second)}

    with pytest.raises(ValueError, match="managed launcher"):
        launcher.install(str(target), str(paths[1]), str(first), apply=apply_change)

    assert {path: (path.read_bytes(), path.stat().st_ino) for path in (first, second)} == before


def test_new_launcher_cannot_target_existing_managed_launcher(paths, tmp_path):
    apply(paths)
    output = tmp_path / "new-bin/dgxm"
    with pytest.raises(ValueError, match="managed launcher"):
        launcher.install(str(paths[2]), str(paths[1]), str(output), apply=True)
    assert not output.parent.exists()


@pytest.mark.parametrize("kind", ["foreign", "symlink", "directory", "edited"])
def test_preserves_existing_commands(paths, kind):
    output = paths[2]
    output.parent.mkdir(parents=True)
    output.parent.chmod(0o755)
    if kind == "foreign":
        output.write_text("keep me")
    elif kind == "symlink":
        output.symlink_to(paths[0])
    elif kind == "directory":
        output.mkdir()
    else:
        apply(paths)
        output.write_bytes(output.read_bytes() + b"# user edit\n")
    before = output.lstat()
    with pytest.raises((ValueError, OSError)):
        apply(paths)
    assert output.lstat().st_ino == before.st_ino
    assert output.lstat().st_size == before.st_size


def test_owned_launcher_rebinds_config(paths, tmp_path):
    apply(paths)
    config = tmp_path / "new.toml"
    config.write_text("[cluster]\n")
    changed = (paths[0], config, paths[2])
    assert apply(changed) == "installed"
    result = subprocess.run([str(paths[2]), "version"], capture_output=True, text=True)
    assert json.loads(result.stdout)[:2] == ["--config", str(config)]


def test_create_race_never_clobbers_other_command(paths, monkeypatch):
    link = os.link
    def race(source, destination, **kwargs):
        paths[2].write_text("foreign")
        return link(source, destination, **kwargs)
    monkeypatch.setattr(launcher.os, "link", race)
    with pytest.raises(FileExistsError):
        apply(paths)
    assert paths[2].read_text() == "foreign"


def test_replace_rechecks_identity_even_for_identical_bytes(paths, tmp_path, monkeypatch):
    apply(paths)
    original = launcher._read_owned
    calls = 0
    def race(parent, name):
        nonlocal calls
        calls += 1
        if calls == 2:
            fresh = tmp_path / "foreign"
            fresh.write_bytes(paths[2].read_bytes())
            fresh.chmod(0o755)
            fresh.replace(paths[2])
        return original(parent, name)
    monkeypatch.setattr(launcher, "_read_owned", race)
    config = tmp_path / "new.toml"
    config.write_text("[cluster]\n")
    with pytest.raises(ValueError, match="changed"):
        apply((paths[0], config, paths[2]))
    assert str(paths[1]) in paths[2].read_text()


def test_symlink_parent_and_self_target_refused(paths, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(OSError):
        apply((paths[0], paths[1], alias / "dgxm"))
    with pytest.raises(ValueError, match="itself"):
        apply((paths[0], paths[1], paths[0]))


def test_cli_reports_path_shadowing_without_shell_changes(paths, tmp_path, monkeypatch, capsys):
    other = tmp_path / "other"
    other.mkdir()
    command = other / "dgxm"
    command.write_text("#!/bin/sh\nexit 0\n")
    command.chmod(0o755)
    monkeypatch.setenv("PATH", str(other))
    args = ["launcher", "--executable", str(paths[0]), "--cluster-config", str(paths[1]), "--destination", str(paths[2])]
    assert main(args) == 0
    assert not paths[2].exists()
    assert "another dgxm" in capsys.readouterr().out
    assert main([*args, "--apply"]) == 0
    monkeypatch.setenv("PATH", str(paths[2].parent))
    assert main(args) == 0
    assert "PATH selects the launcher" in capsys.readouterr().out


def test_missing_config_and_conflicting_modes_fail_without_install(paths):
    paths[1].unlink()
    with pytest.raises(ValueError, match="config"):
        apply(paths)
    assert not paths[2].parent.exists()
    with pytest.raises(SystemExit) as error:
        main(["launcher", "--executable", str(paths[0]), "--cluster-config", str(paths[1]), "--apply", "--check"])
    assert error.value.code == 2


def test_cooperating_installers_refuse_busy_lock(paths):
    import fcntl

    apply(paths)
    lock = paths[2].parent / ".dgxm-launcher.lock"
    with lock.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            apply(paths)
