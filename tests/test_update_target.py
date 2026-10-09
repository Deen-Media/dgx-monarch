from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from dgx_monarch.cli import main
from dgx_monarch.cli.update_command import SystemUpdateOps
from dgx_monarch.cli.update_types import DEFAULT_TARGET
from dgx_monarch.config import ClusterConfig


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.mark.parametrize("branch", ["main", "master"])
def test_default_follows_remote_head_not_stale_tracking_ref(tmp_path: Path, branch: str):
    remote = tmp_path / "origin"
    remote.mkdir()
    git(remote, "init", "-b", "old")
    git(remote, "-c", "user.name=Test", "-c", "user.email=test@example.test",
        "commit", "--allow-empty", "-m", "old")
    local = tmp_path / "local"
    subprocess.run(["git", "clone", str(remote), str(local)], check=True, capture_output=True)
    git(remote, "checkout", "-b", branch)
    git(remote, "-c", "user.name=Test", "-c", "user.email=test@example.test",
        "commit", "--allow-empty", "-m", "new")
    assert git(local, "symbolic-ref", "refs/remotes/origin/HEAD").endswith("/old")
    # A clone restricted to its original branch must also follow a changed default.
    git(local, "config", "remote.origin.fetch", "+refs/heads/old:refs/remotes/origin/old")
    ops = SystemUpdateOps(local, ClusterConfig())
    assert ops.resolve_target(DEFAULT_TARGET) == git(remote, "rev-parse", "HEAD")
    with pytest.raises(RuntimeError, match="repeated"):
        ops.resolve_target(DEFAULT_TARGET)


def test_explicit_reference_still_uses_normal_fetch(tmp_path: Path):
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
    ops = SystemUpdateOps(tmp_path, ClusterConfig(), command_runner=run)
    assert ops.resolve_target("origin/master") == "a" * 40
    assert len(calls) == 2
    assert calls[1][0][-1] == "origin/master^{commit}"
    assert calls[0][1]["env"]["GIT_CONFIG_GLOBAL"] == "/dev/null"


@pytest.mark.parametrize("advertisement", [
    "a" * 40 + "\tHEAD\n",
    "ref: refs/tags/main\tHEAD\n" + "a" * 40 + "\tHEAD\n",
    "ref: refs/heads/main\tHEAD\nnot-a-sha\tHEAD\n",
    "ref: refs/heads/main\tHEAD\n" + ("a" * 40 + "\tHEAD\n") * 2,
])
def test_default_rejects_missing_or_ambiguous_remote_head(tmp_path: Path, advertisement: str):
    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, advertisement, "")
    ops = SystemUpdateOps(tmp_path, ClusterConfig(), command_runner=run)
    with pytest.raises(RuntimeError, match="advertised exactly"):
        ops.resolve_target(DEFAULT_TARGET)


def test_default_rejects_remote_branch_change_during_fetch(tmp_path: Path):
    def run(argv, **kwargs):
        output = "ref: refs/heads/main\tHEAD\n" + "a" * 40 + "\tHEAD\n"
        if "rev-parse" in argv:
            output = "b" * 40 + "\n"
        return subprocess.CompletedProcess(argv, 0, output, "")
    ops = SystemUpdateOps(tmp_path, ClusterConfig(), command_runner=run)
    with pytest.raises(RuntimeError, match="changed during"):
        ops.resolve_target(DEFAULT_TARGET)


def test_plain_update_default_keeps_legacy_path(monkeypatch):
    monkeypatch.setattr(main.legacy_update, "run", lambda *args, **kwargs: 17)
    assert main.main(["update"]) == 17
