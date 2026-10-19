"""Read-only repository observations shared by update controllers."""
from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from .update_types import RepoSnapshot


def git_command(repo: Path, runner: Callable[..., subprocess.CompletedProcess[str]], *args: str,
                cwd: Path | None = None, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return runner(
        ["git", "--no-replace-objects", "-c", "core.hooksPath=/dev/null",
         "-c", "core.fsmonitor=false", "-c", "core.untrackedCache=false",
         "-C", str(cwd or repo), *args],
        capture_output=True, text=True, timeout=timeout,
        env={"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
             "GIT_NO_REPLACE_OBJECTS": "1", "LANG": "C", "LC_ALL": "C", "PATH": os.defpath},
    )


def repository_snapshot(target_sha: str, git: Callable[..., subprocess.CompletedProcess[str]]) -> RepoSnapshot:
    if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", target_sha) is None:
        raise RuntimeError("update target is not a full commit")
    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    assume = git("ls-files", "-v")
    sparse = git("ls-files", "-t")
    if any(result.returncode != 0 for result in (head, status, assume, sparse)):
        raise RuntimeError("repository state query failed")
    hidden = any(line[:1].islower() for line in assume.stdout.splitlines()) or any(
        line.startswith("S ") for line in sparse.stdout.splitlines())
    ff = git("merge-base", "--is-ancestor", head.stdout.strip(), target_sha)
    if ff.returncode not in (0, 1):
        raise RuntimeError("fast-forward query failed")
    return RepoSnapshot(current_sha=head.stdout.strip(), clean=not status.stdout.strip() and not hidden,
                        fast_forward=ff.returncode == 0)
