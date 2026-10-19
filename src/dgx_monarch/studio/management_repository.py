"""Repository-only source review for the existing config-free legacy update."""
from __future__ import annotations

import subprocess
from pathlib import Path

from ..cli.update_repository import git_command, repository_snapshot
from ..cli.update_types import RepoSnapshot


class RepositoryOps:
    def __init__(self, repo: Path):
        self.repo = repo.resolve()

    def _git(self, *args: str, cwd: Path | None = None, timeout: float = 60) -> subprocess.CompletedProcess[str]:
        return git_command(self.repo, subprocess.run, *args, cwd=cwd, timeout=timeout)

    def repo_snapshot(self, target_sha: str) -> RepoSnapshot:
        return repository_snapshot(target_sha, self._git)
