"""Owned detached-worktree cleanup for verified updates."""
from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from .update_types import Certainty, OperationResult


def remove_worktree(
    checkout: Path | None,
    root: Path,
    git: Callable[..., subprocess.CompletedProcess[str]],
) -> OperationResult:
    """Remove the update's own detached checkout and only its registration.

    After a failed ``git worktree remove`` the folder is deleted and git is
    asked to remove that one registration again, which it does for a folder
    that is gone. No other worktree is pruned: one on a disk that is not
    mounted keeps its registration, index and detached HEAD.
    """
    if checkout is None:
        return OperationResult(Certainty.SUCCEEDED)
    if checkout.parent != root or not checkout.name.startswith("update-"):
        return OperationResult(Certainty.UNKNOWN)
    try:
        result = git("worktree", "remove", "--force", str(checkout), timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return OperationResult(Certainty.UNKNOWN)
    if result.returncode == 0 and not checkout.exists():
        return OperationResult(Certainty.SUCCEEDED)
    if result.returncode != 0:
        try:
            if checkout.is_symlink():
                return OperationResult(Certainty.UNKNOWN)
            shutil.rmtree(checkout)
        except FileNotFoundError:
            pass
        except OSError:
            return OperationResult(Certainty.FAILED)
        if not checkout.exists():
            try:
                listed = git("worktree", "list", "--porcelain")
                if listed.returncode == 0 and f"worktree {checkout}\n" in listed.stdout + "\n":
                    git("worktree", "remove", "--force", str(checkout), timeout=60)
                    listed = git("worktree", "list", "--porcelain")
            except (OSError, subprocess.TimeoutExpired):
                return OperationResult(Certainty.UNKNOWN)
            if listed.returncode == 0 and f"worktree {checkout}\n" not in listed.stdout + "\n":
                return OperationResult(Certainty.SUCCEEDED)
    return OperationResult(Certainty.FAILED)
