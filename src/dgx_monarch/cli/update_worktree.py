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
            pruned = git("worktree", "prune", "--expire", "now")
            listed = git("worktree", "list", "--porcelain")
            if (
                pruned.returncode == 0
                and listed.returncode == 0
                and f"worktree {checkout}" not in listed.stdout
            ):
                return OperationResult(Certainty.SUCCEEDED)
    return OperationResult(Certainty.FAILED)
