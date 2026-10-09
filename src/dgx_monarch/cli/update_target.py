"""Resolve a fetched update target, including the remote default branch."""
from __future__ import annotations

import re
import subprocess
from collections.abc import Callable

from .update_types import DEFAULT_TARGET

_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")


def resolve_target(target_ref: str, git: Callable[..., subprocess.CompletedProcess[str]]) -> str:
    if (
        not target_ref or len(target_ref) > 256 or target_ref.startswith("-")
        or any(ord(ch) < 33 for ch in target_ref)
    ):
        raise RuntimeError("update target reference is invalid")
    if target_ref == DEFAULT_TARGET:
        advertised = git("ls-remote", "--symref", "origin", "HEAD", timeout=180)
        lines = advertised.stdout.splitlines()
        refs = [line.removeprefix("ref: ").removesuffix("\tHEAD")
                for line in lines if line.startswith("ref: ") and line.endswith("\tHEAD")]
        shas = [line.removesuffix("\tHEAD") for line in lines
                if not line.startswith("ref: ") and line.endswith("\tHEAD")]
        if (
            advertised.returncode != 0 or len(refs) != 1 or len(shas) != 1
            or not refs[0].startswith("refs/heads/") or _SHA.fullmatch(shas[0]) is None
        ):
            raise RuntimeError("origin default branch was not advertised exactly")
        branch, expected = refs[0], shas[0]
        if git("check-ref-format", branch).returncode != 0:
            raise RuntimeError("origin default branch reference is invalid")
        fetched = git("fetch", "--prune", "origin", branch, timeout=180)
        if fetched.returncode != 0:
            raise RuntimeError("update target fetch failed")
        resolved = git("rev-parse", "--verify", "--end-of-options", "FETCH_HEAD^{commit}")
        if resolved.returncode != 0 or resolved.stdout.strip() != expected:
            raise RuntimeError("origin default branch changed during target resolution")
        return expected
    fetched = git("fetch", "--prune", "origin", timeout=180)
    if fetched.returncode != 0:
        raise RuntimeError("update target fetch failed")
    resolved = git("rev-parse", "--verify", "--end-of-options", f"{target_ref}^{{commit}}")
    target = resolved.stdout.strip()
    if resolved.returncode != 0 or _SHA.fullmatch(target) is None:
        raise RuntimeError("update target did not resolve exactly")
    return target
