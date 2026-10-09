"""Read-only inspection of an update checkout at an exact commit."""
from __future__ import annotations

import ast
import hashlib
import importlib.machinery
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

from ..runtime_provenance import dgx_source_manifest_sha256
from .update_release import ReleaseMetadata

_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_EXTENSIONS = tuple(importlib.machinery.EXTENSION_SUFFIXES)


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "--no-replace-objects", "-c", "core.fsmonitor=false",
         "-c", "core.untrackedCache=false", "-C", str(checkout), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.defpath,
        },
    )


def _literal_assignments(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
    values: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (
            isinstance(target, ast.Name)
            and target.id in {"__version__", "TORCHMONARCH_PIN"}
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values[target.id] = node.value.value
    return values


def _tracked_import_paths(checkout: Path, target_sha: str) -> frozenset[str]:
    result = _git(
        checkout, "ls-tree", "-r", "-z", "--name-only", target_sha,
        "--", "src/dgx_monarch",
    )
    if result.returncode != 0:
        raise RuntimeError("target source inventory failed")
    selected: set[str] = set()
    for raw in result.stdout.split("\0"):
        path = PurePosixPath(raw)
        if len(path.parts) < 3:
            continue
        relative = PurePosixPath(*path.parts[2:]).as_posix()
        if relative.endswith(".py") or relative.endswith(_EXTENSIONS):
            selected.add(relative)
    return frozenset(selected)


def inspect_checkout(
    checkout: Path, target_sha: str
) -> tuple[ReleaseMetadata, tuple[str, ...]]:
    """Check HEAD against target_sha, the dependency and pin mirrors, and the importable file set."""
    if _SHA.fullmatch(target_sha) is None:
        raise RuntimeError("target commit is invalid")
    head = _git(checkout, "rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != target_sha:
        raise RuntimeError("detached checkout is not the exact target commit")
    with open(checkout / "pyproject.toml", "rb") as handle:
        project = tomllib.load(handle)["project"]
    dependencies = tuple(project["dependencies"])
    if not dependencies or not all(isinstance(value, str) for value in dependencies):
        raise RuntimeError("target dependencies are invalid")
    requirements = tuple(
        line.strip()
        for line in (checkout / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if requirements != dependencies:
        raise RuntimeError("requirements.txt does not match pyproject dependencies")
    pins = [
        value.partition("==")[2]
        for value in dependencies
        if value.startswith("torchmonarch==")
    ]
    literals = _literal_assignments(checkout / "src" / "dgx_monarch" / "__init__.py")
    if (
        len(pins) != 1
        or _VERSION.fullmatch(pins[0]) is None
        or literals.get("TORCHMONARCH_PIN") != pins[0]
    ):
        raise RuntimeError("target torchmonarch pin is not one exact version equal to TORCHMONARCH_PIN")
    version = literals.get("__version__", "")
    if _VERSION.fullmatch(version) is None:
        raise RuntimeError("target package version is invalid")
    package = checkout / "src" / "dgx_monarch"
    tracked = _tracked_import_paths(checkout, target_sha)
    actual = frozenset(
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file()
        and (path.name.endswith(".py") or path.name.endswith(_EXTENSIONS))
    )
    if actual != tracked:
        raise RuntimeError("target import source does not equal the tracked commit")
    dependency_manifest = hashlib.sha256(
        json.dumps(
            dependencies, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return (
        ReleaseMetadata(
            target_sha,
            version,
            pins[0],
            dgx_source_manifest_sha256(package),
            dependency_manifest,
        ),
        dependencies,
    )
