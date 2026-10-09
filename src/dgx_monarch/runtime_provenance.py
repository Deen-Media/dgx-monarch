"""Strict, path-redacted provenance for the code loaded by a worker.

This module must stay stdlib-only at import time, so the coordinator and
worker processes share this source inventory without importing
:mod:`dgx_monarch.actor`, which builds the GPU actor class and pulls in torch.
The snapshot function imports its runtime-only dependencies itself.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
from functools import lru_cache
from importlib import machinery
from pathlib import Path, PurePosixPath
from typing import Any

from .runtime_identity import _stat_tuple
from .runtime_identity import machine_identity_sha256 as _machine_identity_sha256
from .runtime_identity import resolved_artifact_identities as _resolved_artifact_identities
from .runtime_identity import runtime_instance_sha256 as _runtime_instance_sha256
from .runtime_identity import runtime_versions as _runtime_versions

_MANIFEST_DOMAIN = b"DGXM_RUNTIME_SOURCE_MANIFEST_V1\0"
_GIT_TIMEOUT_S = 10
_SOURCE_ANYWHERE_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "venv",
}
_COMFY_SOURCE_ROOT_EXCLUDED_DIRS = {
    "input",
    "models",
    "output",
    "temp",
    "user",
}
_EXTENSION_SUFFIXES = tuple(machinery.EXTENSION_SUFFIXES)
_FULL_COMMIT = re.compile(r"[0-9a-f]{40}")
SOURCE_ONLY_PYCACHE_PREFIX = "/proc/self/fd/2147483647"


def enforce_source_only_imports() -> None:
    """Make later imports compile tracked source instead of tree-local pyc files."""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = SOURCE_ONLY_PYCACHE_PREFIX
    sys.dont_write_bytecode = True
    sys.pycache_prefix = SOURCE_ONLY_PYCACHE_PREFIX


def _source_only_imports_enforced() -> bool:
    return bool(
        sys.dont_write_bytecode
        and sys.pycache_prefix == SOURCE_ONLY_PYCACHE_PREFIX
    )


def _is_importable_filename(filename: str) -> bool:
    return filename.endswith(".py") or filename.endswith(_EXTENSION_SUFFIXES)


def _excluded_parts(
    relative: PurePosixPath, *, scope: str, custom_nodes_disabled: bool
) -> bool:
    parts = relative.parts
    if any(part in _SOURCE_ANYWHERE_EXCLUDED_DIRS for part in parts[:-1]):
        return True
    if scope == "comfyui" and parts:
        root_exclusions = _COMFY_SOURCE_ROOT_EXCLUDED_DIRS | (
            {"custom_nodes"} if custom_nodes_disabled else set()
        )
        if parts[0] in root_exclusions:
            return True
    return False


def _selected_logical_path(
    relative: PurePosixPath, *, scope: str, custom_nodes_disabled: bool
) -> bool:
    if _excluded_parts(
        relative, scope=scope, custom_nodes_disabled=custom_nodes_disabled
    ):
        return False
    filename = relative.name
    if filename.endswith(".pyc"):
        raise RuntimeError(
            f"{scope} runtime source contains a direct or sourceless .pyc"
        )
    return _is_importable_filename(filename)


def _raise_walk_error(error: OSError) -> None:
    raise error


def _source_inventory(
    root: Path, *, scope: str, custom_nodes_disabled: bool
) -> list[Path]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError(f"{scope} runtime source root is not a directory")
    paths: list[Path] = []
    for current, directory_names, filenames in os.walk(
        root, followlinks=False, onerror=_raise_walk_error):
        current_path = Path(current)
        current_relative = current_path.relative_to(root)
        retained: list[str] = []
        for name in sorted(directory_names):
            relative = PurePosixPath(*current_relative.parts, name)
            if _excluded_parts(
                relative / "placeholder",
                scope=scope,
                custom_nodes_disabled=custom_nodes_disabled,
            ):
                continue
            candidate = current_path / name
            if candidate.is_symlink():
                raise RuntimeError(f"{scope} source has an included symlink directory")
            retained.append(name)
        directory_names[:] = retained
        for filename in sorted(filenames):
            relative = PurePosixPath(*current_relative.parts, filename)
            if not _selected_logical_path(
                relative,
                scope=scope,
                custom_nodes_disabled=custom_nodes_disabled,
            ):
                continue
            candidate = current_path / filename
            before = candidate.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise RuntimeError(
                    f"{scope} runtime source contains a non-regular import file")
            paths.append(candidate)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _stable_tree_manifest(
    root: Path,
    paths: list[Path],
    *,
    scope: str,
    custom_nodes_disabled: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(_MANIFEST_DOMAIN)
    encoded_scope = scope.encode("ascii")
    digest.update(len(encoded_scope).to_bytes(2, "big"))
    digest.update(encoded_scope)
    digest.update(len(paths).to_bytes(8, "big"))
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        descriptor = os.open(path, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise RuntimeError(
                    f"{scope} runtime source is not a regular file"
                )
            digest.update(int(before.st_size).to_bytes(8, "big"))
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        path_after = path.lstat()
        if (
            _stat_tuple(before) != _stat_tuple(after)
            or _stat_tuple(after) != _stat_tuple(path_after)
        ):
            raise RuntimeError(f"{scope} runtime source changed while hashing")
    if paths != _source_inventory(
        root,
        scope=scope,
        custom_nodes_disabled=custom_nodes_disabled,
    ):
        raise RuntimeError(f"{scope} runtime source set changed while hashing")
    return digest.hexdigest()


def source_manifest_sha256(
    root: Path,
    *,
    scope: str,
    custom_nodes_disabled: bool = False,
    expected_paths: frozenset[str] | None = None,
) -> str:
    """Hash one stable import-capable tree with optional HEAD-set binding."""
    root = root.resolve(strict=True)
    paths = _source_inventory(
        root,
        scope=scope,
        custom_nodes_disabled=custom_nodes_disabled,
    )
    logical_paths = frozenset(path.relative_to(root).as_posix() for path in paths)
    if expected_paths is not None and logical_paths != expected_paths:
        raise RuntimeError(
            f"{scope} live import files do not equal the files tracked at HEAD"
        )
    if not paths:
        raise RuntimeError(f"{scope} runtime source manifest is empty")
    return _stable_tree_manifest(
        root,
        paths,
        scope=scope,
        custom_nodes_disabled=custom_nodes_disabled,
    )


def dgx_source_manifest_sha256(root: Path | None = None) -> str:
    """Return the uncached canonical digest for a dgx_monarch package tree."""
    package_root = (root or Path(__file__).resolve().parent).resolve(strict=True)
    return source_manifest_sha256(package_root, scope="dgx_monarch")


@lru_cache(maxsize=1)
def cached_dgx_source_manifest_sha256() -> str:
    """Process-lifetime source identity for Gate authority and diagnostics."""
    return dgx_source_manifest_sha256()


@lru_cache(maxsize=1)
def _git_executable() -> tuple[str, tuple[int, int, int, int, int]]:
    """Pin the Git executable the operator's PATH selects; ``_run_git`` runs it with the default PATH."""
    candidate = shutil.which("git")
    try:
        resolved = Path(candidate).resolve(strict=True) if candidate else None
        metadata = resolved.stat() if resolved is not None else None
    except OSError as exc:
        raise RuntimeError("runtime provenance Git executable is unavailable") from exc
    if (
        resolved is None
        or metadata is None
        or not stat.S_ISREG(metadata.st_mode)
        or not os.access(resolved, os.X_OK)
    ):
        raise RuntimeError("runtime provenance Git executable is unavailable")
    return str(resolved), _stat_tuple(metadata)


def _run_git(
    root: Path, *arguments: str, binary: bool = False
) -> subprocess.CompletedProcess[Any]:
    executable, executable_identity = _git_executable()
    base = [
        executable,
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(root),
    ]
    try:
        before = os.stat(executable)
        if (
            _stat_tuple(before) != executable_identity
            or not stat.S_ISREG(before.st_mode)
            or not os.access(executable, os.X_OK)
        ):
            raise OSError("Git executable identity changed")
        result = subprocess.run(
            [*base, *arguments],
            check=False,
            capture_output=True,
            env={
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_NO_REPLACE_OBJECTS": "1",
                "LANG": "C",
                "LC_ALL": "C",
                "PATH": os.defpath,
            },
            text=not binary,
            timeout=_GIT_TIMEOUT_S,
        )
        if _stat_tuple(os.stat(executable)) != executable_identity:
            raise OSError("Git executable identity changed")
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("runtime provenance Git query failed") from exc
    if result.returncode != 0:
        raise RuntimeError("runtime provenance Git query failed")
    return result


def _tracked_source_paths(
    payload: bytes,
    *,
    tracked_prefix: str,
    scope: str,
    custom_nodes_disabled: bool,
) -> frozenset[str]:
    prefix = (
        PurePosixPath(tracked_prefix).as_posix().strip("/")
        if tracked_prefix
        else ""
    )
    result: set[str] = set()
    for raw_path in payload.split(b"\0"):
        if not raw_path:
            continue
        try:
            checkout_path = PurePosixPath(raw_path.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise RuntimeError("runtime provenance Git path is not UTF-8") from exc
        checkout_text = checkout_path.as_posix()
        if prefix:
            marker = f"{prefix}/"
            if not checkout_text.startswith(marker):
                continue
            logical = PurePosixPath(checkout_text[len(marker) :])
        else:
            logical = checkout_path
        if _selected_logical_path(
            logical,
            scope=scope,
            custom_nodes_disabled=custom_nodes_disabled,
        ):
            result.add(logical.as_posix())
    return frozenset(result)


def _git_checkout_facts(
    root: Path,
    *,
    tracked_prefix: str,
    scope: str,
    custom_nodes_disabled: bool,
    allow_managed_copy: bool = False,
) -> tuple[dict[str, object], frozenset[str] | None]:
    root = root.resolve(strict=True)
    if not os.path.lexists(root / ".git"):
        if allow_managed_copy:
            return {"mode": "managed_copy"}, None
        raise RuntimeError("runtime provenance requires an exact Git checkout")
    commit = _run_git(root, "rev-parse", "HEAD").stdout.strip()
    top_level = Path(
        _run_git(root, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve(strict=True)
    if top_level != root:
        raise RuntimeError("runtime provenance Git root does not match import root")
    status_arguments = [
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    ]
    pathspec = (
        ("--", ".", ":(top,exclude,literal)custom_nodes")
        if scope == "comfyui" and custom_nodes_disabled
        else ()
    )
    status_arguments.extend(pathspec)
    status_result = _run_git(root, *status_arguments)
    assume_flags = _run_git(root, "ls-files", "-v", *pathspec).stdout.splitlines()
    type_flags = _run_git(root, "ls-files", "-t", *pathspec).stdout.splitlines()
    tracked_result = _run_git(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--name-only",
        commit,
        "--",
        tracked_prefix or ".",
        binary=True,
    )
    if _FULL_COMMIT.fullmatch(commit) is None:
        raise RuntimeError("runtime provenance Git HEAD is not a full commit")
    hidden_flags = any(line[:1].islower() for line in assume_flags) or any(
        line.startswith("S ") for line in type_flags
    )
    tracked = _tracked_source_paths(
        tracked_result.stdout,
        tracked_prefix=tracked_prefix,
        scope=scope,
        custom_nodes_disabled=custom_nodes_disabled,
    )
    return (
        {
            "mode": "git",
            "commit": commit,
            "dirty": bool(status_result.stdout.strip()) or hidden_flags,
        },
        tracked,
    )


def _validate_managed_pythonpath(package_root: Path) -> None:
    pythonpath_root = package_root.parent
    try:
        entries = list(os.scandir(pythonpath_root))
    except OSError as exc:
        raise RuntimeError("managed runtime Python path is unavailable") from exc
    if len(entries) != 1 or entries[0].name != "dgx_monarch":
        raise RuntimeError(
            "managed runtime Python path contains an unexpected sibling entry"
        )
    metadata = entries[0].stat(follow_symlinks=False)
    if entries[0].is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("managed dgx_monarch package root is not a real directory")


def _require_clean_git(facts: dict[str, object], *, scope: str) -> None:
    if facts.get("mode") == "git" and facts.get("dirty") is not False:
        raise RuntimeError(f"{scope} runtime Git checkout is not clean")


def _live_source_paths(
    root: Path, *, scope: str, custom_nodes_disabled: bool
) -> frozenset[str]:
    return frozenset(
        path.relative_to(root).as_posix()
        for path in _source_inventory(
            root,
            scope=scope,
            custom_nodes_disabled=custom_nodes_disabled,
        )
    )


def runtime_provenance_snapshot(
    artifact_manifest: object = None,
    *,
    custom_nodes_disabled: bool,
) -> dict[str, object]:
    """Capture fresh, commit-bound source, package, and artifact facts."""
    if artifact_manifest is not None and not isinstance(artifact_manifest, list):
        raise ValueError("runtime provenance artifact manifest is invalid")
    normalized_artifact_manifest = (
        [] if artifact_manifest is None else artifact_manifest
    )
    if not _source_only_imports_enforced():
        raise RuntimeError(
            "runtime provenance requires source-only imports with bytecode caches disabled"
        )
    import comfy
    import comfy.model_management as comfy_model_management
    import folder_paths

    from .gate_ledger import GATE_PROTOCOL_VERSION
    from .runtime_module_origins import (
        loaded_module_origins,
        module_origin_rows_sha256,
    )

    package_root = Path(__file__).resolve().parent
    checkout_root = package_root.parents[1]
    comfy_root = Path(folder_paths.__file__).resolve().parent
    comfy_origin = Path(comfy_model_management.__file__).resolve()
    comfy_paths = [Path(path).resolve() for path in comfy.__path__]
    if comfy_paths != [(comfy_root / "comfy").resolve()] or not comfy_origin.is_relative_to(
        comfy_root
    ):
        raise RuntimeError("Comfy runtime import escaped its exact checkout")

    dgx_git_before, dgx_tracked_before = _git_checkout_facts(
        checkout_root,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
        allow_managed_copy=True,
    )
    comfy_git_before, comfy_tracked_before = _git_checkout_facts(
        comfy_root,
        tracked_prefix="",
        scope="comfyui",
        custom_nodes_disabled=custom_nodes_disabled,
    )
    _require_clean_git(dgx_git_before, scope="dgx_monarch")
    _require_clean_git(comfy_git_before, scope="ComfyUI")
    if dgx_git_before.get("mode") == "managed_copy":
        _validate_managed_pythonpath(package_root)
    dgx_allowed_before = (
        dgx_tracked_before
        if dgx_tracked_before is not None
        else _live_source_paths(
            package_root,
            scope="dgx_monarch",
            custom_nodes_disabled=False,
        )
    )
    if comfy_tracked_before is None:
        raise RuntimeError("ComfyUI runtime provenance lacks tracked source paths")
    versions = _runtime_versions()
    module_origins_before = loaded_module_origins(
        dgx_root=package_root,
        comfy_root=comfy_root,
        dgx_allowed_paths=dgx_allowed_before,
        comfy_allowed_paths=comfy_tracked_before,
        source_cache_prefix=SOURCE_ONLY_PYCACHE_PREFIX,
    )
    dgx_source_before = source_manifest_sha256(
        package_root,
        scope="dgx_monarch",
        expected_paths=dgx_tracked_before,
    )
    comfy_source_before = source_manifest_sha256(
        comfy_root,
        scope="comfyui",
        custom_nodes_disabled=custom_nodes_disabled,
        expected_paths=comfy_tracked_before,
    )
    artifacts = _resolved_artifact_identities(normalized_artifact_manifest)
    dgx_source_after = source_manifest_sha256(
        package_root,
        scope="dgx_monarch",
        expected_paths=dgx_tracked_before,
    )
    comfy_source_after = source_manifest_sha256(
        comfy_root,
        scope="comfyui",
        custom_nodes_disabled=custom_nodes_disabled,
        expected_paths=comfy_tracked_before,
    )
    dgx_git_after, dgx_tracked_after = _git_checkout_facts(
        checkout_root,
        tracked_prefix="src/dgx_monarch",
        scope="dgx_monarch",
        custom_nodes_disabled=False,
        allow_managed_copy=True,
    )
    comfy_git_after, comfy_tracked_after = _git_checkout_facts(
        comfy_root,
        tracked_prefix="",
        scope="comfyui",
        custom_nodes_disabled=custom_nodes_disabled,
    )
    _require_clean_git(dgx_git_after, scope="dgx_monarch")
    _require_clean_git(comfy_git_after, scope="ComfyUI")
    if dgx_git_after.get("mode") == "managed_copy":
        _validate_managed_pythonpath(package_root)
    dgx_allowed_after = (
        dgx_tracked_after
        if dgx_tracked_after is not None
        else _live_source_paths(
            package_root,
            scope="dgx_monarch",
            custom_nodes_disabled=False,
        )
    )
    if comfy_tracked_after is None:
        raise RuntimeError("ComfyUI runtime provenance lacks tracked source paths")
    module_origins_after = loaded_module_origins(
        dgx_root=package_root,
        comfy_root=comfy_root,
        dgx_allowed_paths=dgx_allowed_after,
        comfy_allowed_paths=comfy_tracked_after,
        source_cache_prefix=SOURCE_ONLY_PYCACHE_PREFIX,
    )
    if (
        dgx_source_before != dgx_source_after
        or comfy_source_before != comfy_source_after
        or dgx_git_before != dgx_git_after
        or comfy_git_before != comfy_git_after
        or dgx_tracked_before != dgx_tracked_after
        or comfy_tracked_before != comfy_tracked_after
        or module_origins_before != module_origins_after
    ):
        raise RuntimeError("runtime provenance changed during the fresh snapshot")
    return {
        "schema": 1,
        "machine_identity_sha256": _machine_identity_sha256(),
        "runtime_instance_sha256": _runtime_instance_sha256(),
        "custom_nodes_disabled": custom_nodes_disabled,
        "source_only_imports": True,
        "loaded_module_origins": module_origins_after,
        "loaded_module_origins_sha256": module_origin_rows_sha256(
            module_origins_after
        ),
        "versions": versions,
        "gate_protocol": GATE_PROTOCOL_VERSION,
        "dgx_monarch": {
            **dgx_git_after,
            "source_manifest_sha256": dgx_source_after,
        },
        "comfyui": {
            **comfy_git_after,
            "source_manifest_sha256": comfy_source_after,
        },
        "resolved_artifacts": artifacts,
    }
