"""Path-redacted validation for the already-loaded dgx_monarch and ComfyUI modules."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from importlib import machinery
from pathlib import Path, PurePosixPath
from types import ModuleType

_ORIGIN_DOMAIN = b"DGXM_RUNTIME_MODULE_ORIGINS_V1\0"
_COMFY_PREFIXES = frozenset({
    "api_server",
    "app",
    "comfy",
    "comfy_api",
    "comfy_api_nodes",
    "comfy_config",
    "comfy_execution",
    "comfy_extras",
    "alembic_db",
    "middleware",
    "utils",
})
_COMFY_ROOT_MODULES = frozenset({
    "comfyui_version",
    "cuda_malloc",
    "execution",
    "folder_paths",
    "hook_breaker_ac10a0",
    "latent_preview",
    "main",
    "node_helpers",
    "nodes",
    "protocol",
    "server",
})
_EXTENSION_SUFFIXES = tuple(machinery.EXTENSION_SUFFIXES)


def _named_scope(name: str) -> str | None:
    if name == "dgx_monarch" or name.startswith("dgx_monarch."):
        return "dgx_monarch"
    top_level = name.partition(".")[0]
    if top_level in _COMFY_PREFIXES or name in _COMFY_ROOT_MODULES:
        return "comfyui"
    return None


def _lexical_absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError("runtime module origin is not absolute")
    return Path(os.path.abspath(path))


def _path_scope(path: Path, dgx_root: Path, comfy_root: Path) -> str | None:
    if path.is_relative_to(dgx_root):
        return "dgx_monarch"
    if path.is_relative_to(comfy_root):
        return "comfyui"
    return None


def _canonical_owned_path(
    value: str,
    *,
    root: Path,
    scope: str,
    regular_file: bool,
) -> tuple[Path, str]:
    lexical = _lexical_absolute(value)
    try:
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"{scope} loaded module origin is unavailable") from exc
    expected_type = stat.S_ISREG if regular_file else stat.S_ISDIR
    if (
        lexical != resolved
        or stat.S_ISLNK(metadata.st_mode)
        or not expected_type(metadata.st_mode)
        or not resolved.is_relative_to(root)
    ):
        raise RuntimeError(f"{scope} loaded module escaped its exact source tree")
    logical = resolved.relative_to(root).as_posix() or "."
    return resolved, logical


def _path_is_selected(logical: str, allowed_paths: frozenset[str]) -> bool:
    if logical == ".":
        return bool(allowed_paths)
    prefix = f"{logical}/"
    return logical in allowed_paths or any(
        candidate.startswith(prefix) for candidate in allowed_paths
    )


def _module_row(
    name: str,
    module: ModuleType,
    *,
    scope: str,
    root: Path,
    allowed_paths: frozenset[str],
    source_cache_prefix: PurePosixPath,
) -> dict[str, object]:
    origin = getattr(module, "__file__", None)
    package_paths = getattr(module, "__path__", None)
    if origin is None and package_paths is None:
        raise RuntimeError(f"{scope} loaded module has no exact source origin")

    row: dict[str, object] = {"name": name, "scope": scope}
    if origin is not None:
        if not isinstance(origin, str):
            raise RuntimeError(f"{scope} loaded module origin is invalid")
        resolved, logical = _canonical_owned_path(
            origin, root=root, scope=scope, regular_file=True
        )
        if not (
            resolved.name.endswith(".py")
            or resolved.name.endswith(_EXTENSION_SUFFIXES)
        ) or logical not in allowed_paths:
            raise RuntimeError(
                f"{scope} loaded module origin is outside its attested import surface"
            )
        spec = getattr(module, "__spec__", None)
        spec_origin = getattr(spec, "origin", None)
        if isinstance(spec_origin, str) and spec_origin not in {"built-in", "frozen"}:
            spec_resolved, _ = _canonical_owned_path(
                spec_origin, root=root, scope=scope, regular_file=True
            )
            if spec_resolved != resolved:
                raise RuntimeError(f"{scope} loaded module spec origin drifted")
        if resolved.name.endswith(".py"):
            cached = getattr(module, "__cached__", None)
            if (
                not isinstance(cached, str)
                or not PurePosixPath(cached).is_absolute()
                or not PurePosixPath(cached).is_relative_to(source_cache_prefix)
            ):
                raise RuntimeError(
                    f"{scope} loaded module predates the source-only import policy"
                )
        row["origin"] = logical

    if package_paths is not None:
        try:
            rendered_paths = list(package_paths)
        except TypeError as exc:
            raise RuntimeError(f"{scope} loaded package path is invalid") from exc
        if len(rendered_paths) != 1 or not isinstance(rendered_paths[0], str):
            raise RuntimeError(f"{scope} loaded package path is not exact")
        _resolved, logical = _canonical_owned_path(
            rendered_paths[0], root=root, scope=scope, regular_file=False
        )
        if not _path_is_selected(logical, allowed_paths):
            raise RuntimeError(
                f"{scope} loaded package path is outside its attested import surface"
            )
        row["package_path"] = logical
    return row


def module_origin_rows_sha256(rows: list[dict[str, object]]) -> str:
    """Digest one already path-redacted, canonically ordered origin map."""
    payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(_ORIGIN_DOMAIN + payload).hexdigest()


def loaded_module_origins(
    *,
    dgx_root: Path,
    comfy_root: Path,
    dgx_allowed_paths: frozenset[str],
    comfy_allowed_paths: frozenset[str],
    source_cache_prefix: str,
    _module_items: tuple[tuple[str, object], ...] | None = None,
) -> list[dict[str, object]]:
    """Validate owned module origins and return their path-redacted map."""
    dgx_root = dgx_root.resolve(strict=True)
    comfy_root = comfy_root.resolve(strict=True)
    cache_prefix = PurePosixPath(source_cache_prefix)
    rows: list[dict[str, object]] = []
    module_items = (
        tuple(sys.modules.items()) if _module_items is None else _module_items
    )
    for name, module in sorted(module_items):
        if not isinstance(name, str):
            continue
        scope = _named_scope(name)
        if not isinstance(module, ModuleType):
            if scope is not None:
                raise RuntimeError(f"{scope} loaded module map is incomplete")
            continue
        origin = getattr(module, "__file__", None)
        if scope is None and isinstance(origin, str):
            try:
                scope = _path_scope(
                    _lexical_absolute(origin), dgx_root, comfy_root
                )
            except RuntimeError:
                scope = None
        if scope is None:
            continue
        root = dgx_root if scope == "dgx_monarch" else comfy_root
        allowed = dgx_allowed_paths if scope == "dgx_monarch" else comfy_allowed_paths
        rows.append(
            _module_row(
                name,
                module,
                scope=scope,
                root=root,
                allowed_paths=allowed,
                source_cache_prefix=cache_prefix,
            )
        )
    return rows


def loaded_module_origins_sha256(
    *,
    dgx_root: Path,
    comfy_root: Path,
    dgx_allowed_paths: frozenset[str],
    comfy_allowed_paths: frozenset[str],
    source_cache_prefix: str,
    _module_items: tuple[tuple[str, object], ...] | None = None,
) -> str:
    """Validate owned module origins and return their canonical map digest."""
    return module_origin_rows_sha256(
        loaded_module_origins(
            dgx_root=dgx_root,
            comfy_root=comfy_root,
            dgx_allowed_paths=dgx_allowed_paths,
            comfy_allowed_paths=comfy_allowed_paths,
            source_cache_prefix=source_cache_prefix,
            _module_items=_module_items,
        )
    )
