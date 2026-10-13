"""Path-redacted Monarch artifact and native-load facts, without hardware probes.

Importing this module is stdlib-only. Explicit collection imports Torch and
Monarch, but never creates an actor, RDMA manager, buffer, or device context.
A loadable native binding is not evidence of a usable GPU or fabric. Metadata
ownership and RECORD identity do not verify installed payload bytes.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any


def _origin(module: Any, distribution: Any) -> dict[str, object]:
    # Ownership only: a digest of the resolved path names the install location
    # to anyone who can guess it, and committed snapshots carry these facts.
    filename = getattr(module, "__file__", None)
    if not filename:
        return {"distribution_owned": False}
    origin = Path(filename).resolve()
    owned = any(
        Path(distribution.locate_file(entry)).resolve() == origin
        for entry in distribution.files or ()
    )
    return {"distribution_owned": owned}


def collect_capabilities() -> dict[str, Any]:
    """Collect native-load identity; errors expose their type, never raw paths."""
    result: dict[str, Any] = {
        "schema": 1,
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "torch": {},
        "torchmonarch": {},
        "rdma": {"native_import": "not_run", "empty_action": "not_run",
                 "hardware": "not_probed"},
    }
    for package in ("torch", "torchmonarch"):
        facts = result[package]
        try:
            distribution = importlib.metadata.distribution(package)
            facts["distribution_version"] = distribution.version
            if package == "torchmonarch":
                record = distribution.read_text("RECORD")
                facts["record_sha256"] = (
                    hashlib.sha256(record.encode()).hexdigest() if record else None
                )
            module = importlib.import_module("monarch" if package == "torchmonarch" else package)
            facts["package"] = _origin(module, distribution)
            facts["import_status"] = "ok"
            if package == "torch":
                facts["runtime_version"] = str(module.__version__)
                facts["cuda_build"] = module.version.cuda
            else:
                native = importlib.import_module("monarch._rust_bindings")
                facts["native"] = _origin(native, distribution)
        except Exception as exc:
            facts["import_status"] = "unavailable"
            facts["error_type"] = type(exc).__name__
    try:
        native_rdma = importlib.import_module("monarch._rust_bindings.rdma")
        result["rdma"]["native_import"] = "ok"
        action = native_rdma._RdmaAction()
        if type(action).__module__ != "monarch._rust_bindings.rdma":
            raise TypeError("unexpected native action implementation")
        result["rdma"]["empty_action"] = "ok"
    except Exception as exc:
        result["rdma"]["error_type"] = type(exc).__name__
        if result["rdma"]["native_import"] == "not_run":
            result["rdma"]["native_import"] = "unavailable"
        else:
            result["rdma"]["empty_action"] = "unavailable"
    return result


def capability_blockers(snapshot: object, *, expected_version: str | None = None) -> list[str]:
    """Require the supported artifact's native load, without claiming hardware."""
    if not isinstance(snapshot, dict) or snapshot.get("schema") != 1:
        return ["Monarch capability snapshot is absent or has an unsupported schema"]
    blockers = []
    for package in ("torch", "torchmonarch"):
        facts = snapshot.get(package, {})
        if not isinstance(facts, dict) or facts.get("import_status") != "ok":
            blockers.append(f"{package} import is unavailable")
            continue
        origins = ("package", "native") if package == "torchmonarch" else ("package",)
        for name in origins:
            origin = facts.get(name, {})
            if not isinstance(origin, dict) or origin.get("distribution_owned") is not True:
                blockers.append(f"{package} {name} import is not owned by its distribution")
        if package == "torchmonarch" and expected_version is not None:
            if facts.get("distribution_version") != expected_version:
                blockers.append("torchmonarch capability version differs from surface snapshot")
    rdma = snapshot.get("rdma", {})
    if not isinstance(rdma, dict) or rdma.get("native_import") != "ok":
        blockers.append("native RDMA bindings are unavailable")
    if not isinstance(rdma, dict) or rdma.get("empty_action") != "ok":
        blockers.append("native RDMA action construction is unavailable")
    return blockers


@lru_cache(maxsize=1)
def _process_capabilities() -> dict[str, Any]:
    return collect_capabilities()


def cached_capabilities() -> dict[str, Any]:
    """Return isolated process-lifetime facts for repeated status consumers.

    Use collect_capabilities for a fresh admission or canary observation.
    """
    return copy.deepcopy(_process_capabilities())


def assert_supported_runtime() -> None:
    """Admit already imported Monarch bindings without importing Torch.

    Call after Monarch's bootstrap loaded its root extension, before GPU setup.
    Distribution ownership proves import placement, not payload byte integrity.
    """
    import sys

    from . import TORCHMONARCH_PIN

    try:
        distribution = importlib.metadata.distribution("torchmonarch")
        if distribution.version != TORCHMONARCH_PIN:
            raise RuntimeError("Monarch version does not match the supported pin")
        for name in ("monarch", "monarch._rust_bindings"):
            module = sys.modules.get(name)
            if module is None or not _origin(module, distribution)["distribution_owned"]:
                raise RuntimeError("Monarch imported bindings are not owned by the pinned distribution")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Monarch runtime identity is unavailable") from exc


@lru_cache(maxsize=1)
def _diagnostic_base() -> dict[str, Any]:
    from . import __version__
    from .runtime_provenance import cached_dgx_source_manifest_sha256

    result: dict[str, Any] = {"dgx_monarch": __version__}
    for package in ("torch", "torchmonarch"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "missing"
    try:
        result["source_manifest_sha256"] = cached_dgx_source_manifest_sha256()
    except (OSError, RuntimeError):
        result["source_manifest_sha256"] = None
    return result


def diagnostic_identity() -> dict[str, Any]:
    """Cached source/package facts with current policy identity; no GPU imports."""
    from .monarch_policy import snapshot

    return {**_diagnostic_base(), "transport_sha256": snapshot().get("transport_sha256")}
