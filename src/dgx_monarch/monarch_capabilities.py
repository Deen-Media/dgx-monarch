"""Path-redacted Monarch artifact and native-load facts, without hardware probes.

Importing this module is stdlib-only. Explicit collection imports Torch and
Monarch, but never creates an actor, RDMA manager, buffer, or device context.
A loadable native binding is not evidence of a usable GPU or fabric. Metadata
ownership does not verify installed payload bytes.
"""
from __future__ import annotations

import copy
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


def assert_supported_runtime() -> str:
    """Admit already imported Monarch bindings without importing Torch; return their version.

    Call after Monarch's bootstrap loaded its root extension, before GPU setup.
    The version is the one this process loaded (``loaded_version``), so a pin
    installed under a running process does not admit it: only a restart does.
    Distribution ownership proves import placement, not payload byte integrity.
    """
    from . import TORCHMONARCH_PIN

    version, owned, cause, replaced = _loaded_identity()
    if replaced:
        raise RuntimeError(_REPLACED)
    if version is not None and version != TORCHMONARCH_PIN:
        raise RuntimeError(
            f"Monarch version does not match the supported pin: this process loaded "
            f"torchmonarch {version} and this dgx-monarch pins {TORCHMONARCH_PIN}. If "
            "dgx-monarch was updated since this process started, restart it before anything "
            "else; otherwise install the pin if it is missing, then restart the process, since "
            f"an install does not change what a running process loaded: {_RESTART}")
    return _identity_or_refuse(version, owned, cause, "pinned")


def loaded_version() -> str:
    """The torchmonarch version this process loaded.

    That is the installed distribution's version, read once per process: at
    the first call made after ``monarch`` is imported, which is the admission
    right after the import (``prepare_driver``, the Worker loop, a spawned
    proc's bootstrap), or ``note_loaded_version`` when ``prepare_driver``
    refuses before its admission. Later calls return that reading, and its
    ownership verdict for the loaded ``monarch`` package and native bindings,
    whatever the disk holds by then: an install under a running process
    changes the files, not the code it runs. Something else may import
    ``monarch`` first and torchmonarch be reinstalled before that reading, so
    the reading also checks that the native binding the process mapped is
    still the file on disk (``_binding_replaced``) and refuses when it is not.
    A read that failed is kept too, since a later read of the disk says
    nothing about what was loaded. A Worker loop records this version for the
    driver's attach gate, which compares it with the same reading in the
    driver (``worker_proc_id.read_values``).
    """
    version, owned, cause, replaced = _loaded_identity()
    if replaced:
        raise RuntimeError(_REPLACED)
    return _identity_or_refuse(version, owned, cause, "installed")


def note_loaded_version() -> None:
    """Keep the loaded identity now if ``monarch`` is imported and none is kept; never raises."""
    _loaded_identity()


_RESTART = ("restart ComfyUI on the driver, or run `dgxm restart` for the Workers "
            "(docs/TROUBLESHOOTING.md #8)")
_REPLACED = ("Monarch's native binding in this process is no longer the file on disk: "
             "torchmonarch was reinstalled after this process loaded it, so the file it runs "
             f"is no longer the one installed. Restart the process: {_RESTART}")
_KEPT = ("; this process keeps that reading, so restart it once torchmonarch is installed "
         f"as the pin says: {_RESTART}")
# Where the kernel lists this process's mapped files (Linux).
_MAPS = "/proc/self/maps"

# The torchmonarch identity this process loaded: its version (None when it
# could not be read), whether that distribution owns the loaded modules, the
# read's error, and whether the mapped native binding is no longer the file
# on disk. Set once, by the first reading taken after ``monarch`` is
# imported, and never read again from disk.
_LOADED: tuple[str | None, bool, BaseException | None, bool] | None = None


def _loaded_identity() -> tuple[str | None, bool, BaseException | None, bool]:
    import sys

    global _LOADED
    if _LOADED is not None:
        return _LOADED
    version: str | None = None
    try:
        distribution = importlib.metadata.distribution("torchmonarch")
        version = str(distribution.version)
        owned = all(
            sys.modules.get(name) is not None
            and bool(_origin(sys.modules[name], distribution)["distribution_owned"])
            for name in ("monarch", "monarch._rust_bindings"))
        replaced = owned and _binding_replaced(sys.modules["monarch._rust_bindings"]) is True
        reading: tuple[str | None, bool, BaseException | None, bool] = (version, owned, None, replaced)
    except Exception as exc:
        reading = (version, False, exc, False)
    # A reading taken before the import describes no loaded code, so only a
    # reading taken after it is kept, refusing or not.
    if "monarch" in sys.modules:
        _LOADED = reading
    return reading


def _binding_replaced(module: Any) -> bool | None:
    """Whether the native binding this process mapped is no longer the file on disk.

    The loaded module's file is the one torchmonarch's RECORD names, which
    the ownership check has just shown. Linux lists each mapped file in
    ``/proc/self/maps`` with its device and inode, under its current path,
    and marks one removed since it was mapped. A reinstall writes a new
    file there: pip first moves the old one aside or unlinks it, and removes
    it when the install ends. So a mapping of that path on the same device
    with another inode, or, with no live mapping of that path, a removed
    file of the binding's name, means the binding was replaced: True. A live
    mapping of that path with its inode is False. With neither, as when
    the list cannot be read, the result is None. On False and None the
    caller keeps the installed version it read, as before this check.
    """
    import os

    filename = getattr(module, "__file__", None)
    if not filename:
        return None
    try:
        located = Path(filename).resolve()
        status = located.stat()
        with open(_MAPS, encoding="utf-8", errors="replace") as maps:
            lines = maps.read().splitlines()
    except OSError:
        return None
    live: list[tuple[int, int]] = []
    removed = False
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) < 6:
            continue
        mapped = fields[5]
        gone = mapped.endswith(" (deleted)")
        mapped = mapped.removesuffix(" (deleted)")
        if os.path.basename(mapped) != located.name:
            continue
        if gone:
            removed = True
        elif mapped == str(located):
            try:
                major, minor = (int(part, 16) for part in fields[3].split(":"))
                live.append((os.makedev(major, minor), int(fields[4])))
            except ValueError:
                return None
    if live:
        return any(device == status.st_dev and inode != status.st_ino for device, inode in live)
    return True if removed else None


def _identity_or_refuse(version: str | None, owned: bool, cause: BaseException | None,
                        which: str) -> str:
    if version is None or cause is not None:
        raise RuntimeError(f"Monarch runtime identity is unavailable{_KEPT}") from cause
    if not owned:
        raise RuntimeError(f"Monarch imported bindings are not owned by the {which} distribution{_KEPT}")
    return version


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
