"""Frozen, typed verified-update controller, isolated from the serving process."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..cli.update_bootstrap import _LOADER, _regular_bytes, _snapshot, _write_private
from ..cli.update_types import UpdateResult
from ..runtime_provenance import dgx_source_manifest_sha256
from .contracts import StudioError

# Reuse the existing source/hash verifier with one fixed, typed entrypoint.
LOADER = _LOADER.replace("from dgx_monarch.cli.update_bootstrap import _child",
                         "from dgx_monarch.studio.update_controller import child as _child")
SOURCE = Path(__file__).resolve().parents[1]


def config_digest(path: Path) -> str:
    return hashlib.sha256(_regular_bytes(path)).hexdigest()


def binding(driver: dict[str, Any] | None) -> dict[str, Any]:
    executable = Path(driver["config"]["executable"] if driver else sys.executable)
    info = executable.stat()
    if not executable.is_absolute() or not stat.S_ISREG(info.st_mode) or info.st_uid not in {0, os.getuid()} or info.st_mode & 0o022:
        raise StudioError("update_interpreter", "Update controller requires the reviewed owner-controlled Python interpreter.", 409)
    environment = []
    for path in (executable.parent / "pyvenv.cfg", executable.parent.parent / "pyvenv.cfg"):
        entry: dict[str, Any] = {"path": str(path), "sha256": None}
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid not in {0, os.getuid()}
                    or metadata.st_mode & 0o022 or metadata.st_size > 65536):
                raise StudioError("update_interpreter", "The Python environment configuration is not owner-controlled.", 409)
            entry["sha256"] = config_digest(path)
        environment.append(entry)
    return {"executable": str(executable), "resolved": str(executable.resolve()),
            "identity": [info.st_dev, info.st_ino, info.st_mtime_ns],
            "environment": environment, "source_manifest": dgx_source_manifest_sha256(SOURCE)}


def prepare(directory: Path, config: Path, repo: Path, reviewed: dict[str, Any],
            driver_host: str, privileged: bool, lock_fd: int) -> dict[str, Any]:
    if config_digest(config) != reviewed["config_sha256"]:
        raise StudioError("update_changed", "Configuration changed after review.", 409)
    request = {"config": str(config.resolve()), "config_sha256": reviewed["config_sha256"],
               "repo": str(repo), "reviewed": reviewed, "driver_host": driver_host,
               "privileged_process_inspection": privileged, "lock_fd": lock_fd}
    root, digest = _snapshot(SOURCE, directory / "update-controllers", request)
    manifest = json.loads(_regular_bytes(root / "manifest.json"))
    if manifest["source_manifest"] != reviewed["controller"]["source_manifest"]:
        raise StudioError("update_changed", "Controller source changed after review; private controller retained.", 409)
    info = root.stat()
    return {"path": str(root), "manifest_sha256": digest, "identity": [info.st_dev, info.st_ino],
            "executable": reviewed["controller"]["executable"], "lock_fd": lock_fd}


def run(prepared: dict[str, Any]) -> UpdateResult:
    root = Path(prepared["path"])
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    process = subprocess.Popen([prepared["executable"], "-I", "-B", "-c", LOADER,
                                str(root), prepared["manifest_sha256"]],
                               cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, pass_fds=(prepared["lock_fd"],), start_new_session=True)
    try:
        _write_private(root / "process.json", json.dumps({"pid": process.pid, "executable": prepared["executable"]}).encode())
    finally:
        # Never abandon a live mutator, even if receipt publication fails.
        code = process.wait()
    payload = json.loads(_regular_bytes(root / "result.json"))
    result = UpdateResult(**payload)
    if code != result.exit_code:
        raise StudioError("update_controller", "Controller exit and retained settlement differ; reconcile its receipt.", 409)
    return result


def cleanup(prepared: dict[str, Any]) -> None:
    """Only called after successful settlement is durably retained by Studio."""
    root = Path(prepared["path"])
    info = root.lstat()
    if root.is_symlink() or [info.st_dev, info.st_ino] != prepared["identity"]:
        raise StudioError("update_controller", "Private controller identity changed; cleanup was refused.", 409)
    shutil.rmtree(root)


def child(root: Path, manifest: dict[str, Any]) -> int:
    from ..cli.update_command import SystemUpdateOps
    from ..cli.update_inspect import inspect_checkout
    from ..cli.update_receipt import interrupted_receipt
    from ..cli.update_transaction import UpdateRequest, execute_verified_update
    from ..config import load_cluster_config
    from .management_update import ReviewedUpdate, checkout_identity

    def interrupt(_signum: int, _frame: object) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    request, reviewed = manifest["request"], manifest["request"]["reviewed"]
    if binding(None) != reviewed["controller"]:
        raise StudioError("update_changed", "Isolated controller source or interpreter differs from review.", 409)
    lock = os.fstat(request["lock_fd"])
    if not stat.S_ISREG(lock.st_mode) or lock.st_uid != os.getuid() or lock.st_mode & 0o077:
        raise StudioError("update_controller", "The inherited update lock is unavailable.", 409)
    config_path = Path(request["config"])
    expected_config = request["config_sha256"]
    if hashlib.sha256(_regular_bytes(config_path)).hexdigest() != expected_config:
        raise StudioError("update_changed", "Configuration changed before isolated update.", 409)
    config = load_cluster_config(config_path)
    if hashlib.sha256(_regular_bytes(config_path)).hexdigest() != expected_config:
        raise StudioError("update_changed", "Configuration changed while loading isolated update.", 409)
    repo = Path(request["repo"])
    if checkout_identity(repo) != reviewed["checkout"]:
        raise StudioError("update_changed", "Selected checkout changed before isolated update.", 409)
    ops = SystemUpdateOps(repo, config, driver_host=request["driver_host"],
                          privileged_process_inspection=request["privileged_process_inspection"])
    prior = ops.repo_snapshot(reviewed["release"]["target_sha"])
    metadata, _ = inspect_checkout(repo, prior.current_sha)
    if not prior.clean or not prior.fast_forward or asdict(metadata) != reviewed["prior_release"]:
        raise StudioError("update_changed", "Selected source changed before isolated update.", 409)
    try:
        result = execute_verified_update(UpdateRequest(reviewed["release"]["target_sha"]), ReviewedUpdate(ops, reviewed["release"]))
    except BaseException as exc:
        receipt = interrupted_receipt(exc)
        if receipt:
            _write_private(root / "interrupted.json", json.dumps(dict(receipt)).encode())
        raise
    _write_private(root / "result.json", json.dumps(asdict(result)).encode())
    return result.exit_code
