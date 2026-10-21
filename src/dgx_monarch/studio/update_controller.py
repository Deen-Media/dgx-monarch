"""Frozen, typed update controller, isolated from the serving process."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..cli import probe_certainty
from ..cli.receipt_sanitize import sanitize_note
from ..cli.update_bootstrap import _LOADER, _regular_bytes, _snapshot, _write_private
from ..cli.update_types import UpdateResult
from ..runtime_provenance import dgx_source_manifest_sha256
from .contracts import StudioError

# Reuse the existing source/hash verifier with one fixed, typed entrypoint.
LOADER = _LOADER.replace("from dgx_monarch.cli.update_bootstrap import _child",
                         "from dgx_monarch.studio.update_controller import child as _child")
SOURCE = Path(__file__).resolve().parents[1]

# The legacy backend returns these exits only before it fetches, merges,
# installs or restarts anything: a refusal, or a probe that never answered.
# Exit 1 covers failures both before and after the checkout moves.
LEGACY_NO_CHANGE_EXITS = frozenset({2, probe_certainty.UNKNOWN_EXIT})
# The isolated controller exits with this code when its read-only admission
# refuses, before it runs any backend.
CONTROLLER_REFUSED_EXIT = 2
# Verified-transaction steps that run before any Worker service changes.
PRE_MUTATION_STEPS = frozenset({"target_resolution", "repository_guard", "activity_guard", "release_staging",
                                "confirmation", "pre_stop_recheck", "stage_cleanup"})
STDERR_NAME = "controller-stderr.log"
STDERR_TAIL_BYTES = 16384
STDERR_LINES = 6
INTERRUPTED_BYTES = 65536


def config_digest(path: Path) -> str:
    return hashlib.sha256(_regular_bytes(path)).hexdigest()


def binding(driver: dict[str, Any] | None) -> dict[str, Any]:
    from .management_update import writers_trusted

    executable = Path(driver["config"]["executable"] if driver else sys.executable)
    info = executable.stat()
    if (not executable.is_absolute() or not stat.S_ISREG(info.st_mode) or info.st_uid not in {0, os.getuid()}
            or not writers_trusted(info)):
        raise StudioError("update_interpreter", "Update controller requires the reviewed owner-controlled Python interpreter.", 409)
    environment = []
    for path in (executable.parent / "pyvenv.cfg", executable.parent.parent / "pyvenv.cfg"):
        entry: dict[str, Any] = {"path": str(path), "sha256": None}
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            # The same rule as the selected checkout: a default-umask venv
            # (0664 under a group that holds only its owner) stays trusted.
            if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid not in {0, os.getuid()}
                    or not writers_trusted(metadata) or metadata.st_size > 65536):
                raise StudioError("update_interpreter", "The Python environment configuration is not owner-controlled.", 409)
            entry["sha256"] = config_digest(path)
        environment.append(entry)
    return {"executable": str(executable), "resolved": str(executable.resolve()),
            "identity": [info.st_dev, info.st_ino, info.st_mtime_ns],
            "environment": environment, "source_manifest": dgx_source_manifest_sha256(SOURCE)}


def prepare(directory: Path, config: Path | None, repo: Path, reviewed: dict[str, Any],
            driver_host: str | None, privileged: bool, lock_fd: int) -> dict[str, Any]:
    if config is None:
        raise StudioError("update_config", "Update requires the reviewed cluster configuration in every mode.", 409)
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
    # Refusals print their reason to stderr; keep it in the private root.
    errors = os.open(root / STDERR_NAME, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        process = subprocess.Popen([prepared["executable"], "-I", "-B", "-c", LOADER,
                                    str(root), prepared["manifest_sha256"]],
                                   cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                   stderr=errors, pass_fds=(prepared["lock_fd"],), start_new_session=True)
    finally:
        os.close(errors)
    try:
        _write_private(root / "process.json", json.dumps({"pid": process.pid, "executable": prepared["executable"]}).encode())
    finally:
        # Never abandon a live mutator, even if receipt publication fails.
        code = process.wait()
    return settled_result(root, code)


def settled_result(root: Path, code: int) -> UpdateResult:
    """Read the child's settlement; a missing or inconsistent one stays an unknown outcome."""
    unsettled: dict[str, Any] = {"source": "isolated_controller", "process_exit": code}
    try:
        interrupted = _regular_bytes(root / "interrupted.json")
        if len(interrupted) <= INTERRUPTED_BYTES:
            unsettled["interrupted"] = json.loads(interrupted)
    except (OSError, ValueError):
        pass
    try:
        result = UpdateResult(**json.loads(_regular_bytes(root / "result.json")))
        if (type(result.exit_code) is not int or not isinstance(result.code, str) or not isinstance(result.message, str)
                or not isinstance(result.receipt, dict)):
            raise TypeError("settlement fields have the wrong types")
    except (OSError, ValueError, TypeError):
        return UpdateResult(code or 1, "controller_unsettled",
                            "The isolated controller ended without a readable settlement.", unsettled)
    if code != result.exit_code:
        return UpdateResult(code or 1, "controller_mismatch",
                            "Controller exit and retained settlement differ; reconcile its receipt.",
                            {**unsettled, "settlement": asdict(result)})
    return result


def stderr_tail(prepared: dict[str, Any]) -> list[str]:
    """A bounded, sanitized copy of the last lines the child wrote to stderr."""
    if not isinstance(prepared.get("path"), str):
        return []
    try:
        descriptor = os.open(Path(prepared["path"]) / STDERR_NAME, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return []
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            return []
        offset = max(0, info.st_size - STDERR_TAIL_BYTES)
        data = os.pread(descriptor, STDERR_TAIL_BYTES, offset)
    finally:
        os.close(descriptor)
    lines = data.decode("utf-8", "replace").splitlines()
    if offset:
        lines = lines[1:]  # the first line may start mid-line
    return [sanitize_note(line) for line in lines if line.strip()][-STDERR_LINES:]


def settlement(mode: str, result: UpdateResult) -> str:
    """Classify a controller result as succeeded, refused, settled_after_change or unknown.

    ``refused`` means the backend stopped before it changed the checkout, the
    packages or any Worker. ``settled_after_change`` means the verified
    transaction changed Worker services and then settled a definite failure by
    restoring the prior release. Anything else stays unknown.
    """
    receipt = result.receipt if isinstance(result.receipt, Mapping) else {}
    if result.code == "controller_refused":
        return ("refused" if result.exit_code == CONTROLLER_REFUSED_EXIT and receipt.get("mutation_started") is False
                else "unknown")
    if result.code in {"controller_unsettled", "controller_mismatch"}:
        return "unknown"
    if mode == "legacy":
        if receipt.get("mode") != "legacy":
            return "unknown"
        if result.exit_code == 0:
            return "succeeded"
        return "refused" if result.exit_code in LEGACY_NO_CHANGE_EXITS else "unknown"
    status, steps = receipt.get("status"), receipt.get("steps")
    if not isinstance(steps, list) or not all(isinstance(step, Mapping) for step in steps):
        return "unknown"
    names = [step.get("name") for step in steps]
    if result.exit_code == 0:
        return "succeeded" if status == "succeeded" else "unknown"
    if status not in {"failed", "planned"}:
        return "unknown"
    if all(name in PRE_MUTATION_STEPS for name in names):
        cleanup = [step.get("status") for step in steps if step.get("name") == "stage_cleanup"]
        if "release_staging" in names and not cleanup:
            return "unknown"
        return "refused" if all(value == "succeeded" for value in cleanup) else "unknown"
    # A failed terminal status after Worker mutation means the backend settled it.
    return "settled_after_change" if status == "failed" and "worker_stop" in names else "unknown"


def cleanup(prepared: dict[str, Any]) -> None:
    """Only called after a definite settlement is durably retained by Studio."""
    root = Path(prepared["path"])
    info = root.lstat()
    if root.is_symlink() or [info.st_dev, info.st_ino] != prepared["identity"]:
        raise StudioError("update_controller", "Private controller identity changed; cleanup was refused.", 409)
    shutil.rmtree(root)


def _admit(request: dict[str, Any], reviewed: dict[str, Any]) -> tuple[Path, Any, Any]:
    """Recheck every reviewed input before the backend runs; nothing here writes."""
    from ..cli.update_command import SystemUpdateOps
    from ..cli.update_inspect import inspect_checkout
    from ..config import load_cluster_config
    from .management_repository import RepositoryOps
    from .management_update import checkout_identity, driver_checkout

    if binding(None) != reviewed["controller"]:
        raise StudioError("update_changed", "Isolated controller source or interpreter differs from review.", 409)
    lock = os.fstat(request["lock_fd"])
    if not stat.S_ISREG(lock.st_mode) or lock.st_uid != os.getuid() or lock.st_mode & 0o077:
        raise StudioError("update_controller", "The inherited update lock is unavailable.", 409)
    if request["config"] is None or request["config_sha256"] is None:
        raise StudioError("update_config", "Update requires the reviewed cluster configuration in every mode.", 409)
    config_path = Path(request["config"])
    expected_config = request["config_sha256"]
    if config_digest(config_path) != expected_config:
        raise StudioError("update_changed", "Configuration changed before isolated update.", 409)
    config = load_cluster_config(config_path)
    if config_digest(config_path) != expected_config:
        raise StudioError("update_changed", "Configuration changed while loading isolated update.", 409)
    repo = Path(request["repo"])
    if checkout_identity(repo) != reviewed["checkout"]:
        raise StudioError("update_changed", "Selected checkout changed before isolated update.", 409)
    if driver_checkout(reviewed["driver_identity"], repo) != reviewed["driver_checkout"]:
        raise StudioError("update_changed", "The driver's checkout link changed before isolated update.", 409)
    # Studio stopped and rechecked the bound driver; an explicit host would
    # make the verified activity guard refuse a stopped driver as unknown.
    ops = (RepositoryOps(repo) if reviewed["mode"] == "legacy" else
           SystemUpdateOps(repo, config, driver_host=None,
                           privileged_process_inspection=request["privileged_process_inspection"]))
    prior = ops.repo_snapshot(reviewed["release"]["target_sha"])
    metadata, _ = inspect_checkout(repo, prior.current_sha)
    if not prior.clean or not prior.fast_forward or asdict(metadata) != reviewed["prior_release"]:
        raise StudioError("update_changed", "Selected source changed before isolated update.", 409)
    return repo, config, ops


def _legacy(request: dict[str, Any], reviewed: dict[str, Any], repo: Path, config: Any) -> UpdateResult:
    from types import SimpleNamespace

    from ..cli import legacy_update
    from ..cli.main import _driver_probe

    code = legacy_update._run_locked(SimpleNamespace(host=request["driver_host"]), repo=repo,
        load_config=lambda args: config, driver_probe=_driver_probe,
        expected_candidate=tuple(reviewed["legacy_candidate"]),
        expected_source_manifest=reviewed["release"]["source_manifest"])
    if code == 0:
        outcome, message = "legacy_updated", "Legacy update completed."
    elif code in LEGACY_NO_CHANGE_EXITS:
        outcome, message = "legacy_refused", "Legacy update refused before it fetched, installed or restarted anything."
    else:
        outcome, message = "legacy_incomplete", "Legacy update did not finish; the checkout, packages or Workers may have changed."
    return UpdateResult(code, outcome, message, {"mode": "legacy", "exit_code": code,
                        "reviewed_target": reviewed["release"]["target_sha"]})


def child(root: Path, manifest: dict[str, Any]) -> int:
    from ..cli.update_receipt import interrupted_receipt
    from ..cli.update_transaction import UpdateRequest, execute_verified_update
    from .management_update import ReviewedUpdate

    def interrupt(_signum: int, _frame: object) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    request, reviewed = manifest["request"], manifest["request"]["reviewed"]
    try:
        repo, config, ops = _admit(request, reviewed)
    except (KeyError, TypeError, OSError, RuntimeError, ValueError) as exc:
        # Admission only reads, so its refusal is a definite no-change result.
        code, message = ((exc.code, str(exc)) if isinstance(exc, StudioError) else
                         ("controller_admission", "The isolated controller could not read its reviewed inputs."))
        refused = UpdateResult(CONTROLLER_REFUSED_EXIT, "controller_refused", message, {"source": "isolated_controller", "refusal": code,
                               "mode": reviewed.get("mode"), "mutation_started": False})
        _write_private(root / "result.json", json.dumps(asdict(refused)).encode())
        return refused.exit_code
    try:
        if reviewed["mode"] == "legacy":
            result = _legacy(request, reviewed, repo, config)
        else:
            result = execute_verified_update(UpdateRequest(reviewed["release"]["target_sha"]), ReviewedUpdate(ops, reviewed["release"]))
        if reviewed.get("receipt_path"):
            from ..cli.operator_receipt import write_receipt
            write_receipt(result.receipt, reviewed["receipt_path"])
    except BaseException as exc:
        receipt = interrupted_receipt(exc)
        if receipt:
            _write_private(root / "interrupted.json", json.dumps(dict(receipt)).encode())
        raise
    _write_private(root / "result.json", json.dumps(asdict(result)).encode())
    return result.exit_code
