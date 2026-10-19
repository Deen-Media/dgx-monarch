"""Source-bound verified updates and explicitly reviewed driver restoration."""
from __future__ import annotations

import asyncio
import grp
import os
import pwd
import tempfile
import tomllib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..cli.update_command import SystemUpdateOps
from ..cli.update_inspect import inspect_checkout
from ..cli.update_lock import UpdateLock
from ..cli.update_types import Certainty
from ..cli.update_worktree import remove_worktree
from ..config import load_cluster_config
from . import update_controller
from .contracts import StudioError, identifier
from .driver_service import DriverService
from .management_config import absolute
from .store import private_directory


async def blocking(function: Callable[..., Any], *args: Any) -> Any:
    """Do not release transaction locks while a cancelled thread still mutates."""
    return await settle(asyncio.to_thread(function, *args))


async def settle(awaitable: Any) -> Any:
    task = asyncio.create_task(awaitable)
    cancellation = None
    while True:
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            if task.cancelled():
                raise
            cancellation = exc
            continue
        except BaseException:
            if cancellation is not None:
                raise cancellation from None
            raise
        if cancellation is not None:
            raise cancellation
        return result


def checkout_identity(repo: Path) -> dict[str, Any]:
    info = repo.stat()
    unsafe_group = False
    if info.st_mode & 0o020:
        try:
            owner = pwd.getpwuid(os.getuid()).pw_name
            group = grp.getgrgid(info.st_gid)
            members = {entry.pw_name for entry in pwd.getpwall() if entry.pw_gid == info.st_gid} | set(group.gr_mem)
            unsafe_group = bool(members - {owner})
        except KeyError:
            unsafe_group = True
    if not repo.is_dir() or info.st_uid != os.getuid() or info.st_mode & 0o002 or unsafe_group:
        raise StudioError("update_repository", "Select an owner-controlled source checkout that other users cannot write.", 409)
    with (repo / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    if project.get("project", {}).get("name") != "dgx-monarch" or not (repo / "src/dgx_monarch/__init__.py").is_file():
        raise StudioError("update_repository", "Select an existing dgx-monarch source checkout.", 409)
    return {"path": str(repo), "device": info.st_dev, "inode": info.st_ino}


class ReviewedUpdate:
    """Use the existing transaction with a fixed target and exact confirmation."""
    def __init__(self, ops: Any, reviewed: dict[str, Any]):
        self.ops, self.reviewed = ops, reviewed

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ops, name)

    def resolve_target(self, target_ref: str) -> str:
        if target_ref != self.reviewed["target_sha"]:
            raise StudioError("update_changed", "Update target differs from the reviewed commit.", 409)
        if self.ops.resolve_target(target_ref) != target_ref:
            raise StudioError("update_changed", "Backend target resolution differs from the reviewed commit.", 409)
        return target_ref

    def confirm(self, staged: Any) -> bool:
        actual = asdict(staged)
        return all(actual.get(key) == value for key, value in self.reviewed.items())


class Updates:
    def __init__(self, config: Path | None, driver: DriverService | None, origin: str, directory: Path):
        self.config, self.driver, self.origin, self.directory = config, driver, origin, directory

    def operations(self, parameters: dict[str, Any]) -> Any:
        if self.config is None:
            raise StudioError("update_config", "Select an existing cluster configuration.")
        repo = absolute(parameters["repo"]).resolve()
        checkout_identity(repo)
        return SystemUpdateOps(repo, load_cluster_config(self.config), driver_host=urlsplit(self.origin).netloc,
                               privileged_process_inspection=parameters["privileged_process_inspection"])

    def preview(self, parameters: dict[str, Any]) -> dict[str, Any]:
        if self.config is None:
            raise StudioError("update_config", "Select an existing cluster configuration.")
        config_sha256 = update_controller.config_digest(self.config)
        ops = self.operations(parameters)
        driver = self.driver.inspect() if self.driver else None
        controller = update_controller.binding(driver)
        if driver and driver["state"] == "active" and not parameters["stop_driver"]:
            raise StudioError("update_driver", "Explicit driver-stop authority is required for the running driver.", 409)
        if parameters["restore_driver"] and (not driver or driver["state"] != "active" or not parameters["stop_driver"]):
            raise StudioError("update_driver", "Restoration requires an explicitly reviewed stop of the running owner driver.", 409)
        with UpdateLock(ops.repo):
            target = ops.resolve_target(parameters["target_ref"])
            prior = ops.repo_snapshot(target)
            if not prior.clean or not prior.fast_forward:
                raise StudioError("update_repository", "The reviewed update requires a clean fast-forward checkout.", 409)
            original, _ = inspect_checkout(ops.repo, prior.current_sha)
            # Inspect only a private source checkout. Package staging stays in
            # the existing transaction, after its mandatory idle activity guard.
            root = private_directory(self.directory / "update-plans")
            checkout = Path(tempfile.mkdtemp(prefix="update-", dir=root))
            try:
                added = ops._git("worktree", "add", "--detach", str(checkout), target, timeout=120)
                if added.returncode:
                    raise StudioError("update_checkout", "The exact target source could not be inspected.", 409)
                metadata, _ = inspect_checkout(checkout, target)
            finally:
                cleaned = remove_worktree(checkout, root, ops._git)
                if cleaned.certainty != Certainty.SUCCEEDED:
                    raise StudioError("update_plan_cleanup", "Private source review cleanup is unconfirmed; inspect retained update-plans.", 409)
        if update_controller.config_digest(self.config) != config_sha256:
            raise StudioError("update_changed", "Configuration changed during review.", 409)
        return {"token": identifier(), "prior_commit": prior.current_sha, "prior_release": asdict(original),
                "checkout": checkout_identity(ops.repo), "release": asdict(metadata),
                "driver_identity": driver, "controller": controller, "config_sha256": config_sha256,
                "review_scope": "Exact source and dependency selection. Existing backend stages and verifies the same selection after idle guards, before Worker mutation.",
                "restart_policy": "Restore only after the verified update succeeds and the stopped driver identity still matches."}

    async def execute(self, parameters: dict[str, Any], reviewed: dict[str, Any],
                      checkpoint: Callable[[dict[str, Any]], None]) -> tuple[str, dict[str, Any]]:
        ops = self.operations(parameters)
        expected = reviewed["release"]
        receipt: dict[str, Any] = {"source": "verified_update", "token": reviewed["token"], "release": expected}
        with UpdateLock(ops.repo) as lock:
            if checkout_identity(ops.repo) != reviewed["checkout"]:
                raise StudioError("update_changed", "The selected source checkout identity changed after review.", 409)
            prior = await blocking(ops.repo_snapshot, expected["target_sha"])
            if not prior.clean or not prior.fast_forward or prior.current_sha != reviewed["prior_commit"]:
                raise StudioError("update_changed", "Driver checkout changed after the source review.", 409)
            original, _ = await blocking(inspect_checkout, ops.repo, prior.current_sha)
            if asdict(original) != reviewed["prior_release"]:
                raise StudioError("update_changed", "The selected prior source content changed after review.", 409)
            current_driver = await blocking(self.driver.inspect) if self.driver else None
            if current_driver != reviewed["driver_identity"] or update_controller.binding(current_driver) != reviewed["controller"]:
                raise StudioError("update_changed", "Driver or controller identity changed after review.", 409)
            if self.config is None:
                raise StudioError("update_config", "The reviewed configuration is unavailable.", 409)
            prepared = await blocking(update_controller.prepare, self.directory, self.config, ops.repo, reviewed,
                                      urlsplit(self.origin).netloc, parameters["privileged_process_inspection"], lock.fileno())
            receipt["controller"] = prepared
            checkpoint({**receipt, "phase": "controller_prepared"})
            stopped = None
            driver_identity = reviewed["driver_identity"]
            if driver_identity and driver_identity["state"] == "active":
                if self.driver is None or parameters["stop_driver"] is not True:
                    raise StudioError("update_driver", "The reviewed owner driver cannot be stopped.", 409)
                checkpoint({**receipt, "phase": "driver_stop_requested", "driver_identity": driver_identity})
                state, stop_receipt = await settle(self.driver.perform("stop", driver_identity))
                receipt["driver_stop"] = stop_receipt
                checkpoint(receipt)
                if state != "succeeded":
                    return "unknown", {**receipt, "recovery": "Driver stop is uncertain; update and restoration were not attempted."}
                stopped = stop_receipt["observed_identity"]
            checkpoint({**receipt, "phase": "verified_update_requested"})

            result = await blocking(update_controller.run, prepared)
            receipt.update(update=dict(result.receipt), result_code=result.code)
            checkpoint(receipt)
            if not result.succeeded:
                return "unknown", {**receipt, "recovery": "Inspect the backend receipt; driver restoration was suppressed after an unsuccessful update."}
            if stopped and parameters["restore_driver"]:
                if self.driver is None:
                    return "unknown", {**receipt, "recovery": "Driver binding is unavailable; restoration was not attempted."}
                current = await blocking(self.driver.inspect)
                if current["config_digest"] != stopped["config_digest"] or current["state"] != "inactive":
                    return "unknown", {**receipt, "recovery": "Stopped driver identity changed; restoration was not attempted."}
                environment = update_controller.binding(current)
                if any(environment[key] != value for key, value in reviewed["controller"].items() if key != "source_manifest"):
                    return "unknown", {**receipt, "recovery": "Driver Python environment changed; restoration was not attempted."}
                checkpoint({**receipt, "phase": "driver_restore_requested", "driver_identity": current})
                state, restored = await settle(self.driver.perform("start", current))
                receipt["driver_restore"] = restored
                checkpoint(receipt)
                if state != "succeeded":
                    return "unknown", receipt
            await blocking(update_controller.cleanup, prepared)
            return "succeeded", {**receipt, "controller_cleaned": True}
