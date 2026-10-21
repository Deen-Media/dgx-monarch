"""Source-bound Studio updates and explicitly reviewed driver restoration."""
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

from ..cli.receipt_sanitize import sanitize_note
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
from .management_repository import RepositoryOps
from .store import private_directory

REVIEW_SCOPE = {
    "verified": ("Exact source and dependency selection. Studio stops the bound driver service if it runs and checks "
                 "it again. The verified transaction then runs its repository, activity and dependency guards, "
                 "stages the release on every host, stops, activates and starts the Workers after exact confirmation, "
                 "runs Doctor, the cluster smoke and the exact readback, and restores the prior release after a "
                 "definite failure."),
    "legacy": ("Exact upstream candidate selection, after the release-slot check of every configured Worker host. "
               "Studio stops the bound driver service if it runs and checks it again. The original in-place update "
               "then runs its driver probe and release-slot check, package contract and dependency admission, "
               "fast-forward merge, driver pin and editable install, Worker pin, Worker restart and Doctor. It stages "
               "no release and has no rollback."),
}
DRIVER_SERVICE_REQUIRED = ("Update requires the ComfyUI driver service selected with --driver-service, which binds "
                           "the driver's interpreter and checkout.")
RESTART_POLICY = ("Restore the stopped driver only after the update succeeds or the backend refuses before it changes "
                  "anything, and only while the stopped driver identity still matches.")


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


def writers_trusted(info: os.stat_result) -> bool:
    """The checkout trust rule: no world write, and group write only through a group that holds no other account."""
    if info.st_mode & 0o002:
        return False
    if not info.st_mode & 0o020:
        return True
    try:
        owner = pwd.getpwuid(os.getuid()).pw_name
        group = grp.getgrgid(info.st_gid)
        members = {entry.pw_name for entry in pwd.getpwall() if entry.pw_gid == info.st_gid} | set(group.gr_mem)
    except KeyError:
        return False
    return not members - {owner}


def checkout_identity(repo: Path) -> dict[str, Any]:
    info = repo.stat()
    if not repo.is_dir() or info.st_uid != os.getuid() or not writers_trusted(info):
        raise StudioError("update_repository", "Select an owner-controlled source checkout that other users cannot write.", 409)
    with (repo / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)
    if project.get("project", {}).get("name") != "dgx-monarch" or not (repo / "src/dgx_monarch/__init__.py").is_file():
        raise StudioError("update_repository", "Select an existing dgx-monarch source checkout.", 409)
    return {"path": str(repo), "device": info.st_dev, "inode": info.st_ino}


def driver_checkout(driver: dict[str, Any] | None, repo: Path) -> dict[str, str]:
    """Bind the selection to the checkout the driver imports: custom_nodes/dgx-monarch beside its reviewed main.py."""
    if driver is None:
        raise StudioError("update_driver_service", DRIVER_SERVICE_REQUIRED, 409)
    mains = [Path(record["path"]) for record in driver["config"]["source_files"] if Path(record["path"]).name == "main.py"]
    if len(mains) != 1:
        raise StudioError("update_repository", "The driver's reviewed ComfyUI entrypoint could not be identified.", 409)
    link = mains[0].parent / "custom_nodes" / "dgx-monarch"
    try:
        resolved = link.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise StudioError("update_repository", "The driver's ComfyUI has no custom_nodes/dgx-monarch checkout to update.", 409) from exc
    if resolved != repo:
        raise StudioError("update_repository", "Select the checkout the bound ComfyUI driver imports, the target of its "
                          "custom_nodes/dgx-monarch.", 409)
    return {"link": str(link), "resolved": str(resolved)}


def legacy_driver_host(origin: str) -> str | None:
    """The legacy probe dials plain HTTP, so only an http origin is passed on; local ports are probed either way."""
    parts = urlsplit(origin)
    return parts.netloc if parts.scheme == "http" else None


def release_slot_refusal(config: Any) -> None:
    """Run the legacy backend's release-slot check during review, before any fetch or driver stop."""
    from ..cli import legacy_update

    if not config.hosts:
        return
    refused = legacy_update._release_slot_refusal(config)
    if refused == 2:
        raise StudioError("update_release_slot", "Legacy update refuses: a Worker runs from a release slot "
                          "(release_slot_worker), which plain update cannot refresh. Select verified mode. "
                          "Nothing was fetched or changed.", 409)
    if refused is not None:
        raise StudioError("update_release_slot_unknown", "Legacy update refuses: a Worker host's source link "
                          "could not be read (release_slot_unknown). Retry once every host answers. "
                          "Nothing was fetched or changed.", 409)


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

    def selected_config(self) -> Path:
        if self.config is None:
            raise StudioError("update_config", "Update requires the cluster configuration selected with --config at "
                              "companion launch, in both modes.", 409)
        return self.config

    def operations(self, parameters: dict[str, Any]) -> Any:
        config = self.selected_config()
        repo = absolute(parameters["repo"]).resolve()
        checkout_identity(repo)
        if parameters.get("mode", "verified") == "legacy":
            return RepositoryOps(repo)
        return SystemUpdateOps(repo, load_cluster_config(config), driver_host=None,
                               privileged_process_inspection=parameters["privileged_process_inspection"])

    def preview(self, parameters: dict[str, Any]) -> dict[str, Any]:
        if parameters.get("receipt"):
            absolute(parameters["receipt"])
        ops = self.operations(parameters)
        config = self.selected_config()
        config_sha256 = update_controller.config_digest(config)
        driver = self.driver.inspect() if self.driver else None
        bound = driver_checkout(driver, ops.repo)
        controller = update_controller.binding(driver)
        if driver and driver["state"] == "active" and not parameters["stop_driver"]:
            raise StudioError("update_driver", "Explicit driver-stop authority is required for the running driver.", 409)
        if parameters["restore_driver"] and (not driver or driver["state"] != "active" or not parameters["stop_driver"]):
            raise StudioError("update_driver", "Restoration requires an explicitly reviewed stop of the running owner driver.", 409)
        with UpdateLock(ops.repo):
            mode = parameters.get("mode", "verified")
            legacy_candidate = None
            if mode == "legacy":
                from ..cli import legacy_update
                if parameters["target_ref"] != "origin/HEAD" or parameters["privileged_process_inspection"] or parameters.get("receipt"):
                    raise StudioError("update_fields", "Target selection, receipts and privileged inspection require verified mode.")
                release_slot_refusal(load_cluster_config(config))
                if not (ops.repo / ".git").is_dir() or legacy_update._contract_edited(ops.repo):
                    raise StudioError("update_repository", "Legacy update requires a plain checkout with unchanged package contract.", 409)
                legacy_candidate = legacy_update._fetch_candidate(ops.repo)
                if legacy_candidate is None:
                    raise StudioError("update_repository", "The legacy upstream candidate could not be established.", 409)
                target = legacy_candidate[0]
            else:
                target = ops.resolve_target(parameters["target_ref"])
            prior = ops.repo_snapshot(target)
            if not prior.clean or not prior.fast_forward:
                raise StudioError("update_repository", "The reviewed update requires a clean fast-forward checkout.", 409)
            original, _ = inspect_checkout(ops.repo, prior.current_sha)
            if controller["source_manifest"] != original.source_manifest:
                raise StudioError("update_controller_source", "The companion runs dgx_monarch source that differs from "
                                  "the selected checkout. Start Studio from the checkout the driver imports.", 409)
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
        if update_controller.config_digest(config) != config_sha256:
            raise StudioError("update_changed", "Configuration changed during review.", 409)
        return {"token": identifier(), "prior_commit": prior.current_sha, "prior_release": asdict(original),
                "mode": mode, "legacy_candidate": legacy_candidate, "receipt_path": parameters.get("receipt") or None,
                "checkout": checkout_identity(ops.repo), "driver_checkout": bound, "release": asdict(metadata),
                "driver_identity": driver, "controller": controller, "config_sha256": config_sha256,
                "review_scope": REVIEW_SCOPE[mode], "restart_policy": RESTART_POLICY}

    async def admit(self, ops: Any, parameters: dict[str, Any], reviewed: dict[str, Any], lock_fd: int) -> dict[str, Any]:
        """Recheck the review and freeze the controller; every refusal here precedes any change."""
        expected = reviewed["release"]
        if checkout_identity(ops.repo) != reviewed["checkout"]:
            raise StudioError("update_changed", "The selected source checkout identity changed after review.", 409)
        prior = await blocking(ops.repo_snapshot, expected["target_sha"])
        if not prior.clean or not prior.fast_forward or prior.current_sha != reviewed["prior_commit"]:
            raise StudioError("update_changed", "Driver checkout changed after the source review.", 409)
        original, _ = await blocking(inspect_checkout, ops.repo, prior.current_sha)
        if asdict(original) != reviewed["prior_release"]:
            raise StudioError("update_changed", "The selected prior source content changed after review.", 409)
        current_driver = await blocking(self.driver.inspect) if self.driver else None
        if driver_checkout(current_driver, ops.repo) != reviewed["driver_checkout"]:
            raise StudioError("update_changed", "The driver's checkout link changed after review.", 409)
        if current_driver != reviewed["driver_identity"] or update_controller.binding(current_driver) != reviewed["controller"]:
            raise StudioError("update_changed", "Driver or controller identity changed after review.", 409)
        driver_identity = reviewed["driver_identity"]
        if driver_identity["state"] == "active" and parameters["stop_driver"] is not True:
            raise StudioError("update_driver", "The reviewed owner driver cannot be stopped.", 409)
        # Verified mode gets no explicit host: Studio rechecked the bound driver,
        # and the transaction's activity guard refuses any named host it cannot reach.
        host = legacy_driver_host(self.origin) if reviewed["mode"] == "legacy" else None
        return await blocking(update_controller.prepare, self.directory, self.config, ops.repo, reviewed,
                              host, parameters["privileged_process_inspection"], lock_fd)

    async def execute(self, parameters: dict[str, Any], reviewed: dict[str, Any],
                      checkpoint: Callable[[dict[str, Any]], None]) -> tuple[str, dict[str, Any]]:
        mode = reviewed.get("mode", "verified")
        receipt: dict[str, Any] = {"source": mode + "_update", "token": reviewed["token"], "release": reviewed["release"]}

        def refused(exc: StudioError) -> tuple[str, dict[str, Any]]:
            return "failed", {**receipt, "refusal": {"code": exc.code, "message": str(exc)},
                              "recovery": "The update was refused before any change; review again."}

        driver = self.driver
        try:
            ops = self.operations(parameters)
            if driver is None:
                raise StudioError("update_driver_service", DRIVER_SERVICE_REQUIRED, 409)
        except StudioError as exc:
            return refused(exc)
        with UpdateLock(ops.repo) as lock:
            try:
                prepared = await self.admit(ops, parameters, {**reviewed, "mode": mode}, lock.fileno())
            except StudioError as exc:
                return refused(exc)
            receipt["controller"] = prepared
            checkpoint({**receipt, "phase": "controller_prepared"})
            stopped = None
            driver_identity = reviewed["driver_identity"]
            if driver_identity["state"] == "active":
                checkpoint({**receipt, "phase": "driver_stop_requested", "driver_identity": driver_identity})
                state, stop_receipt = await settle(driver.perform("stop", driver_identity))
                receipt["driver_stop"] = stop_receipt
                checkpoint(receipt)
                if state != "succeeded":
                    return "unknown", {**receipt, "recovery": "Driver stop is uncertain; update and restoration were not attempted."}
                stopped = stop_receipt["observed_identity"]
            checkpoint({**receipt, "phase": mode + "_update_requested"})

            result = await blocking(update_controller.run, prepared)
            outcome = update_controller.settlement(mode, result)
            receipt.update(update=dict(result.receipt), result_code=result.code, result_message=sanitize_note(result.message),
                           settlement=outcome, controller_stderr=update_controller.stderr_tail(prepared))
            checkpoint(receipt)
            if outcome == "unknown":
                return "unknown", {**receipt, "recovery": "The update outcome is unknown; driver restoration was suppressed. "
                                   "Inspect the backend receipt and the retained private controller before any retry."}
            if outcome == "settled_after_change":
                return "failed", {**receipt, "recovery": "The verified transaction changed Worker services, then restored "
                                  "the prior release after a definite failure; the driver checkout stays at its prior "
                                  "commit. Driver restoration was not attempted: inspect the receipt, then start ComfyUI."}
            if stopped and parameters["restore_driver"]:
                current = await blocking(driver.inspect)
                if current["config_digest"] != stopped["config_digest"] or current["state"] != "inactive":
                    return "unknown", {**receipt, "recovery": "Stopped driver identity changed; restoration was not attempted."}
                environment = update_controller.binding(current)
                if any(environment[key] != value for key, value in reviewed["controller"].items() if key != "source_manifest"):
                    return "unknown", {**receipt, "recovery": "Driver Python environment changed; restoration was not attempted."}
                checkpoint({**receipt, "phase": "driver_restore_requested", "driver_identity": current})
                state, restored = await settle(driver.perform("start", current))
                receipt["driver_restore"] = restored
                checkpoint(receipt)
                if state != "succeeded":
                    return "unknown", {**receipt, "recovery": "Driver restoration is uncertain; inspect the driver service."}
            await blocking(update_controller.cleanup, prepared)
            if outcome == "refused":
                return "failed", {**receipt, "controller_cleaned": True,
                                  "recovery": "The backend refused before it changed anything; see its reason."}
            return "succeeded", {**receipt, "controller_cleaned": True}
