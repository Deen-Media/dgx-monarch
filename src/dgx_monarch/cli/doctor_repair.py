"""Allowlisted, reversible repairs for ``dgxm doctor --repair``.

The report may name many fixes; the only automatic one is the mode of the exact
config file. Network, SSH, package, service, process, mesh and power changes
stay with the operator (`manual_only_effects`).
"""
from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from .operator_receipt import OperatorReceiptBuilder, write_receipt
from .setup_config_lock import SetupConfigLock, SetupConfigLockUnavailable

_PRIVATE_FILE_MODE = 0o600
_CONFIG_PERMISSION_ACTION = "tighten_config_permissions"


class RepairError(RuntimeError):
    """A repair could not be proven safe or complete."""


@dataclass(frozen=True)
class PermissionRepair:
    action_id: str
    needed: bool
    reversible: bool
    previous_mode: int
    desired_mode: int
    config_sha256: str
    detail: str
    inspected_device: int = field(repr=False)
    inspected_inode: int = field(repr=False)

    def wire(self) -> dict[str, object]:
        """Return a path-free receipt fragment."""
        return {
            "action_id": self.action_id,
            "needed": self.needed,
            "reversible": self.reversible,
            "previous_mode": self.previous_mode,
            "desired_mode": self.desired_mode,
            "config_sha256": self.config_sha256,
            "detail": self.detail,
        }


def _regular_config_fd(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RepairError("config_permissions_unreadable") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RepairError("config_permissions_not_regular")
        return fd, before
    except BaseException:
        os.close(fd)
        raise


def inspect_config_permissions(path: str | os.PathLike[str]) -> PermissionRepair:
    """Inspect the exact config inode without following a final symlink."""
    config_path = Path(path).expanduser()
    fd, before = _regular_config_fd(config_path)
    try:
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    except OSError as exc:
        raise RepairError("config_permissions_read_failed") from exc
    finally:
        os.close(fd)
    previous = stat.S_IMODE(before.st_mode)
    needed = bool(previous & 0o077)
    detail = (
        "config is open to users other than its owner; tighten it to 0600"
        if needed
        else "config permissions are already private"
    )
    return PermissionRepair(
        action_id=_CONFIG_PERMISSION_ACTION,
        needed=needed,
        reversible=True,
        previous_mode=previous,
        desired_mode=_PRIVATE_FILE_MODE,
        config_sha256=digest.hexdigest(),
        detail=detail,
        inspected_device=before.st_dev,
        inspected_inode=before.st_ino,
    )


def apply_config_permission_repair(
    path: str | os.PathLike[str],
    plan: PermissionRepair,
    *,
    confirm: Callable[[PermissionRepair], bool],
) -> PermissionRepair:
    """Apply an inspected plan after confirmation and an inode/content recheck."""
    if plan.action_id != _CONFIG_PERMISSION_ACTION:
        raise RepairError("repair_action_not_allowlisted")
    if plan.desired_mode != _PRIVATE_FILE_MODE:
        raise RepairError("repair_mode_not_allowlisted")
    if not plan.needed:
        return plan
    if not confirm(plan):
        raise RepairError("repair_not_confirmed")

    config_path = Path(path).expanduser()
    fd, before = _regular_config_fd(config_path)
    try:
        if (before.st_dev, before.st_ino) != (
            plan.inspected_device,
            plan.inspected_inode,
        ):
            raise RepairError("config_inode_changed_since_inspection")
        if stat.S_IMODE(before.st_mode) != plan.previous_mode:
            raise RepairError("config_permissions_changed_since_inspection")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if digest.hexdigest() != plan.config_sha256:
            raise RepairError("config_changed_since_inspection")
        os.fchmod(fd, _PRIVATE_FILE_MODE)
        os.fsync(fd)
        after = os.fstat(fd)
        if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
            raise RepairError("config_inode_changed_during_repair")
        if stat.S_IMODE(after.st_mode) != _PRIVATE_FILE_MODE:
            raise RepairError("config_permissions_not_confirmed")
    except OSError as exc:
        raise RepairError("config_permissions_write_failed") from exc
    finally:
        os.close(fd)
    return PermissionRepair(
        action_id=plan.action_id,
        needed=False,
        reversible=True,
        previous_mode=plan.previous_mode,
        desired_mode=_PRIVATE_FILE_MODE,
        config_sha256=plan.config_sha256,
        detail="config permissions tightened to 0600",
        inspected_device=plan.inspected_device,
        inspected_inode=plan.inspected_inode,
    )


def manual_only_effects() -> tuple[str, ...]:
    """Effects that diagnostics may recommend but repair must never execute."""
    return ("network", "ssh", "package", "service", "process", "mesh", "power")


def run_config_permission_repair(
    path: str | os.PathLike[str],
    *,
    assume_yes: bool = False,
    receipt_path: str | os.PathLike[str] | None = None,
    input_fn: Callable[[str], str] = input,
    output: TextIO,
) -> tuple[bool, Path | None]:
    """Inspect, optionally repair, and write a disclosure-safe receipt."""
    try:
        plan = inspect_config_permissions(path)
    except RepairError as exc:
        print(f"safe repair unavailable: {exc}", file=output)
        return False, None

    receipt = OperatorReceiptBuilder(
        "doctor_repair", source_hashes={"config": plan.config_sha256}
    )
    destination: Path | None
    receipt.add_step(
        "inspect_config_permissions",
        "succeeded",
        counts={"repairs_needed": int(plan.needed)},
    )
    if not plan.needed:
        receipt.add_step("tighten_config_permissions", "succeeded", counts={"changed": 0})
        final = receipt.finish("succeeded", notes=("No safe repair was needed.",))
        try:
            destination = write_receipt(final, receipt_path)
        except (OSError, ValueError) as exc:
            print(f"config is already private, but writing the repair receipt failed: {exc}", file=output)
            return False, None
        print("config permissions are already owner-only", file=output)
        return True, destination

    print(
        f"safe repair: change config permissions from {plan.previous_mode:04o} to 0600; "
        "no services, processes, network, SSH, packages, power or mesh state will change",
        file=output,
    )
    confirmed = assume_yes or input_fn("apply this reversible filesystem repair? [y/N] ").strip().lower() == "y"
    try:
        if not confirmed:
            raise RepairError("repair_not_confirmed")
        config_path = Path(path).expanduser().absolute()
        with SetupConfigLock(config_path):
            result = apply_config_permission_repair(
                path, plan, confirm=lambda _plan: True
            )
    except SetupConfigLockUnavailable:
        error = RepairError("config_repair_lock_unavailable")
        receipt.add_step(
            "tighten_config_permissions", "failed", counts={"changed": 0}, notes=(str(error),)
        )
        final = receipt.finish("failed", notes=("The safe repair was not applied.",))
        try:
            destination = write_receipt(final, receipt_path)
        except (OSError, ValueError):
            destination = None
        print(f"safe repair stopped: {error}", file=output)
        return False, destination
    except RepairError as exc:
        receipt.add_step(
            "tighten_config_permissions", "failed", counts={"changed": 0}, notes=(str(exc),)
        )
        final = receipt.finish("failed", notes=("The safe repair was not applied.",))
        try:
            destination = write_receipt(final, receipt_path)
        except (OSError, ValueError):
            destination = None
        print(f"safe repair stopped: {exc}", file=output)
        return False, destination

    receipt.add_step(
        "tighten_config_permissions", "succeeded", counts={"changed": 1}
    )
    final = receipt.finish(
        "succeeded",
        notes=(f"Prior mode was {result.previous_mode:04o}; the repair is reversible.",),
    )
    try:
        destination = write_receipt(final, receipt_path)
    except (OSError, ValueError) as exc:
        print(
            f"config permissions tightened to 0600, but writing the repair receipt failed: {exc}",
            file=output,
        )
        return False, None
    print("config permissions tightened to 0600", file=output)
    return True, destination
