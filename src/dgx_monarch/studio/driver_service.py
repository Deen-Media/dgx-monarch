"""Review-bound lifecycle control for an existing owner ComfyUI user unit.

This adapter never installs a unit, invokes a shell, or controls persistent
Monarch Workers. Unsupported unit layouts require an explicit adapter.
"""

from __future__ import annotations

import asyncio
import os
import pwd
import re
import shlex
import stat
import subprocess
from pathlib import Path
from typing import Any

from .contracts import StudioError, digest

_PROPERTIES = (
    "Id", "LoadState", "ActiveState", "SubState", "FragmentPath", "DropInPaths", "NeedDaemonReload",
    "ExecStart", "ExecStartPre", "ExecStartPost", "ExecStop", "ExecStopPost", "ExecReload", "ExecCondition",
    "MainPID", "ControlGroup", "User", "Transient", "InvocationID", "UnitFileState",
    "Conflicts", "PartOf", "BindsTo", "PropagatesStopTo", "Restart",
    "Requires", "Wants", "Requisite", "OnFailure", "OnSuccess",
    "EnvironmentFiles", "KillMode",
)
_MARKERS = {"main.py": ("import comfy.options", "import folder_paths"),
            "server.py": ("class PromptServer",), "execution.py": ("class PromptExecutor",),
            "nodes.py": ("NODE_CLASS_MAPPINGS",), "folder_paths.py": ("folder_names_and_paths",)}
_ACTIONS = {"start", "stop", "restart"}


def _refuse(message: str) -> StudioError:
    return StudioError("driver_service_identity", message, 409)


def validate_unit_name(unit: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]*\.service", unit):
        raise _refuse("Select an existing systemd user service by exact unit name.")
    if "worker" in unit.lower():
        raise _refuse("Persistent Monarch Worker services cannot be selected as the ComfyUI driver.")


def _record(path: Path, uid: int, *, limit: int = 4 * 1024 * 1024) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != uid or metadata.st_mode & 0o022:
            raise _refuse("Driver unit and source files must be regular, owner-controlled, and not group/world writable.")
        if metadata.st_size > limit:
            raise _refuse("Driver identity file exceeds the supported size limit.")
        content = path.read_bytes()
        after = path.lstat()
        before_identity = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
        if before_identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise _refuse("Driver identity changed while it was being inspected.")
        return {"path": str(path), "device": metadata.st_dev, "inode": metadata.st_ino,
                "mode": metadata.st_mode, "uid": metadata.st_uid, "digest": digest(content.hex())}
    except OSError as exc:
        raise _refuse("Driver identity file is unavailable.") from exc


def _direct_command(serialized: str) -> tuple[Path, Path, list[str]]:
    match = re.fullmatch(r"\{\s*path=([^;]+?)\s*;\s*argv\[\]=(.*?)\s*;\s*ignore_errors=no\s*;.*\}", serialized)
    if not match or "} {" in serialized or "\\" in match[2]:
        raise _refuse("Driver ExecStart must be one directly inspectable Python command.")
    executable = Path(match[1])
    try:
        argv = shlex.split(match[2])
    except ValueError as exc:
        raise _refuse("Driver command serialization is ambiguous.") from exc
    if not argv or argv[0] != str(executable) or not executable.is_absolute():
        raise _refuse("Driver command must use an absolute Python executable.")
    if not re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable.name):
        raise _refuse("Driver wrappers and arbitrary service commands are unsupported.")
    arguments = argv[1:]
    while arguments and arguments[0] in {"-u", "-B"}:
        arguments = arguments[1:]
    if not arguments:
        raise _refuse("Driver command has no ComfyUI entrypoint.")
    main = Path(arguments[0])
    if not main.is_absolute() or main.name != "main.py":
        raise _refuse("Driver command must invoke an absolute verified ComfyUI main.py.")
    if any("worker" in argument.lower() for argument in argv):
        raise _refuse("A Worker command cannot be selected as the ComfyUI driver.")
    return executable, main, argv


class DriverService:
    def __init__(self, unit: str):
        validate_unit_name(unit)
        self.unit = unit
        self.uid = os.getuid()
        self.owner_root = Path(pwd.getpwuid(self.uid).pw_dir) / ".config/systemd/user"

    def _show(self) -> dict[str, str]:
        try:
            result = subprocess.run(
                ["systemctl", "--user", "show", "--no-pager", "--property=" + ",".join(_PROPERTIES), "--", self.unit],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise _refuse("The owner user service manager is unavailable.") from exc
        if result.returncode or len(result.stdout) > 131072:
            raise _refuse("The selected driver service cannot be inspected.")
        try:
            lines = result.stdout.decode("utf-8").splitlines()
            fields = dict(line.split("=", 1) for line in lines)
        except (UnicodeError, ValueError) as exc:
            raise _refuse("The user service manager returned an unsupported identity record.") from exc
        if set(fields) != set(_PROPERTIES) or len(fields) != len(lines):
            raise _refuse("The user service manager omitted required identity properties.")
        return fields

    def _unit_file(self, value: str) -> dict[str, Any]:
        path = Path(value)
        if not path.is_absolute() or not path.is_relative_to(self.owner_root):
            raise _refuse("Only units installed in the current owner's user-service directory are supported.")
        if any(part.is_symlink() for part in [path, *path.parents]):
            raise _refuse("Symlinked service units and drop-ins are unsupported.")
        return _record(path, self.uid)

    def _process(self, pid: int, argv: list[str], cgroup: str) -> dict[str, Any]:
        root = Path("/proc") / str(pid)
        try:
            if root.stat().st_uid != self.uid:
                raise _refuse("The driver process belongs to another owner.")
            actual = root.joinpath("cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
            groups = root.joinpath("cgroup").read_text().splitlines()
            start = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, UnicodeError, IndexError) as exc:
            raise _refuse("The driver process identity is unavailable or changed during inspection.") from exc
        if actual != argv or not cgroup or not any(line.split(":", 2)[-1] == cgroup for line in groups):
            raise _refuse("The running process does not match the reviewed ComfyUI command and service cgroup.")
        return {"pid": pid, "start_ticks": start, "command_digest": digest(actual), "cgroup": cgroup}

    def inspect(self) -> dict[str, Any]:
        fields = self._show()
        if fields["Id"] != self.unit or fields["LoadState"] != "loaded" or fields["Transient"] != "no":
            raise _refuse("Driver unit must be an existing loaded, non-transient owner service.")
        if fields["NeedDaemonReload"] != "no":
            raise _refuse("Driver unit files and the service manager differ; reconcile them before review.")
        if fields["EnvironmentFiles"]:
            raise _refuse("External environment files require an additional reviewed identity binding.")
        if fields["KillMode"] not in {"control-group", "mixed"}:
            raise _refuse("Driver service must retain control over its complete process group.")
        if fields["User"] not in {"", str(self.uid), pwd.getpwuid(self.uid).pw_name}:
            raise _refuse("Driver unit specifies another process owner.")
        if any(fields[name] for name in ("ExecStartPre", "ExecStartPost", "ExecStop", "ExecStopPost", "ExecReload", "ExecCondition")):
            raise _refuse("Driver lifecycle helper commands require a separately reviewed adapter.")
        for name in ("Conflicts", "PartOf", "BindsTo", "PropagatesStopTo", "Requires", "Wants", "Requisite", "OnFailure", "OnSuccess"):
            if any(item.endswith(".service") for item in fields[name].split()):
                raise _refuse("Driver lifecycle is coupled to another service; this adapter cannot control it safely.")
        unit_files = [self._unit_file(fields["FragmentPath"])]
        try:
            drop_ins = shlex.split(fields["DropInPaths"])
        except ValueError as exc:
            raise _refuse("Driver drop-in paths cannot be inspected.") from exc
        unit_files.extend(self._unit_file(path) for path in drop_ins)
        executable, main, argv = _direct_command(fields["ExecStart"])
        try:
            interpreter = executable.resolve(strict=True)
            interpreter_stat = interpreter.stat()
            if interpreter_stat.st_uid not in {0, self.uid} or interpreter_stat.st_mode & 0o022:
                raise _refuse("Driver Python interpreter is not owner-controlled.")
        except OSError as exc:
            raise _refuse("Driver Python interpreter is unavailable.") from exc
        source = []
        for filename, markers in _MARKERS.items():
            path = main.parent / filename
            record = _record(path, self.uid)
            text = path.read_text(errors="replace")
            if any(marker not in text for marker in markers):
                raise _refuse("Driver entrypoint lacks the required ComfyUI source identity markers.")
            source.append(record)
        try:
            pid = int(fields["MainPID"])
        except ValueError as exc:
            raise _refuse("Driver process identity is unknown.") from exc
        state = (fields["ActiveState"], fields["SubState"])
        if state == ("active", "running") and pid > 0:
            process = self._process(pid, argv, fields["ControlGroup"])
        elif state == ("inactive", "dead") and pid == 0:
            process = None
        else:
            raise _refuse("Driver is not in a known stable running or stopped state.")
        config = {"unit": self.unit, "owner_uid": self.uid, "unit_files": unit_files,
                  "source_files": source, "exec_digest": digest(argv), "interpreter": str(interpreter),
                  "interpreter_identity": [interpreter_stat.st_dev, interpreter_stat.st_ino, interpreter_stat.st_mtime_ns],
                  "service_policy": {name: fields[name] for name in ("UnitFileState", "Restart", "KillMode")}}
        return {"schema_version": 1, "role": "comfyui_driver", "config": config,
                "config_digest": digest(config), "state": state[0], "substate": state[1],
                "process": process, "invocation_id": fields["InvocationID"]}

    async def perform(self, action: str, reviewed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if action not in _ACTIONS:
            raise _refuse("Unsupported driver service action.")
        before = await asyncio.to_thread(self.inspect)
        if before != reviewed:
            raise _refuse("Driver unit, source, state, or process changed after review; review again.")
        if action == "start" and before["state"] != "inactive":
            raise _refuse("Start requires a reviewed stopped driver.")
        if action in {"stop", "restart"} and before["state"] != "active":
            raise _refuse("Stop or restart requires a reviewed running driver.")
        code: int | None = None
        try:
            result = await asyncio.to_thread(
                subprocess.run, ["systemctl", "--user", action, "--", self.unit],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=30, check=False,
            )
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired):
            pass
        receipt: dict[str, Any] = {"source": "owner_service", "unit": self.unit, "action": action,
                                   "reviewed_identity": reviewed, "exit_code": code}
        try:
            after = await asyncio.to_thread(self.inspect)
        except StudioError:
            receipt["verification"] = "unknown"
            return "unknown", receipt
        receipt["observed_identity"] = after
        expected = "inactive" if action == "stop" else "active"
        verified = code == 0 and after["config_digest"] == before["config_digest"] and after["state"] == expected
        if action == "restart":
            verified = verified and after["process"] != before["process"] and after["invocation_id"] != before["invocation_id"]
        receipt["verification"] = "verified" if verified else "unknown"
        return ("succeeded" if verified else "unknown"), receipt
