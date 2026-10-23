"""Fixed management operations with persisted review, confirmation and receipts."""
from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path
from typing import Any

from .. import __version__
from ..cli.probe_certainty import UNKNOWN_EXIT
from . import operations_lock
from .comfy import ComfyClient
from .contracts import StudioError, digest, identifier, timestamp
from .driver_service import DriverService
from .management_consent import consent_plan
from .management_fields import COMMAND_FIELDS, FIELDS, FIXED_COMMANDS, NO_CONFIG, parse
from .operation_reservations import reserve_review
from .store import Store
from .threads import blocking

LOG = logging.getLogger(__name__)

# Each command and every flag here is owned by Monarch. Browser JSON never becomes argv.
COMMANDS = FIXED_COMMANDS

DRIVER_ACTIONS = {"driver.start": "start", "driver.stop": "stop", "driver.restart": "restart"}
UPDATE_NEEDS_DRIVER_SERVICE = ("Restart the companion with --driver-service naming the existing ComfyUI driver "
                               "unit; update binds that unit's interpreter and checkout.")
HTTP_ACTIONS = {"actors.recycle": "/dgxm/recycle", "driver.free": "/free", "consent.apply": "/dgxm/consent"}
READ_ONLY = frozenset({"version", "status", "doctor", "reap.preview", "gate.inspect", "gate.repair.preview", "top.replay",
                       "diagnostics", "diagnostics.capabilities"})
# A 4xx from these driver routes means the driver refused and applied nothing.
REFUSING_ROUTES = frozenset({"actors.recycle", "consent.apply"})
# A spawned child must import the installed package: -P keeps the working
# directory off sys.path, -s skips user site-packages and -B writes no bytecode.
# PYTHONPATH still applies, as it does for the companion itself.
CHILD_FLAGS = ("-P", "-s", "-B")
DIAGNOSTICS_WRITE = "Restart the companion with --allow-management to write a diagnostics file."
DESCRIPTIONS = {
    "diagnostics": "Reads telemetry only. An output path writes a file and needs --allow-management and confirmation.",
    "consent.apply": "Leave key and id empty to list the driver's pending consent requests; one pending request is selected.",
}
_VERSION_LINE = re.compile(r"dgx-monarch ([A-Za-z0-9.+!_-]{1,64}) \(torchmonarch pin ([A-Za-z0-9.+!_-]{1,64})\)")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def management_write(action: str, parameters: Any) -> bool:
    """Whether this exact request changes host state.

    A diagnostics capture only reads telemetry, but a request that names an
    output path writes a new file there, so it needs the same authority and
    confirmation as any other management action.
    """
    if action not in READ_ONLY:
        return True
    return action == "diagnostics" and isinstance(parameters, dict) and bool(parameters.get("output"))


def absolute_path(value: str) -> str:
    """A browser-typed path must not depend on the companion's working directory."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise StudioError("operation_path", "Management paths must be absolute.")
    return str(path)


def exit_state(code: int) -> str:
    """The receipt state for a spawned command's exit code.

    Every spawned command reports an unanswered critical probe as
    probe_certainty.UNKNOWN_EXIT, and a child ended by a signal reached no
    verdict either. Every other non-zero code is a definite failure or refusal.
    """
    if code == 0:
        return "succeeded"
    return "unknown" if code == UNKNOWN_EXIT or code < 0 else "failed"


def version_receipt(output: bytes) -> dict[str, Any]:
    """Report the version the child printed, never the companion's own."""
    for line in output.decode("utf-8", "replace").splitlines():
        match = _VERSION_LINE.fullmatch(line.strip())
        if match:
            return {"version": match[1], "torchmonarch_pin": match[2], "companion_version": __version__}
    return {"version": None, "torchmonarch_pin": None, "companion_version": __version__,
            "version_reason": "The command printed no recognizable version line."}


def driver_detail(value: Any) -> str | None:
    return _CONTROL.sub(" ", value)[:300] if isinstance(value, str) else None


def command_arguments(action: str, parameters: dict[str, Any]) -> list[str]:
    """Fixed commands and individually typed options; never accept command text."""
    args = list(COMMANDS[action])
    if action in {"workers.up", "workers.restart"} and parameters["no_sync"]:
        args.append("--no-sync")
    if action == "doctor.repair" and parameters.get("receipt"):
        args += ["--receipt", absolute_path(parameters["receipt"])]
    if action.startswith("reap."):
        args += ["--grace", str(parameters["grace_s"])]
    if action.startswith("gate."):
        args += ["--report-dir", str(Path(parameters["report_dir"]).expanduser()), "--timeout", str(parameters["timeout_s"])]
    return args


def fingerprint(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        info = path.stat()
        return digest([info.st_dev, info.st_ino, info.st_mode, path.read_bytes().hex()])
    except OSError:
        return "unavailable"


class Operations:
    def __init__(self, store: Store, client: ComfyClient, *, config: Path | None = None,
                 allow_management: bool = False, driver_service: str | None = None,
                 driver_start: dict[str, str | None] | None = None):
        self.store, self.client = store, client
        # The command the first-run page shows (studio/driver_hint.py), read once at launch; never run.
        self.driver_start = driver_start or {"command": None, "source": None, "config": None}
        # Children run with -P, so the selection cannot lean on the launch directory.
        self.config = config.expanduser().absolute() if config is not None else None
        self.allow_management = allow_management
        self.driver_service = driver_service
        self.driver_binding = (DriverService(driver_service, origin=getattr(client, "origin", None))
                               if driver_service else None)
        self.tasks: set[asyncio.Task[Any]] = set()
        self.execution_lock = asyncio.Lock()
        self._management: Any = None

    @property
    def management(self) -> Any:
        if self._management is None:
            from .management import Management

            self._management = Management(self.store, self.client, self.config, self.driver_binding)
        return self._management

    def capabilities(self) -> list[dict[str, Any]]:
        rows = []
        for action in sorted(set(COMMANDS) | set(HTTP_ACTIONS) | set(DRIVER_ACTIONS) | set(FIELDS)):
            reason = None
            if not reason and action not in READ_ONLY and not self.allow_management:
                reason = "Restart the companion with --allow-management to authorize management controls."
            if not reason and action in DRIVER_ACTIONS and not self.driver_service:
                reason = "No existing owner-managed ComfyUI driver service was selected at companion launch."
            if not reason and (action in COMMANDS or action in FIELDS) and action not in NO_CONFIG | {"version", "setup", "init", "launcher"} and not self.config:
                reason = "Select an existing cluster configuration when starting the companion."
            if not reason and action == "update" and not self.driver_service:
                # Only the driver unit names the interpreter and checkout an update must bind.
                reason = UPDATE_NEEDS_DRIVER_SERVICE
            fields = FIELDS.get(action, COMMAND_FIELDS.get(action, []))
            if action == "driver.free":
                fields = [{"name": "unload_models", "type": "BOOLEAN", "default": True},
                          {"name": "free_memory", "type": "BOOLEAN", "default": True}]
            if action == "consent.apply":
                fields = [{"name": "key", "type": "STRING", "default": ""},
                          {"name": "id", "type": "STRING", "default": ""}]
            rows.append({"id": action, "label": action.replace(".", " · "), "available": reason is None,
                         "reason": reason, "parameters": fields, "requires_confirmation": action not in READ_ONLY,
                         "implementation": "typed_adapter" if action in FIELDS else "adapter",
                         **({"description": DESCRIPTIONS[action]} if action in DESCRIPTIONS else {})})
        return rows

    async def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        """Publish a review under a shared hold on the operations lock, so up, down and --rotate wait for it."""
        hold = operations_lock.hold(self.store.directory)
        try:
            return await self._prepare(request)
        finally:
            operations_lock.release(hold)

    async def _prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        """Obtain authenticated remote evidence before publishing a review."""
        if not isinstance(request.get("parameters", {}), dict):
            raise StudioError("operation_fields", "Operation parameters must be an object.")
        if request.get("action") == "gate.last" and self.allow_management:
            parameters = parse("gate.last", request.get("parameters", {}))
            plan = await self.management.gate_preview(parameters)
            return await blocking(self.preview, request, plan)
        if request.get("action") == "consent.apply" and self.allow_management:
            # Only the driver lists the exact pending key and id an accept needs.
            pending = await self.client.request("GET", "/dgxm/consents")
            return await blocking(self.preview, request, pending)
        return await blocking(self.preview, request)

    def preview(self, request: dict[str, Any], remote_plan: Any = None) -> dict[str, Any]:
        action, parameters = request.get("action"), request.get("parameters", {})
        row = next((row for row in self.capabilities() if row["id"] == action), None)
        if row is None:
            raise StudioError("operation", "Unknown operation.")
        if not isinstance(parameters, dict):
            raise StudioError("operation_fields", "Operation parameters must be an object.")
        allowed = {field["name"] for field in row["parameters"]}
        if set(parameters) - allowed:
            raise StudioError("operation_fields", "Unsupported operation parameter.")
        if action in COMMAND_FIELDS or action == "diagnostics":
            parameters = parse(action, parameters)
        if action == "doctor.repair" and parameters["receipt"]:
            parameters = {**parameters, "receipt": absolute_path(parameters["receipt"])}
        writes = management_write(row["id"], parameters)
        if writes and row["available"] and not self.allow_management:
            row = {**row, "available": False, "reason": DIAGNOSTICS_WRITE}
        row = {**row, "requires_confirmation": writes}
        if action == "driver.free":
            parameters = {"unload_models": parameters.get("unload_models", True), "free_memory": parameters.get("free_memory", True)}
            if any(type(value) is not bool for value in parameters.values()):
                raise StudioError("operation_fields", "Memory controls must be booleans.")
        if action == "consent.apply":
            parameters = {"key": parameters.get("key", ""), "id": parameters.get("id", "")}
            if any(not isinstance(value, str) or len(value) > 512 for value in parameters.values()):
                raise StudioError("operation_fields", "An exact pending consent key and id are required.")
        driver_identity = None
        if action in DRIVER_ACTIONS and row["available"] and self.driver_binding:
            try:
                driver_identity = self.driver_binding.inspect()
                expected = "inactive" if action == "driver.start" else "active"
                if driver_identity["state"] != expected:
                    raise StudioError("driver_service_state", f"{action} requires a reviewed {expected} driver.", 409)
            except StudioError as exc:
                row = {**row, "available": False, "reason": str(exc)}
        adapter_plan = None
        if (action in FIELDS or action == "consent.apply") and row["available"]:
            try:
                if action in {"gate.last", "consent.apply"} and remote_plan is None:
                    raise StudioError("operation_plan", "This review requires authenticated driver evidence.", 409)
                if action == "gate.last":
                    parameters, adapter_plan = parse(action, parameters), remote_plan
                elif action == "consent.apply":
                    parameters, adapter_plan = consent_plan(parameters, remote_plan)
                else:
                    parameters, adapter_plan = self.management.preview(action, parameters)
                if adapter_plan.get("ready") is False:
                    refusal = adapter_plan.get("refusal")
                    row = {**row, "available": False,
                           "reason": refusal if isinstance(refusal, str) else "Backend plan is blocked; inspect its review evidence."}
            except StudioError:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                raise StudioError("operation_plan", "The backend could not prepare a safe management plan; inspect local prerequisites.", 409) from exc
        plan = {"action": action, "parameters": parameters, "config_fingerprint": fingerprint(self.config),
                "adapter_plan": adapter_plan,
                "driver_service": self.driver_service, "driver_identity": driver_identity, "version": __version__}
        return self.store.create("operations", {"id": identifier(), **plan, "digest": digest(plan),
                                 "description": row["label"], "requires_confirmation": row["requires_confirmation"],
                                 "available": row["available"], "reason": row["reason"], "state": "review",
                                 "created_at": timestamp(), "expires_at": timestamp() + 300,
                                 "review_scope": "Adapter request only; execution-time backend guards remain authoritative."})

    def apply(self, operation_id: str, request: dict[str, Any]) -> dict[str, Any]:
        existing = self.store.request("operations", request.get("idempotency_key"),
                                      {**request, "operation_id": operation_id})
        if existing is not None:
            return existing
        plan = self.store.get("operations", operation_id)
        current = next((row for row in self.capabilities() if row["id"] == plan["action"]), None)
        if not current or not current["available"]:
            raise StudioError("operation_unavailable", current["reason"] if current else "Operation removed.", 409)
        writes = management_write(plan["action"], plan.get("parameters"))
        if writes and not self.allow_management:
            raise StudioError("operation_unavailable", DIAGNOSTICS_WRITE, 409)
        if plan["driver_service"] != self.driver_service:
            raise StudioError("service_changed", "The selected driver service changed; review again.", 409)
        if not plan["available"]:
            raise StudioError("operation_unavailable", plan["reason"], 409)
        if request.get("digest") != plan["digest"] or plan["expires_at"] < timestamp():
            raise StudioError("stale_operation", "Operation review is stale or has a different digest.", 409)
        if (plan["requires_confirmation"] or writes) and request.get("confirmed") is not True:
            raise StudioError("confirmation", "Review and explicitly confirm this exact operation.", 409)
        if fingerprint(self.config) != plan["config_fingerprint"]:
            raise StudioError("config_changed", "Configuration changed after review; prepare another operation.", 409)
        value = {**plan, "id": identifier(), "review_id": operation_id, "state": "applying"}
        # The admitted operation holds the operations lock shared until it settles; no await runs in between.
        hold = operations_lock.hold(self.store.directory)
        created = False
        try:
            value, created = reserve_review(self.store, request.get("idempotency_key"),
                                            {**request, "operation_id": operation_id}, value)
            if created:
                task = asyncio.create_task(self.execute(value))
                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)
                task.add_done_callback(lambda _task: operations_lock.release(hold))
        finally:
            if not created:
                operations_lock.release(hold)
        return value

    async def execute(self, operation: dict[str, Any]) -> None:
        from .management_update import settle

        async with (asyncio.Lock() if operation["action"].startswith("top.") else self.execution_lock):
            action = operation["action"]
            try:
                current = next((row for row in self.capabilities() if row["id"] == action), None)
                if (not current or not current["available"] or operation["driver_service"] != self.driver_service
                        or (management_write(action, operation.get("parameters")) and not self.allow_management)):
                    raise StudioError("operation_unavailable", "Operation authority changed after review.", 409)
                if fingerprint(self.config) != operation["config_fingerprint"]:
                    raise StudioError("config_changed", "Configuration changed while awaiting the operation lock.", 409)
                if action in FIELDS:
                    state, receipt = await self.management.execute(operation)
                elif action in DRIVER_ACTIONS:
                    state, receipt = await settle(self.execute_owned(operation))
                elif action in HTTP_ACTIONS:
                    state, receipt = await self.driver_action(action, operation["parameters"])
                else:
                    state, receipt = await settle(self.execute_owned(operation))
                self.store.update("operations", operation["id"], {"state": state, "receipt": receipt,
                                  "completed_at": timestamp()})
            except asyncio.CancelledError:
                retained = self.store.get("operations", operation["id"]).get("receipt", {})
                self.store.update("operations", operation["id"], {"state": "unknown", "receipt": {**retained,
                                  "error": "operation_interrupted", "message": "Reconcile retained receipts before any retry."}})
                raise
            except Exception as exc:
                expected = isinstance(exc, (OSError, ValueError, RuntimeError))
                if not expected:
                    LOG.exception("Studio operation %s ended with an unexpected error", operation["id"])
                code = exc.code if isinstance(exc, StudioError) else "operation_io_failed" if expected else "operation_internal_error"
                retained = self.store.get("operations", operation["id"]).get("receipt", {})
                self.store.update("operations", operation["id"], {"state": "unknown", "receipt": {**retained,
                    "error": code, "message": "Operation outcome requires reconciliation; automatic retry is disabled."}})

    async def driver_action(self, action: str, parameters: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        body = dict(parameters)
        if action == "consent.apply":
            # The driver records this accept in the waiver ledger as a Studio consent.
            body.update(action="accept", source="studio")
        try:
            result = await self.client.request("POST", HTTP_ACTIONS[action], body)
        except StudioError as exc:
            status = getattr(exc, "upstream_status", None)
            if action not in REFUSING_ROUTES or type(status) is not int or not 400 <= status < 500:
                raise  # A 5xx or a lost response leaves the outcome unknown.
            return "failed", {"source": "driver", "error": exc.code, "upstream_status": status,
                              "detail": driver_detail(getattr(exc, "upstream_detail", None))}
        # The driver may return accepted, busy or unknown. Retain its actual result.
        state = "succeeded" if isinstance(result, dict) and result.get("ok") is True else "unknown"
        if action == "driver.free":
            state = "unknown"  # /free acknowledges queuing, not completed memory reclamation.
        return state, {"source": "driver", "result": result}

    async def execute_owned(self, operation: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Settle a local mutator and retain its receipt before releasing its lock."""
        action = operation["action"]
        if action in DRIVER_ACTIONS:
            if self.driver_binding is None or not operation.get("driver_identity"):
                raise StudioError("driver_service_identity", "Review the owner driver service identity first.", 409)
            state, receipt = await self.driver_binding.perform(DRIVER_ACTIONS[action], operation["driver_identity"])
        else:
            python = [sys.executable, *CHILD_FLAGS]
            argv = [*python, "-m", "dgx_monarch"]
            if self.config:
                argv.extend(["--config", str(self.config)])
            argv.extend(command_arguments(action, operation["parameters"]))
            if self.config is None and action in {"doctor", "reap.preview", "reap.apply"}:
                # The public CLI deliberately discovers defaults. Studio's
                # absent config is an explicit local-only selection instead.
                argv = [*python, "-m", "dgx_monarch.studio.local_commands", action]
                if action.startswith("reap."):
                    argv.extend(["--grace", str(operation["parameters"]["grace_s"])])
            proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                                                        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            retained = bytearray()
            total = 0
            try:
                self.store.update("operations", operation["id"], {"process_id": proc.pid})
            finally:
                # Even a failed PID checkpoint cannot abandon the spawned child.
                try:
                    if proc.stdout:
                        while chunk := await proc.stdout.read(8192):
                            total += len(chunk)
                            if len(retained) < 262144:
                                retained.extend(chunk[:262144 - len(retained)])
                finally:
                    code = await proc.wait()
            state = exit_state(code)
            receipt = {"source": "existing_cli", "exit_code": code, "output_bytes": total, "output_retained": False}
            from .management_reports import command_report
            report = command_report(action, bytes(retained))
            if report is not None:
                receipt["report"] = report
            if action == "version":
                receipt.update(version_receipt(bytes(retained)))
                if receipt["version"] is None and state == "succeeded":
                    state = "unknown"
            if action in {"status", "doctor"} and report is None:
                receipt["report"] = {"state": "unavailable", "reason": "Command did not produce one complete JSON report."}
        self.store.update("operations", operation["id"], {"receipt": receipt})
        return state, receipt
