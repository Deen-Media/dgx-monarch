"""Fixed management operations with persisted review, confirmation and receipts."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from .. import __version__
from .comfy import ComfyClient
from .contracts import StudioError, digest, identifier, timestamp
from .driver_service import DriverService
from .management_fields import FIELDS, NO_CONFIG
from .store import Store

# Each command and every flag here is owned by Monarch. Browser JSON never becomes argv.
COMMANDS: dict[str, tuple[str, ...]] = {
    "version": ("version",), "status": ("status", "--json"), "doctor": ("doctor", "--json"),
    "workers.up": ("up", "--no-sync"), "workers.down": ("down",),
    "workers.restart": ("restart", "--no-sync"), "reap.preview": ("reap", "--dry-run"),
    "reap.apply": ("reap",), "service.install": ("install-service",),
    "doctor.repair": ("doctor", "--repair", "--yes"),
    "gate.inspect": ("gate", "--list"), "gate.repair.preview": ("gate", "--repair"),
    "gate.repair.apply": ("gate", "--repair", "--apply"),
}
DRIVER_ACTIONS = {"driver.start": "start", "driver.stop": "stop", "driver.restart": "restart"}
HTTP_ACTIONS = {"actors.recycle": "/dgxm/recycle", "driver.free": "/free", "consent.apply": "/dgxm/consent"}
READ_ONLY = frozenset({"version", "status", "doctor", "reap.preview", "gate.inspect", "gate.repair.preview", "top.replay"})


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
                 allow_management: bool = False, driver_service: str | None = None):
        self.store, self.client = store, client
        self.config = config
        self.allow_management = allow_management
        self.driver_service = driver_service
        self.driver_binding = DriverService(driver_service) if driver_service else None
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
            if not reason and (action in COMMANDS or action in FIELDS) and action not in NO_CONFIG | {"version"} and not self.config:
                reason = "Select an existing cluster configuration when starting the companion."
            fields = FIELDS.get(action, [])
            if action == "driver.free":
                fields = [{"name": "unload_models", "type": "BOOLEAN", "default": True},
                          {"name": "free_memory", "type": "BOOLEAN", "default": True}]
            if action == "consent.apply":
                fields = [{"name": "key", "type": "STRING", "required": True},
                          {"name": "id", "type": "STRING", "required": True}]
            rows.append({"id": action, "label": action.replace(".", " · "), "available": reason is None,
                         "reason": reason, "parameters": fields, "requires_confirmation": action not in READ_ONLY,
                         "implementation": "typed_adapter" if action in FIELDS else "adapter"})
        return rows

    def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        action, parameters = request.get("action"), request.get("parameters", {})
        row = next((row for row in self.capabilities() if row["id"] == action), None)
        if row is None:
            raise StudioError("operation", "Unknown operation.")
        if not isinstance(parameters, dict):
            raise StudioError("operation_fields", "Operation parameters must be an object.")
        allowed = {field["name"] for field in row["parameters"]}
        if set(parameters) - allowed:
            raise StudioError("operation_fields", "Unsupported operation parameter.")
        if action == "driver.free":
            parameters = {"unload_models": parameters.get("unload_models", True), "free_memory": parameters.get("free_memory", True)}
            if any(type(value) is not bool for value in parameters.values()):
                raise StudioError("operation_fields", "Memory controls must be booleans.")
        if action == "consent.apply":
            if set(parameters) != {"key", "id"} or any(not isinstance(v, str) or len(v) > 512 for v in parameters.values()):
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
        if action in FIELDS and row["available"]:
            try:
                parameters, adapter_plan = self.management.preview(action, parameters)
                if adapter_plan.get("ready") is False:
                    row = {**row, "available": False, "reason": "Backend plan is blocked; inspect its review evidence."}
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
                                      {"operation_id": operation_id, **request})
        if existing is not None:
            return existing
        plan = self.store.get("operations", operation_id)
        current = next((row for row in self.capabilities() if row["id"] == plan["action"]), None)
        if not current or not current["available"]:
            raise StudioError("operation_unavailable", current["reason"] if current else "Operation removed.", 409)
        if plan["driver_service"] != self.driver_service:
            raise StudioError("service_changed", "The selected driver service changed; review again.", 409)
        if not plan["available"]:
            raise StudioError("operation_unavailable", plan["reason"], 409)
        if request.get("digest") != plan["digest"] or plan["expires_at"] < timestamp():
            raise StudioError("stale_operation", "Operation review is stale or has a different digest.", 409)
        if plan["requires_confirmation"] and request.get("confirmed") is not True:
            raise StudioError("confirmation", "Review and explicitly confirm this exact operation.", 409)
        if fingerprint(self.config) != plan["config_fingerprint"]:
            raise StudioError("config_changed", "Configuration changed after review; prepare another operation.", 409)
        value = {**plan, "id": identifier(), "review_id": operation_id, "state": "applying"}
        value, created = self.store.reserve("operations", request.get("idempotency_key"),
                                           {"operation_id": operation_id, **request}, value)
        if created:
            task = asyncio.create_task(self.execute(value))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        return value

    async def execute(self, operation: dict[str, Any]) -> None:
        async with (asyncio.Lock() if operation["action"].startswith("top.") else self.execution_lock):
            action = operation["action"]
            try:
                current = next((row for row in self.capabilities() if row["id"] == action), None)
                if not current or not current["available"] or operation["driver_service"] != self.driver_service:
                    raise StudioError("operation_unavailable", "Operation authority changed after review.", 409)
                if fingerprint(self.config) != operation["config_fingerprint"]:
                    raise StudioError("config_changed", "Configuration changed while awaiting the operation lock.", 409)
                if action in FIELDS:
                    state, receipt = await self.management.execute(operation)
                elif action in DRIVER_ACTIONS:
                    if self.driver_binding is None or not operation.get("driver_identity"):
                        raise StudioError("driver_service_identity", "Review the owner driver service identity first.", 409)
                    state, receipt = await self.driver_binding.perform(DRIVER_ACTIONS[action], operation["driver_identity"])
                elif action in HTTP_ACTIONS:
                    parameters = dict(operation["parameters"])
                    if action == "consent.apply":
                        parameters["action"] = "accept"
                    result = await self.client.request("POST", HTTP_ACTIONS[action], parameters)
                    # The driver may return accepted, busy or unknown. Retain its actual result.
                    state = "succeeded" if isinstance(result, dict) and result.get("ok") is True else "unknown"
                    if action == "driver.free":
                        state = "unknown"  # /free acknowledges queuing, not completed memory reclamation.
                    receipt = {"source": "driver", "result": result}
                else:
                    argv = [sys.executable, "-m", "dgx_monarch"]
                    if self.config:
                        argv.extend(["--config", str(self.config)])
                    argv.extend(COMMANDS[action])
                    proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.DEVNULL,
                                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                    self.store.update("operations", operation["id"], {"process_id": proc.pid})
                    retained = bytearray()
                    total = 0
                    if proc.stdout:
                        while chunk := await proc.stdout.read(8192):
                            total += len(chunk)
                            if len(retained) < 262144:
                                retained.extend(chunk[:262144 - len(retained)])
                    code = await proc.wait()
                    # Arbitrary stdout is not returned to browsers: existing command output is not a disclosure-safe receipt.
                    state = "succeeded" if code == 0 else "unknown" if code == 3 else "failed"
                    receipt = {"source": "existing_cli" if action in COMMANDS else "owner_service",
                               "exit_code": code, "output_bytes": total, "output_retained": False}
                    if action == "version":
                        receipt["version"] = __version__
                    if action in {"status", "doctor"}:
                        import json
                        try:
                            receipt["report"] = json.loads(retained)
                        except (ValueError, UnicodeDecodeError):
                            receipt["report"] = {"state": "unavailable", "reason": "Command did not produce one complete JSON report."}
                self.store.update("operations", operation["id"], {"state": state, "receipt": receipt,
                                  "completed_at": timestamp()})
            except asyncio.CancelledError:
                retained = self.store.get("operations", operation["id"]).get("receipt", {})
                self.store.update("operations", operation["id"], {"state": "unknown", "receipt": {**retained,
                                  "error": "operation_interrupted", "message": "Reconcile retained receipts before any retry."}})
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                retained = self.store.get("operations", operation["id"]).get("receipt", {})
                self.store.update("operations", operation["id"], {"state": "unknown", "receipt": {**retained,
                    "error": exc.code if isinstance(exc, StudioError) else "operation_io_failed",
                    "message": "Operation outcome requires reconciliation; automatic retry is disabled."}})
