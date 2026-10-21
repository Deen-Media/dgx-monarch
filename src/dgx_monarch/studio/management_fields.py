"""Closed, typed inputs for owner-authorized management adapters."""
from __future__ import annotations

from typing import Any

from .contracts import StudioError


def field(name: str, kind: str = "STRING", **values: Any) -> dict[str, Any]:
    return {"name": name, "type": kind, **values}


FIXED_COMMANDS: dict[str, tuple[str, ...]] = {
    "version": ("version",), "status": ("status", "--json"), "doctor": ("doctor", "--json"),
    "workers.up": ("up",), "workers.down": ("down",),
    "workers.restart": ("restart",), "reap.preview": ("reap", "--dry-run"),
    "reap.apply": ("reap",), "service.install": ("install-service",),
    "doctor.repair": ("doctor", "--repair", "--yes"),
    "gate.inspect": ("gate", "--list"), "gate.repair.preview": ("gate", "--repair"),
    "gate.repair.apply": ("gate", "--repair", "--apply"),
}

SETUP_FIELDS = [field("hosts", "HOSTS", required=True), field("client_ip", required=True),
                field("profile", "CHOICE", choices=["safe", "balanced", "advanced"], default="balanced"),
                field("fabric_profile"), field("python_bin", default="python3"), field("ssh_key"),
                field("comfy_dir"), field("transport_security", "CHOICE", choices=["trusted_fabric"], required=True),
                field("artifacts", "STRINGS", default=[]), field("install_service", "BOOLEAN", default=False),
                field("start_workers", "BOOLEAN", default=False), field("verify", "BOOLEAN", default=False),
                field("privileged_process_inspection", "BOOLEAN", default=False)]
FIELDS = {
    "launcher": [field("executable", required=True), field("destination", default="~/.local/bin/dgxm"), field("cluster_config")],
    "init": [entry for entry in SETUP_FIELDS if entry["name"] not in
             {"profile", "artifacts", "install_service", "start_workers", "verify", "privileged_process_inspection", "comfy_dir"}],
    "setup": [*SETUP_FIELDS, field("output"), field("receipt")],
    "sol-install": [field("allow_change", "STRINGS", default=[]), field("idle_maintenance", "BOOLEAN", default=False),
                    field("plan"), field("plan_digest")],
    "sol-smoke": [field("idle_maintenance", "BOOLEAN", default=False), field("timeout_s", "INT", default=300, min=30, max=1800)],
    "uninstall": [field("remove_config", "BOOLEAN", default=False)],
    "update": [field("repo", required=True), field("mode", "CHOICE", choices=["verified", "legacy"], default="verified"),
               field("target_ref", default="origin/HEAD"), field("receipt"),
               field("stop_driver", "BOOLEAN", default=False), field("restore_driver", "BOOLEAN", default=False),
               field("privileged_process_inspection", "BOOLEAN", default=False)],
    "top.record": [field("samples", "INT", default=60, min=1, max=600), field("path"),
                   field("interval_s", "FLOAT", default=1, min=0.1, max=3600)],
    "top.replay": [field("recording_id"), field("path")],
    "diagnostics": [field("duration_s", "FLOAT", default=30, min=0.1, max=300),
                    field("max_events", "INT", default=4096, min=1, max=20000),
                    field("max_bytes", "INT", default=1048576, min=65536, max=8388608), field("output")],
    "diagnostics.capabilities": [],
    "gate.last": [field("report_dir", default="~/ComfyUI/output"), field("timeout_s", "INT", default=1800, min=1)],
}
COMMAND_FIELDS = {
    "doctor.repair": [field("receipt")],
    "workers.up": [field("no_sync", "BOOLEAN", default=False)],
    "workers.restart": [field("no_sync", "BOOLEAN", default=False)],
    "reap.preview": [field("grace_s", "FLOAT", default=0.0, min=0)],
    "reap.apply": [field("grace_s", "FLOAT", default=0.0, min=0)],
    **{action: [field("report_dir", default="~/ComfyUI/output"), field("timeout_s", "INT", default=1800, min=1)]
       for action in ("gate.inspect", "gate.repair.preview", "gate.repair.apply")},
}
FIELDS.update({action: COMMAND_FIELDS[action] for action in ("gate.inspect", "gate.repair.preview", "gate.repair.apply")})
# Update is absent on purpose: both modes need the selected cluster configuration
# (the legacy release-slot check and Worker restart read it), so availability
# names the missing --config instead of offering a config-free update.
NO_CONFIG = frozenset({"top.record", "top.replay", "diagnostics", "diagnostics.capabilities", "gate.last",
                       "gate.inspect", "gate.repair.preview", "gate.repair.apply", "doctor", "reap.preview", "reap.apply"})


def parse(action: str, values: dict[str, Any]) -> dict[str, Any]:
    import math

    specs = FIELDS.get(action, COMMAND_FIELDS.get(action, []))
    if set(values) - {entry["name"] for entry in specs}:
        raise StudioError("operation_fields", "Unsupported operation parameter.")
    result = {}
    for entry in specs:
        name, kind = entry["name"], entry["type"]
        value = values.get(name, entry.get("default", ""))
        valid = True
        if kind == "BOOLEAN":
            valid = type(value) is bool
        elif kind in {"INT", "FLOAT"}:
            valid = (type(value) is int if kind == "INT" else type(value) in {int, float})
            valid = valid and math.isfinite(value) and entry.get("min", -math.inf) <= value <= entry.get("max", math.inf)
        elif kind == "STRINGS":
            valid = isinstance(value, list) and len(value) <= 64 and all(isinstance(v, str) and len(v) <= 512 for v in value)
        elif kind == "HOSTS":
            valid = isinstance(value, list) and 1 <= len(value) <= 64 and all(isinstance(v, dict) for v in value)
        else:
            valid = isinstance(value, str) and len(value) <= 4096 and not any(ord(c) < 32 for c in value)
            if valid and kind == "CHOICE":
                valid = value in entry["choices"]
            if valid and entry.get("required"):
                valid = bool(value)
        if not valid:
            raise StudioError("operation_fields", f"Invalid {name} parameter.")
        result[name] = value
    return result
