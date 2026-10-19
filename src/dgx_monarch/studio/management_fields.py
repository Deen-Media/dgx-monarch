"""Closed, typed inputs for owner-authorized management adapters."""
from __future__ import annotations

from typing import Any

from .contracts import StudioError


def field(name: str, kind: str = "STRING", **values: Any) -> dict[str, Any]:
    return {"name": name, "type": kind, **values}


SETUP_FIELDS = [field("hosts", "HOSTS", required=True), field("client_ip", required=True),
                field("profile", "CHOICE", choices=["safe", "balanced", "advanced"], default="balanced"),
                field("fabric_profile"), field("python_bin", default="python3"), field("ssh_key"),
                field("comfy_dir"), field("transport_security", "CHOICE", choices=["trusted_fabric"], required=True),
                field("artifacts", "STRINGS", default=[]), field("install_service", "BOOLEAN", default=False),
                field("start_workers", "BOOLEAN", default=False), field("verify", "BOOLEAN", default=False),
                field("privileged_process_inspection", "BOOLEAN", default=False)]
FIELDS = {
    "launcher": [field("executable", required=True), field("destination", required=True)],
    "init": [entry for entry in SETUP_FIELDS if entry["name"] not in
             {"profile", "artifacts", "install_service", "start_workers", "verify", "privileged_process_inspection", "comfy_dir"}],
    "setup": SETUP_FIELDS,
    "sol-install": [field("allow_change", "STRINGS", default=[]), field("idle_maintenance", "BOOLEAN", default=False)],
    "sol-smoke": [field("idle_maintenance", "BOOLEAN", default=False), field("timeout_s", "INT", default=300, min=30, max=1800)],
    "uninstall": [field("remove_config", "BOOLEAN", default=False)],
    "update": [field("repo", required=True), field("target_ref", default="origin/HEAD"),
               field("stop_driver", "BOOLEAN", default=False), field("restore_driver", "BOOLEAN", default=False),
               field("privileged_process_inspection", "BOOLEAN", default=False)],
    "top.record": [field("samples", "INT", default=60, min=1, max=600),
                   field("interval_s", "FLOAT", default=2, min=0.1, max=60)],
    "top.replay": [field("recording_id", required=True)],
}
NO_CONFIG = frozenset({"top.record", "top.replay"})


def parse(action: str, values: dict[str, Any]) -> dict[str, Any]:
    import math

    specs = FIELDS[action]
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
            valid = valid and math.isfinite(value) and entry["min"] <= value <= entry["max"]
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
