"""Explicit Sol maintenance through the existing staged-plan and idle guards."""
from __future__ import annotations

import re
import subprocess
import uuid
from typing import Any

from ..cli import sol_install, sol_smoke
from ..cli.sol_install_payload import canonical, protected
from ..config import ClusterConfig
from .contracts import StudioError


def allowed_names(values: list[str]) -> list[str]:
    names = sorted({canonical(name) for name in values})
    if any(not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) or protected(name) for name in names):
        raise StudioError("sol_packages", "Protected runtime packages and invalid names cannot be changed.")
    return names


def read_plans(config: ClusterConfig, token: str, allowed: list[str], action: str) -> dict[str, Any]:
    plans = {}
    for host in config.hosts:
        result = sol_install._host(config, host, action, token, allowed)
        if result is None or result.get("status") != "ok":
            raise StudioError("sol_plan", "Sol plan is incomplete; private staging is retained without package apply.", 409)
        plans[host.name] = result["plan"]
    return plans


def preview(action: str, config: ClusterConfig, parameters: dict[str, Any]) -> dict[str, Any]:
    if not config.hosts or len({host.name for host in config.hosts}) != len(config.hosts):
        raise StudioError("sol_hosts", "Sol maintenance requires unique configured hosts.")
    if action == "sol-smoke":
        if any(host.gpus != 1 for host in config.hosts):
            raise StudioError("sol_hosts", "Sol smoke requires single-GPU hosts.")
        return {"hosts": len(config.hosts), "scope": "Explicit BF16 dense kernel control; no model fidelity evidence.",
                "timeout_s": parameters["timeout_s"]}
    allowed = allowed_names(parameters["allow_change"])
    token = uuid.uuid4().hex
    plans = {}
    rows = []
    for index, host in enumerate(config.hosts):
        result = sol_install._host(config, host, "plan", token, allowed)
        if result is not None and result.get("status") == "ok":
            plans[host.name] = result["plan"]
            rows.append({"ordinal": index, "plan": result["plan"]})
        else:
            rows.append({"ordinal": index, "error": sol_install.refusal_reason(result)})
    if len(plans) != len(config.hosts):
        return {"token": token, "ready": False, "allowed": allowed, "hosts": rows,
                "recovery": "Incomplete private staging retained; no package apply attempted."}
    return {"token": token, "fleet_digest": sol_install.fleet_digest(config, token, allowed, plans),
            "ready": True, "allowed": allowed, "hosts": rows}


def execute(action: str, config: ClusterConfig, parameters: dict[str, Any], reviewed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if parameters["idle_maintenance"] is not True:
        raise StudioError("idle_maintenance", "Explicit idle-maintenance authority is required; close ComfyUI and settle actors first.", 409)
    if reviewed.get("ready") is False:
        raise StudioError("sol_plan", "The reviewed plan is incomplete and cannot be applied.", 409)
    rows: list[dict[str, Any]] = []
    if action == "sol-smoke":
        if preview(action, config, parameters) != reviewed:
            raise StudioError("operation_changed", "Sol smoke configuration changed.", 409)
        script = sol_smoke.smoke_script(config.python_bin)
        for index, host in enumerate(config.hosts):
            try:
                result = sol_smoke.run_on_host(config, host, script, timeout=parameters["timeout_s"], require_known_locality=True)
                if result.returncode:
                    return "unknown", {"source": "sol_smoke", "hosts": rows, "incomplete_host": index,
                                       "reason": "Kernel guard or execution failed; remaining hosts were not tested."}
                rows.append({"ordinal": index, "result": sol_smoke._parse_result(result.stdout)})
            except (OSError, subprocess.TimeoutExpired, ValueError):
                return "unknown", {"source": "sol_smoke", "hosts": rows, "incomplete_host": index}
        return "succeeded", {"source": "sol_smoke", "hosts": rows, "scope": reviewed["scope"]}
    token, allowed = reviewed["token"], allowed_names(parameters["allow_change"])
    plans = read_plans(config, token, allowed, "read")
    if sol_install.fleet_digest(config, token, allowed, plans) != reviewed["fleet_digest"]:
        raise StudioError("operation_changed", "The exact staged Sol plan changed after review.", 409)
    for index, host in enumerate(config.hosts):
        applied = sol_install._host(config, host, "apply", token, allowed, plans[host.name]["digest"])
        if applied is None or applied.get("status") != "ok":
            return "unknown", {"source": "sol_install", "token": token, "hosts": rows,
                               "incomplete_host": index, "recovery": "No restart or rollback attempted; staging is retained."}
        if applied["plan"] != plans[host.name]:
            return "unknown", {"source": "sol_install", "token": token, "hosts": rows, "incomplete_host": index}
        rows.append({"ordinal": index, "status": "verified"})
    return "succeeded", {"source": "sol_install", "token": token, "fleet_digest": reviewed["fleet_digest"],
                         "hosts": rows, "kernel_execution": "not_run"}
