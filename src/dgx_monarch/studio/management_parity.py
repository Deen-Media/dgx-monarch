"""Auditable CLI-to-Studio bindings, separate from runtime availability.

This closed registry intentionally does not infer coverage from a matching name.
A new parser option fails validation until its adapter or surface is reviewed.
"""
from __future__ import annotations

from typing import Any

COMMANDS = {
    "version": ["version"], "diagnostics": ["diagnostics", "diagnostics.capabilities"],
    "launcher": ["launcher"], "sol-install": ["sol-install"], "sol-smoke": ["sol-smoke"],
    "doctor": ["doctor", "doctor.repair"], "init": ["init", "service.install", "workers.up"], "setup": ["setup"],
    "up": ["workers.up"], "down": ["workers.down"], "reap": ["reap.apply", "reap.preview"],
    "top": ["top.record", "top.replay"],
    "gate": ["gate.last", "gate.inspect", "gate.repair.preview", "gate.repair.apply"],
    "restart": ["workers.restart"], "status": ["status"], "install-service": ["service.install"],
    "update": ["update"], "uninstall": ["uninstall"],
}

# Explicit parser option -> operation field. Multiple references mean the same
# choice is available for each mode, not that selecting one starts all modes.
PARAMETERS = {
    "diagnostics.duration_s": [("diagnostics", "duration_s")],
    "diagnostics.max_events": [("diagnostics", "max_events")],
    "diagnostics.max_bytes": [("diagnostics", "max_bytes")],
    "diagnostics.output": [("diagnostics", "output")],
    "launcher.executable": [("launcher", "executable")],
    "launcher.cluster_config": [("launcher", "cluster_config")],
    "launcher.destination": [("launcher", "destination")],
    "sol-install.plan": [("sol-install", "plan")],
    "sol-install.plan_digest": [("sol-install", "plan_digest")],
    "sol-install.allow_change": [("sol-install", "allow_change")],
    "sol-install.idle_maintenance": [("sol-install", "idle_maintenance")],
    "sol-smoke.idle_maintenance": [("sol-smoke", "idle_maintenance")],
    "sol-smoke.timeout_s": [("sol-smoke", "timeout_s")],
    "doctor.receipt": [("doctor.repair", "receipt")],
    "setup.host": [("setup", "hosts")], "setup.client_ip": [("setup", "client_ip")],
    "setup.profile": [("setup", "profile")], "setup.fabric_profile": [("setup", "fabric_profile")],
    "setup.python_bin": [("setup", "python_bin")], "setup.ssh_key": [("setup", "ssh_key")],
    "setup.comfy_dir": [("setup", "comfy_dir")], "setup.artifact": [("setup", "artifacts")],
    "setup.output": [("setup", "output")], "setup.receipt": [("setup", "receipt")],
    "setup.install_service": [("setup", "install_service")],
    "setup.start_worker_service": [("setup", "start_workers")],
    "setup.privileged_process_inspection": [("setup", "privileged_process_inspection")],
    "setup.verify": [("setup", "verify")],
    "up.no_sync": [("workers.up", "no_sync")], "restart.no_sync": [("workers.restart", "no_sync")],
    "reap.grace": [("reap.apply", "grace_s"), ("reap.preview", "grace_s")],
    "top.interval": [("top.record", "interval_s")], "top.record": [("top.record", "path")],
    "top.replay": [("top.replay", "path")],
    "gate.report_dir": [(action, "report_dir") for action in COMMANDS["gate"]],
    "gate.timeout": [(action, "timeout_s") for action in COMMANDS["gate"]],
    "update.target": [("update", "target_ref")],
    "update.privileged_process_inspection": [("update", "privileged_process_inspection")],
    "update.receipt": [("update", "receipt")],
}
SELECTORS = {
    "diagnostics.capabilities": (["diagnostics.capabilities"], "Read the same exact-runtime capability report without connecting."),
    "launcher.apply": (["launcher"], "Apply the exact reviewed digest with explicit confirmation."),
    "launcher.check": (["launcher"], "Preview without apply is the default; no launcher is written."),
    "sol-install.apply": (["sol-install"], "Apply the exact fleet plan after digest review and explicit confirmation."),
    "doctor.repair": (["doctor.repair"], "Select the allowlisted permission repair and subsequent Doctor checks."),
    "doctor.yes": (["doctor.repair"], "Explicit confirmed=true on the exact review replaces the terminal prompt."),
    "setup.apply": (["setup"], "Apply the exact reviewed setup plan; preview itself never applies."),
    "setup.yes": (["setup"], "Explicit confirmed=true on the exact review replaces the terminal prompt."),
    "setup.acknowledge_trusted_fabric": (["setup"], "Required transport_security=trusted_fabric is explicit consent; there is no implicit default."),
    "reap.dry_run": (["reap.preview"], "Execute the same dry-run orphan inspection without sending signals."),
    "gate.last": (["gate.last"], "Review the latest history prompt transformed into the same Identity Gate graph."),
    "gate.list": (["gate.inspect"], "Read current trusted gate verdicts from the selected ledger."),
    "gate.repair": (["gate.repair.preview"], "Inspect orphan RETESTING contexts with no writes."),
    "gate.apply": (["gate.repair.apply"], "Apply the reviewed append-only repair with explicit confirmation."),
    "update.verify": (["update"], "mode=verified, the Studio default, selects the verified transaction; mode=legacy selects "
                      "the original in-place update, which the CLI runs without --verify. Both modes need --config."),
    "update.yes": (["update"], "Explicit confirmation replaces the terminal prompt after exact source review."),
}
PRESENTATION = {
    "doctor.json": "Doctor check rows and readiness are returned as sanitized structured JSON in receipt.report.",
    "status.json": "Host rows, mesh and readiness are returned as sanitized structured JSON in receipt.report.",
    "setup.json": "Setup plan and outcome are already structured JSON; apply still requires explicit confirmation.",
    "top.theme": "Terminal palette selection is a presentation-only option; Studio uses its shared semantic palette.",
}
CONTEXT = {
    "global.config": "Owner selects --config at companion launch, or dgxm studio passes the file dgxm itself reads; "
                     "each review binds the selected file identity. The companion never discovers another config.",
    **{f"{command}.host": "Owner selects --driver at companion launch, or dgxm studio passes [studio] driver from "
                          "cluster.toml; all authenticated requests use that fixed origin. Studio never probes for drivers."
       for command in ("diagnostics", "top", "gate")},
    "update.host": "Update binds the driver through --driver-service, then stops it if it runs and checks it again. "
                   "Verified mode passes no host, since with --verify an explicit --host always refuses; legacy mode "
                   "checks an http --driver origin first. Both backends also probe the local ComfyUI ports.",
}

# The studio tree starts, stops, opens or reaches the process that serves
# Studio. A page cannot run these for itself, so each row is waived here by its
# full path, help rows included, and a new studio flag fails until reviewed.
COMPANION_REASON = ("Starts, stops, opens or reaches the companion process that serves Studio; "
                    "the page cannot run these for itself.")
COMPANION_COMMANDS = ("studio", "studio.up", "studio.down", "studio.open", "studio.token", "studio.tunnel")
COMPANION_OPTIONS = (
    "studio.help", "studio.config", "studio.driver", "studio.state_dir", "studio.allow_management", "studio.no_open",
    "studio.up.help", "studio.up.config", "studio.up.allow_management",
    "studio.down.help", "studio.open.help",
    "studio.token.help", "studio.token.value", "studio.token.rotate",
    "studio.tunnel.help", "studio.tunnel.config", "studio.tunnel.host", "studio.tunnel.port", "studio.tunnel.run",
)


def bindings() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for command, actions in COMMANDS.items():
        result[f"cli.command.{command}"] = {"kind": "adapter", "actions": actions,
            "equivalence": "Typed action selection; availability and live backend guards are checked separately."}
    for option, parameters in PARAMETERS.items():
        result[f"cli.option.{option}"] = {"kind": "parameter", "parameters": [list(item) for item in parameters],
                                         "actions": list(dict.fromkeys(action for action, _ in parameters))}
    for option, (actions, meaning) in SELECTORS.items():
        result[f"cli.option.{option}"] = {"kind": "selector", "actions": actions, "equivalence": meaning}
    for kind, group in (("presentation", PRESENTATION), ("context", CONTEXT)):
        for option, meaning in group.items():
            result[f"cli.option.{option}"] = {"kind": kind, "equivalence": meaning}
    for command in ("global", *COMMANDS):
        result[f"cli.option.{command}.help"] = {"kind": "presentation", "equivalence":
            "Capability descriptions, typed parameters and availability reasons replace terminal argparse help."}
    for key in (*(f"cli.command.{command}" for command in COMPANION_COMMANDS),
                *(f"cli.option.{option}" for option in COMPANION_OPTIONS)):
        result[key] = {"kind": "companion", "reason": COMPANION_REASON}
    return result


def validate_bindings(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from .management_fields import COMMAND_FIELDS, FIELDS, FIXED_COMMANDS

    mapping = bindings()
    expected = {row["id"] for row in rows}
    if expected != mapping.keys():
        raise ValueError(f"CLI coverage changed: missing={sorted(expected - mapping.keys())}; stale={sorted(mapping.keys() - expected)}")
    available = set(FIXED_COMMANDS) | FIELDS.keys()
    for row in rows:
        item = mapping[row["id"]]
        if set(item.get("actions", [])) - available:
            raise ValueError(f"Missing executable adapter: {row['id']}")
        for action, field in item.get("parameters", []):
            specs = {entry["name"]: entry for entry in FIELDS.get(action, COMMAND_FIELDS.get(action, []))}
            if field not in specs:
                raise ValueError(f"Missing typed parameter: {row['id']} -> {action}.{field}")
            spec = specs[field]
            # Required wizard answers replace CLI prompts. Optional strings use
            # the wire's empty string in place of argparse None, with the same
            # backend fallback. All other advertised defaults must be exact.
            default = spec.get("default", "" if row["default"] == "" else None)
            if not spec.get("required") and default != row["default"]:
                raise ValueError(f"CLI default differs: {row['id']}")
            if row["choices"] is not None and list(spec.get("choices", [])) != row["choices"]:
                raise ValueError(f"CLI choices differ: {row['id']}")
    return mapping
