"""Structured local reports without forwarding arbitrary command output."""
from __future__ import annotations

import json
from typing import Any

from ..cli import gate_repair
from ..cli.receipt_sanitize import sanitize_note
from ..gate_audit import trust_rows
from ..gate_ledger import GateLedger
from .contracts import StudioError, digest
from .management_config import absolute


def safe_report(value: Any, depth: int = 0) -> Any:
    """Bound report shape and use the existing disclosure-safe note sanitizer."""
    if depth > 10:
        return "[truncated]"
    if isinstance(value, str):
        return sanitize_note(value)
    if value is None or type(value) in {bool, int, float}:
        return value
    if isinstance(value, list):
        return [safe_report(item, depth + 1) for item in value[:4096]]
    if isinstance(value, dict):
        return {sanitize_note(str(key)): safe_report(item, depth + 1) for key, item in list(value.items())[:256]}
    return None


def command_report(action: str, retained: bytes) -> dict[str, Any] | None:
    if action not in {"status", "doctor"}:
        return None
    try:
        value = json.loads(retained)
    except (ValueError, UnicodeDecodeError):
        return None
    keys = {"hosts", "mesh", "readiness"} if action == "status" else {"checks", "summary", "readiness"}
    return safe_report({key: value[key] for key in keys if key in value}) if isinstance(value, dict) else None


def gate_plan(action: str, parameters: dict[str, Any]) -> dict[str, Any]:
    ledger = GateLedger(str(absolute(parameters["report_dir"])))
    entries, damage = ledger.entries_with_integrity()
    if action == "gate.inspect":
        latest = {entry.get("key"): entry for entry in trust_rows(ledger.entries())}
        return {"rows": safe_report(list(latest.values())), "damage_line": damage, "ledger_digest": digest(entries)}
    orphans, inert = gate_repair.find_orphans(entries)
    return {"rows": safe_report([{"line": line, "entry": row, "contexts": contexts} for line, row, contexts in orphans]),
            "inert_rows": inert, "damage_line": damage, "ledger_digest": digest(entries),
            "scope": "Append only INCONCLUSIVE terminal rows; never authorize a previously refused context."}


def gate_execute(action: str, parameters: dict[str, Any], reviewed: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    current = gate_plan(action, parameters)
    if action == "gate.repair.apply":
        if current != reviewed:
            raise StudioError("gate_changed", "Gate ledger changed since review; prepare another repair.", 409)
        code = gate_repair.repair(str(absolute(parameters["report_dir"])), apply=True)
        return ("succeeded" if code == 0 else "unknown", {"source": "gate_ledger", "exit_code": code,
                "before": reviewed, "after": gate_plan(action, parameters)})
    return "succeeded", {"source": "gate_ledger", "report": current}
