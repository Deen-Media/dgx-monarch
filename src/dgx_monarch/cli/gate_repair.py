"""Close identity-gate RETESTING rows that have no terminal result.

The identity check writes a durable RETESTING row before its first side effect,
then records a terminal verdict for each capability context. An interrupted
check can leave contexts blocked indefinitely. With ``--apply``, this command
appends INCONCLUSIVE rows for those unfinished contexts.

Both RETESTING and INCONCLUSIVE resolve to ``inconclusive``; repair does not
allow previously refused work. It only appends rows, without rewriting,
deleting or truncating the ledger.

Run only when no identity check is active. Repair reads open rows before
appending results; a concurrent PASS could be superseded by a later repair row
and require another check.
"""
from __future__ import annotations

import sys
from typing import Any

from .. import __version__
from ..gate_ledger import GATE_PROTOCOL_VERSION, GateLedger

TERMINAL_VERDICTS = ("PASS", "FAIL", "INCONCLUSIVE")
REPAIR_REASON = "repaired: ceremony ended without a terminal row"
_APPLY_ERRORS = (OSError, RuntimeError, TypeError, ValueError, RecursionError)


def _is_inert(row: dict) -> bool:
    """Return whether the row belongs to another protocol or package version.

    These rows cannot affect current decisions, so they need no repair entry.
    """
    return (row.get("gate_protocol") != GATE_PROTOCOL_VERSION
            or row.get("dgx_monarch") != __version__)


def _has_successor(entries: list[tuple[int, dict]], line: int, row: dict,
                   context: str, artifacts: str) -> bool:
    """Return whether a later terminal result closed this exact context.

    Ignore ``dgx_source``: a source change invalidates reuse of a result but does
    not reopen its completed transaction. A later RETESTING row starts another
    transaction and does not close this one.
    """
    return any(
        entry.get("key") == row.get("key")
        and GateLedger._artifact_matches(entry, artifacts)
        and entry.get("gate_protocol") == row.get("gate_protocol")
        and entry.get("dgx_monarch") == row.get("dgx_monarch")
        and entry.get("capability_context") == context
        and entry.get("verdict") in TERMINAL_VERDICTS
        for later, entry in entries if later > line
    )


def find_orphans(entries: list[tuple[int, dict]]) -> tuple[list[tuple[int, dict, list[str]]], int]:
    """Every open RETESTING row with its unanswered contexts, plus inert count."""
    orphans: list[tuple[int, dict, list[str]]] = []
    inert = 0
    for line, row in entries:
        if row.get("verdict") != "RETESTING":
            continue
        if _is_inert(row):
            inert += 1
            continue
        artifacts = str(row.get("artifacts") or "")
        open_contexts = [
            context for context in row.get("blocked_contexts") or []
            if not _has_successor(entries, line, row, context, artifacts)
        ]
        if open_contexts:
            orphans.append((line, row, open_contexts))
    return orphans, inert


def _describe(line: int, row: dict, contexts: list[str]) -> str:
    key = str(row.get("key", "unknown"))
    return (f"line {line}  {row.get('time', 'unknown time')}  "
            f"key={key[:16]}{'..' if len(key) > 16 else ''}  "
            f"model={row.get('model', 'unknown')}  "
            f"protocol={row.get('gate_protocol')}  "
            f"version={row.get('dgx_monarch')}  "
            f"open_contexts={len(contexts)}")


def _repair_detail(line: int, row: dict) -> dict[str, Any]:
    return {
        "model": row.get("model", "unknown"),
        "loras": row.get("loras", 0),
        "origin": "gate_repair",
        "run_id": "",
        "inconclusive_reasons": [REPAIR_REASON],
        "repaired_from_time": row.get("time"),
        "repaired_from_line": line,
    }


def repair(report_dir: str, *, apply: bool) -> int:
    """List, or with ``apply`` close, every orphaned RETESTING transaction."""
    ledger = GateLedger(report_dir)
    entries, last_damage_line = ledger.entries_with_integrity()
    if last_damage_line:
        print(f"note: the newest damaged or invalid line is {last_damage_line}; "
              "every intact line is still read")
    orphans, inert = find_orphans(entries)
    print("this does not change what is refused: an open RETESTING row already "
          "denies, and closing it only ends the transaction")
    for line, row, contexts in orphans:
        print(_describe(line, row, contexts))
    total = sum(len(contexts) for _line, _row, contexts in orphans)
    print(f"{total} open context(s) on {len(orphans)} RETESTING row(s) have no "
          "terminal successor")
    print(f"{inert} inert rows skipped (protocol/version mismatch)")
    if not apply:
        return 1 if orphans else 0
    written = 0
    try:
        for line, row, contexts in orphans:
            detail = _repair_detail(line, row)
            for context in contexts:
                if not ledger.record(row["key"], row["artifacts"], row["comfy"],
                                     "INCONCLUSIVE", detail, context):
                    print(f"error: could not append the terminal row for "
                          f"{_describe(line, row, contexts)}", file=sys.stderr)
                    return 2
                written += 1
    except _APPLY_ERRORS as exc:
        # `record` builds its entry outside its own write guard, and the build
        # reads the source manifest; begin_retest_required catches this same
        # tuple for that reason.
        print(f"error: the repair could not be written: {exc}", file=sys.stderr)
        return 2
    print(f"appended {written} terminal INCONCLUSIVE row(s)")
    return 0
