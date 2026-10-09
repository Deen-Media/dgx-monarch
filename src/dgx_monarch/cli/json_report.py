"""Machine-readable output for ``dgxm doctor`` and ``dgxm status``.

JSON mode runs the same checks as prose mode, keeps their prose off stdout, and
prints the same rows as JSON, so the checks and the exit code do not change.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from .. import operator_readiness

T = TypeVar("T")

# Stable host-row keys, each present even when its probe gave a blank.
_HOST_FIELDS = ("host", "loop", "port", "address", "mode")

# Display badge to stable machine token (`doctor_state`).
_DOCTOR_STATES = {"ok": "ok", "WARN": "warn", "FAIL": "fail"}


def subcommand(
    sub: argparse._SubParsersAction, name: str, help_text: str
) -> argparse.ArgumentParser:
    """Register a subcommand that speaks both prose and `--json`."""
    parser = sub.add_parser(name, help=help_text)
    parser.add_argument(
        "--json", action="store_true",
        help="emit the rows as JSON on stdout instead of the prose report")
    return parser


def quietly(work: Callable[[], T]) -> T:
    """Run `work` with its prose kept off stdout."""
    with contextlib.redirect_stdout(io.StringIO()):
        return work()


def emit(payload: Mapping[str, object]) -> None:
    print(json.dumps(payload, indent=2))


def doctor_state(badge: object) -> str:
    """Return a stable machine token for a doctor badge.

    Unknown badges are lowercased. Tests cover known tokens as an API contract.
    """
    text = str(badge).strip()
    return _DOCTOR_STATES.get(text, text.lower())


def doctor_payload(
    rows: Sequence[Mapping[str, object]], failures: int
) -> dict[str, object]:
    """Doctor rows and the summary counts the prose report prints.

    The caller supplies the failure count the exit code uses.
    """
    checks = [
        {
            "check": str(row.get("name", "")),
            "state": doctor_state(row.get("status", "")),
            "detail": str(row.get("detail", "")),
        }
        for row in rows
    ]
    return {
        "checks": checks,
        "summary": {"checks": len(checks), "failures": failures},
        "readiness": operator_readiness.readiness_from_doctor_rows(rows),
    }


def status_payload(
    hosts: Sequence[Mapping[str, object]],
    mesh_state: str,
    mesh_detail: str,
    *,
    mesh_block: object = None,
) -> dict[str, object]:
    """The per-host rows and mesh row `dgxm status` prints, plus readiness.

    Status keeps only the `mesh` block of the driver's telemetry and passes no
    render state, so its overall readiness reads `unknown` at best
    (docs/CONCEPTS.md, Readiness).
    """
    host_rows = [
        {field: str(row.get(field, "")) for field in _HOST_FIELDS}
        for row in hosts
    ]
    normalized_mesh = (
        mesh_block
        if mesh_block is not None
        else {"state": mesh_state if mesh_state in {"ok", "idle", "busy", "dirty", "unknown"} else "unknown"}
    )
    return {
        "hosts": host_rows,
        "mesh": {"state": mesh_state, "detail": mesh_detail},
        "readiness": operator_readiness.readiness_from_telemetry(
            host_rows, normalized_mesh, None
        ),
    }
