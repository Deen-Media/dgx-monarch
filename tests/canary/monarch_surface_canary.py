#!/usr/bin/env python3
"""Snapshot and compare the torchmonarch surface used by dgx-monarch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from dgx_monarch.monarch_surface import (  # noqa: E402
    collect_snapshot,
)
from dgx_monarch.monarch_surface_report import (  # noqa: E402
    compare_snapshots,
    render_comparison_markdown,
    snapshot_blockers,
)


def _write_text(path: str, text: str) -> None:
    if path == "-":
        sys.stdout.write(text)
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def _write_json(path: str, payload: Any) -> None:
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"snapshot {path} is not a JSON object")
    return payload


def _snapshot(args: argparse.Namespace) -> int:
    snapshot = collect_snapshot()
    _write_json(args.output, snapshot)
    blockers = snapshot_blockers(snapshot)
    for blocker in blockers:
        print(f"torchmonarch surface blocker: {blocker}", file=sys.stderr)
    return 1 if blockers else 0


def _compare(args: argparse.Namespace) -> int:
    baseline = _read_json(args.baseline)
    candidate = _read_json(args.candidate)
    comparison = compare_snapshots(baseline, candidate)
    _write_json(args.diff_json, comparison.to_dict())
    _write_text(
        args.report,
        render_comparison_markdown(baseline, candidate, comparison),
    )
    for blocker in comparison.blockers:
        print(f"torchmonarch comparison blocker: {blocker}", file=sys.stderr)
    return 0 if comparison.compatible else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="snapshot and compare dgx-monarch's torchmonarch API surface"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="inspect the installed torchmonarch")
    snapshot.add_argument("--output", required=True, help="JSON path, or - for stdout")
    snapshot.set_defaults(run=_snapshot)

    compare = subparsers.add_parser("compare", help="compare two snapshot JSON files")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--diff-json", required=True)
    compare.add_argument("--report", required=True)
    compare.set_defaults(run=_compare)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
