#!/usr/bin/env python3
"""Snapshot and compare the torchmonarch surface used by dgx-monarch."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import canary_sessions  # noqa: E402 - tests/canary, beside this runner

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


# A smoke script's own deadline comes this long before the runner's limit, so
# its cleanup, not the hard kill, is the usual end of a stuck run.
DEADLINE_MARGIN_S = 30.0


def _run_smoke(script: Path, output: Path, env: dict[str, str], timeout_s: float,
               deadline_s: float | None = None) -> tuple[int | str, list[int]]:
    """One smoke script in a fresh session: its return code or "timeout", and any group left.

    The script records each session it starts (canary_sessions). However the
    script ended, the runner then stops every recorded group and returns the
    ones it could not stop. On a timeout it first kills the script's own
    group, which a hang in native code leaves as the only way out.
    """
    sessions = output.with_name(output.name + ".sessions")
    sessions.unlink(missing_ok=True)
    deadline = timeout_s - DEADLINE_MARGIN_S if deadline_s is None else deadline_s
    env = {**env, canary_sessions.SESSIONS_ENV: str(sessions),
           canary_sessions.DEADLINE_ENV: f"{max(deadline, 1.0):g}"}
    with subprocess.Popen([sys.executable, str(script), str(output)], env=env,
                          start_new_session=True) as process:
        try:
            returncode: int | str = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Descendant actors must not outlive a timed-out controller.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            returncode = "timeout"
    return returncode, canary_sessions.stop_groups(canary_sessions.recorded(sessions))


def _cpu_smoke(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="dgxm-monarch-cpu-") as scratch:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "",
               "MONARCH_RDMA_DISABLE_IBVERBS": "1", "XDG_RUNTIME_DIR": scratch,
               "PYTHONDONTWRITEBYTECODE": "1"}
        observations: dict[str, object] = {}
        # monarch_cpu_smoke: about 40 s on a 20-core CPU host, a dead proc's
        # faults taking the most (90 s of their own). monarch_lost_rank_smoke:
        # about 50 s, the stop report coming at proc_stop_max_idle (30 s).
        for script, timeout_s in (("monarch_cpu_smoke.py", 240), ("monarch_lost_rank_smoke.py", 300)):
            output = Path(scratch) / f"{script}.json"
            returncode, left = _run_smoke(Path(__file__).with_name(script), output, env, timeout_s)
            if left:
                _write_json(args.output, {"status": "failed", "script": script,
                                          "reason": "process groups left running", "groups": left})
                return 1
            if returncode == "timeout":
                _write_json(args.output, {"status": "failed", "script": script, "reason": "timeout"})
                return 1
            if returncode == canary_sessions.DEADLINE_EXIT:
                _write_json(args.output, {"status": "failed", "script": script,
                                          "reason": "its own deadline"})
                return 1
            if returncode or not output.is_file():
                _write_json(args.output, {"status": "failed", "script": script, "returncode": returncode})
                return 1
            observations.update(_read_json(str(output)))
        _write_json(args.output, {"status": "passed", "scope": "local_cpu_only",
                                  "observations": observations})
        return 0


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
    smoke = subparsers.add_parser("cpu-smoke", help="spawn and stop only local CPU actors")
    smoke.add_argument("--output", required=True)
    smoke.set_defaults(run=_cpu_smoke)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
