"""Summarize a completed sweep session and its findings.

Leave source records in place. Pass report text through the matrix runner's
redactor and refuse to write if private identifiers remain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .matrix import FULL_RENDER_BAR, WAIVED_BAR, load_config, run_matrix_module

COLUMNS = ("cell", "template", "label", "observed", "legs", "cold", "warm", "nrms",
           "max_abs", "frames", "verdict")


def _row(record: dict) -> dict:
    cell, legs = record["cell"], record.get("legs") or {}
    fidelity = record.get("fidelity") or {}
    return {
        "cell": cell["id"], "label": cell["label"], "verdict": record.get("verdict", ""),
        "template": f"{cell['template']} {cell['preset']} {cell['attention']}",
        "observed": record.get("observed", "not-run"),
        # The verdict scores the cold leg; a warm or probe leg that crashed
        # would otherwise leave no mark on the table.
        "legs": " ".join(f"{name}={leg.get('outcome', '')}" for name, leg in legs.items()),
        "cold": legs.get("cold", {}).get("wall_s", ""),
        "warm": legs.get("warm", {}).get("wall_s", ""),
        "nrms": fidelity.get("nrms", ""), "max_abs": fidelity.get("max_abs", ""),
        "frames": legs.get("warm", {}).get("frames", legs.get("cold", {}).get("frames", "")),
    }


def records_for(out: Path, session: str) -> list[dict]:
    """Every current cell record for one session.

    A retained attempt is not a row: the runner keeps it under ``.attempt`` so
    an infra failure stays visible, and counting it would list the cell twice
    and could publish a dead pin as the session's.
    """
    records = []
    for path in sorted((out / "cells").glob("*.json")):
        if ".attempt" in path.name:
            continue
        record = json.loads(path.read_text())
        if record["cell"]["session"] == session:
            records.append(record)
    return records


def build(records: list[dict], session: str) -> dict:
    """The table, the findings, and the numbers measured against no floor, apart.

    A full-render line is a measurement against no floor, unlike a step-nrms
    line over a calibrated one; mixed into the findings list it would read as a
    fault, and on this matrix it would outnumber them. The split is per line,
    not per record, and it matches the line the runner stored under
    ``full_render`` rather than a finding's wording: one cell can carry a real
    fault and a full-render number at once, and routing the whole record would
    file the fault here.

    A waived-render number never reaches the findings list: the class K waiver
    admits that deviation, whether the matrix names the kernel as approximate
    or the leg rendered under a granted class K card, so the runner files it as
    a note. It is collected here from its own key, because the number is the
    deliverable of those cells and a table column alone does not say what it
    measured.
    """
    rows = [_row(record) for record in records]
    findings: list[str] = []
    full_render: list[str] = []
    for record, row in zip(records, rows, strict=True):
        measured = record.get("full_render")
        for detail in record.get("findings") or []:
            line = f"- `{record['cell']['id']}` {row['template']}: {detail}"
            (full_render if measured and detail == measured else findings).append(line)
    waived = [f"- `{record['cell']['id']}` {row['template']}: {record['waived_render']}"
              for record, row in zip(records, rows, strict=True)
              if record.get("waived_render")]
    return {"session": session, "rows": rows, "findings": findings,
            "full_render": full_render, "waived": waived,
            "pin": records[0].get("pin") if records else {}}


def render(payload: dict) -> str:
    """The markdown for one payload. main passes the redacted one, so no
    private path reaches the file the operator pastes."""
    lines = [f"# sweep summary: {payload['session']} session", "",
             "| " + " | ".join(COLUMNS) + " |",
             "|" + "|".join("---" for _ in COLUMNS) + "|"]
    lines += ["| " + " | ".join(str(row[name]) for name in COLUMNS) + " |"
              for row in payload["rows"]]
    lines += ["", "## findings", ""]
    lines += payload["findings"] or ["- none"]
    lines += ["", f"## full-render measurements ({FULL_RENDER_BAR})", "",
              "These cells have no probe leg, so each number compares two full warm "
              "renders and has no floor. Step-nrms checks are in the findings list above. "
              "A cell listed here may also carry a fault there; read the verdict column.",
              ""]
    lines += payload.get("full_render") or ["- none"]
    lines += ["", f"## waived-render measurements ({WAIVED_BAR})", "",
              "These cells hold a render the runtime already knows to be wrong: a kernel the matrix "
              "names as approximate, or a leg that rendered under a granted class K card. Each "
              "number is the deviation the waiver admits, measured against no floor. None of "
              "them is a check: the render is the PASS and the number is the result.", ""]
    lines += payload.get("waived") or ["- none"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.sweep.report",
        description="Write a sanitized summary table and findings list under the out dir.")
    parser.add_argument("--config", required=True, help="sweep TOML (see sweep.example.toml)")
    parser.add_argument("--session", required=True, choices=("local", "cluster"))
    args = parser.parse_args(argv)
    config = load_config(Path(args.config))
    out = config["out_dir"]
    records = records_for(out, args.session)
    payload = build(records, args.session)
    run_matrix = run_matrix_module()
    secrets = list(run_matrix.private_path_secrets(payload))
    sanitized = run_matrix.sanitize_report(payload)
    text = render(sanitized)
    if leaked := run_matrix.leaked_secrets(text, secrets):
        parser.error(f"{len(leaked)} private value(s) survive in the summary text, so nothing was written")
    (out / f"summary_{args.session}.md").write_text(text)
    (out / f"summary_{args.session}.json").write_text(
        json.dumps(sanitized, indent=2, default=str))
    print(text)
    print(f"wrote {out / f'summary_{args.session}.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
