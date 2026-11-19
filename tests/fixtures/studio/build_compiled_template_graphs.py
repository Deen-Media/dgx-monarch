"""Build compiled_template_graphs.json: each bundled template as the API prompt a job records.

Usage::

    python tests/fixtures/studio/build_compiled_template_graphs.py

A job's preview compiles the draft's workflow (``prepare_preview``), the job
records that API prompt, and Reuse settings and Upscale load it back as the new
draft's workflow. Studio then names the draft's fields from the API prompt, not
from the saved workflow. For each bundled workflow this keeps, per node, its
class and the inputs a field's name can depend on: every link, and every text
input with its value emptied. studio/src/textThrough.test.ts names each
template's fields in this form and in its saved form, and
tests/test_studio_fixture_builder.py checks the file against a fresh build.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dgx_monarch.studio.preview_drafts import prepare_preview

REPO = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO / "src/dgx_monarch/studio/data/workflows.json"
SCHEMA = Path(__file__).with_name("object_info_bundled.json")
OUTPUT = Path(__file__).with_name("compiled_template_graphs.json")


def _is_text(schema: dict[str, Any], kind: str, name: str) -> bool:
    spec = schema.get(kind, {}).get("input", {})
    found = (spec.get("required") or {}).get(name) or (spec.get("optional") or {}).get(name)
    return isinstance(found, list) and bool(found) and found[0] == "STRING"


def graph(prompt: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """An API prompt's links and emptied text inputs, node by node."""
    kept = {}
    for node_id, node in prompt.items():
        inputs = {}
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                inputs[name] = value
            elif isinstance(value, str) and _is_text(schema, node["class_type"], name):
                inputs[name] = ""
        kept[node_id] = {"class_type": node["class_type"], "inputs": inputs}
    return kept


def build(workflows: list[dict[str, Any]], schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    graphs = {}
    for row in sorted(workflows, key=lambda row: row["name"]):
        prompt, _values, _transformations = prepare_preview({"workflow": row["workflow"], "values": {}}, schema)
        graphs[row["name"]] = graph(prompt, schema)
    return graphs


def write(graphs: dict[str, dict[str, Any]], path: Path) -> None:
    """One template to a line, so a change to one template reads as a one-line diff."""
    lines = [f"  {json.dumps(name)}: {json.dumps(nodes, separators=(',', ':'))}" for name, nodes in graphs.items()]
    about = ("Each bundled template compiled as a job's preview compiles it, the API prompt Reuse settings "
             "loads: per node its class, its links and its text inputs, values emptied. "
             "build_compiled_template_graphs.py writes it.")
    path.write_text("{\n" + f' "about": {json.dumps(about)},\n "graphs": {{\n' + ",\n".join(lines) + "\n }\n}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    graphs = build(json.loads(WORKFLOWS.read_text())["workflows"], json.loads(SCHEMA.read_text()))
    write(graphs, args.out)
    print(f"wrote {len(graphs)} compiled template graphs to {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
