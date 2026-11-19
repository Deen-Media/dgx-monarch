"""A refused draft's button opens the Create tab the companion's refusal names.

The companion refuses a draft value as ``<node>.<input>: <reason>``. The node can
be a subgraph execution id with "/" and the input a dotted dynamic option such as
Save Video's ``format.codec``. These tests make each refusal through the real
``/jobs/preview`` route and run Studio's ``adjustTarget`` under Node on the error
the browser receives.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from test_studio_integration import companion

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "studio/src/draftRefusal.ts"
BUNDLED = json.loads((ROOT / "tests/fixtures/studio/object_info_bundled.json").read_text())
ALL_INPUTS = {"tab": "advanced", "label": "Open All inputs"}
LAYOUT = {"tab": "workflow", "label": "Open Layout"}


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to run Studio's refusal reader")
    return node


def targets(errors: list[dict]) -> list:
    """The tab and button Studio offers for each refusal the browser received."""
    program = (f"import {{ adjustTarget }} from {json.dumps(MODULE.as_uri())};\n"
               f"const errors = {json.dumps(errors)};\n"
               "process.stdout.write(JSON.stringify(errors.map((error) => adjustTarget(error))));")
    result = subprocess.run([_node(), "--input-type=module", "-e", program], capture_output=True,
                            text=True, timeout=60, check=True)
    return json.loads(result.stdout)


def subgraph_workflow() -> dict:
    """A workflow whose subgraph instance 12 holds a loader, node 5, set to a file the driver no longer lists."""
    child = {"id": "child", "inputNode": {"id": -10}, "outputNode": {"id": -20},
             "inputs": [], "outputs": [{"name": "value", "type": "STRING"}],
             "nodes": [{"id": 5, "type": "Pick", "widgets_values": ["removed.safetensors"]}],
             "links": [{"id": 1, "origin_id": 5, "origin_slot": 0, "target_id": -20, "target_slot": 0}]}
    return {"id": "main", "nodes": [
        {"id": 12, "type": "child", "inputs": []},
        {"id": 3, "type": "Save", "inputs": [{"name": "value", "link": 2}]},
    ], "links": [[2, 12, 0, 3, 0, "STRING"]], "definitions": {"subgraphs": [child]},
        "extra": {"frontendVersion": "1.41.13"}}


def refusals(tmp_path, monkeypatch) -> list[dict]:
    """The errors /jobs/preview answers for three refused drafts."""
    template = json.loads((ROOT / "example_workflows/dgx-monarch-wan22-ti2v.json").read_text())
    drafts = [
        # A dotted dynamic input of a shipped template's Save Video.
        {"workflow": template, "values": {f"{template['id']}:13:format.codec": "no-such-codec"},
         "operation": "video"},
        # A value inside a subgraph: the companion names it by its execution id.
        {"workflow": subgraph_workflow(), "values": {}, "operation": "workflow"},
        # A graph fault names no input, so Layout repairs it.
        {"workflow": {**subgraph_workflow(), "nodes": [{"id": 3, "type": "Gone"}], "links": []},
         "values": {}, "operation": "workflow"},
    ]

    async def scenario():
        async with companion(tmp_path, monkeypatch) as system:
            system.protocol.schema = {
                **BUNDLED,
                "Pick": {"input": {"required": {"name": [["kept.safetensors"]]}}, "output": ["STRING"]},
                "Save": {"input": {"required": {"value": ["STRING", {"forceInput": True}]}}, "output": [],
                         "output_node": True},
            }
            return [(await system.request("POST", "/jobs/preview", expected=400,
                                          json={**draft, "seed_policy": "fixed", "project_id": "project-one"}))["error"]
                    for draft in drafts]

    return asyncio.run(scenario())


def test_a_refused_value_opens_all_inputs_and_a_graph_fault_opens_layout(tmp_path, monkeypatch):
    errors = refusals(tmp_path, monkeypatch)
    assert [error["code"] for error in errors] == ["workflow"] * 3
    assert errors[0]["message"].startswith("13.format.codec: "), errors[0]
    assert errors[1]["message"].startswith("__studio_subgraph__12/5.name: "), errors[1]
    assert errors[2]["message"] == "Node type unavailable: Gone", errors[2]
    assert targets(errors) == [ALL_INPUTS, ALL_INPUTS, LAYOUT]
