"""Each editing workflow Send to edit offers compiles with the output bound.

Send to edit (studio/src/editHandoff.ts) copies the output into the driver's
input store as a `studio/` upload and binds it to the image input the editing
workflow's App Mode layout lists first. At Generate the companion compiles that
draft against the driver schema through a view whose image upload inputs also
list the request's stored uploads. This runs the same path for each editing
workflow Studio offers on a driver with the recorded schema, the list
studio/src/editHandoff.test.ts holds, so a target that cannot take the copy
fails here. CPU only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dgx_monarch.studio.preview_drafts import prepare_preview
from dgx_monarch.studio.submission_guard import upload_view
from dgx_monarch.studio.workflows import WorkflowError, binding_id

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "tests/fixtures/studio/object_info_bundled.json").read_text())
WORKFLOWS = {row["name"]: row["workflow"] for row in
             json.loads((ROOT / "src/dgx_monarch/studio/data/workflows.json").read_text())["workflows"]}
EDITING = ("dgx-monarch-flux3-klein-base-edit", "dgx-monarch-flux3-klein-edit",
           "dgx-monarch-mage-flow-edit-turbo", "dgx-monarch-mage-flow-edit",
           "dgx-monarch-minimax-h3-image-edit", "dgx-monarch-qwen-image21-edit-mask-rgba",
           "dgx-monarch-qwen-image21-edit-rgba")
COPY = "studio/copy.png"


def edit_target(workflow: dict) -> tuple[str, str]:
    """The image input of a Load Image node that the App Mode layout lists first."""
    types = {node["id"]: node["type"] for node in workflow["nodes"]}
    return next((str(node), name) for node, name in workflow["extra"]["linearData"]["inputs"]
                if name == "image" and types[node] == "LoadImage")


def test_the_minimax_h3_image_edit_takes_the_output_on_node_11():
    assert edit_target(WORKFLOWS["dgx-monarch-minimax-h3-image-edit"]) == ("11", "image")


@pytest.mark.parametrize("name", EDITING)
def test_send_to_edit_binds_the_copy_and_the_draft_compiles(name):
    workflow = copy.deepcopy(WORKFLOWS[name])
    node, input_name = edit_target(workflow)
    target = binding_id(workflow["id"], node, input_name)
    request = {"workflow": workflow, "values": {target: COPY}, "operation": "edit",
               "styles": [], "lora_stack": []}
    prompt, values, _ = prepare_preview(request, SCHEMA, view=upload_view(SCHEMA, frozenset({COPY})))
    assert prompt[node]["inputs"][input_name] == COPY
    assert values[target] == COPY
    # The driver's own schema lists no stored upload, so the bound copy needs the view.
    with pytest.raises(WorkflowError, match=f"{node}.{input_name}: Value is not an available choice"):
        prepare_preview(copy.deepcopy(request), SCHEMA)
