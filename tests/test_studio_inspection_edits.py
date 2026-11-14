"""An inspection validates the draft's edited values, so a repaired value loses its error."""
from __future__ import annotations

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from dgx_monarch.studio import SCHEMA_VERSION
from dgx_monarch.studio.comfy import ComfyClient
from dgx_monarch.studio.server import create_app
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.workflows import inspect_workflow

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = "workflow.dgx-monarch-kromoss-t2i"
MISSING = "kromoss_bf16.safetensors"
AVAILABLE = "kroma-v0.3.1-turbo-opd.safetensors"
ERROR = "3.unet_name: Value is not an available choice"


@pytest.fixture(scope="module")
def kromoss():
    """The packaged Kromoss template and a driver schema without the model file it names."""
    rows = json.loads((ROOT / "src/dgx_monarch/studio/data/workflows.json").read_text())["workflows"]
    workflow = next(row["workflow"] for row in rows if row["id"] == TEMPLATE)
    info = json.loads((ROOT / "tests/fixtures/studio/object_info_bundled.json").read_text())
    info["DGXMonarchUNETLoader"]["input"]["required"]["unet_name"][0].remove(MISSING)
    return workflow, info


def ids(inspection):
    return {row["input_name"]: row["id"] for row in inspection["bindings"] if row["node_id"] in {"3", "10"}}


def test_an_edit_that_repairs_a_saved_choice_clears_its_inspection_error(kromoss):
    # A Studio tester's case: the template names a model file the driver lacks.
    workflow, info = kromoss
    template = inspect_workflow(workflow, info)
    assert template["errors"] == [ERROR] and template["prompt"] is None and template["connections"] == []
    unet = ids(template)["unet_name"]
    repaired = inspect_workflow(workflow, info, {unet: AVAILABLE})
    assert repaired["errors"] == []
    assert repaired["prompt"]["3"]["inputs"]["unet_name"] == AVAILABLE
    # The compiled draft now lists the model links the LoRA tab offers.
    assert repaired["connections"]
    # The binding keeps the saved graph's value; the draft holds the edit.
    assert next(row for row in repaired["bindings"] if row["id"] == unet)["value"] == MISSING


def test_a_genuine_error_survives_an_inspection_with_edits(kromoss):
    workflow, info = kromoss
    fields = ids(inspect_workflow(workflow, info))
    # An edit to another field leaves the saved value, and its error, in place.
    assert inspect_workflow(workflow, info, {fields["seed"]: 5})["errors"] == [ERROR]
    # A file the driver also lacks is still refused.
    assert inspect_workflow(workflow, info, {fields["unet_name"]: "absent.safetensors"})["errors"] == [ERROR]
    # A repaired choice beside an invalid edit names the edit's own problem.
    invalid = inspect_workflow(workflow, info, {fields["unet_name"]: AVAILABLE, fields["steps"]: 0})
    assert invalid["errors"] == ["10.steps: Value violates min=1"] and invalid["prompt"] is None


def test_inspection_leaves_out_edits_that_compile_would_refuse():
    # A connected input or an input the bindings no longer list stays the graph's.
    info = {
        "Loader": {"input": {"required": {"unet_name": [["a.safetensors", "b.safetensors"]]}}, "output": ["MODEL"]},
        "Sampler": {"input": {"required": {"model": ["MODEL"], "steps": ["INT", {"min": 1, "max": 100}]}},
                    "output": ["IMAGE"], "output_node": True},
    }
    workflow = {"3": {"class_type": "Loader", "inputs": {"unet_name": "missing.safetensors"}},
                "4": {"class_type": "Sampler", "inputs": {"model": ["3", 0], "steps": 20}}}
    edits = {"root:3:unet_name": "b.safetensors", "root:4:model": "b.safetensors", "root:9:gone": 1}
    inspection = inspect_workflow(workflow, info, edits)
    assert inspection["errors"] == []
    assert inspection["prompt"]["4"]["inputs"]["model"] == ["3", 0]
    assert inspect_workflow(workflow, info, {"root:4:model": "b.safetensors"})["errors"] == [
        "3.unet_name: Value is not an available choice"]


@asynccontextmanager
async def companion(tmp_path, info):
    driver = web.Application()

    async def object_info(_request):
        return web.json_response(info)

    async def empty(_request):
        return web.json_response({})

    driver.router.add_get("/object_info", object_info)
    for route in ("/history", "/system_stats"):
        driver.router.add_get(route, empty)
    async with TestServer(driver) as upstream:
        store = Store(tmp_path / "state")
        token = "inspection-edits-" + "x" * 32
        hosts: set[str] = set()
        app = create_app(store, ComfyClient(str(upstream.make_url("/")).rstrip("/")), token,
                         allowed_hosts=hosts, poll=False)
        try:
            async with TestClient(TestServer(app)) as browser:
                hosts.add(str(browser.make_url("/")).split("://", 1)[1].rstrip("/"))
                headers = {"Authorization": "Bearer " + token, "X-DGXM-Schema": SCHEMA_VERSION}

                async def inspect(body):
                    response = await browser.post("/api/v1/workflows/inspect", headers=headers, json=body)
                    assert response.status == 200
                    return await response.json()
                yield inspect
        finally:
            store.close()


def test_the_inspect_route_validates_the_drafts_values(tmp_path, kromoss):
    # Studio inspects the draft again with its values after an edit; the route
    # must answer for those values, not for the template as saved.
    workflow, info = kromoss

    async def scenario():
        async with companion(tmp_path, info) as inspect:
            template = await inspect({"workflow": workflow})
            assert template["errors"] == [ERROR]
            fields = ids(template)
            values = {row["id"]: row["value"] for row in template["bindings"] if row["resolved"]}
            assert (await inspect({"workflow": workflow, "values": values}))["errors"] == [ERROR]
            repaired = await inspect({"workflow": workflow, "values": {**values, fields["unet_name"]: AVAILABLE}})
            assert repaired["errors"] == [] and repaired["connections"]
            assert repaired["workflow"] == workflow
    asyncio.run(scenario())


def test_the_template_inspects_clean_when_the_driver_has_its_file():
    # The fixture only removes one file; the packaged template itself is sound.
    rows = json.loads((ROOT / "src/dgx_monarch/studio/data/workflows.json").read_text())["workflows"]
    workflow = copy.deepcopy(next(row["workflow"] for row in rows if row["id"] == TEMPLATE))
    info = json.loads((ROOT / "tests/fixtures/studio/object_info_bundled.json").read_text())
    assert inspect_workflow(workflow, info)["errors"] == []
