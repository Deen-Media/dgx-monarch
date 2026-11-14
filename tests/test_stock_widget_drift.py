"""Master-shaped stock widgets on an older ComfyUI (dgx_monarch/stock_widget_drift.py).

The templates follow ComfyUI master, whose Create Video gained
chroma_subsampling between color_space and codec in ComfyUI 0.44.0. ComfyUI
0.43 and older lack it. Each reader that maps widgets_values by a running
ComfyUI's own order must read the later values in their own slots there, and
leave a graph saved on the older ComfyUI as it was:

* the sweep converter (benchmark/sweep/convert.py),
* Studio's widget layout (studio/widget_values.py), checked against the
  recorded driver schema, which predates the widget,
* the frontend heal (web/js/dgx_monarch_widgets.js), driven under node.

The real-ComfyUI half is the comfy canary (tests/canary/template_widget_canary.py,
video_export_canary.py). CI runs it against master and the newest release,
which both declare the widget, so the older side rests on these tests and on
a canary run against a 0.43 checkout.
"""
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from benchmark.sweep.convert import convert, object_info_widget_names, repo_widget_names

from dgx_monarch import stock_widget_drift
from dgx_monarch.studio.widget_values import layout_values
from dgx_monarch.studio.workflows import WorkflowError

REPO = Path(__file__).resolve().parents[1]
OLDER_INFO = json.loads((REPO / "tests/fixtures/studio/object_info_bundled.json").read_text())["CreateVideo"]
TEMPLATE = REPO / "example_workflows/dgx-monarch-minimax-h3-mp4.json"
OLDER = ["fps", "bit_depth", "color_space", "codec"]
MASTER = ["fps", "bit_depth", "color_space", "chroma_subsampling", "codec"]


def _master_info() -> dict:
    """The recorded schema with master's widget added where master declares it."""
    info = copy.deepcopy(OLDER_INFO)
    optional = info["input"]["optional"]
    chroma = ["COMBO", {"default": "auto", "multiselect": False, "options": ["auto", "444", "422", "420"]}]
    order = ["audio", "bit_depth", "color_space", "chroma_subsampling", "codec"]
    info["input"]["optional"] = {name: optional.get(name, chroma) for name in order}
    info["input_order"]["optional"] = order
    return info


def _create_video(graph: dict) -> dict:
    return next(node for node in graph["nodes"] if node["type"] == "CreateVideo")


def _names(order: list[str]):
    return lambda node_class: order if node_class == "CreateVideo" else repo_widget_names()(node_class)


def test_the_table_puts_a_missing_widget_back_after_its_anchor():
    assert object_info_widget_names(OLDER_INFO["input"]) == OLDER
    assert object_info_widget_names(_master_info()["input"]) == MASTER
    assert stock_widget_drift.missing("CreateVideo", OLDER) == {"chroma_subsampling": "auto"}
    assert stock_widget_drift.with_newer("CreateVideo", OLDER) == MASTER
    assert stock_widget_drift.missing("CreateVideo", MASTER) == {}
    assert stock_widget_drift.with_newer("CreateVideo", MASTER) == MASTER
    # No anchor, no guess: a reordered node keeps its own order.
    assert stock_widget_drift.missing("CreateVideo", ["fps", "codec"]) == {}
    assert stock_widget_drift.with_newer("SaveVideo", OLDER) == OLDER


def test_every_shipped_create_video_holds_the_default_in_the_newer_slot():
    generated = repo_widget_names()("CreateVideo")
    assert generated == MASTER
    seen = 0
    for path in [*(REPO / "example_workflows").glob("*.json"),
                 *(REPO / "tests/fixtures/workflows/generated").glob("*.json")]:
        for node in json.loads(path.read_text()).get("nodes", []):
            if node["type"] != "CreateVideo":
                continue
            seen += 1
            values = dict(zip(generated, node["widgets_values"], strict=True))
            assert values["chroma_subsampling"] == "auto", path.name
            assert node["dgxm_widgets"] == values, path.name
    assert seen >= 8


def test_the_sweep_converter_reads_codec_in_its_own_slot_on_both_lines():
    ui = json.loads(TEMPLATE.read_text())
    on_older = _create_video_inputs(convert(ui, _names(OLDER)))
    on_master = _create_video_inputs(convert(ui, _names(MASTER)))
    assert on_older == {"fps": 24.0, "bit_depth": 8, "color_space": "sRGB", "codec": "none"}
    assert on_master == {**on_older, "chroma_subsampling": "auto"}
    # A graph saved on the older ComfyUI keeps reading as it did.
    saved = copy.deepcopy(ui)
    _create_video(saved)["widgets_values"] = [24.0, 8, "sRGB", "h264"]
    assert _create_video_inputs(convert(saved, _names(OLDER)))["codec"] == "h264"
    # A choice the older ComfyUI cannot honor stays in the prompt, so the canary reports it.
    _create_video(saved)["widgets_values"] = [24.0, 8, "sRGB", "444", "none"]
    inputs = _create_video_inputs(convert(saved, _names(OLDER)))
    assert inputs["chroma_subsampling"] == "444" and inputs["codec"] == "none"


def _create_video_inputs(graph: dict) -> dict:
    node = next(node for node in graph.values() if node["class_type"] == "CreateVideo")
    return {name: value for name, value in node["inputs"].items() if name not in ("images", "audio")}


def test_studio_lays_out_master_shaped_values_against_the_recorded_older_schema():
    node = _create_video(json.loads(TEMPLATE.read_text()))
    assert layout_values(node, OLDER_INFO) == {"fps": 24.0, "bit_depth": 8, "color_space": "sRGB",
                                                 "codec": "none"}
    assert layout_values(node, _master_info()) == {"fps": 24.0, "bit_depth": 8, "color_space": "sRGB",
                                                   "chroma_subsampling": "auto", "codec": "none"}
    saved = {**node, "widgets_values": [24.0, 8, "sRGB", "av1"]}
    assert layout_values(saved, OLDER_INFO)["codec"] == "av1"
    with pytest.raises(WorkflowError, match="chroma_subsampling holds '420'"):
        layout_values({**node, "widgets_values": [24.0, 8, "sRGB", "420", "none"]}, OLDER_INFO)
    with pytest.raises(WorkflowError, match="unsupported extra values"):
        layout_values({**node, "widgets_values": [24.0, 8, "sRGB", "auto", "none", "x"]}, OLDER_INFO)


def test_the_frontend_heal_lists_the_table_and_restores_create_video_by_name(tmp_path):
    source = (REPO / "web/js/dgx_monarch_widgets.js").read_text()
    listed = re.search(r"const NAMED_STOCK = \[([^\]]*)\];", source)
    assert listed, "NAMED_STOCK literal not found in dgx_monarch_widgets.js"
    assert sorted(json.loads(f"[{listed.group(1)}]")) == sorted(stock_widget_drift.NEWER)
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required for the frontend heal check")
    (tmp_path / "package.json").write_text('{"type":"module"}')
    (tmp_path / "widgets.js").write_text(source.replace('"../../scripts/app.js"', '"./mocks.js"'))
    (tmp_path / "mocks.js").write_text(
        "export const app = { registerExtension(e) { globalThis.extension = e; }, extensionManager: {} };\n")
    saved = _create_video(json.loads(TEMPLATE.read_text()))
    (tmp_path / "saved.json").write_text(json.dumps(saved))
    (tmp_path / "older.json").write_text(json.dumps({"name": "CreateVideo", "category": "video", **OLDER_INFO}))
    (tmp_path / "master.json").write_text(json.dumps({"name": "CreateVideo", "category": "video", **_master_info()}))
    (tmp_path / "driver.js").write_text(_DRIVER)
    result = subprocess.run([node, "driver.js"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "heal green"


# Mirrors the frontend: build the widgets a schema declares, restore the saved
# array by position (frontend 1.57.4 with "Restore widget values by name" off,
# its default), then run the extension's onConfigure.
_DRIVER = r"""
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
await import('./widgets.js');
const load = name => JSON.parse(readFileSync(name, 'utf8'));
const saved = load('saved.json');

function configured(nodeData) {
    function NodeType() {}
    NodeType.prototype = {};
    globalThis.extension.beforeRegisterNodeDef(NodeType, nodeData);
    const node = new NodeType();
    node.type = nodeData.name;
    node.id = saved.id;
    node.widgets = [];
    for (const group of ['required', 'optional']) {
        for (const name of nodeData.input_order[group]) {
            const [kind, config] = nodeData.input[group][name];
            if (!['FLOAT', 'COMBO'].includes(kind)) continue;
            node.widgets.push({ name, value: config.default, options: kind === 'COMBO' ? { values: config.options } : {} });
        }
    }
    saved.widgets_values.forEach((value, index) => { if (node.widgets[index]) node.widgets[index].value = value; });
    node.onConfigure(saved);
    return node;
}
const values = node => Object.fromEntries(node.widgets.map(w => [w.name, w.value]));

const older = configured(load('older.json'));
assert.deepEqual(values(older), { fps: 24, bit_depth: 8, color_space: 'sRGB', codec: 'none' });
const master = configured(load('master.json'));
assert.deepEqual(values(master), { fps: 24, bit_depth: 8, color_space: 'sRGB', chroma_subsampling: 'auto', codec: 'none' });
const out = {};
master.widgets.find(w => w.name === 'codec').value = 'h264';
master.onSerialize(out);
assert.deepEqual(out.dgxm_widgets, { fps: 24, bit_depth: 8, color_space: 'sRGB', chroma_subsampling: 'auto', codec: 'h264' });

// A stock node outside the list keeps the frontend's own restore.
function Other() {}
Other.prototype = {};
globalThis.extension.beforeRegisterNodeDef(Other, { name: 'SaveVideo', category: 'image/video', input: {} });
assert.equal(Other.prototype.onConfigure, undefined);
console.log('heal green');
"""
