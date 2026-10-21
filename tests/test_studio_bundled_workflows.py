"""Bundled and frontend-saved UI workflows compile against a recorded driver schema.

``fixtures/studio/object_info_bundled.json`` is trimmed from a real /object_info
recording; ``fixtures/studio/README.md`` says how. The upload loaders, V3 dynamic
combos, union widgets and Autogrow sockets below all come from that recording,
and the widget shapes come from frontend-saved workflows.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from dgx_monarch.studio.workflows import WorkflowError, compile_workflow, inspect_workflow

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = json.loads((ROOT / "src/dgx_monarch/studio/data/workflows.json").read_text())["workflows"]
OBJECT_INFO = json.loads((ROOT / "tests/fixtures/studio/object_info_bundled.json").read_text())
# UI-only nodes that never reach a driver, as build_object_info_bundled.py lists them.
VIRTUAL = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}
RECORD = ("record a new /object_info and rebuild the fixture as tests/fixtures/studio/README.md "
          "(Recording a new driver schema) says")


@pytest.fixture
def info():
    return copy.deepcopy(OBJECT_INFO)


def _classes(graph: dict) -> set[str]:
    """Node classes in a graph and its subgraphs; a subgraph instance names its subgraph, not a class."""
    subgraphs = graph.get("definitions", {}).get("subgraphs", [])
    nodes = [*graph.get("nodes", []), *(node for subgraph in subgraphs for node in subgraph.get("nodes", []))]
    return {node["type"] for node in nodes} - {subgraph.get("id") for subgraph in subgraphs}


def missing_classes(workflow: dict) -> list[str]:
    """Node classes a workflow uses that the recorded schema lacks."""
    return sorted(_classes(workflow) - set(OBJECT_INFO) - VIRTUAL)


def test_fixture_covers_every_node_class_the_bundled_set_uses():
    # Counted from the checkout, so adding or removing a template changes only the generated data.
    assert len(WORKFLOWS) == len(list((ROOT / "example_workflows").glob("*.json")))
    missing = sorted({name for row in WORKFLOWS for name in missing_classes(row["workflow"])})
    if missing:
        pytest.skip(f"The recorded schema lacks node classes {', '.join(missing)}; {RECORD}.")


def test_a_class_the_recording_lacks_is_named_for_the_skip_reason():
    recorded = next(name for name in sorted(OBJECT_INFO) if name not in VIRTUAL)
    workflow = {"nodes": [{"type": "BrandNewNode"}, {"type": "Note"}, {"type": recorded}, {"type": "sub-1"}],
                "definitions": {"subgraphs": [{"id": "sub-1", "nodes": [{"type": "AnotherNewNode"}]}]}}
    assert missing_classes(workflow) == ["AnotherNewNode", "BrandNewNode"]


@pytest.mark.parametrize("row", WORKFLOWS, ids=[row["id"] for row in WORKFLOWS])
def test_bundled_workflow_inspects_and_compiles(row, info):
    if missing := missing_classes(row["workflow"]):
        pytest.skip(f"{row['id']} uses node classes the recorded schema lacks ({', '.join(missing)}); {RECORD}.")
    inspection = inspect_workflow(row["workflow"], info)
    assert inspection["errors"] == []
    assert compile_workflow(row["workflow"], info) == inspection["prompt"]


def _ui(*nodes, links=()):
    return {"id": "main", "nodes": list(nodes), "links": [list(link) for link in links],
            "extra": {"linearData": {"inputs": [], "outputs": []}}}


def _node(nid, kind, values, sockets=()):
    return {"id": nid, "type": kind, "mode": 0, "widgets_values": values, "inputs": list(sockets)}


def _save_image(nid, link):
    return _node(nid, "SaveImage", ["out"], [{"name": "images", "type": "IMAGE", "link": link}])


@pytest.mark.parametrize("values", [["example.png", "image"], ["example.png", ""], ["example.png"]])
def test_upload_helper_slot_carries_any_value_or_none(values, info):
    workflow = _ui(_node(1, "LoadImage", values), _save_image(2, 1), links=[(1, 1, 0, 2, 0, "IMAGE")])
    assert compile_workflow(workflow, info)["1"] == {"class_type": "LoadImage", "inputs": {"image": "example.png"}}


@pytest.mark.parametrize("values", [["wan_dancer_reference_audio.mp3", None, None],
                                    ["wan_dancer_reference_audio.mp3", None, ""],
                                    ["wan_dancer_reference_audio.mp3", ""],
                                    ["wan_dancer_reference_audio.mp3", None]])
def test_audio_loader_player_and_upload_slots_are_accepted(values, info):
    info["Sink"] = {"input": {"required": {"audio": ["AUDIO"]}}, "output": [], "output_node": True}
    workflow = _ui(_node(1, "LoadAudio", values),
                   _node(2, "Sink", [], [{"name": "audio", "type": "AUDIO", "link": 1}]),
                   links=[(1, 1, 0, 2, 0, "AUDIO")])
    assert compile_workflow(workflow, info)["1"]["inputs"] == {"audio": "wan_dancer_reference_audio.mp3"}


def test_unlinked_upload_widget_socket_is_ignored_but_an_undeclared_link_refuses(info):
    sockets = [{"name": "image", "type": "COMBO", "widget": {"name": "image"}, "link": None},
               {"name": "upload", "type": "IMAGEUPLOAD", "widget": {"name": "upload"}, "link": None}]
    workflow = _ui(_node(1, "LoadImage", ["example.png", "image"], sockets), _save_image(2, 1),
                   links=[(1, 1, 0, 2, 0, "IMAGE")])
    assert compile_workflow(workflow, info)["1"]["inputs"] == {"image": "example.png"}
    workflow["nodes"][1]["inputs"].append({"name": "extension_only", "type": "IMAGE", "link": 2})
    workflow["links"].append([2, 1, 0, 2, 1, "IMAGE"])
    with pytest.raises(WorkflowError, match="Custom input sockets"):
        compile_workflow(workflow, info)


def test_extra_slots_still_refuse_on_a_node_without_an_upload_input(info):
    workflow = _ui(_node(1, "LoadImage", ["example.png", "image"]), _save_image(2, 1),
                   links=[(1, 1, 0, 2, 0, "IMAGE")])
    workflow["nodes"][1]["widgets_values"] = ["out", "unexpected"]
    inspection = inspect_workflow(workflow, info)
    assert any("unsupported extra values" in error for error in inspection["errors"])
    with pytest.raises(WorkflowError, match="unsupported extra values"):
        compile_workflow(workflow, info)


def _save_video(values):
    info_sink = _node(1, "LoadVideo", ["wan_animate2_driving.mp4", "image"])
    return _ui(info_sink, _node(2, "SaveVideo", values, [{"name": "video", "type": "VIDEO", "link": 1}]),
               links=[(1, 1, 0, 2, 0, "VIDEO")])


def test_dynamic_combo_children_follow_their_parent_in_frontend_order(info):
    # The 1.55.18 frontend saves format, format.codec, the hidden top-level codec and its encoding.
    prompt = compile_workflow(_save_video(["clip", "mp4", "auto", "h264", "auto"]), info)
    assert prompt["2"]["inputs"] == {"video": ["1", 0], "filename_prefix": "clip", "format": "mp4",
                                     "format.codec": "auto", "codec": "h264", "codec.encoding": "auto"}
    inspection = inspect_workflow(_save_video(["clip", "mp4", "auto", "h264", "auto"]), info)
    names = [row["input_name"] for row in inspection["bindings"] if row["node_id"] == "2"]
    assert names == ["filename_prefix", "format", "format.codec", "codec", "codec.encoding"]
    with pytest.raises(WorkflowError, match="unsupported extra values"):
        compile_workflow(_save_video(["clip", "mp4", "auto", "h264", "auto", "surplus"]), info)


def test_dynamic_combo_edit_reselects_child_inputs(info):
    workflow = _save_video(["clip", "mp4", "h264"])
    inputs = compile_workflow(workflow, info)["2"]["inputs"]
    assert (inputs["format"], inputs["format.codec"], inputs["format.codec.encoding"]) == ("mp4", "h264", "auto")
    assert compile_workflow(workflow, info, {"main:2:format": "mkv"})["2"]["inputs"]["format.codec"] == "h264"
    # WebM offers no H.264, so the kept codec is refused rather than silently dropped.
    with pytest.raises(WorkflowError, match=re.escape("format.codec: Value is not an available choice")):
        compile_workflow(workflow, info, {"main:2:format": "webm"})
    inputs = compile_workflow(workflow, info, {"main:2:format": "webm", "main:2:format.codec": "av1"})["2"]["inputs"]
    assert (inputs["format"], inputs["format.codec"], inputs["format.codec.encoding"]) == ("webm", "av1", "auto")


def test_union_widget_takes_its_widget_type(info):
    schema = info["LTXVEmptyLatentAudio"]["input"]["required"]["frame_rate"]
    assert schema[0] == "FLOAT,INT" and schema[1]["widgetType"] == "FLOAT"
    workflow = next(row["workflow"] for row in WORKFLOWS if row["id"] == "workflow.dgx-monarch-ltx25-t2v")
    inspection = inspect_workflow(workflow, info)
    rate = next(row for row in inspection["bindings"] if row["input_name"] == "frame_rate"
                and row["schema"].get("widgetType") == "FLOAT")
    assert rate["schema"]["type"] == "FLOAT"
    node = rate["node_id"]
    with pytest.raises(WorkflowError, match="Expected a number"):
        compile_workflow(workflow, info, {rate["id"]: "fast"})
    assert compile_workflow(workflow, info, {rate["id"]: 30})[node]["inputs"]["frame_rate"] == 30
    # Each edit applies once, so a value unequal to itself still reaches the value check.
    with pytest.raises(WorkflowError, match="Number must be finite"):
        compile_workflow(workflow, info, {rate["id"]: float("nan")})


def test_audio_save_option_child_and_player_socket_from_a_frontend_export(info):
    # A 1.57 export: widget sockets for every widget, the option child and the audioUI player.
    sockets = [{"name": "audio", "type": "AUDIO", "link": 1}]
    sockets += [{"name": name, "type": "COMBO", "widget": {"name": name}, "link": None}
                for name in ("filename_prefix", "format", "format.quality", "audioUI")]
    workflow = _ui(_node(1, "LoadAudio", ["wan_dancer_reference_audio.mp3", None]),
                   _node(2, "SaveAudioAdvanced", ["audio/clip", "mp3", "V0"], sockets),
                   links=[(1, 1, 0, 2, 0, "AUDIO")])
    assert compile_workflow(workflow, info)["2"]["inputs"] == {
        "audio": ["1", 0], "filename_prefix": "audio/clip", "format": "mp3", "format.quality": "V0"}
