"""Serialized virtual interfaces, separate instances and typed bypass behavior."""
import copy

import pytest

from dgx_monarch.studio.workflows import WorkflowError, compile_workflow, inspect_workflow

INFO = {
    "Text": {"input": {"required": {"text": ["STRING", {"default": ""}]}}, "output": ["STRING"]},
    "Pass": {"input": {"required": {"value": ["STRING", {"forceInput": True}]}}, "output": ["STRING"]},
    "Save": {"input": {"required": {"value": ["STRING", {"forceInput": True}]}}, "output": [], "output_node": True},
}


def workflow():
    child = {"id": "child", "inputNode": {"id": -10}, "outputNode": {"id": -20},
             "inputs": [{"name": "value", "type": "STRING"}], "outputs": [{"name": "value", "type": "STRING"}],
             "nodes": [{"id": 1, "type": "Pass", "inputs": [{"name": "value", "link": 1}]}],
             "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0},
                       {"id": 2, "origin_id": 1, "origin_slot": 0, "target_id": -20, "target_slot": 0}]}
    return {"id": "main", "nodes": [
        {"id": 1, "type": "Text", "widgets_values": ["source"]},
        {"id": 2, "type": "child", "inputs": [{"name": "value", "link": 1}]},
        {"id": 3, "type": "Save", "inputs": [{"name": "value", "link": 2}]},
    ], "links": [[1, 1, 0, 2, 0, "STRING"], [2, 2, 0, 3, 0, "STRING"]],
        "definitions": {"subgraphs": [child]}, "extra": {"frontendVersion": "1.41.13"}}


def test_subgraph_virtual_links_compile_without_mutating_serialization():
    source = workflow()
    before = copy.deepcopy(source)
    prompt = compile_workflow(source, INFO, {"main:1:text": "edited"})
    child_id = prompt["3"]["inputs"]["value"][0]
    assert prompt[child_id] == {"class_type": "Pass", "inputs": {"value": ["1", 0]}}
    assert prompt["1"]["inputs"]["text"] == "edited"
    assert source == before
    assert inspect_workflow(source, INFO)["errors"] == []


def test_nested_and_shared_instances_keep_separate_execution_identities():
    source = workflow()
    child = source["definitions"]["subgraphs"][0]
    nested = copy.deepcopy(child)
    nested["id"] = "nested"
    child["nodes"][0]["type"] = "nested"
    source["definitions"]["subgraphs"].append(nested)
    source["nodes"].extend([
        {"id": 4, "type": "child", "inputs": [], "widgets_values": ["other"]},
        {"id": 5, "type": "Save", "inputs": [{"name": "value", "link": 3}]},
    ])
    source["links"].append([3, 4, 0, 5, 0, "STRING"])
    prompt = compile_workflow(source, INFO)
    first, second = prompt["3"]["inputs"]["value"][0], prompt["5"]["inputs"]["value"][0]
    assert first != second
    assert prompt[first]["inputs"]["value"] == ["1", 0]
    assert prompt[second]["inputs"]["value"] == "other"


def test_missing_output_and_recursive_definition_refuse():
    source = workflow()
    source["definitions"]["subgraphs"][0]["links"].pop()
    with pytest.raises(WorkflowError, match="exactly one source"):
        compile_workflow(source, INFO)
    source = workflow()
    source["definitions"]["subgraphs"][0]["nodes"][0]["type"] = "child"
    with pytest.raises(WorkflowError, match="Recursive"):
        compile_workflow(source, INFO)


def test_promoted_typed_value_and_shared_definition_edit_refusal():
    source = workflow()
    child = source["definitions"]["subgraphs"][0]
    child["nodes"][0] = {"id": 1, "type": "Text", "widgets_values": ["default"]}
    child["links"].pop(0)
    source["nodes"][1].update(properties={"proxyWidgets": [["1", "text"]]}, widgets_values=["promoted"])
    assert compile_workflow(source, INFO)["__studio_subgraph__2/1"]["inputs"]["text"] == "promoted"
    source["nodes"].append({"id": 4, "type": "child"})
    with pytest.raises(WorkflowError, match="ambiguous"):
        compile_workflow(source, INFO, {"child:1:text": "both?"})
    child["id"] = "child:encoded"
    for node in source["nodes"]:
        if node["type"] == "child":
            node["type"] = child["id"]
    with pytest.raises(WorkflowError, match="ambiguous"):
        compile_workflow(source, INFO, {"child:encoded:1:text": "both?"})


def test_unambiguous_bypass_resolves_while_muted_dependency_refuses():
    source = workflow()
    source["definitions"]["subgraphs"][0]["nodes"].append(
        {"id": 9, "type": "Save", "inputs": [{"name": "value", "link": 3}]})
    source["definitions"]["subgraphs"][0]["links"].append([3, 1, 0, 9, 0, "STRING"])
    source["nodes"][1].update(mode=4, outputs=[{"type": "STRING"}])
    source["nodes"][1]["inputs"][0]["type"] = "STRING"
    prompt = compile_workflow(source, INFO)
    assert prompt["3"]["inputs"]["value"] == ["1", 0]
    assert not any(key.startswith("__studio_subgraph__") for key in prompt)
    source["nodes"][1]["mode"] = 2
    with pytest.raises(WorkflowError, match="muted"):
        compile_workflow(source, INFO)


def promoted_workflow():
    source = workflow()
    child = source["definitions"]["subgraphs"][0]
    child["nodes"][0] = {"id": 1, "type": "Text", "widgets_values": ["default"]}
    child["links"].pop(0)
    source["nodes"][1].update(properties={"proxyWidgets": [["1", "text"]]}, widgets_values=["first"])
    source["nodes"].extend([
        {"id": "4/a:b", "type": "child", "properties": {"proxyWidgets": [["1", "text"]]}, "widgets_values": ["second"]},
        {"id": 5, "type": "Save", "inputs": [{"name": "value", "link": 3}]},
    ])
    source["links"].append([3, "4/a:b", 0, 5, 0, "STRING"])
    return source


def test_shared_instances_project_distinct_editable_binding_ids_and_preserve_source():
    source = promoted_workflow()
    before = copy.deepcopy(source)
    inspected = inspect_workflow(source, INFO)
    rows = {tuple(row["instance_path"]): row for row in inspected["bindings"] if row["graph_id"] == "child"}
    first, second = rows[("2",)], rows[("4/a:b",)]
    assert first["id"] != second["id"] and not first["aliases"] and not second["aliases"]
    assert second["id"] == "@instance/main/4%2Fa%3Ab:1:text"
    assert (first["value"], second["value"]) == ("first", "second")
    prompt = compile_workflow(source, INFO, {second["id"]: "changed"})
    assert prompt[first["execution_node_id"]]["inputs"]["text"] == "first"
    assert prompt[second["execution_node_id"]]["inputs"]["text"] == "changed"
    source["extra"]["linearData"] = {"inputs": [[second["id"], "text"]]}
    assert not inspect_workflow(source, INFO)["errors"]
    source["extra"].pop("linearData")
    assert source == before


def test_nested_promoted_values_follow_explicit_widget_mapping_per_instance():
    source = promoted_workflow()
    child = source["definitions"]["subgraphs"][0]
    leaf = copy.deepcopy(child)
    leaf["id"] = "leaf"
    child["nodes"][0] = {"id": 1, "type": "leaf", "properties": {"proxyWidgets": [["1", "text"]]},
                         "widgets_values": ["inner default"]}
    source["definitions"]["subgraphs"].append(leaf)
    inspection = inspect_workflow(source, INFO)
    assert not inspection["errors"]
    rows = {tuple(row["instance_path"]): row for row in inspection["bindings"] if row["graph_id"] == "leaf"}
    first, second = rows[("2", "1")], rows[("4/a:b", "1")]
    assert (first["value"], second["value"]) == ("first", "second")
    prompt = compile_workflow(source, INFO, {first["id"]: "explicit"})
    assert prompt[first["execution_node_id"]]["inputs"]["text"] == "explicit"
    assert prompt[second["execution_node_id"]]["inputs"]["text"] == "second"
    source["extra"]["linearData"] = {"inputs": [["2:1:1", "text"]]}
    assert next(row for row in inspect_workflow(source, INFO)["bindings"] if row["id"] == first["id"])["prominent"]
    child["nodes"][0]["properties"]["proxyWidgets"].append(["9", "text"])
    child["nodes"][0]["widgets_values"].append("ambiguous")
    with pytest.raises(WorkflowError, match=r"missing node|matching widget"):
        compile_workflow(source, INFO)


def test_unique_legacy_alias_still_works_and_duplicate_alias_edits_refuse():
    source = promoted_workflow()
    source["nodes"] = source["nodes"][:3]
    source["links"].pop()
    row = next(row for row in inspect_workflow(source, INFO)["bindings"] if row["graph_id"] == "child")
    assert row["aliases"] == ["child:1:text"]
    assert compile_workflow(source, INFO, {"child:1:text": "legacy"})[row["execution_node_id"]]["inputs"]["text"] == "legacy"
    with pytest.raises(WorkflowError, match="Both instance and legacy"):
        compile_workflow(source, INFO, {"child:1:text": "one", row["id"]: "two"})


def test_styles_apply_to_one_nested_instance_after_legacy_alias_edits():
    from dgx_monarch.studio.preview_drafts import prepare_preview

    source = promoted_workflow()
    source["nodes"] = source["nodes"][:3]
    source["links"].pop()
    row = next(row for row in inspect_workflow(source, INFO)["bindings"] if row["graph_id"] == "child")
    prompt, values, effective = prepare_preview({"workflow": source, "values": {"child:1:text": "draft"},
                                                "styles": [{"id": "paint", "prompt": "painted {prompt}"}],
                                                "style_bindings": {"prompt": row["id"]}}, INFO)
    assert prompt[row["execution_node_id"]]["inputs"]["text"] == "painted draft"
    assert values[row["id"]] == "painted draft" and "child:1:text" not in values
    assert effective["styles"]["unexpanded"]["prompt"] == "draft"


def test_duplicate_promoted_targets_refuse_instead_of_last_write_wins():
    source = promoted_workflow()
    source["nodes"][1]["properties"]["proxyWidgets"].append(["1", "text"])
    source["nodes"][1]["widgets_values"].append("different")
    with pytest.raises(WorkflowError, match="more than one value"):
        compile_workflow(source, INFO)
