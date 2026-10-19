"""CPU-only compatibility tests for explicit graph and draft contracts."""

import copy
import json
from pathlib import Path

import pytest

from dgx_monarch.studio.drafts import DraftError, apply_preset, apply_styles, expand_grid, resolve_loras, undo_preset
from dgx_monarch.studio.workflows import (
    WorkflowError,
    binding_id,
    compile_workflow,
    export_workflow,
    inspect_workflow,
    rebind_workflow,
)


@pytest.fixture
def info():
    return {
        "Text": {"input": {"required": {"text": ["STRING", {"default": ""}]}}, "output": ["STRING"]},
        "Sampler": {"input": {"required": {
            "text": ["STRING", {"forceInput": True}],
            "seed": ["INT", {"min": 0, "max": 2**64 - 1}],
            "steps": ["INT", {"min": 1, "max": 100}],
            "sampler": [["euler", "heun"]],
        }}, "output": ["IMAGE"]},
        "Save": {"input": {"required": {"image": ["IMAGE"]}}, "output": [], "output_node": True},
    }


@pytest.fixture
def workflow():
    return json.loads((Path(__file__).parent / "fixtures/studio/app_mode_canonical.json").read_text())


def test_lossless_roundtrip_and_advanced_bindings(workflow, info):
    original = copy.deepcopy(workflow)
    result = inspect_workflow(workflow, info)
    assert result["errors"] == []
    assert export_workflow(result) == original == workflow
    assert result["prompt"]["2"]["inputs"]["steps"] == 20
    binding = next(item for item in result["bindings"] if item["id"] == "main:1:text")
    assert binding["prominent"] is True
    assert binding["config"] == {"height": 120, "description": "Prompt", "future": {"x": 1}}
    assert any(item["input_name"] == "sampler" and not item["prominent"] for item in result["bindings"])
    assert result["outputs"][0]["node_id"] == "3"


def test_values_do_not_change_workflow_or_graph_labels(workflow, info):
    workflow["nodes"][0]["title"] = "Label with no identity meaning"
    workflow["extra"]["dgxm_studio"] = {"bindings": {"main:1:text": {"label": "Rename", "group": "Text"}}}
    result = inspect_workflow(workflow, info)
    assert next(item for item in result["bindings"] if item["id"] == "main:1:text")["label"] == "Rename"
    prompt = compile_workflow(workflow, info, {"main:1:text": "changed"})
    assert prompt["1"]["inputs"]["text"] == "changed"
    assert workflow["nodes"][0]["widgets_values"] == ["original"]


def test_legacy_and_encoded_identity(workflow, info):
    workflow["nodes"][0]["id"] = "one:two"
    workflow["links"][0][1] = "one:two"
    workflow["extra"]["linearData"]["inputs"] = [["main:one%3Atwo:text", "Display label"]]
    assert inspect_workflow(workflow, info)["errors"] == []
    workflow["nodes"][0]["id"] = 1
    workflow["links"][0][1] = 1
    workflow["extra"]["linearData"]["inputs"] = [[1, "text"]]
    assert inspect_workflow(workflow, info)["bindings"][0]["source_identity"] == 1
    assert binding_id("main", "one:two", "words / :") == "main:one%3Atwo:words%20%2F%20%3A"


def test_legacy_source_version_fixture(info):
    path = Path(__file__).parent / "fixtures/studio/app_mode_1_41_13.json"
    workflow = json.loads(path.read_text())
    result = inspect_workflow(workflow, info)
    assert result["errors"] == []
    assert result["bindings"][0]["id"] == "main:1:text"
    assert export_workflow(result) == workflow


def test_missing_and_custom_widget_are_preserved_and_rebound(workflow, info):
    workflow["extra"]["linearData"]["inputs"] = [[99, "removed", {"unknown": True}]]
    result = inspect_workflow(workflow, info)
    missing = result["bindings"][-1]
    assert missing["resolved"] is False
    assert missing["config"]["unknown"] is True
    with pytest.raises(WorkflowError, match="Resolve App Mode"):
        compile_workflow(workflow, info)
    repaired = rebind_workflow(workflow, missing["id"], "main:1:text", info)
    assert compile_workflow(repaired, info)["1"]["inputs"]["text"] == "original"
    assert repaired["extra"]["linearData"]["inputs"] == workflow["extra"]["linearData"]["inputs"]
    workflow["extra"]["linearData"]["inputs"] = [[1, "javascript_widget"]]
    assert "unsupported" in inspect_workflow(workflow, info)["bindings"][-1]["reason"]


def test_nested_graph_resolution_preserves_source_binding(workflow, info):
    child = {"id": "nested", "nodes": [{"id": 1, "type": "Text", "widgets_values": ["child"]}]}
    workflow["definitions"] = {"subgraphs": [child]}
    workflow["nodes"].append({"id": 8, "type": "nested"})
    workflow["extra"]["linearData"]["inputs"] = [["nested:1:text", "text"]]
    result = inspect_workflow(workflow, info)
    selected = next(item for item in result["bindings"] if item["id"] == "nested:1:text")
    assert selected["value"] == "child" and selected["resolved"]
    assert result["errors"] == []
    workflow["extra"]["linearData"]["inputs"] = [["8:1", "text"]]
    assert next(item for item in inspect_workflow(workflow, info)["bindings"]
                if item["id"] == "nested:1:text")["prominent"]


def test_ambiguous_legacy_node_is_not_resolved_by_label(workflow, info):
    workflow["definitions"] = {"subgraphs": [
        {"id": gid, "nodes": [{"id": 22, "type": "Text", "widgets_values": ["x"]}]}
        for gid in ["a", "b"]]}
    workflow["extra"]["linearData"]["inputs"] = [[22, "text"]]
    assert "Ambiguous" in inspect_workflow(workflow, info)["bindings"][-1]["reason"]


@pytest.mark.parametrize("value", [0, 101, True, "20", float("nan")])
def test_invalid_values_are_not_normalized(workflow, info, value):
    with pytest.raises(WorkflowError):
        compile_workflow(workflow, info, {"main:2:steps": value})


def test_unknown_and_linked_override_refused(workflow, info):
    for key in ["main:2:text", "main:9:missing"]:
        with pytest.raises(WorkflowError, match="Unknown, connected"):
            compile_workflow(workflow, info, {key: "oops"})


def test_seed_control_does_not_shift_widget_order(workflow, info):
    prompt = compile_workflow(workflow, info)
    assert prompt["2"]["inputs"] == {"text": ["1", 0], "seed": 42, "steps": 20, "sampler": "euler"}
    workflow["nodes"][1]["widgets_values"].append("custom button")
    with pytest.raises(WorkflowError, match="unsupported extra"):
        compile_workflow(workflow, info)


def test_primitive_and_reroute_are_resolved(workflow, info):
    workflow["extra"]["linearData"]["inputs"] = []
    workflow["nodes"][0].update(type="PrimitiveNode", widgets_values=["primitive text"])
    assert compile_workflow(workflow, info)["2"]["inputs"]["text"] == "primitive text"
    workflow["extra"]["linearData"]["inputs"] = [[1, "value"]]
    result = inspect_workflow(workflow, info)
    assert result["bindings"][0]["schema"]["type"] == "STRING"
    assert compile_workflow(workflow, info, {"main:1:value": "new"})["2"]["inputs"]["text"] == "new"
    workflow["nodes"].append({"id": 4, "type": "Reroute", "inputs": [{"name": "", "link": 3}]})
    workflow["links"].append([3, 1, 0, 4, 0, "STRING"])
    workflow["links"][0][1] = 4
    assert compile_workflow(workflow, info)["2"]["inputs"]["text"] == "primitive text"
    workflow["links"][-1][1] = 4
    workflow["extra"]["linearData"]["inputs"] = []
    with pytest.raises(WorkflowError, match="cycle"):
        compile_workflow(workflow, info)


def test_link_endpoint_and_slot_validation(workflow, info):
    workflow["links"][0][3] = 3
    with pytest.raises(WorkflowError, match="endpoint mismatch"):
        compile_workflow(workflow, info)
    workflow["links"][0][3] = 2
    workflow["links"][0][2] = 9
    with pytest.raises(WorkflowError, match="output slot"):
        compile_workflow(workflow, info)


def test_api_prompt_roundtrip_validation_and_cycle(workflow, info):
    prompt = compile_workflow(workflow, info)
    assert compile_workflow(prompt, info, {"root:2:steps": 15})["2"]["inputs"]["steps"] == 15
    prompt["1"]["inputs"]["text"] = ["1", 0]
    with pytest.raises(WorkflowError, match="cycle"):
        compile_workflow(prompt, info)


def test_video_output_and_duplicate_nodes(workflow, info):
    info["Sampler"]["output"] = ["VIDEO"]
    info["Save"]["input"]["required"]["image"] = ["VIDEO"]
    assert compile_workflow(workflow, info)["3"]["inputs"]["image"] == ["2", 0]
    workflow["nodes"].append(copy.deepcopy(workflow["nodes"][0]))
    with pytest.raises(WorkflowError, match="Duplicate node"):
        inspect_workflow(workflow, info)


@pytest.mark.parametrize("mutation", [
    lambda w: w.update(extra=[]),
    lambda w: w["extra"].update(linearData=[]),
    lambda w: w.update(definitions=[]),
    lambda w: w["nodes"][0].update(type=[]),
])
def test_malformed_graph_metadata_returns_typed_refusal(workflow, info, mutation):
    mutation(workflow)
    with pytest.raises(WorkflowError):
        inspect_workflow(workflow, info)


def test_presets_undo_and_ordered_styles():
    draft = {"prompt": "cat", "seed": 4, "loras": [{"id": "a", "enabled": False}]}
    applied = apply_preset(draft, {"fields": {"seed": 8}})
    assert applied["draft"]["loras"] == draft["loras"]
    assert applied["replaced_fields"] == ["seed"]
    assert undo_preset(applied) == draft
    styles = [{"id": "a", "prompt": "painting of {prompt}"}, {"id": "b", "prompt": "framed {prompt}"}]
    transformed = apply_styles("cat", "bad", styles)
    assert transformed["prompt"] == "framed painting of cat"
    assert transformed["unexpanded"] == {"prompt": "cat", "negative_prompt": "bad"}


def test_grid_enumerates_invalid_cells_and_explicit_exclusions():
    axes = [{"name": "steps", "values": [1, 0]}, {"name": "cfg", "values": [1.0, 2.0]}]
    schemas = {"steps": {"type": "INT", "min": 1}, "cfg": {"type": "FLOAT"}}
    result = expand_grid({}, axes, schemas, seed_policy="increment", base_seed=10)
    assert result["total"] == 4 and not result["ready"]
    assert [cell["parameters"]["seed"] for cell in result["cells"]] == [10, 11, 12, 13]
    result = expand_grid({}, axes, schemas, seed_policy="fixed", base_seed=10, excluded=[2, 3])
    assert result["ready"] and result["job_count"] == 2 and len(result["cells"]) == 4
    with pytest.raises(DraftError, match="1024"):
        expand_grid({}, [{"name": "steps", "values": list(range(1025))}], schemas,
                    seed_policy="fixed", base_seed=1)


def test_lora_uses_catalog_compatibility_and_preserves_disabled_draft():
    stack = [{"id": "missing", "enabled": False, "strength": 1}, {"id": "known", "strength": 0.8}]
    catalog = {"known": {"available": True, "compatible_models": ["model"],
                         "strengths": {"strength": {"type": "FLOAT", "min": -2, "max": 2}}}}
    assert resolve_loras(stack, catalog, "model")[0]["strengths"] == {"strength": 0.8}
    assert len(stack) == 2
    with pytest.raises(DraftError, match="compatibility"):
        resolve_loras(stack, catalog, "other")
