"""Layout metadata stays editable without becoming an executable API node."""

from copy import deepcopy

from dgx_monarch.studio.jobs import Jobs
from dgx_monarch.studio.store import Store
from dgx_monarch.studio.workflows import compile_workflow, inspect_workflow, is_api_prompt

SCHEMA = {"Text": {"input": {"required": {"text": ["STRING", {}]}}, "output": [], "output_node": True}}


def test_api_layout_edits_preserve_bindings_and_strip_execution_metadata(tmp_path):
    workflow = {"1": {"class_type": "Text", "inputs": {"text": "cat"}},
                "extra": {"dgxm_studio": {"bindings": {"root:1:text": {
                    "label": "Subject", "prominent": True, "order": 2, "group": "Prompt"}}},
                          "future_metadata": {"keep": True}}}
    original = deepcopy(workflow)
    assert is_api_prompt(workflow)
    inspection = inspect_workflow(workflow, SCHEMA)
    assert inspection["errors"] == []
    binding = inspection["bindings"][0]
    assert (binding["id"], binding["label"], binding["prominent"], binding["order"], binding["group"]) == (
        "root:1:text", "Subject", True, 2, "Prompt")
    prompt = compile_workflow(workflow, SCHEMA, {"root:1:text": "dog"})
    assert prompt == {"1": {"class_type": "Text", "inputs": {"text": "dog"}}}
    store = Store(tmp_path / "state")
    try:
        jobs = Jobs(store, None)
        preview = jobs.build_preview({"workflow": workflow, "values": {"root:1:text": "dog"}}, schema=SCHEMA)
        assert preview["workflow"] == original and preview["prompt"] == prompt
        assert store.get("previews", preview["id"])["workflow"]["extra"] == original["extra"]
    finally:
        store.close()
    assert workflow == original


def test_existing_api_node_named_extra_remains_executable():
    workflow = {"extra": {"class_type": "Text", "inputs": {"text": "cat"}}}
    assert is_api_prompt(workflow)
    assert compile_workflow(workflow, SCHEMA) == workflow


def test_unrecognized_non_node_fields_are_not_silently_dropped():
    assert not is_api_prompt({"1": {"class_type": "Text", "inputs": {"text": "cat"}}, "surprise": {}})
