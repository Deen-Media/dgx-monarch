"""API-format envelope and node-allowlist checks shared by the workflow tests.

Each caller passes its own workflow and its own `_KNOWN_CLASS_TYPES` allowlist.
"""
from __future__ import annotations

import json


def load_workflow_prompt(fixture_path: str) -> dict:
    """Return a fixture's `prompt` graph; its envelope must be exactly `_meta` and
    `prompt`, with `title` and `usage` in `_meta`."""
    with open(fixture_path, encoding="utf-8") as fh:
        document = json.load(fh)
    assert set(document.keys()) == {"_meta", "prompt"}
    assert {"title", "usage"} <= set(document["_meta"].keys())
    return document["prompt"]


def assert_uses_only_known_node_classes(prompt: dict, allowlist: frozenset[str]) -> None:
    """Every node in `prompt` must carry `class_type` and `inputs`, with its class_type in `allowlist`."""
    assert prompt, "workflow fixture must not be empty"
    for node_id, node in prompt.items():
        assert "class_type" in node, f"node {node_id!r} missing class_type"
        assert "inputs" in node, f"node {node_id!r} missing inputs"
        assert node["class_type"] in allowlist, (
            f"node {node_id!r} uses unrecognized class_type {node['class_type']!r}"
        )
