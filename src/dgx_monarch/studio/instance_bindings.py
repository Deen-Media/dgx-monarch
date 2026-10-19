"""Stable per-instance projections of serialized subgraph widget identities."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .subgraph_compile import Compiler
from .workflows import WorkflowError


def project_bindings(workflow: dict[str, Any], info: dict[str, Any], bindings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compiler = Compiler(workflow, info, {})
    projected = []
    for context in compiler.contexts:
        for original in bindings:
            if original["graph_id"] != context.gid:
                continue
            nid, name = original["node_id"], original["input_name"]
            binding = deepcopy(original)
            binding.update(id=context.binding(nid, name), instance_path=list(context.path),
                           execution_node_id=context.execution_id(nid), definition_binding_id=original["id"],
                           aliases=[original["id"]] if context.path and compiler.counts[context.gid] == 1 else [])
            if (nid, name) in context.overrides:
                binding["value"] = deepcopy(context.overrides[nid, name])
            if context.path:
                binding["label"] = f"{' / '.join(context.path)} · {original['label']}"
            projected.append(binding)
    return projected


def select_binding(identity: Any, name: str, bindings: list[dict[str, Any]], canonical: str) -> dict[str, Any] | None:
    direct = [row for row in bindings if identity == row["id"] or canonical == row["id"]
              or canonical in row.get("aliases", [])]
    if len(direct) == 1:
        return direct[0]
    # Legacy paths name actual host instances, whereas canonical definition IDs may be shared.
    path = str(identity).split(":")
    if len(path) > 1:
        from urllib.parse import unquote
        path = [unquote(part) for part in path]
        exact = [row for row in bindings if [*row.get("instance_path", []), row["node_id"]] == path
                 and row["input_name"] == name]
        if len(exact) == 1:
            return exact[0]
    if any(row.get("definition_binding_id") == canonical for row in bindings):
        raise WorkflowError("Shared subgraph binding is ambiguous; choose a specific instance")
    return None
