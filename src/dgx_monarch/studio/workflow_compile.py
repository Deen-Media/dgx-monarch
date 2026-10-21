"""Conservative conversion of serialized workflows into Comfy API prompts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .drafts import DraftError, validate_value
from .widget_values import (
    DYNAMIC_COMBO,
    LINKED,
    MISSING,
    field_schema,
    finalize_inputs,
    is_link,
    layout_values,
    link_accepts,
    resolve_inputs,
)
from .workflows import SCALAR_TYPES, WorkflowError, api_prompt_nodes, binding_id, graph_index, is_api_prompt

_DECORATIVE = {"Note", "MarkdownNote"}
_VIRTUAL = {"PrimitiveNode", "Reroute"}


def _link_rows(graph: Mapping[str, Any]) -> dict[str, tuple[str, int, str, int]]:
    links = {}
    for row in graph.get("links", []):
        if isinstance(row, dict):
            values = [row.get(key) for key in ("id", "origin_id", "origin_slot", "target_id", "target_slot")]
        elif isinstance(row, list) and len(row) >= 5:
            values = row[:5]
        else:
            raise WorkflowError("Unsupported serialized link")
        identity, source, source_slot, target, target_slot = values
        if (identity is None or source is None or target is None
                or type(source_slot) is not int or type(target_slot) is not int
                or source_slot < 0 or target_slot < 0):
            raise WorkflowError("Invalid serialized link endpoint")
        if str(identity) in links:
            raise WorkflowError("Duplicate link identity")
        links[str(identity)] = (str(source), source_slot, str(target), target_slot)
    return links


def _cycle_check(prompt: Mapping[str, Any]) -> None:
    active: set[str] = set()
    done: set[str] = set()

    def visit(identity: str) -> None:
        if identity in active:
            raise WorkflowError("Workflow contains a dependency cycle")
        if identity in done:
            return
        active.add(identity)
        for value in prompt[identity]["inputs"].values():
            if is_link(value) and value[0] in prompt:
                visit(value[0])
        active.remove(identity)
        done.add(identity)

    try:
        for identity in prompt:
            visit(identity)
    except RecursionError as exc:
        raise WorkflowError("Workflow dependency depth is unsupported") from exc


def _validate_prompt(prompt: dict[str, Any], info: Mapping[str, Any]) -> dict[str, Any]:
    """Validate as ComfyUI's validate_inputs does, on its finalized declared inputs.

    ComfyUI skips inputs a node does not declare and still passes them to the
    node, as rgthree's Power Lora Loader relies on for ``lora_1..N``. Studio
    keeps them in the prompt and checks only that a link among them names a node.
    """
    for nid, node in prompt.items():
        definition = info.get(node["class_type"])
        if definition is None:
            raise WorkflowError(f"Node type unavailable: {node['class_type']}")
        fields = finalize_inputs(definition, node["inputs"])
        declared = {name for name, _, _ in fields}
        for name, value in node["inputs"].items():
            if name not in declared and is_link(value) and value[0] not in prompt:
                raise WorkflowError(f"Link references missing node: {nid}.{name}")
        for name, spec, required in fields:
            if name not in node["inputs"]:
                if required:
                    raise WorkflowError(f"Required input is missing: {nid}.{name}")
                continue
            value = node["inputs"][name]
            if isinstance(value, list):
                if len(value) != 2 or not isinstance(value[0], str) or type(value[1]) is not int:
                    raise WorkflowError(f"Malformed input link: {nid}.{name}")
                source = prompt.get(value[0])
                if source is None:
                    raise WorkflowError(f"Link references missing node: {nid}.{name}")
                source_info = info.get(source["class_type"], {})
                output_types = source_info.get("output", [])
                if not 0 <= value[1] < len(output_types):
                    raise WorkflowError(f"Link references invalid output slot: {nid}.{name}")
                # ComfyUI never type-checks a linked DynamicCombo: the link
                # selects no option, so the input is not declared there.
                if spec[0] != DYNAMIC_COMBO and not link_accepts(output_types[value[1]], spec[0]):
                    raise WorkflowError(f"Incompatible input link: {nid}.{name}")
            else:
                try:
                    validate_value(value, field_schema(name, spec))
                except DraftError as exc:
                    raise WorkflowError(f"{nid}.{name}: {exc}") from exc
    _cycle_check(prompt)
    if not any(info[node["class_type"]].get("output_node", False) for node in prompt.values()):
        raise WorkflowError("Workflow has no available output node")
    return prompt


def compile_graph(workflow: Mapping[str, Any], info: Mapping[str, Any],
                  values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert explicitly supported core serialization; refuse custom graph code."""
    if not info:
        raise WorkflowError("Driver object_info is required to validate workflow inputs")
    graphs = graph_index(workflow)
    gid = next(iter(graphs))
    root = graphs[gid]
    nodes = root["nodes"]
    if is_api_prompt(workflow):
        prompt = copy.deepcopy(api_prompt_nodes(workflow))
        for nid, node in prompt.items():
            if not isinstance(node.get("inputs"), dict):
                raise WorkflowError(f"Node {nid} inputs must be an object")

            def edit(name: str, nid: str = str(nid)) -> Any:
                identity = binding_id("root", nid, name)
                return copy.deepcopy(values[identity]) if identity in values else MISSING

            for name, _, _ in resolve_inputs(info.get(node["class_type"], {}), node["inputs"], edit,
                                             fill_defaults=False):
                if (value := edit(name)) is not MISSING:
                    node["inputs"][name] = value
        return _validate_prompt(prompt, info)
    if len(graphs) > 1 or any(node.get("mode", 0) in {2, 4} for node in nodes.values()):
        from .subgraph_compile import compile_subgraphs
        return compile_subgraphs(dict(workflow), dict(info), dict(values))
    links = _link_rows(root["graph"])
    prompt = {}

    def source_value(source_id: str, slot: int, seen: frozenset[str]) -> Any:
        if len(seen) > 256:
            raise WorkflowError("Virtual node dependency depth exceeds 256")
        if source_id in seen:
            raise WorkflowError("Virtual node link cycle")
        seen = seen | {source_id}
        source = nodes.get(source_id)
        if source is None:
            raise WorkflowError("Link references a missing node")
        kind = source.get("type")
        if source.get("mode", 0) != 0:
            raise WorkflowError("Muted or bypassed connected nodes require explicit graph conversion")
        if kind == "PrimitiveNode":
            serialized = source.get("widgets_values", [])
            if slot != 0 or not isinstance(serialized, list) or not serialized:
                raise WorkflowError("Unsupported primitive serialization")
            if len(serialized) > 2 or (len(serialized) == 2 and serialized[1] not in
                                      {"fixed", "increment", "decrement", "randomize"}):
                raise WorkflowError("Custom primitive widgets require an explicit adapter")
            return copy.deepcopy(values.get(binding_id(gid, source_id, "value"), serialized[0]))
        if kind == "Reroute":
            sockets = source.get("inputs", [])
            if slot != 0 or len(sockets) != 1 or sockets[0].get("link") is None:
                raise WorkflowError("Reroute has no unique connected source")
            link = links.get(str(sockets[0]["link"]))
            if link is None or link[2:] != (source_id, 0):
                raise WorkflowError("Reroute link endpoint mismatch")
            return source_value(link[0], link[1], seen)
        return [source_id, slot]

    for nid, node in nodes.items():
        kind = node.get("type")
        if kind in _DECORATIVE or kind in _VIRTUAL:
            continue
        if kind in graphs or "subgraph" in node:
            raise WorkflowError("Subgraph execution needs a frontend-exported API prompt; graph metadata is preserved")
        if node.get("mode", 0) != 0:
            raise WorkflowError("Muted or bypassed nodes require a frontend-exported API prompt")
        definition = info.get(kind)
        if definition is None:
            raise WorkflowError(f"Node type unavailable: {kind}")
        serialized = layout_values(node, definition)
        inputs = {}
        sockets = node.get("inputs", [])
        if not isinstance(sockets, list):
            raise WorkflowError("UI workflow inputs must be an array")
        names = [socket.get("name") for socket in sockets if isinstance(socket, dict)]
        if len(set(names)) != len(sockets):
            raise WorkflowError("Duplicate or malformed input sockets")
        by_name = {socket["name"]: (index, socket) for index, socket in enumerate(sockets)}
        linked = {name for name, (_, socket) in by_name.items() if socket.get("link") is not None}

        def edit(name: str, nid: str = nid) -> Any:
            return values.get(binding_id(gid, nid, name), MISSING)

        fields = resolve_inputs(definition, {**serialized, **dict.fromkeys(linked, LINKED)}, edit,
                                fill_defaults=True)
        # Unlinked helper sockets (upload buttons, media players) carry nothing;
        # a link into an input the schema does not declare cannot be typed.
        if linked - {name for name, _, _ in fields}:
            raise WorkflowError("Custom input sockets require an explicit typed adapter")
        for name, spec, _ in fields:
            schema = field_schema(name, spec)
            identity = binding_id(gid, nid, name)
            socket_index, socket = by_name.get(name, (-1, {}))
            if socket.get("link") is not None:
                if identity in values:
                    raise WorkflowError("Connected input cannot be overridden; edit its source")
                link = links.get(str(socket["link"]))
                if link is None or link[2:] != (nid, socket_index):
                    raise WorkflowError("Input link endpoint mismatch")
                inputs[name] = source_value(link[0], link[1], frozenset())
            elif identity in values:
                inputs[name] = copy.deepcopy(values[identity])
            elif name in serialized:
                inputs[name] = serialized[name]
            elif "default" in schema and schema["type"] in SCALAR_TYPES:
                inputs[name] = copy.deepcopy(schema["default"])
        prompt[nid] = {"class_type": kind, "inputs": inputs}
    return _validate_prompt(prompt, info)
