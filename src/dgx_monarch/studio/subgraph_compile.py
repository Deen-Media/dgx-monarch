"""Compile serialized subgraph instances without loading frontend JavaScript.

Subgraph interfaces use their serialized virtual input/output node identities.
Every instance gets separate execution IDs; source graph identities remain the
keys for draft edits. Ambiguous edits to shared definitions are refused.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from .workflows import SCALAR_TYPES, WorkflowError, binding_id, graph_index, node_inputs, widget_values

_MISSING = object()


@dataclass
class Context:
    gid: str
    graph: dict[str, Any]
    nodes: dict[str, Any]
    links: dict[str, tuple[str, int, str, int]]
    path: tuple[str, ...]
    parent: Context | None = None
    host: dict[str, Any] | None = None
    children: dict[str, Context] = field(default_factory=dict)
    overrides: dict[tuple[str, str], Any] = field(default_factory=dict)

    def execution_id(self, nid: str) -> str:
        if not self.path:
            return nid
        return "__studio_subgraph__" + "/".join(quote(part, safe="") for part in (*self.path, nid))


class Compiler:
    def __init__(self, workflow: dict[str, Any], info: dict[str, Any], values: dict[str, Any]):
        from .workflow_compile import _link_rows
        self.info, self.values = info, values
        self.graphs = graph_index(workflow)
        self.contexts: list[Context] = []

        def expand(gid: str, path: tuple[str, ...], ancestry: frozenset[str],
                   parent: Context | None = None, host: dict[str, Any] | None = None) -> Context:
            if gid in ancestry or len(path) > 32:
                raise WorkflowError("Recursive subgraph definitions cannot execute")
            graph = self.graphs[gid]
            context = Context(gid, graph["graph"], graph["nodes"], _link_rows(graph["graph"]), path, parent, host)
            self.contexts.append(context)
            if sum(len(item.nodes) for item in self.contexts) > 10000:
                raise WorkflowError("Expanded workflow exceeds 10000 nodes")
            for nid, node in context.nodes.items():
                if node.get("mode", 0) in {2, 4}:
                    continue
                child = node.get("subgraph")
                child_id = str(child.get("id", f"{gid}/node/{nid}")) if isinstance(child, dict) else node["type"]
                if child_id in self.graphs:
                    context.children[nid] = expand(child_id, (*path, nid), ancestry | {gid}, context, node)
            return context

        self.root = expand(next(iter(self.graphs)), (), frozenset())
        counts: dict[str, int] = {}
        for context in self.contexts:
            counts[context.gid] = counts.get(context.gid, 0) + 1
        for key in values:
            if counts.get(key.rsplit(":", 2)[0], 0) > 1:
                raise WorkflowError("Shared subgraph binding is ambiguous; select an instance-specific API workflow")
        for context in self.contexts:
            self.promoted_values(context)

    def promoted_values(self, context: Context) -> None:
        if context.host is None:
            return
        values = context.host.get("widgets_values", [])
        proxies = context.host.get("properties", {}).get("proxyWidgets", [])
        if not proxies:
            return  # Scalar interface values are resolved by name in external_input.
        if not isinstance(proxies, list) or not isinstance(values, list) or len(values) != len(proxies):
            raise WorkflowError("Promoted subgraph widgets need an explicit, ordered typed mapping")
        for proxy, value in zip(proxies, values, strict=True):
            if not isinstance(proxy, list) or len(proxy) != 2 or not isinstance(proxy[1], str):
                raise WorkflowError("Malformed promoted widget identity")
            nid, name = str(proxy[0]), proxy[1]
            node = context.nodes.get(nid)
            if node is None or nid in context.children:
                raise WorkflowError("Promoted widget must resolve to one typed node input")
            fields = {name: schema for name, schema, _ in node_inputs(self.info.get(node["type"], {}))}
            if name not in fields or fields[name]["type"] not in SCALAR_TYPES:
                raise WorkflowError("Promoted widget has no supported scalar type")
            context.overrides[nid, name] = copy.deepcopy(value)

    def linked(self, context: Context, nid: str, index: int, socket: dict[str, Any], seen: frozenset[Any]) -> Any:
        if socket.get("link") is None:
            return _MISSING
        link = context.links.get(str(socket["link"]))
        if link is None or link[2:] != (nid, index):
            raise WorkflowError("Subgraph input link endpoint mismatch")
        return self.source(context, link[0], link[1], seen)

    def external_input(self, context: Context, slot: int, seen: frozenset[Any]) -> Any:
        interfaces = context.graph.get("inputs", [])
        if not isinstance(interfaces, list) or not 0 <= slot < len(interfaces):
            raise WorkflowError("Subgraph input slot is missing")
        interface = interfaces[slot]
        if context.parent is None or context.host is None:
            raise WorkflowError("Subgraph has no owning instance")
        sockets = context.host.get("inputs", [])
        for index, socket in enumerate(sockets):
            if socket.get("name") == interface.get("name"):
                result = self.linked(context.parent, str(context.host["id"]), index, socket, seen)
                if result is not _MISSING:
                    return result
        # Promoted scalar interface widgets are serialized in declaration order.
        scalar = [item for item in interfaces if item.get("type") in SCALAR_TYPES]
        serialized = context.host.get("widgets_values", [])
        if interface in scalar and not context.host.get("properties", {}).get("proxyWidgets"):
            position = scalar.index(interface)
            if isinstance(serialized, list) and position < len(serialized):
                return copy.deepcopy(serialized[position])
            if isinstance(serialized, dict) and interface.get("name") in serialized:
                return copy.deepcopy(serialized[interface["name"]])
        return _MISSING

    def source(self, context: Context, nid: str, slot: int, seen: frozenset[Any]) -> Any:
        key = (context.path, nid, slot)
        if key in seen or len(seen) > 256:
            raise WorkflowError("Subgraph link cycle or excessive depth")
        seen = seen | {key}
        input_node = str(context.graph.get("inputNode", {}).get("id", "-10"))
        if context.parent is not None and nid == input_node:
            return self.external_input(context, slot, seen)
        node = context.nodes.get(nid)
        if node is None:
            raise WorkflowError("Subgraph link references a missing node")
        if node.get("mode", 0) == 2:
            raise WorkflowError("A required source node is muted")
        if node.get("mode", 0) == 4:
            outputs, inputs = node.get("outputs", []), node.get("inputs", [])
            if not 0 <= slot < len(outputs):
                raise WorkflowError("Bypass output slot is missing")
            candidates = [(index, item) for index, item in enumerate(inputs)
                          if item.get("type") == outputs[slot].get("type") and item.get("link") is not None]
            if len(candidates) != 1:
                raise WorkflowError("Bypass needs one unambiguous input of the output type")
            return self.linked(context, nid, *candidates[0], seen)
        if node.get("mode", 0) != 0:
            raise WorkflowError("Unsupported node execution mode")
        if nid in context.children:
            child = context.children[nid]
            output_node = str(child.graph.get("outputNode", {}).get("id", "-20"))
            output_links = [link for link in child.links.values() if link[2:] == (output_node, slot)]
            if len(output_links) != 1:
                raise WorkflowError("Subgraph output requires exactly one source")
            return self.source(child, output_links[0][0], output_links[0][1], seen)
        if node["type"] == "Reroute":
            sockets = node.get("inputs", [])
            if slot != 0 or len(sockets) != 1:
                raise WorkflowError("Reroute needs one input")
            return self.linked(context, nid, 0, sockets[0], seen)
        if node["type"] == "PrimitiveNode":
            serialized = node.get("widgets_values", [])
            if slot != 0 or not isinstance(serialized, list) or not 1 <= len(serialized) <= 2:
                raise WorkflowError("Unsupported primitive serialization")
            if len(serialized) == 2 and serialized[1] not in {"fixed", "increment", "decrement", "randomize"}:
                raise WorkflowError("Unsupported primitive control")
            return copy.deepcopy(self.values.get(binding_id(context.gid, nid, "value"), serialized[0]))
        return [context.execution_id(nid), slot]

    def compile(self) -> dict[str, Any]:
        from .workflow_compile import _validate_prompt
        prompt: dict[str, Any] = {}
        for context in self.contexts:
            for nid, node in context.nodes.items():
                if nid in context.children or node["type"] in {"PrimitiveNode", "Reroute", "Note", "MarkdownNote"}:
                    continue
                if node.get("mode", 0) in {2, 4}:
                    continue
                if node.get("mode", 0) != 0:
                    raise WorkflowError("Unsupported node execution mode")
                definition = self.info.get(node["type"])
                if definition is None:
                    raise WorkflowError(f"Node type unavailable: {node['type']}")
                fields = node_inputs(definition)
                serialized = widget_values(node, definition)
                sockets = node.get("inputs", [])
                if not isinstance(sockets, list) or any(not isinstance(item, dict) for item in sockets):
                    raise WorkflowError("Malformed subgraph input sockets")
                by_name = {item.get("name"): (index, item) for index, item in enumerate(sockets)}
                if len(by_name) != len(sockets) or set(by_name) - {name for name, _, _ in fields}:
                    raise WorkflowError("Duplicate or unsupported subgraph input sockets")
                inputs: dict[str, Any] = {}
                for name, schema, _ in fields:
                    key = binding_id(context.gid, nid, name)
                    index, socket = by_name.get(name, (-1, {}))
                    value = self.linked(context, nid, index, socket, frozenset())
                    if value is not _MISSING and key in self.values:
                        raise WorkflowError("Connected input cannot be overridden")
                    if value is _MISSING:
                        value = self.values.get(key, context.overrides.get((nid, name), serialized.get(name, schema.get("default", _MISSING))))
                    if value is not _MISSING:
                        inputs[name] = copy.deepcopy(value)
                execution_id = context.execution_id(nid)
                if execution_id in prompt:
                    raise WorkflowError("Expanded subgraph execution identity collision")
                prompt[execution_id] = {"class_type": node["type"], "inputs": inputs}
        return _validate_prompt(prompt, self.info)


def compile_subgraphs(workflow: dict[str, Any], info: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    return Compiler(workflow, info, values).compile()
