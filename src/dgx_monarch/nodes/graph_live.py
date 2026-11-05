"""Which nodes of a prompt graph ComfyUI will run.

The loader preflight prices the models a graph loads on the driver
(``loader_graph``), and a model that sits only behind a lazy switch's other
branch never loads. Pure: no ComfyUI import, no file read. A graph this cannot
read keeps every node, which is the charge the preflight made before this walk.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# Lazy switches whose literal `switch` lets ComfyUI evaluate one branch only
# (comfy_extras/nodes_logic.py, check_lazy_status); read-only after import.
SWITCH_CLASSES = ("ComfySwitchNode", "ComfySoftSwitchNode")
# ComfyUI's display names for them, read-only after import.
SWITCH_TITLES = {"ComfySwitchNode": "If/Else Switch", "ComfySoftSwitchNode": "Soft Switch"}


def is_link(value: object) -> bool:
    """Comfy API-format link: ``[node_id, output_index]``."""
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[0], (str, int))
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def links_in(value: object) -> list[Any]:
    """The links one input holds, through one Autogrow container level."""
    if is_link(value):
        return [value]
    items = value.values() if isinstance(value, Mapping) else (
        value if isinstance(value, (list, tuple)) else ())
    return [item for item in items if is_link(item)]


def skipped_branch(node: Mapping) -> str | None:
    """The input a lazy switch never evaluates, when its literal choice says so.

    Only a literal boolean with both branches linked decides it: a linked
    switch, or one branch left open, keeps every input, as ComfyUI may
    evaluate either.
    """
    inputs = node.get("inputs")
    if node.get("class_type") not in SWITCH_CLASSES or not isinstance(inputs, Mapping):
        return None
    choice = inputs.get("switch")
    if not isinstance(choice, bool) or not all(
            is_link(inputs.get(key)) for key in ("on_true", "on_false")):
        return None
    return "on_false" if choice else "on_true"


def live_nodes(prompt: Mapping) -> frozenset[str]:
    """The nodes ComfyUI will run: those a sink reaches through links.

    A sink is a node nothing consumes, an output node among them. The walk
    follows links backwards and skips a lazy switch's unselected branch, so a
    model loader that feeds only that branch never loads and is not priced.
    """
    nodes = {str(key): node for key, node in prompt.items() if isinstance(node, Mapping)}
    consumed: set[str] = set()
    for node in nodes.values():
        inputs = node.get("inputs")
        for value in inputs.values() if isinstance(inputs, Mapping) else ():
            consumed.update(str(link[0]) for link in links_in(value))
    pending = [key for key in nodes if key not in consumed]
    live: set[str] = set()
    while pending:
        key = pending.pop()
        if key in live or key not in nodes:
            continue
        live.add(key)
        inputs = nodes[key].get("inputs")
        skipped = skipped_branch(nodes[key])
        for name, value in inputs.items() if isinstance(inputs, Mapping) else ():
            if name != skipped:
                pending.extend(str(link[0]) for link in links_in(value))
    return frozenset(live)
