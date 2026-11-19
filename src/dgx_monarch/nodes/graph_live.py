"""Which nodes of a prompt graph ComfyUI will run.

The loader preflight prices the models a graph loads on the driver
(``loader_graph``), and a model that sits only behind a lazy switch's other
branch never loads. Pure: no ComfyUI import, no file read. A graph this cannot
read keeps every node, which is the charge the preflight made before this walk.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# Lazy switches whose literal `switch` lets ComfyUI evaluate one branch only
# (comfy_extras/nodes_logic.py, check_lazy_status); read-only after import.
SWITCH_CLASSES = ("ComfySwitchNode", "ComfySoftSwitchNode")
# ComfyUI's display names for them, read-only after import.
SWITCH_TITLES = {"ComfySwitchNode": "If/Else Switch", "ComfySoftSwitchNode": "Soft Switch"}
# The two inputs a switch may leave unevaluated, read-only after import.
BRANCH_INPUTS = frozenset({"on_true", "on_false"})
# ComfyUI's lazy node this walk does not decide (comfy_extras/nodes_logic.py,
# BranchNode at 0.45.0): which of its inputs runs is not read here.
UNREAD_LAZY_CLASSES = ("BranchNode",)


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


def output_node(class_type: object) -> bool | None:
    """ComfyUI's own answer to whether a class is an output node, or None unknown.

    The test is the one execution.py makes (``OUTPUT_NODE is True``), read from
    the registry ComfyUI fills in the driver process; headless, nothing is known.
    """
    try:
        import nodes as comfy_nodes

        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
    except Exception:
        return None
    return None if cls is None else getattr(cls, "OUTPUT_NODE", False) is True


def live_nodes(prompt: Mapping, *, roots: Iterable[str] | None = None,
               sure: bool = False) -> frozenset[str]:
    """The nodes ComfyUI will run: those an output node reaches through links.

    ComfyUI runs the ancestors of its output nodes. A node whose class the
    registry cannot place counts as one when nothing consumes it, so an
    unknown class is charged rather than dropped. The walk follows links
    backwards and skips a lazy switch's unselected branch, so a model loader
    that feeds only that branch, or nothing that runs, is not priced.

    ``roots`` starts the same walk from the named nodes in place of the output
    nodes: what those nodes take their inputs from, and nothing a preview or a
    save elsewhere in the graph runs beside them.

    A switch whose choice the graph does not state keeps both branches, the
    safe side for a price. ``sure`` asks the other question, which nodes
    ComfyUI is certain to run: such a switch still runs whatever feeds its
    ``switch`` input, since ComfyUI needs that value to choose, and neither
    branch is followed. A Branch node's inputs are not followed at all.
    """
    nodes = {str(key): node for key, node in prompt.items() if isinstance(node, Mapping)}
    if roots is not None:
        pending = [str(key) for key in roots]
    else:
        consumed: set[str] = set()
        for node in nodes.values():
            inputs = node.get("inputs")
            for value in inputs.values() if isinstance(inputs, Mapping) else ():
                consumed.update(str(link[0]) for link in links_in(value))
        outputs = {key: output_node(node.get("class_type")) for key, node in nodes.items()}
        pending = [key for key in nodes
                   if outputs[key] is True or (outputs[key] is None and key not in consumed)]
    live: set[str] = set()
    while pending:
        key = pending.pop()
        if key in live or key not in nodes:
            continue
        live.add(key)
        inputs = nodes[key].get("inputs")
        kind = nodes[key].get("class_type")
        if sure and kind in UNREAD_LAZY_CLASSES:
            continue
        branch = skipped_branch(nodes[key])
        skipped = BRANCH_INPUTS if sure and branch is None and kind in SWITCH_CLASSES else {branch}
        for name, value in inputs.items() if isinstance(inputs, Mapping) else ():
            if name not in skipped:
                pending.extend(str(link[0]) for link in links_in(value))
    return frozenset(live)
