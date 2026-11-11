"""Refuse a prompt expander that cannot receive the request it is meant to rewrite.

ComfyUI's MiniMax-H3-Image templates splice the request into a task prompt with
Replace Text, hand the result to Generate Text, and read the caption out of the
answer with Extract Text from JSON. The DGX Monarch templates keep that chain,
with the switch that selects the expander off. The two shipped templates carry
this project's own task prompt in the box Replace Text reads; the testing graph
ships that box empty, and a user may empty or replace it. Replace Text over an
empty text is an empty text whatever the request holds, so with the switch on
Generate Text answers an empty prompt and the render follows what comes of
that, with no error anywhere. A task prompt that lacks the marker Replace Text
looks for fails the same way more quietly: the expander runs, and the request
never reaches it.

A DGX Monarch loader, sampler or Identity Gate receives the whole prompt graph,
so it reads this chain there and refuses, class P, before the DiT loads or a
picture renders. The loader answers on a first queue. On a re-queue ComfyUI
keeps the loader's output and runs the expander before it reaches the sampler,
so the sampler's refusal comes after the expander has answered, and still
ahead of the render.

Only what the graph states is read, because a refusal of a graph that would
have rendered well is worse than none. Generate Text must be a node ComfyUI
is sure to run for a DGX Monarch render: the walk (graph_live, ``sure``)
starts at the samplers, Identity Gate and Fleet sampler that are themselves
sure to run, not at every output node, so an expander that runs only for a
Preview as Text beside the path, with the switch off, refuses nothing.
Generate Text must be fed by a Replace Text node whose text is typed or comes
from a text box.

A switch counts by what it states: a typed boolean, or a Boolean box wired to
it, as ComfyUI's own edit template has. Behind a switch that states no
choice, neither branch is sure to run, so an expander there refuses nothing,
and the same holds behind a Branch node. What feeds such a switch's own
``switch`` input is sure to run, though: ComfyUI needs that value to choose.
The shipped graphs use this. Their empty-caption fallback is a second If/Else
whose ``switch`` reads Compare Text over the expander's caption, so with the
outer switch true the expander runs whichever way the fallback goes, and the
refusal still answers. Any graph this cannot read refuses nothing. A lazy node
of another pack is not known here and reads as running all its inputs.

Pure: no ComfyUI import and no file read.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..adapters.base import UnsupportedModelError
from ..refusal import RefusalClass, refusal
from . import graph_live

GENERATE_CLASS = "TextGenerateV2"
REPLACE_CLASS = "StringReplace"
BOOLEAN_CLASS = "PrimitiveBoolean"
# Text boxes whose one widget is the text they hand on; read-only after import.
TEXT_CLASSES = ("PrimitiveStringMultiline", "PrimitiveString")
# The DGX Monarch nodes that render and call refuse_dead_expander themselves.
# Read-only after import.
RENDER_CLASSES = (
    "DGXMonarchKSampler", "DGXMonarchKSamplerPipeline", "DGXMonarchKSamplerAdvanced",
    "DGXMonarchSamplerCustom", "DGXMonarchIdentityGate")
# Where the walk starts: every node a prompt reaches a picture through. The
# Fleet sampler renders too. It receives no prompt graph, so it cannot refuse
# itself and relies on the UNET loader's call, which reads the whole graph.
ROOT_CLASSES = (*RENDER_CLASSES, "DGXMonarchFleetKSampler")
TROUBLESHOOTING = 153

EMPTY_TASK_PROMPT = (
    "The prompt expander is on and its task prompt is empty. Replace Text (node {replace}) "
    "puts the request into the text on its `string` input, and that text is empty, so "
    "Generate Text (node {generate}) answers an empty prompt, and a render that takes its "
    "prompt from that answer follows whatever it wrote. No picture was rendered. Fill the "
    "text box wired to Replace Text with the task prompt, or send the request to the model "
    "without the expander instead: in the DGX Monarch templates, set the switch titled "
    "If/Else Switch (expander on or off) to false.")
MISSING_MARKER = (
    "The prompt expander is on and the request cannot reach it. Replace Text (node "
    "{replace}) puts the request where the text on its `string` input holds {find!r}, and "
    "that text does not hold it, so Generate Text (node {generate}) rewrites the task "
    "prompt alone, and a render that takes its prompt from that answer ignores the request. "
    "No picture was rendered. Put {find!r} in the task prompt where the request belongs, or "
    "send the request to the model without the expander instead: in the DGX Monarch "
    "templates, set the switch titled If/Else Switch (expander on or off) to false.")


def _node(prompt: Mapping, key: object) -> Mapping | None:
    node = None if key is None else prompt.get(str(key))
    return node if isinstance(node, Mapping) else None


def _inputs(node: Mapping | None) -> Mapping:
    inputs = node.get("inputs") if node is not None else None
    return inputs if isinstance(inputs, Mapping) else {}


def _source(value: object) -> str | None:
    """The id of the node a link comes from, or None for a value that is no link."""
    if isinstance(value, (list, tuple)) and graph_live.is_link(value):
        return str(value[0])
    return None


def _text(prompt: Mapping, value: object) -> str | None:
    """The text an input holds: a literal, or the widget of the text box it is wired to."""
    if isinstance(value, str):
        return value
    source = _node(prompt, _source(value))
    if source is None or source.get("class_type") not in TEXT_CLASSES:
        return None
    held = _inputs(source).get("value")
    return held if isinstance(held, str) else None


def _stated(nodes: Mapping[str, Mapping]) -> dict[str, Mapping]:
    """The graph with each switch a Boolean box drives set to that box's value."""
    stated = dict(nodes)
    for key, node in nodes.items():
        if node.get("class_type") not in graph_live.SWITCH_CLASSES:
            continue
        inputs = _inputs(node)
        box = _node(nodes, _source(inputs.get("switch")))
        if box is None or box.get("class_type") != BOOLEAN_CLASS:
            continue
        value = _inputs(box).get("value")
        if isinstance(value, bool):
            stated[key] = {**node, "inputs": {**inputs, "switch": value}}
    return stated


def _finding(prompt: Any) -> tuple[str, dict[str, str]] | None:
    """The sentence and its fields for a running expander that cannot see the request."""
    if not isinstance(prompt, Mapping):
        return None
    nodes = _stated({str(key): node for key, node in prompt.items()
                     if isinstance(node, Mapping)})
    # What ComfyUI is sure to run, then what the renders among those are sure to read.
    running = graph_live.live_nodes(nodes, sure=True)
    renders = [key for key in running if nodes[key].get("class_type") in ROOT_CLASSES]
    live = graph_live.live_nodes(nodes, roots=renders, sure=True)
    for key in sorted(live):
        node = nodes[key]
        if node.get("class_type") != GENERATE_CLASS:
            continue
        source = _source(_inputs(node).get("prompt"))
        replace = _node(nodes, source)
        if replace is None or replace.get("class_type") != REPLACE_CLASS:
            continue
        inputs = _inputs(replace)
        template = _text(nodes, inputs.get("string"))
        if template is None:
            continue
        fields = {"replace": str(source), "generate": key}
        if not template.strip():
            return EMPTY_TASK_PROMPT, fields
        find = inputs.get("find")
        if (isinstance(find, str) and find and find not in template
                and graph_live.is_link(inputs.get("replace"))):
            return MISSING_MARKER, {**fields, "find": find}
    return None


def dead_expander(prompt: Any) -> str | None:
    """The refusal sentence for a running expander that cannot see the request, or None."""
    found = _finding(prompt)
    return None if found is None else found[0].format(**found[1])


def refuse_dead_expander(prompt: Any) -> None:
    """Raise the class P refusal when the graph's expander cannot see the request.

    Each sentence is named at its own raise, where the refusal ledger reads it.
    """
    try:
        found = _finding(prompt)
    except Exception:  # unreadable evidence answers nothing
        return
    if found is None:
        return
    sentence, fields = found
    if sentence is MISSING_MARKER:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, MISSING_MARKER.format(**fields),
            troubleshooting=TROUBLESHOOTING))
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS, EMPTY_TASK_PROMPT.format(**fields),
        troubleshooting=TROUBLESHOOTING))
