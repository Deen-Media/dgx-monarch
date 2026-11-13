"""A guide encode against the samplers: what proves its length, and whether it comes first.

``loader_graph`` prices a MiniMax H3 guide encode and the render activations.
They are a sum unless this module proves every guide is a still, from its
file, that encodes, and ComfyUI returns its memory, before any sampler runs;
then they are alternatives and the larger one is charged (issue #187). A clip
keeps the sum: no reading backs its charge alone. ComfyUI with DynamicVRAM empties its
device pool after every node: the executor's ``finally`` calls
``reset_cast_buffers``, which ends in ``soft_empty_cache``
(tests/canary/comfy_seam_contracts.py holds both). An order that cannot be
read keeps the sum, and the words say why either way.

No ComfyUI import at module scope; every read here fails toward the sum.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from ..constants import GUIDER_TYPE, MODEL_TYPE
from .graph_live import is_link, live_nodes

_GIB = 2 ** 30

# The widget a loader carries its filename on, so a guide's length can be
# proved from the file rather than trusted from a node name or an extension.
_GUIDE_FILE_KEYS = ("image",)

# A node with one of these inputs linked uses a model, and it counts as one
# that samples unless ComfyUI's registry shows that it returns only a patched
# model, a guider or a sigma schedule (stock's types and the pack's).
# Read-only after import.
_MODEL_INPUT_KEYS = ("model", "guider")
_NO_RENDER_OUTPUTS = frozenset({"MODEL", MODEL_TYPE, "GUIDER", GUIDER_TYPE, "SIGMAS"})
# Samplers whose renders can overlap other driver work, or whose overlap on one
# box this price does not read, each with the words a budget clause gives.
# Read-only after import.
_OVERLAPPING_SAMPLERS = {
    "DGXMonarchKSamplerPipeline": ("the pipeline sampler can sample one render while "
                                   "the next one's guide encodes"),
    "DGXMonarchFleetKSampler": ("the fleet sampler's renders can overlap, which this "
                                "price does not read"),
}
GUIDE_FIRST = ("it is a still its file proves, and it encodes and ComfyUI returns its "
               "memory before any sampler runs")
NO_POOL_RESET = ("ComfyUI here runs without DynamicVRAM, so it keeps the encode's "
                 "memory until the decode")
MODEL_BEFORE_GUIDE = "a node that uses a model runs before a guide encodes"
SAMPLER_BEFORE_GUIDE = "a sampler in this graph can run before a guide encodes"
ORDER_UNREAD = "the graph does not show where the guide encodes"
# Only a still has a reading beside the render: no hardware cell reads a clip
# encode's peak against its charge with the DiT resident, where the clip charge
# alone would hold the floor (issue #187, filed with the cell that would read it).
NOT_A_PROVEN_STILL = ("a guide here is a clip, or its file does not prove it is a still, and only "
                      "a still is charged as the alternative to the render")


def _gib(value: float) -> str:
    return f"{value / _GIB:.1f}"


def named_file_frames(prompt: object, node_id: object) -> int:
    """Frames in the file this node names, or 0 when nothing proves a count.

    The file is opened and its frames counted, so an animated PNG or WebP
    cannot pass as a still behind an image extension, and a video that no
    image decoder opens reads unreadable rather than cheap. Missing files,
    unreadable files and sources that name no file all return 0, which the
    guide price reads as the declared length or else the worst case.

    A file replaced between this preflight read and execution is charged on
    the proof taken, the same window every other pre-execution read here has.
    """
    if not isinstance(prompt, Mapping) or node_id is None:
        return 0
    node = prompt.get(node_id)
    if node is None:
        node = prompt.get(str(node_id))
    inputs = node.get("inputs") if isinstance(node, Mapping) else None
    if not isinstance(inputs, Mapping):
        return 0
    for key in _GUIDE_FILE_KEYS:
        name = inputs.get(key)
        if not isinstance(name, str) or not name:
            continue
        try:
            import folder_paths
            from PIL import Image

            path = folder_paths.get_annotated_filepath(name)
            if not path or not os.path.isfile(path):
                return 0
            with Image.open(path) as image:
                frames = getattr(image, "n_frames", 1)
            return int(frames) if isinstance(frames, int) and frames > 0 else 0
        except Exception:  # unreadable evidence charges the worst case
            return 0
    return 0


def comfy_returns_pool_after_each_node() -> bool:
    """Whether ComfyUI in this process empties its device pool after every node.

    It does with DynamicVRAM on. Off, or with no ComfyUI in this process, an
    encode's freed blocks stay reserved until the decode.
    """
    try:
        import comfy.memory_management as memory_management
    except Exception:  # no ComfyUI here: nothing returns the pool
        return False
    return getattr(memory_management, "aimdo_enabled", False) is True


def _class_outputs(class_type: object) -> tuple[str, ...] | None:
    """The output types ComfyUI's registry gives a node class, or None when it cannot say."""
    try:
        import nodes as comfy_nodes

        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_type)
        outputs = None if cls is None else cls.RETURN_TYPES
    except Exception:  # no registry, or a class it cannot read
        return None
    if not isinstance(outputs, (list, tuple)):
        return None
    return tuple(str(kind) for kind in outputs)


def _uses_model(node: Mapping) -> bool:
    inputs = node.get("inputs")
    return isinstance(inputs, Mapping) and any(
        is_link(inputs.get(key)) for key in _MODEL_INPUT_KEYS)


def _may_sample(node: Mapping) -> bool:
    """A node that uses a model, unless the registry shows every output is one
    that carries no render: a model patch, a guider and a scheduler use a
    model and sample nothing. Identity Gate (DGX Monarch) returns a latent, and
    a class the registry cannot place counts as sampling."""
    if not _uses_model(node):
        return False
    outputs = _class_outputs(node.get("class_type"))
    return not outputs or not set(outputs) <= _NO_RENDER_OUTPUTS


def proved(prompt: Mapping, live: frozenset[str], guides: tuple[str, ...],
           stills: bool = False) -> tuple[bool, str]:
    """Whether every guide is a still that encodes, and returns its memory,
    before any sampler runs; the words for either answer.

    Five things must hold. ComfyUI here returns its pool after every node. No
    sampler that overlaps other driver work is in the graph. No guide has a
    node that uses a model among its ancestors (a second pass, a guide image
    decoded from a render), read with both branches of an undecided switch.
    Every node that may sample has every guide among the nodes ComfyUI is
    certain to run before it, so no sampler can run first. And ``stills``:
    every guide's file proves it a still, as the price proves it for the still
    charge (``named_file_frames``); a clip, a declared length or an unread one
    keeps the sum. Any doubt, and any fault, keeps the sum.
    """
    try:
        if not comfy_returns_pool_after_each_node():
            return False, NO_POOL_RESET
        nodes = {str(key): node for key, node in prompt.items() if isinstance(node, Mapping)}
        for key in sorted(live):
            words = _OVERLAPPING_SAMPLERS.get(str(nodes.get(key, {}).get("class_type", "")))
            if words:
                return False, words
        for guide in guides:
            if any(_uses_model(nodes[key]) for key in live_nodes(nodes, roots=[guide])
                   if key != guide and key in nodes):
                return False, MODEL_BEFORE_GUIDE
        for key in sorted(live):
            node = nodes.get(key)
            if node is None or not _may_sample(node):
                continue
            ran_first = live_nodes(nodes, roots=[key], sure=True)
            if not all(guide in ran_first for guide in guides):
                return False, SAMPLER_BEFORE_GUIDE
        if not stills:
            return False, NOT_A_PROVEN_STILL
        return True, GUIDE_FIRST
    except Exception:  # an unread order keeps the sum
        return False, ORDER_UNREAD


def charged_note(note: str, guide_bytes: int, activation_bytes: int, first: bool, order_note: str) -> str:
    """The guide's line under "Charged:", ending with which term was charged and why."""
    if first:
        charged = ("this guide encode" if guide_bytes > activation_bytes else
                   f"the {_gib(activation_bytes)} GiB of render activations")
        return (f"{note}; only the larger of this and the render activations is charged, here "
                f"{charged}, since {order_note}")
    return f"{note}; charged beside the render activations because {order_note}"


def budget_words(guide_bytes: int, activation_bytes: int, first: bool,
                 order_note: str) -> tuple[int, int, str]:
    """(guide, activations, words) as a budget clause names them: the term left out reads 0, and the
    words after the clause say which was charged and why, or why both are."""
    if not (guide_bytes and activation_bytes):
        return guide_bytes, activation_bytes, ""
    if first and guide_bytes > activation_bytes:
        return guide_bytes, 0, (f"; its {_gib(activation_bytes)} GiB of render activations come after "
                                "ComfyUI returns the encode's memory, so only the larger of the two is charged")
    if first:
        return 0, activation_bytes, (f"; its {_gib(guide_bytes)} GiB guide encode runs first and ComfyUI "
                                     "returns that memory before sampling, so only the larger of the two is "
                                     "charged")
    return guide_bytes, activation_bytes, (f"; both are charged because {order_note}" if order_note else "")


__all__ = ["GUIDE_FIRST", "MODEL_BEFORE_GUIDE", "NOT_A_PROVEN_STILL", "NO_POOL_RESET", "ORDER_UNREAD",
           "SAMPLER_BEFORE_GUIDE", "budget_words", "charged_note", "comfy_returns_pool_after_each_node",
           "named_file_frames", "proved"]
