"""What the UNET loader charged for each spend of a prompt, so the driver spend price does not charge it twice.

Where the loader's graph price admits a prompt at the loader site on a graph it
read, ``graph_load_price.note_admitted`` records here, per (prompt id, term),
the figure it charged: the guide encode at the room the loader held for the
encode's window (the larger of the guide and the activations, or both), the
reference encode, and each driver load node's files, by node id, where no
held-model credit took them. ``driver_spend_price`` steps aside for a spend
charged at a figure at least its own, and prices every other spend live: a
spend charged at 0 (a guide on a checkpoint whose family the loader could not
read, another family's loader in a mixed graph, a load the held-model credit
took) is priced at the node.
"""
from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

GUIDE_TERM = "guide"
REFERENCE_TERM = "reference"
_LIMIT = 512
_LOCK = threading.Lock()
# Session lifetime, capped at _LIMIT, oldest first: what the loader charged per
# (prompt id, term). A lost entry prices a spend the loader held; it never
# skips one.
_CHARGED: dict[str, int] = {}


def remember(store: dict, key: str, value: Any) -> None:
    """Insert into a capped, insertion-ordered store, oldest entry evicted first."""
    with _LOCK:
        store[key] = value
        while len(store) > _LIMIT:
            store.pop(next(iter(store)), None)


def node_term(node_id: object) -> str:
    """The term a driver load node is recorded under."""
    return f"node\x00{node_id}"


def load_bytes(inputs: Mapping[str, Any]) -> int:
    """What the loader charges for a driver node's inputs: text encoders at the loader's ratio, models whole."""
    from .nodes import loader_graph as lg

    te = model = 0
    for key, value in inputs.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value:
            continue
        if key.startswith(lg._TE_NAME_KEYS):
            te += lg._artifact_bytes(lg._TE_FOLDERS, value)
        elif key in lg._DRIVER_MODEL_KEYS:
            model += lg._artifact_bytes(lg._DRIVER_MODEL_KEYS[key], value)
    return int(te * lg.TE_RESIDENT_RATIO) + model


def note_charged(prompt_id: str, terms: Any, prompt: object) -> None:
    """Record what an admitting loader charged for each spend of this prompt."""
    if getattr(terms, "guide_bytes", 0):
        record(prompt_id, GUIDE_TERM, int(terms.encode_and_render_bytes))
    if getattr(terms, "ref_bytes", 0):
        record(prompt_id, REFERENCE_TERM, int(terms.ref_bytes))
    if getattr(terms, "credited", True) or not isinstance(prompt, Mapping):
        return  # a held-model credit took the loads, at a figure no file names
    from .nodes.graph_live import live_nodes

    live = live_nodes(prompt)
    for node_id, node in prompt.items():
        if (str(node_id) in live and isinstance(node, Mapping)
                and not str(node.get("class_type", "")).startswith("DGXMonarch")):
            figure = load_bytes(node.get("inputs") or {})
            if figure:
                record(prompt_id, node_term(node_id), figure)


def record(prompt_id: str, term: str, figure: int) -> None:
    """Keep the larger figure the loader charged for this term in this prompt."""
    key = f"{prompt_id}\x00{term}"
    remember(_CHARGED, key, max(figure, _CHARGED.get(key, 0)))


def charged(prompt_id: str, term: str) -> int:
    """What the loader charged for this term in this prompt; 0 where it charged nothing."""
    return _CHARGED.get(f"{prompt_id}\x00{term}", 0)
