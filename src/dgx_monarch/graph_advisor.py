"""Advise queued workflows about missing DGX Monarch nodes.

The browser notice identifies stock graphs, missing Init nodes, and stock
loaders or samplers left beside Init. An allowlist limits advice to known
replacements; VAE, CLIP, encode, latent, image, and save nodes remain on the
driver and need no replacement.

This is advisory: every path, including failures, returns the queue payload
unchanged. See docs/DESIGN.md section 5.9 and docs/TROUBLESHOOTING.md #63.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from . import first_render
from .log import get_logger

log = get_logger(__name__)

TROUBLESHOOTING = 63
PHASE = "graph_advisor"
# Titles the toast. The notice spine's default says "first render", which this
# advisory is not: it fires at queue time on a graph that may never render.
SUMMARY = "DGX Monarch: workflow advice"
KILL_SWITCH = "DGXM_GRAPH_ADVISOR"
INIT_CLASS = "DGXMonarchInit"
_PREFIX = "DGXMonarch"
_OFF = frozenset({"0", "false", "off", "no"})
# A long swap list is too much for a toast, so past this many the message
# counts the rest instead of naming them. The driver log carries the same text.
_MAX_LISTED = 8

# Stock class_type -> the DGX Monarch class that replaces it one for one.
# Values are NODE_CLASS_MAPPINGS keys, and DISPLAY_NAMES below copies rows of
# NODE_DISPLAY_NAME_MAPPINGS. Both are literals so this module stays a leaf
# with no node-package import; tests/test_graph_advisor.py pins every row
# against the real mappings in nodes/__init__.py.
SWAPS: dict[str, str] = {
    "UNETLoader": "DGXMonarchUNETLoader",
    # The Monarch LoRA node patches the model only (nodes/loaders.py), which is
    # stock LoraLoaderModelOnly's shape, not model+CLIP LoraLoader's.
    "LoraLoaderModelOnly": "DGXMonarchLoraLoader",
    "KSampler": "DGXMonarchKSampler",
    "KSamplerAdvanced": "DGXMonarchKSamplerAdvanced",
    # Stock SamplerCustom takes a model and a cfg, so it has no one-for-one
    # node and sits in REBUILDS. The DGXMonarchSamplerCustom docstring
    # (nodes/samplers.py) explains its name and the interface it mirrors.
    "SamplerCustomAdvanced": "DGXMonarchSamplerCustom",
    "BasicScheduler": "DGXMonarchBasicScheduler",
    "BasicGuider": "DGXMonarchBasicGuider",
    "CFGGuider": "DGXMonarchCFGGuider",
    "ModelSamplingSD3": "DGXMonarchModelSamplingSD3",
}

DISPLAY_NAMES: dict[str, str] = {
    "DGXMonarchInit": "DGX Monarch Init",
    "DGXMonarchUNETLoader": "Load Diffusion Model (DGX Monarch)",
    "DGXMonarchLoraLoader": "Load LoRA (DGX Monarch)",
    "DGXMonarchModelSamplingSD3": "Model Sampling SD3 (DGX Monarch)",
    "DGXMonarchKSampler": "KSampler (DGX Monarch)",
    "DGXMonarchKSamplerAdvanced": "KSampler Advanced (DGX Monarch)",
    "DGXMonarchSamplerCustom": "Sampler Custom (DGX Monarch)",
    "DGXMonarchBasicScheduler": "Basic Scheduler (DGX Monarch)",
    "DGXMonarchBasicGuider": "Basic Guider (DGX Monarch)",
    "DGXMonarchCFGGuider": "CFG Guider (DGX Monarch)",
}

# Stock classes with no one-for-one Monarch node. The value completes the
# sentence "#<id> <StockClass> -> ...".
REBUILDS: dict[str, str] = {
    "CheckpointLoaderSimple": (
        "Load Diffusion Model (DGX Monarch) for the diffusion model, plus the "
        "stock CLIP and VAE loaders for the other two parts"
    ),
    "LoraLoader": (
        "Load LoRA (DGX Monarch), which patches the model only; there is no "
        "cluster path for the CLIP half"
    ),
    "SamplerCustom": (
        "CFG Guider (DGX Monarch) into Sampler Custom (DGX Monarch), which "
        "takes a guider where the stock node takes a model and a cfg"
    ),
}

# Conditions required by suggested Monarch nodes. Each value completes
# the sentence "<Display> ...".
CAVEATS: dict[str, str] = {
    # Raise site: nodes/samplers.py DGXMonarchBasicScheduler.get_sigmas.
    "DGXMonarchBasicScheduler": (
        "needs an explicit topology on the Init node (pick ring2, uly2 or "
        "cfg2: auto resolves at the first render, after the sigmas are "
        "already needed, and the node refuses)"
    ),
}

_POINTER = f"docs/TROUBLESHOOTING.md #{TROUBLESHOOTING}"


@dataclass(frozen=True, slots=True)
class Advisory:
    """One thing to say about one graph, already worded for the operator."""

    kind: str
    severity: str
    message: str
    note: str
    fingerprint: str


def enabled() -> bool:
    """False when DGXM_GRAPH_ADVISOR is off. Read per call, so it flips live."""
    return os.environ.get(KILL_SWITCH, "1").strip().lower() not in _OFF


def _order(row: tuple[str, str]) -> tuple[int, int, str]:
    """Decimal node ids sort as numbers; every other Unicode id sorts as text."""
    node_id = row[0]
    # `str.isdigit()` accepts superscripts and circled digits that `int()`
    # cannot parse. Queue advice must never disappear because a custom node
    # used one of those valid JSON object keys as its id.
    if node_id.isdecimal():
        try:
            return 0, int(node_id), ""
        except ValueError:
            pass
    return 1, 0, node_id


def _rows(prompt: object) -> list[tuple[str, str]]:
    """(node id, class_type) for every readable node, in display order."""
    if not isinstance(prompt, Mapping):
        return []
    rows: list[tuple[str, str]] = []
    for node_id, node in prompt.items():
        if not isinstance(node, Mapping):
            continue
        class_type = node.get("class_type")
        if isinstance(class_type, str) and class_type:
            rows.append((str(node_id), class_type))
    return sorted(rows, key=_order)


def _fingerprint(kind: str, rows: Iterable[tuple[str, str]]) -> str:
    """Identify this advice about this graph, stably across requeues."""
    payload = "|".join(f"{node_id}:{class_type}" for node_id, class_type in rows)
    return hashlib.sha256(f"{kind}\n{payload}".encode()).hexdigest()[:16]


def _listed(entries: list[str]) -> str:
    if len(entries) <= _MAX_LISTED:
        return "; ".join(entries)
    head = "; ".join(entries[:_MAX_LISTED])
    return f"{head}; and {len(entries) - _MAX_LISTED} more"


def _swap_line(node_id: str, class_type: str) -> str:
    target = SWAPS.get(class_type)
    if target is None:
        return f"#{node_id} {class_type} -> {REBUILDS[class_type]}"
    line = f"#{node_id} {class_type} -> {DISPLAY_NAMES[target]}"
    caveat = CAVEATS.get(target)
    return f"{line}, which {caveat}" if caveat else line


def _conditions(classes: Iterable[str]) -> str:
    """The standing conditions of the Monarch nodes named, once each."""
    said = [f"{DISPLAY_NAMES[name]} {CAVEATS[name]}"
            for name in dict.fromkeys(classes) if name in CAVEATS]
    return f" {'; '.join(said)}." if said else ""


def _stock_only(swappable: list[tuple[str, str]]) -> Advisory:
    lines = [_swap_line(node_id, class_type) for node_id, class_type in swappable]
    return Advisory(
        kind="stock_only",
        severity="info",
        message=(
            "DGX Monarch: this workflow has no DGX Monarch nodes, so it renders on "
            "the driver box alone and the rest of the cluster stays idle. To run it "
            f"on the cluster, add the {DISPLAY_NAMES[INIT_CLASS]} node and swap: "
            f"{_listed(lines)}. Leave everything else (CLIP, VAE, latents and save) as "
            f"it is ({_POINTER})."
        ),
        note=f"stock workflow: {len(lines)} nodes to swap for the cluster",
        fingerprint=_fingerprint("stock_only", swappable),
    )


def _missing_init(monarch: list[tuple[str, str]]) -> Advisory:
    # Every node in this pack reaches the cluster through a mesh handle or
    # through a model handle a loader made from one, and Init is the only node
    # that returns a mesh, so this advisory has no exceptions.
    named = _listed([f"#{node_id} {class_type}" for node_id, class_type in monarch])
    # The Init node these graphs are told to add carries the widget the caveats
    # are about, so its conditions belong in this advisory too.
    conditions = _conditions(class_type for _node_id, class_type in monarch)
    return Advisory(
        kind="missing_init",
        severity="warn",
        message=(
            f"DGX Monarch: this workflow uses DGX Monarch nodes ({named}) but has no "
            f"{DISPLAY_NAMES[INIT_CLASS]} node. Init attaches the cluster and produces "
            "the mesh every other node in the pack needs, so add it and wire its mesh "
            f"output into the loader.{conditions} ({_POINTER})."
        ),
        note=f"no {DISPLAY_NAMES[INIT_CLASS]} node in the graph",
        fingerprint=_fingerprint("missing_init", monarch),
    )


def _leftover_stock(swappable: list[tuple[str, str]]) -> Advisory:
    lines = [_swap_line(node_id, class_type) for node_id, class_type in swappable]
    return Advisory(
        kind="leftover_stock",
        severity="warn",
        message=(
            "DGX Monarch: these nodes still run on the driver box while the rest of "
            f"the workflow runs on the cluster: {_listed(lines)} ({_POINTER})."
        ),
        note=f"{len(lines)} stock nodes left on the driver",
        fingerprint=_fingerprint("leftover_stock", swappable),
    )


def analyze(prompt: object) -> list[Advisory]:
    """Read one queue payload's prompt graph and say what is missing.

    Pure: no ComfyUI import, no driver state, no publication. Anything that is
    not a readable node is skipped rather than guessed at.
    """
    if not enabled():
        return []
    rows = _rows(prompt)
    monarch = [row for row in rows if row[1].startswith(_PREFIX)]
    swappable = [row for row in rows if row[1] in SWAPS or row[1] in REBUILDS]
    if not monarch:
        # A graph with nothing to swap gets no advice: an upscale or a
        # file-shuffling graph is not a failed cluster render.
        return [_stock_only(swappable)] if swappable else []
    if not any(class_type == INIT_CLASS for _, class_type in monarch):
        # Init first: a graph missing it has no cluster to move leftovers onto.
        return [_missing_init(monarch)]
    return [_leftover_stock(swappable)] if swappable else []


def publish(advisories: Iterable[Advisory]) -> int:
    """Announce each advisory once per driver process. Returns how many spoke."""
    spoken = 0
    for advisory in advisories:
        if first_render.notice(
            PHASE,
            advisory.message,
            advisory.note,
            once_key=f"{PHASE}:{advisory.kind}:{advisory.fingerprint}",
            severity=advisory.severity,
            summary=SUMMARY,
        ):
            spoken += 1
    return spoken


def advise(prompt: object) -> int:
    """Analyze and publish. Never raises: advice is not worth a failed queue."""
    try:
        return publish(analyze(prompt))
    except Exception as exc:
        log.debug("graph advisor skipped this queue: %r", exc)
        return 0


def on_prompt(json_data: Any) -> Any:
    """ComfyUI on_prompt handler: advise, then return the payload untouched.

    The outer guard covers the payload itself: `json_data` comes off the wire
    and a mapping whose own lookup raises must still queue.
    """
    try:
        if isinstance(json_data, Mapping):
            advise(json_data.get("prompt"))
    except Exception as exc:
        log.debug("graph advisor could not read the queue payload: %r", exc)
    return json_data


def register_on_prompt(prompt_server: Any) -> bool:
    """Attach the advisor to a PromptServer once. True when this call attached."""
    if getattr(prompt_server, "_dgxm_graph_advisor", False):
        return False
    prompt_server.add_on_prompt_handler(on_prompt)
    prompt_server._dgxm_graph_advisor = True
    log.info("queue-time graph advisor registered")
    return True
