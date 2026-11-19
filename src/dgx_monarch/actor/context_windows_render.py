"""Worker application of MiniMax H3 context windows on the per-render clone.

The windows are ComfyUI's own: ``MiniMaxH3ContextWindowsNode.execute`` puts its
context handler and its two sampling wrappers on the clone, stock
``calc_cond_batch`` runs each window as one ordinary model call on a shorter
clip, and the sharded H3 forward splits that call's packed sequence as it
splits a whole clip, rebuilding the layout from the windowed payload
(adapters/minimax_h3.py). Windows are cut above the forward and sequence
parallelism splits inside it, so the two never cut the same axis twice.

This module adds only the belt in front of it. ``assert_supported`` runs inside
run_sample's load guard (actor/sample_protocol.py): a checkpoint that is not
MiniMax H3 and a two-checkpoint guider read the same request and header on
every rank, while a Worker whose ComfyUI lacks the stock node is a host fact,
so a one-sided refusal there takes the readiness exchange and the healthy peer
refuses too rather than waiting at the first collective.
"""
from __future__ import annotations

from typing import Any

from ..h3_context_windows import (
    DUAL_MODEL_REASON,
    FIELD,
    STOCK_NODE_MISSING,
    ContextWindowsWire,
    family_refusal_reason,
    normalize_context_windows,
)
from ..refusal import RefusalClass, refusal

TROUBLESHOOTING = 142


def spec_of(request: dict) -> ContextWindowsWire | None:
    """The request's context-window wire value, or None without the node."""
    return normalize_context_windows((request.get("model") or {}).get(FIELD))


def stock_node() -> Any:
    """ComfyUI's MiniMax H3 context-window node, or None on ComfyUI 0.43 and older."""
    try:
        from comfy_extras.nodes_context_windows import MiniMaxH3ContextWindowsNode
    except ImportError:
        return None
    return MiniMaxH3ContextWindowsNode


def assert_supported(request: dict, family: str | None) -> None:
    """Refuse, class P, a windowed render this Worker cannot run as stock does."""
    uncond = request.get("uncond_model") or {}
    reason = DUAL_MODEL_REASON if uncond.get(FIELD) is not None else None
    if reason is None and spec_of(request) is not None:
        reason = family_refusal_reason(family)
        if reason is None and stock_node() is None:
            reason = STOCK_NODE_MISSING
    if reason is not None:
        from ..adapters.base import UnsupportedModelError

        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, f"{reason} Without the node, the graph samples the clip whole "
            "instead.", troubleshooting=TROUBLESHOOTING))


def apply(render_patcher: Any, request: dict) -> Any:
    """Return ``render_patcher`` with the request's windows on a clone, or unchanged."""
    wire = spec_of(request)
    if wire is None:
        return render_patcher
    node = stock_node()
    if node is None:  # assert_supported refused this render inside the load guard
        from ..adapters.base import UnsupportedModelError

        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS, f"{STOCK_NODE_MISSING} Without the node, the graph samples the "
            "clip whole instead.", troubleshooting=TROUBLESHOOTING))
    return node.execute(
        render_patcher, wire["context_length"], wire["context_overlap"], wire["fuse_method"],
        wire["freenoise"], split_conds_to_windows=wire["split_conds_to_windows"])[0]
