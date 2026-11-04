"""Worker application of a model-sampling patch on the per-render clone.

Both kinds run ComfyUI's own node on the clone that
``actor/sampling.model_sampling_render_clone`` builds: ``sd3`` through
ModelSamplingSD3, and ``minimax_h3`` through ``MiniMaxH3SigmaShift``
(ModelSamplingMiniMaxH3), which puts the paired video and audio sampling on the
clone, keeps the stock object's noise scale, and sets the two transformer_options
keys the H3 forward reads, the sharded one included (adapters/minimax_h3.py).
The clone owns its model_options, so no shift outlives its render.

``assert_supported`` is the belt in front of it. It runs inside run_sample's load
guard (actor/sample_protocol.py), as the context-window belt does: a checkpoint
that is not MiniMax H3 reads the same request and header on every rank, while a
Worker whose ComfyUI lacks the stock node is a host fact, so a one-sided refusal
there takes the readiness exchange and the healthy peer refuses too.
"""
from __future__ import annotations

from typing import Any

from ..h3_sigma_shift import STOCK_NODE_MISSING, TROUBLESHOOTING, family_refusal_reason
from ..model_sampling import MINIMAX_H3, AnyModelSamplingWire, normalize_model_sampling
from ..refusal import RefusalClass, refusal


def h3_stock_node() -> Any:
    """ComfyUI's MiniMax H3 shift node, or None on a ComfyUI without MiniMax H3."""
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    except ImportError:
        return None
    return MiniMaxH3SigmaShift


def _refuse(reason: str) -> None:
    from ..adapters.base import UnsupportedModelError

    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS, f"{reason} Use the shifts the model's own workflow names "
        "(docs/MODELS.md).", troubleshooting=TROUBLESHOOTING))


def assert_supported(request: dict, family: str | None) -> None:
    """Refuse, class P, an H3 shift this Worker cannot apply as stock does."""
    wires = [normalize_model_sampling((request.get(slot) or {}).get("model_sampling"))
             for slot in ("model", "uncond_model")]
    if not any(wire is not None and wire["kind"] == MINIMAX_H3 for wire in wires):
        return
    reason = family_refusal_reason(family)
    if reason is None and h3_stock_node() is None:
        reason = STOCK_NODE_MISSING
    if reason is not None:
        _refuse(reason)


def apply_model_sampling(render_patcher: Any, wire: AnyModelSamplingWire | None) -> Any:
    """Return ``render_patcher`` with the wire's sampling node applied, or unchanged."""
    if wire is None:
        return render_patcher
    if wire["kind"] == MINIMAX_H3:
        node = h3_stock_node()
        if node is None:  # assert_supported refused this render inside the load guard
            _refuse(STOCK_NODE_MISSING)
        return node.execute(render_patcher, wire["shift_video"], wire["shift_audio"])[0]
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3

    shifted, = ModelSamplingSD3().patch(render_patcher, wire["shift"])
    return shifted
