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

from ..h3_sigma_shift import (
    FAMILY,
    STOCK_NODE_MISSING,
    TROUBLESHOOTING,
    family_refusal_reason,
    is_image_latent,
    missing_shift_reason,
)
from ..model_sampling import MINIMAX_H3, AnyModelSamplingWire, normalize_model_sampling
from ..refusal import RefusalClass, refusal


def h3_stock_node() -> Any:
    """ComfyUI's MiniMax H3 shift node, or None on a ComfyUI without it."""
    try:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    except ImportError:
        return None
    return MiniMaxH3SigmaShift


def _error(text: str) -> Exception:
    from ..adapters.base import UnsupportedModelError

    return UnsupportedModelError(text)


def _refuse_stock_node_missing() -> None:
    raise _error(refusal(
        RefusalClass.PHYSICS, f"{STOCK_NODE_MISSING} Run every host on a ComfyUI that has the "
        "node instead.", troubleshooting=TROUBLESHOOTING))


def assert_supported(request: dict, family: str | None) -> None:
    """Refuse, class P, an H3 shift this Worker cannot apply as stock does.

    Three arms, each read from facts every rank holds alike except the host's
    ComfyUI: the node on another family, a ComfyUI without the stock node, and
    p-h3-image's one-frame image latent on a MiniMax H3 model that no H3 shift
    reaches. The last repeats the driver's check (nodes/model_patch_preflight.py)
    for every path that misses it, such as a header the driver could not read;
    the family here is the loaded model's.
    """
    wires = [normalize_model_sampling((request.get(slot) or {}).get("model_sampling"))
             for slot in ("model", "uncond_model")]
    if any(wire is not None and wire["kind"] == MINIMAX_H3 for wire in wires):
        reason = family_refusal_reason(family)
        if reason is not None:
            raise _error(refusal(
                RefusalClass.PHYSICS, f"{reason} Remove the node and render with the "
                "checkpoint's own sampling instead, or load a MiniMax H3 or p-h3-image "
                "checkpoint.", troubleshooting=TROUBLESHOOTING))
        if h3_stock_node() is None:
            _refuse_stock_node_missing()
    if family != FAMILY or not is_image_latent(request.get("latent")):
        return
    primary = wires[0]
    reason = missing_shift_reason(None if primary is None else primary["kind"])
    if reason is not None:
        raise _error(refusal(
            RefusalClass.PHYSICS, f"{reason} Use the shifts the model's own workflow names "
            "(docs/MODELS.md).", troubleshooting=TROUBLESHOOTING))


def apply_model_sampling(render_patcher: Any, wire: AnyModelSamplingWire | None) -> Any:
    """Return ``render_patcher`` with the wire's sampling node applied, or unchanged."""
    if wire is None:
        return render_patcher
    if wire["kind"] == MINIMAX_H3:
        node = h3_stock_node()
        if node is None:  # assert_supported refused this render inside the load guard
            _refuse_stock_node_missing()
        return node.execute(render_patcher, wire["shift_video"], wire["shift_audio"])[0]
    from comfy_extras.nodes_model_advanced import ModelSamplingSD3

    shifted, = ModelSamplingSD3().patch(render_patcher, wire["shift"])
    return shifted
