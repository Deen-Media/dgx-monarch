"""Worker application of Skip Layer Guidance on the per-render clone.

The skip itself is ComfyUI's own ``SkipLayerGuidanceDiTSimple.execute``: its
window test, its options copy and its two model calls run unchanged. This module
adds only the belt in front of it, the same refusal the driver preflight raises
(nodes/model_patch_preflight.py), read from facts every rank holds alike, so
every rank refuses in the same call and the lease retires CONSUMED.
"""
from __future__ import annotations

from typing import Any

from ..refusal import RefusalClass, refusal
from ..skip_layer_guidance import (
    normalize_skip_layer_guidance,
    skip_layer_guidance_active,
    skip_layer_guidance_refusal_reason,
)


def _specs(request: dict) -> tuple[Any, Any]:
    uncond = request.get("uncond_model") or {}
    return (normalize_skip_layer_guidance((request.get("model") or {}).get("skip_layer_guidance")),
            normalize_skip_layer_guidance(uncond.get("skip_layer_guidance")))


def apply_skip_layer_guidance(render_patcher: Any, request: dict, worker: Any,
                              adapter: Any = None) -> Any:
    """Return ``render_patcher`` with the request's skip installed, or unchanged.

    No active skip returns the clone untouched, with no import and no adapter
    lookup. ``adapter`` defaults to the one ``adapter_for`` binds for this
    Worker's family override: run_sample's own local adapter is None on every
    sequence-parallel render, so the family gate never reads it.
    """
    spec, uncond_spec = _specs(request)
    if not (skip_layer_guidance_active(spec) or skip_layer_guidance_active(uncond_spec)):
        return render_patcher
    if adapter is None:
        from ..adapters import adapter_for
        from ..family_select import override_for_worker

        adapter = adapter_for(render_patcher.model, override_for_worker(worker))
    topology = getattr(worker, "topology", {}) or {}
    reason = skip_layer_guidance_refusal_reason(
        cfg=int(topology.get("cfg", 1)),
        sp=int(topology.get("ulysses", 1)) * int(topology.get("ring", 1)),
        family=str(getattr(adapter, "family", "model")),
        family_honours=bool(getattr(adapter, "sp_honours_block_replace", False)),
        dual_model=request.get("uncond_model") is not None,
    )
    if reason is not None:
        from ..adapters.base import UnsupportedModelError

        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"{reason} To keep this topology and guider instead, clear both layer lists on "
            "the node; the render then runs without the skip.",
            troubleshooting=126))
    if spec is None:
        return render_patcher  # unreachable: an active uncond spec alone is dual_model, refused above
    from comfy_extras.nodes_slg import SkipLayerGuidanceDiTSimple

    return SkipLayerGuidanceDiTSimple.execute(
        render_patcher, spec["start_percent"], spec["end_percent"],
        double_layers=spec["double_layers"], single_layers=spec["single_layers"])[0]
