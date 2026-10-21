"""Driver-side node that carries ComfyUI's Skip Layer Guidance to the Workers."""
from __future__ import annotations

from dataclasses import replace

from ..constants import MODEL_TYPE, NODE_CATEGORY
from ..skip_layer_guidance import freeze_skip_layer_guidance
from .common import ModelSpec


class DGXMonarchSkipLayerGuidance:
    """Stock SkipLayerGuidanceDiTSimple's widgets, applied on every Worker.

    The names, order, defaults and ranges are stock's, so a graph advisor swap
    keeps every value. Each Worker applies ComfyUI's own node to its render
    clone (actor/guidance_render.py).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "double_layers": ("STRING", {"default": "7, 8, 9"}),
                "single_layers": ("STRING", {"default": "7, 8, 9"}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = (
        "Skips the listed layers on the unconditional pass inside the step window on every "
        "Worker, as ComfyUI's SkipLayerGuidanceDiTSimple does; empty lists change nothing. "
        "It refuses on a cfg topology, with a two-checkpoint guider, and under sequence "
        "parallelism on any family but MOSS-Image (docs/TROUBLESHOOTING.md #126)."
    )

    def patch(self, model: ModelSpec, double_layers: str, single_layers: str,
              start_percent: float, end_percent: float):
        spec = freeze_skip_layer_guidance({
            "double_layers": double_layers,
            "single_layers": single_layers,
            "start_percent": start_percent,
            "end_percent": end_percent,
        })
        return (replace(model, skip_layer_guidance=spec),)
