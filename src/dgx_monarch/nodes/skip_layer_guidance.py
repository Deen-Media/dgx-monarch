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

    Chained nodes behave as stock's do. Stock returns its input model when both
    lists hold no digit (comfy_extras/nodes_slg.py SkipLayerGuidanceDiTSimple),
    so an empty node keeps a skip an earlier node set. An active node installs
    the one calc_cond_batch function a model holds
    (ModelPatcher.set_model_sampler_calc_cond_batch_function), so a later
    active node replaces an earlier one.
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
        "Worker, as ComfyUI's SkipLayerGuidanceDiTSimple does. Empty lists change nothing, "
        "so they do not remove a skip an earlier node in the chain set: clear or remove that "
        "node instead. A later node with layers replaces an earlier one. It refuses on a cfg "
        "topology, with a two-checkpoint guider, and under sequence parallelism on any "
        "family but MOSS-Image (docs/TROUBLESHOOTING.md #126)."
    )

    def patch(self, model: ModelSpec, double_layers: str, single_layers: str,
              start_percent: float, end_percent: float):
        spec = freeze_skip_layer_guidance({
            "double_layers": double_layers,
            "single_layers": single_layers,
            "start_percent": start_percent,
            "end_percent": end_percent,
        })
        if spec is None:
            # Stock returns its input unchanged here, so an earlier skip stands.
            return (model,)
        return (replace(model, skip_layer_guidance=spec),)
