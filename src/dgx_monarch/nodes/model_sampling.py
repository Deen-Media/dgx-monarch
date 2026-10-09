"""Driver-side nodes for per-render ComfyUI model-sampling patches."""
from __future__ import annotations

from ..constants import MODEL_TYPE, NODE_CATEGORY
from .common import ModelSpec


class DGXMonarchModelSamplingSD3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = NODE_CATEGORY

    def patch(self, model: ModelSpec, shift: float):
        return (model.with_model_sampling({"kind": "sd3", "shift": shift}),)
