"""Driver-side nodes for per-render ComfyUI model-sampling patches."""
from __future__ import annotations

from ..constants import MODEL_TYPE, NODE_CATEGORY
from .common import ModelSpec
from .model_patch_preflight import refuse_shift_after_skip


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
        refuse_shift_after_skip(model)
        return (model.with_model_sampling({"kind": "sd3", "shift": shift}),)


class DGXMonarchModelSamplingMiniMaxH3:
    """Stock ModelSamplingMiniMaxH3's two widgets, applied on every Worker.

    The names, order, defaults and ranges are stock's (``MiniMaxH3SigmaShift``,
    comfy_extras/nodes_minimax_h3.py), so a graph advisor swap keeps both
    values. Each Worker applies ComfyUI's own node to its render clone
    (actor/model_sampling_render.py). Like stock, a later sampling node
    replaces an earlier one. The schema stays literal for the Studio inventory;
    tests/test_h3_sigma_shift.py holds it to h3_sigma_shift.py's constants, and
    the comfy test to stock's.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = (
        "Sets the MiniMax H3 video and audio flow shifts on every Worker, as ComfyUI's "
        "ModelSamplingMiniMaxH3 node does. p-h3-image needs 1.0 and 1.0. It refuses on a "
        "checkpoint that is not MiniMax H3 (docs/TROUBLESHOOTING.md #144)."
    )

    def patch(self, model: ModelSpec, shift_video: float, shift_audio: float):
        refuse_shift_after_skip(model, "ModelSamplingMiniMaxH3 (DGX Monarch)")
        return (model.with_model_sampling(
            {"kind": "minimax_h3", "shift_video": shift_video, "shift_audio": shift_audio}),)
