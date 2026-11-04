"""Driver-side node that carries ComfyUI's MiniMax H3 context windows to the Workers."""
from __future__ import annotations

from dataclasses import replace

from ..constants import MODEL_TYPE, NODE_CATEGORY
from ..h3_context_windows import freeze_context_windows
from .common import ModelSpec


class DGXMonarchMiniMaxH3ContextWindows:
    """Stock MiniMaxH3ContextWindows's widgets, applied on every Worker.

    The names, order, defaults and ranges are stock's (ComfyUI 0.44.0,
    comfy_extras/nodes_context_windows.py), so a graph advisor swap keeps every
    value. Each Worker applies ComfyUI's own node to its render clone
    (actor/context_windows_render.py). Chained nodes behave as stock's do: the
    stock node replaces the one context handler a model holds, so a later node
    replaces an earlier one. The schema stays literal for the Studio inventory;
    tests/test_h3_context_windows.py holds it to h3_context_windows.py's
    constants, and the comfy test to stock's.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "context_length": ("INT", {"default": 124, "min": 22, "max": 16384, "step": 17}),
                "context_overlap": ("INT", {"default": 39, "min": 5, "step": 17}),
                "fuse_method": (["pyramid", "relative", "flat", "overlap-linear"], {"default": "pyramid"}),
                "freenoise": ("BOOLEAN", {"default": True, "advanced": True}),
                "split_conds_to_windows": ("BOOLEAN", {"default": False, "advanced": True}),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = (
        "Samples a long MiniMax H3 clip as overlapping windows on every Worker, as ComfyUI's "
        "MiniMax H3 Context Windows node does: each window is one model call on a shorter "
        "clip, and sequence parallelism splits each window as it splits a whole clip. Lengths "
        "are frames of the form 17n + 5. It needs ComfyUI 0.44.0 or newer on every host, and "
        "it refuses on a checkpoint that is not MiniMax H3 (docs/TROUBLESHOOTING.md #142)."
    )

    def patch(self, model: ModelSpec, context_length: int, context_overlap: int,
              fuse_method: str, freenoise: bool, split_conds_to_windows: bool = False):
        spec = freeze_context_windows({
            "context_length": context_length,
            "context_overlap": context_overlap,
            "fuse_method": fuse_method,
            "freenoise": freenoise,
            "split_conds_to_windows": split_conds_to_windows,
        })
        return (replace(model, context_windows=spec),)
