"""DGX model-handle cache policy for Qwen Image 2.1."""
from __future__ import annotations

from ..constants import MODEL_TYPE, NODE_CATEGORY
from .common import ModelSpec


class DGXMonarchQwenImage21Cache:
    """Configure the remote Qwen 2.1 cache; defaults to the safe off state."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "device": (["off", "auto", "gpu", "cpu"], {"default": "off"}),
                "dtype": (["default", "int8", "int4"], {"default": "default"}),
            },
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = NODE_CATEGORY
    DESCRIPTION = (
        "Sets Qwen Image 2.1's replicated prefix KV cache on every worker; other "
        "families ignore it. The cache is off by default. `default` storage is lossless; "
        "`int8` and `int4` use ComfyUI's native compressed K/V cache and are approximate."
    )

    def patch(self, model: ModelSpec, device: str, dtype: str):
        return (model.with_qwen_image21_cache({"device": device, "dtype": dtype}),)
