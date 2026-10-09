"""Render-time Qwen Image 2.1 cache policy on a loaded Comfy patcher."""
from __future__ import annotations

from collections.abc import Mapping

from ..qwen_image21_cache import normalize_qwen_image21_cache


def apply_loaded_cache_policy(patcher, family: str, node_options: object) -> None:
    """Attach Qwen's cache policy after Comfy has created its ModelPatcher.

    ``comfy.sd.load_diffusion_model`` consumes load options but does not carry
    arbitrary transformer options into the returned patcher, so the cache
    policy is set after the load. Qwen defaults to off; every other family
    keeps its patcher options untouched.
    """
    if family != "qwen_image21":
        return
    options = node_options if isinstance(node_options, Mapping) else None
    configured = normalize_qwen_image21_cache(
        None if options is None else options.get("qwen_image21_cache"))
    policy: dict[str, str]
    if configured is None:
        policy = {"device": "off", "dtype": "default"}
    else:
        policy = {"device": configured["device"], "dtype": configured["dtype"]}
    current = getattr(patcher, "model_options", {})
    if not isinstance(current, Mapping):
        raise TypeError("loaded Qwen patcher model_options must be a mapping")
    updated = dict(current)
    transformer = updated.get("transformer_options", {})
    if not isinstance(transformer, Mapping):
        raise TypeError("loaded Qwen patcher transformer_options must be a mapping")
    transformer = dict(transformer)
    transformer["qwen_image21_cache"] = policy
    updated["transformer_options"] = transformer
    patcher.model_options = updated
