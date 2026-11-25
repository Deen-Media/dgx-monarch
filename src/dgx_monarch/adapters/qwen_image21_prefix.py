"""Qwen-Image-2.1 references and prefix cache: when a cached prefix may be reused.

The sharded forward (adapters/qwen_image21.py) keeps prefix K/V replicated on
every rank and reuses them only for the complete, unchanged prefix, keyed on
comfy's native prefix key plus the sequence-parallel cohort. ComfyUI
503d785f0's writable prefix cache, which stock admits by a local memory check,
refuses on every rank alike before any collective (TROUBLESHOOTING #134).
"""
from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError


def validate_reference_count(ref_latents) -> tuple:
    """Normalize the stock optional reference list and enforce its public bound."""
    refs = tuple(ref_latents or ())
    if len(refs) > 10:
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                f"qwen_image21 accepts at most 10 reference images, got {len(refs)}. "
                "Use at most 10 ordered reference images.",
            )
        )
    return refs


def prefix_cache_is_exact(*, enabled: bool, hooks_present: bool,
                          refs, image_slots, cache_owner: str | None) -> bool:
    """Return whether an explicitly scoped prefix cache can be reused.

    The cache entries are replicated on every sequence-parallel rank: each rank
    needs the full prefix K/V while it owns only target-query rows.
    ``cache_owner`` names that cohort, not a process-local cache; the key
    (``distributed_prefix_cache_key``) adds the sequence-parallel degree, world
    and rank to comfy's native prefix key. This guards reuse identity only:
    comfy's optional int8 and int4 cache storage is lossy by design. Raises on
    more than 10 references, or more image slots than references.
    """
    validate_reference_count(refs)
    if image_slots is not None and len(image_slots) > len(tuple(refs or ())):
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                "qwen_image21 image_slots has more entries than reference images. "
                "Use one slot per reference image.",
            )
        )
    return bool(enabled and not hooks_present and cache_owner == "distributed")


def distributed_prefix_cache_key(native_key: torch.Tensor, *, topology_sp: int,
                                 world: int, rank: int) -> torch.Tensor:
    """Bind a native full-prefix key to the exact sequence-parallel cohort.

    Comfy's native key carries every prefix-affecting value (text embeddings,
    reference latent bytes, their slots, and target geometry). It omits target
    *values*, because denoising changes them on every step. A replicated K/V
    entry cannot cross a topology or rank assignment boundary, so append that
    identity without replacing native equality semantics.
    """
    if native_key.ndim != 2:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "qwen_image21 native prefix key must be rank two. "
            "Use the supported native Qwen Image 2.1 conditioning path.",
        ))
    if topology_sp < 1 or world != topology_sp or not 0 <= rank < world:
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                "qwen_image21 prefix cache requires the live sequence-parallel cohort "
                "to match the adapter topology. Use a fresh topology whose sequence-"
                "parallel degree matches the active worker cohort.",
            )
        )
    cohort = native_key.new_tensor((topology_sp, world, rank)).expand(native_key.shape[0], -1)
    return torch.cat((native_key, cohort), dim=1)


# The sharded forward raises this on every rank alike; TROUBLESHOOTING #134.
WRITABLE_PREFIX_CACHE_REFUSAL = refusal(
    RefusalClass.PHYSICS,
    "qwen_image21: the Qwen Image 2.1 cache device is 'auto' on a ComfyUI whose forward takes "
    "a writable prefix K/V cache (prefix_cache, ComfyUI 503d785f0), under dynamic VRAM. There "
    "stock keeps the prefix keys and values in that cache, or recomputes them when it admits "
    "none, where the sequence-parallel forward fills its own cache; no hardware test has "
    "compared the two on a cluster. "
    "Set the Qwen Image 2.1 Cache (DGX Monarch) node's device to 'gpu', 'cpu' or 'off', or run "
    "on topology 'single' (mode=local with gpus_per_host=1), which runs ComfyUI's own forward.",
    troubleshooting=134,
)


def stock_forward_takes_prefix_cache(model_class: Any) -> bool:
    """Whether a model class's stock ``_forward`` takes ``prefix_cache`` (503d785f0 on)."""
    try:
        return "prefix_cache" in inspect.signature(model_class._forward).parameters
    except (AttributeError, TypeError, ValueError):
        return False


def _patcher_is_dynamic(model: Any) -> bool:
    """Whether the attached patcher is dynamic. One that cannot say counts as dynamic,
    where stock's own ``is_dynamic()`` call would raise."""
    is_dynamic = getattr(getattr(model, "current_patcher", None), "is_dynamic", None)
    return True if not callable(is_dynamic) else bool(is_dynamic())


def refuse_writable_prefix_cache(model: Any, prefix_cache: Any, cache_options: Any,
                                 writable_seam: bool) -> None:
    """Refuse stock's writable prefix cache on every rank alike, before any collective (#134).

    The rule reads facts every rank shares: the cache policy the actor sets
    on every load (absent reads ``off``, as in the sharded forward), the
    tree's forward and the patcher class. Stock's admission of the cache is a
    local memory check, so a rank can hold one where its peer holds none;
    ``auto`` refuses whether or not it was admitted, because stock without one
    recomputes the prefix where ``select_prefix_cache`` would fill a cache. A
    ``prefix_cache`` that still arrives refuses too; no ComfyUI path sends one
    for another device. ``off``, ``gpu``, ``cpu`` and a static patcher pass:
    stock takes ``select_prefix_cache`` there as the sharded forward does.
    """
    device = cache_options.get("device", "off") if isinstance(cache_options, Mapping) else None
    if prefix_cache is None and not (
            writable_seam and device == "auto" and _patcher_is_dynamic(model)):
        return
    raise UnsupportedModelError(WRITABLE_PREFIX_CACHE_REFUSAL)
