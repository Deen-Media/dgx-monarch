"""Shared fail-closed contracts for driver and worker sampling requests."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def custom_schedule_steps(sigmas: Any) -> int:
    """Return the denoise-step count for one valid custom sigma schedule.

    A finite one-element vector is the only zero-step schedule. Empty, scalar,
    multidimensional, integral or non-finite values raise ValueError; they must
    never pass as a zero-step render, which skips sampling.
    """
    if (
        not isinstance(sigmas, torch.Tensor)
        or sigmas.ndim != 1
        or sigmas.numel() == 0
        or not torch.is_floating_point(sigmas)
        or not bool(torch.isfinite(sigmas).all().item())
    ):
        raise ValueError(
            "custom sampler sigmas must be a finite, non-empty, 1-D floating Tensor"
        )
    return int(sigmas.numel()) - 1


# The empty latent's size tags, spelled as stock spells them. Empty-latent
# nodes record the downscale ratio they divided by; every stock sampler hands
# both to comfy.sample.fix_empty_latent_channels, which rescales an all-zero
# latent built for another ratio onto the model's grid; the sampler then drops
# both from the LATENT it returns (nodes.py common_ksampler; docs/VALIDATION.md
# "Worker empty-latent size tags", 2026-10-05).
LATENT_SIZE_TAG_KEYS = ("downscale_ratio_spacial", "downscale_ratio_temporal")


def latent_size_tags(latent: Mapping) -> tuple[Any, Any]:
    """The (spatial, temporal) tags exactly as stock reads them; absent is None."""
    spatial, temporal = (latent.get(key, None) for key in LATENT_SIZE_TAG_KEYS)
    return spatial, temporal


def latent_without_size_tags(latent: Mapping) -> dict:
    """A sampler's output LATENT: stock drops the tags its fix consumed.

    Never apply this to a request latent: the worker must still read them.
    """
    return {key: value for key, value in latent.items() if key not in LATENT_SIZE_TAG_KEYS}


def fixed_latent_shape(samples: Any, latent_format: Any,
                       spatial: Any = None, temporal: Any = None) -> tuple[int, ...]:
    """The shape ``comfy.sample.fix_empty_latent_channels`` returns, without building it.

    Shape arithmetic only, in stock's order: a nested latent comes back as it
    is; an all-zero latent takes the format's channel count and, when its
    spatial tag differs from the format's ratio, a grid of round(side * tag /
    ratio); a 3D format lifts a 4D latent to one frame; an all-zero latent
    whose temporal tag differs takes max(1, round(t * tag / ratio)) on axis 2.
    A latent with values is never rescaled. The format's own
    ``fix_empty_latent`` is not modelled: only MiniMax H3 AV overrides it, to
    split its audio stream off the video, and it keeps the video grid. The
    driver's size estimates (``latent_scale.rendered_latent_shape``) and the
    worker's cfg pad equalizer read this, so neither builds a second latent.
    """
    shape = [int(size) for size in samples.shape]
    if getattr(samples, "is_nested", False):
        return tuple(shape)
    empty = bool(torch.count_nonzero(samples) == 0)
    if empty:
        shape[1] = int(latent_format.latent_channels)
        if spatial is not None and spatial != latent_format.spacial_downscale_ratio:
            scale = spatial / latent_format.spacial_downscale_ratio
            shape[-2:] = [round(shape[-2] * scale), round(shape[-1] * scale)]
    if latent_format.latent_dimensions == 3 and len(shape) == 4:
        shape.insert(2, 1)
    if empty and temporal is not None and temporal != latent_format.temporal_downscale_ratio:
        scale = temporal / latent_format.temporal_downscale_ratio
        shape[2] = max(1, round(shape[2] * scale))
    return tuple(shape)
