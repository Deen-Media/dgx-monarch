"""Per-frame spatial megapixels for automatic topology and capacity estimates.

Megapixels count sampled pixel height times width, excluding batch, channels,
and time. Runtime estimates convert the latent grid using ComfyUI's detected
``latent_format.spacial_downscale_ratio``; the sweep matrix uses graph pixel
dimensions. Both call ``spatial_megapixels``. They agree when the graph uses
the model's ratio and the canvas divides evenly, as shipped finite-bound
templates do.

Undetected headers fall back to ratio 8, or 1 for 4D three-channel latents.
This undercounts 16x families by four. ``rendered_latent_shape`` applies stock
empty-latent size-tag arithmetic through ``sampling_contract`` so estimates
use the grid workers will sample. Auto topology, render memory, H3 rows, and
reference/pose estimates share that shape.
"""
from __future__ import annotations

from typing import Any

from .log import get_logger

log = get_logger(__name__)

# The fallback for a header comfy cannot name.
LEGACY_SPATIAL_DOWNSCALE = 8
_PIXEL_SPACE_CHANNELS = 3


def spatial_megapixels(height_px: int, width_px: int) -> float:
    """Per-frame spatial megapixels: the quantity every auto-table row bounds."""
    return height_px * width_px / 1e6


def latent_megapixels(samples: Any, downscale: int) -> float:
    """Per-frame spatial megapixels of a sampler latent, or its shape, at ``downscale``.

    The last two axes are height and width for every latent a sampler takes:
    ``[B, C, H, W]`` images, ``[B, C, T, H, W]`` video, and a comfy
    NestedTensor, whose ``shape`` is its first tensor's. For the packed
    audio-video latents (LTX AV, MiniMax H3) that first tensor is the video.
    """
    shape = getattr(samples, "shape", samples)
    return spatial_megapixels(int(shape[-2]) * downscale, int(shape[-1]) * downscale)


def legacy_spatial_downscale(samples: Any) -> int:
    """The fallback ratio: 1 for a 4D 3-channel (pixel-space) latent, else 8.

    Only a header comfy's detection cannot name reaches this. It is right for
    every 8x and pixel-space family and reads a 16x or 32x family low.
    """
    pixel = samples.ndim == 4 and samples.shape[1] == _PIXEL_SPACE_CHANNELS
    return 1 if pixel else LEGACY_SPATIAL_DOWNSCALE


def config_spatial_downscale(config: Any) -> int | None:
    """The spatial ratio a comfy model config's latent format declares, or None."""
    ratio = getattr(getattr(config, "latent_format", None), "spacial_downscale_ratio", None)
    if isinstance(ratio, float) and ratio.is_integer():
        ratio = int(ratio)
    if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 1:
        return None
    return ratio


def comfy_spatial_downscale(path: str) -> int | None:
    """The ratio comfy declares for this checkpoint, or None. Fails open.

    Auto resolves several times per render; ``detect_comfy_model_config``
    caches by file identity, so this and the render-memory price share one
    detection per checkpoint.
    """
    try:
        from .render_memory_price import detect_comfy_model_config

        return config_spatial_downscale(detect_comfy_model_config(path))
    except Exception as exc:  # comfy absent, unreadable header: fall back
        log.debug("latent ratio unavailable from comfy for %s; using the fallback ratio: %r", path, exc)
        return None


def spatial_downscale_for(path: str, samples: Any) -> int:
    """Pixels per latent cell for auto topology: comfy's ratio, else the legacy guess."""
    ratio = comfy_spatial_downscale(path)
    return legacy_spatial_downscale(samples) if ratio is None else ratio


def rendered_latent_shape(path: str, samples: Any, spatial: Any = None,
                          temporal: Any = None) -> tuple[int, ...]:
    """The shape the workers will sample for this latent and its size tags.

    Stock's fix applied to the shape, on the latent format of the config
    comfy's detection names for the checkpoint (cached, shared with the ratio
    and the render memory price). Fails open to the shape as built: comfy
    absent, a header comfy cannot name, or a latent that is not a tensor.
    """
    built = tuple(int(size) for size in samples.shape)
    try:
        from .render_memory_price import detect_comfy_model_config
        from .sampling_contract import fixed_latent_shape

        latent_format = getattr(detect_comfy_model_config(path), "latent_format", None)
        if latent_format is None:
            return built
        return fixed_latent_shape(samples, latent_format, spatial, temporal)
    except Exception as exc:  # comfy absent, unreadable header: as built
        log.debug("rendered latent shape unavailable for %s; using the shape as built: %r", path, exc)
        return built
