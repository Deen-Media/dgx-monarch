"""The upstream gate's two family tests, anchored on the keys ComfyUI's detection reads.

The header detector (``detect._SIGNATURES``) matches needles at any dotted
boundary, strips a prefix per key, and takes the first row that matches. That
suits choosing a topology. For a refusal it was a hole: one extra tensor
(``aux.dec_net.x.weight``, ``aux.audio_patch_proj.weight``, or a needle of any
earlier row) moved the detector's answer, while ComfyUI, which tests exact keys
after one prefix strip for the whole file, still built the model the gate
must refuse.

So each test here takes ComfyUI's own reading first. Where ComfyUI's keys name
a model of the family, its exact test decides. Where they do not, ComfyUI
builds no such model and the detector's reading stands, which can only add a
refusal. Every file the detector's reading refused is still refused (the
module's tests hold the cases), and an extra tensor cannot remove one.

Not replayed: ComfyUI's detection is a chain of branches, and these are the
two branches the families own (``comfy/model_detection.py``), not the chain.
"""
from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Final

from . import minimax_h3_image_header as h3_image
from .detect import (
    _SIGNATURES,
    _ZIMAGE_L2P_PRESENT,
    _has_segment,
    _strip_prefix,
    detect_family_from_keys,
    is_zimage_l2p_from_keys,
)

# ComfyUI builds a MiniMax H3 model from these two keys and names it the image
# model when the audio patch projection is absent.
_H3_KEYS: Final = ("video_patch_proj.weight", "condition_proj.weight")
_H3_AUDIO_KEY: Final = "audio_patch_proj.weight"
# ComfyUI's Lumina 2 branch, which Z-Image takes; the pixel-space model only
# with the decoder's conditioning embed. It has no L2P contract, so an L2P file
# takes this branch and builds latent Z-Image (docs/TROUBLESHOOTING.md #91).
_LUMINA2_KEYS: Final = ("cap_embedder.1.weight", "noise_refiner.0.attention.k_norm.weight")
_ZIMAGE_PIXEL_KEY: Final = "dec_net.cond_embed.weight"
_ZIMAGE_NEEDLES: Final = next(needles for family, needles in _SIGNATURES if family == "zimage")


def comfy_detection_keys(dtypes: dict[str, str], metadata: dict[str, str]) -> frozenset[str]:
    """The key names ComfyUI's detection reads for this header. It never raises.

    ``upstream_gate.comfy_load_keys`` replays ComfyUI's diffusion-model load on
    names, and a canary test holds it key for key against that load. When the
    quantization table makes the replay raise, ComfyUI's load raises before
    detection too; the names after the one prefix strip then stand in, because
    the family tests answer before any quantization metadata counts (#91 does).
    """
    from .. import upstream_gate

    try:
        return frozenset(upstream_gate.comfy_load_keys(dtypes, metadata))
    except Exception:
        prefix = upstream_gate._unet_prefix(dtypes)
        stripped = frozenset(key[len(prefix):] for key in dtypes if key.startswith(prefix))
        return stripped or frozenset(dtypes)


def reads_minimax_h3_image(names: Sequence[str], comfy_keys: Collection[str]) -> bool:
    """Whether the gate's MiniMax-H3-Image refusals (#149, #150) apply to this file."""
    if all(key in comfy_keys for key in _H3_KEYS):
        return _H3_AUDIO_KEY not in comfy_keys  # ComfyUI's own test for the image model
    return detect_family_from_keys(list(names)) == h3_image.FAMILY


def reads_zimage_l2p(names: Sequence[str], comfy_keys: Collection[str]) -> bool:
    """Whether the file is the Z-Image L2P surface (#91).

    The Z-Image needles and the L2P decoder must be there by the detector's
    boundary match, which can only widen the refusal. Then ComfyUI's keys
    decide whether it would build the pixel-space model, the one thing that
    makes the file safe to load.
    """
    stripped = [_strip_prefix(name) for name in names]
    if not all(_has_segment(stripped, needle)
               for needle in (*_ZIMAGE_NEEDLES, *_ZIMAGE_L2P_PRESENT)):
        return False
    if all(key in comfy_keys for key in _LUMINA2_KEYS):
        return _ZIMAGE_PIXEL_KEY not in comfy_keys  # ComfyUI's own test for pixel space
    return is_zimage_l2p_from_keys(list(names))
