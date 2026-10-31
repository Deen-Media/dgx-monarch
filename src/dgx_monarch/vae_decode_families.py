"""Which ComfyUI VAEs split, and through which stock decode (docs/MODELS.md).

A family splits only where ComfyUI's single-host tiled decode for that VAE is
a loop whose units decode independently and that the split can replay:

* MiniMax H3's video VAE tiles itself (``handles_tiling``): its decode runs
  temporal chunks, each spatially tiled and blended inside the model, then
  blends the chunks. Each chunk decodes from its own latent window, so the unit
  is one chunk. ComfyUI's VAE Decode and VAE Decode (Tiled) both run that same
  decode on this VAE, so both split.
* The Wan 2.1 and 2.2, HunyuanVideo 1.0 and 1.5 and LTX video VAEs tile through
  ``tiled_scale_multidim`` from ``VAE.decode_tiled_3d``, with each tile one call
  of the stock decode function. Only VAE Decode (Tiled) runs that loop; VAE
  Decode runs the whole latent through the model at once, a different result,
  so it never splits on these families.

Left out on purpose: CogVideoX tiles frames inside its spatial loop, so its
tiles are not independent; an LTX VAE whose decode draws unseeded noise
(timestep conditioning or injected noise) has no reproducible reference;
image VAEs and every unknown class run the reference decode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

H3_CHUNK = "h3_chunk"
MULTIDIM = "multidim"
DECODE = "decode"  # VAEDecode
DECODE_TILED = "decode_tiled"  # VAEDecodeTiled
# Stills at or under this many output pixels decode in about a second on
# one GB10; a split would add a transfer and buy nothing (issue #93).
STILL_SPLIT_MIN_PIXELS = 1_200_000


@dataclass(frozen=True)
class Family:
    """One splittable VAE family."""

    name: str
    kind: str
    entries: frozenset[str]


_H3 = Family("minimax_h3", H3_CHUNK, frozenset({DECODE, DECODE_TILED}))
_TILED = frozenset({DECODE_TILED})
_CLASSES = {
    "comfy.ldm.wan.vae.WanVAE": "wan21",
    "comfy.ldm.wan.vae2_2.WanVAE": "wan22",
    "comfy.ldm.lightricks.vae.causal_video_autoencoder.VideoVAE": "ltx",
    "comfy.ldm.lightricks.vae.na_diffusion_decoder.CausalDiffusionVAE": "ltx",
}
_HUNYUAN15_DECODER = "comfy.ldm.hunyuan_video.vae_refiner.Decoder"
_SUPPORTED = ("MiniMax H3, Wan 2.1 and 2.2, HunyuanVideo 1.0 and 1.5 and LTX video VAEs "
              "split (docs/MODELS.md)")


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _formula(vae: Any) -> tuple | None:
    value = getattr(vae, "upscale_index_formula", None)
    return tuple(value) if isinstance(value, (tuple, list)) else None


def classify(vae: Any) -> tuple[Family | None, str]:
    """``(family, "")`` for a splittable VAE, else ``(None, why not)``."""
    fsm = getattr(vae, "first_stage_model", None)
    if fsm is None:
        return None, "the VAE has no decoder model"
    name = _qualified(type(fsm))
    if name == "comfy.ldm.minimax.vae.MiniMaxH3VideoVAE":
        if not getattr(vae, "handles_tiling", False) or getattr(fsm, "decoder", None) is None:
            return None, ("this ComfyUI decodes the MiniMax H3 VAE some other way than its own "
                          "chunked decode, so the split cannot replay it")
        return _H3, ""
    if name == "comfy.ldm.cogvideo.vae.AutoencoderKLCogVideoX":
        return None, "CogVideoX tiles frames inside its spatial loop, so its tiles are not independent"
    if getattr(vae, "latent_dim", None) != 3 or getattr(vae, "handles_tiling", False):
        return None, f"no split for {name}: {_SUPPORTED}"
    family = _CLASSES.get(name)
    if family == "wan22" and _formula(vae) != (4, 16, 16):
        family = None
    if name.endswith("causal_video_autoencoder.VideoVAE") and (
            getattr(fsm, "timestep_conditioning", False)
            or any(getattr(module, "inject_noise", False) for module in fsm.modules())):
        return None, ("this LTX VAE's decode draws unseeded noise, so no two decodes match "
                      "and a byte-exact split has no reference")
    if family is None and name == "comfy.ldm.models.autoencoder.AutoencoderKL":
        if _formula(vae) == (4, 8, 8) and getattr(vae, "latent_channels", None) == 16:
            family = "hunyuan_video"
    if family is None and name == "comfy.ldm.models.autoencoder.AutoencodingEngine":
        decoder = getattr(fsm, "decoder", None)
        if (decoder is not None and _qualified(type(decoder)) == _HUNYUAN15_DECODER
                and not getattr(vae, "not_video", False)):
            family = "hunyuan_video15"
    if family is None:
        return None, f"no split for {name}: {_SUPPORTED}"
    return Family(family, MULTIDIM, _TILED), ""


def small_still(vae: Any, samples: Any) -> int | None:
    """The output pixel count of a still at or under the split floor, else None."""
    shape = tuple(int(n) for n in getattr(samples, "shape", ()))
    if len(shape) < 4 or (len(shape) == 5 and shape[2] > 1):
        return None
    try:
        ratio = int(vae.spacial_compression_decode())
    except Exception:
        ratio = 8
    pixels = shape[-2] * shape[-1] * ratio * ratio
    return pixels if pixels <= STILL_SPLIT_MIN_PIXELS else None
