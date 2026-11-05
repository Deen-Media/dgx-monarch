"""The MiniMax H3 sampling shift on the Workers, and the one-frame image latent it guards.

ComfyUI's ModelSamplingMiniMaxH3 node (``MiniMaxH3SigmaShift``,
comfy_extras/nodes_minimax_h3.py) patches a stock MODEL, so it cannot connect to
a DGX Monarch loader's DGXM_MODEL. ModelSamplingMiniMaxH3 (DGX Monarch) carries
its two widgets on the model_sampling wire (model_sampling.py), and every Worker
applies the stock node to its render clone (actor/model_sampling_render.py), so
the shifts, the audio scale and the two transformer_options keys the H3 forward
reads are stock's.

p-h3-image (ComfyUI #17506) is a MiniMax H3 DiT: ComfyUI builds it as
``MiniMaxH3``, and ``PH3ImageConditioning`` hands the sampler one video latent
frame beside two audio latent frames. A MiniMax H3 video latent never has one
frame (frames 17k + 5 map to latents 5k + 2), so a one-frame latent is this
model's image graph. Its workflows set both shifts to 1.0; without the node the
Workers would sample at H3's defaults, 12.0 and 3.0. Standard library only, like
h3_context_windows.py; tests/test_h3_sigma_shift_comfy.py pins the widgets,
the keys and the latent shape to ComfyUI's own code.
"""
from __future__ import annotations

from typing import Any, Final

FAMILY: Final = "minimax_h3"
STOCK_NODE: Final = "MiniMaxH3SigmaShift"
STOCK_DISPLAY: Final = "ModelSamplingMiniMaxH3"
NODE_NAME: Final = "ModelSamplingMiniMaxH3 (DGX Monarch)"
CONDITIONING_NODE: Final = "PH3ImageConditioning"
TROUBLESHOOTING: Final = 144

# The stock widgets: names, order, defaults and ranges.
SHIFT_VIDEO: Final = {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}
SHIFT_AUDIO: Final = {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}
# The transformer_options keys the stock node sets and the H3 forward reads.
VIDEO_KEY: Final = "minimax_h3_sigma_shift_video"
AUDIO_KEY: Final = "minimax_h3_sigma_shift_audio"
# p-h3-image's shifts in every vendor and ComfyUI workflow.
P_H3_IMAGE_SHIFT: Final = 1.0
# PH3ImageConditioning's latent: one video frame of 24 channels, two audio frames.
IMAGE_LATENT_T: Final = 1
IMAGE_AUDIO_T: Final = 2


def family_refusal_reason(family: str | None) -> str | None:
    """Why this checkpoint cannot take the node, or None when it can.

    An unreadable header (None) and one no signature matches ("unknown") pass:
    the Worker belt reads the loaded model's class instead. The text names the
    cause; each refusal site adds its own remedy.
    """
    if family is None or family in (FAMILY, "unknown"):
        return None
    return (f"{NODE_NAME} applies ComfyUI's {STOCK_DISPLAY} node, which replaces the model's "
            "sampling with MiniMax H3's paired video and audio schedule, and this checkpoint "
            f"reads as {family}.")


# The stock node has shipped with MiniMax H3 since ComfyUI 0.30.0 (#15224), so
# its absence means a ComfyUI without MiniMax H3, or one that moved the node.
STOCK_NODE_PATH: Final = f"comfy_extras.nodes_minimax_h3.{STOCK_NODE}"
STOCK_NODE_MISSING: Final = (
    f"{NODE_NAME} runs ComfyUI's {STOCK_DISPLAY} node, {STOCK_NODE_PATH}, which has shipped "
    "with MiniMax H3 since ComfyUI 0.30.0, and this Worker's ComfyUI has none: it has no "
    "MiniMax H3 support, or a later release moved or renamed the node. Without it the Worker "
    "would sample at the checkpoint's own shifts.")


def _dims(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return None


def is_image_latent(latent: Any) -> bool:
    """Whether a latent dict holds MiniMax H3's audio-video pair with one video frame.

    Reads duck-typed: a NestedTensor's ``tensors``, a video of five dimensions
    whose time axis is 1, then an audio tensor. Anything else is False.
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    tensors = getattr(samples, "tensors", None)
    if not isinstance(tensors, (list, tuple)) or len(tensors) < 2:
        return False
    video = _dims(tensors[0])
    return video is not None and len(video) == 5 and video[2] == IMAGE_LATENT_T


def missing_shift_reason(kind: str | None) -> str | None:
    """Why a one-frame H3 latent cannot sample under this model's sampling, or None.

    Any value on the node passes: only a graph with no MiniMax H3 shift at all refuses.
    """
    if kind == "minimax_h3":
        return None
    told = ("Model Sampling SD3 (DGX Monarch) sets an SD3 shift, which the H3 forward does not "
            "read. " if kind == "sd3" else "")
    return ("this render samples a MiniMax H3 audio-video latent with one video frame, the image "
            f"latent {CONDITIONING_NODE} makes for p-h3-image, and nothing in the graph sets the "
            f"MiniMax H3 shifts. {told}The Workers would sample it at H3's defaults, 12.0 for "
            "video and 3.0 for audio, where p-h3-image's own workflows use 1.0 and 1.0, so the "
            f"image would come out wrong. Add {NODE_NAME} between the loader and the guider or "
            "sampler, at 1.0 and 1.0.")
