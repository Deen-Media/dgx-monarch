"""The wire value, refusal reasons, window schedule and row count for MiniMax H3 context windows.

ComfyUI's MiniMax H3 Context Windows node (#17425, in ComfyUI 0.44.0 or newer,
comfy_extras/nodes_context_windows.py) samples a long clip as overlapping
windows of the video latent's time axis. Stock ``calc_cond_batch`` hands each
window to the model as one ordinary call on a shorter clip and blends the
results, so the forward itself never changes. The driver node
(nodes/h3_context_windows.py) carries that node's five widgets to the Workers,
and each Worker applies the stock node to its render clone
(actor/context_windows_render.py), so the windows are ComfyUI's own.

The driver and the Worker share what this module holds without ComfyUI: the
wire value, the refusal reasons, and the static schedule the stock node runs,
which lets the sample deadline count the rows every window samples
(nodes/render_rows.py). Standard library only, like skip_layer_guidance.py;
tests/test_h3_context_windows_comfy.py pins the schedule, the frame
arithmetic and the widgets to ComfyUI's own code.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypedDict

from .h3_rows import latent_t_for_frames

FIELD: Final = "context_windows"
FAMILY: Final = "minimax_h3"
STOCK_NODE: Final = "MiniMaxH3ContextWindows"
NODE_NAME: Final = "MiniMax H3 Context Windows (DGX Monarch)"

# The stock node's widgets: names, order, defaults and ranges. Lengths are
# pixel frames of the form 17n + 5; ComfyUI's MAX_RESOLUTION bounds the length,
# and the overlap has no maximum.
MAX_RESOLUTION: Final = 16384
CONTEXT_LENGTH: Final = {"default": 124, "min": 22, "max": MAX_RESOLUTION, "step": 17}
CONTEXT_OVERLAP: Final = {"default": 39, "min": 5, "step": 17}
FUSE_METHODS: Final = ("pyramid", "relative", "flat", "overlap-linear")
FUSE_DEFAULT: Final = "pyramid"
FREENOISE_DEFAULT: Final = True
SPLIT_CONDS_DEFAULT: Final = False

# Pixel frames per video latent within one 17-frame chunk of five latents
# (comfy.ldm.minimax.model.FRAME_PER_TOKEN), and the chunk the stock node caps
# the overlap by so that every window keeps at least one chunk of new frames.
FRAME_PER_TOKEN: Final = (1, 4, 4, 4, 4)
CHUNK_LATENTS: Final = 5

_KEYS = frozenset({"context_length", "context_overlap", "fuse_method", "freenoise",
                   "split_conds_to_windows"})


class ContextWindowsWire(TypedDict):
    context_length: int
    context_overlap: int
    fuse_method: str
    freenoise: bool
    split_conds_to_windows: bool


@dataclass(frozen=True)
class ContextWindowsSpec:
    """Canonical immutable graph-side form of the node's widgets."""

    context_length: int
    context_overlap: int
    fuse_method: str
    freenoise: bool
    split_conds_to_windows: bool

    def to_wire(self) -> ContextWindowsWire:
        return {
            "context_length": self.context_length,
            "context_overlap": self.context_overlap,
            "fuse_method": self.fuse_method,
            "freenoise": self.freenoise,
            "split_conds_to_windows": self.split_conds_to_windows,
        }


def _frames(value: object, name: str, bounds: Mapping[str, int]) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{FIELD} {name} must be an integer")
    if value < bounds["min"] or value > bounds.get("max", value):
        top = f" to {bounds['max']}" if "max" in bounds else " or more"
        raise ValueError(f"{FIELD} {name} must be {bounds['min']}{top}")
    return value


def _flag(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{FIELD} {name} must be true or false")
    return value


def normalize_context_windows(value: object | None) -> ContextWindowsWire | None:
    """Validate and copy a context-window wire value, or None for no windows.

    The exact-key check keeps the protocol fail-closed, as
    skip_layer_guidance.py does. Bounds are the ones ComfyUI's prompt
    validation enforces (min and max); the 17-frame step is the frontend's,
    and the stock node rounds any length down to a chunk boundary itself.
    """
    if value is None:
        return None
    if isinstance(value, ContextWindowsSpec):
        value = value.to_wire()
    if not isinstance(value, Mapping):
        raise ValueError(f"{FIELD} must be a mapping or null")
    if set(value) != _KEYS:
        raise ValueError(f"{FIELD} must contain exactly {', '.join(sorted(_KEYS))}")
    fuse = value["fuse_method"]
    if fuse not in FUSE_METHODS:
        raise ValueError(f"{FIELD} fuse_method must be one of {', '.join(FUSE_METHODS)}")
    return {
        "context_length": _frames(value["context_length"], "context_length", CONTEXT_LENGTH),
        "context_overlap": _frames(value["context_overlap"], "context_overlap", CONTEXT_OVERLAP),
        "fuse_method": str(fuse),
        "freenoise": _flag(value["freenoise"], "freenoise"),
        "split_conds_to_windows": _flag(value["split_conds_to_windows"], "split_conds_to_windows"),
    }


def freeze_context_windows(value: object | None) -> ContextWindowsSpec | None:
    """The immutable graph-side form, or None for no windows."""
    wire = normalize_context_windows(value)
    return None if wire is None else ContextWindowsSpec(**wire)


def context_windows_field(value: object | None) -> dict[str, ContextWindowsWire]:
    """The model request's context-window entry, empty without the node.

    A graph without the node keeps its request byte for byte, so no Gate
    binding, Studio row or pinned request moves.
    """
    wire = normalize_context_windows(value)
    return {} if wire is None else {FIELD: wire}


def latent_window_settings(wire: ContextWindowsWire) -> tuple[int, int]:
    """(length, overlap) in video latents, as the stock node converts its widgets."""
    length = latent_t_for_frames(wire["context_length"])
    overlap = min(latent_t_for_frames(wire["context_overlap"]), length - CHUNK_LATENTS)
    return length, overlap


def latent_windows(latent_t: int, wire: ContextWindowsWire) -> list[list[int]]:
    """The video-latent windows of one step: stock's static standard schedule.

    A clip no longer than one window runs as one call over every latent, the
    path stock takes when it declines to window.
    """
    length, overlap = latent_window_settings(wire)
    if latent_t <= length:
        return [list(range(latent_t))]
    windows = []
    for start in range(0, latent_t, length - overlap):
        if start + length >= latent_t:
            final = latent_t - length
            windows.append(list(range(final, final + length)))
            break
        windows.append(list(range(start, start + length)))
    return windows


def audio_span(window: list[int], latent_t: int, audio_t: int) -> int:
    """Audio latent frames the stock mapping gives one video window.

    MiniMaxH3.map_context_window_to_modalities splits the shared time axis at
    the video latents' pixel-frame boundaries, scaled to the audio length.
    """
    starts = [0]
    for k in range(latent_t):
        starts.append(starts[-1] + FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)])
    edges = [round(s * audio_t / starts[-1]) for s in starts]
    return sum(edges[k + 1] - edges[k] for k in window)


def windowed_rows(*, latent_t: int, frame_rows: int, audio_t: int, fixed_rows: int,
                  wire: ContextWindowsWire) -> int:
    """Packed rows one sampler step runs across every window.

    ``fixed_rows`` is the text, keyframe and reference rows of one call; each
    window carries them again. Stock crops keyframe guides to the window, so
    counting the whole guide per window counts high, the safe side for a
    deadline. One window gives the whole clip's rows.
    """
    return sum(fixed_rows + len(window) * frame_rows + 2 * audio_span(window, latent_t, audio_t)
               for window in latent_windows(latent_t, wire))


def family_refusal_reason(family: str | None) -> str | None:
    """Why this checkpoint cannot take the node, or None when it can.

    The stock node cuts windows on MiniMax H3's 17-frame chunks and maps its
    audio by H3's frame layout, and only H3's sharded forward is shown to
    rebuild its packed layout for each window. An unreadable family is None:
    the Worker belt reads the loaded model instead.
    """
    if family is None or family == FAMILY:
        return None
    return (f"{NODE_NAME} runs ComfyUI's MiniMax H3 context windows, which cut the clip on "
            "MiniMax H3's 17-frame chunks and split its audio by H3's frame layout, and only "
            f"the MiniMax H3 forward here is shown to rebuild its packed sequence for each "
            f"window; this checkpoint reads as {family}. Load a MiniMax H3 checkpoint to keep "
            "the windows.")


DUAL_MODEL_REASON: Final = (
    f"{NODE_NAME} feeds a two-checkpoint guider's negative model here, and the Workers carry "
    "context windows on the primary model only. Wire the node on the model the sampler or "
    "the guider's positive input takes to keep the windows.")

STOCK_NODE_MISSING: Final = (
    f"{NODE_NAME} needs ComfyUI's {STOCK_NODE} node, in ComfyUI 0.44.0 or newer, and this "
    "Worker's ComfyUI has none, so it would render the whole clip at once instead of the "
    "windows the graph asks for. Update ComfyUI to 0.44.0 or newer on every host to keep the "
    "windows.")
