"""Integer-only MiniMax H3 packed-row accounting.

The capacity total includes text, keyframe condition, reference, target audio,
and target video rows. A generic latent shape exposes only video and would
undercount. Formulas follow the H3 packed-layout and temporal-grid contracts
exposed by ComfyUI. Parsing failures return zero or unresolved rows, avoiding
false refusals. H3 is structurally batch-1, so no batch multiplier applies.

This standard-library leaf reads ints, mappings and anything with a ``shape``.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

# Checkpoint and latent-grid constants.
H3_PATCH: Final = 2
H3_VAE_SPATIAL: Final = 16
H3_FPS: Final = 24
H3_AUDIO_LATENT_FPS: Final = 40

# Legal frame counts are at least 5 and congruent to 5 modulo 17.
H3_FRAME_PERIOD: Final = 17
H3_FRAME_PHASE: Final = 5
H3_LATENT_PERIOD: Final = 5
H3_MIN_LATENT_T: Final = 2

# Tests pin these stdlib-local keys to driver_footprint's profile.
REFS_KEY: Final = "minimax_refs"
KEYFRAMES_KEY: Final = "minimax_keyframes"
REF_EXTRAS_KEYS: Final[tuple[str, str]] = (REFS_KEY, KEYFRAMES_KEY)

# The empty-latent node is the only canvas node outside the family prefix.
H3_NODE_PREFIX: Final = "MiniMaxH3"
H3_EMPTY_LATENT_CLASS: Final = "EmptyMiniMaxH3LatentAV"
# MiniMax-H3-Image's conditioning node carries its canvas; 0 on a side means 2048
# without images (comfy_extras/nodes_minimax_h3.py), the stand-in used here.
H3_IMAGE_CANVAS_CLASS: Final = "MiniMaxH3ImageConditioning"
H3_IMAGE_DEFAULT_SIDE: Final = 2048


@dataclass(frozen=True, slots=True)
class H3Rows:
    """Packed rows by stream; unresolved geometry charges nothing."""

    video: int = 0
    audio: int = 0
    text: int = 0
    cond: int = 0          # keyframe condition segments
    ref: int = 0           # every reference block, image, video or audio
    frame_rows: int = 0    # the target grid, ceil(lat_h/2) * ceil(lat_w/2)
    latent_t: int = 0
    resolved: bool = False

    @property
    def total(self) -> int:
        return self.video + self.audio + self.text + self.cond + self.ref

    def per_rank(self, sp_degree: int) -> int:
        """Ceiling rows per rank; pad rows still consume memory."""
        degree = max(int(sp_degree), 1)
        return -(-self.total // degree)

    def stream_summary(self) -> str:
        return (f"{self.video} video, {self.audio} audio, {self.text} text, "
                f"{self.cond} keyframe, {self.ref} reference")


def _dim(value: object) -> int:
    """Return one integer dimension, or zero."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def _dims(shape: object) -> list[int] | None:
    """Return an integer shape, or None when unreadable."""
    if shape is None or isinstance(shape, (str, bytes, Mapping)):
        return None
    if not isinstance(shape, Iterable):
        return None
    return [_dim(item) for item in shape]


def frame_rows(latent_h: int, latent_w: int) -> int:
    """Target-frame rows: the grid rounds up to even, then packs by halving.

    The two compose to ceil(latent/2) per axis. Stock canvases are even; the
    ceiling also covers a hand-set odd dimension.
    """
    return -(-max(latent_h, 0) // H3_PATCH) * -(-max(latent_w, 0) // H3_PATCH)


def ref_frame_rows(latent_h: int, latent_w: int) -> int:
    """Reference-frame rows using the packer's floor halving.

    References skip target-grid rounding. Ceiling on an odd dimension would
    overcount and risk a false refusal. Stock nodes emit 32-pixel multiples.
    """
    return (max(latent_h, 0) // H3_PATCH) * (max(latent_w, 0) // H3_PATCH)


def align_frame_count(length: int) -> int:
    """Smallest legal frame count at or above the request."""
    count = max(int(length), H3_FRAME_PHASE)
    while count % H3_FRAME_PERIOD != H3_FRAME_PHASE:
        count += 1
    return count


def latent_t_for_frames(frame_count: int) -> int:
    """Latent frames for a legal generation frame count."""
    if frame_count <= H3_FRAME_PHASE:
        return H3_MIN_LATENT_T
    return (((frame_count - H3_FRAME_PHASE) // H3_FRAME_PERIOD)
            * H3_LATENT_PERIOD + H3_MIN_LATENT_T)


def frames_for_latent_t(latent_t: int) -> int:
    """Inverse of ``latent_t_for_frames`` on the legal grid.

    A test pins this stdlib-local formula to driver_footprint's reference
    geometry calculation.
    """
    if latent_t <= H3_MIN_LATENT_T:
        return H3_FRAME_PHASE
    return (H3_FRAME_PERIOD * (latent_t - H3_MIN_LATENT_T) // H3_LATENT_PERIOD
            + H3_FRAME_PHASE)


def audio_t_for_frames(frame_count: int) -> int:
    """Audio latent frames for a generation frame count."""
    return round(max(frame_count, 0) * H3_AUDIO_LATENT_FPS / H3_FPS)


def _audio_t_from_samples(samples: object, latent_t: int) -> int:
    """Audio latent frames from the nested latent's audio tensor, else the frame grid."""
    tensors = getattr(samples, "tensors", None)
    if isinstance(tensors, (list, tuple)) and len(tensors) >= 2:
        dims = _dims(getattr(tensors[1], "shape", None))
        if dims:
            return max(dims[-1], 0)
    return audio_t_for_frames(frames_for_latent_t(latent_t))


def _keyframe_block_rows(block: Mapping[str, Any], rows_per_frame: int) -> int:
    """Rows one anchored guide adds, from the latents it already carries.

    A guide reaches this call encoded: the node pops its image and audio and
    stores ``latent`` and ``audio_latent``, so the video latent's frame count
    times the target grid, plus two rows per audio latent frame, is exact. A
    guide may carry either latent alone. Unreadable shapes charge nothing,
    which undercounts rather than refusing a render that fits.
    """
    rows = 0
    video_dims = _dims(getattr(block.get("latent"), "shape", None))
    if video_dims is not None and len(video_dims) >= 3:
        rows += video_dims[2] * rows_per_frame
    audio_dims = _dims(getattr(block.get("audio_latent"), "shape", None))
    if audio_dims:
        rows += 2 * audio_dims[-1]
    return rows


def _ref_block_rows(block: Mapping[str, Any]) -> int:
    """Rows for one reference block; missing integers undercount permissively."""
    latent_t = _dim(block.get("latent_t"))
    grid = ref_frame_rows(_dim(block.get("latent_h")), _dim(block.get("latent_w")))
    audio_rows = 2 * _dim(block.get("ref_audio_t"))
    kind = block.get("kind")
    if kind == "image":
        return grid
    if kind == "audio":
        return audio_rows
    if kind in ("video", "video_audio"):
        return audio_rows + latent_t * grid
    # Unknown kinds use only the geometry they provide.
    return audio_rows + (latent_t * grid if latent_t else grid)


def _positive_entries(request: Any) -> tuple[Any, ...]:
    positive = request.get("positive") if isinstance(request, Mapping) else None
    if not isinstance(positive, (list, tuple)):
        return ()
    return tuple(positive)


def rows_from_request(request: Any, latent: Any, *,
                      video_shape: Any = None, image: bool = False) -> H3Rows:
    """Return the dispatch's packed total from duck-typed runtime inputs.

    ``video_shape``, when given, is the grid the workers will sample (size
    tags applied) and stands in for the latent's own shape. ``image`` counts
    MiniMax-H3-Image's stream: one plain latent frame and no audio rows.
    """
    if not isinstance(latent, Mapping):
        return H3Rows()
    samples = latent.get("samples")
    dims = _dims(video_shape if video_shape is not None
                 else getattr(samples, "shape", None))
    if dims is None or len(dims) < 5:
        return H3Rows()
    latent_t, lat_h, lat_w = dims[2], dims[3], dims[4]
    rows_per_frame = frame_rows(lat_h, lat_w)
    audio_t = 0 if image else _audio_t_from_samples(samples, latent_t)
    text = cond = ref = 0
    # Extras are copied to every positive entry, not appended. Each entry is a
    # separate model call, as are cond and mirrored uncond, so peak memory is
    # the largest entry. Summing would multiply references and falsely refuse.
    for entry in _positive_entries(request):
        if not isinstance(entry, (list, tuple)) or not entry:
            continue
        shape = _dims(getattr(entry[0], "shape", None))
        if shape is not None and len(shape) >= 2:
            text = max(text, shape[1])
        extra = entry[1] if len(entry) > 1 else None
        if not isinstance(extra, Mapping):
            continue
        keyframes = extra.get(KEYFRAMES_KEY)
        if isinstance(keyframes, (list, tuple)):
            # Keyframe latents are already encoded on the target canvas.
            cond = max(cond, sum(_keyframe_block_rows(block, rows_per_frame)
                                 for block in keyframes
                                 if isinstance(block, Mapping)))
        refs = extra.get(REFS_KEY)
        if isinstance(refs, (list, tuple)):
            ref = max(ref, sum(_ref_block_rows(block) for block in refs
                               if isinstance(block, Mapping)))
    return H3Rows(
        video=latent_t * rows_per_frame, audio=2 * audio_t, text=text,
        cond=cond, ref=ref, frame_rows=rows_per_frame, latent_t=latent_t,
        resolved=True)


# The exact count below is for divisibility, not capacity. The capacity count
# above takes the largest entry and undercounts what it cannot read; this one
# counts one model call as stock's ``PackedLayout.seq_len`` does, or returns
# None, so a caller that refuses on it refuses only what it read. The canary
# (tests/canary/comfy_seam_contracts.py) holds it to the installed ComfyUI.


def _whole(value: object) -> int | None:
    """A non-negative integer, or None."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _positive_dims(value: object, least: int) -> list[int] | None:
    """A tensor's shape with at least ``least`` positive dimensions, or None."""
    dims = _dims(getattr(value, "shape", None))
    if dims is None or len(dims) < least or min(dims) <= 0:
        return None
    return dims


def exact_geometry(latent: Any, *, video_shape: Any = None,
                   image: bool = False) -> tuple[int, int, int] | None:
    """(latent_t, rows per frame, audio_t) of the grid the workers sample, or None.

    ``video_shape`` is that grid with size tags applied, as in
    ``rows_from_request``. The audio length comes from the nested latent's
    audio tensor only; MiniMax-H3-Image has none.
    """
    if not isinstance(latent, Mapping):
        return None
    samples = latent.get("samples")
    dims = _dims(video_shape if video_shape is not None
                 else getattr(samples, "shape", None))
    if dims is None or len(dims) < 5 or min(dims[2:5]) <= 0:
        return None
    audio_t = 0
    if not image:
        tensors = getattr(samples, "tensors", None)
        if not isinstance(tensors, (list, tuple)) or len(tensors) < 2:
            return None
        audio = _positive_dims(tensors[1], 1)
        if audio is None:
            return None
        audio_t = audio[-1]
    return dims[2], frame_rows(dims[3], dims[4]), audio_t


def _exact_keyframe_rows(block: object, rows_per_frame: int) -> int | None:
    """One anchored guide's rows as stock packs them: its video latent's frames
    on the target grid, two per audio latent frame; None when unreadable."""
    if not isinstance(block, Mapping):
        return None
    rows = 0
    if block.get("latent") is not None:
        video = _positive_dims(block["latent"], 3)
        if video is None:
            return None
        rows += video[2] * rows_per_frame
    if block.get("audio_latent") is not None:
        audio = _positive_dims(block["audio_latent"], 1)
        if audio is None:
            return None
        rows += 2 * audio[-1]
    return rows


def _exact_ref_rows(block: object) -> int | None:
    """One reference block's rows as stock packs them, or None.

    Stock packs nothing for a kind it does not name, and a newer ComfyUI may
    name more, so an unknown kind returns None rather than zero.
    """
    if not isinstance(block, Mapping):
        return None
    kind = block.get("kind")
    if kind == "image":
        lat_h, lat_w = _whole(block.get("latent_h")), _whole(block.get("latent_w"))
        if lat_h is None or lat_w is None:
            return None
        return ref_frame_rows(lat_h, lat_w)
    if kind == "audio":
        audio_t = _whole(block.get("ref_audio_t"))
        return None if audio_t is None else 2 * audio_t
    if kind in ("video", "video_audio"):
        values = [_whole(block.get(key))
                  for key in ("ref_audio_t", "latent_t", "latent_h", "latent_w")]
        if any(value is None for value in values):
            return None
        audio_t, latent_t, lat_h, lat_w = (int(value or 0) for value in values)
        return 2 * audio_t + latent_t * ref_frame_rows(lat_h, lat_w)
    return None


def exact_call_total(entry: object, geometry: tuple[int, int, int], *,
                     image: bool = False) -> int | None:
    """One model call's packed length, as stock's ``PackedLayout.seq_len``, or None.

    ``entry`` is one ``[text, extras]`` conditioning entry. Each entry is its
    own model call with its own layout, so each pads or not on its own.
    ``geometry`` is ``exact_geometry``'s answer. MiniMax-H3-Image packs text,
    references and its one frame: stock hands its layout no guides and no audio.
    """
    if not isinstance(entry, (list, tuple)) or not entry:
        return None
    text = _positive_dims(entry[0], 2)
    if text is None:
        return None
    extra = entry[1] if len(entry) > 1 else {}
    if not isinstance(extra, Mapping):
        return None
    latent_t, rows_per_frame, audio_t = geometry
    total = text[1] + latent_t * rows_per_frame + (0 if image else 2 * audio_t)
    keys = (REFS_KEY,) if image else (KEYFRAMES_KEY, REFS_KEY)
    for key in keys:
        blocks = extra.get(key)
        if blocks is None:
            continue
        if not isinstance(blocks, (list, tuple)):
            return None
        for block in blocks:
            rows = (_exact_keyframe_rows(block, rows_per_frame) if key == KEYFRAMES_KEY
                    else _exact_ref_rows(block))
            if rows is None:
                return None
            total += rows
    return total


def _canvas_node(node: object) -> Mapping[str, Any] | None:
    """A node's inputs when it is one of this family's canvas-bearing nodes."""
    if not isinstance(node, Mapping):
        return None
    class_type = node.get("class_type")
    if not isinstance(class_type, str):
        return None
    if not (class_type.startswith(H3_NODE_PREFIX)
            or class_type == H3_EMPTY_LATENT_CLASS):
        return None
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, Mapping) else None


def canvas_pixels_of_node(prompt: Any, node_id: Any) -> int:
    """Return the canvas area of one node, when that node carries a canvas.

    A guide is resized to the canvas of the latent it is anchored into, which
    is a link, not the largest canvas in the graph.
    """
    if not isinstance(prompt, Mapping) or node_id is None:
        return 0
    node = prompt.get(node_id)
    if node is None:
        node = prompt.get(str(node_id))
    inputs = _canvas_node(node)
    if inputs is None:
        return 0
    width = _dim(inputs.get("width"))
    height = _dim(inputs.get("height"))
    return width * height if width > 0 and height > 0 else 0


def canvas_pixels_from_prompt_graph(prompt: Any) -> int:
    """Return the largest canvas area this graph declares, in pixels.

    Add Guide resizes its frames to the target canvas before encoding them, so
    this area is the encode geometry. Canvases are alternative renders, and a
    guide encode costs the most at the largest.
    """
    if not isinstance(prompt, Mapping):
        return 0
    best = 0
    for node in prompt.values():
        inputs = _canvas_node(node)
        if inputs is None:
            continue
        width = _dim(inputs.get("width"))
        height = _dim(inputs.get("height"))
        if width > 0 and height > 0:
            best = max(best, width * height)
    return best


def _image_canvas_rows(inputs: Mapping[str, Any]) -> H3Rows:
    """One MiniMax-H3-Image target frame from the conditioning node's canvas, no audio."""
    width = _dim(inputs.get("width")) or H3_IMAGE_DEFAULT_SIDE
    height = _dim(inputs.get("height")) or H3_IMAGE_DEFAULT_SIDE
    rows_per_frame = frame_rows(height // H3_VAE_SPATIAL, width // H3_VAE_SPATIAL)
    return H3Rows(video=rows_per_frame, frame_rows=rows_per_frame, latent_t=1, resolved=True)


def rows_from_prompt_graph(prompt: Any, *, image: bool = False) -> H3Rows:
    """Return the largest canvas's loader-time packed estimate.

    Graph literals resolve target video and audio only; linked keyframe and
    reference geometry remains zero, a permissive undercount. Canvases are
    alternative renders, so peak memory is their maximum rather than their sum.
    ``image`` reads MiniMax-H3-Image's conditioning nodes instead: one target
    frame each, at 2048 on a side the node leaves at 0 (an edit that sizes from
    its first picture may differ), text and reference rows unseen.
    """
    if not isinstance(prompt, Mapping):
        return H3Rows()
    best = H3Rows()
    for node in prompt.values():
        inputs = _canvas_node(node)
        if inputs is None:
            continue
        if image != (node.get("class_type") == H3_IMAGE_CANVAS_CLASS):
            continue  # each family reads only its own canvas nodes
        if image:
            rows = _image_canvas_rows(inputs)
            best = rows if rows.total > best.total else best
            continue
        width = _dim(inputs.get("width"))
        height = _dim(inputs.get("height"))
        length = _dim(inputs.get("length"))
        if width <= 0 or height <= 0 or length <= 0:
            continue
        frames = align_frame_count(length)
        latent_t = latent_t_for_frames(frames)
        rows_per_frame = frame_rows(height // H3_VAE_SPATIAL,
                                    width // H3_VAE_SPATIAL)
        rows = H3Rows(
            video=latent_t * rows_per_frame,
            audio=2 * audio_t_for_frames(frames),
            frame_rows=rows_per_frame, latent_t=latent_t, resolved=True)
        if rows.total > best.total:
            best = rows
    return best
