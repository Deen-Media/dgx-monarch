"""LTX dual-stream activation pricing for the loader-site preflight.

LTX runs two token streams at two hidden widths, video at 4096 and audio at
2048, coupled per block by bidirectional cross attention. ``adapters/ltx.py``
shards each stream with its own ``shard_seq`` call, so each axis divides by the
sequence-parallel degree on its own and each pads on its own. Nothing here
packs them onto one row axis the way ``h3_rows`` does.

The charge is the graph's unspent render activations plus the per-rank worker
floor. ``nodes/loader_graph`` folds it into the settled driver-stack window, so
an oversized request is refused at the loader node, before the DiT is placed.
Both preflight kill switches stand this leaf down, and every parse failure
omits the charge rather than raising: uncertain evidence must not invent a
capacity refusal.

Calibration lives in ``ltx25_calibration``. Keep this leaf free of
runtime-heavy dependencies.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from . import driver_footprint, mesh_safety
from .ltx25_calibration import (
    LTX_RANK_FLOOR_BYTES,
    LTX_REPLICATED_AUDIO_ROW_BYTES,
    LTX_REPLICATED_VIDEO_ROW_BYTES,
    LTX_SHARDED_AUDIO_ROW_BYTES,
    LTX_SHARDED_VIDEO_ROW_BYTES,
)

LTX_FAMILY: Final = "ltx"

# Latent grid, from comfy's latent_formats.LTXV and EmptyLTXVLatentVideo.
LTX_SPATIAL_DOWNSCALE: Final = 32
LTX_TEMPORAL_DOWNSCALE: Final = 8
# 16000 Hz / 160 mel hop / 4 = 25 audio latents per second, read from the 2.5
# audio VAE's own metadata and comfy/ldm/lightricks/vae/audio_vae.py.
LTX_AUDIO_LATENTS_PER_SECOND: Final = 25

# Matched on class_type. Both nodes carry a literal pixel canvas; every other
# LTX latent node derives its geometry from a link this leaf cannot follow.
LTX_VIDEO_CANVAS_CLASSES: Final[tuple[str, ...]] = (
    "EmptyLTXVLatentVideo", "LTXVImgToVideo")
LTX_AUDIO_CANVAS_CLASSES: Final[tuple[str, ...]] = ("LTXVEmptyLatentAudio",)

# A guide appends rows the canvas node never declared. LTXV Add Guide always
# appends rather than replacing (comfy nodes_lt.py LTXVAddGuide.append_keyframe)
# and encodes the guide at the canvas latent resolution (LTXVAddGuide.encode),
# so a guide costs whole grids of the canvas it attaches to: 510 rows per latent
# frame at 960x544, 880 at 1280x704.
LTX_GUIDE_CLASSES: Final[tuple[str, ...]] = ("LTXVAddGuide",)
# Pixel frames a guide's image source declares in a widget, by class. A still
# image is one frame; a clip's length reaches the guide over an image link, and
# nothing in comfy core states it statically except a batch repeat. An
# unreadable source charges the one frame a still costs and the note says so.
LTX_GUIDE_FRAME_WIDGETS: Final[Mapping[str, str]] = {"RepeatImageBatch": "amount"}
# Add Guide chains through its own latent output, so a first-and-last-frame
# graph is two guides deep. The cap bounds the walk on a malformed graph.
LTX_GUIDE_CHAIN_LIMIT: Final = 16

# Presets whose parallel degree does not divide a row axis. Sequence parallel
# is the only degree that does, and the auto table gives ltx ulysses 2 at every
# canvas size (topology.AUTO_TABLE row 40), so auto prices at 2, capped at world.
_NON_SEQUENCE_PRESET_PREFIXES: Final[tuple[str, ...]] = (
    "cfg", "dp", "single", "stock", "local")


@dataclass(frozen=True, slots=True)
class LTXRows:
    """Rows by stream; unresolved geometry charges nothing."""

    video: int = 0
    audio: int = 0
    latent_t: int = 0
    frame_rows: int = 0
    guides: int = 0
    guide_video: int = 0
    unattributed_guides: int = 0
    resolved: bool = False

    @property
    def total(self) -> int:
        return self.video + self.audio

    def video_per_rank(self, sp_degree: int) -> int:
        """Ceiling rows per rank; a pad row leaves attention but still costs."""
        return -(-self.video // max(int(sp_degree), 1))

    def audio_per_rank(self, sp_degree: int) -> int:
        return -(-self.audio // max(int(sp_degree), 1))

    def stream_summary(self) -> str:
        return f"{self.video} video, {self.audio} audio"


def _dim(value: object) -> int:
    """Return one non-negative integer input, or zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _rate(value: object) -> float:
    """Return a positive frame rate, or zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value) if value > 0 else 0.0


def latent_frames(length: int) -> int:
    """Latent frames for a generation frame count (comfy nodes_lt.py:82)."""
    return (max(int(length), 1) - 1) // LTX_TEMPORAL_DOWNSCALE + 1


def video_rows(width: int, height: int, length: int, batch: int = 1) -> int:
    """Video rows: latent frames times the 32-pixel latent grid, times batch.

    The DiT patchifies 1x1 on the LTX latent grid, so a latent cell is a row.
    """
    grid = (_dim(height) // LTX_SPATIAL_DOWNSCALE) * (
        _dim(width) // LTX_SPATIAL_DOWNSCALE)
    return max(int(batch), 1) * latent_frames(length) * grid


def audio_rows(frames: int, frame_rate: object, batch: int = 1) -> int:
    """Audio rows: seconds of clip times 25 latents per second, times batch."""
    rate = _rate(frame_rate)
    if not rate:
        return 0
    return max(int(batch), 1) * round(
        _dim(frames) / rate * LTX_AUDIO_LATENTS_PER_SECOND)


def activation_bytes(rows: LTXRows, sp_degree: int) -> int:
    """Per-rank bytes: both sharded axes, both replicated axes, the floor."""
    degree = max(int(sp_degree), 1)
    return (rows.video_per_rank(degree) * LTX_SHARDED_VIDEO_ROW_BYTES
            + rows.audio_per_rank(degree) * LTX_SHARDED_AUDIO_ROW_BYTES
            + rows.video * LTX_REPLICATED_VIDEO_ROW_BYTES
            + rows.audio * LTX_REPLICATED_AUDIO_ROW_BYTES
            + LTX_RANK_FLOOR_BYTES)


def max_video_rows(budget: int, sp_degree: int, audio: int = 0) -> int:
    """Largest video-row total that clears ``budget`` at this topology.

    The closed form omits ``ceil(video/N)``; a bounded walk of at most ``N - 1``
    rows corrects the floor-division result.
    """
    degree = max(int(sp_degree), 1)
    left = (int(budget) - LTX_RANK_FLOOR_BYTES
            - (-(-max(audio, 0) // degree)) * LTX_SHARDED_AUDIO_ROW_BYTES
            - max(audio, 0) * LTX_REPLICATED_AUDIO_ROW_BYTES)
    if left <= 0:
        return 0
    rows = (left * degree
            // (LTX_SHARDED_VIDEO_ROW_BYTES
                + LTX_REPLICATED_VIDEO_ROW_BYTES * degree))
    while rows > 0 and (-(-rows // degree) * LTX_SHARDED_VIDEO_ROW_BYTES
                        + rows * LTX_REPLICATED_VIDEO_ROW_BYTES) > left:
        rows -= 1
    return rows


def _is_class(node: object, classes: tuple[str, ...]) -> bool:
    """Whether the node declares one of ``classes``, inputs or not."""
    return isinstance(node, Mapping) and node.get("class_type") in classes


def _canvas_inputs(node: object, classes: tuple[str, ...]) -> Mapping[str, Any] | None:
    """A node's inputs when its class_type is one of ``classes``."""
    if not _is_class(node, classes):
        return None
    inputs = node.get("inputs") if isinstance(node, Mapping) else None
    return inputs if isinstance(inputs, Mapping) else None


def _link_target(
    prompt: Mapping[str | int, Any], node: object, name: str,
) -> str | int | None:
    """The node id an input link points at, in API format ``[id, slot]``."""
    inputs = node.get("inputs") if isinstance(node, Mapping) else None
    link = inputs.get(name) if isinstance(inputs, Mapping) else None
    if not isinstance(link, (list, tuple)) or not link:
        return None
    target = link[0]
    if isinstance(target, bool) or not isinstance(target, (str, int)):
        return None
    # A graph may key nodes by strings or numbers. Return the actual key so the
    # caller can associate a guide with its canvas. Decimal strings get the
    # opposite spelling too, but ``"030"`` stays distinct from node ``30``.
    keys: tuple[str | int, ...] = (
        (target, str(target)) if isinstance(target, int) else (target,)
    )
    if (isinstance(target, str) and target.isascii() and target.isdecimal()
            and str(int(target)) == target):
        keys += (int(target),)
    for key in keys:
        if key in prompt:
            return key
    return None


def guide_pixel_frames(prompt: Mapping[str | int, Any], guide: object) -> int:
    """Pixel frames the guide's image source declares, defaulting to one.

    Only a source whose own widget states the count is read. Guessing a longer
    clip would invent a charge, and this leaf refuses a render it over-prices.
    """
    source_id = _link_target(prompt, guide, "image")
    source = prompt.get(source_id) if source_id is not None else None
    if not isinstance(source, Mapping):
        return 1
    widget = LTX_GUIDE_FRAME_WIDGETS.get(str(source.get("class_type")))
    inputs = source.get("inputs")
    if widget is None or not isinstance(inputs, Mapping):
        return 1
    return _dim(inputs.get(widget)) or 1


def guide_canvas_id(
    prompt: Mapping[str | int, Any], guide_id: str | int,
) -> str | int | None:
    """The video canvas a guide attaches to, following its latent chain.

    Add Guide takes the latent it appends to and returns the longer one, so a
    first-and-last-frame graph reaches its canvas through the guide ahead of it.
    """
    seen = {guide_id}
    current: object = prompt.get(guide_id)
    for _ in range(LTX_GUIDE_CHAIN_LIMIT):
        next_id = _link_target(prompt, current, "latent")
        if next_id is None or next_id in seen:
            return None
        seen.add(next_id)
        candidate = prompt.get(next_id)
        if _is_class(candidate, LTX_VIDEO_CANVAS_CLASSES):
            return next_id
        if not _is_class(candidate, LTX_GUIDE_CLASSES):
            return None
        current = candidate
    return None


def rows_from_prompt_graph(prompt: Any) -> LTXRows:
    """Return the largest declared canvas on each stream, plus its own guides.

    Canvases in one graph are alternative or sequential renders, so peak memory
    is their maximum rather than their sum, exactly as ``h3_rows`` treats an H3
    graph. Guides are charged to the canvas they attach to and the maximum is
    taken over those totals, so a guide never inflates a canvas it never
    touches. Comfy crops a guide clip to 8n + 1 pixel frames before encoding
    (nodes_lt.py LTXVAddGuide.encode), which is exactly the rounding
    ``latent_frames`` already applies, so the crop moves no rows.

    Three permissive undercounts remain, all deliberate: a clip whose source
    states no length charges one frame, a guide whose canvas is ambiguous
    charges nothing, and a two-stage graph hides its second canvas behind
    ``LTXVLatentUpsampler``, whose output geometry lives on a link rather than
    in a widget. An undercount omits a charge; an invented one would refuse a
    render that fits.

    A video canvas whose own geometry cannot be read is not one of those three.
    It leaves nothing to undercount from, so the whole graph reads unresolved
    and the caller charges nothing and says so.
    """
    if not isinstance(prompt, Mapping):
        return LTXRows()
    canvases: dict[str | int, tuple[int, int, int]] = {}
    best_audio = 0
    unreadable_canvas = False
    guide_ids: list[str | int] = []
    for node_id, node in prompt.items():
        if _is_class(node, LTX_GUIDE_CLASSES):
            guide_ids.append(node_id)
            continue
        inputs = _canvas_inputs(node, LTX_VIDEO_CANVAS_CLASSES)
        if inputs is not None:
            width = _dim(inputs.get("width"))
            height = _dim(inputs.get("height"))
            length = _dim(inputs.get("length"))
            batch = _dim(inputs.get("batch_size")) or 1
            if width and height and length:
                canvases[node_id] = (
                    video_rows(width, height, length, batch),
                    latent_frames(length),
                    (height // LTX_SPATIAL_DOWNSCALE) * (width // LTX_SPATIAL_DOWNSCALE),
                )
            else:
                # A widget converted to an input is a link, not a dimension. Skipping an
                # unreadable video canvas would omit its guides while an audio canvas
                # still made the graph appear resolved.
                unreadable_canvas = True
            continue
        inputs = _canvas_inputs(node, LTX_AUDIO_CANVAS_CLASSES)
        if inputs is None:
            continue
        rows = audio_rows(_dim(inputs.get("frames_number")),
                          inputs.get("frame_rate"),
                          _dim(inputs.get("batch_size")) or 1)
        best_audio = max(best_audio, rows)

    # A guide whose latent chain reaches no canvas is still unambiguous when
    # the graph declares exactly one, because there is nothing else to attach
    # to. With several, charging a guess would price a canvas it never touched.
    lone_canvas = next(iter(canvases)) if len(canvases) == 1 else None
    charged: dict[str | int, list[int]] = {}
    unattributed = 0
    for guide_id in guide_ids:
        canvas_id = guide_canvas_id(prompt, guide_id)
        if canvas_id is None:
            canvas_id = lone_canvas
        if canvas_id is None or canvas_id not in canvases:
            unattributed += 1
            continue
        frames = latent_frames(guide_pixel_frames(prompt, prompt.get(guide_id)))
        tally = charged.setdefault(canvas_id, [0, 0])
        tally[0] += 1
        tally[1] += frames

    if unreadable_canvas or (not canvases and not best_audio):
        return LTXRows()
    best = LTXRows(audio=best_audio, unattributed_guides=unattributed,
                   resolved=True)
    for canvas_id, (rows, latent_t, grid) in canvases.items():
        # Batch stays out of the guide charge: a guide appends a batch-1 latent
        # and comfy's concatenate rejects any wider canvas.
        count, frames = charged.get(canvas_id, (0, 0))
        guide_video = frames * grid
        if rows + guide_video <= best.video:
            continue
        best = LTXRows(
            video=rows + guide_video, audio=best_audio, latent_t=latent_t,
            frame_rows=grid, guides=count, guide_video=guide_video,
            unattributed_guides=unattributed, resolved=True)
    return best


def sp_degree_for_mesh(mesh: Any) -> int:
    """Return the sequence-parallel degree this render will use.

    An explicit cfg, dp or single-rank preset shards no row axis, so it prices
    at one. A uly or ring preset prices at the product of its named degrees,
    and ``auto`` at 2, because the ltx auto rule is ulysses 2 at every canvas
    size and carries no cfg row; both cap at world. The packed audio+video path
    refuses cfg and dp outright. Any other preset, and unreadable state, prices
    at one, which keeps the larger charge.
    """
    try:
        preset = str(getattr(mesh, "topology_preset", "auto") or "auto").lower()
        if preset.startswith(_NON_SEQUENCE_PRESET_PREFIXES):
            return 1
        world = getattr(mesh, "world", None)
        if isinstance(world, bool) or not isinstance(world, int) or world < 1:
            world = getattr(getattr(mesh, "handle", None), "world", 1)
        if isinstance(world, bool) or not isinstance(world, int) or world < 1:
            return 1
        degree = 1
        for token in ("uly", "ring"):
            match = re.search(token + r"(\d+)", preset)
            if match:
                degree *= int(match.group(1))
        if degree <= 1:
            # Under auto, a world above 2 leaves dp behind, which the packed
            # preflight refuses, so the smaller divisor only ever charges more.
            degree = 2 if preset == "auto" else 1
        return max(1, min(degree, world))
    except Exception:
        return 1


_DISABLE_LOGGED = False


def preflight_disabled() -> bool:
    """Return the shared override and log the disabled boundary once."""
    global _DISABLE_LOGGED
    if not mesh_safety.env_enabled(mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV):
        return False
    if not _DISABLE_LOGGED:
        _DISABLE_LOGGED = True
        from .log import get_logger

        get_logger(__name__).warning(
            "ltx activation preflight DISABLED by %s; for this driver process the "
            "loader-site footprint charges no LTX render activations and cannot refuse on them",
            mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV)
    return True


def _guide_clause(rows: LTXRows) -> str:
    """Name the guide charge, and what geometry still escapes this leaf."""
    parts = []
    if rows.guides:
        frames = rows.guide_video // rows.frame_rows if rows.frame_rows else 0
        parts.append(f"{rows.guides} guide(s) add {rows.guide_video} of those "
                     f"video rows over {frames} latent frame(s), counting a "
                     "clip whose source states no length as one frame")
    if rows.unattributed_guides:
        parts.append(f"{rows.unattributed_guides} further guide(s) reach no "
                     "canvas this preflight can identify and are not charged")
    parts.append("latent-upsampler geometry is not visible in the graph")
    return "; ".join(parts)


def graph_activation_bytes(prompt: Any, sp_degree: int, *,
                           family: str) -> tuple[int, str]:
    """Return graph-declared bytes for the loader's settled memory window.

    Both guard overrides apply. This helper is total because propagation would
    discard the complete loader-site footprint refusal.
    """
    try:
        if family != LTX_FAMILY:
            return 0, "not a dual-stream LTX family"
        if preflight_disabled():
            return 0, f"{mesh_safety.ACTIVATION_PREFLIGHT_DISABLE_ENV} is set"
        if driver_footprint.preflight_disabled():
            return 0, f"{driver_footprint.DRIVER_PREFLIGHT_DISABLE_ENV} is set"
        degree = int(sp_degree)
        if degree < 1:
            return 0, "no sequence-parallel degree resolved"
        rows = rows_from_prompt_graph(prompt)
        if not rows.resolved:
            return 0, "no LTX canvas node with readable geometry in this graph"
        return activation_bytes(rows, degree), (
            f"{rows.video} video and {rows.audio} audio rows over {degree} "
            f"rank(s), {rows.video_per_rank(degree)} video rows/rank; a video "
            f"row costs {LTX_SHARDED_VIDEO_ROW_BYTES // 1024} KiB per rank "
            f"plus {LTX_REPLICATED_VIDEO_ROW_BYTES // 1024} KiB that no "
            f"topology divides, over a "
            f"{LTX_RANK_FLOOR_BYTES / 2 ** 30:.1f} GiB per-rank worker floor, "
            f"so the shortfall converts straight into rows; {_guide_clause(rows)}")
    except Exception:
        return 0, "row extraction failed; activations not charged"
