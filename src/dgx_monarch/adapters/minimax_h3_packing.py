"""Per-token MiniMax H3 and MiniMax-H3-Image packing, timestep embeddings and the head's grids.

The adapter owns sharding, pad exclusion, and topology refusals. This module
uses ComfyUI only to move the curve table onto the compute device and to read
the installed output head's signature.
"""
from __future__ import annotations

import inspect
from typing import Any

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

# Modality tag per packed segment kind, mirroring ComfyUI's own seg_tag map.
# Its keys are the segment kinds this adapter has vetted for sharding: a kind
# outside it refuses rather than reaching a stream.
H3_SEG_MODALITY: dict[str, int] = {
    "text": 1, "video": 0, "audio": 2,
    "cond": 0, "ref_img": 0, "cond_audio": 2, "ref_audio": 2,
}


# The name of the ``FinalLayer.forward`` parameter after ``sample_sigmas``.
# ComfyUI ede4e5f (v0.45.0) made it one decreasing sigma grid per output
# stream; 0.44.x took the two flow shifts there and built each grid inside the
# head.
HEAD_GRIDS = "grids"
HEAD_SHIFTS = "shifts"
H3_COMFY_FLOOR = "0.45.0"
HEAD_FORM_TROUBLESHOOTING = 147


def head_blocks(final_layer: Any) -> int:
    """The PDD head blocks the output bank holds; 1 for a plain head.

    Read from the live weight, as stock reads ``pdd_steps``.
    """
    return int(final_layer.video_out.weight.shape[0]) // int(final_layer.video_dim)


def head_form(final_layer: Any, family: str = "minimax_h3", model: str = "MiniMax H3") -> str:
    """The installed head's form, read once at inject, or a class P refusal.

    ``grids`` is the head of ComfyUI 0.45.0 or newer. ``shifts`` is the older
    head's: a one-block head never reads that slot, so the shifts render what v1.4.0 rendered,
    bit for bit; a stacked head there, or a name this build does not know,
    refuses here, before any collective, instead of failing inside the head.
    """
    values = list(inspect.signature(type(final_layer).forward).parameters.values())
    positional = [p for p in values if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                                  inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    name = positional[7].name if len(positional) > 7 else None
    # The call passes the receiver and seven arguments by position and no keyword:
    # it fits when eight positional slots exist (or *args takes the rest) and no
    # parameter beyond them, positional or keyword-only, lacks a default.
    takes_eight = len(positional) >= 8 or any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in values)
    required_after = [p for p in positional[8:] if p.default is inspect.Parameter.empty] + [
        p for p in values
        if p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty]
    if not takes_eight or required_after:
        name = None
    if name == HEAD_GRIDS or (name == HEAD_SHIFTS and head_blocks(final_layer) == 1):
        return name
    if name == HEAD_SHIFTS:
        what = (f"this checkpoint's output head stacks {head_blocks(final_layer)} PDD blocks, "
                "and this Worker's ComfyUI is older than 0.45.0")
    elif name is None:
        what = ("this Worker's ComfyUI has a MiniMax H3 output head whose arguments do not "
                "match the call this build makes")
    else:
        what = (f"this Worker's ComfyUI has a MiniMax H3 output head that takes {name!r} "
                f"where this build passes {HEAD_GRIDS!r}")
    # Class P: a collective render's artifact_identity preflight (mesh_rpc.py) holds every
    # rank to the driver's ComfyUI commit first, so every rank reads this head alike. A
    # ComfyUI that is not a git checkout reports "unknown" there and is not held.
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family}: {what}. This build drives {model} "
        f"on ComfyUI {H3_COMFY_FLOOR} or newer, where the head takes one sigma grid "
        "per stream. Update ComfyUI on every host to "
        f"{H3_COMFY_FLOOR} or newer, then reset the Attached mesh; or take a true "
        "single-GPU reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
        troubleshooting=HEAD_FORM_TROUBLESHOOTING,
    ))


def pdd_grid(steps: int, flow_shift: float, time_shift_sigma: Any) -> list[float]:
    """Stock's ``pdd_grid`` (ComfyUI 0.45.0): the uniform base grid through a flow shift.

    Built from the declared ``time_shift_sigma`` with stock's own arithmetic, so
    the adapter imports no name 0.44.x lacks; the pdd head seam holds it equal
    to stock's grid.
    """
    return [time_shift_sigma(k / steps, 1.0, flow_shift) for k in range(steps, -1, -1)]


def head_schedule(form: str, final_layer: Any, shifts: tuple[float, ...],
                  time_shift_sigma: Any, grid: Any = None) -> tuple:
    """``FinalLayer.forward``'s last argument for the installed head form.

    ``shifts`` holds one flow shift per output stream, video first. ``grid`` is
    a checkpoint's own explicit sigma grid, the one grid of a model without an
    audio stream, which the head takes as given. No caller passes ``grid`` yet;
    the MiniMax-H3-Image adapter will, for its ``pdd_sigmas``.
    """
    if form == HEAD_SHIFTS:
        return tuple(shifts)
    if grid is not None:
        return ([float(sigma) for sigma in grid],)
    steps = head_blocks(final_layer)
    return tuple(pdd_grid(steps, shift, time_shift_sigma) for shift in shifts)


# The segment kinds a MiniMax-H3-Image layout packs: the text, the reference
# pictures and the target, in that order (ComfyUI's PackedLayout with no audio).
H3_IMAGE_SEGMENT_KINDS: frozenset[str] = frozenset({"text", "ref_img", "video"})


def h3_image_modulation_plan(
    layout: Any,
    payload: dict[str, Any],
    t: float,
    visual_cond_t: float,
) -> tuple[list[float], torch.Tensor, int]:
    """MiniMax-H3-Image's per-token modulation rows, as its stock ``_forward`` numbers them.

    Text and the target take ``t``. A reference picture takes ``max(t,
    visual_cond_t)`` with ComfyUI's constant, never the payload's noise
    augmentation, and no audio time joins the set, so the row numbers equal
    stock's ``mod_segments``. Returns the unique times, the int64 row per token
    (it shards with the stream, as H3's does) and the target's row.
    """
    seg_time = {"text": t, "video": t, "ref_img": max(t, visual_cond_t)}
    present = {kind for _, _, kind in layout.segments}
    unknown = sorted(present - H3_IMAGE_SEGMENT_KINDS)
    if unknown:
        # The layout is built from the request on every rank, so this refuses
        # every rank alike before the first collective.
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"minimax_h3_image: the packed layout carries segment kind(s) {unknown}, which "
            "this adapter has not vetted. MiniMax-H3-Image packs text, reference pictures and "
            "the target only, and stock ComfyUI's image forward has no timestep for any other "
            "kind, so it fails on this conditioning too and no topology renders it. Use "
            "MiniMax H3 Image Conditioning to build the references, or report the workflow "
            "(docs/MODELS.md).",
        ))
    unique_t = sorted({seg_time[kind] for kind in present})
    t_row = {value: index for index, value in enumerate(unique_t)}
    tags = payload.get("text_token_tags")
    if tags is not None:
        tags = tags.reshape(-1).to(device="cpu", dtype=torch.long)
    rows = torch.zeros(layout.seq_len, dtype=torch.long)
    for start, stop, kind in layout.segments:
        base_row = t_row[seg_time[kind]] * 3
        if kind == "text" and tags is not None:
            rows[start:stop] = tags[: stop - start] + base_row
        else:
            rows[start:stop] = base_row + H3_SEG_MODALITY[kind]
    return unique_t, rows, t_row[t]


def assert_image_grid(final_layer: Any, pdd_sigmas: Any) -> None:
    """Refuse, class P, a stacked head whose block count the grid does not match.

    ComfyUI's PDD head splits its weight into one block per grid interval and
    fails inside the head when the two disagree; a one-block head never reads
    the grid, as in stock.
    """
    blocks, points = head_blocks(final_layer), len(pdd_sigmas)
    if blocks == 1 or points == blocks + 1:
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"minimax_h3_image: this checkpoint's output head has {blocks} blocks and its "
        f"pdd_sigmas grid has {points} points. ComfyUI pairs each head block with one "
        "interval of that grid, so the grid must hold one point more than the head has "
        "blocks. Stock ComfyUI fails inside the head on this file too, so no topology "
        "renders it. Use the file as Comfy-Org/MiniMax-H3-Image publishes it, or report the "
        "checkpoint (docs/MODELS.md).",
    ))


def _build_packed_layout(packed_layout_cls, payload, text_len, latent_t,
                         lat_h, lat_w, audio_t):
    """Build the stock PackedLayout across the frame_count seam.

    Comfy e01fb4c56 removed PackedLayout's frame_count parameter (the new
    guide anchoring replaced it). Older comfy revisions still take it and use
    it for last-frame keyframes, so it is passed only where the installed
    class accepts it.
    """
    extra = {}
    init = inspect.signature(packed_layout_cls.__init__)
    if "frame_count" in init.parameters:
        extra["frame_count"] = payload.get("frame_count")
    return packed_layout_cls(text_len, latent_t, lat_h, lat_w, audio_t,
                             keyframes=payload.get("keyframes"),
                             refs=payload.get("refs"),
                             **extra)


def h3_timestep_embedding(model: Any, t_vals: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Embed timesteps for curve or MLP checkpoints.

    Curve interpolation remains fp32 for fp32 adaLN linears. MLP results use
    the compute dtype. Unknown checkpoint regimes fail closed.
    """
    if getattr(model, "use_adaln_curves", False):
        table = getattr(model, "adaln_t_table", None)
        if table is None:
            raise UnsupportedModelError(
                "minimax_h3: this checkpoint declares the adaLN curve regime but carries no "
                "'adaln_t_table' buffer, so there is no timestep embedding to read. Run it on a "
                "true single-GPU reference (mode=local, gpus_per_host=1) and report the "
                "checkpoint (docs/MODELS.md)."
            )
        import comfy.model_management

        table = comfy.model_management.cast_to(table, device=t_vals.device)
        # Preserve the trained curve's clamp, lower row, and fractional blend.
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
    embedder = getattr(model, "time_embedder", None)
    if embedder is None:
        raise UnsupportedModelError(
            "minimax_h3: this checkpoint carries neither an 'adaln_t_table' buffer nor a "
            "'time_embedder' module, so its timestep-embedding regime is unknown. Run it on a "
            "true single-GPU reference (mode=local, gpus_per_host=1) and report the checkpoint "
            "(docs/MODELS.md)."
        )
    return embedder(t_vals).to(dtype)


def h3_modulation_plan(
    layout: Any,
    payload: dict[str, Any],
    t_v: float,
    t_a: float,
    visual_cond_default: float,
    audio_cond_default: float,
) -> tuple[list[float], torch.Tensor, int, int]:
    """Return unique times, per-token modulation rows, and final-layer rows.

    The int64 row vector shards with hidden states; ``row_runs`` reconstructs
    local runs without rank-offset arithmetic. It stays on CPU because runs are
    read as Python integers. Final-layer rows omit ``* 3`` because that adaLN
    projection has one modality.
    """
    vis_aug = float(payload.get("visual_cond_noise_aug", visual_cond_default))
    aud_aug = float(payload.get("audio_cond_noise_aug", audio_cond_default))
    # Each DiT adaLN projection has 3*M rows: video, text, and audio for each
    # timestep. Condition rows use their augmentation time. An anchored guide
    # emits "cond" for its frames and "cond_audio" for its soundtrack, both
    # carrying the same times and tags their reference-block twins carry.
    seg_time = {"text": t_v, "video": t_v, "audio": t_a,
                "cond": max(t_v, vis_aug), "ref_img": max(t_v, vis_aug),
                "cond_audio": max(t_a, aud_aug), "ref_audio": max(t_a, aud_aug)}
    seg_modality = H3_SEG_MODALITY
    present = {kind for _, _, kind in layout.segments}
    unknown = sorted(present - seg_time.keys())
    if unknown:
        # Vet segment kinds before adapter assembly indexes them. The layout is
        # built from the request on every rank, so this decision is rank
        # symmetric and the tag retires the sample lease consumed: act on the
        # refusal and re-queue on the same fleet.
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"minimax_h3: the packed layout carries segment kind(s) {unknown}, which this "
            "adapter has not vetted for sharding. Report the workflow, or take a true "
            "single-GPU reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
        ))
    unique_t = sorted({t_v, t_a} | {seg_time[kind] for kind in present})
    t_row = {value: index for index, value in enumerate(unique_t)}

    tags = payload.get("text_token_tags")
    if tags is not None:
        tags = tags.reshape(-1).to(device="cpu", dtype=torch.long)
    rows = torch.zeros(layout.seq_len, dtype=torch.long)
    for start, stop, kind in layout.segments:
        base_row = t_row[seg_time[kind]] * 3
        if kind == "text" and tags is not None:
            # Presentation prompts mark vision-pad positions with the video tag.
            rows[start:stop] = tags[: stop - start] + base_row
        else:
            rows[start:stop] = base_row + seg_modality[kind]
    return unique_t, rows, t_row[seg_time["video"]], t_row[seg_time["audio"]]


def refuse_indivisible_dit_heads(diffusion_model: Any, family: str) -> None:
    """Refuse, class P, a Ulysses degree the DiT's own head count does not divide.

    ``auto``'s world-4 fold names 56 heads from the table
    (``topology.VALIDATED_ULYSSES_FOLDS``) and never reads the file, and the
    all-to-all's reshape would fail untyped on a count the degree does not
    divide. So the check reads the live first block's head count before any
    bind. A model with no readable count (a test double) is left alone.
    """
    from .flux3_packing import refuse_indivisible_heads, ulysses_degree

    block = next(iter(getattr(diffusion_model, "blocks", None) or ()), None)
    heads = getattr(getattr(block, "attn", None), "heads", None)
    if isinstance(heads, int) and not isinstance(heads, bool):
        refuse_indivisible_heads(heads, ulysses_degree(), family=family,
                                 divisors="2, 4 or 8 for every shipped H3 file's 56 heads")


def pad_row_index(rows: torch.Tensor, padded_len: int) -> torch.Tensor:
    """Extend row indices across divisibility padding.

    Pads repeat the last real row, which comfy's packing order makes the
    target-video row, so contiguous runs survive for the modulation helpers.
    """
    seq_len = int(rows.shape[0])
    if padded_len <= seq_len:
        return rows
    padded = torch.zeros(padded_len, dtype=torch.long)
    padded[:seq_len] = rows
    if seq_len:
        padded[seq_len:] = rows[seq_len - 1]
    return padded


def splice_condition_rows(
    target_rows: torch.Tensor,
    cond_rows: torch.Tensor | None,
    update_mask: torch.Tensor,
    device: Any,
) -> torch.Tensor:
    """Interleave sampler targets and pinned conditions by ``update_mask``.

    False positions consume condition rows in layout order. The masked writes
    are disjoint.
    """
    if cond_rows is None:
        return target_rows
    merged = torch.empty(int(update_mask.shape[0]), int(target_rows.shape[1]),
                         dtype=torch.float32, device=device)
    merged[update_mask] = target_rows
    merged[~update_mask] = cond_rows
    return merged


def row_runs(rows: torch.Tensor) -> list[tuple[int, int, int]]:
    """Encode local row indices as ``(start, stop, row)`` runs.

    Merging equal adjacent runs is elementwise identical because modulation
    scales, shifts, and gates each row independently.
    """
    n = int(rows.shape[0])
    if n == 0:
        return []
    edges = (rows[1:] != rows[:-1]).nonzero().flatten().add(1).tolist()
    bounds = [0, *[int(e) for e in edges], n]
    return [(bounds[i], bounds[i + 1], int(rows[bounds[i]])) for i in range(len(bounds) - 1)]
