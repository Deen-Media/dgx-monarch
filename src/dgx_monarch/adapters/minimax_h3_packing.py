"""Per-token MiniMax H3 packing and checkpoint timestep embeddings.

The adapter owns sharding, pad exclusion, and topology refusals. This module
uses ComfyUI only to move the curve table onto the compute device.
"""
from __future__ import annotations

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
