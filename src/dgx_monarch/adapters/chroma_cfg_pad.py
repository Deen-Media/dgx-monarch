"""Chroma's cfg2 pad: the trailing text pad and joint-key bias, undone by mask.

Chroma's pad rule is "pad+mask" (actor/sampling.py): the Worker pads the text
and ships an additive bias over the joint keys, and Chroma's cfg-pad forward
(adapters/flux_family.ChromaAdapter.inject_cfg_pad_forward) reads each row's
real length back from that bias. The other cfg2 families read a stamped count
instead (adapters/cfg_pad_rows.py).
"""
from __future__ import annotations

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError


def _chroma_cfg_pad_trim(context: torch.Tensor, attention_mask):
    """Undo the cfg2 equalizer's trailing text pad and joint-key bias.

    `equalize_cond_lengths` (actor/sampling.py, rule "pad+mask") zero-pads
    each prompt's context to one length (the longest, aligned so the joint key
    count is a multiple of 8) and attaches an additive bias over the joint
    [text, image] keys: 0 on real text rows and every image key, the dtype
    minimum on pad rows. Each row's bias is a real-then-pad prefix, so its zero
    count is the row's real length. Trimming context to that length and
    dropping the bias hands stock's forward exactly the unpadded, maskless
    tensors a one-GPU call for that cond builds.

    Pad is a hard sentinel, at or below float16's minimum: ComfyUI casts the
    equalizer's cond-dtype minimum to the model dtype (-inf from float32 in
    bfloat16), and attention gives any such key exactly zero weight, so the
    trim is the same math. A finite bias such as a custom producer's soft
    mask attenuates real tokens and is no pad. Such a mask, a nonzero image
    column, or a text zone that is not zeros then pad is not this construction
    (a conditioning that ships its own attention_mask skips the equalizer,
    sampling.py "already ships its own") and is returned unchanged. A batch
    whose rows disagree on the real length cannot be trimmed and is refused.
    """
    if attention_mask is None:
        return context, None
    text_len = context.shape[1]
    bias = attention_mask.reshape(attention_mask.shape[0], -1)
    if bias.shape[-1] < text_len:
        return context, attention_mask
    text_zone, image_zone = bias[:, :text_len], bias[:, text_len:]
    real = text_zone == 0
    if (image_zone.numel() and bool((image_zone != 0).any())) or not bool(
            (real | (text_zone <= torch.finfo(torch.float16).min)).all()):
        return context, attention_mask
    lengths = real.sum(dim=1)
    is_prefix = real == (
        torch.arange(text_len, device=real.device).unsqueeze(0) < lengths.unsqueeze(1)
    )
    if not bool(is_prefix.all()):
        return context, attention_mask
    rows = int(lengths[0])
    if bool((lengths != rows).any()):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "chroma: this cfg-parallel rank's batch holds more than one real text "
            "length after the trailing pad is removed, so no single trim reproduces "
            "stock's unpadded call for every row. Use one prompt length across the "
            "image batch, or a non-cfg topology.",
        ))
    if rows == text_len:
        return context, None
    return context[:, :rows], None
