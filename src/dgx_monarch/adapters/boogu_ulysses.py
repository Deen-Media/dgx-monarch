"""Boogu's pure-Ulysses exactness treatments.

A Boogu double block makes two attention calls on one ``transformer_options``:
the joint ``[instruct, image]`` attention, then the image self-attention. For
the first each rank holds ``[instruct_r, image_r]``, so Ulysses' head
all-to-all gathers ``[instruct0, image0, instruct1, image1]`` where stock
attends ``[instruct0, instruct1, image0, image1]``, and flash rounding depends
on key order. The second gathers ``[image0, image1]``, which is stock's order
already, and a joint descriptor there would refuse. One block-scoped override
cannot carry both, so the override built here picks each call's own options by
its local query length: the joint call holds ``instruct_local + image_local``
rows and the image call ``image_local``, and the instruct stream always holds
at least one row, so the two never coincide.

The same split carries the divisibility pad rows ``shard_seq`` adds. After
modulation a zero row is a real key, so each call names its own pad set at
Ulysses' full-sequence point: the joint call both streams' pads in rank-major
coordinates, which the descriptor remaps, and the image call the image pads
alone. Handing the joint set to the image call would drop a real image row,
because a joint coordinate can fall inside the image axis.
"""
from __future__ import annotations

from typing import Any

import torch

from ..refusal import RefusalClass, refusal
from .base import (
    USP_ATTENTION_OVERRIDE_ATTR,
    UnsupportedModelError,
    padded_row_indices,
    usp_options,
)
from .quant_activation_scale import ACTIVATION_SCALE_RULES
from .usp_sequence_order import RankMajorJointOrder

_OVERRIDE = "optimized_attention_override"


def _query_rows(args, kwargs) -> int:
    """Local query length: ``(B, H, L, D)`` under ``skip_reshape``, else ``(B, L, H*D)``."""
    query = args[0]
    return int(query.shape[2] if kwargs.get("skip_reshape", False) else query.shape[1])


def double_block_options(transformer_options: dict, attention, text: tuple[int, int],
                         image: tuple[int, int], *, pure_ulysses: bool) -> dict:
    """Options for the double-stream loop.

    ``text`` and ``image`` are each stream's ``(original, local)`` row counts,
    the image stream counting reference tokens. Without pure Ulysses this is
    the plain override with no order and no pad rows, so ring and hybrid keep
    their path.
    """
    if not pure_ulysses:
        return usp_options(transformer_options, attention)
    text_local, image_local = text[1], image[1]
    joint = usp_options(transformer_options, attention, padded_row_indices([text, image]),
                        RankMajorJointOrder(text_local, image_local))
    alone = usp_options(transformer_options, attention, padded_row_indices([image]))
    calls = {text_local + image_local: joint[_OVERRIDE], image_local: alone[_OVERRIDE]}

    def override(func, *args, **kwargs):
        rows = _query_rows(args, kwargs)
        call = calls.get(rows)
        if call is None:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"boogu double block: an attention call carried {rows} local query "
                f"rows, which is neither the joint instruct and image pair "
                f"({text_local + image_local}) nor the image stream ({image_local}), "
                "so its key order and pad rows are unknown. This is a dgx-monarch "
                "fault, not a workflow setting: report the shapes. Use topology "
                "'single' (mode=local with gpus_per_host=1) until it is fixed.",
            ))
        return call(func, *args, **kwargs)

    setattr(override, USP_ATTENTION_OVERRIDE_ATTR, True)
    joint[_OVERRIDE] = override
    return joint


def stream_options(transformer_options: dict, attention, stream: tuple[int, int], *,
                   pure_ulysses: bool) -> dict:
    """Options for a loop over one sharded stream, ``(original, local)`` rows.

    Under pure Ulysses its tail pad rows are named so attention drops them.
    Elsewhere none are named: the ring and hybrid guard refuses a named pad
    row, and those topologies attend their pads.
    """
    drop_rows = padded_row_indices([stream]) if pure_ulysses else None
    return usp_options(transformer_options, attention, drop_rows)


def _row_independent(module) -> bool:
    """Whether a full-row call quantizes this Linear's input as stock does.

    A Linear is wrapped when it is a ``torch.nn.Linear`` (dense, or comfy's
    ``fp8_ops``) or a comfy mixed-precision Linear whose layout
    quant_activation_scale rates "constant" (the fp8 layouts), because comfy
    quantizes such an input with the checkpoint's static ``input_scale`` or
    the constant 1.0, never an amax, so the row count cannot change the
    quantization. NVFP4, MXFP8, the int8 family and any layout that table does
    not name keep the shard. Layers loaded with ``full_precision_matrix_mult``
    keep their fp8 layout name and run a plain GEMM on the dequantized weight.
    """
    if isinstance(module, torch.nn.Linear):
        return True
    layout = getattr(module, "layout_type", None)
    return isinstance(layout, str) and ACTIVATION_SCALE_RULES.get(layout) == "constant"


def instruct_projections(block) -> tuple[tuple[Any, bool], ...]:
    """A double block's instruct-stream Linears, for chroma_text's full-row helper.

    Stock runs them on every instruct row; a rank runs only its shard, and the
    BLAS can pick another kernel for another row count. ``instruct_out`` reads a slice
    of the joint attention output, so it takes the joint batch stride. The
    shipped fp8_scaled file loads every one through comfy's mixed-precision
    Linear with an fp8 layout and no ``input_scale``, so ``_row_independent``
    admits them.
    """
    processor: Any = getattr(getattr(block, "img_instruct_attn", None), "processor", None)
    feed_forward: Any = getattr(block, "instruct_feed_forward", None)
    candidates = (
        (getattr(processor, "instruct_to_q", None), False),
        (getattr(processor, "instruct_to_k", None), False),
        (getattr(processor, "instruct_to_v", None), False),
        (getattr(processor, "instruct_out", None), True),
        (getattr(feed_forward, "linear_1", None), False),
        (getattr(feed_forward, "linear_3", None), False),
        (getattr(feed_forward, "linear_2", None), False),
    )
    return tuple((module, joint_stride) for module, joint_stride in candidates
                 if module is not None and _row_independent(module))
