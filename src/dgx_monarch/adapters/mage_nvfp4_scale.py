"""Mage-only NVFP4 activation-scale row scope for exact Ulysses execution.

`call` records how many of this rank's text and image rows are real for one
block call, and `real_input` hands quant_activation_scale only those rows, so
divisibility pads never enter the shared amax. Text projections that stay
dense run on full rows through mage_flow._text_modules instead.
"""
from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any

import torch

_ROWS: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar("mage_nvfp4_rows", default=None)


def _role(path: str) -> int | None:
    """0=text, 1=image; only native double-block projection namespaces."""
    if not path.startswith("transformer_blocks."):
        return None
    if ".attn.add_" in path or ".attn.to_add_out" in path or ".txt_mlp." in path:
        return 0
    if ".attn.to_" in path or ".img_mlp." in path:
        return 1
    return None


def real_input(path: str, value: torch.Tensor) -> torch.Tensor:
    rows = _ROWS.get()
    role = _role(path)
    if rows is None or role is None or value.ndim < 2:
        return value
    count = rows[role]
    # The adapter's shard_seq padding is a stream-local trailing tail. Empty
    # real spans contribute zero to the max reduction, never a padded outlier.
    return value[:, :count] if count > 0 else value[:, :0]


def call(block: Callable[..., Any], *, text_original: int, text_local: int,
         image_original: int, image_local: int, rank: int, **kwargs: Any) -> Any:
    def count(original: int, local: int) -> int:
        return max(0, min(local, original - rank * local))
    token = _ROWS.set((count(text_original, text_local), count(image_original, image_local)))
    try:
        return block(**kwargs)
    finally:
        _ROWS.reset(token)
