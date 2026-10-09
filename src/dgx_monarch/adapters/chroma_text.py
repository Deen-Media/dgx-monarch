"""Pure-Ulysses full-row text projections for dual-stream blocks."""
from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

import torch

from .base import pad_seq_to_multiple, sp_gather, sp_rank, sp_world


def _modules(block) -> tuple[tuple[torch.nn.Linear, bool], ...]:
    """The two text projections whose GEMM depends on row count.

    Selection is ``isinstance(module, torch.nn.Linear)``. ComfyUI's
    ``disable_weight_init``, ``manual_cast``, ``fp8_ops`` and ``cublas_ops``
    Linear classes all subclass ``torch.nn.Linear``, so unquantized weights and
    plain fp8 weights without quantization metadata are selected. ComfyUI's
    ``mixed_precision_ops`` Linear subclasses ``torch.nn.Module`` instead, so
    checkpoints with quantization metadata (scaled fp8, int8, ConvRot, NVFP4,
    MXFP8) are skipped. The "nvfp4 activation scale" seam in
    tests/canary/comfy_seam_contracts.py pins that base class.
    """
    attention = getattr(getattr(block, "txt_attn", None), "proj", None)
    mlp: Any = getattr(block, "txt_mlp", ())
    tail = mlp[2] if len(mlp) > 2 else None
    candidates = ((attention, True), (tail, False))
    return tuple((module, joint_stride) for module, joint_stride in candidates
                 if isinstance(module, torch.nn.Linear))


@contextlib.contextmanager
def full_row_text_projections(
    block, text_rows: int, image_rows: int, *, pure_ulysses: bool,
    select: Callable[[Any], tuple[tuple[torch.nn.Linear, bool], ...]] | None = None,
    gather: Callable[..., torch.Tensor] | None = None,
):
    """Run a block's text projections at full text length under pure Ulysses.

    `select` names another family's projections as `(module, joint_stride)`
    pairs: Boogu, Hunyuan, Lens, Mage Flow and Qwen-Image pass their own, and
    Chroma takes the default. `gather` stands in for `sp_gather` with the same
    call; Mage Flow passes one that gathers an input its projections share
    only once.

    Each output is re-sharded exactly as the text stream was. A projection
    selected with ``joint_stride`` receives stock's joint-output view stride,
    whose batch row span includes the image stream; the others receive the
    contiguous gathered text tensor. Only the modules the selector returns are
    patched (`_modules` says which Chroma's admits); Mage Flow's also admits
    comfy's mixed-precision Linear when its weight is unquantized.

    Restore removes an instance ``forward`` the module did not own before,
    unless the block already removed it, so a missing attribute never masks
    the block's own exception.
    """
    if not pure_ulysses:
        yield
        return
    targets = (select or _modules)(block)
    if not targets:
        yield
        return

    sentinel = object()
    patched: list[tuple[torch.nn.Linear, object]] = []
    forward_attr = "forward"

    def wrapped(joint_stride: bool, original):
        def forward(x):
            gathered = (sp_gather if gather is None else gather)(x, text_rows, dim=1)
            linear_input = gathered.contiguous()
            if joint_stride:
                batch, rows, hidden = gathered.shape
                # Stock slices text from the joint [text, image] attention output.
                linear_input = torch.empty_strided(
                    (batch, rows, hidden), ((rows + image_rows) * hidden, hidden, 1),
                    dtype=gathered.dtype, device=gathered.device)
                linear_input.copy_(gathered)
            output = original(linear_input)
            padded, _ = pad_seq_to_multiple(output, sp_world(), dim=1)
            return torch.chunk(padded, sp_world(), dim=1)[sp_rank()].contiguous()
        return forward

    try:
        for module, joint_stride in targets:
            own_forward = vars(module).get(forward_attr, sentinel)
            original = module.forward
            setattr(module, forward_attr, wrapped(joint_stride, original))
            patched.append((module, own_forward))
        yield
    finally:
        for restored_module, own_forward in reversed(patched):
            if own_forward is not sentinel:
                setattr(restored_module, forward_attr, own_forward)
            elif forward_attr in vars(restored_module):
                delattr(restored_module, forward_attr)
