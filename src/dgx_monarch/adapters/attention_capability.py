"""Select and probe kernels for a family's declared attention head dimension.

Ideogram4 uses head dimension 256, beyond the tabled cuDNN and Sage limits.
Substitute flash when it supports the declared shape, matching ComfyUI's
unsharded fallback. Refuse with class P when no tabled kernel supports the
head or the selected implementation's live probe fails without a substitute.
A declared head dimension is required so the runtime can report its selection
before rendering.

Never substitute TORCH_MATH for Ring: yunchang's math and cuDNN branches return
zero log-sum-exp values, which cannot correctly merge partial softmax states
above Ring degree 1. ``sol_attention_guards`` applies the same class-P rule to
unsupported geometry. See docs/TROUBLESHOOTING.md #98 and docs/VALIDATION.md.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

log = get_logger(__name__)

# The largest head dimension each kernel carries, and where the number is
# written down. cuDNN: the frontend inside libtorch_cuda.so, "hidden_dim d_qk
# should be less than or equal to 128 ... unless d_qk == 192 and d_v == 128"
# (cudnn_frontend sdpa_support_surface.h); torch declines before it. Sage:
# sageattention/core.py raises "Unsupported head_dim" above 128. Flash: torch's
# gate, "FlashAttention forward only supports head dimension at most 256". A
# kernel absent here makes no static claim.
KERNEL_HEAD_DIM_LIMIT: Final[Mapping[str, int]] = MappingProxyType({
    "TORCH_CUDNN": 128,
    "SAGE_AUTO": 128,
    "SAGE_FP16": 128,
    "SAGE_FP8": 128,
    "TORCH_FLASH": 256,
})

# The substitutes, in order, for a kernel that cannot carry a head dimension.
# One entry: flash is the only tabled kernel wider than 128, and it is
# what ComfyUI's own unsharded fallback lands on for this shape.
SUBSTITUTE_ORDER: Final = ("TORCH_FLASH",)

# The probe's geometry: a sequence length that is a multiple of 64, so no
# kernel declines it for alignment and the head dimension is what is under
# test. A decline for another reason still refuses, since the probe runs the
# real call rather than re-deriving its preconditions.
PROBE_BATCH: Final = 1
PROBE_HEADS: Final = 2
PROBE_SEQUENCE: Final = 128

_RECORDS: dict[tuple[str, str], str] = {}


def capability_records() -> Mapping[tuple[str, str], str]:
    """What this worker measured, per family and kernel. The answer depends on
    the hardware, so no setup generation clears it."""
    return MappingProxyType(dict(_RECORDS))


def kernels_carrying(head_dim: int) -> tuple[str, ...]:
    """The tabled kernels whose published limit admits this head dimension."""
    return tuple(sorted(name for name, limit in KERNEL_HEAD_DIM_LIMIT.items()
                        if head_dim <= limit))


def _kernels_that_carry(head_dim: int) -> str:
    """The kernel names for the refusal text. The cfg-or-single sentence stays
    at each raise site as a literal: the ledger walker reads class P
    messages statically."""
    admitted = kernels_carrying(head_dim)
    if not admitted:
        return "No kernel in this build's table carries that head dimension."
    return (f"Kernels that carry head dimension {head_dim}: "
            f"{', '.join(admitted)}. Pick one with the Init node's attention input.")


def effective_kernel(requested: str, family: str, head_dim: int | None) -> str:
    """The kernel that will run, after the substitution.

    Pure and table driven, so the dispatcher and the sweep matrix read one
    answer. Unchanged unless the table says the selection cannot carry this
    family's declared head dimension and a substitute can.
    """
    if head_dim is None:
        return requested
    limit = KERNEL_HEAD_DIM_LIMIT.get(requested)
    if limit is None or head_dim <= limit:
        return requested
    for name in SUBSTITUTE_ORDER:
        if head_dim <= KERNEL_HEAD_DIM_LIMIT.get(name, 0):
            return name
    return requested


def substitution_note(requested: str, family: str, head_dim: int | None) -> str:
    """One sentence naming a substitution, or empty when none happened."""
    effective = effective_kernel(requested, family, head_dim)
    if effective == requested:
        return ""
    return (f"{effective} substituted for {requested}: {family} attends at head "
            f"dimension {head_dim}, and {requested} carries at most "
            f"{KERNEL_HEAD_DIM_LIMIT[requested]}")


def assert_static_capability(kernel: str, family: str, head_dim: int | None) -> None:
    """Refuse a kernel whose published limit excludes this family.

    Takes the kernel that will run, so any substitution has already happened
    and this is the arm where nothing carries the head. It touches no device,
    so the sweep matrix can call it.
    """
    if head_dim is None:
        return
    limit = KERNEL_HEAD_DIM_LIMIT.get(kernel)
    if limit is None or head_dim <= limit:
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{kernel} carries head dimensions up to {limit}, and {family} has "
        f"{head_dim}, so the sharded attention this topology installs would "
        "abort inside the ring instead of rendering. "
        f"{_kernels_that_carry(head_dim)} Run this workflow "
        "on a cfg or single topology instead (for a batch of one, single needs "
        "mode=local with gpus_per_host=1): neither installs the sharded "
        "attention.",
        troubleshooting=98,
    ))


def _compute_dtype(model) -> torch.dtype:
    """The dtype this model attends in, which is never its stored fp8 weights."""
    for value in (getattr(model, "manual_cast_dtype", None),
                  getattr(getattr(model, "diffusion_model", None), "dtype", None)):
        if (isinstance(value, torch.dtype) and value.is_floating_point
                and value.itemsize >= 2):
            return value
    return torch.bfloat16


def _ring_forward(kernel: str):
    """The exact function the ring path selects for this kernel, and no other."""
    from yunchang.kernels import AttnType, select_flash_attn_impl

    return select_flash_attn_impl(AttnType[kernel], stage="fwd-only")


def _run_probe(probe, head_dim: int, dtype: torch.dtype, device) -> None:
    """One forward with the keywords xFuser's ring forward passes, verbatim."""
    shape = (PROBE_BATCH, PROBE_SEQUENCE, PROBE_HEADS, head_dim)
    tensors = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(3)]
    probe(*tensors, dropout_p=0.0, softmax_scale=None, causal=False,
          window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
          return_softmax=False)


def assert_kernel_capability(kernel: str, family: str, head_dim: int | None,
                             model=None, *, probe=None, device=None) -> None:
    """Prove the kernel runs this family's shape here, before any collective.

    Takes the kernel that will run, so the substitution has already happened.
    Rank replicated by construction: every input comes from the declared head
    dimension, the kernel name and the model's compute dtype, which every rank
    shares. It enters no process group, so it cannot desynchronize a rank.
    """
    from .sol_attention import is_sol_kernel

    if is_sol_kernel(kernel):
        # The sol kernels own this question already, geometry and family both.
        return
    assert_static_capability(kernel, family, head_dim)
    if head_dim is None:
        return
    if probe is None:
        if not torch.cuda.is_available():
            return
        probe, device = _ring_forward(kernel), device or torch.device("cuda")
    dtype = _compute_dtype(model)
    try:
        _run_probe(probe, head_dim, dtype, device)
    except torch.cuda.OutOfMemoryError:
        # A transient shortfall is not a statement about the kernel.
        raise
    except Exception as exc:
        _RECORDS[(family, kernel)] = f"declined: {type(exc).__name__}"
        log.warning(
            "attention capability: %s declined %s at head dim %d, dtype %s, "
            "shape (%d, %d, %d, %d), mask none: %s",
            kernel, family, head_dim, dtype, PROBE_BATCH, PROBE_SEQUENCE,
            PROBE_HEADS, head_dim, exc)
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"{kernel} declined a probe of this render's attention on this "
            f"hardware. The probe for {family} ran {PROBE_HEADS} heads over "
            f"{PROBE_SEQUENCE} tokens at head dimension {head_dim} in {dtype} "
            f"with no mask, and the kernel raised {type(exc).__name__}. "
            "The sharded attention would have aborted inside the ring after "
            f"the model loaded. {_kernels_that_carry(head_dim)} Run this "
            "workflow on a cfg or single topology instead (for a batch of one, "
            "single needs mode=local with gpus_per_host=1): neither installs "
            "the sharded attention.",
            troubleshooting=98,
        )) from exc
    _RECORDS[(family, kernel)] = "admitted"
    log.info("attention capability: %s admits %s at head dim %d, dtype %s",
             kernel, family, head_dim, dtype)
