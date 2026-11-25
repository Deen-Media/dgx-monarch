"""Carry each cfg conditioning's own text row count past the Worker's pad.

On a pure cfg topology the Worker pads the shorter conditioning of a pair with
zero rows so ComfyUI batches the two (actor/sampling.equalize_cond_lengths),
and a family whose cfg-pad forward restores stock's call cuts those rows off
again on each rank. The cut needs each entry's count from before the pad. A
zero row cannot give it: a zeroed negative (ConditioningZeroOut) is all zero
rows, every one of them real conditioning, and a prompt can end in zero rows
of its own.

So the pad stamps every entry of the padded pair with its count, as a model
cond. ComfyUI keeps a conditioning's own ``model_conds`` beside what the
family's ``extra_conds`` adds (comfy/sampler_helpers.convert_cond,
comfy/samplers.encode_model_conds), repeats and concatenates it with the batch
as it does the text (``CONDRegular``), and hands every model cond to the
diffusion model's forward as a keyword (``BaseModel._apply_model``, which keeps
an integer tensor's dtype); the cfg split slices it with the rest of the batch
(adapters/cfg_parallel.py). The forward pops it and trims to it. A call with no
stamp was not padded and is left as stock sent it.

The trim costs one host read of a short integer tensor per call, in place of
the amax over the whole text tensor that inferring the rows took.

Which forwards read the stamp is explicit: ``stamped(adapter)`` decides
whether the pad stamps a family's pair, and each such family's cfg-pad forward
carries the ``reads_rows`` mark. The cfg split checks the mark on every
stamped call (``assert_read``), so a stamp that would reach a forward that
ignores it refuses, typed, naming the family. The stamp lives only in the
copies the Worker's pad makes of one sample's conditioning; the lists the
request carried are not changed, and nothing the Worker sends back holds it.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import torch

from ..refusal import RefusalClass, refusal
from .base import UnsupportedModelError

# The model-cond key, and so the forward keyword, the stamp travels under.
TEXT_ROWS = "dgxm_text_rows"
# Set on a cfg-pad forward that pops the stamp and cuts the pad to it.
_READS_ROWS = "_dgxm_reads_text_rows"

_F = TypeVar("_F", bound=Callable[..., Any])
# Worker process state, one render at a time: the family whose pair this
# render's pad stamped, else None. actor/sample_protocol.run_sample clears it at
# every render and the pad sets it (``stamps_for``), as
# cfg_dispatch.record_cfg_pair_lengths does for the fold fact.
_stamped_render: str | None = None


def stamped(adapter: Any) -> bool:
    """Whether the cfg pad stamps this family's pair with its row counts."""
    return (getattr(adapter, "cfg_cond_padding", None) == "pad"
            and bool(getattr(adapter, "cfg_pad_restores_stock_call", False)))


def record_render(family: str | None) -> None:
    """Note the family whose pair this render's pad stamped, or clear it (None)."""
    global _stamped_render
    _stamped_render = family


def stamps_for(adapter: Any) -> bool:
    """Whether the pad stamps this pair, noting the render as stamped when it does.

    A zero row cannot say whether the pad added it, since a zeroed negative is
    all zero rows of real conditioning, so a family whose cfg-pad forward cuts
    the pad back off reads each entry's own count from the stamp. Every entry
    of the pair is stamped, padded or not, so the pair keeps one key set and
    still batches (comfy's cond_equal_size).
    """
    if not stamped(adapter):
        return False
    record_render(str(getattr(adapter, "family", "model")))
    return True


def assert_not_lost(text: Any, family: str) -> None:
    """Refuse a call of a stamped render that carries text but no stamp.

    Every text entry of a stamped render carries its count, so a call with text
    and no count lost it between the pad and the forward (a ComfyUI change that
    drops an unknown model cond, say), and its forward would attend the pad as
    real text. A call with no text tensor, and every call of an unstamped
    render, passes. Each rank meets the same call, so every rank refuses
    before any collective.
    """
    if _stamped_render is None or not torch.is_tensor(text):
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family}: this render's cfg pair was padded and each conditioning stamped with "
        "its own row count, but a model call reached the cfg split with text and no count, "
        "so its forward could not cut the pad and would attend it as real text. This is a "
        "dgx-monarch fault or a ComfyUI change, not a workflow setting: report both Workers' "
        "journals. Use a ulysses topology, or topology 'single' (mode=local with "
        "gpus_per_host=1), until it is fixed.",
    ))


def reads_rows(forward: _F) -> _F:
    """Mark a cfg-pad forward as one that reads the stamp and removes it."""
    setattr(forward, _READS_ROWS, True)
    return forward


def assert_read(model: Any, family: str) -> None:
    """Refuse a stamped model call whose forward would not read the stamp.

    ``model`` is the object the cfg split wraps: the diffusion model, or the
    model base around it. Each stamping family binds its reading forward as
    the diffusion model's ``_forward``.
    """
    for target in (model, getattr(model, "diffusion_model", None)):
        bound = getattr(target, "__dict__", {}).get("_forward")
        if getattr(getattr(bound, "__func__", None), _READS_ROWS, False):
            return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family}: this model call carries the cfg pad's row counts, but the forward "
        "that cuts the pad back off is not installed on this model, so the pad rows would "
        "reach it as real text. This is a dgx-monarch fault, not a workflow setting. Use "
        "topology 'single' (mode=local with gpus_per_host=1) until it is fixed.",
    ))


def stamp(extras: dict[str, Any], rows: int) -> dict[str, Any]:
    """A copy of one entry's conditioning dict carrying its pre-pad row count.

    One value serves every batch row of the entry, since its text tensor has
    one length; ComfyUI repeats it to the latent batch as it repeats the text.
    """
    import comfy.conds

    out = dict(extras)
    model_conds = dict(out.get("model_conds") or {})
    model_conds[TEXT_ROWS] = comfy.conds.CONDRegular(torch.tensor([int(rows)], dtype=torch.long))
    out["model_conds"] = model_conds
    return out


def _refuse(family: str, detail: str) -> None:
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family}: the cfg pad's row counts do not match this model call ({detail}). "
        "This is a dgx-monarch fault, not a workflow setting: report the shapes. Use "
        "topology 'single' (mode=local with gpus_per_host=1) until it is fixed.",
    ))


def trim(context: torch.Tensor, rows: torch.Tensor | None, family: str
         ) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Cut the Worker's pad rows off ``context`` by count.

    Returns the context cut to the longest real count in the call, and the
    per-row real counts when the call is ragged (a shorter entry keeps pad rows
    inside the cut), else None. ``rows`` None means the call was not padded:
    the context comes back as it was. The cut is a contiguous copy: stock's
    own call holds a contiguous text tensor, and a GEMM over a strided view of
    several rows rounds apart from one over the same values laid out densely.
    """
    if rows is None:
        return context, None
    counts = rows.reshape(-1).to(torch.long)
    if counts.numel() != context.shape[0]:
        _refuse(family, f"{counts.numel()} counts for {context.shape[0]} batch rows")
    host = counts.tolist()
    longest = max(host)
    if min(host) < 1 or longest > context.shape[1]:
        _refuse(family, f"counts {host} for {context.shape[1]} text rows")
    if longest < context.shape[1]:
        context = context[:, :longest].contiguous()
    if min(host) == longest:
        return context, None
    return context, counts.to(context.device)


def cut_mask(mask: torch.Tensor, rows: int) -> torch.Tensor:
    """A shipped (B, text) mask cut to the text rows the trim kept.

    The pad extended it with zeros over the rows it appended, so the cut
    restores the entry's own mask; a mask that already fits is returned as is.
    """
    return mask if mask.shape[-1] == rows else mask[..., :rows]


def residual_bias(counts: torch.Tensor, rows: int, dtype: torch.dtype) -> torch.Tensor:
    """An additive (B, rows) text-key bias: 0 on real rows, the dtype minimum past them."""
    positions = torch.arange(rows, device=counts.device)
    past = positions.unsqueeze(0) >= counts.unsqueeze(1)
    return past.to(dtype) * torch.finfo(dtype).min


def residual_keep(counts: torch.Tensor, rows: int) -> torch.Tensor:
    """A boolean (B, rows) text keep-mask: True on real rows."""
    positions = torch.arange(rows, device=counts.device)
    return positions.unsqueeze(0) < counts.unsqueeze(1)
