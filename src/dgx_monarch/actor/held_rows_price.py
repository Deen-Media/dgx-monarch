"""Price a sample-time load at the rows the forward holds.

ComfyUI prices a model call before it samples. ``sampler_helpers.estimate_memory``
adds the latent's batch times its height and width to the height times width
of every reference latent (``Flux.extra_conds_shapes``) and multiplies by the
model's ``memory_usage_factor`` (``BaseModel.memory_required``).
``load_models_gpu`` then keeps on the device only the weights that free memory
plus what is loaded, less that estimate and the reserve, leaves room for, and
offloads the rest.

With UltraTex token masks that area is not what the forward holds. Stock
``Flux._forward`` (ComfyUI 0.45.0) drops the background tokens of the target
and of each masked reference before the blocks, so the blocks run on the kept
rows only, and the estimate still counts every view, the whole normal atlas
and the whole reference. On 2026-11-11 six views at 2048 a view on FLUX.2
Klein Base 9B were priced at 212,992 latent cells, 81.1 GiB, where the masked
call held 90,597 rows and the one-GPU render of it held about 15 GiB beyond
its weights. On two Sparks with 75 and 81 GiB available each rank offloaded
most of a 17 GiB model and the partial-load guard refused the render class C
(docs/TROUBLESHOOTING.md #79); on one GPU ComfyUI offloaded the whole model
and streamed it a block at a time.

A ``PREPARE_SAMPLING`` wrapper counts the kept tokens with ComfyUI's own mask
rule (``comfy.ldm.flux.model.foreground_tokens``, the rule
``preprocess_token_masks`` applies), and for the length of ComfyUI's own
``prepare_sampling`` call hands its estimate those counts in place of the full
shapes. The factor, the dtype, the batch rule, the reserve and the load
decision stay ComfyUI's. The count is in ComfyUI's unit, latent cells, so a
kept token of a patch-2 model (Flux 1, LongCat) counts four.

ComfyUI's minimum, the figure the load decision reads, is one cond's call: the
latent plus the largest reference count of any cond. Stock can add those
because every cond holds the same latent. With masks the conds differ: a
masked cond holds its kept target and kept references, and a cond with no
masks (a zeroed negative wired from the bare prompt) holds the whole latent
and no reference. So the minimum is priced at the cond whose own call holds
most, never a mix of one cond's latent with another's references. For the
render above that is the unmasked negative's 98,304 cells, 37.4 GiB, and
34.5 GiB where the negative carries the masks too. ComfyUI's other figure,
for cond and uncond in one call, takes the largest target of any cond at
twice the batch and every cond's kept references.

What it never does:

- Price above stock. Each count is capped at stock's own, and the estimate is
  read before and after: a priced figure above stock's drops the correction.
- Guess. A cond with no masks keeps stock's counts, and a batch the view grid
  does not divide, a mask count that is not the grid's, a reference count
  that is not stock's, or any error keeps stock's figure.
- Reach the cond batch rule. ``_calc_cond_batch`` reads the same
  ``memory_required`` during sampling to decide whether cond and uncond share
  a call; the correction is gone before sampling starts, so that decision is
  the one a Worker made before this module.
- Discount a rank's share. Under sequence parallelism a rank runs the per-token
  layers on its window, but the measured renders hold a pair rank at about
  what one GPU holds for the whole stream (``h3_calibration``, the measured
  points), so every rank is priced at the whole kept stream.
- Force a full load. The partial-load guard still reads ComfyUI's own choice,
  and a rank that cannot hold the model at the kept-row price still refuses.

It runs at every world size: a one-GPU Worker that offloads a LoRA-patched
model casts its small patched weights by another recipe than a whole load
bakes them, so one GPU and a pair agree only when both load whole.
"""
from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..log import get_logger
from ..transfer_utils import failure_summary

log = get_logger(__name__)

WRAPPER_KEY = "dgx_monarch_held_rows_price"
_SHADOWED = ("extra_conds_shapes", "memory_required")
_REF_SHAPE_KEY = "ref_latents"
_ABSENT = object()


@dataclass(frozen=True)
class HeldCells:
    """The latent cells one render's calls hold, in ComfyUI's estimate unit.

    Target counts cover the whole latent batch. ``stock`` is what ComfyUI's
    minimum counts: the latent and the largest reference count of any cond.
    ``largest`` is the target and reference cells of the cond whose own call
    holds most, ``target`` the largest target any cond holds, and ``refs``
    maps a masked cond's ``token_masks`` object to its kept and its stock
    reference cells.
    """

    batch: int
    stock: tuple[int, int]
    largest: tuple[int, int]
    target: int
    refs: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def corrects(self) -> bool:
        return (sum(self.largest) < sum(self.stock) or self.target < self.stock[0]
                or any(kept < full for kept, full in self.refs.values()))


def _cond_entries(conds: Any) -> Iterator[Mapping]:
    for entries in conds.values() if isinstance(conds, Mapping) else ():
        for cond in entries if isinstance(entries, list | tuple) else ():
            if isinstance(cond, Mapping):
                yield cond


def _cells(shape: Any) -> int:
    """The cells ``memory_required`` reads from one shape: batch times everything past the channels."""
    return int(shape[0]) * math.prod(int(side) for side in shape[2:])


def _kept_cells(foreground_tokens: Any, mask: Any, height: int, width: int, patch: int) -> int:
    """Cells the kept tokens of ``mask`` cover on a latent of this size.

    The token grid is ``Flux.preprocess_token_masks``'s own rounding, and the
    kept set is ComfyUI's ``foreground_tokens``.
    """
    rows, cols = (height + patch // 2) // patch, (width + patch // 2) // patch
    return int(foreground_tokens(mask, rows, cols).sum().item()) * patch * patch


def _target_cells(foreground_tokens: Any, cond: Mapping, noise_shape: Any, patch: int) -> int | None:
    """Cells the kept target tokens of one cond cover over the whole batch, or None for stock's."""
    masks, grid = cond.get("token_masks"), cond.get("view_grid")
    if not isinstance(masks, Mapping) or not grid or len(noise_shape) != 4:
        return None
    target, views = masks.get("target"), len(grid)
    batch, height, width = int(noise_shape[0]), int(noise_shape[2]), int(noise_shape[3])
    if target is None or batch % views or int(target.shape[0]) != views:
        return None
    kept = (batch // views) * _kept_cells(foreground_tokens, target, height, width, patch)
    return min(kept, batch * height * width)


def _ref_cells(foreground_tokens: Any, cond: Mapping, patch: int) -> tuple[int, int] | None:
    """Kept and full cells of one cond's reference latents, or None where it carries no masks."""
    masks, latents = cond.get("token_masks"), cond.get("reference_latents")
    if not isinstance(masks, Mapping) or latents is None:
        return None
    ref_masks = list(masks.get("refs") or ())
    kept = full_total = 0
    for index, latent in enumerate(latents):
        full = math.prod(latent.shape[2:])
        mask = ref_masks[index] if index < len(ref_masks) else None
        full_total += full
        kept += full if mask is None else min(full, _kept_cells(
            foreground_tokens, mask, int(latent.shape[-2]), int(latent.shape[-1]), patch))
    return kept, full_total


def held_cells(base_model: Any, noise_shape: Any, conds: Any) -> HeldCells | None:
    """What this render's calls hold, or None where stock's estimate already says it.

    Each cond is read as its own call: a masked cond holds its kept target and
    kept references, any other cond stock's latent and stock's count of its
    references (``extra_conds_shapes``, asked of stock itself).
    """
    entries = list(_cond_entries(conds))
    if not any(cond.get("token_masks") is not None for cond in entries):
        return None
    import comfy.ldm.flux.model as flux_model

    foreground_tokens = getattr(flux_model, "foreground_tokens", None)
    patch = getattr(getattr(base_model, "diffusion_model", None), "patch_size", None)
    if foreground_tokens is None or not isinstance(patch, int) or patch < 1 or len(noise_shape) < 3:
        return None
    stock_target = _cells(noise_shape)
    calls: list[tuple[int, int]] = []
    stock_refs = 0
    refs: dict[int, tuple[int, int]] = {}
    for cond in entries:
        shape = base_model.extra_conds_shapes(**cond).get(_REF_SHAPE_KEY)
        full = _cells(shape) if shape is not None else 0
        stock_refs = max(stock_refs, full)
        target = _target_cells(foreground_tokens, cond, noise_shape, patch)
        counted = _ref_cells(foreground_tokens, cond, patch)
        # Only a reference count this module reads as stock does may be replaced.
        kept = counted[0] if counted is not None and counted[1] == full else full
        if kept < full:
            refs[id(cond["token_masks"])] = (kept, full)
        calls.append((stock_target if target is None else target, kept))
    held = HeldCells(
        batch=int(noise_shape[0]), stock=(stock_target, stock_refs), refs=refs,
        largest=max(calls, key=sum), target=max(target for target, _ in calls))
    return held if held.corrects else None


def _shadow(base_model: Any, noise_shape: Any, held: HeldCells) -> dict:
    """Bind the two estimate inputs on this model object; returns what to restore."""
    stock_shapes, stock_required = base_model.extra_conds_shapes, base_model.memory_required
    latent_tail = [int(side) for side in noise_shape[1:]]

    def extra_conds_shapes(**cond):
        out = stock_shapes(**cond)
        cells = held.refs.get(id(cond.get("token_masks")))
        shape = out.get(_REF_SHAPE_KEY) if isinstance(out, Mapping) else None
        # Only the shape this module read stock's count from: [1, channels, cells].
        if cells is None or shape is None or len(shape) != 3 or shape[0] != 1 or shape[2] != cells[1]:
            return out
        return {**out, _REF_SHAPE_KEY: [shape[0], shape[1], cells[0]]}

    def memory_required(input_shape, cond_shapes={}):  # stock's own signature and default
        shape = list(input_shape)
        # Stock passes this render's latent shape, at its batch or twice it.
        if [int(side) for side in shape[1:]] != latent_tail or int(shape[0]) % held.batch:
            return stock_required(input_shape, cond_shapes=cond_shapes)
        calls = int(shape[0]) // held.batch
        if calls != 1:
            return stock_required([calls, shape[1], held.target], cond_shapes=cond_shapes)
        # The minimum: one cond's call, the one that holds most.
        target, refs = held.largest
        shapes = {key: value for key, value in cond_shapes.items() if key != _REF_SHAPE_KEY}
        given: list[Any] = list(cond_shapes.get(_REF_SHAPE_KEY) or ())
        if refs and (len(given) != 1 or len(given[0]) != 3):
            return stock_required(input_shape, cond_shapes=cond_shapes)
        if refs:
            shapes[_REF_SHAPE_KEY] = [[given[0][0], given[0][1], refs]]
        return stock_required([1, shape[1], target], cond_shapes=shapes)

    saved = {name: base_model.__dict__.get(name, _ABSENT) for name in _SHADOWED}
    base_model.extra_conds_shapes = extra_conds_shapes
    base_model.memory_required = memory_required
    return saved


def _restore(base_model: Any, saved: Mapping) -> None:
    for name, value in saved.items():
        if value is _ABSENT:
            base_model.__dict__.pop(name, None)
        else:
            setattr(base_model, name, value)


def _minimum(patcher: Any, noise_shape: Any, conds: Any) -> float:
    import comfy.sampler_helpers as sampler_helpers

    return float(sampler_helpers.estimate_memory(patcher, noise_shape, conds)[1])


def make_wrapper():
    """Build the ``PREPARE_SAMPLING`` wrapper that prices the kept rows."""

    def wrapper(executor, *args, **kwargs):
        patcher = args[0] if args else kwargs.get("model")
        noise_shape = args[1] if len(args) > 1 else kwargs.get("noise_shape")
        conds = args[2] if len(args) > 2 else kwargs.get("conds")
        base_model = getattr(patcher, "model", None)
        saved = None
        try:
            held = held_cells(base_model, noise_shape, conds)
            if held is not None:
                stock = _minimum(patcher, noise_shape, conds)
                saved = _shadow(base_model, noise_shape, held)
                priced = _minimum(patcher, noise_shape, conds)
                if priced > stock:
                    raise ValueError(f"priced {priced:.0f} B above stock's {stock:.0f} B")
                log.info(
                    "held-rows price: ComfyUI's sample-time estimate counts %d latent cells "
                    "for this render (%d target, %d reference); after the token drop the "
                    "largest call a cond makes holds %d (%d target, %d reference); the load "
                    "is priced at %.1f GiB in place of %.1f GiB",
                    sum(held.stock), *held.stock, sum(held.largest), *held.largest,
                    priced / 2**30, stock / 2**30)
        except Exception as exc:
            if saved is not None:
                _restore(base_model, saved)
                saved = None
            log.warning("held-rows price: left at ComfyUI's own estimate (%s)", failure_summary(exc))
        try:
            return executor(*args, **kwargs)
        finally:
            if saved is not None:
                _restore(base_model, saved)

    return wrapper


def install(base_patcher: Any, family: str | None) -> None:
    """Register the wrapper on a freshly loaded base of a family whose forward drops tokens.

    Every world size, one GPU included. Read the key before writing it: comfy
    appends on a second add under one key.
    """
    from ..adapters.flux_views import VIEW_FAMILIES

    if family not in VIEW_FAMILIES:
        return
    import comfy.patcher_extension as pe

    options = base_patcher.model_options
    if pe.get_wrappers_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, options, is_model_options=True
    ):
        return
    pe.add_wrapper_with_key(
        pe.WrappersMP.PREPARE_SAMPLING, WRAPPER_KEY, make_wrapper(),
        options, is_model_options=True,
    )
