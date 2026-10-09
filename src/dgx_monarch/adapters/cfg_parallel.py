"""Split a batched CFG model call across ranks and gather its predictions.

ComfyUI batches concatenable conditionings when its memory check admits them.
The keyed wrapper gives each CFG rank a contiguous ``batch // world`` slice,
runs the inner forward, and all-gathers the result. The slice unit supports
latent batches larger than one.

Each family declares the wrapper interface its forward actually invokes.
Boogu, Ernie, and OmniGen2 do not construct a DIFFUSION_MODEL executor, so
they declare ``cfg_split_seam = "apply_model"`` and use the executor in
``BaseModel.apply_model`` instead.

When conditionings require separate calls, ``cfg_dispatch`` distributes them
at CALC_COND_BATCH. Cases it cannot classify retain this wrapper's typed
refusal. ``actor.sampling`` can also pad opted-in families to make their
conditionings concatenate. See DESIGN.md §5.4.
"""
from __future__ import annotations

import inspect

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .attention_patches import assert_cfg_attention_output_patches_safe
from .base import Adapter, UnsupportedModelError, cfg_rank, cfg_world

log = get_logger(__name__)

# Shared metadata: execution structure or patch callables, never sliced, even
# when one holds a tensor whose leading size happens to equal the model batch
# (a four-step sigma schedule at batch four). Every other entry may hold
# per-sample tensor trees and is localized recursively below.
_STATIC_TRANSFORMER_OPTIONS = frozenset({
    "activations_shape",
    "block",
    "block_index",
    "block_type",
    "callbacks",
    "grid_sizes",
    "img_slice",
    "optimized_attention_override",
    "original_shape",
    "patches",
    "patches_replace",
    "prefetch_dynamic_vbars",
    "rope_options",
    "sample_sigmas",
    "sigmas",
    "total_blocks",
    "transformer_index",
    "wrappers",
})


def _leading_batch(value) -> int | None:
    """Batch size of the first tensor found inside a forward argument."""
    queue = [value]
    while queue:
        item = queue.pop(0)
        if torch.is_tensor(item):
            return int(item.shape[0]) if item.ndim > 0 else None
        if isinstance(item, dict):
            queue.extend(item.values())
        elif isinstance(item, (list, tuple)):
            queue.extend(item)
    return None


def _take_slice(value, batch: int, world: int, rank: int):
    """This rank's contiguous rows of every tensor whose dim 0 equals the full
    call batch. Broadcast tensors (dim 0 of 1 or the per-rank size) and
    non-tensor leaves pass through untouched; containers recurse (controlnet
    ships its per-block tensors inside dicts/lists).
    """
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch:
            rows = batch // world
            return value.narrow(0, rank * rows, rows)
        return value
    if isinstance(value, dict):
        return {key: _take_slice(item, batch, world, rank) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        sliced = [_take_slice(item, batch, world, rank) for item in value]
        return tuple(sliced) if isinstance(value, tuple) else sliced
    return value


def _condition_chunk_slice(batch: int, world: int, rank: int, chunks: int) -> slice:
    """Map this rank's tensor-row interval to whole condition chunks."""
    if batch % chunks != 0:
        raise RuntimeError(
            f"cfg-parallel got {chunks} condition chunks for batch {batch}; "
            "the model-call batch must divide evenly into condition chunks"
        )
    rows_per_rank = batch // world
    rows_per_chunk = batch // chunks
    row_start = rank * rows_per_rank
    row_end = row_start + rows_per_rank
    if row_start % rows_per_chunk or row_end % rows_per_chunk:
        raise RuntimeError(
            "cfg-parallel rank slicing would split a cond_or_uncond chunk; "
            f"batch={batch}, chunks={chunks}, world={world}"
        )
    return slice(row_start // rows_per_chunk, row_end // rows_per_chunk)


def _local_condition_metadata(value: dict, batch: int, world: int, rank: int) -> dict:
    """Slice Comfy's parallel condition labels and UUIDs with one boundary.

    Both entries hold one item per condition chunk, and a chunk can span
    several tensor rows. Deriving one slice for both also makes a mismatched
    pair fail instead of pairing a local branch with the wrong cache UUID.
    """
    metadata = {
        key: value[key]
        for key in ("cond_or_uncond", "uuids")
        if isinstance(value.get(key), (list, tuple)) and value[key]
    }
    if not metadata:
        return {}

    chunk_counts = {len(item) for item in metadata.values()}
    if len(chunk_counts) != 1:
        raise RuntimeError(
            "cfg-parallel requires cond_or_uncond and uuids to describe the "
            f"same condition chunks; lengths={sorted(chunk_counts)}"
        )

    chunk_slice = _condition_chunk_slice(batch, world, rank, chunk_counts.pop())
    return {key: item[chunk_slice] for key, item in metadata.items()}


def _slice_transformer_options(value, batch: int, world: int, rank: int):
    """Copy transformer metadata and localize its per-batch leaves.

    Tensor trees (custom per-sample patch metadata included) follow the batch
    slicing rule of ordinary forward arguments; callable patch hooks, strings,
    scalars and other static leaves keep their identity. ``cond_or_uncond`` and
    ``uuids`` hold one entry per condition chunk, not per tensor row
    (`_local_condition_metadata`).
    """
    if not isinstance(value, dict):
        return _take_slice(value, batch, world, rank)
    localized = {
        key: (
            item if key in _STATIC_TRANSFORMER_OPTIONS
            else _take_slice(item, batch, world, rank)
        )
        for key, item in value.items()
    }
    localized.update(_local_condition_metadata(value, batch, world, rank))
    return localized


def _slice_forward_argument(name: str, value, batch: int, world: int, rank: int):
    if name == "transformer_options":
        return _slice_transformer_options(value, batch, world, rank)
    return _take_slice(value, batch, world, rank)


def assert_cfg_parallel_supported(
    adapter: Adapter, cfg_degree: int, dual_model_cfg: bool = False
) -> None:
    """Refuse a family without a combined cond/uncond call before mutation.

    A family may carry its own ``cfg_parallel_refusal_reason`` when the
    exclusion is a hardware measurement rather than the separate-models
    architecture (PixelDiT: cfg2 measured over the fidelity floor).

    ``dual_model_cfg`` grants the split-guider route (dm-cfg2): a family that
    runs cond and uncond as separate models has no batched call to split, but
    the split guider runs one model per rank and exchanges the predictions, so
    the batched-call requirement does not apply.
    """
    if dual_model_cfg:
        return
    if cfg_degree <= 1 or getattr(adapter, "cfg_parallel_supported", True):
        return
    family = getattr(adapter, "family", "model")
    reason = getattr(adapter, "cfg_parallel_refusal_reason", None) or (
        f"{family} guidance runs the conditional and unconditional models "
        "as separate calls, so cfg-parallel has no combined batch to split. "
        "Choose topology 'auto', or an explicit ring topology whose product "
        "matches the worker world."
    )
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{reason} Use mode=local with gpus_per_host=1 for a stock one-GPU path.",
    ))


def refuse_pixel_space_pad_mask(family: str) -> None:
    """Pixel-space Chroma cannot map the joint pad mask onto a latent grid.

    Stock mask upscaling factorizes the joint [text, image] bias onto a latent
    grid; a pixel-space token layout has none, so the forward dies inside
    einops instead of rendering. Refuse before dispatch with the remedy.
    """
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{family} pixel-space cfg-parallel cannot equalize unequal prompt "
        "lengths: the joint attention mask's spatial factorization assumes a "
        "latent grid this pixel-space model does not have, and the render "
        "dies inside stock mask upscaling (measured 2026-08-21). Use "
        "equal-length prompts, a Conditioning Zero Out negative (the shipped "
        "template's wiring), or a non-cfg topology.",
    ))


def assert_sp_quant_supported(adapter: Adapter, sp: int, quant_kind: str, *,
                              cfg: int = 1, dual_model_cfg: bool = False) -> None:
    """Refuse a quantization a family has measured wrong under sharding.

    ``sp_validated_quants`` is opt-in: families without it keep the default
    (quantization orthogonal to topology, docs/MODELS.md), and a declaring
    family carries its basis in ``sp_quant_refusal_reason``.

    cfg-parallel shards the batch rather than the token axis, which splits an
    activation statistic as sequence parallelism does (nvfp4,
    docs/VALIDATION.md), so a declaring family is asked about cfg2
    too. dm-cfg2 is not sharded: each rank runs a full batch call for its own
    checkpoint.
    """
    validated = getattr(adapter, "sp_validated_quants", None)
    sharded = sp > 1 or (cfg > 1 and not dual_model_cfg)
    if not sharded or validated is None or quant_kind in validated:
        return
    family = getattr(adapter, "family", "model")
    reason = getattr(adapter, "sp_quant_refusal_reason", None) or (
        f"{family} has no sharded-fidelity evidence for quant={quant_kind}; "
        f"validated: {sorted(validated)}.")
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"{reason} Use mode=local with gpus_per_host=1 for a stock one-GPU path.",
    ))


# The two seams a family may declare. Comfy applies both itself; only the
# DIFFUSION_MODEL one depends on the family's own forward building an executor.
CFG_SPLIT_SEAMS = ("diffusion_model", "apply_model")


def cfg_split_seam_type(adapter: Adapter) -> str:
    """The comfy wrapper type this family's cfg-parallel split installs on.

    Each member is read by name so the touchpoint manifest carries it.
    """
    import comfy.patcher_extension as pe

    seam = getattr(adapter, "cfg_split_seam", "diffusion_model")
    if seam == "diffusion_model":
        return pe.WrappersMP.DIFFUSION_MODEL
    if seam == "apply_model":
        return pe.WrappersMP.APPLY_MODEL
    raise RuntimeError(
        f"{getattr(adapter, 'family', 'model')} declares cfg_split_seam={seam!r}, "
        f"which is not one of {list(CFG_SPLIT_SEAMS)}"
    )


def install_cfg_parallel_wrapper(adapter: Adapter, model_options: dict) -> str:
    """Install the split once, on the seam this family's forward reaches.

    Read before write: ``add_wrapper_with_key`` appends under one key, so a
    reload that ran this twice would slice twice.
    """
    import comfy.patcher_extension as pe

    from ..constants import CFG_WRAPPER_KEY

    wrapper_type = cfg_split_seam_type(adapter)
    if not pe.get_wrappers_with_key(
            wrapper_type, CFG_WRAPPER_KEY, model_options, is_model_options=True):
        pe.add_wrapper_with_key(
            wrapper_type, CFG_WRAPPER_KEY, make_cfg_parallel_wrapper(adapter),
            model_options, is_model_options=True)
    return wrapper_type


def cfg_dispatches_per_cond(adapter: Adapter) -> bool:
    """Whether two prompts of different length always take the per-cond
    dispatch. A declared ``cfg_batch_constant`` is a real caption length comfy
    compares by value, so a differing pair never folds; for other families
    comfy's concat rule decides."""
    return (getattr(adapter, "cfg_batch_constant", None) is not None
            and getattr(adapter, "cfg_parallel_supported", True))


def _all_gather_cfg(local):
    """Reassemble the full call from every cfg rank's slice.

    NCCL rejects non-contiguous inputs and several family forwards end in a
    crop view, so the slice is made contiguous before all_gather_into_tensor.
    """
    from xfuser.core.distributed import get_cfg_group

    return get_cfg_group().all_gather(local.contiguous(), dim=0)


def make_cfg_parallel_wrapper(adapter: Adapter):
    """Build the keyed cfg-split wrapper for `adapter`'s family."""
    family = getattr(adapter, "family", "model")
    assert_cfg_parallel_supported(adapter, 2)

    def cfg_parallel_wrapper(executor, *args, **kwargs):
        from .cfg_dispatch import dispatch_in_progress

        world = cfg_world()
        # A rank inside its own per-cond call owns every row of it.
        if world <= 1 or dispatch_in_progress():
            return executor(*args, **kwargs)
        rank = cfg_rank()

        # executor.original is the bound comfy forward; binding the call to
        # its signature lets the slice rule see every argument by name no
        # matter how comfy passed it.
        bound = inspect.signature(executor.original).bind_partial(*args, **kwargs)

        batch = _leading_batch(bound.arguments.get("x"))
        if batch is None or batch < world or batch % world != 0:
            # Class P, so the driver retires the sample lease CONSUMED: the
            # check reads x's leading batch and cfg_world() ahead of this
            # call's slice and all-gather, every earlier call's all-gather has
            # returned, and no rank packs a latent until the sampler returns.
            # x's batch matches across ranks unless comfy's free-memory check
            # splits a concatenable pair on one rank only (docs/VALIDATION.md).
            #
            # A batch of one comes from a CFG 1.0 sampler that drops the
            # unconditional pass, from that memory split, or from a pair that
            # cfg_dispatch.conds_fold leaves on this path unread (a default
            # cond, an area, mask, gligen patch, timestep window, hook group,
            # different control objects, or conds it cannot read) when comfy
            # runs it as separate calls.
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"cfg-parallel got a model call of batch={batch}, which does not split into "
                f"{world} equal slices. ComfyUI batches cond+uncond into one call only when it "
                "runs both passes and the two conditionings share a cross-attention shape. It "
                "runs both above CFG 1.0, and at CFG 1.0 only under a cfg++ sampler, whose name "
                "carries cfg_pp and which keeps the unconditional pass; every other sampler "
                "drops that pass at CFG 1.0 and hands this wrapper one row. A blank or much "
                "longer negative prompt forces separate calls that cfg-parallel cannot split. "
                "On a pure cfg topology dgx-monarch equalizes the two prompt lengths "
                "on every family that declares a padding rule; the adapter's "
                "cfg_cond_padding says which. Otherwise raise CFG above 1.0 or "
                "pick a cfg_pp sampler, keep the prompts comparable in length, or use a "
                "ulysses or ring topology instead.",
            ))

        transformer_options = bound.arguments.get("transformer_options")
        variadic_kwargs = bound.arguments.get("kwargs")
        if transformer_options is None and isinstance(variadic_kwargs, dict):
            transformer_options = variadic_kwargs.get("transformer_options")
        if isinstance(transformer_options, dict):
            assert_cfg_attention_output_patches_safe(transformer_options, family)

        for name, value in list(bound.arguments.items()):
            if name == "kwargs" and isinstance(value, dict):
                bound.arguments[name] = {
                    key: _slice_forward_argument(key, item, batch, world, rank)
                    for key, item in value.items()
                }
            else:
                bound.arguments[name] = _slice_forward_argument(
                    name, value, batch, world, rank)

        return _all_gather_cfg(executor(*bound.args, **bound.kwargs))

    return cfg_parallel_wrapper
