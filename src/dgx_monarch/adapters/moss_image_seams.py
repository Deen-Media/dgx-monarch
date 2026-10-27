"""MOSS-Image sharded-forward seams: what the bound forward checks, routes and cuts.

Stock MOSS-Image (comfy/ldm/moss_image/model.py) runs 42 identical blocks over
one joint ``[image, text]`` stream. The bound ``_forward`` (adapters/moss_image.py)
keeps stock's replicated prelude and runs stock's own ``Block.forward`` code on
one contiguous window of that stream per rank. This module holds every part of
that which can refuse, route or cut: the inject-time tree and ``rope`` checks,
the vetting of ``("single_block", i)`` replacements (an identity function, read
from its bytecode, or a callable marked ``sequence_shard_local``), the window
cut with its pad rows, the fused-norm route that launches both of a block's
norm calls at stock's row count (FLUX.3's ``at_stock_rows``, prefix placement),
the attention ledger and the refusals of reference images and of a text
attention bias. Every raise sits in its own module-level function and goes
through ``refusal()``, class P. Nothing imports ComfyUI at module level.

ComfyUI 60aade0bb (2026-10-22) added reference images for edit LoRAs. Its
``Block.forward`` reaches the fused norm through ``modulate`` and the gated
residual through ``residual``, both module globals, where the earlier form
calls ``modulated_rms_norm`` and ``gated_residual`` itself; without a reference
split the two forms run the same calls. ``rope`` gained the reference shapes
and the f offset, and ``_forward`` the reference arguments. Both trees are
supported: the route follows whichever global the block reads. ComfyUI
9fe7ef764 (2026-10-23) added ``context_attention_bias``, an additive logit
bias on the text keys that the edit node sets from its image_strength;
067f85330 (2026-10-24) frees tensors sooner in ``Block.forward``, which makes
the same calls in the same order.
"""
from __future__ import annotations

import dis
import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NoReturn

import torch

from ..refusal import RefusalClass, refusal
from .attention_inputs import refuse_unadopted_keywords
from .attention_patches import SEQUENCE_SHARD_LOCAL_CAPABILITY, _has_capability
from .base import UnsupportedModelError, _mask_is_noop, padded_row_indices, shard_seq
from .flux3_adaln_parity import at_stock_rows, loads_global, route_global

MOSS_MODEL_MODULE = "comfy.ldm.moss_image.model"
NORM_GLOBAL = "modulated_rms_norm"
# Since ComfyUI 60aade0bb Block.forward calls these two, and modulate calls the norm.
MODULATE_GLOBAL = "modulate"
RESIDUAL_GLOBAL = "residual"
ATTENTION_GLOBAL = "attention"
# The keywords usp_attention applies itself (adapters/base.py); it would drop
# any other one for an ordinary kernel.
ADOPTED_ATTENTION_KEYWORDS = frozenset(
    {"mask", "attn_precision", "skip_reshape", "skip_output_reshape", "enable_gqa"})
HEAD_DIM = 128
_OPTIONS_ROPE = ("h", "w", "text_len", "device", "transformer_options")
_V042_ROPE = ("h", "w", "text_len", "device")
_REFERENCE_ROPE = (*_OPTIONS_ROPE, "ref_shapes", "f_offset")
_SKIPPED_OPS = frozenset({"RESUME", "NOP", "CACHE", "EXTENDED_ARG"})
_VARIADIC = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


def rope_with_options(model, h, w, text_len, device, options):
    """ComfyUI b9b290e43 up to 60aade0bb: the options reach stock's table."""
    return model.rope(h, w, text_len, device, options)


def rope_with_references(model, h, w, text_len, device, options):
    """ComfyUI master since 60aade0bb, called as stock's forward calls it with no reference."""
    return model.rope(h, w, text_len, device, options, [], 0)


def rope_without_options(model, h, w, text_len, device, options):
    """v0.42.0 and v0.42.1, whose rope reads no options; stock ignores them there too."""
    return model.rope(h, w, text_len, device)


def rope_call(diffusion_model: Any) -> Callable[..., torch.Tensor]:
    """The call that builds stock's rotary table, chosen once from ``rope``'s form."""
    try:
        names = tuple(inspect.signature(type(diffusion_model).rope).parameters)[1:]
    except (AttributeError, TypeError, ValueError):
        names = ()
    if names == _REFERENCE_ROPE:
        return rope_with_references
    if names == _OPTIONS_ROPE:
        return rope_with_options
    if names == _V042_ROPE:
        return rope_without_options
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"moss_image: ComfyUI's MOSSImage.rope takes ({', '.join(names)}) after self, and this "
        "release knows only the v0.42 form (h, w, text_len, device), the later form that adds "
        "transformer_options and the form that also adds ref_shapes and f_offset, so it cannot "
        "show that the sharded forward builds stock's rotary table. "
        "Use topology 'single' (mode=local with gpus_per_host=1), or a dgx-monarch release that "
        "supports this ComfyUI (docs/MODELS.md).",
    ))


def assert_stock_tree(diffusion_model: Any) -> None:
    """Refuse a tree that is not ComfyUI's stock MOSS-Image, naming each difference."""
    import comfy.ldm.moss_image.model as moss

    found: list[str] = []
    cls = type(diffusion_model)
    if not isinstance(diffusion_model, moss.MOSSImage):
        found.append(f"the diffusion model is {cls.__name__}, not MOSSImage")
    for name, stock in (("_forward", moss.MOSSImage._forward), ("forward", moss.MOSSImage.forward),
                        ("rope", moss.MOSSImage.rope)):
        if getattr(cls, name, None) is not stock:
            found.append(f"{cls.__name__}.{name} is not MOSSImage.{name}")
    layers = list(getattr(diffusion_model, "layers", None) or ())
    if not layers:
        found.append("it holds no layers")
    strays = [i for i, layer in enumerate(layers) if not isinstance(layer, moss.Block)
              or type(layer).forward is not moss.Block.forward]
    if strays:
        found.append(f"layer {strays[0]} is {type(layers[strays[0]]).__name__}, not stock's Block")
    text = getattr(diffusion_model, "text_embedder", None)
    if not isinstance(text, moss.TextEmbedder) or type(text).forward is not moss.TextEmbedder.forward:
        found.append(f"its text embedder is {type(text).__name__}, not stock's TextEmbedder")
    route = block_norm_route(moss.Block.forward)
    modulate = getattr(moss, MODULATE_GLOBAL, None)
    if route is None:
        found.append(f"Block.forward reads neither the module global {NORM_GLOBAL} nor the pair "
                     f"{MODULATE_GLOBAL} and {RESIDUAL_GLOBAL} alone")
    elif route == MODULATE_GLOBAL and not (callable(modulate) and loads_global(modulate, NORM_GLOBAL)):
        found.append(f"{MODULATE_GLOBAL} no longer reads the module global {NORM_GLOBAL}")
    if not loads_global(moss.Block.forward, ATTENTION_GLOBAL):
        found.append(f"Block.forward no longer reads the module global {ATTENTION_GLOBAL}")
    width = getattr(getattr(diffusion_model, "video_embedder", None), "out_features", None)
    heads = getattr(layers[0], "num_heads", None) if layers else None
    if layers and (not isinstance(heads, int) or width != heads * HEAD_DIM):
        found.append(f"its model width {width} over {heads} heads is not {HEAD_DIM} a head")
    if found:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"moss_image: the loaded model is not ComfyUI's stock MOSS-Image tree ({'; '.join(found)}), "
            "so the sharded forward cannot show that it computes what one GPU computes. Use topology "
            "'single' (mode=local with gpus_per_host=1) for a derivative, or a dgx-monarch release "
            "that supports this ComfyUI (docs/MODELS.md).",
        ))


def is_identity_block_patch(patch: Any) -> bool:
    """Whether ``patch`` is a plain two-argument function that returns its first argument.

    It hands its input back, so the block is the identity on every row and
    exact on any window; ComfyUI's Skip Layer Guidance ``skip`` is one. The
    test names no module: ComfyUI registers node modules under path names.
    """
    if not isinstance(patch, types.FunctionType) or patch.__closure__ is not None:
        return False
    code = patch.__code__
    if code.co_argcount != 2 or code.co_kwonlyargcount or code.co_flags & _VARIADIC:
        return False
    ops = [ins for ins in dis.get_instructions(patch) if ins.opname not in _SKIPPED_OPS]
    return (len(ops) == 2 and ops[0].opname.startswith("LOAD_FAST")
            and ops[0].argval == code.co_varnames[0] and ops[1].opname == "RETURN_VALUE")


def vet_block_replacements(transformer_options: dict, total_blocks: int) -> dict[int, Callable[..., Any]]:
    """The ``("single_block", i)`` replacements stock would call, each exact on a rank's rows.

    One lookup a block, as stock reads the table, so every other key is ignored.
    """
    replace = transformer_options.get("patches_replace", {})
    table = "'dit' replacement table" if isinstance(replace, dict) else "patches_replace option"
    dit = replace.get("dit", {}) if isinstance(replace, dict) else replace
    vetted: dict[int, Callable[..., Any]] = {}
    problem = None
    if not isinstance(dit, dict):
        problem = (f"the {table} is a {type(dit).__name__}, not the dict "
                   "stock reads, so this release cannot tell which blocks it replaces")
    else:
        for index in range(total_blocks):
            key = ("single_block", index)
            if key not in dit:
                continue
            patch = dit[key]
            if is_identity_block_patch(patch) or _has_capability(patch, SEQUENCE_SHARD_LOCAL_CAPABILITY):
                vetted[index] = patch
                continue
            problem = (f"the 'dit' single_block replacement on block {index} receives this rank's rows "
                       "of the joint [image, text] stream and its rotary rows, and it is neither an "
                       f"identity nor marked {SEQUENCE_SHARD_LOCAL_CAPABILITY!r}, so it could compute "
                       "on the wrong rows")
            break
    if problem is not None:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"moss_image: under sequence parallelism {problem}. Run this workflow on cfg2, dp2 or "
            "topology 'single' (mode=local with gpus_per_host=1), where ComfyUI's own forward hands "
            "the patch the full stream, or mark a row-local patch sequence_shard_local (docs/MODELS.md).",
        ))
    return vetted


def refuse_reshaped_block_rows(index: int, args: dict, batch: int, rows: int) -> None:
    """Refuse a marked replacement that hands stock's block other rows than this rank's window."""
    shape = tuple(getattr(args.get("img"), "shape", ()))
    table = tuple(getattr(args.get("pe"), "shape", ()))[1:2]
    if len(shape) == 3 and shape[0] == batch and shape[1] == rows and table == (rows,):
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"moss_image: the sequence_shard_local replacement on block {index} hands stock's block a "
        f"{shape} stream with {table[0] if table else 'no'} rotary rows, where this rank holds a "
        f"[{batch}, {rows}, D] window with {rows} rotary rows, so the block would run on rows the "
        "shard never cut. Run this workflow on dp2 or topology 'single' (mode=local with "
        "gpus_per_host=1), where ComfyUI's own forward hands the patch the full stream.",
    ))


@dataclass(frozen=True)
class JointShard:
    """One rank's contiguous window of the joint stream and its rotary rows."""

    tokens: torch.Tensor   # [B, n, D], contiguous, pads at the global tail
    pe: torch.Tensor       # [1, n, 1, 64, 2, 2], stock's rotary rows for these tokens
    drop_rows: list[int]   # the pad rows in Ulysses' gathered order
    orig: int              # N, the pre-pad length sp_gather restores


def shard_joint(stream: torch.Tensor, pe: torch.Tensor) -> JointShard:
    """Cut the ``[B, N, D]`` stream and its ``[1, N, ...]`` table into this rank's window.

    Both take one zero pad and one chunk, so a rank's table rows are the rows
    stock hands those tokens (a pad row gets a zero rotation). Both are copied
    contiguous: ``shard_seq`` returns a strided chunk at batch 2, stock's gated
    residual writes in place, and comfy-kitchen's fused q/k norm and RoPE takes
    its CUDA path only when the table's rows equal q's.
    """
    local, orig = shard_seq(stream, dim=1, name="moss_image joint-token")
    pe_local, pe_orig = shard_seq(pe, dim=1, name="moss_image joint-position")
    tokens, pe_local = local.contiguous(), pe_local.contiguous()
    assert orig == pe_orig == stream.shape[1], (
        f"moss_image: {orig} stream rows against {pe_orig} rotary rows")
    assert pe_local.shape[1] == tokens.shape[1], (
        f"moss_image: a rank holds {tokens.shape[1]} rows and {pe_local.shape[1]} rotary rows")
    return JointShard(tokens=tokens, pe=pe_local, orig=orig,
                      drop_rows=padded_row_indices([(orig, tokens.shape[1])]))


def block_norm_route(forward: Any) -> str | None:
    """The global through which a ``Block.forward`` reaches its fused norm, or None.

    ``modulated_rms_norm`` before ComfyUI 60aade0bb; ``modulate``, read with
    ``residual`` and no direct norm call, since. Any other mix is unknown.
    """
    reads = {name for name in (NORM_GLOBAL, MODULATE_GLOBAL, RESIDUAL_GLOBAL)
             if loads_global(forward, name)}
    if reads == {NORM_GLOBAL}:
        return NORM_GLOBAL
    if reads == {MODULATE_GLOBAL, RESIDUAL_GLOBAL}:
        return MODULATE_GLOBAL
    return None


def bind_norm_parity(block: Any) -> bool:
    """Bind ``block.forward`` so its two fused-norm calls run at stock's row count.

    With ``stock_norm_rows`` (N), stock's own ``Block.forward`` code runs with
    the global it reaches the norm through (``block_norm_route``) sent through
    ``at_stock_rows``: each ``modulated_rms_norm`` call on the earlier form,
    each ``modulate`` call, stock's own, on the later one. Without it, the
    call is the class forward's own, reference arguments included, as stock's
    ``preprocess_refs`` makes it. A repeat bind replaces the last; the class
    and module never change. Returns whether a route exists.
    """
    stock_forward = type(block).forward
    route = block_norm_route(stock_forward)

    def forward(self, x, mod, pe, transformer_options={}, *references, stock_norm_rows=None,
                **reference_options):
        if stock_norm_rows is None or route is None:
            return stock_forward(self, x, mod, pe, transformer_options, *references,
                                 **reference_options)
        refuse_reference_images(*references, *reference_options.values())
        if not callable(stock_forward.__globals__.get(route)):
            refuse_stray_norm_call(f"its module's {route} global is missing or not callable")
        batch, rows = x.shape[0], x.shape[1]
        calls = [0]

        def check(y, scale, shift):
            calls[0] += 1
            fits = (calls[0] <= 2 and y.ndim == 3 and y.shape[0] == batch and y.shape[1] == rows
                    and tuple(getattr(scale, "shape", ())) == tuple(getattr(shift, "shape", ()))
                    == (batch, 1, y.shape[2]))
            if not fits:
                refuse_stray_norm_call(
                    f"fused-norm call {calls[0]} takes a {tuple(y.shape)} input and a "
                    f"{tuple(getattr(scale, 'shape', ()))} scale, where stock's block makes two "
                    f"calls over its [{batch}, {rows}, D] stream with [{batch}, 1, D] modulation")

        def make(stock):
            def parity(y, norm, scale, shift):
                check(y, scale, shift)
                return at_stock_rows(lambda call: stock(call, norm, scale, shift), y, stock_norm_rows)

            def modulate(y, norm, block_mod, index, ref):
                if ref is not None:
                    refuse_stray_norm_call("a modulate call carries a reference split")
                try:
                    scale, shift = block_mod[index + 1], block_mod[index]
                except (IndexError, KeyError, TypeError):
                    scale = shift = None
                check(y, scale, shift)
                return at_stock_rows(lambda call: stock(call, norm, block_mod, index, None), y,
                                     stock_norm_rows)
            return modulate if route == MODULATE_GLOBAL else parity

        out = route_global(stock_forward, route, make)(self, x, mod, pe, transformer_options)
        if calls[0] != 2:
            refuse_stray_norm_call(f"the block made {calls[0]} fused-norm calls where stock's makes two")
        return out

    block.forward = types.MethodType(forward, block)
    return route is not None


def refuse_stray_norm_call(detail: str) -> NoReturn:
    """A fused-norm call stock's block does not make, at block 0 on every rank alike."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"moss_image: ComfyUI changed MOSS-Image's Block.forward ({detail}). This release launches "
        "each of the block's two fused-norm calls at stock's row count, and it cannot show that a "
        "call it does not know is exact. Use an updated dgx-monarch, or topology 'single' (mode=local "
        "with gpus_per_host=1); canary seam 43 (moss_image block surface) names the change.",
    ))


# The driver preflight (nodes/model_patch_preflight.py), the Worker's cond
# bind (adapters/moss_image.py) and the sharded forward raise the same text,
# so a render reads alike wherever it stops.
REFERENCE_IMAGES_REFUSAL = refusal(
    RefusalClass.PHYSICS,
    "moss_image: this render carries reference images, which ComfyUI's Text Encode MOSS-Image "
    "Edit node (TextEncodeMOSSImageEdit) attaches for an edit LoRA. Stock MOSS-Image appends "
    "their tokens to the joint stream, or their keys and values to every block's attention, so "
    "every row attends to them, and this release has not validated that on any topology but one "
    "GPU: no hardware test has compared a sharded, cfg-parallel, data-parallel or FSDP edit "
    "with stock. "
    "Run this workflow on topology 'single' (mode=local with gpus_per_host=1), which runs "
    "ComfyUI's own forward, or in plain ComfyUI on one GPU instead.",
    troubleshooting=130,
)


def _carries(value: Any) -> bool:
    """Whether a reference argument holds anything: stock's defaults are ``[]`` and None."""
    if value is None:
        return False
    if isinstance(value, list | tuple):
        return len(value) > 0
    return True


def refuse_reference_images(*values: Any) -> None:
    """Refuse reference images on every rank alike, before any collective (#130)."""
    if not any(_carries(value) for value in values):
        return
    raise UnsupportedModelError(REFERENCE_IMAGES_REFUSAL)


# The sharded forward alone raises this: cfg2 and dp2 bind no forward, so
# stock's own applies the bias there.
CONTEXT_ATTENTION_BIAS_REFUSAL = refusal(
    RefusalClass.PHYSICS,
    "moss_image: this render carries an attention bias on its text tokens, which ComfyUI's Text "
    "Encode MOSS-Image Edit node (TextEncodeMOSSImageEdit) attaches when its image_strength is not "
    "1.0. Stock MOSS-Image adds the bias to every query's attention logits for those keys, and the "
    "sharded attention kernel cannot apply an additive key bias, so a sharded render would drop it "
    "and ignore image_strength. "
    "Run this workflow on topology 'single' (mode=local with gpus_per_host=1), which runs "
    "ComfyUI's own forward, or in plain ComfyUI on one GPU instead.",
    troubleshooting=135,
)


def refuse_context_attention_bias(bias: Any) -> None:
    """Refuse an effective text attention bias on every rank alike, before any collective (#135).

    None and an all-zero floating bias pass: stock adds zero to every logit,
    so the forward runs without it. Anything else refuses, so no bias is ever
    dropped unread: a boolean one too, which stock pads with False into a
    mask that hides the image keys, and a value that is not a tensor.
    """
    if bias is None or (isinstance(bias, torch.Tensor) and bias.is_floating_point()
                        and _mask_is_noop(bias)):
        return
    raise UnsupportedModelError(CONTEXT_ATTENTION_BIAS_REFUSAL)


def worker_on_one_gpu() -> bool:
    """Whether this process renders alone: it has no process group, or a group of one.

    A Worker's setup opens its group at the render's world before any load,
    and Fleet KSampler's at one (actor/worker_env.py, mesh_setup.py). Every
    FSDP preset has a degree of two, so a group of one matches the driver's
    rule (nodes/model_patch_preflight.renders_on_one_gpu).
    """
    import torch.distributed as dist

    return not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1


def refuse_control_channels(channels: int, expected: int) -> None:
    """Refuse a latent wider than the checkpoint's, which only a control adapter feeds."""
    if channels == expected:
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"moss_image: this render feeds the model {channels} latent channels where the checkpoint has "
        f"{expected}. A color or depth control adapter concatenates them, and this release has not "
        "validated control under sequence parallelism. Run this workflow in plain ComfyUI on one GPU "
        "instead.",
        troubleshooting=127,
    ))


@dataclass
class AttentionLedger:
    """The block attentions one forward started, and the ones the sharded override ran."""

    started: int = 0
    attended: int = 0

    def guard(self, attn: Any) -> Callable[..., torch.Tensor]:
        """``attn`` behind the keyword check, counting each call that reaches it."""

        def guarded(q, k, v, heads, *, drop_rows=None, **kwargs):
            refuse_unadopted_keywords("moss_image block attention", set(kwargs) - ADOPTED_ATTENTION_KEYWORDS)
            self.attended += 1
            return attn(q, k, v, heads, drop_rows=drop_rows, **kwargs)

        return guarded


def refuse_unsharded_attention(started: int, attended: int) -> None:
    """Refuse a forward in which a block attended outside the sharded override."""
    if started == attended:
        return
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"moss_image: {started} blocks ran and {attended} of them attended through the sharded "
        "override, so at least one block attended over this rank's rows only: this ComfyUI changed "
        "the block's attention call. Use topology 'single' (mode=local with gpus_per_host=1), or a "
        "dgx-monarch release that supports this ComfyUI (docs/MODELS.md).",
    ))
