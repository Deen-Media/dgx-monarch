"""MOSS-Image sharded-forward seams: what the bound forward checks, routes and cuts.

Stock MOSS-Image (comfy/ldm/moss_image/model.py) runs 42 identical blocks over
one joint ``[image, text]`` stream. The bound ``_forward`` (adapters/moss_image.py)
keeps stock's replicated prelude and runs stock's own ``Block.forward`` code on
one contiguous window of that stream per rank. This module holds every part of
that which can refuse, route or cut: the inject-time tree and ``rope`` checks,
the vetting of ``("single_block", i)`` replacements (an identity function, read
from its bytecode, or a callable marked ``sequence_shard_local``), the window
cut with its pad rows, the fused-norm route that launches both of a block's
norm calls at stock's row count (FLUX.3's ``at_stock_rows``, prefix placement)
and the attention ledger. Every raise sits in its own module-level function
and goes through ``refusal()``, class P. Nothing imports ComfyUI at module level.
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
from .base import UnsupportedModelError, padded_row_indices, shard_seq
from .flux3_adaln_parity import at_stock_rows, loads_global, route_global

MOSS_MODEL_MODULE = "comfy.ldm.moss_image.model"
NORM_GLOBAL = "modulated_rms_norm"
ATTENTION_GLOBAL = "attention"
# The keywords usp_attention applies itself (adapters/base.py); it would drop
# any other one for an ordinary kernel.
ADOPTED_ATTENTION_KEYWORDS = frozenset(
    {"mask", "attn_precision", "skip_reshape", "skip_output_reshape", "enable_gqa"})
HEAD_DIM = 128
_MASTER_ROPE = ("h", "w", "text_len", "device", "transformer_options")
_V042_ROPE = ("h", "w", "text_len", "device")
_SKIPPED_OPS = frozenset({"RESUME", "NOP", "CACHE", "EXTENDED_ARG"})
_VARIADIC = inspect.CO_VARARGS | inspect.CO_VARKEYWORDS


def rope_with_options(model, h, w, text_len, device, options):
    """ComfyUI master since b9b290e43: the options reach stock's table."""
    return model.rope(h, w, text_len, device, options)


def rope_without_options(model, h, w, text_len, device, options):
    """v0.42.0 and v0.42.1, whose rope reads no options; stock ignores them there too."""
    return model.rope(h, w, text_len, device)


def rope_call(diffusion_model: Any) -> Callable[..., torch.Tensor]:
    """The call that builds stock's rotary table, chosen once from ``rope``'s form."""
    try:
        names = tuple(inspect.signature(type(diffusion_model).rope).parameters)[1:]
    except (AttributeError, TypeError, ValueError):
        names = ()
    if names == _MASTER_ROPE:
        return rope_with_options
    if names == _V042_ROPE:
        return rope_without_options
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"moss_image: ComfyUI's MOSSImage.rope takes ({', '.join(names)}) after self, and this "
        "release knows only the v0.42 form (h, w, text_len, device) and the later form that adds "
        "transformer_options, so it cannot show that the sharded forward builds stock's rotary table. "
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
    for name in (NORM_GLOBAL, ATTENTION_GLOBAL):
        if not loads_global(moss.Block.forward, name):
            found.append(f"Block.forward no longer reads the module global {name}")
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


def bind_norm_parity(block: Any) -> bool:
    """Bind ``block.forward`` so its two fused-norm calls run at stock's row count.

    With ``stock_norm_rows`` (N), stock's own ``Block.forward`` code runs with
    its ``modulated_rms_norm`` global sent through ``at_stock_rows``; without
    it, the call is the class forward's own. A repeat bind replaces the last;
    the class and module never change. Returns whether the global is read.
    """
    stock_forward = type(block).forward
    routable = loads_global(stock_forward, NORM_GLOBAL)

    def forward(self, x, mod, pe, transformer_options={}, *, stock_norm_rows=None):
        if stock_norm_rows is None or not routable:
            return stock_forward(self, x, mod, pe, transformer_options)
        if not callable(stock_forward.__globals__.get(NORM_GLOBAL)):
            refuse_stray_norm_call(f"its module's {NORM_GLOBAL} global is missing or not callable")
        batch, rows = x.shape[0], x.shape[1]
        calls = [0]

        def make(stock_norm):
            def parity(y, norm, scale, shift):
                calls[0] += 1
                fits = (calls[0] <= 2 and y.ndim == 3 and y.shape[0] == batch and y.shape[1] == rows
                        and tuple(scale.shape) == tuple(shift.shape) == (batch, 1, y.shape[2]))
                if not fits:
                    refuse_stray_norm_call(
                        f"fused-norm call {calls[0]} takes a {tuple(y.shape)} input and a "
                        f"{tuple(getattr(scale, 'shape', ()))} scale, where stock's block makes two "
                        f"calls over its [{batch}, {rows}, D] stream with [{batch}, 1, D] modulation")
                return at_stock_rows(lambda call: stock_norm(call, norm, scale, shift), y, stock_norm_rows)
            return parity

        out = route_global(stock_forward, NORM_GLOBAL, make)(self, x, mod, pe, transformer_options)
        if calls[0] != 2:
            refuse_stray_norm_call(f"the block made {calls[0]} fused-norm calls where stock's makes two")
        return out

    block.forward = types.MethodType(forward, block)
    return routable


def refuse_stray_norm_call(detail: str) -> NoReturn:
    """A fused-norm call stock's block does not make, at block 0 on every rank alike."""
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"moss_image: ComfyUI changed MOSS-Image's Block.forward ({detail}). This release launches "
        "each of the block's two fused-norm calls at stock's row count, and it cannot show that a "
        "call it does not know is exact. Use an updated dgx-monarch, or topology 'single' (mode=local "
        "with gpus_per_host=1); canary seam 43 (moss_image block surface) names the change.",
    ))


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
