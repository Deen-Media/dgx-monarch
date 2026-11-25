"""Wan 2.1 and 2.2 video DiT sequence-parallel adapter.

Tokens shard after embedding. RoPE frequencies and per-token modulation take
the same shard. Attention uses the configured USP kernel, and gathering precedes
the output head. A stream the degree does not divide is padded with zero rows,
which the block loop names: pure Ulysses leaves them out of every softmax
exactly, and ring or hybrid, which cannot, refuses class K by default
(``refuse_padded_ring``, the ring_pad waiver) naming frame counts that divide.
The gather trims them.
"""
from __future__ import annotations

import contextlib
import math
from collections.abc import Callable
from typing import Any

import torch

from .. import accuracy_waiver
from ..log import get_logger
from . import base, scale_rows
from .base import (
    RING_PAD_GUARD,
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    assert_ulysses_only_padding,
    padded_row_indices,
    shard_seq,
    sp_gather,
)
from .stock_attention import stock_attention_options

log = get_logger(__name__)


def _ring_degree() -> int:
    """The initialized ring degree, one where xfuser answers nothing."""
    try:
        from xfuser.core.distributed import get_ring_parallel_world_size

        return int(get_ring_parallel_world_size())
    except Exception:
        return 1


def dividing_frames(rows_at: Callable[[int], int], frames: int, world: int) -> list[int]:
    """The nearest video frame counts, one below and one above, whose stream divides ``world``.

    ``frames`` is the video's own latent frame count and ``rows_at(f)`` the
    whole sharded stream's length at ``f`` latent frames, so rows that grow with
    the video (pose rows, a context clip of the video's length) and rows that do
    not (reference frames, reference images) both count. Wan's VAE keeps one
    latent frame for the first video frame and one for each four after it. The
    current count is never named, since it is the one that pads.
    """
    below = next((f for f in range(frames - 1, 0, -1) if rows_at(f) % world == 0), None)
    above = next((f for f in range(frames + 1, frames + world + 1)
                  if rows_at(f) % world == 0), None)
    return [4 * (f - 1) + 1 for f in (below, above) if f is not None]


def stream_rows_at(rows: int, frames: int, per_frame: int) -> Callable[[int], int]:
    """``rows_at`` for a stream of ``rows`` at ``frames`` latent frames that gains
    ``per_frame`` rows with each frame (``dividing_frames``)."""
    return lambda f: rows + (f - frames) * per_frame


def refuse_padded_ring(family: str, rows: int, world: int, pad_rows: int, heads: int,
                       rows_at: Callable[[int], int] | None = None,
                       frames: int | None = None) -> None:
    """Refuse, class K by default, pad rows on ring or hybrid, naming what avoids them.

    Ring attention has no full-sequence point where the pad rows could be left
    out, so the generic refusal at the attention would fire anyway; this one
    runs before the block loop. It offers pure Ulysses at the same world where
    the ``heads`` divide it, and the nearest video frame counts whose stream
    divides (``dividing_frames``) where the family can count its rows by frame.
    The ring_pad waiver lets the render attend them, stamped.
    """
    ring = _ring_degree()
    if ring <= 1 or not pad_rows or accuracy_waiver.waived(RING_PAD_GUARD):
        return
    counts = (dividing_frames(rows_at, frames, world)
              if rows_at is not None and frames is not None and frames > 1 else [])
    options = []
    if heads and heads % world == 0:
        options.append(f"Use uly{world}, which leaves the pad rows out exactly")
    if counts:
        options.append("render " + " or ".join(f"{count} frames" for count in counts)
                       + ", where the patched count divides")
    options.append("render on topology 'single'")
    assert_ulysses_only_padding(ring, pad_rows, remedy=(
        f"{family}: {rows} patched tokens do not divide the {world} ranks. "
        + "; or ".join(options)))


@contextlib.contextmanager
def named_pads(pads: list, orig_len: int, local_len: int, family: str, heads: int,
               frames: int | None = None, per_frame: int = 0):
    """Name one sharded stream's tail pad rows for a Wan variant's block loop.

    The zero rows an odd stream took to divide the shard get zero RoPE, so
    each is a zero key, and logit 0 still takes softmax weight. Named on every
    topology, pure Ulysses drops them and ring or hybrid refuses them before
    the loop (class K, ring_pad waiver), naming the video frame counts whose
    stream divides where ``frames`` (the video's own latent frames) and
    ``per_frame`` (the rows each frame adds) say how the stream grows
    (``refuse_padded_ring``). The same rows stay out of the shared nvfp4
    activation scale, as in plain Wan. The outer value comes back after, so a
    forward re-entered from a patch leaves this one's rows named.
    """
    saved = pads[0]
    pads[0] = padded_row_indices([(orig_len, local_len)]) or None
    if pads[0]:
        refuse_padded_ring(
            family, orig_len, base.sp_world(), len(pads[0]), heads,
            None if frames is None else stream_rows_at(orig_len, frames, per_frame), frames)
    try:
        with scale_rows.real_rows(WAN_SCALE_ROLES,
                                  {"x": scale_rows.stream((orig_len, local_len))}):
            yield
    finally:
        pads[0] = saved


@contextlib.contextmanager
def video_frames(cell: list, frames: int):
    """Hold the video's own latent frame count in ``cell`` for one forward.

    Stock ``_forward`` concatenates reference frames (``time_dim_concat``) onto
    the time axis before the block forward runs, so the count is read where
    ``_forward`` receives the latent. The previous value comes back afterwards,
    so a forward re-entered from a patch leaves the outer one's count intact.
    """
    saved = cell[0]
    cell[0] = int(frames)
    try:
        yield
    finally:
        cell[0] = saved

# The Wan block Linears that see the sharded token stream: self-attention,
# the cross-attention query and output, and the feed-forward. The
# cross-attention keys and values read the replicated text and image context,
# and WanDancer's audio injector reads the replicated audio embedding.
WAN_SCALE_ROLES = scale_rows.by_pattern([
    (r"blocks\.\d+\.(self_attn\.(q|k|v|o)|cross_attn\.(q|o)|ffn\.\d+)", "x"),
])


# Only these types share the plain t2v/i2v forward. Other Wan variants add
# streams or conditioning that require dedicated adapters. FlowRVS changes
# model-base sampling, not this diffusion forward.
_WAN_SUPPORTED_EXACT = ("WAN21", "WAN22", "WAN21_FlowRVS")


class WanAdapter(Adapter):
    family = "wan"
    model_base_classes = ("WAN21",)  # WAN22 subclasses WAN21
    exact_model_base_classes = _WAN_SUPPORTED_EXACT
    # UMT5 pads every prompt to at least 512 tokens, so two prompts under that length
    # batch; a pair that does not fold runs one conditioning per cfg rank (cfg_dispatch.py).
    cfg_cond_padding = "none"

    def attention_dispatch(self, dispatch):
        return dispatch.for_wan() if type(self) is WanAdapter else dispatch

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in self.exact_model_base_classes:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"wan variant {type(base_model).__name__} is not in the dgx-monarch launch set, "
            "which holds plain Wan 2.1 and 2.2 (t2v, i2v) and FlowRVS only; other Wan "
            "variants need their own adapter. Render it in stock ComfyUI on one GPU "
            "(docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn: Any = ctx.usp_attention
        # This forward's pad rows, set around its block loop and read by the
        # rebound self-attention (one cell per bound model), and the video's own
        # latent frame count, read where _forward receives the latent.
        pads: list[list[int] | None] = [None]
        frames: list[int | None] = [None]
        # A model whose class has no _forward of its own names no frame count.
        stock_forward: Any = getattr(type(diffusion_model), "_forward", None)  # class fn

        def frames_forward(self, x, *args, **kwargs):
            with video_frames(frames, x.shape[2]):
                return stock_forward(self, x, *args, **kwargs)

        def usp_self_attention(self, x, freqs, transformer_options={}):
            # q/k use already-sharded RoPE; the USP call owns the collective.
            from comfy.ldm.flux.math import apply_rope1

            b, s = x.shape[:2]
            n, d = self.num_heads, self.head_dim
            q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
            k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
            v = self.v(x).view(b, s, n * d)
            out = attn(q.view(b, s, n * d), k.view(b, s, n * d), v, heads=n,
                       drop_rows=pads[0])

            patches = transformer_options.get("patches", {})
            for p in patches.get("attn1_patch", []):
                out = p({"x": out, "q": q, "k": k, "transformer_options": transformer_options})
            return self.o(out)

        def usp_forward_orig(self, x, t, context, clip_fea=None, freqs=None,
                             transformer_options={}, **kwargs):
            # Cross-attention uses local q with replicated context k/v. Only x,
            # freqs, and per-token e0 shard; the head requires a gathered stream.
            # The blocks keep stock cross-attention, which must not run a
            # block's pinned kernel (adapters/stock_attention.py).
            from comfy.ldm.wan.model import sinusoidal_embedding_1d

            transformer_options = stock_attention_options(transformer_options)

            x = self.patch_embedding(x.float()).to(x.dtype)
            grid_sizes = x.shape[2:]
            transformer_options["grid_sizes"] = grid_sizes
            x = x.flatten(2).transpose(1, 2)

            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
            e = e.reshape(t.shape[0], -1, e.shape[-1])
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            ref_len = 0
            if self.ref_conv is not None:
                ref = kwargs.get("reference_latent", None)
                if ref is not None:
                    ref = self.ref_conv(ref).flatten(2).transpose(1, 2)
                    ref_len = ref.shape[1]
                    x = torch.concat((ref, x), dim=1)

            main_len = x.shape[1]
            # The rows each further video frame would add: the video's own, and
            # a context clip's that runs the video's length (a reference image
            # or the ref_conv rows stay fixed).
            per_frame = int(grid_sizes[1]) * int(grid_sizes[2])
            context_latents = kwargs.get("context_latents", None)
            if context_latents is not None:
                for lat in context_latents:
                    cl = self.patch_embedding(lat.float().to(x.device)).to(x.dtype).flatten(2).transpose(1, 2)
                    x = torch.cat([x, cl], dim=1)
                    if frames[0] is not None and int(lat.shape[-3]) == frames[0]:
                        per_frame += cl.shape[1] // frames[0]

            # Expand per-frame timesteps to tokens before sharding. Otherwise
            # each rank would stretch the full frame schedule over its local chunk.
            if e0.shape[1] > 1:
                tokens_per_e = math.ceil(x.shape[1] / e0.shape[1])
                e0 = torch.repeat_interleave(e0, tokens_per_e, dim=1)[:, :x.shape[1]]
                e0, _ = shard_seq(e0, dim=1)

            x, orig_len = shard_seq(x, dim=1)
            freqs, _ = shard_seq(freqs, dim=1)

            context = self.text_embedding(context)
            context_img_len = None
            if clip_fea is not None:
                if self.img_emb is not None:
                    context = torch.concat([self.img_emb(clip_fea), context], dim=1)
                context_img_len = clip_fea.shape[-2]

            blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
            transformer_options["total_blocks"] = len(self.blocks)
            transformer_options["block_type"] = "double"
            # The zero rows an odd stream took to divide the shard get zero RoPE,
            # so each is a zero key: logit 0 still takes softmax weight. Named on
            # every topology, pure Ulysses drops them and ring or hybrid refuses
            # them (class K, ring_pad waiver). The same rows stay out of the
            # shared nvfp4 activation scale. The outer value comes back after,
            # so a forward re-entered from a patch leaves this one's rows named.
            saved_pads = pads[0]
            pads[0] = padded_row_indices([(orig_len, x.shape[1])]) or None
            if pads[0]:
                refuse_padded_ring(
                    "wan", orig_len, base.sp_world(), len(pads[0]), getattr(self, "num_heads", 0),
                    None if frames[0] is None else stream_rows_at(orig_len, frames[0], per_frame),
                    frames[0])
            try:
                with scale_rows.real_rows(WAN_SCALE_ROLES,
                                          {"x": scale_rows.stream((orig_len, x.shape[1]))}):
                    for i, block in enumerate(self.blocks):
                        transformer_options["block_index"] = i
                        if ("double_block", i) in blocks_replace:
                            def block_wrap(args, block=block):
                                return {"img": block(args["img"], context=args["txt"],
                                                     e=args["vec"], freqs=args["pe"],
                                                     context_img_len=context_img_len,
                                                     transformer_options=args["transformer_options"])}

                            out = blocks_replace[("double_block", i)](
                                {"img": x, "txt": context, "vec": e0, "pe": freqs,
                                 "transformer_options": transformer_options},
                                {"original_block": block_wrap})
                            x = out["img"]
                        else:
                            x = block(x, e=e0, freqs=freqs, context=context,
                                      context_img_len=context_img_len,
                                      transformer_options=transformer_options)
            finally:
                pads[0] = saved_pads

            x = sp_gather(x, orig_len, dim=1)

            x = self.head(x, e)
            if context_latents is not None:
                x = x[:, :main_len]
            if ref_len > 0:
                x = x[:, ref_len:]
            return self.unpatchify(x, grid_sizes)

        for block in diffusion_model.blocks:
            self.bind(block.self_attn, "forward", usp_self_attention)
        self.bind(diffusion_model, "forward_orig", usp_forward_orig)
        if stock_forward is not None:
            self.bind(diffusion_model, "_forward", frames_forward)
        log.info("wan USP injected: %d blocks, sp=%d", len(diffusion_model.blocks), ctx.topology_sp)
