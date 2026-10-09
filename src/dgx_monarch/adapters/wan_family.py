"""Wan 2.1 and 2.2 video DiT sequence-parallel adapter.

Tokens shard after embedding. RoPE frequencies and per-token modulation take
the same shard. Attention uses the configured USP kernel, and gathering precedes
the output head. Non-divisible sequences use aligned zero padding that is
trimmed after gathering.
"""
from __future__ import annotations

import math

import torch

from ..log import get_logger
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    shard_seq,
    sp_gather,
)

log = get_logger(__name__)


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
        attn = ctx.usp_attention

        def usp_self_attention(self, x, freqs, transformer_options={}):
            # q/k use already-sharded RoPE; the USP call owns the collective.
            from comfy.ldm.flux.math import apply_rope1

            b, s = x.shape[:2]
            n, d = self.num_heads, self.head_dim
            q = apply_rope1(self.norm_q(self.q(x)).view(b, s, n, d), freqs)
            k = apply_rope1(self.norm_k(self.k(x)).view(b, s, n, d), freqs)
            v = self.v(x).view(b, s, n * d)
            out = attn(q.view(b, s, n * d), k.view(b, s, n * d), v, heads=n)

            patches = transformer_options.get("patches", {})
            for p in patches.get("attn1_patch", []):
                out = p({"x": out, "q": q, "k": k, "transformer_options": transformer_options})
            return self.o(out)

        def usp_forward_orig(self, x, t, context, clip_fea=None, freqs=None,
                             transformer_options={}, **kwargs):
            # Cross-attention uses local q with replicated context k/v. Only x,
            # freqs, and per-token e0 shard; the head requires a gathered stream.
            from comfy.ldm.wan.model import sinusoidal_embedding_1d

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
            context_latents = kwargs.get("context_latents", None)
            if context_latents is not None:
                for lat in context_latents:
                    cl = self.patch_embedding(lat.float().to(x.device)).to(x.dtype).flatten(2).transpose(1, 2)
                    x = torch.cat([x, cl], dim=1)

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
            for i, block in enumerate(self.blocks):
                transformer_options["block_index"] = i
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args, block=block):
                        return {"img": block(args["img"], context=args["txt"], e=args["vec"],
                                             freqs=args["pe"], context_img_len=context_img_len,
                                             transformer_options=args["transformer_options"])}

                    out = blocks_replace[("double_block", i)](
                        {"img": x, "txt": context, "vec": e0, "pe": freqs,
                         "transformer_options": transformer_options},
                        {"original_block": block_wrap})
                    x = out["img"]
                else:
                    x = block(x, e=e0, freqs=freqs, context=context,
                              context_img_len=context_img_len, transformer_options=transformer_options)

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
        log.info("wan USP injected: %d blocks, sp=%d", len(diffusion_model.blocks), ctx.topology_sp)
