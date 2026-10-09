"""Sequence-parallel adapters for SCAIL, SCAIL-2 and WanDancer.

These Wan variants add reference, pose, mask, or audio streams. After embedding
and concatenation, tokens and RoPE frequencies take identical shards. Block
self-attention uses USP, cross-attention keeps replicated K/V, and gathering
precedes the output head. WanDancer's single-key audio attention reduces to a
per-frame residual that can be indexed per local token. Animate2 shards another
way and lives in wan_animate2.py; this module re-exports it.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    shard_seq,
    sp_gather,
)
from .wan_animate2 import WanAnimate2Adapter  # noqa: F401

log = get_logger(__name__)


def _bind_usp_self_attention(diffusion_model, attn) -> None:
    """Route each variant's Wan self-attention through USP."""

    def usp_self_attention(self, x, freqs, transformer_options={}):
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

    for block in diffusion_model.blocks:
        Adapter.bind(block.self_attn, "forward", usp_self_attention)


class SCAILAdapter(Adapter):
    """SCAIL with time-concatenated references and appended pose tokens.

    SCAIL-2 uses the same forward with additional mask embeddings.
    """

    family = "wan_scail"
    model_base_classes = ("WAN21_SCAIL",)
    # New subclasses require an explicit forward-compatibility review.
    _EXACT = ("WAN21_SCAIL",)
    # UMT5 text needs no pad, as in plain Wan; reference and pose conditions are shared or batch-split.
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in self._EXACT:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"wan-scail variant {type(base_model).__name__} is not in the dgx-monarch "
            "launch set, which holds SCAIL and SCAIL-2 only. Render it in stock ComfyUI "
            "on one GPU, or, for a finetune that keeps the SCAIL or SCAIL-2 forward, set "
            "family_adapter='wan_scail' on the Init node (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward_orig(self, x, t, context, clip_fea=None, freqs=None,
                             transformer_options={}, pose_latents=None,
                             reference_latent=None, ref_mask_latents=None,
                             sam_latents=None, **kwargs):
            # References join before patch embedding; pose tokens join after
            # flattening. The resulting token axis and freqs shard identically.
            from comfy.ldm.wan.model import sinusoidal_embedding_1d

            if reference_latent is not None:
                x = torch.cat((reference_latent, x), dim=2)

            x = self.patch_embedding(x.float()).to(x.dtype)
            if ref_mask_latents is not None:
                # The mask embedding joins the video grid before flattening.
                x = x + self.patch_embedding_mask(ref_mask_latents.float()).to(x.dtype)
            grid_sizes = x.shape[2:]
            transformer_options["grid_sizes"] = grid_sizes
            x = x.flatten(2).transpose(1, 2)

            scail_pose_seq_len = 0
            if pose_latents is not None:
                scail_x = self.patch_embedding_pose(pose_latents.float()).to(x.dtype)
                if sam_latents is not None:  # SCAIL-2 pose-token mask embedding.
                    scail_x = scail_x + self.patch_embedding_mask(sam_latents.float()).to(x.dtype)
                scail_x = scail_x.flatten(2).transpose(1, 2)
                scail_pose_seq_len = scail_x.shape[1]
                x = torch.cat([x, scail_x], dim=1)

            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
            e = e.reshape(t.shape[0], -1, e.shape[-1])
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            # Scalar e0 broadcasts safely. Replicating a multi-row table would
            # restart the schedule on every rank, and no expansion is proven for
            # SCAIL's combined stream, so it refuses (docs/TROUBLESHOOTING.md #73).
            if e0.shape[1] != 1:
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    "SCAIL/SCAIL-2 sequence parallelism supports only scalar timesteps "
                    "(one modulation row per batch item); "
                    f"got {e0.shape[1]} modulation rows. Use the stock scalar-timestep "
                    "workflow, or topology 'single' (mode=local with gpus_per_host=1).",
                ))
            x, orig_len = shard_seq(x, dim=1)
            freqs, _ = shard_seq(freqs, dim=1)

            context = self.text_embedding(context)
            context_img_len = None
            if clip_fea is not None:
                if self.img_emb is not None:
                    context = torch.cat([self.img_emb(clip_fea), context], dim=1)
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
            # Trim pose tokens before unpatchify, then reference frames from the
            # restored temporal axis.
            if scail_pose_seq_len > 0:
                x = x[:, :-scail_pose_seq_len]
            x = self.unpatchify(x, grid_sizes)
            if reference_latent is not None:
                x = x[:, :, reference_latent.shape[2]:]
            return x

        _bind_usp_self_attention(diffusion_model, attn)
        self.bind(diffusion_model, "forward_orig", usp_forward_orig)
        log.info("%s USP injected: %d blocks, sp=%d", self.family,
                 len(diffusion_model.blocks), ctx.topology_sp)


class SCAIL2Adapter(SCAILAdapter):
    """SCAIL-2 with additive video and pose mask embeddings.

    It inherits the SCAIL forward but keeps an exact type and Ring2 dispatch.
    Registry order must place this subclass before SCAILAdapter.
    """

    model_base_classes = ("WAN21_SCAIL2",)
    _EXACT = ("WAN21_SCAIL2",)

    def attention_dispatch(self, dispatch):
        """Use validated plain-Wan Ring2 math for this exact variant only."""
        return dispatch.for_wan() if type(self) is SCAIL2Adapter else dispatch


class WanDancerAdapter(Adapter):
    """WanDancer audio-driven i2v with per-frame residual injection.

    Each audio group has one key, so its cross-attention output depends only on
    that frame's value and can be indexed by local token group.
    """

    family = "wan_dancer"
    model_base_classes = ("WAN22_WanDancer",)
    _EXACT = ("WAN22_WanDancer",)
    # UMT5 text needs no pad, as in plain Wan; audio and image conditions are shared or batch-split.
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in self._EXACT:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"wandancer variant {type(base_model).__name__} is not in the dgx-monarch "
            "launch set. Render it in stock ComfyUI on one GPU, or, for a finetune that "
            "keeps the WanDancer forward, set family_adapter='wan_dancer' on the Init node "
            "(docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward_orig(self, x, t, context, clip_fea=None, clip_fea_ref=None,
                             freqs=None, audio_embed=None, fps=30,
                             audio_inject_scale=1.0, transformer_options={}, **kwargs):
            if self.ref_conv is not None and kwargs.get("reference_latent") is not None:
                # seq_len is captured before full_ref is prepended, which would
                # misalign every per-frame residual index.
                raise UnsupportedModelError(
                    "WanDancer + reference_latent is not supported: the audio injector's "
                    "seq_len predates the full_ref prepend and would misalign the per-frame "
                    "residual (docs/MODELS.md)."
                )
            from comfy.ldm.wan.model import sinusoidal_embedding_1d

            # Non-30 fps selects the global embedding and output head.
            off_30 = int(fps + 0.5) != 30
            x = (self.patch_embedding_global if off_30 else self.patch_embedding)(x.float()).to(x.dtype)
            grid_sizes = x.shape[2:]
            latent_frames = grid_sizes[0]
            transformer_options["grid_sizes"] = grid_sizes
            x = x.flatten(2).transpose(1, 2)
            seq_len = x.size(1)

            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x[0].dtype))
            e = e.reshape(t.shape[0], -1, e.shape[-1])
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))

            # The small audio encoder remains replicated and never enters USP.
            audio_emb = None
            if audio_embed is not None:
                music_feature = self.music_projection(audio_embed)
                music_seq_len = music_feature.shape[1]
                music_ids = torch.arange(music_seq_len, device=music_feature.device,
                                         dtype=music_feature.dtype).reshape(1, -1, 1)
                music_freqs = self.music_rope_embedder(music_ids).movedim(1, 2)
                for layer in self.music_encoder:
                    music_feature = layer(music_feature, music_freqs)
                audio_emb = torch.nn.functional.interpolate(
                    music_feature.unsqueeze(1), size=(latent_frames * 8, self.dim),
                    mode="bilinear").squeeze(1)

            # Image tokens prepend the replicated cross-attention context.
            context = self.text_embedding(context)
            context_img_len = 0
            if self.img_emb is not None and clip_fea is not None:
                context = torch.cat([self.img_emb(clip_fea), context], dim=1)
                context_img_len += clip_fea.shape[-2]
            if self.img_emb_refimage is not None and clip_fea_ref is not None:
                context = torch.cat([self.img_emb_refimage(clip_fea_ref), context], dim=1)
                context_img_len += clip_fea_ref.shape[-2]

            # Expand per-frame e0 to token level before sharding. A replicated
            # frame table makes each rank stretch the whole schedule over its
            # local chunk: silently wrong modulation (docs/TROUBLESHOOTING.md #73).
            # Scalar e0 broadcasts safely.
            if e0.shape[1] > 1:
                tokens_per_e = -(-x.shape[1] // e0.shape[1])
                e0 = torch.repeat_interleave(e0, tokens_per_e, dim=1)[:, :x.shape[1]]
                e0, _ = shard_seq(e0, dim=1)

            x, orig_len = shard_seq(x, dim=1)
            freqs, _ = shard_seq(freqs, dim=1)

            # Shard frame-group IDs with the tokens. Each of the num_frames audio
            # groups covers seq_len // num_frames tokens, as in comfy's
            # AudioInjector_WAN; valid_local masks any divisibility padding.
            group_local = valid_local = None
            num_frames = 0
            if audio_emb is not None:
                num_frames = audio_emb.shape[1]
                tokens_per_group = max(seq_len // num_frames, 1)
                group_full = (torch.arange(seq_len, device=x.device) // tokens_per_group).view(1, -1, 1)
                group_local, _ = shard_seq(group_full, dim=1)
                group_local = group_local.reshape(-1).clamp_(max=num_frames - 1).long()
                valid_local, _ = shard_seq(
                    torch.ones(1, seq_len, 1, device=x.device, dtype=x.dtype), dim=1)

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
                if audio_emb is not None:
                    audio_attn_id = self.music_injector.injected_block_id.get(i)
                    if audio_attn_id is not None:
                        inj = self.music_injector.injector[audio_attn_id]
                        # One key makes softmax equal one, so q/k drop out. Index
                        # o(v(audio)) by each local token's frame group.
                        residual = inj.o(inj.v(audio_emb))          # (B, num_frames, dim)
                        x = x + residual[:, group_local, :] * valid_local * audio_inject_scale

            x = sp_gather(x, orig_len, dim=1)

            x = (self.head_global if off_30 else self.head)(x, e)
            x = self.unpatchify(x, grid_sizes)
            return x

        _bind_usp_self_attention(diffusion_model, attn)
        self.bind(diffusion_model, "forward_orig", usp_forward_orig)
        log.info("wan_dancer USP injected: %d blocks, %d audio-inject layers, sp=%d",
                 len(diffusion_model.blocks), len(diffusion_model.music_injector.injector),
                 ctx.topology_sp)
