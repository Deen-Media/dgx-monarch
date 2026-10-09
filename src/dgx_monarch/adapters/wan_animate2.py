"""Exact spatial-within-frame sequence parallelism for Wan-Animate2."""
from __future__ import annotations

import json

import torch

from ..refusal import RefusalClass, refusal
from .attention_patches import (
    assert_no_foreign_attention_override,
    assert_usp_attention_patches_safe,
)
from .base import Adapter, InjectionContext, UnsupportedModelError, sp_rank, sp_world


def is_wan_animate2_metadata(header: dict) -> bool:
    """Return whether Comfy's preserved config selects native Animate2."""
    config = header.get("config")
    if not isinstance(config, str):
        return False
    try:
        parsed = json.loads(config)
    except (TypeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    transformer = parsed.get("transformer")
    return isinstance(transformer, dict) and transformer.get("model_type") == "animate2"


def is_exact_animate2_pair(base_model, diffusion_model) -> bool:
    """Return whether both are exactly Comfy's Animate2 classes.

    It lives here so adapters/fsdp.py needs no optional Comfy import.
    """
    try:
        from comfy import model_base
        from comfy.ldm.wan.model_animate2 import WanAnimate2Model
    except (AttributeError, ImportError):
        return False
    return type(base_model) is model_base.WAN_Animate2 and type(diffusion_model) is WanAnimate2Model


class WanAnimate2Adapter(Adapter):
    """Keep every temporal frame on each rank and shard only spatial tokens."""

    family = "wan_animate2"
    model_base_classes = ("WAN_Animate2",)
    _EXACT = ("WAN_Animate2",)
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, "WAN_Animate2", None)
        if cls is not None and type(base_model) is cls:
            return True
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"wan-animate2 variant {type(base_model).__name__} is not in the launch set. "
            "Use an exact WAN_Animate2 checkpoint or topology 'single'.",
        ))

    @staticmethod
    def _spatial_local(t, frames, pixels, name):
        world = sp_world()
        if pixels % world:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"Wan-Animate-2 {name} spatial token count {pixels} is not divisible by "
                f"the sequence-parallel degree {world}; masked spatial padding is unproven. "
                "Use a topology whose degree divides the spatial grid, or topology 'single'.",
            ))
        if t.shape[1] != frames * pixels:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"Wan-Animate-2 {name} has {t.shape[1]} tokens, expected {frames * pixels}; "
                "this Comfy forward surface is not reviewed for spatial parallelism. "
                "Use topology 'single'.",
            ))
        shape = t.shape
        local_pixels = pixels // world
        return t.reshape(shape[0], frames, pixels, *shape[2:])[
            :, :, sp_rank() * local_pixels : (sp_rank() + 1) * local_pixels
        ].reshape(shape[0], frames * local_pixels, *shape[2:])

    @staticmethod
    def _gather_spatial(t, frames, pixels):
        from xfuser.core.distributed import get_sp_group

        world = sp_world()
        local_pixels = pixels // world
        gathered = get_sp_group().all_gather(t.contiguous(), dim=1)
        return (gathered.reshape(t.shape[0], world, frames, local_pixels, *t.shape[2:])
                .transpose(1, 2).reshape(t.shape[0], frames * pixels, *t.shape[2:]))

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        def stock_attention(q, k, v, heads, transformer_options, preferred_attention):
            from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

            return optimized_attention(
                AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                heads=heads, preferred_attention=preferred_attention, transformer_options=transformer_options)

        def pose_attn(self, x, freqs, transformer_options={}):
            b, s = x.shape[:2]
            n, d = self.num_heads, self.head_dim
            q, k, v = self.qkv(x, freqs)
            frames = transformer_options.get("_dgxm_animate2_pose_frames")
            pixels = transformer_options.get("_dgxm_animate2_pixels")
            if not isinstance(frames, int) or not isinstance(pixels, int):
                raise UnsupportedModelError(refusal(RefusalClass.PHYSICS,
                    "Wan-Animate-2 pose attention lacks reviewed spatial ownership metadata. "
                    "Use topology 'single'."))
            kg = self._dgxm_adapter._gather_spatial(k, frames, pixels)
            vg = self._dgxm_adapter._gather_spatial(v, frames, pixels)
            out = stock_attention(q.reshape(b, s, n * d), kg.reshape(b, -1, n * d),
                                  vg.reshape(b, -1, n * d), n, transformer_options,
                                  self.comfy_attention)
            return self.o(self._attn1_patch(out, q, k, transformer_options)), k, v

        def gen_attn(self, x, freqs, k_pose, v_pose, f_gen, hw, buffers,
                     ref_strength=1.0, transformer_options={}):
            del buffers
            b, s = x.shape[:2]
            n, d = self.num_heads, self.head_dim
            q, k, v = self.qkv(x, freqs)
            if ref_strength != 1.0:
                v[:, :hw] *= ref_strength
            pixels = transformer_options.get("_dgxm_animate2_pixels")
            if not isinstance(pixels, int):
                raise UnsupportedModelError(refusal(RefusalClass.PHYSICS,
                    "Wan-Animate-2 generation attention lacks reviewed spatial ownership metadata. "
                    "Use topology 'single'."))
            adapter = self._dgxm_adapter
            kg = adapter._gather_spatial(k, f_gen, pixels)
            vg = adapter._gather_spatial(v, f_gen, pixels)
            if k_pose is not None:
                kpg = adapter._gather_spatial(k_pose, f_gen - 1, pixels)
                vpg = adapter._gather_spatial(v_pose, f_gen - 1, pixels)
            else:
                kpg = vpg = None
            out = torch.empty_like(q)
            for j in range(f_gen):
                lo, hi = j * hw, (j + 1) * hw
                if j == 0 or kpg is None:
                    kk, vv = kg, vg
                else:
                    pose_lo, pose_hi = (j - 1) * pixels, j * pixels
                    kk = torch.cat((kg, kpg[:, pose_lo:pose_hi]), 1)
                    vv = torch.cat((vg, vpg[:, pose_lo:pose_hi]), 1)
                out[:, lo:hi] = stock_attention(
                    q[:, lo:hi].reshape(b, hw, n * d), kk.reshape(b, -1, n * d),
                    vv.reshape(b, -1, n * d), n, transformer_options, self.comfy_attention,
                ).reshape(b, hw, n, d)
            return self.o(self._attn1_patch(out.reshape(b, s, n * d), q, k, transformer_options))

        def forward_orig(self, x, t, context, clip_fea=None, freqs=None, freqs_pose=None,
                         pose_latents=None, clip_fea_pose=None, context_pose=None,
                         pose_strength=1.0, reference_strength=1.0, transformer_options={}, **kwargs):
            # Frame-indexed K/V runs native Comfy attention after the spatial
            # gather, not the xFuser dispatcher. A warm resident can change
            # selectors, so refuse before the patch embedding or any collective
            # when the selector no longer names TORCH_FLASH.
            if getattr(ctx.usp_attention, "effective_kernel", "") != "TORCH_FLASH":
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    "Wan-Animate-2 sequence parallelism calls native Comfy attention and "
                    "requires TORCH_FLASH. Use TORCH_FLASH or topology 'single'.",
                ))
            assert_no_foreign_attention_override(transformer_options, "wan_animate2")
            assert_usp_attention_patches_safe(
                transformer_options, "wan_animate2",
                ("attn1_patch", "attn1_output_patch", "attn2_patch"),
            )
            from comfy.ldm.wan.model import sinusoidal_embedding_1d

            if (transformer_options.get("patches_replace", {}).get("dit")
                    or transformer_options.get("patches", {}).get("double_block")):
                raise UnsupportedModelError(refusal(RefusalClass.PHYSICS,
                    "Wan-Animate-2 sequence parallelism cannot apply dit block-replacement or double_block patches. "
                    "Remove the patch, or run on topology 'single'."))
            x = self.patch_embedding(x.float()).to(x.dtype)
            grid = x.shape[2:]
            f_gen, gh, gw = grid
            pixels = gh * gw
            transformer_options["grid_sizes"] = grid
            transformer_options["_dgxm_animate2_pixels"] = pixels
            transformer_options["_dgxm_animate2_pose_frames"] = f_gen - 1
            x = self._dgxm_adapter._spatial_local(x.flatten(2).transpose(1, 2), f_gen, pixels, "generation")
            freqs = self._dgxm_adapter._spatial_local(freqs, f_gen, pixels, "generation RoPE")
            apply_pose = pose_latents is not None
            if apply_pose and pose_latents.shape[2] != f_gen - 1:
                raise UnsupportedModelError(refusal(RefusalClass.PHYSICS,
                    f"Wan-Animate-2 pose branch has {pose_latents.shape[2]} latent frames, expected {f_gen - 1}. "
                    "Use the native conditioner shape or topology 'single'."))
            cache = transformer_options.get("animate2_cache") if apply_pose else None
            if cache is not None:
                cache.select(pose_latents[:1])
            cached = cache is not None and cache.filled(len(self.blocks))
            x_pose = None
            if apply_pose and not cached:
                pose = self.patch_embedding(torch.cat(
                    [pose_latents, torch.ones_like(pose_latents[:, :4]), pose_latents], 1).float()).to(x.dtype)
                x_pose = self._dgxm_adapter._spatial_local(
                    pose.flatten(2).transpose(1, 2), f_gen - 1, pixels, "pose")
            if apply_pose:
                freqs_pose = self._dgxm_adapter._spatial_local(freqs_pose, f_gen - 1, pixels, "pose RoPE")
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).to(dtype=x.dtype))
            e = e.reshape(t.shape[0], -1, e.shape[-1])
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            e0_pose = None
            if apply_pose:
                e_pose = self.time_embedding(sinusoidal_embedding_1d(
                    self.freq_dim, torch.ones_like(t.flatten())).to(dtype=x.dtype))
                e_pose = e_pose.reshape(t.shape[0], -1, e_pose.shape[-1])
                e0_pose = self.time_projection(e_pose).unflatten(2, (6, self.dim))
            context_gen = self.text_embedding(context)
            context_img_len = None
            if clip_fea is not None:
                if self.img_emb is not None:
                    context_gen = torch.cat([self.img_emb(clip_fea), context_gen], 1)
                context_img_len = clip_fea.shape[-2]
            native_context_pose = context_pose
            context_pose = context_img_len_pose = None
            if apply_pose and not cached:
                context_pose = self.text_embedding(context if native_context_pose is None else native_context_pose)
                clip_fea_pose = clip_fea if clip_fea_pose is None else clip_fea_pose
                if clip_fea_pose is not None:
                    if self.img_emb is not None:
                        context_pose = torch.cat([self.img_emb(clip_fea_pose), context_pose], 1)
                    context_img_len_pose = clip_fea_pose.shape[-2]
            transformer_options["total_blocks"] = len(self.blocks)
            transformer_options["block_type"] = "double"
            if cache is not None and not cached and "context_window" in transformer_options:
                for i, block in enumerate(self.blocks):
                    transformer_options["block_index"] = i
                    cache.put(i, x_pose[:1])
                    x_pose = block.forward_pose(x_pose, e0_pose, freqs_pose, context_pose,
                                                context_img_len=context_img_len_pose,
                                                transformer_options=transformer_options)[0]
                x_pose = None
                cached = True
            local_pixels = pixels // sp_world()
            for i, block in enumerate(self.blocks):
                transformer_options["block_index"] = i
                if not apply_pose:
                    k_pose = v_pose = None
                elif cached:
                    x_pose_in = cache.take(i, x.device, x.dtype, x.shape[0])
                    cache.prefetch(i + 1, x.device, x.dtype)
                    k_pose, v_pose = block.kv_from_input(x_pose_in, e0_pose, freqs_pose,
                                                         transformer_options=transformer_options)
                else:
                    if cache is not None:
                        cache.put(i, x_pose[:1])
                    x_pose, k_pose, v_pose = block.forward_pose(
                        x_pose, e0_pose, freqs_pose, context_pose,
                        context_img_len=context_img_len_pose, transformer_options=transformer_options)
                if v_pose is not None and pose_strength != 1.0:
                    v_pose = v_pose * pose_strength
                x = block.forward_gen(x, e0, freqs, context_gen, k_pose, v_pose, f_gen,
                                      local_pixels, None, ref_strength=reference_strength,
                                      context_img_len=context_img_len,
                                      transformer_options=transformer_options)
            x = self._dgxm_adapter._gather_spatial(x, f_gen, pixels)
            return self.unpatchify(self.head(x, e), grid)

        diffusion_model._dgxm_adapter = self
        for block in diffusion_model.blocks:
            block.self_attn._dgxm_adapter = self
            Adapter.bind(block.self_attn, "forward_pose", pose_attn)
            Adapter.bind(block.self_attn, "forward_gen", gen_attn)
        Adapter.bind(diffusion_model, "forward_orig", forward_orig)
