"""HunyuanImage 2.1 and HunyuanVideo 1.5 sequence-parallel adapter.

The image, refiner, video, and SR variants share a double-stream/single-stream
DiT. TokenRefiner remains replicated because it attends only over text and may
use a text mask. Additional text tokens join before sharding. Image and text
streams shard separately with matching local position IDs, then the joined
single stream is reshared. Gathering precedes the output head and all trims.

USP rejects nontrivial attention masks, global guiding-frame modulation, and a
text mask combined with byt5 tokens because those contracts cannot be expressed
on the sharded sequence.
"""
from __future__ import annotations

from typing import Any

import torch

from ..log import get_logger
from .attention_patches import assert_usp_attention_patches_safe
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    padded_row_indices,
    shard_seq,
    sp_gather,
    sp_rank,
    usp_options,
)
from .chroma_text import full_row_text_projections
from .flux_family import _control_span_overlap  # Remap control spans in both loops.
from .usp_sequence_order import RankMajorJointOrder

log = get_logger(__name__)


# Exact types prevent related variants with global guiding-frame modulation from
# inheriting an incompatible sharded forward.
_HUNYUAN_SUPPORTED_EXACT = (
    "HunyuanImage21",
    "HunyuanImage21Refiner",
    "HunyuanVideo15",
    "HunyuanVideo15_SR_Distilled",
)


def _img_slice_bounds(txt_len: int, img_len: int, ref_len: int) -> tuple[int, int]:
    """Return the real-image bounds within gathered ``[text, ref, image]``."""
    return txt_len + ref_len, txt_len + img_len


def _text_modules(block) -> tuple[tuple[torch.nn.Linear, bool], ...]:
    """The double block's text-stream Linears, which stock runs on all text rows.

    `txt_attn.proj` reads a slice of the joint `[text, image]` attention
    output, so it takes the joint batch stride. Selection is
    `isinstance(module, torch.nn.Linear)` as for Chroma (see
    chroma_text._modules): plain fp8 and bf16 weights are wrapped, and
    quantized-metadata Linears are skipped.
    """
    attention = getattr(block, "txt_attn", None)
    mlp: Any = getattr(block, "txt_mlp", ())
    candidates = (
        (getattr(attention, "qkv", None), False),
        (getattr(attention, "proj", None), True),
        (mlp[0] if len(mlp) > 0 else None, False),
        (mlp[2] if len(mlp) > 2 else None, False),
    )
    return tuple((module, joint_stride) for module, joint_stride in candidates
                 if isinstance(module, torch.nn.Linear))


class HunyuanAdapter(Adapter):
    family = "hunyuan"
    # Broad base classes enter dispatch; matches() then enforces the exact list.
    model_base_classes = ("HunyuanImage21", "HunyuanVideo")
    exact_model_base_classes = _HUNYUAN_SUPPORTED_EXACT
    # TokenRefiner consumes a text-only mask. A joint additive mask has the wrong
    # shape, so CFG padding uses the dedicated text-mask rule.
    cfg_cond_padding = "pad+text-mask"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in self.exact_model_base_classes:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"hunyuan variant {type(base_model).__name__} is not in the dgx-monarch "
            "launch set (HunyuanImage 2.1 + refiner and HunyuanVideo 1.5 + SR only). "
            "HunyuanVideo 1.0 / i2v / skyreels use guiding-frame modulation over "
            "global token ranges, which sequence sharding cannot express; run "
            "those on a cfg2 or single topology (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        family = self.family

        def usp_forward_orig(self, img, img_ids, txt, txt_ids, txt_mask, timesteps,
                             y=None, txt_byt5=None, clip_fea=None, guidance=None,
                             guiding_frame_index=None, ref_latent=None,
                             disable_time_r=False, control=None, transformer_options={}):
            # Guards precede ComfyUI imports so lightweight contract tests reach them.
            if guiding_frame_index is not None:
                raise UnsupportedModelError(
                    "hunyuan: guiding_frame_index modulates GLOBAL [start, end) token "
                    "ranges, which do not survive sequence sharding. Run this i2v "
                    "workflow on a cfg2 or single topology (docs/MODELS.md)."
                )
            byt5_present = self.byt5_in is not None and txt_byt5 is not None
            if txt_mask is not None:
                # model_base removes all-ones masks, so this mask is nontrivial.
                if byt5_present:
                    # The context-width mask does not cover the byt5-grown stream.
                    raise UnsupportedModelError(
                        "hunyuan: a real attention mask together with byt5 glyph tokens "
                        "hits an upstream mask-width mismatch (model.py:388) AND the "
                        "sharded USP kernel cannot apply a mask. Make the byt5 conds "
                        "equal-length between cond/uncond (drops the mask), or run "
                        "this workflow on a cfg2 or single topology (docs/MODELS.md)."
                    )
                raise UnsupportedModelError(
                    "hunyuan: this render carries a real attention mask, which the "
                    "sharded USP attention kernel has no way to apply. Run masked "
                    "workflows on a cfg2 or single topology (docs/MODELS.md)."
                )
            assert_usp_attention_patches_safe(transformer_options, family)

            from comfy.ldm.flux.layers import timestep_embedding

            transformer_options = transformer_options.copy()
            patches_replace = transformer_options.get("patches_replace", {})

            initial_shape = list(img.shape)
            img = self.img_in(img)
            vec = self.time_in(timestep_embedding(timesteps, 256, time_factor=1.0).to(img.dtype))

            if (self.time_r_in is not None) and (not disable_time_r):
                # Meanflow/SR time conditioning is per-batch and remains replicated.
                w = torch.where(transformer_options['sigmas'][0] == transformer_options['sample_sigmas'])[0]
                if len(w) > 0:
                    timesteps_r = transformer_options['sample_sigmas'][w[0] + 1]
                    timesteps_r = timesteps_r.unsqueeze(0).to(device=timesteps.device, dtype=timesteps.dtype)
                    vec_r = self.time_r_in(timestep_embedding(timesteps_r, 256, time_factor=1000.0).to(img.dtype))
                    vec = (vec + vec_r) if self.params.meanflow_sum else (vec + vec_r) / 2

            ref_len = 0
            if ref_latent is not None:
                # Reference tokens and IDs grow together before sharding and are
                # removed after gathering by _img_slice_bounds.
                ref_latent_ids = self.img_ids(ref_latent)
                ref_latent = self.img_in(ref_latent)
                img = torch.cat([ref_latent, img], dim=-2)
                ref_latent_ids[..., 0] = -1
                ref_latent_ids[..., 2] += (initial_shape[-1] // self.patch_size[-1])
                img_ids = torch.cat([ref_latent_ids, img_ids], dim=-2)
                ref_len = ref_latent.shape[1]

            # Supported paths keep vector conditioning per-batch.
            if self.vector_in is not None:
                vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

            if self.params.guidance_embed:
                if guidance is not None:
                    vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))

            # Preserve the TokenRefiner's integer-to-additive mask conversion.
            if txt_mask is not None and not torch.is_floating_point(txt_mask):
                txt_mask = (txt_mask - 1).to(img.dtype) * torch.finfo(img.dtype).max

            # TokenRefiner attends only over replicated text, outside USP.
            txt = self.txt_in(txt, timesteps, txt_mask, transformer_options=transformer_options)

            if self.cond_type_embedding is not None:
                self.cond_type_embedding.to(txt.device)
                cond_emb = self.cond_type_embedding(torch.zeros_like(txt[:, :, 0], device=txt.device, dtype=torch.long))
                txt = txt + cond_emb.to(txt.dtype)

            if byt5_present:
                # byt5 tokens and IDs shard together. Their zero position IDs make
                # the differing token/ID concatenation order RoPE-invariant.
                txt_byt5 = self.byt5_in(txt_byt5)
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(torch.ones_like(txt_byt5[:, :, 0], device=txt_byt5.device, dtype=torch.long))
                    txt_byt5 = txt_byt5 + cond_emb.to(txt_byt5.dtype)
                    txt = torch.cat((txt_byt5, txt), dim=1)  # byt5 first for HunyuanVideo 1.5
                else:
                    txt = torch.cat((txt, txt_byt5), dim=1)
                txt_byt5_ids = torch.zeros((txt_ids.shape[0], txt_byt5.shape[1], txt_ids.shape[-1]), device=txt_ids.device, dtype=txt_ids.dtype)
                txt_ids = torch.cat((txt_ids, txt_byt5_ids), dim=1)

            if clip_fea is not None:
                # Vision tokens join the text stream with zero position IDs.
                txt_vision_states = self.vision_in(clip_fea)
                if self.cond_type_embedding is not None:
                    cond_emb = self.cond_type_embedding(2 * torch.ones_like(txt_vision_states[:, :, 0], dtype=torch.long, device=txt_vision_states.device))
                    txt_vision_states = txt_vision_states + cond_emb
                txt = torch.cat((txt_vision_states.to(txt.dtype), txt), dim=1)
                extra_txt_ids = torch.zeros((txt_ids.shape[0], txt_vision_states.shape[1], txt_ids.shape[-1]), device=txt_ids.device, dtype=txt_ids.dtype)
                txt_ids = torch.cat((txt_ids, extra_txt_ids), dim=1)

            img_len = img.shape[1]  # Includes reference rows; captured before sharding.

            # Shard image and text separately, with matching local position IDs.
            img, img_orig_len = shard_seq(img, dim=1)
            txt, txt_orig_len = shard_seq(txt, dim=1)
            # Each double block joins [txt_local, img_local].
            txt_ids_local, _ = shard_seq(txt_ids, dim=1)
            img_ids_local, _ = shard_seq(img_ids, dim=1)
            pe_double = self.pe_embedder(torch.cat((txt_ids_local, img_ids_local), dim=1))

            # Ulysses' head all-to-all gathers each rank's [txt_local, img_local]
            # pair rank-major, but stock attends [all text, all image] and flash
            # rounding depends on key order. Pure Ulysses restores stock order at
            # its full-axis point; with no text rows the order is already stock's.
            # The zero row each odd stream took to divide the shard is a real key
            # after modulation, so the same point drops it. Ring and hybrid keep
            # their path: they cannot exclude pads, and naming them would refuse.
            sequence_order = drop_rows = None
            if ctx.pure_ulysses:
                if txt.shape[1]:
                    sequence_order = RankMajorJointOrder(txt.shape[1], img.shape[1])
                drop_rows = padded_row_indices(
                    [(txt_orig_len, txt.shape[1]), (img_orig_len, img.shape[1])])
            usp_opts = usp_options(transformer_options, attn, drop_rows=drop_rows,
                                   sequence_order=sequence_order)

            blocks_replace = patches_replace.get("dit", {})
            usp_opts["total_blocks"] = len(self.double_blocks)
            usp_opts["block_type"] = "double"
            rows = img.shape[1]
            row0 = sp_rank() * rows
            for i, block in enumerate(self.double_blocks):
                usp_opts["block_index"] = i
                # A shard would run the text Linears on half the text rows, and a
                # BLAS may pick another kernel for another row count. Under pure
                # Ulysses each runs on all text rows in stock's layout.
                with full_row_text_projections(
                        block, txt_orig_len, img_orig_len,
                        pure_ulysses=ctx.pure_ulysses, select=_text_modules):
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args, block=block):
                            out = {}
                            out["img"], out["txt"] = block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims_img=args["modulation_dims_img"], modulation_dims_txt=args["modulation_dims_txt"], transformer_options=args["transformer_options"])
                            return out

                        out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe_double, "attention_mask": None, 'modulation_dims_img': None, 'modulation_dims_txt': None, 'transformer_options': usp_opts}, {"original_block": block_wrap})
                        txt = out["txt"]
                        img = out["img"]
                    else:
                        img, txt = block(img=img, txt=txt, vec=vec, pe=pe_double, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None, transformer_options=usp_opts)

                if control is not None:  # ControlNet targets image rows [0, add_len).
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            # Remap the global control span onto this rank's rows.
                            ov = _control_span_overlap(row0, rows, 0, add.shape[1])
                            if ov is not None:
                                img[:, ov[0]:ov[1]] += add[:, ov[2]:ov[3]]

            img = sp_gather(img, img_orig_len, dim=1)
            txt = sp_gather(txt, txt_orig_len, dim=1)

            img = torch.cat((txt, img), 1)
            txt_len_full = txt.shape[1]

            # Reshard the joined stream and its concatenated IDs identically.
            joint_ids_local, _ = shard_seq(torch.cat((txt_ids, img_ids), dim=1), dim=1)
            pe_single = self.pe_embedder(joint_ids_local)
            img, joint_orig_len = shard_seq(img, dim=1)

            # Contiguous chunks of one joined stream gather in stock order, so
            # the single loop takes no order, and its pad is the stream's tail.
            # Copying the double-loop options keeps what stock's one shared dict
            # carries between the loops.
            single_drop = None
            if ctx.pure_ulysses:
                single_drop = padded_row_indices([(joint_orig_len, img.shape[1])])
            single_opts = usp_options(usp_opts, attn, drop_rows=single_drop)
            single_opts["total_blocks"] = len(self.single_blocks)
            single_opts["block_type"] = "single"
            # Record the full [text, image] split; the block tensors are shards.
            single_opts["img_slice"] = [txt_len_full, joint_orig_len]

            rows = img.shape[1]
            row0 = sp_rank() * rows
            for i, block in enumerate(self.single_blocks):
                single_opts["block_index"] = i
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args, block=block):
                        out = {}
                        out["img"] = block(args["img"], vec=args["vec"], pe=args["pe"], attn_mask=args["attention_mask"], modulation_dims=args["modulation_dims"], transformer_options=args["transformer_options"])
                        return out

                    out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe_single, "attention_mask": None, 'modulation_dims': None, 'transformer_options': single_opts}, {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=vec, pe=pe_single, attn_mask=None, modulation_dims=None, transformer_options=single_opts)

                if control is not None:  # ControlNet targets joint-stream image rows.
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            ov = _control_span_overlap(row0, rows, txt_len_full, add.shape[1])
                            if ov is not None:
                                img[:, ov[0]:ov[1]] += add[:, ov[2]:ov[3]]

            img = sp_gather(img, joint_orig_len, dim=1)
            lo, hi = _img_slice_bounds(txt_len_full, img_len, ref_len)
            img = img[:, lo:hi]

            # Guiding-frame modulation was rejected, so vec remains per-batch.
            img = self.final_layer(img, vec, modulation_dims=None)  # (N, T, patch**2 * out_ch)

            # Restore spatial dimensions after the gathered output head.
            shape = initial_shape[-len(self.patch_size):]
            for i in range(len(shape)):
                shape[i] = shape[i] // self.patch_size[i]
            img = img.reshape([img.shape[0], *shape, self.out_channels, *self.patch_size])
            if img.ndim == 8:
                img = img.permute(0, 4, 1, 5, 2, 6, 3, 7)
                img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3], initial_shape[4])
            else:
                img = img.permute(0, 3, 1, 4, 2, 5)
                img = img.reshape(initial_shape[0], self.out_channels, initial_shape[2], initial_shape[3])
            return img

        self.bind(diffusion_model, "forward_orig", usp_forward_orig)
        log.info(
            "hunyuan USP injected: %d double + %d single blocks, sp=%d",
            len(diffusion_model.double_blocks), len(diffusion_model.single_blocks), ctx.topology_sp,
        )
