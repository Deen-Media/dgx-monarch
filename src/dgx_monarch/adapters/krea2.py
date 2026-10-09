"""Krea2 single-stream image DiT adapter.

Text, image, positions, and other token-aligned tensors take identical shards.
Ulysses excludes divisibility-pad rows before attention, and the gathered image
stream is trimmed before the output head. Under pure Ulysses the text path runs
on every text row before the text shards, and the joint blocks attend in stock
key order.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    padded_row_indices,
    shard_seq,
    sp_gather,
    usp_options,
)
from .usp_sequence_order import RankMajorJointOrder

log = get_logger(__name__)


def krea2_reference_latents_summary(cond_list) -> tuple[bool, str | None]:
    """Read reference presence and method from a Comfy conditioning list.

    Krea2 has no model default method, so only conditioning metadata resolves
    one. Empty or malformed input returns `(False, None)`.
    """
    has_refs = False
    method: str | None = None
    for entry in cond_list or ():
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        cond_dict = entry[1] or {}
        if not isinstance(cond_dict, dict):
            continue
        if cond_dict.get("reference_latents"):
            has_refs = True
        candidate = cond_dict.get("reference_latents_method")
        if candidate is not None:
            method = candidate
    return has_refs, method


def krea2_ref_latents_would_reject(has_ref_latents: bool, ref_method: str | None) -> bool:
    """Share the preflight and forward decision; unresolved references are inert."""
    return ref_method is not None and has_ref_latents


KREA2_REF_LATENTS_REFUSAL_MESSAGE = refusal(
    RefusalClass.PHYSICS,
    "krea2: reference latents (used by identity-edit reference LoRAs) concatenate reference "
    "tokens onto the image stream, and ref_latents_method='index_timestep_zero' "
    "modulates every block at a global token boundary; neither survives sequence "
    "sharding or cfg-parallel padding. Run reference-image renders on topology 'single' "
    "(docs/MODELS.md).",
)


def _trim_trailing_pad(context):
    """Trim common trailing padding and return any residual ragged bias.

    Trimming is exact because pad keys have negative-infinity bias, their query
    outputs are discarded, and text RoPE is constant. Uniform input returns no
    mask for the flash path; ragged input keeps an additive bias. Fully zero
    conditioning remains real input and is not trimmed.
    """
    entries, seq = context.shape[0], context.shape[1]
    content = context.reshape(entries, seq, -1).abs().amax(dim=2) > 0  # (B, seq)
    positions = torch.arange(seq, device=context.device)
    last = torch.where(content, positions.expand_as(content), positions.new_full((), -1)).amax(dim=1)
    cutoff = torch.where(last < 0, positions.new_full((), seq), last + 1)  # (B,) real len (seq if fully-zero)
    trim = int(cutoff.amax())  # longest real length across the batch
    if trim < seq:
        context = context[:, :trim]
        positions = positions[:trim]
    if bool((cutoff >= trim).all()):
        return context, None  # every entry is full after the trim: maskless flash path
    trailing = positions.unsqueeze(0) >= cutoff.unsqueeze(1)  # (B, trim)
    return context, trailing.unsqueeze(1).to(context.dtype) * torch.finfo(context.dtype).min


def _krea2_reject_global_span_features(model, ref_latents, transformer_options,
                                       kwargs, path: str,
                                       reject_attn_patches: bool = False) -> None:
    """Reject features whose global token spans this forward cannot preserve.

    Reference modulation uses a global boundary, while `post_input` may rewrite
    complete streams and position IDs. Neither has a faithful sharded mapping.
    """
    ref_method = kwargs.get("ref_latents_method", getattr(model, "default_ref_method", None))
    has_refs = ref_latents is not None and len(ref_latents) > 0
    if krea2_ref_latents_would_reject(has_refs, ref_method):
        raise UnsupportedModelError(KREA2_REF_LATENTS_REFUSAL_MESSAGE)
    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        raise UnsupportedModelError(
            "krea2: a 'post_input' patch rewrites the GLOBAL img/txt token streams and their "
            f"position ids before the block loop, which the {path} distributed forward cannot apply "
            "faithfully. Run this workflow on topology 'single'."
        )
    # The cfg path publishes complete q/k/v. The SP path would expose rank-local
    # tensors and indices, so reject those hooks rather than change their domain.
    if reject_attn_patches:
        for hook in ("attn1_patch", "attn1_output_patch"):
            if patches.get(hook):
                raise UnsupportedModelError(
                    f"krea2: an '{hook}' patch receives q/k/v for the whole sequence in stock "
                    "comfy, but under sequence parallelism each rank holds only its own shard, "
                    "so the patch would see rank-local tokens and indices. Run this workflow on "
                    "topology 'cfg2' or 'single'."
                )


class Krea2Adapter(Adapter):
    family = "krea2"
    model_base_classes = ("Krea2",)
    # cfg2 runs native attention; the cfg-pad forward trims the equalizer's
    # zero rows or turns them into a key bias.
    cfg_cond_padding = "pad"
    cfg_pad_restores_stock_call = True

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timesteps, context, attention_mask=None,
                        ref_latents=None, transformer_options={}, **kwargs):
            # Joint blocks use USP. Under pure Ulysses the text path runs whole
            # on every rank; otherwise refiner blocks use USP on the text shard
            # and layerwise text blocks remain token-local. `ref_latents` keeps
            # its positional API slot: references with a method refuse, and
            # without one they are inert, as in stock.
            _krea2_reject_global_span_features(self, ref_latents, transformer_options,
                                               kwargs, "sharded (USP)",
                                               reject_attn_patches=True)

            import comfy.ldm.common_dit
            from comfy.ldm.flux.layers import timestep_embedding
            from einops import rearrange

            temporal = x.ndim == 5
            if temporal:
                b5, c5, t5, h5, w5 = x.shape
                x = x.reshape(b5 * t5, c5, h5, w5)
            bs, _c, H_orig, W_orig = x.shape
            patch = self.patch
            x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
            H, W = x.shape[-2], x.shape[-1]
            h_, w_ = H // patch, W // patch

            context = self._unpack_context(context)

            img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
            img = self.first(img)

            t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
            tvec = self.tproj(t)

            device = img.device
            txtpos = torch.zeros(bs, context.shape[1], 3, device=device, dtype=torch.float32)
            imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
            imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
            imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
            imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)

            if ctx.pure_ulysses:
                # A shard runs the text Linears and norms on half the text rows,
                # and the BLAS can pick another kernel for another row count. The
                # conditioning is replicated, so every rank runs stock's text
                # path on all rows, with native attention, and shards its output.
                context = self.txtmlp(self.txtfusion(
                    context, mask=None, transformer_options=transformer_options))
            context, txt_orig_len = shard_seq(context, dim=1)
            txtpos, _ = shard_seq(txtpos, dim=1)
            img, img_orig_len = shard_seq(img, dim=1)
            imgpos, _ = shard_seq(imgpos, dim=1)

            # Exclude synthetic keys from text-only and joint attention.
            txt_seg = (txt_orig_len, context.shape[1])
            joint_drop = padded_row_indices([txt_seg, (img_orig_len, img.shape[1])])
            if not ctx.pure_ulysses:
                usp_opts = usp_options(transformer_options, attn,
                                       drop_rows=padded_row_indices([txt_seg]))

                # Layerwise blocks are token-local; refiner blocks attend across text.
                tf = self.txtfusion
                b_, l_, n_, d_ = context.shape
                ctx_tokens = context.reshape(b_ * l_, n_, d_)
                for block in tf.layerwise_blocks:
                    ctx_tokens = block(ctx_tokens.contiguous(), mask=None,
                                       transformer_options=transformer_options)
                ctx_tokens = rearrange(ctx_tokens, "(b l) n d -> b l d n", b=b_, l=l_)
                ctx_tokens = tf.projector(ctx_tokens).squeeze(-1)
                for block in tf.refiner_blocks:
                    ctx_tokens = block(ctx_tokens, mask=None, transformer_options=usp_opts)
                context = self.txtmlp(ctx_tokens)

            txtlen, imglen = context.shape[1], img.shape[1]
            combined = torch.cat((context, img), dim=1)
            pos = torch.cat((txtpos, imgpos), dim=1)
            freqs = self.pe_embedder(pos)

            # Each rank joins [text, image] from its own shards, so the head
            # all-to-all gathers [text0, image0, text1, image1] where stock
            # attends all text then all image, and flash rounding depends on
            # key order. Pure Ulysses restores stock order at its full-axis
            # point; ring and hybrid keep their path.
            sequence_order = (RankMajorJointOrder(txtlen, imglen)
                              if ctx.pure_ulysses else None)
            joint_opts = usp_options(transformer_options, attn, drop_rows=joint_drop,
                                     sequence_order=sequence_order)
            for block in self.blocks:
                combined = block(combined, tvec, freqs, None, transformer_options=joint_opts)

            final = self.last(combined, t)
            out = final[:, txtlen:txtlen + imglen, :]
            out = sp_gather(out, img_orig_len, dim=1)

            out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                            h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
            out = out[:, :, :H_orig, :W_orig]
            if temporal:
                out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
            return out

        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("krea2 USP injected: %d blocks, sp=%d", len(diffusion_model.blocks), ctx.topology_sp)

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Support asymmetric cfg prompts without attending zero-row padding.

        `_trim_trailing_pad` decides trim or bias; a ragged batch's text-key
        bias reaches the refiner and joint blocks.
        """

        def cfg_pad_forward(self, x, timesteps, context, attention_mask=None,
                            ref_latents=None, transformer_options={}, **kwargs):
            _krea2_reject_global_span_features(self, ref_latents, transformer_options,
                                               kwargs, "cfg-pad")

            import comfy.ldm.common_dit
            from comfy.ldm.flux.layers import timestep_embedding
            from einops import rearrange

            temporal = x.ndim == 5
            if temporal:
                b5, c5, t5, h5, w5 = x.shape
                x = x.reshape(b5 * t5, c5, h5, w5)
            bs, _c, H_orig, W_orig = x.shape
            patch = self.patch
            x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch, patch))
            H, W = x.shape[-2], x.shape[-1]
            h_, w_ = H // patch, W // patch

            context = self._unpack_context(context)
            context, text_bias = _trim_trailing_pad(context)

            img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
            img = self.first(img)

            t = self.tmlp(timestep_embedding(timesteps, self.tdim).unsqueeze(1).to(img.dtype))
            tvec = self.tproj(t)

            # Only sequence-attending refiner and joint blocks receive the bias.
            context = self.txtfusion(context, mask=text_bias, transformer_options=transformer_options)
            context = self.txtmlp(context)

            txtlen, imglen = context.shape[1], img.shape[1]
            combined = torch.cat((context, img), dim=1)

            device = combined.device
            txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)
            imgids = torch.zeros(h_, w_, 3, device=device, dtype=torch.float32)
            imgids[..., 1] = torch.arange(h_, device=device, dtype=torch.float32)[:, None]
            imgids[..., 2] = torch.arange(w_, device=device, dtype=torch.float32)[None, :]
            imgpos = imgids.reshape(1, h_ * w_, 3).repeat(bs, 1, 1)
            pos = torch.cat((txtpos, imgpos), dim=1)
            freqs = self.pe_embedder(pos)

            joint_bias = None
            if text_bias is not None:
                # Extend the text bias with always-visible image keys.
                joint_bias = torch.cat((text_bias, text_bias.new_zeros(bs, 1, imglen)), dim=2)

            # Complete q/k/v and global image indices preserve attention hooks.
            block_opts = dict(transformer_options)
            block_opts["total_blocks"] = len(self.blocks)
            block_opts["block_type"] = "single"
            block_opts["img_slice"] = [txtlen, combined.shape[1]]
            for i, block in enumerate(self.blocks):
                block_opts["block_index"] = i
                combined = block(combined, tvec, freqs, joint_bias, transformer_options=block_opts)

            final = self.last(combined, t)
            out = final[:, txtlen:txtlen + imglen, :]
            out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                            h=h_, w=w_, ph=patch, pw=patch, c=self.channels)
            out = out[:, :, :H_orig, :W_orig]
            if temporal:
                out = out.reshape(b5, t5, self.channels, H_orig, W_orig).movedim(1, 2)
            return out

        self.bind(diffusion_model, "_forward", cfg_pad_forward)
        log.info("krea2 cfg-pad forward installed")
