"""Z-Image latent and pixel-space NextDiT adapter.

Caption and image tokens are learned-padded to multiples of 32, refined
separately, and concatenated into one maskless joint-attention stream.
``patchify_and_embed`` stays replicated; the concatenated tokens and RoPE
frequencies shard identically across the joint blocks and gather before the
head. The x32 invariant prevents synthetic zero rows, which this maskless model
would otherwise attend.

Omni/edit mode uses global token spans and doubled timesteps that do not survive
sequence sharding, so it refuses. The sharded and cfg-pad forwards also refuse
Ming-Image inputs if they reach a Z-Image call (`_reject_ming_image`).
The pixel-space variant shares the backbone and applies its per-patch decoder
after gather.

CFG parallel trims driver-added trailing caption padding for batch-1 ranks and
refuses ragged batches. Asymmetric-prompt trimming has not been tested on
hardware; the passing CFG2 test used equal-token prompts (docs/VALIDATION.md).
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    _mask_is_noop,
    shard_seq,
    sp_gather,
    sp_world,
    usp_options,
)
from .krea2 import _trim_trailing_pad

log = get_logger(__name__)

# ``Lumina2`` is shared by Lumina 2.0 and latent Z-Image; learned x32 pad tokens
# select Z-Image. Ming-Image has its own comfy subclass, which the exact-type
# match skips, and the pixel-space type is distinct.
_ZIMAGE_LATENT_BASE = "Lumina2"
_ZIMAGE_PIXEL_BASE = "ZImagePixelSpace"


def _ulysses_degree() -> int:
    """The active Ulysses degree, or 1 when model-parallel is not initialized.

    The fallback keeps CPU injection tests independent of distributed state.
    """
    try:
        from xfuser.core.distributed import get_ulysses_parallel_world_size

        return int(get_ulysses_parallel_world_size())
    except Exception:
        return 1


def _assert_ulysses_divides_heads(n_heads: int, ulysses_degree: int) -> None:
    """Require the Ulysses degree to divide the attention-head count.

    Z-Image has 30 heads, so uly2 is valid and uly4 is not. Ring has no head
    constraint (docs/MODELS.md, How a family qualifies).
    """
    if ulysses_degree > 1 and n_heads % ulysses_degree != 0:
        raise UnsupportedModelError(
            f"zimage: {n_heads} attention heads are not divisible by the Ulysses degree "
            f"{ulysses_degree} (Ulysses splits the heads across ranks). Z-Image has 30 heads: "
            "uly2 works (30 % 2 == 0) but uly4 does not. Use a ring topology (no head "
            "constraint) or uly2; a two-rank world is unaffected."
        )


def _assert_concat_divisible(concat_len: int, sp: int) -> None:
    """Keep the learned-padded joint stream divisible without synthetic rows.

    Caption and image are each padded to x32. A forced pad must use the model's
    learned ``x_pad_token`` because maskless attention would consume zero rows.
    """
    if sp > 1 and concat_len % sp != 0:
        raise UnsupportedModelError(
            f"zimage: joint sequence length {concat_len} is not divisible by the "
            f"sequence-parallel degree {sp}. Z-Image learned-pads caption and image each to "
            "a multiple of 32, so this is unreachable for sp in {2,4,8}; a forced pad must "
            "use the model's x_pad_token, never zeros: Z-Image is maskless, so a zero pad "
            "row would be attended."
        )


def _trim_context_or_raise(context):
    """Trim trailing CFG padding or refuse a ragged caption batch.

    ``cap_embedder`` maps zero rows to bias-valued keys, and maskless attention
    has no residual-mask fallback. Batch-1 CFG ranks trim exactly.
    """
    context, residual = _trim_trailing_pad(context)
    if residual is not None:
        raise UnsupportedModelError(
            "zimage: one model call batches captions of different real lengths, but Z-Image "
            "is maskless (cap_mask is hard-None), so the residual pad, bias-valued via "
            "cap_embedder, cannot be masked off. Equalize the prompt lengths or run on "
            "topology 'single'."
        )
    return context


def _reject_ming_image(x, direct_context, ref_frames) -> None:
    """Refuse Ming-Image inputs before any shape-specific code can misread them.

    ComfyUI 3b4c0b0e defines a 5-D `(B, C, F, H, W)` latent (the
    target frame, or a composite plus layers for Ming-Image-Layer), a
    `direct_context` caption extension, and `ref_frames` clean frames appended
    to the token axis after the target. Each is a global token span that does
    not survive sequence sharding, as with the omni/edit `ref_latents` that
    `_sharded_backbone` refuses. cfg-pad refuses them too instead of trimming:
    Ming-Image's `masked_pad_multiple` makes some encoder zero rows shift image
    positions, and the trim was validated only against Z-Image's own plain
    trailing padding. `x.ndim == 5` is checked first because a 5-D latent would
    otherwise fail an unrelated 4-tuple shape unpack before this message.
    """
    if x.ndim == 5 or direct_context is not None or len(ref_frames) > 0:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "zimage: this checkpoint carries Ming-Image conditioning (a frame axis on "
            "the latent, a direct-context caption extension, or reference frames "
            "appended to the token axis), which this build does not re-express, on "
            "either the sharded or the cfg-pad forward. Run this workflow on topology "
            "'single' (docs/MODELS.md).",
        ))


def _sharded_backbone(self, x, timesteps, context, num_tokens, attention_mask,
                      ref_latents, ref_contexts, siglip_feats, direct_context, ref_frames,
                      transformer_options, attn, kwargs):
    """Run replicated refiners and a sequence-sharded NextDiT joint block loop.

    Return the gathered stream and head inputs for latent or pixel decoding.
    """
    _reject_ming_image(x, direct_context, ref_frames)
    # Refuse omni/edit before ComfyUI import; ref_latents select that path.
    if len(ref_latents) > 0 or len(ref_contexts) > 0 or len(siglip_feats) > 0:
        raise UnsupportedModelError(
            "zimage: reference latents (omni/edit mode) modulate GLOBAL token spans via "
            "timestep_zero_index and batch-double the timesteps; neither survives sequence "
            "sharding. Run edit/omni renders on a cfg2 or single topology (docs/MODELS.md)."
        )
    if attention_mask is not None and not _mask_is_noop(attention_mask):
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                "zimage USP: this render carries an effective caption attention mask, "
                "but the maskless sharded backbone cannot apply it. Use mask-free "
                "conditioning. Prompts of unequal length take the exact trim path on "
                "pure cfg2; any other caption mask has no supported topology "
                "(docs/MODELS.md).",
            )
        )

    import comfy.ldm.common_dit

    t = 1.0 - timesteps
    cap_feats = context
    cap_mask = attention_mask
    _bs, _c, h, w = x.shape
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))

    t = self.t_embedder(t * self.time_scale, dtype=x.dtype)
    adaln_input = t
    if self.clip_text_pooled_proj is not None:
        # Z-Image leaves this unset; the shared NextDiT path supports it.
        pooled = kwargs.get("clip_text_pooled", None)
        if pooled is not None:
            pooled = self.clip_text_pooled_proj(pooled)
        else:
            pooled = torch.zeros((x.shape[0], self.clip_text_dim), device=x.device, dtype=x.dtype)
        adaln_input = self.time_text_embed(torch.cat((t, pooled), dim=-1))

    x_is_tensor = isinstance(x, torch.Tensor)
    # Use original options so replicated refiners keep stock local attention.
    img, mask, img_size, cap_size, freqs_cis, _timestep_zero_index = self.patchify_and_embed(
        x, cap_feats, cap_mask, adaln_input, num_tokens,
        ref_latents=ref_latents, ref_contexts=ref_contexts,
        siglip_feats=siglip_feats, transformer_options=transformer_options)
    freqs_cis = freqs_cis.to(img.device)

    # double_block patches use the global caption/image boundary, which a local
    # shard does not contain. Refiner patches have already run replicated.
    if "double_block" in transformer_options.get("patches", {}):
        raise UnsupportedModelError(
            "zimage: a 'double_block' patch addresses global caption/image token spans of "
            "the joint stream, which do not exist on a sequence-sharded rank. Run this "
            "workflow on a cfg2 or single topology."
        )

    sp = sp_world()
    _assert_concat_divisible(img.shape[1], sp)
    img, orig_len = shard_seq(img, dim=1)
    freqs_cis, _ = shard_seq(freqs_cis, dim=1)

    usp_opts = usp_options(transformer_options, attn)
    usp_opts["total_blocks"] = len(self.layers)
    usp_opts["block_type"] = "double"
    # With omni rejected, per-token modulation commutes with sharding.
    for i, layer in enumerate(self.layers):
        usp_opts["block_index"] = i
        img = layer(img, mask, freqs_cis, adaln_input, timestep_zero_index=None,
                    transformer_options=usp_opts)

    img = sp_gather(img, orig_len, dim=1)
    return img, adaln_input, img_size, cap_size, h, w, x_is_tensor, x


def _pixel_decode(img, cap_len, n_patches, dim, pixel_values, dec_net,
                  unpatchify_fn, img_size, x_is_tensor, h, w):
    """Decode real image patches from the gathered joint stream.

    ``dec_net`` runs one MLP per patch with ``B*N`` as batch. A zero caption
    placeholder preserves unpatchify indexing after learned padding is removed.
    """
    B = img.shape[0]
    img_hidden = img[:, cap_len:cap_len + n_patches, :]        # (B, N, dim)
    decoder_cond = img_hidden.reshape(B * n_patches, dim)       # (B*N, dim)
    output = dec_net(pixel_values, decoder_cond).reshape(B, n_patches, -1)  # (B, N, P^2*C)
    cap_placeholder = output.new_zeros(B, cap_len, output.shape[-1])
    full = torch.cat([cap_placeholder, output], dim=1)
    return -unpatchify_fn(full, img_size, [cap_len] * B, return_tensor=x_is_tensor)[:, :, :h, :w]


class ZImageAdapter(Adapter):
    """Bind latent and pixel-space Z-Image NextDiT variants."""

    family = "zimage"
    model_base_classes = (_ZIMAGE_LATENT_BASE, _ZIMAGE_PIXEL_BASE)
    model_detection_attrs = ("diffusion_model", "diffusion_model.pad_tokens_multiple")
    cfg_cond_padding = "pad"
    cfg_pad_restores_stock_call = True

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        pixel_cls = getattr(model_base, _ZIMAGE_PIXEL_BASE, None)
        if pixel_cls is not None and type(base_model) is pixel_cls:
            return True
        latent_cls = getattr(model_base, _ZIMAGE_LATENT_BASE, None)
        if latent_cls is not None and type(base_model) is latent_cls:
            # Lumina 2.0 shares this type and has no learned x32 pads.
            dm = getattr(base_model, "diffusion_model", None)
            if dm is not None and getattr(dm, "pad_tokens_multiple", None):
                return True
        return False

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        n_heads = getattr(diffusion_model, "n_heads", None)
        if n_heads is not None:
            _assert_ulysses_divides_heads(n_heads, _ulysses_degree())

        attn = ctx.usp_attention
        pixel_space = hasattr(diffusion_model, "dec_net")  # NextDiTPixelSpace deletes final_layer

        def latent_forward(self, x, timesteps, context, num_tokens, attention_mask=None,
                           ref_latents=[], ref_contexts=[], siglip_feats=[],
                           direct_context=None, ref_frames=[],
                           transformer_options={}, **kwargs):
            img, adaln_input, img_size, cap_size, h, w, x_is_tensor, _ = _sharded_backbone(
                self, x, timesteps, context, num_tokens, attention_mask,
                ref_latents, ref_contexts, siglip_feats, direct_context, ref_frames,
                transformer_options, attn, kwargs)
            img = self.final_layer(img, adaln_input, timestep_zero_index=None)
            img = self.unpatchify(img, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]
            return -img

        def pixel_forward(self, x, timesteps, context, num_tokens, attention_mask=None,
                          ref_latents=[], ref_contexts=[], siglip_feats=[],
                          transformer_options={}, **kwargs):
            # NextDiTPixelSpace._forward takes no direct_context or ref_frames
            # (Ming-Image has no pixel-space variant); pass None and [] so
            # _sharded_backbone's one guard covers both variants.
            img, _adaln_input, img_size, cap_size, h, w, x_is_tensor, x_padded = _sharded_backbone(
                self, x, timesteps, context, num_tokens, attention_mask,
                ref_latents, ref_contexts, siglip_feats, None, [],
                transformer_options, attn, kwargs)
            # Replicated pre-embed pixels, one token per patch: [B*N, 1, P^2*C].
            pH = pW = self.patch_size
            B, C, H, W = x_padded.shape
            pixel_patches = (
                x_padded.view(B, C, H // pH, pH, W // pW, pW)
                .permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2)
            )  # (B, N, pH*pW*C)
            n_patches = pixel_patches.shape[1]
            pixel_values = pixel_patches.reshape(B * n_patches, 1, pH * pW * C)
            return _pixel_decode(img, cap_size[0], n_patches, self.dim, pixel_values,
                                 self.dec_net, self.unpatchify, img_size, x_is_tensor, h, w)

        self.bind(diffusion_model, "_forward", pixel_forward if pixel_space else latent_forward)
        log.info("zimage USP injected: %d joint blocks, sp=%d, pixel_space=%s",
                 len(diffusion_model.layers), ctx.topology_sp, pixel_space)

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Install maskless CFG asymmetric-prompt support.

        Trim only driver-added trailing zero rows for batch-1 CFG ranks; refuse
        ragged batches. CFG ranks have ``sp == 1``, so the stock forward retains
        omni/edit behavior. Unpadded inputs, including ConditioningZeroOut, are
        unchanged.
        """
        stock_forward = type(diffusion_model)._forward  # class fn: idempotent, never the bind

        def cfg_pad_forward(self, x, timesteps, context, num_tokens, attention_mask=None,
                            direct_context=None, ref_frames=[],
                            transformer_options={}, **kwargs):
            _reject_ming_image(x, direct_context, ref_frames)
            context = _trim_context_or_raise(context)
            return stock_forward(self, x, timesteps, context, context.shape[1],
                                 attention_mask=attention_mask,
                                 direct_context=direct_context, ref_frames=ref_frames,
                                 transformer_options=transformer_options, **kwargs)

        self.bind(diffusion_model, "_forward", cfg_pad_forward)
        log.info("zimage cfg-pad forward installed")
