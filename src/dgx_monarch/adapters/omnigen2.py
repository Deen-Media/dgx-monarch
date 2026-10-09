"""Omnigen2 sequence-parallel adapter.

The masked text and image refiners remain replicated. Only the maskless main
stack shards the joined ``[text, ref, image]`` stream and its rotary embedding.
Gathering must precede normalization and the final image-tail slice.

The detected 21 query heads cannot be split by Ulysses degree 2, so the adapter
requires ring sequence parallelism. CFG prompts of different lengths are never
padded to one batch: caption length shifts image RoPE positions, and padding
would change those positions while adding unmasked keys. On cfg2 such a pair
takes per-condition dispatch instead (``cfg_batch_constant``), preserving each
caption's original positions.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    shard_seq,
    sp_gather,
    usp_options,
)

log = get_logger(__name__)

# Detected ComfyUI configuration. Ulysses must divide the query-head count.
_Q_HEADS = 21
_KV_HEADS = 7


def _assert_ring_only_topology(ulysses_degree: int) -> None:
    """Refuse Ulysses degrees that cannot divide the expanded query heads."""
    if ulysses_degree > 1 and _Q_HEADS % ulysses_degree != 0:
        raise UnsupportedModelError(
            f"omnigen2 has {_Q_HEADS} attention heads ({_KV_HEADS} kv), not divisible by the "
            f"requested ulysses degree {ulysses_degree}: the Ulysses all-to-all gives each rank "
            "an equal share of the heads. Use a ring topology (auto picks ring2), which shards "
            "the sequence and keeps every head whole (docs/MODELS.md)."
        )


def _assert_uniform_batch(seq_lengths) -> None:
    """Reject a ragged batch the maskless sharded main loop cannot serve.

    Shorter rows would require a key-padding mask that USP cannot carry.
    """
    lengths = [int(s) for s in seq_lengths]
    if lengths and min(lengths) != max(lengths):
        raise UnsupportedModelError(
            f"omnigen2 USP: this batch has non-uniform per-entry sequence lengths {lengths} "
            "(ragged ref-image or prompt sizes). The maskless sharded main loop cannot key-pad "
            "the shorter rows. Render such batches on topology 'single' (mode=local with "
            "gpus_per_host=1), or split them into uniform-size sub-batches (docs/MODELS.md)."
        )


def _requested_ulysses_degree() -> int:
    """Read the initialized Ulysses degree, defaulting to one if unavailable."""
    try:
        from xfuser.core.distributed import get_ulysses_parallel_world_size

        return int(get_ulysses_parallel_world_size())
    except Exception:
        return 1


class Omnigen2Adapter(Adapter):
    family = "omnigen2"
    model_base_classes = ("Omnigen2",)
    # num_tokens sets the image RoPE offset, so no exact padding rule exists
    # (module docstring).
    cfg_cond_padding = "none"
    # Stock OmniGen2 builds no DIFFUSION_MODEL executor. Install CFG splitting
    # on BaseModel.apply_model so it reaches the forward.
    cfg_split_seam = "apply_model"
    # extra_conds publishes the real caption length as a CONDConstant, so two
    # prompts of different real lengths arrive as two calls of batch one.
    cfg_batch_constant = "num_tokens"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        # Boogu subclasses Omnigen2 but has a different dual-stream forward.
        cls = getattr(model_base, "Omnigen2", None)
        if cls is not None and type(base_model) is cls:
            return True
        raise UnsupportedModelError(
            f"omnigen2 variant {type(base_model).__name__} is not in the dgx-monarch launch set, "
            "so every topology refuses it. The launch set holds plain Omnigen2 only (Boogu has "
            "its own adapter). Render it in stock ComfyUI on one GPU, or, for a finetune that "
            "keeps the plain Omnigen2 forward, set family_adapter='omnigen2' on the Init node "
            "(docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        _assert_ring_only_topology(_requested_ulysses_degree())
        attn = ctx.usp_attention

        def usp_forward(self, x, timesteps, context, num_tokens, ref_latents=None,
                        attention_mask=None, transformer_options={}, **kwargs):
            # Only the joined stream and its rotary enter the sharded main loop.
            import comfy.ldm.common_dit
            from einops import rearrange

            _B, _C, H, W = x.shape
            hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
            _, _, H_padded, W_padded = hidden_states.shape
            timestep = 1.0 - timesteps
            text_hidden_states = context
            text_attention_mask = attention_mask
            ref_image_hidden_states = ref_latents
            device = hidden_states.device

            temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

            (
                hidden_states, ref_image_hidden_states,
                img_mask, ref_img_mask,
                l_effective_ref_img_len, l_effective_img_len,
                ref_img_sizes, img_sizes,
            ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

            (
                context_rotary_emb, ref_img_rotary_emb, noise_rotary_emb,
                rotary_emb, _encoder_seq_lengths, seq_lengths,
            ) = self.rope_embedder(
                hidden_states.shape[0], text_hidden_states.shape[1], [num_tokens] * text_hidden_states.shape[0],
                l_effective_ref_img_len, l_effective_img_len,
                ref_img_sizes, img_sizes, device,
            )

            # Maskless attention requires one sequence length across the batch.
            _assert_uniform_batch(seq_lengths)

            # Refiners remain replicated so the text stage can apply its mask.
            for layer in self.context_refiner:
                text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb,
                                           transformer_options=transformer_options)

            img_len = hidden_states.shape[1]
            combined_img_hidden_states = self.img_patch_embed_and_refine(
                hidden_states, ref_image_hidden_states,
                img_mask, ref_img_mask,
                noise_rotary_emb, ref_img_rotary_emb,
                l_effective_ref_img_len, l_effective_img_len,
                temb,
                transformer_options=transformer_options,
            )

            hidden_states = torch.cat([text_hidden_states, combined_img_hidden_states], dim=1)

            # Shard the joined stream and rotary identically. temb is per-batch
            # and remains replicated. Omnigen2 is ring-only at world 2, and Ring
            # has no full-sequence point where synthetic pad rows can be excluded,
            # so refuse rather than silently attending zero keys.
            usp_opts = usp_options(transformer_options, attn)
            hidden_states, combined_len = shard_seq(
                hidden_states, dim=1, allow_padding=False,
                name="omnigen2 joined-token",
            )
            rotary_emb, _ = shard_seq(
                rotary_emb, dim=1, allow_padding=False,
                name="omnigen2 joined-rotary",
            )

            for layer in self.layers:
                hidden_states = layer(hidden_states, None, rotary_emb, temb, transformer_options=usp_opts)

            # Recover the full joined stream before keeping the image tail.
            hidden_states = sp_gather(hidden_states, combined_len, dim=1)
            hidden_states = self.norm_out(hidden_states, temb)

            p = self.patch_size
            output = rearrange(hidden_states[:, -img_len:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                               h=H_padded // p, w=W_padded // p, p1=p, p2=p)[:, :, :H, :W]
            return -output

        self.bind(diffusion_model, "forward", usp_forward)
        log.info("omnigen2 USP injected: %d layers, sp=%d", len(diffusion_model.layers), ctx.topology_sp)
