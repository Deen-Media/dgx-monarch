"""Ernie-Image adapter: 36 identical maskless single-stream blocks.

Sharding contract: tokens are the joint [image, text] sequence, image first
(stock ErnieImageModel.forward builds `cat([img, text], dim=1)`), with a
per-token 3-axis RoPE whose image temporal index equals the text length Tmax.
The rotary tensor is built at full length from the full joint ids (the fp64
rope path is per-position elementwise), then sharded with the same pad and
chunk as the tokens; never reindex it. adaLN is per-batch: six (B, 1, hidden)
chunks broadcast over the local shard, so it stays replicated. The sequence is
all-gathered before the final norm; the head keeps the first N_img tokens of
the full sequence.

`ErnieImageAttention.forward` calls optimized_attention with no
transformer_options kwarg, so the usp_options override never fires here; USP
is routed by binding each block's attention module forward instead (the
WanAdapter self-attention pattern). Divisibility padding is zero-pad + trim.
After modulation the zero pad row is a real key, so under pure Ulysses the
forward names it as a drop row and the full-axis call attends without it, as
stock attends no such key. Ring and hybrid keep attending it and do not
refuse, because they have no full-sequence point to drop it. Under pure
Ulysses the bound attention builds q and k on stock's own branch (the fused
norm-and-rope kernel at inference), and the first block norm and the final
norm read contiguous rows, as stock's do; ring and hybrid keep RMSNorm then
the rope and, at batch 2, the strided shard and gather views. docs/ADAPTERS.md
("Ernie-Image pad row and q/k branch") holds the measurements.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    padded_row_indices,
    shard_seq,
    sp_gather,
)

log = get_logger(__name__)


def _joint_rope_ids(bs: int, hp: int, wp: int, tmax: int, rope_options: dict | None, device) -> torch.Tensor:
    """Per-token 3-axis position ids for the joint [image, text] sequence.

    Mirrors stock `ErnieImageModel.forward`'s joint id construction (text ids
    then image ids), image first to match the token concat. Image tokens sit at
    temporal index Tmax (+shift_t); text tokens occupy temporal 0..Tmax-1 with
    zero spatial axes. Built at full length: the image index depends on the
    whole text length, so the ids cannot be assembled from a local text shard.
    """
    text_ids = torch.zeros((bs, tmax, 3), device=device, dtype=torch.float32)
    text_ids[:, :, 0] = torch.linspace(0, tmax - 1, steps=tmax, device=device, dtype=torch.float32)

    index = float(tmax)
    h_len, w_len = float(hp), float(wp)
    h_offset, w_offset = 0.0, 0.0
    if rope_options is not None:
        h_len = (h_len - 1.0) * rope_options.get("scale_y", 1.0) + 1.0
        w_len = (w_len - 1.0) * rope_options.get("scale_x", 1.0) + 1.0
        index += rope_options.get("shift_t", 0.0)
        h_offset += rope_options.get("shift_y", 0.0)
        w_offset += rope_options.get("shift_x", 0.0)

    image_ids = torch.zeros((hp, wp, 3), device=device, dtype=torch.float32)
    image_ids[:, :, 0] = image_ids[:, :, 0] + index
    image_ids[:, :, 1] = image_ids[:, :, 1] + torch.linspace(
        h_offset, h_len - 1 + h_offset, steps=hp, device=device, dtype=torch.float32).unsqueeze(1)
    image_ids[:, :, 2] = image_ids[:, :, 2] + torch.linspace(
        w_offset, w_len - 1 + w_offset, steps=wp, device=device, dtype=torch.float32).unsqueeze(0)
    image_ids = image_ids.reshape(1, hp * wp, 3).expand(bs, -1, -1)

    return torch.cat([image_ids, text_ids], dim=1)


def _attention_inputs(attention, x, rotary, *, fused: bool):
    """q, k and v for one block as stock's `ErnieImageAttention.forward` builds them.

    `fused` takes stock's inference branch (ComfyUI a7169322): with a rotary
    tensor and outside training, stock hands the cast norm weights to
    comfy-kitchen's fused `rms_rope_split_half`, whose own reduction can round
    differently on CUDA from RMSNorm followed by the rope. Otherwise this is
    stock's other branch, which ring and hybrid keep.
    """
    import comfy.quant_ops

    b, s, _ = x.shape
    query = attention.to_q(x).view(b, s, attention.heads, attention.head_dim)
    key = attention.to_k(x).view(b, s, attention.heads, attention.head_dim)
    value = attention.to_v(x)
    if fused and rotary is not None:
        import comfy.model_management

        if not comfy.model_management.in_training:
            import comfy.ops

            q_scale, _, q_offload_stream = comfy.ops.cast_bias_weight(
                attention.norm_q, query, offloadable=True)
            k_scale, _, k_offload_stream = comfy.ops.cast_bias_weight(
                attention.norm_k, key, offloadable=True)
            query, key = comfy.quant_ops.ck.rms_rope_split_half(
                query, key, rotary, q_scale, k_scale, attention.norm_q.eps)
            comfy.ops.uncast_bias_weight(attention.norm_q, q_scale, None, q_offload_stream)
            comfy.ops.uncast_bias_weight(attention.norm_k, k_scale, None, k_offload_stream)
            return query, key, value
    query = attention.norm_q(query)
    key = attention.norm_k(key)
    if rotary is not None:
        query, key = comfy.quant_ops.ck.apply_rope_split_half(query, key, rotary)
    return query, key, value


class ErnieAdapter(Adapter):
    family = "ernie"
    model_base_classes = ("ErnieImage",)
    # A driver pad for ragged cond/uncond prompts would be inexact here: the
    # image RoPE temporal index is the text length Tmax, so padding the shorter
    # conditioning shifts every image token's rotary position, and the 36
    # maskless blocks would attend the pad rows as real keys. Ragged prompts run
    # as separate calls instead, and cfg_dispatch gives each cfg rank its own
    # call. Ernie unequal-prompt fidelity has not been tested on hardware.
    cfg_cond_padding = "none"
    # Stock's forward builds no DIFFUSION_MODEL executor (cfg_parallel module docstring).
    cfg_split_seam = "apply_model"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        if type(base_model) is model_base.ErnieImage:
            return True
        raise UnsupportedModelError(
            f"ernie variant {type(base_model).__name__} subclasses ErnieImage with a "
            "forward this adapter has not vetted, so every topology refuses it. Render it "
            "in stock ComfyUI on one GPU, or, for a finetune that keeps the plain ErnieImage "
            "forward, set family_adapter='ernie' on the Init node (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        # The joint stream's pad rows for the forward now running. The stock
        # block calls attention with no options, so the forward that owns the
        # layer loop leaves them here for the bound attention to read.
        pads: list[list[int] | None] = [None]

        def usp_attention_forward(self, x, attention_mask=None, image_rotary_emb=None):
            # ErnieImageAttention.forward with the attention call routed
            # through USP; this bind is the only seam (module docstring). No
            # GQA: q, k and v all project to heads*head_dim, as stock. q and k
            # take per-token RoPE from the already sharded rotary tensor, so
            # every collective sits inside the xfuser attention call.
            if attention_mask is not None:
                raise UnsupportedModelError(
                    "ernie USP: a real attention mask reached ErnieImageAttention. Stock "
                    "comfy hardcodes None through the block loop and the sharded attention "
                    "kernel cannot honor a mask; run this workflow on topology 'single'."
                )
            b, s, _ = x.shape
            query, key, v_flat = _attention_inputs(self, x, image_rotary_emb,
                                                   fused=ctx.pure_ulysses)
            out = attn(query.reshape(b, s, -1), key.reshape(b, s, -1), v_flat, heads=self.heads,
                       drop_rows=pads[0])
            return self.to_out[0](out)

        def usp_forward(self, x, timesteps, context, **kwargs):
            # Shard and gather around the layer loop. The gather must precede
            # the final norm: the head slices the first N_img tokens of the
            # full sequence.
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None:
                # Stock never applies one (the cond exists but the layer loop
                # hardcodes None); if a future comfy threads a real mask here,
                # fail typed instead of silently mis-rendering.
                raise UnsupportedModelError(
                    "ernie USP: a real attention mask reached the forward. Stock comfy "
                    "never applies one and the sharded attention kernel cannot honor a "
                    "mask; run this workflow on topology 'single'."
                )

            device, dtype = x.device, x.dtype
            bs, _c, height, width = x.shape
            p = self.patch_size
            hp, wp = height // p, width // p
            n_img = hp * wp

            img_bsh = self.x_embedder(x)

            text_bth = context
            if self.text_proj is not None and text_bth.numel() > 0:
                text_bth = self.text_proj(text_bth)
            tmax = text_bth.shape[1]

            hidden_states = torch.cat([img_bsh, text_bth], dim=1)

            transformer_options = kwargs.get("transformer_options", {})
            ids = _joint_rope_ids(bs, hp, wp, tmax, transformer_options.get("rope_options", None), device)
            rotary_pos_emb = self.pos_embed(ids)
            del ids

            # Tokens and rotary take the same pad and chunk (module docstring).
            # Pad rows carry zero content and zero rotary; the gather trims them.
            hidden_states, orig_len = shard_seq(hidden_states, dim=1)
            rotary_pos_emb, _ = shard_seq(rotary_pos_emb, dim=1)
            if ctx.pure_ulysses:
                # Stock hands the fused kernel a contiguous rotary tensor. A
                # batch-2 chunk is a strided view, and comfy-kitchen's layout
                # check reads shapes, not strides.
                rotary_pos_emb = rotary_pos_emb.contiguous()
                # Stock's first block norm reads contiguous rows too.
                hidden_states = hidden_states.contiguous()

            sample = self.time_proj(timesteps).to(dtype)
            c = self.time_embedding(sample)

            # Per-batch adaLN, never sharded (module docstring).
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
                t.unsqueeze(1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)
            ]

            temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
            # Only pure Ulysses names the pad: ring and hybrid would refuse a
            # named row, and an even stream yields none and keeps xfuser's call.
            pads[0] = (padded_row_indices([(orig_len, hidden_states.shape[1])])
                       if ctx.pure_ulysses else None)
            try:
                for layer in self.layers:
                    hidden_states = layer(hidden_states, rotary_pos_emb, temb)
            finally:
                pads[0] = None

            hidden_states = sp_gather(hidden_states, orig_len, dim=1)
            if ctx.pure_ulysses:
                # The gather trims an odd stream's pad with a narrow, a strided
                # view at batch 2; stock's final norm reads contiguous rows.
                hidden_states = hidden_states.contiguous()

            hidden_states = self.final_norm(hidden_states, c).type_as(hidden_states)

            patches = self.final_linear(hidden_states)[:, :n_img, :]
            return (
                patches.view(bs, hp, wp, p, p, self.out_channels)
                .permute(0, 5, 1, 3, 2, 4)
                .contiguous()
                .view(bs, self.out_channels, height, width)
            )

        for layer in diffusion_model.layers:
            self.bind(layer.self_attention, "forward", usp_attention_forward)
        self.bind(diffusion_model, "forward", usp_forward)
        log.info("ernie USP injected: %d layers, sp=%d", len(diffusion_model.layers), ctx.topology_sp)
