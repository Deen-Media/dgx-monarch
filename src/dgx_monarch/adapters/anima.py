"""Anima adapter for the MiniTrainDIT image denoiser.

The block grid is `(B, T, H, W, D)`. Sequence parallelism shards H and applies
the identical H-row shard to the 3D RoPE table; zero-padding is whole H rows
followed by gather-time trim, with fidelity inside the documented benchmark
noise floor. Per-`(B, T)` time and adaLN tensors stay replicated because they
broadcast over H and W.

Only `blocks[i].self_attn.attn_op` routes through USP. Cross-attention keeps local
image q against replicated text k/v. `llm_adapter` runs before this denoise path
and must remain unbound, unsharded, and outside any FSDP collective; under FSDP
its parameters go in ``ignored_params`` (adapters/fsdp_islands.py).
"""
from __future__ import annotations

import torch

from ..log import get_logger
from .base import Adapter, InjectionContext, UnsupportedModelError, shard_seq, sp_gather

log = get_logger(__name__)


def _shard_rope_h(rope, t: int, h: int, w: int):
    """Shard `(T*H*W, head_dim/2, 2, 2)` RoPE rows on H.

    The table and block flattening are both t-major, h-mid, w-minor. Matching
    token padding is required; otherwise later ranks receive wrong positions.
    Padded H rows are removed after the token gather.
    """
    rest = rope.shape[1:]
    rope_thw = rope.reshape(t, h, w, *rest)
    rope_thw, _ = shard_seq(rope_thw, dim=1)
    return rope_thw.reshape(-1, *rest)


def _make_usp_attn_op(attn):
    """Adapt `(B, S, H, D)` q/k/v to USP and return `(B, S, H*D)`.

    The returned plain function is assigned to an instance attribute directly;
    method binding would add an invalid `self` argument.
    """

    def usp_attn_op(q, k, v, transformer_options={}):
        heads = q.shape[-2]
        q = q.transpose(1, 2)  # (B, S, H, D) -> (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return attn(q, k, v, heads=heads, skip_reshape=True)  # -> (B, S, H*D)

    return usp_attn_op


class AnimaAdapter(Adapter):
    family = "anima"
    model_base_classes = ("Anima",)
    # This family pads no conditioning of its own. Comfy's own 512-token pad
    # lands inside the forward, after comfy has decided whether the two
    # conditionings batch, so it equalizes nothing the cfg wrapper can see: a
    # prompt pair of different encoded length never shares a model call, and
    # cfg-parallel runs it through the per-cond dispatch (adapters/cfg_dispatch.py).
    cfg_cond_padding = "none"
    # These raw tensors have shape `(num_t5_tokens,)`, not a batch dimension.
    # extra_conds expands them to `(1, L)` on every rank; DP slicing them would
    # corrupt the token sequence and cause cross-rank divergence.
    dp_cond_exempt_keys = frozenset({"t5xxl_ids", "t5xxl_weights"})

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        # Exact type excludes sibling and future subclass forwards that have not
        # been validated for this sharding contract.
        if type(base_model) is model_base.Anima:
            return True
        raise UnsupportedModelError(
            f"anima variant {type(base_model).__name__} subclasses Anima with a forward "
            "this adapter has not vetted, so every topology refuses it. Render it in "
            "stock ComfyUI on one GPU, or, for a finetune that keeps the plain Anima "
            "forward, set family_adapter='anima' on the Init node (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
            # Patchify, RoPE construction, and the output head use the full grid;
            # only the block loop carries the H shard.
            import comfy.ldm.common_dit

            orig_shape = list(x.shape)
            x = comfy.ldm.common_dit.pad_to_patch_size(
                x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))

            # Build RoPE on full `(T, H, W)` before sharding it with the grid.
            x_B_T_H_W_D, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(
                x, fps=fps, padding_mask=padding_mask)

            timesteps_B_T = timesteps
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_emb_B_T_D, adaln_lora_B_T_3D = self.t_embedder[1](
                self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype))
            t_emb_B_T_D = self.t_embedding_norm(t_emb_B_T_D)

            # Keep T and the per-`(B, T)` embeddings whole; shard H and its RoPE.
            T, H, W = x_B_T_H_W_D.shape[1], x_B_T_H_W_D.shape[2], x_B_T_H_W_D.shape[3]
            x_B_T_H_W_D, h_orig = shard_seq(x_B_T_H_W_D, dim=2)
            rope_local = _shard_rope_h(rope_emb, T, H, W)
            if extra_pos_emb is not None:
                # Absolute position residuals must use the same H shard.
                extra_pos_emb, _ = shard_seq(extra_pos_emb, dim=2)
                assert x_B_T_H_W_D.shape == extra_pos_emb.shape

            block_kwargs = {
                "rope_emb_L_1_1_D": rope_local.unsqueeze(1).unsqueeze(0),
                "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
                "extra_per_block_pos_emb": extra_pos_emb,
                "transformer_options": kwargs.get("transformer_options", {}),
            }

            # Preserve the stock fp32 residual stream for fp16 blocks.
            if x_B_T_H_W_D.dtype == torch.float16:
                x_B_T_H_W_D = x_B_T_H_W_D.float()

            for block in self.blocks:
                x_B_T_H_W_D = block(x_B_T_H_W_D, t_emb_B_T_D, context, **block_kwargs)

            # The output head requires the full `(B, T, H, W, D)` grid.
            x_B_T_H_W_D = sp_gather(x_B_T_H_W_D, h_orig, dim=2)

            x_B_T_H_W_O = self.final_layer(
                x_B_T_H_W_D.to(context.dtype), t_emb_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
            return self.unpatchify(x_B_T_H_W_O)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]

        usp_attn_op = _make_usp_attn_op(attn)
        for block in diffusion_model.blocks:
            # Cross-attention and llm_adapter stay stock and replicated.
            block.self_attn.attn_op = usp_attn_op
        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("anima USP injected: %d blocks, sp=%d", len(diffusion_model.blocks), ctx.topology_sp)
