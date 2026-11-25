"""Anima adapter for the MiniTrainDIT image denoiser.

The block grid is `(B, T, H, W, D)`. Sequence parallelism shards H and applies
the identical H-row shard to the 3D RoPE table; zero-padding is whole H rows,
which the block loop names, so pure Ulysses leaves them out of every softmax
exactly and ring or hybrid refuses them class K (ring_pad waiver), and the
gather trims them. Per-`(B, T)` time and adaLN tensors stay replicated because
they broadcast over H and W.

Only `blocks[i].self_attn.attn_op` routes through USP. Cross-attention keeps local
image q against replicated text k/v through cosmos' stock `torch_attention_op`,
which runs with no per-block kernel pin (adapters/stock_attention.py).
`llm_adapter` runs before this denoise path and must remain unbound,
unsharded, and outside any FSDP collective; under FSDP its parameters go in
``ignored_params`` (adapters/fsdp_islands.py).

``prepare_base_model`` makes every Worker take stock's text branch: stock
``Anima.extra_conds`` runs ``llm_adapter`` and the 512-row pad once per
conditioning only under ``torch.inference_mode()``, which ComfyUI enters
around a prompt and a Worker's sampling thread does not. Off that branch the
forward reruns them on every call, and a prompt pair whose raw rows differ
splits into two batch-1 calls where stock makes one batch-2 call.
docs/ADAPTERS.md has the paragraph.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from . import scale_rows
from .attention_inputs import (
    note_preferred_attention,
    refuse_unadopted_keywords,
    take_qkv,
)
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    padded_row_indices,
    shard_seq,
    sp_gather,
)
from .flux3 import _normal
from .stock_attention import stock_attention_options

log = get_logger(__name__)

# Which block Linears see the H-sharded grid, and in which layout. The
# cross-attention keys and values read the replicated text; the adaLN
# modulation reads the per-(B, T) time embedding.
_SCALE_ROLES = scale_rows.by_pattern([
    (r"blocks\.\d+\.(self_attn\.\w+|cross_attn\.(q_proj|output_proj))", "tokens"),
    (r"blocks\.\d+\.mlp\.\w+", "grid"),
])


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


def _pad_rows(t: int, h_orig: int, h_local: int, w: int) -> list[int]:
    """The gathered Ulysses coordinates of the zero H rows ``shard_seq`` added.

    Each rank flattens its ``(T, h_local, W)`` chunk t-major, so its tokens are
    T segments of ``h_local * W`` rows, each ending in that frame's padded H
    rows on the last rank or ranks.
    """
    return padded_row_indices([(h_orig * w, h_local * w)] * t)


def _make_usp_attn_op(attn, pads: list | None = None):
    """Adapt `(B, S, H, D)` q/k/v to USP and return `(B, S, H*D)`.

    The returned plain function is assigned to an instance attribute directly;
    method binding would add an invalid `self` argument. Comfy's cosmos
    `Attention.compute_attention` calls it outside `wrap_attn`, with q/k/v in
    `AttentionTensorContainer` and a `preferred_attention` keyword. The op
    takes the containers once and ignores the preference, because the Init
    `attention` widget and the dispatcher choose the kernel
    (adapters/attention_inputs.py). Any other keyword refuses with a typed
    class P refusal before a container is taken: the dispatcher drops an
    unknown keyword for an ordinary kernel, so forwarding it would let a
    ComfyUI change pass unapplied. ``pads[0]``, when set, holds the running
    forward's pad rows, which the attention excludes under pure Ulysses and
    refuses on ring.
    """

    def usp_attn_op(q, k, v, transformer_options={}, preferred_attention=None,
                    **unadopted):
        refuse_unadopted_keywords("anima self-attention", unadopted)
        note_preferred_attention("anima self-attention", preferred_attention)
        q, k, v = take_qkv(q, k, v)
        heads = q.shape[-2]
        q = q.transpose(1, 2)  # (B, S, H, D) -> (B, H, S, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        rows = pads[0] if pads is not None else None
        named = {"drop_rows": rows} if rows else {}
        return attn(q, k, v, heads=heads, skip_reshape=True, **named)  # -> (B, S, H*D)

    return usp_attn_op


class AnimaAdapter(Adapter):
    family = "anima"
    model_base_classes = ("Anima",)
    # The driver pads no conditioning for this family. Comfy's own 512-row pad
    # runs in extra_conds (prepare_base_model), before comfy decides whether
    # the two conditionings batch: a pair of at most 512 T5 ids folds into one
    # call, as on one GPU, and cfg-parallel slices it; only a pair whose padded
    # rows differ dispatches one cond per rank (adapters/cfg_dispatch.py).
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

    def prepare_base_model(self, base_model) -> None:
        """Run stock Anima.extra_conds under inference mode, as stock ComfyUI does.

        FLUX.3's bind (adapters/flux3.py), on this family's method, on every
        topology and every load path. Stock's own predicate still picks the
        branch, so the text leaves here at stock's 512 rows and folds as
        stock's does. Every inference tensor in the output is cloned out, and
        the first text call per load logs the branch.
        """
        stock = type(base_model).extra_conds
        reported: list[bool] = []

        def extra_conds(self, **kwargs):
            with torch.inference_mode():
                out = stock(self, **kwargs)
            for cond in out.values():
                if hasattr(cond, "cond"):
                    cond.cond = _normal(cond.cond)
            if not reported and "c_crossattn" in out:
                reported.append(True)
                log.info("anima text branch: %s", "computed in the forward"
                         if "t5xxl_ids" in out else "preprocessed once (extra_conds)")
            return out

        self.bind(base_model, "extra_conds", extra_conds)

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        # This forward's pad rows, set around its block loop and read by the
        # rebound self-attention op (one cell per bound model).
        pads: list[list[int] | None] = [None]

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
                # Stock cross-attention ignores a pinned kernel.
                "transformer_options": stock_attention_options(kwargs.get("transformer_options")),
            }

            # Preserve the stock fp32 residual stream for fp16 blocks.
            if x_B_T_H_W_D.dtype == torch.float16:
                x_B_T_H_W_D = x_B_T_H_W_D.float()

            # The zero H rows an odd patched height took to divide the shard are
            # real keys after the block's adaLN. Named on every topology, pure
            # Ulysses drops them and ring or hybrid refuses them (class K,
            # ring_pad waiver).
            # The same rows stay out of the shared nvfp4 activation scale: the
            # attention projections read the grid flattened t-major, one
            # segment per frame, and the MLP reads it as a grid, split on H.
            h_local = x_B_T_H_W_D.shape[2]
            h_real = scale_rows.real_count(h_orig, h_local)
            streams = {
                "tokens": scale_rows.Stream(((h_local * W, h_real * W),) * T),
                "grid": scale_rows.Stream(((h_local, h_real),), axis=-3),
            }
            # Restored after, so a forward re-entered from a patch keeps this one's rows named.
            saved_pads, pads[0] = pads[0], _pad_rows(T, h_orig, h_local, W) or None
            try:
                with scale_rows.real_rows(_SCALE_ROLES, streams):
                    for block in self.blocks:
                        x_B_T_H_W_D = block(x_B_T_H_W_D, t_emb_B_T_D, context, **block_kwargs)
            finally:
                pads[0] = saved_pads

            # The output head requires the full `(B, T, H, W, D)` grid.
            x_B_T_H_W_D = sp_gather(x_B_T_H_W_D, h_orig, dim=2)

            x_B_T_H_W_O = self.final_layer(
                x_B_T_H_W_D.to(context.dtype), t_emb_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
            return self.unpatchify(x_B_T_H_W_O)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]

        usp_attn_op = _make_usp_attn_op(attn, pads)
        for block in diffusion_model.blocks:
            # Cross-attention and llm_adapter stay stock and replicated.
            block.self_attn.attn_op = usp_attn_op
        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("anima USP injected: %d blocks, sp=%d", len(diffusion_model.blocks), ctx.topology_sp)
