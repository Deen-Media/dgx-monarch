"""Kandinsky5 video and image sequence-parallel adapter.

Visual tokens and `(1, L, 1, ...)` RoPE frequencies take the same contiguous
sequence shard, then gather before the output reshape. Divisibility padding is
zero-pad plus trim. Odd-token testing stayed within the 0.10 one-step NRMS
threshold; see docs/VALIDATION.md for the evidence requirements.
Text blocks stay replicated; cross-attention uses local q against replicated
text k/v.

Self-attention is bound directly because decoder blocks share one options dict
between self- and cross-attention. The replacement also removes the exact
`>8192`-token chunk-project-concatenate q/k path because each rank already holds
fewer tokens.
For the 4096-dim video model, full-grid `h*w >= 14080` controls RoPE scaling, so
USP leaves `_forward` stock: it computes the frequencies from the full grid
before `forward_orig` shards them.
i2v conditioning is carried on x's channel axis and therefore follows x through
patchify and sequence sharding.
"""
from __future__ import annotations

from ..log import get_logger
from .base import Adapter, InjectionContext, UnsupportedModelError, shard_seq, sp_gather
from .krea2 import _trim_trailing_pad

log = get_logger(__name__)

# Only these exact model-base forwards have been validated. The image class
# subclasses Kandinsky5 and shares its forward.
_KANDINSKY5_SUPPORTED_EXACT = ("Kandinsky5", "Kandinsky5Image")


def _trim_context_or_raise(context):
    """Trim a uniform trailing text pad and reject ragged residual padding.

    Kandinsky5 has no mask input for its text blocks or cross-attention. cfg2
    places cond and uncond on separate ranks, so each local batch normally has
    one real length; any residual mask requirement receives a typed refusal.
    """
    context, residual = _trim_trailing_pad(context)
    if residual is not None:
        raise UnsupportedModelError(
            "kandinsky5: one model call batches conditionings with different real text "
            "lengths; the model is maskless, so the residual pad cannot be masked off. "
            "Equalize the prompt lengths or run this workflow on topology 'single'."
        )
    return context


class Kandinsky5Adapter(Adapter):
    family = "kandinsky5"
    model_base_classes = ("Kandinsky5",)  # Kandinsky5Image subclasses Kandinsky5
    exact_model_base_classes = _KANDINSKY5_SUPPORTED_EXACT
    # Asymmetric prompts use driver padding, then trim before biased text
    # embeddings because this model has no attention-mask path.
    cfg_cond_padding = "pad"
    cfg_pad_restores_stock_call = True

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in self.exact_model_base_classes:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"kandinsky5 variant {type(base_model).__name__} is not in the dgx-monarch "
            "launch set, which holds plain Kandinsky 5 video and image only. Render it in "
            "stock ComfyUI on one GPU, or, for a finetune that keeps the plain Kandinsky 5 "
            "forward, set family_adapter='kandinsky5' on the Init node (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_self_attention(self, x, freqs, transformer_options={}):
            # q/k use already-sharded per-token RoPE frequencies.
            from comfy.ldm.flux.math import apply_rope1

            b, s = x.shape[:2]
            n, d = self.num_heads, self.head_dim
            q = apply_rope1(self.query_norm(self.to_query(x).view(b, s, n, d)), freqs)
            k = apply_rope1(self.key_norm(self.to_key(x).view(b, s, n, d)), freqs)
            v = self.to_value(x).view(b, s, n * d)
            out = attn(q.view(b, s, n * d), k.view(b, s, n * d), v, heads=n)
            return self.out_layer(out)

        def usp_forward_orig(self, x, timestep, context, y, freqs, freqs_text,
                             transformer_options={}, **kwargs):
            # Shard visual embeddings and frequencies around the visual loop.
            # Text stays replicated, time_embed is per-batch, and gather must
            # precede the `(B, T, H, W, D)` reshape consumed by out_layer.
            patches_replace = transformer_options.get("patches_replace", {})
            context = self.text_embeddings(context)
            time_embed = self.time_embeddings(timestep, x.dtype) + self.pooled_text_embeddings(y)

            for block in self.text_transformer_blocks:
                context = block(context, time_embed, freqs_text, transformer_options=transformer_options)

            visual_embed = self.visual_embeddings(x)
            visual_shape = visual_embed.shape[:-1]
            visual_embed = visual_embed.flatten(1, -2)

            visual_embed, orig_len = shard_seq(visual_embed, dim=1)
            freqs, _ = shard_seq(freqs, dim=1)

            blocks_replace = patches_replace.get("dit", {})
            transformer_options["total_blocks"] = len(self.visual_transformer_blocks)
            transformer_options["block_type"] = "double"
            for i, block in enumerate(self.visual_transformer_blocks):
                transformer_options["block_index"] = i
                if ("double_block", i) in blocks_replace:
                    # The block parameters are positional visual/text embeddings.
                    def block_wrap(args, block=block):
                        return block(args["x"], args["context"], args["time_embed"],
                                     freqs=args["freqs"],
                                     transformer_options=args.get("transformer_options"))

                    visual_embed = blocks_replace[("double_block", i)](
                        {"x": visual_embed, "context": context, "time_embed": time_embed,
                         "freqs": freqs, "transformer_options": transformer_options},
                        {"original_block": block_wrap})["x"]
                else:
                    visual_embed = block(visual_embed, context, time_embed, freqs=freqs,
                                         transformer_options=transformer_options)

            visual_embed = sp_gather(visual_embed, orig_len, dim=1)

            visual_embed = visual_embed.reshape(*visual_shape, -1)
            return self.out_layer(visual_embed, time_embed)

        for block in diffusion_model.visual_transformer_blocks:
            self.bind(block.self_attention, "forward", usp_self_attention)
        self.bind(diffusion_model, "forward_orig", usp_forward_orig)
        log.info("kandinsky5 USP injected: %d visual blocks, sp=%d",
                 len(diffusion_model.visual_transformer_blocks), ctx.topology_sp)

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Install exact cfg2 handling for asymmetric prompt lengths.

        Trim the driver's trailing zero rows before biased text embeddings and
        rebuild 1D RoPE for the real `0..L-1` range. Ragged batches receive a
        typed refusal. Unpadded inputs, including fully zeroed real conditioning,
        remain untrimmed and take the stock path.
        """

        def cfg_pad_forward(self, x, timestep, context, y, time_dim_replace=None,
                            transformer_options={}, **kwargs):
            # Trim before building text RoPE so it uses the real text length.
            import comfy.ldm.common_dit

            context = _trim_context_or_raise(context)

            original_dims = x.ndim
            if original_dims == 4:
                x = x.unsqueeze(2)
            _b, _c, t_len, h, w = x.shape
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)

            if time_dim_replace is not None:
                time_dim_replace = comfy.ldm.common_dit.pad_to_patch_size(time_dim_replace, self.patch_size)
                x[:, :time_dim_replace.shape[1], :time_dim_replace.shape[2]] = time_dim_replace

            freqs = self.rope_encode_3d(t_len, h, w, device=x.device, dtype=x.dtype,
                                        transformer_options=transformer_options)
            freqs_text = self.rope_encode_1d(context.shape[1], device=x.device, dtype=x.dtype,
                                             transformer_options=transformer_options)

            out = self.forward_orig(x, timestep, context, y, freqs, freqs_text,
                                    transformer_options=transformer_options, **kwargs)
            if original_dims == 4:
                out = out.squeeze(2)
            return out

        self.bind(diffusion_model, "_forward", cfg_pad_forward)
        log.info("kandinsky5 cfg-pad forward installed")
