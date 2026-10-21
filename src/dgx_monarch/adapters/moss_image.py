"""MOSS-Image adapter: one contiguous window per rank over stock's joint stream.

Stock MOSS-Image (comfy/ldm/moss_image/model.py) builds one rotary table for
the joint ``[image, text]`` stream, a time embedding, six shared block
modulations and a head pair, runs a two-block text embedder over the text
rows, joins the embedded image rows and the text rows, runs 42 identical
blocks over that stream and normalizes the image rows for the head. SFT, Base
and Kromoss share one class, one key set and one set of shapes, so one family
serves all three, and the adapter reads every size from the live module.

The bound ``_forward`` keeps every replicated step on stock's rows, on every
rank: the table, the embeddings, the modulations, the text embedder (on
ComfyUI's default attention backend with any pin dropped,
adapters/stock_attention.py) and the image embedding. It then cuts the joint
stream and its table into one contiguous window per rank
(adapters/moss_image_seams.py), so Ulysses' head all-to-all concatenates the
windows in stock's key order with no reorder descriptor. Each block runs
stock's own ``Block.forward`` code on its window, with both fused-norm calls
launched at stock's row count and its attention through the sharded override,
which drops the divisibility pad rows at the full-sequence point. A
``("single_block", i)`` replacement stock would call is honored when it is
exact on a window (an identity, which Skip Layer Guidance installs, or a
``sequence_shard_local`` callable) and refused before any compute otherwise.
The head norm runs after the gather, on stock's rows. The malloc graph is not
opened in the sharded loop (the FLUX.3 precedent); the prefetch queue and its
block-scope pops are kept. A padded ring or hybrid render refuses class K.

single, cfg2 and dp2 bind nothing and run stock's forward. No pad is exact on
cfg2 (image rows sit at RoPE f = T and the joint attention has no mask), so an
unequal prompt pair dispatches one cond per rank. ``cfg_dispatch.
cross_attn_pair_folds`` models CONDCrossAttn's repeat rule and overstates
MOSS-Image folding (CONDRegular folds only at equal shapes); it decides only
NVFP4's cfg scale, which this family refuses under sharding, so a later NVFP4
admission must revisit it. docs/ADAPTERS.md has the family paragraph.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .attention_inputs import STOCK_BACKEND_RUNS, note_preferred_attention
from .attention_patches import assert_no_foreign_attention_override, assert_usp_attention_patches_safe
from .base import Adapter, InjectionContext, UnsupportedModelError, sp_gather, usp_options
from .flux3_packing import joint_pad_count, refuse_indivisible_heads, refuse_padded_ring, ulysses_degree
from .moss_image_seams import (
    AttentionLedger,
    assert_stock_tree,
    bind_norm_parity,
    refuse_control_channels,
    refuse_reshaped_block_rows,
    refuse_unsharded_attention,
    rope_call,
    rope_with_options,
    shard_joint,
    vet_block_replacements,
)
from .stock_attention import stock_attention_options

log = get_logger(__name__)

# Exact ComfyUI model class; detect.py owns checkpoint discrimination.
_MOSS_EXACT = ("MOSSImage",)
_OVERRIDE = "optimized_attention_override"


class MOSSImageAdapter(Adapter):
    """The exact ComfyUI MOSSImage base: SFT, Base, Kromoss and same-layout finetunes."""

    family = "moss_image"
    model_base_classes = _MOSS_EXACT
    exact_model_base_classes = _MOSS_EXACT
    # Image rows sit at RoPE f = T and the joint attention has no mask, so no
    # driver pad is exact; an unequal pair dispatches.
    cfg_cond_padding = "none"
    attention_head_dim = 128
    sp_validated_quants = frozenset({"bf16", "int8"})
    sp_quant_refusal_reason = (
        "moss_image shards its joint stream with divisibility pad rows, and an NVFP4 file's "
        "shared activation scale is reduced over every row a rank holds, the pads included "
        "(only Mage-Flow's projections trim them, adapters/mage_nvfp4_scale.py), so an NVFP4 "
        "render would quantize with a scale one GPU never computes. On cfg2 nothing shards "
        "the token axis, but the same rule covers the split of each step's [cond, uncond] "
        "batch, which this release has not measured either. FP8 and MXFP8 have no "
        "MOSS-Image file and no sharded measurement, an fp8 weight_dtype cast on a BF16 file "
        "carries the same unmeasured kind, and this release reads an unrecognized "
        "quantization format as fp8, so only the two shipped kinds are admitted: BF16, and "
        "INT8 ConvRot, which takes no statistic across rows: comfy-kitchen's INT8 GEMM scales "
        "each activation row on its own, and every INT8 linear reaches that GEMM on a rank as "
        "on one GPU, ffn_down through its fused call and the others with their activation "
        "handed over as 2-D rows (adapters/int8_linear_layout.py)."
    )
    # The sharded forward applies patches_replace["dit"] as stock does, for
    # every patch the vetting admits; the Skip Layer Guidance carrier reads it.
    sp_honours_block_replace = True

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, "MOSSImage", None)
        if cls is not None and type(base_model) is cls:
            return True
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"moss_image variant {type(base_model).__name__} is a MOSSImage subclass this "
            "adapter has not vetted, so every topology refuses it. Use stock ComfyUI on one GPU "
            "for an unreviewed derivative, or, for a finetune that keeps the plain MOSSImage "
            "forward, set family_adapter='moss_image' on the Init node (docs/MODELS.md).",
        ))

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        assert_stock_tree(diffusion_model)
        refuse_indivisible_heads(diffusion_model.layers[0].num_heads, ulysses_degree(),
                                 family="moss_image",
                                 divisors="2, 4 or 8 for the 24-head MOSS-Image files")
        rope = rope_call(diffusion_model)
        attn = ctx.usp_attention

        def usp_forward(self, x, timestep, context, transformer_options={}, **kwargs):
            import comfy.ldm.moss_image.model as moss
            import comfy.model_prefetch
            from comfy.ldm.flux.layers import timestep_embedding

            # A. Guards: shapes, options and the tree only, identical on every rank.
            assert_no_foreign_attention_override(transformer_options, "moss_image")
            assert_usp_attention_patches_safe(transformer_options, "moss_image")
            B, C, H, W = x.shape
            refuse_control_channels(C, self.latent_channels)
            hw = H * W
            rows = hw + context.shape[1]
            vetted = vet_block_replacements(transformer_options, len(self.layers))
            refuse_padded_ring(0, joint_pad_count(rows))
            # B. The replicated prelude: stock's calls on stock's rows.
            dtype, device = x.dtype, x.device
            opts0 = transformer_options.copy()
            pe = rope(self, H, W, context.shape[1], device, opts0)
            vec = self.time_embedding(timestep_embedding(timestep, 256).to(dtype))
            mod = self.shared_block_modulation(vec).unflatten(-1, (6, -1)).unsqueeze(2).unbind(1)
            head_shift, head_scale = (
                self.video_head_modulation(vec).unflatten(-1, (2, -1)).unsqueeze(2).unbind(1))
            note_preferred_attention("moss_image block attention", self.layers[0].comfy_attention)
            for text_block in self.text_embedder.blocks:
                note_preferred_attention("moss_image text block attention",
                                         getattr(text_block, "comfy_attention", None),
                                         chooser=STOCK_BACKEND_RUNS)
            text_opts = stock_attention_options({key: value for key, value in opts0.items()
                                                 if key != _OVERRIDE})
            txt = self.text_embedder(context, pe[:, hw:], text_opts)
            img = self.video_embedder(x.flatten(2).transpose(1, 2))
            stream = torch.cat((img, txt), dim=1)
            del img, txt
            # C. One contiguous window per rank, the table cut alike.
            shard = shard_joint(stream, pe)
            del stream, pe
            # D. Block options: the sharded override with this call's pad rows.
            ledger = AttentionLedger()
            opts = usp_options(opts0, ledger.guard(attn), drop_rows=shard.drop_rows)
            opts["total_blocks"], opts["block_type"] = len(self.layers), "single"
            # E. Stock's block loop on the window, without the malloc graph.
            queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), device, opts)
            h = shard.tokens
            for index, block in enumerate(self.layers):
                comfy.model_prefetch.prefetch_queue_pop(queue, device, block, dtype,
                                                        malloc_scope="block")
                opts["block_index"] = index
                if index in vetted:
                    def wrap(args, _block=block, _index=index):
                        refuse_reshaped_block_rows(_index, args, B, shard.pe.shape[1])
                        ledger.started += 1
                        return {"img": _block(args["img"], args["mod"], args["pe"],
                                              args["transformer_options"], stock_norm_rows=rows)}

                    h = vetted[index]({"img": h, "vec": vec, "mod": mod, "pe": shard.pe,
                                       "transformer_options": opts}, {"original_block": wrap})["img"]
                else:
                    ledger.started += 1
                    h = block(h, mod, shard.pe, opts, stock_norm_rows=rows)
            comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")
            refuse_unsharded_attention(ledger.started, ledger.attended)
            # F. The gather, then stock's head on stock's rows.
            full = sp_gather(h, rows, dim=1).contiguous()
            out = self.final_linear_video(moss.modulated_rms_norm(
                full[:, :hw], self.final_norm_video, head_scale, head_shift))
            return out.transpose(1, 2).reshape(B, self.latent_channels, H, W)

        # Bind every block first; a generator would stop binding at the first miss.
        routable = [bind_norm_parity(block) for block in diffusion_model.layers]
        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("moss_image USP injected: %d blocks, sp=%d, rope=%s", len(routable),
                 ctx.topology_sp, "options" if rope is rope_with_options else "no options")
