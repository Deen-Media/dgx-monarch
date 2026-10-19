"""FLUX.3 Image Klein adapter: a two-phase sharded forward and stock's cond branch.

Stock FLUX.3 (comfy/ldm/flux3/model.py) runs five per-mode image blocks over
the image rows, joins ``[image, references, text]`` and runs 28 single blocks
over the joint stream, all 33 with one ``Block`` class whose modulation comes
as ``(start, end, ModulationOut)`` row spans. The bound ``_forward`` keeps
stock's replicated prelude, then shards each phase as one contiguous window of
rows: the image rows for the five image blocks, gathered and rejoined, then
the joint stream for the single blocks. Spans are clipped per rank
(adapters/flux3_packing.py), and Ulysses' head all-to-all concatenates the
windows in rank order, so every kernel sees stock's key order with no reorder
descriptor. Each of the 33 sharded blocks runs stock's ``Block.forward`` with
every fused-norm call launched at stock's row count for its span
(adapters/flux3_adaln_parity.py), because comfy-kitchen's fused AdaLN picks
its kernel by the call's rows. Divisibility pad rows are named as
``drop_rows`` on every topology; a padded ring or hybrid render refuses class
K before the first block. The final layer runs on the gathered image rows,
stock's call on stock's rows.

The ten early text and reference blocks stay in stock's ``preprocess_conds``,
replicated, and comfy calls them with no options, so ``inject_usp`` binds
their attention to the stock backend. ``prepare_base_model`` makes every
Worker take stock's ``extra_conds`` cache branch: ComfyUI runs prompts under
``torch.inference_mode()``, a Worker's sampling thread does not, and the
cache is stock's once-per-conditioning t = 0 precompute, nothing more.
docs/ADAPTERS.md has the family paragraph.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .attention_patches import assert_usp_attention_patches_safe
from .base import Adapter, InjectionContext, UnsupportedModelError, sp_gather, usp_options
from .flux3_adaln_parity import bind_adaln_parity
from .flux3_packing import (
    assert_joint_layout,
    joint_pad_count,
    refuse_indivisible_heads,
    refuse_padded_ring,
    shard_stream,
    ulysses_degree,
)
from .stock_attention import bind_stock_attention

log = get_logger(__name__)

# Exact ComfyUI model class; detect.py owns checkpoint discrimination.
_FLUX3_EXACT = ("Flux3",)


def require_sample_sigmas(transformer_options: dict):
    """The sampler's sigma table, which the step-distilled checkpoint reads."""
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "flux3: the step-distilled FLUX.3 Klein checkpoint reads the sampler's "
            "sigma schedule to embed the time each step jumps to, and this sampler "
            "path did not set it, so stock ComfyUI fails the same call. Use KSampler, "
            "KSampler Advanced or SamplerCustom (docs/MODELS.md).",
        ))
    return sigmas


def _normal(value):
    """An exact copy of an inference tensor; lists are walked, other values pass."""
    if torch.is_tensor(value) and value.is_inference():
        return value.clone()
    if isinstance(value, list):
        return [_normal(item) for item in value]
    return value


class Flux3Adapter(Adapter):
    """The exact ComfyUI Flux3 base, step-distilled and base checkpoints alike."""

    family = "flux3"
    model_base_classes = _FLUX3_EXACT
    exact_model_base_classes = _FLUX3_EXACT
    # Every text row is attended, and stock pads with real tokens at RoPE
    # position 1, so no driver pad is exact; an unequal pair dispatches.
    cfg_cond_padding = "none"
    attention_head_dim = 128

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, "Flux3", None)
        if cls is not None and type(base_model) is cls:
            return True
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"flux3 variant {type(base_model).__name__} is a Flux3 subclass this "
            "adapter has not vetted, so every topology refuses it. Use stock ComfyUI "
            "on one GPU for an unreviewed derivative, or, for a finetune that keeps "
            "the plain Flux3 forward, set family_adapter='flux3' on the Init node "
            "(docs/MODELS.md).",
        ))

    def prepare_base_model(self, base_model) -> None:
        """Run stock Flux3.extra_conds under inference mode, as stock ComfyUI does.

        GPUWorker._inject_for_topology calls this on every load path, before
        any extra_conds call of a sample, on every topology. The bind sits on
        the shared BaseModel instance, so LoRA clones and the baked hot-swap
        arm keep it, and a repeat bind replaces rather than stacks because it
        wraps the class function. Stock's own predicate still decides the
        branch: a flow-map model and hooked conds stay in the forward. Every
        inference tensor in the output is cloned out, so nothing downstream
        meets one outside inference mode. The first call per load logs which
        branch the forward will take.
        """
        stock = type(base_model).extra_conds
        reported: list[bool] = []

        def extra_conds(self, **kwargs):
            with torch.inference_mode():
                out = stock(self, **kwargs)
            for cond in out.values():
                if hasattr(cond, "cond"):
                    cond.cond = _normal(cond.cond)
            if not reported and ("txt_states" in out or "c_crossattn" in out):
                reported.append(True)
                log.info("flux3 cond branch: %s", "states given (extra_conds cache)"
                         if "txt_states" in out else "states computed in the forward")
            return out

        self.bind(base_model, "extra_conds", extra_conds)

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        refuse_indivisible_heads(diffusion_model.single_blocks[0].num_heads, ulysses_degree())

        def usp_forward(self, x, timestep, context=None, ref_latents=None,
                        transformer_options={}, txt_states=None, ref_states=None,
                        ref_shapes=None, **kwargs):
            assert_usp_attention_patches_safe(transformer_options, "flux3")
            import comfy.model_prefetch
            from comfy.ldm.flux.layers import timestep_embedding

            B, _, H, W = x.shape
            dtype, device = x.dtype, x.device
            hw = H * W
            # A. Replicated prelude, stock's lines and stock's methods.
            vec_target = None
            if self.flow_map:
                sample_sigmas = require_sample_sigmas(transformer_options)
                i = int((sample_sigmas - timestep[0]).abs().argmin())
                target = sample_sigmas[min(i + 1, sample_sigmas.shape[0] - 1)]
                vec_target = self.ref_time_in(timestep_embedding(
                    target.reshape(1).to(device=device, dtype=dtype), 256))
            if txt_states is None:
                txt_states, ref_states = self.preprocess_conds(
                    context, ref_latents, vec_target, transformer_options)
                ref_shapes = [ref.shape[-2:] for ref in ref_latents] if ref_latents else None
            transformer_options = transformer_options.copy()
            vec = self.time_in(timestep_embedding(
                torch.cat([timestep, timestep.new_zeros(1)]).to(dtype), 256))
            if vec_target is not None:
                vec = vec + vec_target
            vec_img, vec_zero = vec[:-1], vec[-1:]
            early = self.early_stream_modulations["image"](vec_img)[0]
            img = self.emb_in["image"](x.flatten(2).transpose(1, 2))
            parts = [(img, "image", self.image_ids(H, W, 0, device))]
            if ref_states is not None:
                parts.append((ref_states, "image_cond", self.ref_ids(ref_shapes, device)))
            parts.append((txt_states, "txt", self.text_ids(txt_states.shape[1], device)))
            spans, start = [], 0
            for tokens, mode, _ids in parts:
                spans.append((start, start + tokens.shape[1], self.single_stream_modulations[mode](
                    vec_img if mode == "image" else vec_zero)[0]))
                start += tokens.shape[1]
            ids = torch.cat([part_ids for _tokens, _mode, part_ids in parts])
            assert_joint_layout(parts, ids, spans, B, ref_shapes)
            # B. The five image blocks on a contiguous window of the image rows.
            image = shard_stream(img, parts[0][2], [(0, hw, early)], self.rope, "flux3 image")
            refuse_padded_ring(len(image.drop_rows), joint_pad_count(start))
            queue = comfy.model_prefetch.make_prefetch_queue(
                list(self.content_mode_blocks["image"]) + list(self.single_blocks),
                device, transformer_options)
            # Stock's image blocks get the options before the block keys are set.
            opts = usp_options(transformer_options, attn, drop_rows=image.drop_rows)
            h = image.tokens
            for block in self.content_mode_blocks["image"]:
                comfy.model_prefetch.prefetch_queue_pop(queue, device, block, dtype,
                                                        malloc_scope="block")
                h = block(h, image.spans, image.pe, opts, stock_norm_rows=image.norm_rows)
            img = sp_gather(h, image.orig, dim=1)
            # C. The single blocks on a contiguous window of the joint stream.
            joint = shard_stream(torch.cat([img] + [t for t, _m, _i in parts[1:]], dim=1),
                                 ids, spans, self.rope, "flux3 joint")
            del img, parts
            opts = usp_options(transformer_options, attn, drop_rows=joint.drop_rows)
            opts["total_blocks"], opts["block_type"] = len(self.single_blocks), "single"
            h = joint.tokens
            for index, block in enumerate(self.single_blocks):
                comfy.model_prefetch.prefetch_queue_pop(queue, device, block, dtype,
                                                        malloc_scope="block")
                opts["block_index"] = index
                h = block(h, joint.spans, joint.pe, opts, stock_norm_rows=joint.norm_rows)
            comfy.model_prefetch.prefetch_queue_pop(queue, device, None, malloc_scope="block")
            # D. Stock's final layer on the gathered image rows.
            h = sp_gather(h, joint.orig, dim=1)
            out = self.final_layer["image"](h[:, :hw], vec_img)
            return out.transpose(1, 2).reshape(B, self.in_channels, H, W)

        # comfy runs the early text and reference blocks from preprocess_conds
        # with no options, on every rank; their stock attention ignores a pin.
        for block in diffusion_model.txt_mode_blocks:
            bind_stock_attention(block, "flux3 text early block attention")
        for block in diffusion_model.content_mode_blocks["image_cond"]:
            bind_stock_attention(block, "flux3 reference early block attention")
        # Bind every block first; all() over a generator would stop at the first miss.
        sharded = [*diffusion_model.content_mode_blocks["image"], *diffusion_model.single_blocks]
        routable = [bind_adaln_parity(block) for block in sharded]
        if not all(routable):
            log.warning("flux3: stock Block.forward reads no modulated_norm global, so "
                        "the fused-norm parity route is off and a sharded render can "
                        "differ from one GPU by a few bf16 steps (docs/ADAPTERS.md)")
        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("flux3 USP injected: %d image blocks, %d single blocks, sp=%d",
                 len(diffusion_model.content_mode_blocks["image"]),
                 len(diffusion_model.single_blocks), ctx.topology_sp)
