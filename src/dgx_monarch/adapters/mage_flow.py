"""Mage-Flow native-resolution MMDiT adapter.

Mage shares Qwen-Image's double-stream block class but not its token geometry:
text RoPE is the identity, image positions centre odd axes differently, and
the 128-channel Mage-VAE stream is unpatched. Keep this forward independent
so a future Qwen change cannot silently redefine Mage sharding.
"""
from __future__ import annotations

from typing import Any

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from . import mage_nvfp4_scale
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    _mask_is_noop,
    padded_row_indices,
    shard_seq,
    sp_gather,
    sp_rank,
    usp_options,
)
from .chroma_text import full_row_text_projections
from .flux_family import _control_span_overlap
from .krea2 import _trim_trailing_pad
from .usp_sequence_order import RankMajorJointOrder

log = get_logger(__name__)


def _dense(module) -> bool:
    """Whether this Linear's GEMM runs on unquantized weights.

    comfy's mixed-precision Linear is not a `torch.nn.Linear`. It names its
    quantized layout in `layout_type` and leaves it unset for a weight the
    checkpoint stores unquantized, such as the BF16 `to_add_out` of the MXFP8
    and NVFP4 native-view files or the BF16 layers of FP8 mixed and mixed
    NVFP4 files. Those take the same dense GEMM as a bf16 checkpoint.
    """
    if getattr(module, "layout_type", None) is not None:
        return False
    return isinstance(module, torch.nn.Linear) or isinstance(
        getattr(module, "weight", None), torch.Tensor)


def _text_modules(block) -> tuple[tuple[Any, bool], ...]:
    """Select full-row text projections for pure Ulysses.

    BLAS can choose different kernels and rounding at different row counts.
    Chroma projections matching the 3072-by-3072 ``add_*``/``to_add_out`` and
    12288-to-3072 ``net[2]`` shapes showed this sensitivity on GB10.
    ``net[0].proj`` also runs on all rows because that probe did not cover the shortest supported
    negative prompt. ``to_add_out`` retains the joint attention output's batch
    stride. Quantized Linears keep their shard; see ``_dense``.
    """
    attention = getattr(block, "attn", None)
    net: Any = getattr(getattr(block, "txt_mlp", None), "net", ())
    candidates = (
        (getattr(attention, "add_q_proj", None), False),
        (getattr(attention, "add_k_proj", None), False),
        (getattr(attention, "add_v_proj", None), False),
        (getattr(attention, "to_add_out", None), True),
        (getattr(net[0], "proj", None) if len(net) > 0 else None, False),
        (net[2] if len(net) > 2 else None, False),
    )
    return tuple((module, joint_stride) for module, joint_stride in candidates
                 if module is not None and _dense(module))


def _shared_gather():
    """An `sp_gather` that gathers each input tensor once.

    Stock hands the same `txt_modulated` tensor to `add_q_proj`, `add_k_proj`
    and `add_v_proj`, so one gather and one contiguous copy serve all three.
    The last input is held and compared by identity, not by `id()`: the block
    frees `txt_modulated` before it builds the MLP input, which could then
    reuse the address.
    """
    last: list[torch.Tensor] = []

    def gather(value: torch.Tensor, length: int, dim: int = 1) -> torch.Tensor:
        if last and last[0] is value:
            return last[1]
        gathered = sp_gather(value, length, dim=dim).contiguous()
        last[:] = [value, gathered]
        return gathered
    return gather


class MageFlowAdapter(Adapter):
    """The exact ComfyUI MageFlow base and its 24-head MMDiT forward."""

    family = "mage_flow"
    model_base_classes = ("MageFlow",)
    cfg_cond_padding = "pad"
    cfg_pad_restores_stock_call = True
    attention_head_dim = 128

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, "MageFlow", None)
        if cls is not None and type(base_model) is cls:
            return True
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"mage_flow variant {type(base_model).__name__} is not in the dgx-monarch "
            "launch set, so every topology refuses it. Use stock ComfyUI on one GPU for "
            "an unreviewed derivative, or, for a finetune that keeps the plain MageFlow "
            "forward, set family_adapter='mage_flow' on the Init node (docs/MODELS.md).",
        ))

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timestep, context, attention_mask=None, ref_latents=None,
                        transformer_options={}, control=None, **kwargs):
            # TextEncodeMageFlowEdit can carry the VLM's all-valid mask. It equals
            # no bias in real arithmetic, though not in bits: stock still passes
            # the all-zero bias to every block. Discard only that exact no-op
            # form; a real key bias remains a typed USP refusal.
            if attention_mask is not None and _mask_is_noop(attention_mask):
                attention_mask = None
            if attention_mask is not None:
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    "mage_flow: this render carries an encoder attention mask. The "
                    "sharded Ulysses kernel cannot preserve that text-key bias. Use "
                    "cfg2, or topology 'single' (mode=local with gpus_per_host=1).",
                ))

            hidden_states, img_ids, orig_shape = self.process_img(x)
            target_tokens = hidden_states.shape[1]
            if ref_latents is not None:
                ref_tokens = []
                for index, ref in enumerate(ref_latents, start=1):
                    reference, reference_ids, _ = self.process_img(ref, index=index)
                    hidden_states = torch.cat([hidden_states, reference], dim=1)
                    img_ids = torch.cat([img_ids, reference_ids], dim=1)
                    ref_tokens.append(reference.shape[1])
                transformer_options = transformer_options.copy()
                transformer_options["reference_image_num_tokens"] = ref_tokens

            # Mage leaves text unrotated (zero ids). Stock process_img builds the
            # image IDs, odd-axis centring included, before they are sharded and
            # embedded locally below.
            txt_ids = torch.zeros((x.shape[0], context.shape[1], 3), device=x.device)
            hidden_states = self.img_in(hidden_states)
            context = self.txt_in(self.txt_norm(context))
            temb = self.time_text_embed(timestep, hidden_states)

            patches = transformer_options.get("patches", {})
            blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
            if "post_input" in patches:
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    "mage_flow: a post_input patch rewrites global img/txt streams and "
                    "position IDs before sequence sharding. Run this workflow on topology "
                    "'single' (mode=local with gpus_per_host=1).",
                ))

            hidden_states, image_original = shard_seq(hidden_states, dim=1)
            context, text_original = shard_seq(context, dim=1)
            txt_ids, _ = shard_seq(txt_ids, dim=1)
            img_ids, _ = shard_seq(img_ids, dim=1)
            image_rotary_emb = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1)).contiguous()

            # Padding is per stream, then each rank joins [text, image]. The
            # rank-interleaved indices name those synthetic rows at Ulysses'
            # full-sequence attention point; the shared attention helper
            # refuses a padded ring or hybrid call.
            drop_rows = padded_row_indices([
                (text_original, context.shape[1]),
                (image_original, hidden_states.shape[1]),
            ])
            # xFuser gathers the locally paired streams rank-major as
            # [text0,image0,text1,image1]. Stock self-attention sees
            # [text0,text1,image0,image1]; restore that native order only at
            # pure Ulysses' full-axis attention point, including per-stream
            # padding coordinates and any appended reference image rows.
            sequence_order = (
                RankMajorJointOrder(context.shape[1], hidden_states.shape[1])
                if ctx.pure_ulysses else None
            )
            options = usp_options(transformer_options, attn, drop_rows=drop_rows,
                                  sequence_order=sequence_order)
            options["total_blocks"] = len(self.transformer_blocks)
            options["block_type"] = "double"
            rows = hidden_states.shape[1]
            row0 = sp_rank() * rows
            for index, block in enumerate(self.transformer_blocks):
                options["block_index"] = index
                # Pure Ulysses only: ring and hybrid keep their sharded GEMMs.
                with full_row_text_projections(
                        block, text_original, image_original,
                        pure_ulysses=ctx.pure_ulysses, select=_text_modules,
                        gather=_shared_gather()):
                    if ("double_block", index) in blocks_replace:
                        def block_wrap(args, block=block, text_original=text_original,
                                       image_original=image_original):
                            txt, img = mage_nvfp4_scale.call(
                                block, text_original=text_original, text_local=args["txt"].shape[1],
                                image_original=image_original, image_local=args["img"].shape[1],
                                rank=sp_rank(),
                                hidden_states=args["img"], encoder_hidden_states=args["txt"],
                                encoder_hidden_states_mask=None, temb=args["vec"],
                                image_rotary_emb=args["pe"],
                                transformer_options=args["transformer_options"],
                            )
                            return {"txt": txt, "img": img}
                        out = blocks_replace[("double_block", index)](
                            {"img": hidden_states, "txt": context, "vec": temb,
                             "pe": image_rotary_emb, "transformer_options": options},
                            {"original_block": block_wrap},
                        )
                        context, hidden_states = out["txt"], out["img"]
                    else:
                        context, hidden_states = mage_nvfp4_scale.call(
                            block, text_original=text_original, text_local=context.shape[1],
                            image_original=image_original, image_local=hidden_states.shape[1],
                            rank=sp_rank(),
                            hidden_states=hidden_states, encoder_hidden_states=context,
                            encoder_hidden_states_mask=None, temb=temb,
                            image_rotary_emb=image_rotary_emb, transformer_options=options,
                        )
                for patch in patches.get("double_block", ()):
                    out = patch({"img": hidden_states, "txt": context, "x": x,
                                 "block_index": index, "transformer_options": options})
                    hidden_states, context = out["img"], out["txt"]
                if control is not None:
                    additions = control.get("input")
                    if additions is not None and index < len(additions):
                        add = additions[index]
                        if add is not None:
                            overlap = _control_span_overlap(row0, rows, 0, add.shape[1])
                            if overlap is not None:
                                hidden_states[:, overlap[0]:overlap[1]] += add[:, overlap[2]:overlap[3]]

            hidden_states = sp_gather(hidden_states, image_original, dim=1)
            hidden_states = self.proj_out(self.norm_out(hidden_states, temb))
            h, w = orig_shape
            return hidden_states[:, :target_tokens].reshape(
                x.shape[0], h, w, self.out_channels).movedim(-1, 1)

        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("mage_flow USP injected: %d blocks, sp=%d", len(diffusion_model.transformer_blocks), ctx.topology_sp)

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Restore Mage's stock text call after cfg2 right-padding."""
        def cfg_pad_forward(self, x, timestep, context, attention_mask=None, ref_latents=None,
                            transformer_options={}, control=None, **kwargs):
            if attention_mask is None:
                context, bias = _trim_trailing_pad(context)
                if bias is not None:
                    attention_mask = bias.squeeze(1)
            return type(self)._forward(
                self, x, timestep, context, attention_mask=attention_mask,
                ref_latents=ref_latents, transformer_options=transformer_options,
                control=control, **kwargs,
            )

        self.bind(diffusion_model, "_forward", cfg_pad_forward)
        log.info("mage_flow cfg-pad forward installed")
