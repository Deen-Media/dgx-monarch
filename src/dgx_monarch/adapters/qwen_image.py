"""Qwen-Image dual-stream MMDiT adapter.

Each block joins text then image for self-attention. The two streams shard
separately, so RoPE must embed this rank's local IDs in that same order. The
per-batch time embedding remains replicated, and only image rows gather before
the output head.

Sequence parallelism refuses a real text mask because the sharded attention
cannot apply its additive bias. It also refuses `index_timestep_zero` reference
editing: that method splits modulation at a global image-row boundary that has
no rank-local equivalent. Standard index, negative-index, and offset references
only extend the image stream and remain supported. Comfy's own attention writes
`img_slice` from the local query tensors, so an attention patch here sees this
rank's rows; the flux-family single-block path publishes full-stream bounds.
"""
from __future__ import annotations

from typing import Any

import torch

from ..log import get_logger
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
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


# Each text-stream Linear's attribute path in a block, and whether it takes the
# joint batch stride.
_TEXT_PATHS = (
    ("attn.add_q_proj", False),
    ("attn.add_k_proj", False),
    ("attn.add_v_proj", False),
    ("attn.to_add_out", True),
    ("txt_mlp.net.0.proj", False),
    ("txt_mlp.net.2", False),
)
_warned_paths: set[str] = set()  # process lifetime: one warning per path


def _resolve(block, path: str) -> Any:
    node: Any = block
    for step in path.split("."):
        if step.isdigit():
            try:
                node = node[int(step)]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            node = getattr(node, step, None)
        if node is None:
            return None
    return node


def _text_modules(block) -> tuple[tuple[torch.nn.Linear, bool], ...]:
    """The text-stream Linears whose GEMM can depend on row count.

    Stock runs them on all text rows; a shard would run half. The BLAS can pick
    a different kernel for another row count, so the bits can differ.
    `to_add_out` reads a slice of the joint attention output, so it takes the
    joint batch stride. Selection is `isinstance(module, torch.nn.Linear)` as
    for Chroma (see chroma_text._modules): plain fp8 and bf16 weights are
    wrapped, and quantized-metadata Linears, which keep `in_features`, are
    skipped. Any other module, or none, at a path means ComfyUI changed the
    block layout. That Linear would then run on sharded rows with nothing
    logged, so a warning names the path, once per process.
    """
    selected: list[tuple[torch.nn.Linear, bool]] = []
    for path, joint_stride in _TEXT_PATHS:
        module = _resolve(block, path)
        if isinstance(module, torch.nn.Linear):
            selected.append((module, joint_stride))
        elif getattr(module, "in_features", None) is None and path not in _warned_paths:
            _warned_paths.add(path)
            log.warning(
                "qwen_image: block.%s is %s, not a Linear layer, so it runs on sharded "
                "text rows and pure-Ulysses output can differ from one GPU. ComfyUI "
                "changed the Qwen block layout that dgx-monarch selects by path.",
                path, "missing" if module is None else type(module).__name__)
    return tuple(selected)


class QwenImageAdapter(Adapter):
    """Qwen-Image dual-stream MMDiT, restricted to the exact base type."""

    family = "qwen_image"
    model_base_classes = ("QwenImage",)
    # Qwen expects a text-length mask, not the driver's joint Chroma mask.
    # The local cfg wrapper derives the native mask from zero-row padding.
    cfg_cond_padding = "pad"
    cfg_pad_restores_stock_call = True

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, "QwenImage", None)
        if cls is not None and type(base_model) is cls:
            return True
        # Future subclasses require a separate forward review.
        raise UnsupportedModelError(
            f"qwen_image variant {type(base_model).__name__} is not in the dgx-monarch "
            "launch set, which holds plain Qwen-Image t2i and index or offset reference "
            "editing only. Render it in stock ComfyUI on one GPU, or, for a finetune that "
            "keeps the plain Qwen-Image forward, set family_adapter='qwen_image' on the "
            "Init node (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timesteps, context, attention_mask=None, ref_latents=None,
                        additional_t_cond=None, transformer_options={}, control=None, **kwargs):
            # Shard both streams around the joint-attention block loop.
            if attention_mask is not None:
                raise UnsupportedModelError(
                    "qwen_image: this render carries an encoder_hidden_states_mask, which "
                    "the sharded USP attention kernel has no way to apply. Run masked / "
                    "asymmetric-length-prompt workflows on a cfg2 or single topology "
                    "(docs/MODELS.md)."
                )

            hidden_states, img_ids, orig_shape = self.process_img(x)
            num_embeds = hidden_states.shape[1]

            if ref_latents is not None:
                # Append references before sharding; gather before trimming them
                # from the output. Only global-boundary modulation is refused.
                ref_num_tokens = []
                h = w = index = 0
                ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
                index_ref_method = ref_method in ("index", "index_timestep_zero")
                negative_ref_method = ref_method == "negative_index"
                timestep_zero = ref_method == "index_timestep_zero"
                for ref in ref_latents:
                    if index_ref_method:
                        index += 1
                        h_offset = w_offset = 0
                    elif negative_ref_method:
                        index -= 1
                        h_offset = w_offset = 0
                    else:
                        index = 1
                        h_offset = w_offset = 0
                        if ref.shape[-2] + h > ref.shape[-1] + w:
                            w_offset = w
                        else:
                            h_offset = h
                        h = max(h, ref.shape[-2] + h_offset)
                        w = max(w, ref.shape[-1] + w_offset)
                    kontext, kontext_ids, _ = self.process_img(
                        ref, index=index, h_offset=h_offset, w_offset=w_offset)
                    hidden_states = torch.cat([hidden_states, kontext], dim=1)
                    img_ids = torch.cat([img_ids, kontext_ids], dim=1)
                    ref_num_tokens.append(kontext.shape[1])
                if timestep_zero and index > 0:
                    # The global modulation boundary has no rank-local index.
                    raise UnsupportedModelError(
                        "qwen_image: reference latents use the 'index_timestep_zero' method "
                        "(Qwen-Image-Edit 2511), which splits per-block modulation at a "
                        "global token index that does not survive sequence sharding. Switch "
                        "the reference method to 'index'/'offset', or run this workflow on a "
                        "cfg2/single topology (docs/MODELS.md)."
                    )
                transformer_options = transformer_options.copy()
                transformer_options["reference_image_num_tokens"] = ref_num_tokens

            txt_start = round(max(((x.shape[-1] + (self.patch_size // 2)) // self.patch_size) // 2,
                                  ((x.shape[-2] + (self.patch_size // 2)) // self.patch_size) // 2))
            txt_ids = torch.arange(
                txt_start, txt_start + context.shape[1], device=x.device).reshape(1, -1, 1).repeat(x.shape[0], 1, 3)

            hidden_states = self.img_in(hidden_states)
            encoder_hidden_states = self.txt_norm(context)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)

            # Per-batch, not per-token.
            temb = self.time_text_embed(timesteps, hidden_states, additional_t_cond)

            patches = transformer_options.get("patches", {})
            blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})

            if "post_input" in patches:
                # Patches may resize tokens and IDs together, so run them first.
                for p in patches["post_input"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states,
                             "img_ids": img_ids, "txt_ids": txt_ids,
                             "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]
                    img_ids = out["img_ids"]
                    txt_ids = out["txt_ids"]

            # Shard the two streams separately.
            hidden_states, img_orig_len = shard_seq(hidden_states, dim=1)
            encoder_hidden_states, txt_orig_len = shard_seq(encoder_hidden_states, dim=1)
            # Embed local IDs in the block's [text, image] joint order. Chunking
            # an embedded full sequence would assign later ranks wrong positions.
            txt_ids_local, _ = shard_seq(txt_ids, dim=1)
            img_ids_local, _ = shard_seq(img_ids, dim=1)
            image_rotary_emb = self.pe_embedder(
                torch.cat((txt_ids_local, img_ids_local), dim=1)).to(x.dtype).contiguous()

            rows = hidden_states.shape[1]
            # Each rank joins [text, image]; the head all-to-all gathers those
            # pairs rank-major, but stock attends [all text, all image] and
            # flash rounding depends on key order. Restore stock's order at
            # pure Ulysses' full-axis point. The zero pad each stream took to
            # divide the shard is a real key after modulation, so the same
            # point drops those rows. Ring and hybrid keep their path: they
            # cannot exclude pads, and naming drop rows there would refuse.
            sequence_order = drop_rows = None
            if ctx.pure_ulysses:
                sequence_order = RankMajorJointOrder(encoder_hidden_states.shape[1], rows)
                drop_rows = padded_row_indices([
                    (txt_orig_len, encoder_hidden_states.shape[1]),
                    (img_orig_len, rows),
                ])
            usp_opts = usp_options(transformer_options, attn, drop_rows=drop_rows,
                                   sequence_order=sequence_order)
            usp_opts["total_blocks"] = len(self.transformer_blocks)
            usp_opts["block_type"] = "double"
            row0 = sp_rank() * rows
            for i, block in enumerate(self.transformer_blocks):
                usp_opts["block_index"] = i
                with full_row_text_projections(
                        block, txt_orig_len, img_orig_len,
                        pure_ulysses=ctx.pure_ulysses, select=_text_modules):
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args, block=block):
                            out = {}
                            out["txt"], out["img"] = block(
                                hidden_states=args["img"], encoder_hidden_states=args["txt"],
                                encoder_hidden_states_mask=None, temb=args["vec"],
                                image_rotary_emb=args["pe"], timestep_zero_index=None,
                                transformer_options=args["transformer_options"])
                            return out

                        out = blocks_replace[("double_block", i)](
                            {"img": hidden_states, "txt": encoder_hidden_states, "vec": temb,
                             "pe": image_rotary_emb, "transformer_options": usp_opts},
                            {"original_block": block_wrap})
                        hidden_states = out["img"]
                        encoder_hidden_states = out["txt"]
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_hidden_states_mask=None,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            timestep_zero_index=None,
                            transformer_options=usp_opts,
                        )

                if "double_block" in patches:
                    for p in patches["double_block"]:
                        out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x,
                                 "block_index": i, "transformer_options": usp_opts})
                        hidden_states = out["img"]
                        encoder_hidden_states = out["txt"]

                if control is not None:  # ControlNet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            # Remap global image span [0, add_len) to this shard.
                            ov = _control_span_overlap(row0, rows, 0, add.shape[1])
                            if ov is not None:
                                hidden_states[:, ov[0]:ov[1]] += add[:, ov[2]:ov[3]]

            hidden_states = sp_gather(hidden_states, img_orig_len, dim=1)

            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            hidden_states = hidden_states[:, :num_embeds].view(
                orig_shape[0], orig_shape[-3], orig_shape[-2] // 2, orig_shape[-1] // 2,
                orig_shape[1], 2, 2)
            hidden_states = hidden_states.permute(0, 4, 1, 2, 5, 3, 6)
            return hidden_states.reshape(orig_shape)[:, :, :, :x.shape[-2], :x.shape[-1]]

        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("qwen_image USP injected: %d blocks, sp=%d",
                 len(diffusion_model.transformer_blocks), ctx.topology_sp)

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Mask cfg padding with Qwen's native text-length mask.

        A uniform trailing pad is trimmed for mask-free attention. Ragged input
        keeps a `(B, trim)` additive text bias. Fully zero conditioning remains
        real input, so it is neither trimmed nor masked. Calling the class method
        avoids re-entering this instance wrapper.
        """

        def cfg_pad_forward(self, x, timesteps, context, attention_mask=None, ref_latents=None,
                            additional_t_cond=None, transformer_options={}, control=None, **kwargs):
            if attention_mask is None:
                context, bias = _trim_trailing_pad(context)
                if bias is not None:
                    attention_mask = bias.squeeze(1)  # (B, 1, trim) -> (B, trim) text-key
            return type(self)._forward(
                self, x, timesteps, context, attention_mask=attention_mask,
                ref_latents=ref_latents, additional_t_cond=additional_t_cond,
                transformer_options=transformer_options, control=control, **kwargs)

        self.bind(diffusion_model, "_forward", cfg_pad_forward)
        log.info("qwen_image cfg-pad forward installed")
