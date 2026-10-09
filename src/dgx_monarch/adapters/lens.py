"""Microsoft Lens dual-stream MMDiT adapter.

Lens has no patchify: its image sequence is the `h*w` latent cells. Each block
attends jointly in `[image, text]` order while retaining separate streams. Image
and text are sharded independently, then each rank embeds its own ID shards in
that same order. Per-batch `temb` stays replicated, and only the image stream is
gathered for the output head.

xFuser's USP kernels take no key mask. An absent or all-ones text mask is
equivalent to no mask in real arithmetic, though not in bits: stock still
passes its all-zero bias, which moves torch's SDPA off flash. Ring and hybrid
trim a uniform trailing text pad, which is exact in real arithmetic because
text outputs are discarded and image RoPE does not depend on text length.
Ragged lengths or masked positions inside the kept text receive a typed
cfg2/single refusal on every topology. Divisibility pad rows are excluded in
`[image, text]` order through `drop_rows`; ring and hybrid padding retain the
shared class-K refusal and waiver contract. Under pure Ulysses,
`LensJointOrder` hands the kernel stock's key order, `_stock_key_bias` hands
comfy's SDPA stock's joint bias (a trailing pad stays, under its bias, as in
stock), and each block's text projections run on all text rows
(`_text_modules`).
comfy runs Lens in bf16 or fp32 only, because fp16 produces NaNs. Its 24 heads
of width 64 divide cleanly across the validated uly2 and ring2 layouts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..log import get_logger
from .base import (
    USP_ATTENTION_OVERRIDE_ATTR,
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


def _text_modules(block) -> tuple[tuple[Any, bool], ...]:
    """The text-stream projections whose GEMM depends on row count.

    Stock runs them on all text rows; a shard runs half, and the BLAS picks a
    kernel per row count. `to_add_out` reads the text slice of the joint
    attention output, so it takes the joint batch stride. The SwiGLU `txt_mlp`
    is wrapped whole, so its three Linears share one gather; the silu and the
    product between them are elementwise. Selection is `isinstance(module,
    torch.nn.Linear)` as for Chroma (see chroma_text._modules): plain fp8 and
    bf16 weights are wrapped, and quantized-metadata Linears are skipped.
    """
    attention = getattr(block, "attn", None)
    mlp = getattr(block, "txt_mlp", None)
    candidates = ((getattr(attention, "txt_qkv", None), False),
                  (getattr(attention, "to_add_out", None), True))
    selected: list[tuple[Any, bool]] = [
        (module, joint_stride) for module, joint_stride in candidates
        if isinstance(module, torch.nn.Linear)]
    if mlp is not None and all(isinstance(getattr(mlp, name, None), torch.nn.Linear)
                               for name in ("w1", "w2", "w3")):
        selected.append((mlp, False))
    return tuple(selected)


@dataclass(frozen=True)
class LensJointOrder(RankMajorJointOrder):
    """Stock's `[image, text]` key order at Ulysses' full-axis point.

    Each rank joins `[image_r, text_r]`, so the head all-to-all hands every rank
    `[image0, text0, image1, text1]`, while stock attends all image rows, then
    all text rows, and flash rounding depends on key order. `text_rows` and
    `image_rows` are this rank's rows per stream, pad included; the `*_real`
    fields are the lengths before the divisibility pad. The permutation lists
    the real rows in stock's order, then the pad rows: the kernel never sees a
    pad row, so its place does not change the math, and at the tail they pass
    the shared check that a query-bias group's pads trail every group. The
    inverse and the output restore are the parent's.
    """

    image_real: int
    text_real: int

    def canonicalize(self, q, k, v, drop_rows, kv_drop_rows, groups):
        """The parent's reorder, admitting only `_stock_key_bias`'s group.

        That group starts at row 0 and spans every real row, so it names the
        same queries in any order once the pads are gone. Any other group set
        would need its coordinates mapped, and the parent refuses it.
        """
        whole = ((0, self.image_real + self.text_real),)
        if groups and tuple((start, end) for start, end, _bias in groups) == whole:
            groups = None
        return super().canonicalize(q, k, v, drop_rows, kv_drop_rows, groups)

    def permutation(self, rows: int, device: torch.device) -> torch.Tensor:
        """Indices mapping this order to the rank-major gathered rows."""
        local = self.image_rows + self.text_rows
        world = rows // local if self.image_rows > 0 and self.text_rows >= 0 else 0
        if (world < 2 or rows != world * local
                or not 0 < self.image_real <= world * self.image_rows
                or not 0 <= self.text_real <= world * self.text_rows):
            self._refuse("the gathered sequence does not tile the local image/text pair")
        starts = torch.arange(world, device=device).unsqueeze(1) * local
        image = (starts + torch.arange(self.image_rows, device=device)).reshape(-1)
        text = (starts + self.image_rows
                + torch.arange(self.text_rows, device=device)).reshape(-1)
        return torch.cat((image[:self.image_real], text[:self.text_real],
                          image[self.image_real:], text[self.text_real:]))


def _joint_key_bias(keep, batch: int, image_rows: int, text_rows: int,
                    device) -> torch.Tensor:
    """Stock's joint additive key bias, (B, 1, 1, image + text) in float32.

    The values of comfy's `_build_joint_attention_mask` with stock's all-ones
    default for a missing mask. It is built here so the forward does not lean
    on that private method; the real-model tests compare the two outputs for exact equality.
    The float32 minimum becomes -inf in bf16, which is what stock attends
    under.
    """
    if keep is None:
        keep = torch.ones(batch, text_rows, dtype=torch.bool, device=device)
    keep = keep.reshape(keep.shape[0], text_rows).bool()
    image = torch.ones(keep.shape[0], image_rows, dtype=torch.bool, device=keep.device)
    joint = torch.cat((image, keep), dim=1)
    bias = torch.zeros_like(joint, dtype=torch.float32)
    bias.masked_fill_(~joint, torch.finfo(torch.float32).min)
    return bias[:, None, None, :]


def _stock_key_bias(override, joint_bias: torch.Tensor):
    """Give each joint attention stock's additive key bias.

    Stock Lens always builds its joint key mask, all zeros for an unmasked
    prompt, and hands it to comfy's attention, so torch's SDPA runs a masked
    backend (flash takes no mask). xFuser's kernels take no mask at all. One
    query group over every real row carries that bias to the full-axis point,
    where `usp_query_bias` runs comfy's own SDPA wrapper with it: stock's
    function, bias, dtype and key order, on this rank's half of the heads.
    Stock casts the bias to the query dtype per call, and so does this.
    """
    real_rows = joint_bias.shape[3]

    def biased(func, q, k, v, heads, **kwargs):
        bias = joint_bias.to(q.dtype)
        return override(func, q, k, v, heads,
                        query_bias_groups=((0, real_rows, bias),), **kwargs)

    setattr(biased, USP_ATTENTION_OVERRIDE_ATTR, True)
    return biased


def _local_pe_ids(img_ids: torch.Tensor, txt_ids: torch.Tensor) -> torch.Tensor:
    """Return this rank's separately sharded IDs in `[image, text]` order.

    IDs must be sharded before embedding; chunking full embedded IDs assigns
    later ranks the wrong positions.
    """
    img_local, _ = shard_seq(img_ids, dim=1)
    txt_local, _ = shard_seq(txt_ids, dim=1)
    return torch.cat((img_local, txt_local), dim=1)


def _usp_text_trim(attention_mask: torch.Tensor, text_seq_len: int) -> int:
    """Return the uniform real text length for maskless sharded blocks.

    The input is a text keep-mask. Uniform trailing padding can be removed
    exactly; ragged lengths or a hole inside the kept region raise a typed
    refusal on every topology, pure Ulysses included.
    """
    entries = attention_mask.shape[0]
    probe = attention_mask.reshape(entries, text_seq_len, 1).to(torch.float32)
    probe, residual = _trim_trailing_pad(probe)
    trim = probe.shape[1]
    if residual is not None or not bool(attention_mask[:, :trim].bool().all()):
        raise UnsupportedModelError(
            "lens: this render's text attention mask is not a uniform trailing pad "
            "(ragged real lengths across the batch, or a masked position inside the "
            "kept text), which the sharded USP attention kernel has no way to apply. "
            "Run masked / asymmetric-length-prompt workflows on a cfg2 or single "
            "topology (docs/MODELS.md)."
        )
    return trim


class LensAdapter(Adapter):
    """Bind only the exact validated `model_base.Lens` forward."""

    family = "lens"
    model_base_classes = ("Lens",)
    # Lens expects a text-length boolean keep-mask. The shared joint additive
    # mask has incompatible shape, order, and polarity, so cfg padding supplies
    # zero rows rather than a bias. inject_cfg_pad_forward says how a shipped
    # keep-mask follows those rows.
    cfg_cond_padding = "pad"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, "Lens", None)
        if cls is not None and type(base_model) is cls:
            return True
        # Refuse subclasses whose forward has not been validated.
        raise UnsupportedModelError(
            f"lens variant {type(base_model).__name__} is not in the dgx-monarch launch set, "
            "which holds plain Microsoft Lens t2i only. Render it in stock ComfyUI on one "
            "GPU, or, for a finetune that keeps the plain Lens forward, set "
            "family_adapter='lens' on the Init node (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        pure_ulysses = ctx.pure_ulysses

        def usp_forward(self, x, timestep, context, attention_mask=None,
                        transformer_options={}, control=None, **kwargs):
            # Shard both streams around the joint block loop; stock modules run
            # the embeddings, position IDs, blocks and output head.
            transformer_options = transformer_options.copy()
            patches = transformer_options.get("patches", {})
            blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})

            B, C, h, w = x.shape
            hidden_states = x.permute(0, 2, 3, 1).reshape(B, h * w, C)

            # Multi-layer text features are stacked along channels.
            if self.multi_layer_encoder_feature:
                L = len(self.selected_layer_index)
                enc_dim = context.shape[-1] // L
                encoder_hidden_states = list(context.reshape(B, -1, L, enc_dim).unbind(dim=2))
                text_seq_len = encoder_hidden_states[0].shape[1]
            else:
                encoder_hidden_states = context
                text_seq_len = context.shape[1]

            hidden_states = self.img_in(hidden_states)
            timestep = timestep.to(hidden_states.dtype)

            if self.multi_layer_encoder_feature:
                normed = [self.txt_norm[i](encoder_hidden_states[i]) for i in range(L)]
                encoder_hidden_states = torch.cat(normed, dim=-1)
            else:
                encoder_hidden_states = self.txt_norm(encoder_hidden_states)
            encoder_hidden_states = self.txt_in(encoder_hidden_states)

            if "post_input" in patches:
                # Apply stock patches before sharding either stream.
                for p in patches["post_input"]:
                    out = p({"img": hidden_states, "txt": encoder_hidden_states,
                             "transformer_options": transformer_options})
                    hidden_states = out["img"]
                    encoder_hidden_states = out["txt"]

            # Refuse mask semantics no topology here can carry. Pure Ulysses
            # keeps a uniform trailing pad, because stock attends those keys
            # under its bias and their count changes the kernel's bits; ring
            # and hybrid have no bias to carry, so they trim it.
            if attention_mask is not None:
                trim = _usp_text_trim(attention_mask, text_seq_len)
                if trim < text_seq_len and not pure_ulysses:
                    encoder_hidden_states = encoder_hidden_states[:, :trim]
                    text_seq_len = trim

            temb = self.time_text_embed(timestep, hidden_states)  # per-batch

            # Axial IDs follow the block's `[image, text]` joint order.
            from comfy.ldm.lens.model import _lens_position_ids

            ids = _lens_position_ids(1, h, w, text_seq_len, device=hidden_states.device).unsqueeze(0)
            img_len = h * w
            img_ids, txt_ids = ids[:, :img_len], ids[:, img_len:]

            # Shard streams separately, then embed this rank's local IDs.
            hidden_states, img_orig_len = shard_seq(hidden_states, dim=1)
            encoder_hidden_states, txt_orig_len = shard_seq(encoder_hidden_states, dim=1)
            freqs_cis = self.pos_embed(_local_pe_ids(img_ids, txt_ids))

            # Zero-pad rows are real maskless keys. Exclude them in the block's
            # `[image, text]` concatenation order.
            img_seg = (img_orig_len, hidden_states.shape[1])
            txt_seg = (txt_orig_len, encoder_hidden_states.shape[1])
            joint_drop = padded_row_indices([img_seg, txt_seg])
            # Restore stock key order and stock's key bias only where every
            # rank holds the whole key axis; ring and hybrid keep their path.
            order = None
            if pure_ulysses:
                order = LensJointOrder(
                    text_rows=txt_seg[1], image_rows=img_seg[1],
                    image_real=img_orig_len, text_real=txt_orig_len)

            usp_opts = usp_options(transformer_options, attn, drop_rows=joint_drop,
                                   sequence_order=order)
            if order is not None:
                usp_opts["optimized_attention_override"] = _stock_key_bias(
                    usp_opts["optimized_attention_override"],
                    _joint_key_bias(attention_mask, B, img_orig_len, txt_orig_len, x.device))
            usp_opts["total_blocks"] = len(self.transformer_blocks)
            usp_opts["block_type"] = "double"
            rows = hidden_states.shape[1]
            row0 = sp_rank() * rows
            for i, block in enumerate(self.transformer_blocks):
                usp_opts["block_index"] = i
                with full_row_text_projections(
                        block, txt_orig_len, img_orig_len,
                        pure_ulysses=pure_ulysses, select=_text_modules):
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args, block=block):
                            out = {}
                            out["txt"], out["img"] = block(
                                hidden_states=args["img"], encoder_hidden_states=args["txt"],
                                temb=args["vec"], freqs_cis=args["pe"],
                                attention_mask=args.get("attn_mask"),
                                transformer_options=args.get("transformer_options"))
                            return out

                        out = blocks_replace[("double_block", i)](
                            {"img": hidden_states, "txt": encoder_hidden_states, "vec": temb,
                             "pe": freqs_cis, "attn_mask": None, "transformer_options": usp_opts},
                            {"original_block": block_wrap})
                        hidden_states = out["img"]
                        encoder_hidden_states = out["txt"]
                    else:
                        encoder_hidden_states, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb, freqs_cis=freqs_cis, attention_mask=None,
                            transformer_options=usp_opts)

                if "double_block" in patches:
                    for p in patches["double_block"]:
                        out = p({"img": hidden_states, "txt": encoder_hidden_states, "x": x,
                                 "block_index": i, "transformer_options": usp_opts})
                        hidden_states = out["img"]
                        encoder_hidden_states = out["txt"]

                if control is not None:  # ControlNet
                    control_i = control.get("input")
                    if control_i is not None and i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            # Map the global image span `[0, add_len)` to this
                            # rank's image window; text remains separate.
                            ov = _control_span_overlap(row0, rows, 0, add.shape[1])
                            if ov is not None:
                                hidden_states[:, ov[0]:ov[1]] += add[:, ov[2]:ov[3]]

            # Gather the image stream before the head; txt is discarded.
            hidden_states = sp_gather(hidden_states, img_orig_len, dim=1)
            hidden_states = self.norm_out(hidden_states, temb)
            out = self.proj_out(hidden_states)
            return out.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()

        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("lens USP injected: %d blocks, sp=%d",
                 len(diffusion_model.transformer_blocks), ctx.topology_sp)

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Support asymmetric CFG prompts using Lens's stock text-mask contract.

        The worker equalizer pads text and extends its keep-mask with zeros.
        Lens publishes both as CONDRegular values, so matching shapes let ComfyUI
        batch them. Pass the resulting ``attention_mask`` to stock, which excludes
        the added keys from joint attention.

        Without a supplied mask, trim a uniform zero tail or convert residual
        ragged bias to Lens's boolean keep polarity (``bias == 0``). Unpadded and
        fully zeroed real conditioning stays intact. Call ``type(self)._forward``
        to avoid re-entering the instance wrapper.
        """

        def cfg_pad_forward(self, x, timestep, context, attention_mask=None,
                            transformer_options={}, control=None, **kwargs):
            if attention_mask is None:
                context, bias = _trim_trailing_pad(context)
                if bias is not None:
                    attention_mask = bias.squeeze(1) == 0
            return type(self)._forward(
                self, x, timestep, context, attention_mask=attention_mask,
                transformer_options=transformer_options, control=control, **kwargs)

        self.bind(diffusion_model, "_forward", cfg_pad_forward)
        log.info("lens cfg-pad forward installed")
