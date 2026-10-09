"""flux-family adapters: Chroma (+ChromaRadiance), Flux 1.x, Flux2, LongCat.

All of these run comfy's double-stream/single-stream block layout with pure
joint self-attention, so no attention forward is replaced: the sharded forward
threads the USP attention override through transformer_options and comfy's
own dispatch does the rest. Under pure Ulysses, Chroma also runs up to two
text projections per double block on full text rows: the Linear modules
`chroma_text._modules` admits, none on a checkpoint with quantization metadata.

RoPE under sharding is strictly per-token: every block loop embeds the id
shard that matches its own token shard (docs/ADAPTERS.md, "RoPE alignment
under sharding"). The double loop sees [txt_local, img_local] per rank, so
its pe is the two local id shards embedded side by side; the single loop sees
a shard of the re-concatenated [txt, img] stream, so its pe embeds the same
shard of the concatenated ids. The ids take the same pad and chunk as the
tokens they position. In the double loop, a chunk of the embedded full
[txt, img] sequence would hand a rank a contiguous run of the wrong positions
and corrupt every multi-rank render with no error.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .attention_patches import assert_usp_attention_patches_safe
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    _mask_is_noop,
    sp_gather,
    sp_rank,
    usp_options,
)
from .chroma_text import full_row_text_projections
from .flux_shared import (  # Other adapters import _control_span_overlap from here.
    PAD_FIDELITY_PROBE,
    _assert_ids_shard_with_tokens,
    _control_span_overlap,
    _family_drop_rows,
    _family_shard,
)
from .usp_sequence_order import joint_double_options

log = get_logger(__name__)


def _chroma_cfg_pad_trim(context: torch.Tensor, attention_mask):
    """Undo the cfg2 equalizer's trailing text pad and joint-key bias.

    `equalize_cond_lengths` (actor/sampling.py, rule "pad+mask") zero-pads
    each prompt's context to one length (the longest, aligned so the joint key
    count is a multiple of 8) and attaches an additive bias over the joint
    [text, image] keys: 0 on real text rows and every image key, the dtype
    minimum on pad rows. Each row's bias is a real-then-pad prefix, so its zero
    count is the row's real length. Trimming context to that length and
    dropping the bias hands stock's forward exactly the unpadded, maskless
    tensors a one-GPU call for that cond builds.

    A mask with a nonzero image column, or a text zone that is not a clean
    prefix, is not this construction (a conditioning that ships its own
    attention_mask skips the equalizer, sampling.py "already ships its own")
    and is returned unchanged. A batch whose rows disagree on the real length
    cannot be trimmed to one shape and is refused.
    """
    if attention_mask is None:
        return context, None
    text_len = context.shape[1]
    bias = attention_mask.reshape(attention_mask.shape[0], -1)
    if bias.shape[-1] < text_len:
        return context, attention_mask
    text_zone, image_zone = bias[:, :text_len], bias[:, text_len:]
    if image_zone.numel() and bool((image_zone != 0).any()):
        return context, attention_mask
    real = text_zone == 0
    lengths = real.sum(dim=1)
    is_prefix = real == (
        torch.arange(text_len, device=real.device).unsqueeze(0) < lengths.unsqueeze(1)
    )
    if not bool(is_prefix.all()):
        return context, attention_mask
    rows = int(lengths[0])
    if bool((lengths != rows).any()):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "chroma: this cfg-parallel rank's batch holds more than one real text "
            "length after the trailing pad is removed, so no single trim reproduces "
            "stock's unpadded call for every row. Use one prompt length across the "
            "image batch, or a non-cfg topology.",
        ))
    if rows == text_len:
        return context, None
    return context[:, :rows], None


class ChromaAdapter(Adapter):
    family = "chroma"
    model_base_classes = ("Chroma",)  # ChromaRadiance subclasses Chroma
    # Chroma's forward honors an additive attention mask over the joint
    # [text+image] keys, so asymmetric cfg-parallel prompts get pad + mask.
    # At cfg2, inject_cfg_pad_forward trims both back off before stock's
    # forward runs rather than letting it attend the bias.
    cfg_cond_padding = "pad+mask"
    usp_pad_exclusion_probe = "benchmark/reports/chroma_pad_fidelity_matrix.toml"
    cfg_pad_restores_stock_call = True

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        family = self.family
        pad_vouched = self.usp_pad_exclusion_probe is not None
        pure_ulysses = ctx.pure_ulysses

        def usp_forward_orig(self, img, img_ids, txt, txt_ids, timesteps, guidance=None,
                             control=None, transformer_options={}, attn_mask=None, **kwargs):
            assert_usp_attention_patches_safe(transformer_options, family)
            if attn_mask is not None and not _mask_is_noop(attn_mask):
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    "chroma: this render carries an effective attention mask, which "
                    "the sharded USP attention kernel cannot apply. Use cfg2 when the "
                    "graph batches cond and uncond, or a one-GPU local run "
                    "(mode=local, gpus_per_host=1; docs/MODELS.md).",
                ))

            from comfy.ldm.flux.layers import timestep_embedding

            patches_replace = transformer_options.get("patches_replace", {})

            img = self.img_in(img)

            # Distilled modulation vectors are per-batch, not per-token; no shard.
            mod_index_length = 344
            distill_timestep = timestep_embedding(timesteps.detach().clone(), 16).to(img.device, img.dtype)
            distill_guidance = timestep_embedding(guidance.detach().clone(), 16).to(img.device, img.dtype)
            modulation_index = timestep_embedding(
                torch.arange(mod_index_length, device=img.device), 32).to(img.device, img.dtype)
            modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
            timestep_guidance = torch.cat(
                [distill_timestep, distill_guidance], dim=1).unsqueeze(1).repeat(1, mod_index_length, 1)
            mod_vectors = self.distilled_guidance_layer(
                torch.cat([timestep_guidance.to(img.dtype), modulation_index], dim=-1))

            txt = self.txt_in(txt)

            img, img_orig_len = _family_shard(img, family, "image-token", pad_vouched=pad_vouched)
            txt, txt_orig_len = _family_shard(txt, family, "text-token", pad_vouched=pad_vouched)
            # Per-token RoPE (module docstring): this rank's own id shards,
            # embedded in [txt_local, img_local] order.
            txt_ids_local, _ = _family_shard(txt_ids, family, "text-position", pad_vouched=pad_vouched)
            img_ids_local, _ = _family_shard(img_ids, family, "image-position", pad_vouched=pad_vouched)
            _assert_ids_shard_with_tokens((txt_ids_local, txt), (img_ids_local, img))
            pe_double = self.pe_embedder(torch.cat((txt_ids_local, img_ids_local), dim=1))
            # Each rank contributes [txt_local, img_local] to the gathered
            # sequence, so the pad rows of both segments are named in that
            # order. The set is fixed in the override when it is built.
            usp_opts = joint_double_options(transformer_options, attn, _family_drop_rows(
                pad_vouched, [(txt_orig_len, txt.shape[1]), (img_orig_len, img.shape[1])]),
                ctx, txt.shape[1], img.shape[1])
            blocks_replace = patches_replace.get("dit", {})
            usp_opts["total_blocks"] = len(self.double_blocks)
            usp_opts["block_type"] = "double"
            for i, block in enumerate(self.double_blocks):
                usp_opts["block_index"] = i
                if i in self.skip_mmdit:
                    continue
                double_mod = (
                    self.get_modulations(mod_vectors, "double_img", idx=i),
                    self.get_modulations(mod_vectors, "double_txt", idx=i),
                )
                with full_row_text_projections(block, txt_orig_len, img_orig_len, pure_ulysses=pure_ulysses):
                    if ("double_block", i) in blocks_replace:
                        def block_wrap(args, block=block):
                            out_img, out_txt = block(img=args["img"], txt=args["txt"], vec=args["vec"],
                                                     pe=args["pe"], attn_mask=args.get("attn_mask"),
                                                     transformer_options=args.get("transformer_options"))
                            return {"img": out_img, "txt": out_txt}

                        out = blocks_replace[("double_block", i)](
                            {"img": img, "txt": txt, "vec": double_mod, "pe": pe_double,
                             "attn_mask": attn_mask, "transformer_options": usp_opts},
                            {"original_block": block_wrap})
                        img, txt = out["img"], out["txt"]
                    else:
                        img, txt = block(img=img, txt=txt, vec=double_mod, pe=pe_double,
                                         attn_mask=attn_mask, transformer_options=usp_opts)

                if control is not None:
                    control_i = control.get("input")
                    if i < len(control_i) and control_i[i] is not None:
                        add = control_i[i]
                        # Stock adds over global img rows [0, len); remap the
                        # span onto this rank's window (_control_span_overlap).
                        rows = img.shape[1]
                        row0 = sp_rank() * rows
                        ov = _control_span_overlap(row0, rows, 0, add.shape[1])
                        if ov is not None:
                            img[:, ov[0]:ov[1]] += add[:, ov[2]:ov[3]]

            img = sp_gather(img, img_orig_len, dim=1)
            txt = sp_gather(txt, txt_orig_len, dim=1)
            img = torch.cat((txt, img), 1)
            txt_len_full = txt.shape[1]
            # The same shard of the concatenated ids (module docstring).
            joint_ids_local, _ = _family_shard(
                torch.cat((txt_ids, img_ids), dim=1), family, "joint-position",
                pad_vouched=pad_vouched)
            pe_single = self.pe_embedder(joint_ids_local)
            img, joint_orig_len = _family_shard(img, family, "joint-token", pad_vouched=pad_vouched)
            _assert_ids_shard_with_tokens((joint_ids_local, img))
            # One joint segment, so its pads are a tail of the gathered
            # sequence, in an options dict of its own (one dict cannot carry two
            # drop sets). img_slice keeps stock's [txt, img] split of the full
            # stream (chroma/model.py), though the block tensors are shards.
            single_opts = usp_options(transformer_options, attn, drop_rows=_family_drop_rows(
                pad_vouched, [(joint_orig_len, img.shape[1])]))
            single_opts["img_slice"] = [txt_len_full, joint_orig_len]
            single_opts["total_blocks"] = len(self.single_blocks)
            single_opts["block_type"] = "single"
            for i, block in enumerate(self.single_blocks):
                single_opts["block_index"] = i
                if i in self.skip_dit:
                    continue
                single_mod = self.get_modulations(mod_vectors, "single", idx=i)
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args, block=block):
                        return {"img": block(args["img"], vec=args["vec"], pe=args["pe"],
                                             attn_mask=args.get("attn_mask"),
                                             transformer_options=args.get("transformer_options"))}

                    out = blocks_replace[("single_block", i)](
                        {"img": img, "vec": single_mod, "pe": pe_single,
                         "attn_mask": attn_mask, "transformer_options": single_opts},
                        {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_mod, pe=pe_single, attn_mask=attn_mask,
                                transformer_options=single_opts)

                if control is not None:
                    control_o = control.get("output")
                    if i < len(control_o) and control_o[i] is not None:
                        add = control_o[i]
                        # The control tensor covers global joint rows
                        # [txt_len_full, txt_len_full + len); this rank holds
                        # [row0, row0 + rows), so add the overlap in place.
                        rows = img.shape[1]
                        row0 = sp_rank() * rows
                        lo = max(row0, txt_len_full)
                        hi = min(row0 + rows, txt_len_full + add.shape[1])
                        if lo < hi:
                            img[:, lo - row0:hi - row0] += add[:, lo - txt_len_full:hi - txt_len_full]

            img = sp_gather(img, joint_orig_len, dim=1)
            img = img[:, txt_len_full:, ...]
            if hasattr(self, "final_layer"):
                final_mod = self.get_modulations(mod_vectors, "final")
                img = self.final_layer(img, vec=final_mod)
            return img

        self.bind(diffusion_model, "forward_orig", usp_forward_orig)
        log.info(
            "chroma USP injected: %d double + %d single blocks, sp=%d",
            len(diffusion_model.double_blocks), len(diffusion_model.single_blocks), ctx.topology_sp,
        )

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Remove the CFG equalizer's padding and mask before stock attention.

        At cfg2 with sp == 1, each rank runs the captured stock ``_forward`` on its
        own batch. Capturing the existing method covers both Chroma and
        ChromaRadiance without copying their block loops. Keeping the mask would
        select a different SDPA kernel than the unmasked single-GPU reference;
        trimming restores the reference's inputs and kernel choice.
        """
        stock_forward = diffusion_model._forward

        def cfg_pad_forward(self, x, timestep, context, guidance, control=None,
                            transformer_options={}, **kwargs):
            context, mask = _chroma_cfg_pad_trim(context, kwargs.get("attention_mask"))
            kwargs = dict(kwargs)
            if mask is None:
                kwargs.pop("attention_mask", None)
            else:
                kwargs["attention_mask"] = mask
            return stock_forward(x, timestep, context, guidance, control,
                                 transformer_options, **kwargs)

        self.bind(diffusion_model, "_forward", cfg_pad_forward)
        log.info("chroma cfg-pad forward installed")


class FluxAdapter(Adapter):
    """Bind the exact ``model_base.Flux`` type for dev, schnell, and inpaint.

    Chroma, Flux2, and LongCatImage also inherit Flux but have separate adapters
    registered first. Reject unknown subclasses so an upstream addition cannot
    silently inherit this forward.
    """

    family = "flux"
    model_base_classes = ("Flux",)
    # The single exact model_base type this adapter (or a subclass) binds.
    exact_model_base = "Flux"
    # Flux's blocks honor an additive attention mask over the joint keys
    # (comfy.ldm.flux.math.attention mask kwarg), so asymmetric cfg-parallel
    # prompts get pad + key mask. On Chroma that mask moved the call to another
    # SDPA kernel than stock's unmasked one. Distilled dev/schnell
    # run cfg 1.0 and skip cfg-parallel rows in the auto table anyway.
    cfg_cond_padding = "pad+mask"
    usp_pad_exclusion_probe = PAD_FIDELITY_PROBE   # inherited by flux2, longcat

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, self.exact_model_base, None)
        if cls is not None and type(base_model) is cls:
            return True
        # Reached only for a flux subclass no earlier registry entry claimed
        # (Chroma/Flux2/LongCat adapters all precede this root adapter).
        raise UnsupportedModelError(
            f"flux variant {type(base_model).__name__} is not in the dgx-monarch launch set "
            "(Flux 1.x, Flux2, LongCat-Image, Chroma; docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        family = self.family  # the bound forward's `self` is the model, not this adapter
        pad_vouched = self.usp_pad_exclusion_probe is not None

        def usp_forward_orig(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None,
                             control=None, timestep_zero_index=None, transformer_options={},
                             attn_mask=None, **kwargs):
            # Flux._forward appends Kontext/ref-latent tokens and their ids
            # before this call, so the shard covers them and _forward's
            # out[:, :img_tokens] trim sees the full gathered sequence.
            transformer_options = transformer_options.copy()
            assert_usp_attention_patches_safe(transformer_options, family)

            from comfy.ldm.flux.layers import timestep_embedding

            patches = transformer_options.get("patches", {})
            patches_replace = transformer_options.get("patches_replace", {})

            if timestep_zero_index is not None:
                raise UnsupportedModelError(
                    f"{family}: reference latents use the 'index_timestep_zero' method, "
                    "which modulates global [start, end) token spans; those spans do not "
                    "survive sequence sharding. Set the reference method (FluxKontext"
                    "MultiReferenceLatentMethod) to 'offset' or 'index', or run on a cfg2 or single topology."
                )
            if attn_mask is not None and not _mask_is_noop(attn_mask):
                # Only this sharded forward refuses: cfg-parallel and single run
                # the stock forward, which honors the mask (the pad+mask cond
                # rule this adapter declares).
                raise UnsupportedModelError(
                    f"{family}: this render carries an attention mask, which the "
                    "sharded USP attention kernel cannot apply. Run masked "
                    "workflows on a cfg2 or single topology (docs/MODELS.md)."
                )
            if img.ndim != 3 or txt.ndim != 3:
                raise ValueError("Input img and txt tensors must have 3 dimensions.")

            img = self.img_in(img)
            # vec is per-batch, never per-token; no shard. Flux 1.x blocks
            # self-modulate from it through their own img_mod/txt_mod Modulation
            # modules (nothing like Chroma's distilled mod-vector table exists
            # here); under global_modulation (Flux2) the ModulationOut tuples
            # are computed once below and copied through to every block.
            vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
            if self.params.guidance_embed:
                if guidance is not None:
                    vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
            if self.vector_in is not None:
                if y is None:
                    y = torch.zeros((img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype)
                vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])

            if self.txt_norm is not None:  # Flux2 branch: plain per-token RMSNorm
                txt = self.txt_norm(txt)
            txt = self.txt_in(txt)

            # post_input patches run PRE-shard on the full streams + ids, like
            # stock: they may resize both tokens and ids together, and the
            # shard below re-derives everything from the patched tensors.
            if "post_input" in patches:
                for p in patches["post_input"]:
                    out = p({"img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids,
                             "transformer_options": transformer_options})
                    img, txt = out["img"], out["txt"]
                    img_ids, txt_ids = out["img_ids"], out["txt_ids"]

            img, img_orig_len = _family_shard(img, family, "image-token", pad_vouched=pad_vouched)
            txt, txt_orig_len = _family_shard(txt, family, "text-token", pad_vouched=pad_vouched)
            if img_ids is not None:
                # Per-token RoPE from local id shards (module docstring).
                txt_ids_local, _ = _family_shard(txt_ids, family, "text-position", pad_vouched=pad_vouched)
                img_ids_local, _ = _family_shard(img_ids, family, "image-position", pad_vouched=pad_vouched)
                _assert_ids_shard_with_tokens((txt_ids_local, txt), (img_ids_local, img))
                pe_double = self.pe_embedder(torch.cat((txt_ids_local, img_ids_local), dim=1))
            else:
                pe_double = None
            # Pads sit in rank-major [txt_local, img_local]; pure Ulysses restores stock order.
            usp_opts = joint_double_options(transformer_options, attn, _family_drop_rows(
                pad_vouched, [(txt_orig_len, txt.shape[1]), (img_orig_len, img.shape[1])]),
                ctx, txt.shape[1], img.shape[1])

            vec_orig = vec
            if self.params.global_modulation:
                # Flux2 branch: double-block ModulationOut tuples computed once,
                # per-batch copy-through. Stock's txt mod input differs from
                # vec_orig only on the (rejected) timestep_zero_index path.
                vec = (self.double_stream_modulation_img(vec_orig),
                       self.double_stream_modulation_txt(vec_orig))

            blocks_replace = patches_replace.get("dit", {})
            usp_opts["total_blocks"] = len(self.double_blocks)
            usp_opts["block_type"] = "double"
            rows = img.shape[1]
            row0 = sp_rank() * rows
            for i, block in enumerate(self.double_blocks):
                usp_opts["block_index"] = i
                if ("double_block", i) in blocks_replace:
                    def block_wrap(args, block=block):
                        out_img, out_txt = block(img=args["img"], txt=args["txt"], vec=args["vec"],
                                                 pe=args["pe"], attn_mask=args.get("attn_mask"),
                                                 transformer_options=args.get("transformer_options"))
                        return {"img": out_img, "txt": out_txt}

                    out = blocks_replace[("double_block", i)](
                        {"img": img, "txt": txt, "vec": vec, "pe": pe_double,
                         "attn_mask": attn_mask, "transformer_options": usp_opts},
                        {"original_block": block_wrap})
                    img, txt = out["img"], out["txt"]
                else:
                    img, txt = block(img=img, txt=txt, vec=vec, pe=pe_double,
                                     attn_mask=attn_mask, transformer_options=usp_opts)

                if control is not None:  # ControlNet
                    control_i = control.get("input")
                    if i < len(control_i):
                        add = control_i[i]
                        if add is not None:
                            # Stock adds over global img rows [0, len); remap the
                            # span onto this rank's window.
                            ov = _control_span_overlap(row0, rows, 0, add.shape[1])
                            if ov is not None:
                                img[:, ov[0]:ov[1]] += add[:, ov[2]:ov[3]]

            if img.dtype == torch.float16:
                # Stock fp16 stabilization; elementwise, so shard-then-gather
                # commutes with it (byte-identical to stock placement).
                img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

            img = sp_gather(img, img_orig_len, dim=1)
            txt = sp_gather(txt, txt_orig_len, dim=1)

            img = torch.cat((txt, img), 1)
            txt_len_full = txt.shape[1]

            if self.params.global_modulation:  # Flux2 branch, per-batch copy-through
                single_vec, _ = self.single_stream_modulation(vec_orig)
            else:
                single_vec = vec_orig

            if img_ids is not None:
                # The same shard of the concatenated ids (module docstring).
                joint_ids_local, _ = _family_shard(
                    torch.cat((txt_ids, img_ids), dim=1), family, "joint-position",
                    pad_vouched=pad_vouched)
                pe_single = self.pe_embedder(joint_ids_local)
            else:
                pe_single = None
            img, joint_orig_len = _family_shard(img, family, "joint-token", pad_vouched=pad_vouched)
            if img_ids is not None:
                _assert_ids_shard_with_tokens((joint_ids_local, img))

            # As in ChromaAdapter: the pads are a tail, in options of their
            # own, and img_slice keeps stock's full-stream [txt, img] split.
            single_opts = usp_options(transformer_options, attn, drop_rows=_family_drop_rows(
                pad_vouched, [(joint_orig_len, img.shape[1])]))
            single_opts["img_slice"] = [txt_len_full, joint_orig_len]
            single_opts["total_blocks"] = len(self.single_blocks)
            single_opts["block_type"] = "single"
            rows = img.shape[1]
            row0 = sp_rank() * rows
            for i, block in enumerate(self.single_blocks):
                single_opts["block_index"] = i
                if ("single_block", i) in blocks_replace:
                    def block_wrap(args, block=block):
                        return {"img": block(args["img"], vec=args["vec"], pe=args["pe"],
                                             attn_mask=args.get("attn_mask"),
                                             transformer_options=args.get("transformer_options"))}

                    out = blocks_replace[("single_block", i)](
                        {"img": img, "vec": single_vec, "pe": pe_single,
                         "attn_mask": attn_mask, "transformer_options": single_opts},
                        {"original_block": block_wrap})
                    img = out["img"]
                else:
                    img = block(img, vec=single_vec, pe=pe_single, attn_mask=attn_mask,
                                transformer_options=single_opts)

                if control is not None:  # ControlNet
                    control_o = control.get("output")
                    if i < len(control_o):
                        add = control_o[i]
                        if add is not None:
                            # Stock adds over global joint rows
                            # [txt_len, txt_len + len), offset by the text block.
                            ov = _control_span_overlap(row0, rows, txt_len_full, add.shape[1])
                            if ov is not None:
                                img[:, ov[0]:ov[1]] += add[:, ov[2]:ov[3]]

            img = sp_gather(img, joint_orig_len, dim=1)
            img = img[:, txt_len_full:, ...]
            img = self.final_layer(img, vec_orig)
            return img

        self.bind(diffusion_model, "forward_orig", usp_forward_orig)
        log.info(
            "%s USP injected: %d double + %d single blocks, sp=%d",
            self.family, len(diffusion_model.double_blocks),
            len(diffusion_model.single_blocks), ctx.topology_sp,
        )


class Flux2Adapter(FluxAdapter):
    """Flux2 (~32B): the same stock forward. global_modulation and txt_norm
    are per-batch/per-token branches the parent forward mirrors inline. Split
    out only for the exact-type bind and the cond-padding rule."""

    family = "flux2"
    model_base_classes = ("Flux2",)
    exact_model_base = "Flux2"
    # comfy front-pads a flux2 text cond shorter than 512 tokens to 512
    # (model_base.Flux2.extra_conds), so cond+uncond concat for cfg-parallel
    # needs zero-pad only, no key mask.
    cfg_cond_padding = "pad"


class LongCatAdapter(FluxAdapter):
    """LongCat-Image: stock Flux forward end to end; what sets it apart is
    conditioning geometry (rope_options shift_t/y/x) applied model_base-side
    before Flux._forward builds img_ids, so it arrives here as ordinary ids
    and shards like any other."""

    family = "longcat"
    model_base_classes = ("LongCatImage",)
    exact_model_base = "LongCatImage"
    cfg_cond_padding = "pad+mask"
