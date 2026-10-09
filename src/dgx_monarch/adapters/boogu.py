"""Boogu adapter: an OmniGen2 derivative with an added dual-stream stage.

Four attention regimes run in sequence. The masked text refiner remains
replicated. Noise and reference-image refiners shard their separate streams.
Double-stream blocks keep instruction and image shards separate, with matching
local RoPE for joint and image-only attention. Both streams are then gathered,
joined, and reshared for the maskless single-stream blocks. Under pure Ulysses
every sharded loop drops its pad rows, the joint attention keeps stock's key
order, and the instruct Linears run on every instruct row
(adapters/boogu_ulysses.py).

Caption length sets the image RoPE offset. The sharded loops still need one
value for the whole batch, so the USP path keeps that contract. The CFG path
does not: ``inject_cfg_pad_forward`` reads each row's own caption length and
rebuilds the joint rope at the padded width.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    shard_seq,
    sp_gather,
)
from .boogu_ulysses import double_block_options, instruct_projections, stream_options
from .chroma_text import full_row_text_projections

log = get_logger(__name__)

# USP expands 7 K/V heads to 28 before Ulysses partitions the head axis.
_Q_HEADS = 28
_KV_HEADS = 7


def _split_joint_rope(rotary_emb: torch.Tensor, l_instruct: int):
    """Split ``[caption, reference, image]`` RoPE at the caption boundary."""
    return rotary_emb[:, :l_instruct], rotary_emb[:, l_instruct:]


def _assert_uniform_seq_lengths(seq_lengths) -> None:
    """Reject a ragged batch the maskless sharded loops cannot serve.

    Shorter rows need key padding, and one ``num_tokens`` cannot represent
    differing caption lengths.
    """
    lengths = [int(s) for s in seq_lengths]
    if lengths and min(lengths) != max(lengths):
        raise UnsupportedModelError(
            f"boogu USP: this batch has non-uniform per-entry sequence lengths {lengths} "
            "(ragged ref-image or prompt sizes). The maskless sharded loops cannot key-pad "
            "the shorter rows, and one `num_tokens` cannot carry differing caption lengths. "
            "Render such batches on topology 'single', or split them into uniform-size batches "
            "(docs/MODELS.md)."
        )


def _keep_mask_caption_lengths(attention_mask, width: int) -> list[int]:
    """Per-row real caption length read off boogu's text keep-mask.

    ComfyUI publishes this mask as ones on real tokens and zeros on padding,
    and sums it to make ``num_tokens``. Boogu numbers the caption 0..len-1 in
    stream order, so the kept positions have to be a prefix of the row: real
    tokens that start anywhere else have no single-GPU render to match.
    """
    keep = attention_mask
    if keep.ndim == 3 and keep.shape[1] == 1:
        keep = keep[:, 0]
    if keep.ndim != 2 or int(keep.shape[-1]) != width or bool((keep < 0).any()):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "boogu cfg-parallel: the text attention mask is not the keep-mask this "
            f"family publishes (shape {tuple(attention_mask.shape)} against {width} "
            "caption tokens, or negative entries where ComfyUI writes ones and zeros). "
            "Boogu reads that mask to place every image token's rope, so a mask in "
            "another convention cannot be honored. Run this workflow on topology "
            "'single' or 'uly2', or drop the node that rewrites the text mask "
            "(docs/MODELS.md).",
        ))
    lengths = keep.ne(0).sum(dim=1)
    positions = torch.arange(width, device=keep.device)
    if not bool(torch.equal(positions.unsqueeze(0) < lengths.unsqueeze(1), keep.ne(0))):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "boogu cfg-parallel: this render's text mask keeps tokens that are not a "
            "prefix of the caption (padding inside or ahead of the real tokens). Boogu "
            "numbers caption positions 0..n-1 in stream order and offsets every image "
            "token by that count, so a gapped caption has no single-GPU render to be "
            "exact against. Run this workflow on topology 'single' or 'uly2', or use "
            "a prompt whose padding is trailing (docs/MODELS.md).",
        ))
    return [int(value) for value in lengths]


def _content_caption_lengths(context) -> list[int]:
    """Per-row real caption length from trailing all-zero rows.

    The cfg cond equalization appends zero rows to the shorter prompt, and a
    conditioning whose every token is real ships no mask at all. An entry that
    is zero throughout stays full length, as krea2._trim_trailing_pad treats it.
    """
    entries, seq = context.shape[0], context.shape[1]
    content = context.reshape(entries, seq, -1).abs().amax(dim=2) > 0
    positions = torch.arange(seq, device=context.device)
    last = torch.where(
        content, positions.expand_as(content), positions.new_full((), -1)).amax(dim=1)
    cutoff = torch.where(last < 0, positions.new_full((), seq), last + 1)
    return [int(value) for value in cutoff]


def _text_pad_key_bias(cap_lens: list[int], width: int, image_len: int, dtype, device):
    """Additive bias dropping the padded caption columns from every text key.

    Shape ``(B, 1, 1, width + image_len)``: the joint and single-stream
    attentions broadcast it over heads and queries, and the text refiner takes
    its text half. Dropping those columns leaves the real tokens attending over
    exactly the unpadded key set, so each row matches its unpadded single-GPU render.
    """
    positions = torch.arange(width, device=device)
    lengths = torch.tensor(cap_lens, device=device, dtype=positions.dtype).unsqueeze(1)
    dropped = (positions.unsqueeze(0) >= lengths).to(dtype)
    bias = dropped * torch.finfo(dtype).min
    if image_len:
        bias = torch.cat([bias, bias.new_zeros(bias.shape[0], image_len)], dim=1)
    return bias[:, None, None, :]


def _padded_caption_forward(model, x, timesteps, context, cap_lens, ref_latents,
                            transformer_options):
    """Boogu's forward for a caption whose real tokens do not fill the stream.

    Stock boogu builds the joint rope at the caption's real length and then
    concatenates the padded text stream, so with a padded caption the two
    disagree in length and the stock forward cannot run. The stock rope
    embedder already accepts one caption length per row and zero-fills a
    caption slot wider than the real length, so the three padded pieces it
    returns concatenate into the joint rope the padded stream needs.
    """
    import comfy.ldm.common_dit
    from einops import rearrange

    _B, _C, H, W = x.shape
    hidden_states = comfy.ldm.common_dit.pad_to_patch_size(
        x, (model.patch_size, model.patch_size))
    _, _, H_padded, W_padded = hidden_states.shape
    device = hidden_states.device

    temb, text_hidden_states = model.time_caption_embed(
        1.0 - timesteps, context, hidden_states[0].dtype)
    width = int(text_hidden_states.shape[1])

    (
        flat_hidden_states, flat_ref_img_hidden_states, _img_mask, _ref_img_mask,
        l_effective_ref_img_len, l_effective_img_len, ref_img_sizes, img_sizes,
    ) = model.flat_and_pad_to_seq(hidden_states, ref_latents)

    (
        context_rotary_emb, ref_img_rotary_emb, noise_rotary_emb,
        _tight_joint, _caps, _seq_lengths,
    ) = model.rope_embedder(
        flat_hidden_states.shape[0], width, list(cap_lens),
        l_effective_ref_img_len, l_effective_img_len, ref_img_sizes, img_sizes, device,
    )
    rotary_emb = torch.cat(
        [context_rotary_emb, ref_img_rotary_emb, noise_rotary_emb], dim=1)

    img_len = flat_hidden_states.shape[1]
    combined_img_hidden_states = model.img_patch_embed_and_refine(
        flat_hidden_states, flat_ref_img_hidden_states, None, None,
        noise_rotary_emb, ref_img_rotary_emb,
        l_effective_ref_img_len, l_effective_img_len, temb,
        transformer_options=transformer_options,
    )

    key_bias = _text_pad_key_bias(
        cap_lens, width, int(combined_img_hidden_states.shape[1]),
        combined_img_hidden_states.dtype, device)
    # The refiner sees text keys only, so it takes the text half of the bias.
    for layer in model.context_refiner:
        text_hidden_states = layer(
            text_hidden_states, key_bias[..., :width], context_rotary_emb,
            transformer_options=transformer_options)

    combined_img_rotary_emb = rotary_emb[:, width:]
    for layer in model.double_stream_layers:
        combined_img_hidden_states, text_hidden_states = layer(
            combined_img_hidden_states, text_hidden_states,
            rotary_emb, combined_img_rotary_emb, temb,
            joint_attention_mask=key_bias, img_attention_mask=None,
            transformer_options=transformer_options,
        )

    hidden_states = torch.cat([text_hidden_states, combined_img_hidden_states], dim=1)
    for layer in model.single_stream_layers:
        hidden_states = layer(hidden_states, key_bias, rotary_emb, temb,
                              transformer_options=transformer_options)

    hidden_states = model.norm_out(hidden_states, temb)
    p = model.patch_size
    output = rearrange(
        hidden_states[:, -img_len:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
        h=H_padded // p, w=W_padded // p, p1=p, p2=p)[:, :, :H, :W]
    return -output  # The model contract negates the prediction.


class BooguAdapter(Adapter):
    """Boogu-Image (~10B), a dual-stream OmniGen2 derivative.

    Exact-type matching prevents this dual-stream forward from reaching the
    single-stack Omnigen2 adapter. Registry order must keep Boogu first.
    """

    family = "boogu"
    model_base_classes = ("Boogu",)
    # Subclasses require an explicit compatibility review.
    exact_model_base = "Boogu"
    # Zero rows plus the shipped keep-mask stretched to match. The padded rows
    # carry no rope of their own and are dropped from every text key, so each
    # row keeps the image rope offset of its own caption length.
    cfg_cond_padding = "pad"
    cfg_pad_restores_stock_call = True
    # Stock Boogu builds no DIFFUSION_MODEL executor. Install CFG splitting on
    # BaseModel.apply_model so it reaches the forward.
    cfg_split_seam = "apply_model"
    # Boogu inherits Omnigen2.extra_conds, which publishes the real caption
    # length as a CONDConstant. Two prompts of different real lengths therefore
    # never enter one batched call, whatever the pad rule does to the rows.
    cfg_batch_constant = "num_tokens"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, self.exact_model_base, None)
        if cls is not None and type(base_model) is cls:
            return True
        # A new subclass may change the forward contract.
        raise UnsupportedModelError(
            f"boogu variant {type(base_model).__name__} is not in the dgx-monarch launch set "
            "(plain Boogu-Image only). Run it on topology 'single' (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timesteps, context, num_tokens, ref_latents=None,
                        attention_mask=None, transformer_options={}, **kwargs):
            import comfy.ldm.common_dit
            import comfy.model_management
            from einops import rearrange

            _B, _C, H, W = x.shape
            hidden_states = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
            _, _, H_padded, W_padded = hidden_states.shape
            timestep = 1.0 - timesteps
            text_hidden_states = context
            text_attention_mask = attention_mask
            ref_image_hidden_states = ref_latents
            device = hidden_states.device

            temb, text_hidden_states = self.time_caption_embed(
                timestep, text_hidden_states, hidden_states[0].dtype)

            (
                flat_hidden_states, flat_ref_img_hidden_states,
                _img_mask, _ref_img_mask,
                l_effective_ref_img_len, l_effective_img_len,
                ref_img_sizes, img_sizes,
            ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

            (
                context_rotary_emb, ref_img_rotary_emb, noise_rotary_emb,
                rotary_emb, _encoder_seq_lengths, seq_lengths,
            ) = self.rope_embedder(
                flat_hidden_states.shape[0], text_hidden_states.shape[1],
                [num_tokens] * text_hidden_states.shape[0],
                l_effective_ref_img_len, l_effective_img_len,
                ref_img_sizes, img_sizes, device,
            )

            L_instruct = text_hidden_states.shape[1]
            # num_tokens and the caption boundary must select the same RoPE split.
            if int(num_tokens) != L_instruct:
                raise UnsupportedModelError(
                    f"boogu USP: num_tokens ({int(num_tokens)}) != caption length ({L_instruct}). "
                    "Boogu places image tokens right after `num_tokens` caption positions in the "
                    "joint rope; a padded or masked caption where these differ mis-slices the image "
                    "rope under sharding. Run this workflow on a cfg2 or single topology "
                    "(docs/MODELS.md)."
                )
            # Maskless sharded loops require a uniform batch length.
            _assert_uniform_seq_lengths(seq_lengths)

            # Under pure Ulysses each sharded loop names its own pad rows and the
            # joint attention its key order (adapters/boogu_ulysses.py). Ring and
            # hybrid get the plain override and keep their path.
            pure = ctx.pure_ulysses

            # Stage 1: replicated text refinement preserves its attention mask.
            for layer in self.context_refiner:
                text_hidden_states = layer(
                    text_hidden_states, text_attention_mask, context_rotary_emb,
                    transformer_options=transformer_options)

            img_len = flat_hidden_states.shape[1]  # noise-token count (head keeps this tail)

            # Stage 2: shard the maskless noise and reference-image refiners.
            noise_hs = self.x_embedder(flat_hidden_states)
            if noise_rotary_emb.shape[1] != noise_hs.shape[1]:
                raise UnsupportedModelError(
                    "boogu USP: noise rope length does not match the noise-token count "
                    f"({noise_rotary_emb.shape[1]} vs {noise_hs.shape[1]}); the batch has ragged "
                    "image sizes, which the sharded refiner cannot align. Run on topology 'single' "
                    "(docs/MODELS.md)."
                )
            noise_local, noise_orig = shard_seq(noise_hs, dim=1)
            noise_rope_local, _ = shard_seq(noise_rotary_emb, dim=1)
            noise_opts = stream_options(transformer_options, attn,
                                        (noise_orig, noise_local.shape[1]), pure_ulysses=pure)
            for layer in self.noise_refiner:
                noise_local = layer(noise_local, None, noise_rope_local, temb,
                                    transformer_options=noise_opts)
            noise_full = sp_gather(noise_local, noise_orig, dim=1)

            if flat_ref_img_hidden_states is not None:
                ref_hs = self.ref_image_patch_embedder(flat_ref_img_hidden_states)
                image_index_embedding = comfy.model_management.cast_to(
                    self.image_index_embedding, dtype=ref_hs.dtype, device=ref_hs.device)
                for i in range(ref_hs.shape[0]):
                    shift = 0
                    for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                        ref_hs[i, shift:shift + ref_img_len, :] = (
                            ref_hs[i, shift:shift + ref_img_len, :] + image_index_embedding[j])
                        shift += ref_img_len
                if ref_img_rotary_emb.shape[1] != ref_hs.shape[1]:
                    raise UnsupportedModelError(
                        "boogu USP: ref rope length does not match the ref-token count "
                        f"({ref_img_rotary_emb.shape[1]} vs {ref_hs.shape[1]}); the batch has ragged reference-image "
                        "sizes, which the sharded refiner cannot align. Run on topology 'single' (docs/MODELS.md)."
                    )
                ref_local, ref_orig = shard_seq(ref_hs, dim=1)
                ref_rope_local, _ = shard_seq(ref_img_rotary_emb, dim=1)
                ref_opts = stream_options(transformer_options, attn,
                                          (ref_orig, ref_local.shape[1]), pure_ulysses=pure)
                for layer in self.ref_image_refiner:
                    ref_local = layer(ref_local, None, ref_rope_local, temb,
                                      transformer_options=ref_opts)
                ref_full = sp_gather(ref_local, ref_orig, dim=1)
                combined_img_full = torch.cat([ref_full, noise_full], dim=1)
            else:
                combined_img_full = noise_full

            # Stage 3: keep instruction and image shards separate. Joint
            # attention concatenates their local RoPE; image attention uses only
            # the image-local RoPE.
            instruct_rope_full, img_rope_full = _split_joint_rope(rotary_emb, L_instruct)
            instruct_local, instruct_orig = shard_seq(text_hidden_states, dim=1)
            img_local, img_orig = shard_seq(combined_img_full, dim=1)
            instruct_rope_local, _ = shard_seq(instruct_rope_full, dim=1)
            img_rope_local, _ = shard_seq(img_rope_full, dim=1)
            joint_rope_local = torch.cat([instruct_rope_local, img_rope_local], dim=1)
            double_opts = double_block_options(
                transformer_options, attn, (instruct_orig, instruct_local.shape[1]),
                (img_orig, img_local.shape[1]), pure_ulysses=pure)

            for layer in self.double_stream_layers:
                # Pure Ulysses runs the instruct Linears on every instruct row
                # in stock's layout and re-shards their output.
                with full_row_text_projections(
                        layer, instruct_orig, img_orig, pure_ulysses=pure,
                        select=instruct_projections):
                    img_local, instruct_local = layer(
                        img_local, instruct_local,
                        joint_rope_local, img_rope_local, temb,
                        joint_attention_mask=None, img_attention_mask=None,
                        transformer_options=double_opts)

            instruct_full = sp_gather(instruct_local, instruct_orig, dim=1)
            img_full = sp_gather(img_local, img_orig, dim=1)
            hidden_states = torch.cat([instruct_full, img_full], dim=1)  # [text, ref, img]

            # Stage 4: reshard the joined stream and full RoPE identically.
            hidden_local, joint_orig = shard_seq(hidden_states, dim=1)
            rotary_local, _ = shard_seq(rotary_emb, dim=1)
            single_opts = stream_options(transformer_options, attn,
                                         (joint_orig, hidden_local.shape[1]), pure_ulysses=pure)
            for layer in self.single_stream_layers:
                hidden_local = layer(hidden_local, None, rotary_local, temb,
                                     transformer_options=single_opts)
            hidden_states = sp_gather(hidden_local, joint_orig, dim=1)

            # The image-tail slice requires the complete gathered sequence.
            hidden_states = self.norm_out(hidden_states, temb)
            p = self.patch_size
            output = rearrange(
                hidden_states[:, -img_len:], 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                h=H_padded // p, w=W_padded // p, p1=p, p2=p)[:, :, :H, :W]
            return -output  # The model contract negates the prediction.

        self.bind(diffusion_model, "forward", usp_forward)
        log.info(
            "boogu USP injected: %d refiner-depth (x3 stages) + %d double + %d single blocks, sp=%d",
            len(diffusion_model.context_refiner),
            len(diffusion_model.double_stream_layers),
            len(diffusion_model.single_stream_layers), ctx.topology_sp,
        )

    def inject_cfg_pad_forward(self, diffusion_model) -> None:
        """Install the per-row CFG forward for padded captions.

        The rope offset is per sample and known: each row's real caption length
        is the sum of its text keep-mask, the number ComfyUI sums to make
        ``num_tokens`` (with no mask, ``_content_caption_lengths`` reads it).
        Where the batch shares one caption length the pad is trimmed away and
        the stock class forward runs unchanged. Where the lengths differ
        the per-row forward above carries each row's own offset. Calling the
        class method avoids re-entering this bound forward.
        """

        def cfg_pad_forward(self, x, timesteps, context, num_tokens, ref_latents=None,
                            attention_mask=None, transformer_options={}, **kwargs):
            width = int(context.shape[1])
            if attention_mask is None:
                cap_lens = _content_caption_lengths(context)
            else:
                cap_lens = _keep_mask_caption_lengths(attention_mask, width)
                if len(cap_lens) == 1 and context.shape[0] > 1:
                    cap_lens = cap_lens * int(context.shape[0])
            trim = max(cap_lens)
            context = context[:, :trim]
            if min(cap_lens) == trim:
                # One caption length across the batch: with the pad trimmed off,
                # the stock forward is the single-GPU render of these prompts.
                return type(self).forward(
                    self, x, timesteps, context, trim, ref_latents=ref_latents,
                    attention_mask=None, transformer_options=transformer_options,
                    **kwargs)
            return _padded_caption_forward(
                self, x, timesteps, context, cap_lens, ref_latents,
                transformer_options)

        self.bind(diffusion_model, "forward", cfg_pad_forward)
        log.info("boogu cfg-pad forward installed")
