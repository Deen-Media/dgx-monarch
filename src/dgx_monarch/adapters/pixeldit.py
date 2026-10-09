"""PixelDiT-family adapter: Ideogram4 (head_dim 256).

The auto table (row 30) picks Ulysses2 and never cfg-parallel. On the tested
FP8 dual-model pair at 1024x1024, Ulysses2 matched DP2 at one-step
NRMS 0.000; Ring2 measured 0.102 against the 0.100 limit and remains an
explicit diagnostic only. Other quantizations inherit the head-divisible
topology choice without a hardware claim. Ideogram4's unconditional pass runs a
separate model (dual-model asymmetric CFG, docs/DESIGN.md section 5.7), so
there is no cond/uncond batch to split.

Blocks are pure packed-sequence self-attention, so USP rides the attention
override. The distributed forward assembles packed [text, image] tokens with
per-rank position ids. Under pure Ulysses the forward leaves the divisibility
pad row of an odd packed stream out of attention, because stock has no such key.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..log import get_logger
from .base import (
    Adapter,
    InjectionContext,
    ThrottledWarning,
    padded_row_indices,
    shard_seq,
    sp_gather,
    usp_options,
)

log = get_logger(__name__)


_segment_mask_dropped = ThrottledWarning()


def _drop_padded_text_segment_mask(indicator, attention_mask, l_text: int) -> None:
    """Zero Ideogram text-pad indicators and warn that USP drops the segment mask."""
    if attention_mask is None:
        return
    pad = attention_mask == 0
    indicator[:, :l_text][pad] = 0
    if bool(pad.any()):
        _segment_mask_dropped.warn(
            "ideogram4 USP: padded text tokens are present. The USP attention "
            "kernels cannot apply the block-diagonal segment mask, so this render "
            "drops it (pad rows are still zeroed out of the content path). For "
            "exact mask semantics run topology 'single' (mode=local with "
            "gpus_per_host=1)."
        )


class Ideogram4Adapter(Adapter):
    family = "ideogram4"
    model_base_classes = ("Ideogram4",)
    cfg_cond_padding = "none"  # cfg-parallel is excluded for this family (auto table row 30)
    cfg_parallel_supported = False  # dual-model guidance executes separate model calls
    dual_model_cfg_supported = True  # runs the split-guider dm-cfg2 route at cfg2 instead
    # comfy's Ideogram4Transformer2DModel fixes attention_head_dim at 256, which
    # exceeds the supported cuDNN and Sage head dimensions.
    attention_head_dim = 256

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timesteps, context=None, attention_mask=None,
                        transformer_options={}, **kwargs):
            # Distributed Ideogram4 forward with shard/gather around the
            # conditional and image-only layer loop. Interleaved MRoPE uses
            # the local position-id shard, which is exactly per-token.
            from comfy.ldm.ideogram4.model import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, _split_half_rope_matrix
            from comfy.text_encoders.llama import precompute_freqs_cis

            bs, _c, gh, gw = x.shape
            device = x.device
            timesteps = 1.0 - timesteps

            t_cond = self.t_embedding(timesteps, dtype=x.dtype)
            if timesteps.dim() == 1:
                t_cond = t_cond.unsqueeze(1)
            adaln_input = F.silu(self.adaln_proj(t_cond))

            img_tokens = self._img_to_tokens(x)
            l_img = img_tokens.shape[1]

            if context is None:
                # Image-only pass: the unconditional model of dual-model CFG.
                position_ids = self._image_position_ids(gh, gw, device).unsqueeze(0).expand(bs, l_img, 3)
                h = self.input_proj(img_tokens)
                h = h + self.embed_image_indicator(
                    torch.ones((bs, l_img), dtype=torch.long, device=device), out_dtype=h.dtype)
                l_text = 0
            else:
                l_text = context.shape[1]
                seq_len = l_text + l_img
                x_full = torch.zeros(bs, seq_len, img_tokens.shape[-1], dtype=img_tokens.dtype, device=device)
                x_full[:, l_text:] = img_tokens

                text_pos = torch.arange(l_text, device=device).view(-1, 1).expand(l_text, 3)
                img_pos = self._image_position_ids(gh, gw, device)
                position_ids = torch.cat([text_pos, img_pos], dim=0).unsqueeze(0).expand(bs, seq_len, 3)
                del img_pos, text_pos

                indicator = torch.empty(bs, seq_len, dtype=torch.long, device=device)
                indicator[:, :l_text] = LLM_TOKEN_INDICATOR
                indicator[:, l_text:] = OUTPUT_IMAGE_INDICATOR
                _drop_padded_text_segment_mask(indicator, attention_mask, l_text)

                image_row = (indicator == OUTPUT_IMAGE_INDICATOR)
                h = self.input_proj(x_full * image_row.to(x_full.dtype).unsqueeze(-1))
                h = h * image_row.to(h.dtype).unsqueeze(-1)

                text_row = (indicator[:, :l_text] == LLM_TOKEN_INDICATOR).to(x_full.dtype).unsqueeze(-1)
                llm = self.llm_cond_norm(context * text_row)
                llm = self.llm_cond_proj(llm) * text_row
                h[:, :l_text] = h[:, :l_text] + llm
                h = h + self.embed_image_indicator(image_row.to(torch.long), out_dtype=h.dtype)

            # Divisibility pad rows get zero position ids and zero content; the
            # gather trims them, so they never reach the output.
            h, orig_len = shard_seq(h, dim=1)
            position_ids, _ = shard_seq(position_ids, dim=1)

            freqs_cis = precompute_freqs_cis(
                self.head_dim, position_ids[0].transpose(0, 1), self.rope_theta,
                rope_dims=self.mrope_section, interleaved_mrope=True, device=device,
            )
            # ComfyUI f966a2b3 (#15080) uses comfy-kitchen's rms_rope_split_half
            # kernel, which takes a stacked rotation matrix rather than the
            # (cos, sin, neg_sin) tuple. Mirror
            # comfy's own forward (ideogram4/model.py) exactly: precompute on
            # the sharded positions, then the same matrix conversion, so each
            # rank's tokens keep their own rotations.
            freqs_cis = _split_half_rope_matrix(freqs_cis)

            # One packed stream chunked in order gathers back in stock's key
            # order, but an odd row count leaves a zero pad row at the tail.
            # It stays zero through the first block's norm and scale-only
            # modulation, yet it is still a key: logit 0 takes softmax weight,
            # and its zero query averages the values, so from the second block
            # on it carries content. Pure Ulysses drops it at the full-axis
            # point so the kernel sees stock's key set. Ring and hybrid keep
            # attending it: they cannot exclude pads, and naming drop rows
            # there would refuse.
            drop_rows: list[int] | None = None
            if ctx.pure_ulysses:
                drop_rows = padded_row_indices([(orig_len, h.shape[1])])
            usp_opts = usp_options(transformer_options, attn, drop_rows=drop_rows)
            for layer in self.layers:
                h = layer(h, None, freqs_cis, adaln_input, transformer_options=usp_opts)

            h = sp_gather(h, orig_len, dim=1)
            out = self.final_layer(h, adaln_input)
            return -self._tokens_to_img(out[:, l_text:], gh, gw)

        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("ideogram4 USP injected: %d layers, sp=%d", len(diffusion_model.layers), ctx.topology_sp)
