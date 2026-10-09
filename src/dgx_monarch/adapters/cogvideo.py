"""CogVideoX T2V, I2V, and inpaint adapter.

Blocks use one joint [text, image] self-attention, so the scoped USP override
cannot affect cross-attention. Patch embedding runs before sharding because the
2B positional table is applied at global positions. Text and image then shard
separately, and each block derives its text split from the local stream.

For 5B/1.5 models, precomputed image RoPE rows take the identical pad and chunk
as image tokens. T2V, I2V, and inpaint differ only through channel conditioning
and per-batch offsets, so they share this sequence contract.
"""
from __future__ import annotations

from ..log import get_logger
from .base import Adapter, InjectionContext, UnsupportedModelError, shard_seq, sp_gather, usp_options

log = get_logger(__name__)


def _shard_rope(rope):
    """Shard `(cos, sin)` rows with the image-token pad and chunk.

    Any differing pad would shift later ranks' rotations. Zero RoPE pad rows
    zero the corresponding q/k entries and are trimmed after gather.
    """
    cos, sin = rope
    cos = shard_seq(cos.unsqueeze(0), dim=1)[0].squeeze(0)
    sin = shard_seq(sin.unsqueeze(0), dim=1)[0].squeeze(0)
    return (cos, sin)


class CogVideoXAdapter(Adapter):
    family = "cogvideo"
    # All supported variants build the same model_base class.
    model_base_classes = ("CogVideoX",)
    # ComfyUI pads T5 conditioning to at least 226 tokens, so prompts up to that
    # length match without a pad.
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        if type(base_model) is model_base.CogVideoX:
            return True
        raise UnsupportedModelError(
            f"cogvideo variant {type(base_model).__name__} subclasses CogVideoX with a "
            "forward this adapter has not vetted, so every topology refuses it. Render it "
            "in stock ComfyUI on one GPU, or, for a finetune that keeps the plain CogVideoX "
            "forward, set family_adapter='cogvideo' on the Init node (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timestep, context, ofs=None, transformer_options=None, **kwargs):
            if kwargs.get("attention_mask") is not None:
                raise UnsupportedModelError(
                    "cogvideo: this render carries an attention mask, which the USP "
                    "kernels cannot honor. Use topology 'single'."
                )
            if transformer_options is None:
                transformer_options = {}

            import comfy.ldm.common_dit
            from comfy.ldm.cogvideo.model import get_timestep_embedding

            batch_size, _channels, t, h, w = x.shape
            p_t = self.patch_size_t if self.patch_size_t is not None else 1
            x = comfy.ldm.common_dit.pad_to_patch_size(x, (p_t, self.patch_size, self.patch_size))
            x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
            batch_size, num_frames, _channels, height, width = x.shape

            # Time and 1.5-I2V offsets are per-batch, not per-token.
            t_emb = get_timestep_embedding(timestep, self.time_proj_dim, self.time_proj_flip, self.time_proj_shift)
            t_emb = t_emb.to(dtype=x.dtype)
            emb = self.time_embedding_linear_2(self.time_embedding_act(self.time_embedding_linear_1(t_emb)))
            if self.ofs_embedding_linear_1 is not None and ofs is not None:
                ofs_emb = get_timestep_embedding(ofs, self.ofs_proj_dim, self.time_proj_flip, self.time_proj_shift)
                ofs_emb = ofs_emb.to(dtype=x.dtype)
                emb = emb + self.ofs_embedding_linear_2(self.ofs_embedding_act(self.ofs_embedding_linear_1(ofs_emb)))

            # The 2B positional table requires full-sequence patch embedding.
            hidden_states = self.patch_embed(context, x)

            text_seq_length = context.shape[1]
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]

            # The 5B/1.5 RoPE table follows the image-token shard.
            image_rotary_emb = None
            if self.use_rotary_positional_embeddings:
                post_time = num_frames if self.patch_size_t is None else num_frames // self.patch_size_t
                image_rotary_emb = _shard_rope(self._get_rotary_emb(
                    height // self.patch_size, width // self.patch_size, post_time, device=x.device))

            # Local blocks see [text_local, image_local], aligned with local RoPE.
            encoder_hidden_states, _ = shard_seq(encoder_hidden_states, dim=1)
            hidden_states, img_orig_len = shard_seq(hidden_states, dim=1)

            usp_opts = usp_options(transformer_options, attn)
            for block in self.blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    transformer_options=usp_opts,
                )

            # Only image rows reach the output head.
            hidden_states = sp_gather(hidden_states, img_orig_len, dim=1)

            hidden_states = self.norm_final(hidden_states)
            hidden_states = self.norm_out(hidden_states, temb=emb)
            hidden_states = self.proj_out(hidden_states)

            p = self.patch_size
            p_t = self.patch_size_t
            if p_t is None:
                output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
                output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
            else:
                output = hidden_states.reshape(
                    batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
                )
                output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

            return output.permute(0, 2, 1, 3, 4)[:, :, :t, :h, :w]

        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("cogvideo USP injected: %d blocks, sp=%d", len(diffusion_model.blocks), ctx.topology_sp)
