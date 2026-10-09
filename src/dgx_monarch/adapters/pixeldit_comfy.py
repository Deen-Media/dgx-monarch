"""Comfy PixelDiT T2I and PiD adapter. Ideogram4 has its own, in pixeldit.py.

The patch stage jointly attends text and `L = Hs*Ws` image tokens. The pixel
stage represents each patch as `(B*L, P^2, pixel_dim)`, compresses its pixels for
global attention across L, then expands them. Both stages carry the same
contiguous L shard: folded tensors are viewed as `(B, L, ...)`, sharded, and
refolded. One gather after `final_layer` restores L before `F.fold`. L padding
uses zero-pad plus trim and remains within the documented fidelity noise floor.

Text takes a separate shard for maskless joint attention and is discarded before
the pixel stage. Projections and MLPs stay sequence-sharded, but each attention
call gathers Q/K/V into stock stream order with the full stock head layout, runs
stock attention, and returns each rank's rows. Sharded Ulysses and Ring attention
miss the floor at 1024 (NRMS 0.265 on uly2, 0.338 on ring2; docs/VALIDATION.md,
PixelDiT/PiD exact-gather evidence). PiD adds replicated LQ
projection features shaped `(B, L, hidden)`; they take the same L shard as patch
conditioning, including its optional pixel-stage feature. The fixed 300-token
context needs no cfg padding.
"""
from __future__ import annotations

from typing import NoReturn

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    shard_seq,
    sp_gather,
    sp_rank,
    sp_world,
)

log = get_logger(__name__)

# PiD subclasses PixelDiTT2I; exact types refuse unknown future forwards.
_PIXELDIT_SUPPORTED_EXACT = ("PixelDiTT2I", "PiD")


def _raise_exact_gather_unsupported(detail: str) -> NoReturn:
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"PixelDiT exact-gather attention cannot use this ComfyUI surface: "
        f"{detail}. Use topology 'dp2' or local single-GPU stock execution.",
    ))


def _shard_bl(t_bl: torch.Tensor, batch: int) -> tuple[torch.Tensor, int]:
    """Shard `(B*L, ...)` through a `(B, L, ...)` view and refold it.

    This must use the same padded contiguous L shard as patch conditioning so
    `s_cond` and pixel tokens retain identical patch indices. Return the local
    tensor and pre-pad L for gather-time trim.
    """
    rest = tuple(t_bl.shape[1:])
    length = t_bl.shape[0] // batch
    local, orig = shard_seq(t_bl.reshape(batch, length, *rest), dim=1)
    return local.reshape(-1, *rest), orig


def _gather_bl(t_bl_local: torch.Tensor, batch: int, orig_len: int) -> torch.Tensor:
    """Gather `(B*L_local, ...)`, trim L to `orig_len`, and return `(B*L, ...)`."""
    rest = tuple(t_bl_local.shape[1:])
    local_len = t_bl_local.shape[0] // batch
    full = sp_gather(t_bl_local.reshape(batch, local_len, *rest), orig_len, dim=1)
    return full.reshape(-1, *rest)


def _shard_lq_features(lq_features):
    """Apply the patch stream's L shard to each `(B, L, hidden)` LQ feature."""
    return [shard_seq(f, dim=1)[0] for f in lq_features]


def _gather_rank_major(t: torch.Tensor) -> torch.Tensor:
    """Gather equal local token rows in process-rank order."""
    from xfuser.core.distributed import get_sp_group

    return get_sp_group().all_gather(t.contiguous(), dim=2)


def _global_segments(
    rank_major: torch.Tensor,
    segments: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Turn `[rank0 segments, rank1 segments, ...]` into stock stream order."""
    world = sp_world()
    rank_width = sum(local for _, local in segments)
    if rank_major.shape[2] != world * rank_width:
        _raise_exact_gather_unsupported("the gathered sequence width is inconsistent")
    streams = []
    local_offset = 0
    for original, local in segments:
        if original < 0 or local < 0 or original > world * local:
            _raise_exact_gather_unsupported("the stream lengths are invalid")
        parts = [
            rank_major.narrow(2, rank * rank_width + local_offset, local)
            for rank in range(world)
        ]
        streams.append(torch.cat(parts, dim=2).narrow(2, 0, original))
        local_offset += local
    return torch.cat(streams, dim=2)


def _local_segments(
    global_output: torch.Tensor,
    segments: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    """Return this rank's padded local rows from a stock-order full output."""
    rank = sp_rank()
    outputs = []
    global_offset = 0
    for original, local in segments:
        valid = max(0, min(local, original - rank * local))
        output = global_output.new_zeros(
            global_output.shape[0], global_output.shape[1], local, global_output.shape[3]
        )
        if valid:
            output[:, :, :valid] = global_output.narrow(
                2, global_offset + rank * local, valid
            )
        outputs.append(output)
        global_offset += original
    return torch.cat(outputs, dim=2)


def _gather_stock_attention(
    stock_attention,
    q,
    k,
    v,
    heads,
    *,
    segments: tuple[tuple[int, int], ...],
    mask=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    """Run stock full-head attention on gathered rows, then return local rows."""
    if mask is not None or not skip_reshape:
        _raise_exact_gather_unsupported("attention is not maskless `(B,H,L,D)`")
    if any(t.ndim != 4 for t in (q, k, v)) or not (q.shape == k.shape == v.shape):
        _raise_exact_gather_unsupported("q/k/v are not matching rank-4 tensors")
    local_width = sum(local for _, local in segments)
    if q.shape[2] != local_width:
        _raise_exact_gather_unsupported("stream lengths do not match q/k/v")

    q_full, k_full, v_full = (
        _global_segments(_gather_rank_major(t), segments) for t in (q, k, v)
    )
    full_output = stock_attention(
        q_full,
        k_full,
        v_full,
        heads,
        mask=None,
        skip_reshape=True,
        skip_output_reshape=True,
        **kwargs,
    )
    if not isinstance(full_output, torch.Tensor) or full_output.ndim != 4:
        _raise_exact_gather_unsupported("stock attention returned an invalid output")
    local_output = _local_segments(full_output, segments)
    if skip_output_reshape:
        return local_output
    batch, local_heads, local_length, head_dim = local_output.shape
    return local_output.transpose(1, 2).reshape(
        batch, local_length, local_heads * head_dim
    )


def _stock_gather_options(
    transformer_options: dict,
    segments: tuple[tuple[int, int], ...],
) -> dict:
    """Route one PixelDiT stage through exact gathered stock attention."""
    options = dict(transformer_options)

    def override(stock_attention, *args, **kwargs):
        kwargs.pop("_inside_attn_wrapper", None)
        kwargs.pop("transformer_options", None)
        return _gather_stock_attention(
            stock_attention, *args, segments=segments, **kwargs
        )

    options["optimized_attention_override"] = override
    return options


def _usp_pixel_block_forward(self, x, s_cond, image_height, image_width, patch_size,
                             mask=None, transformer_options={}):
    """Run PiTBlock on `(B*L_local, ...)` with gathered stock attention across L.

    B comes from options, L_local from local rows, and full-grid `pos_comp` takes
    the matching L shard. Keep stock's chunked MLP and its early `del` of the
    adaLN tensors.
    """
    from comfy.ldm.pixeldit.modules import apply_adaln_

    bl_local, p2, _ = x.shape
    batch = transformer_options["pixeldit_batch"]
    l_local = bl_local // batch
    hs, ws = image_height // patch_size, image_width // patch_size

    msa_params = self.adaLN_modulation_msa(s_cond).view(bl_local, p2, 3 * self.pixel_dim)
    shift_msa, scale_msa, gate_msa = msa_params.chunk(3, dim=-1)
    x_norm = apply_adaln_(self.norm1(x), shift_msa, scale_msa)
    x_flat = x_norm.view(bl_local, p2 * self.pixel_dim)

    x_comp = self.compress_to_attn(x_flat).view(batch, l_local, self.attn_dim)
    pos_comp = self._fetch_pos(hs, ws, x.device, x.dtype, **(transformer_options.get("rope_options") or {}))
    pos_comp, _ = shard_seq(pos_comp, dim=0)  # (L_local, ...) aligned with x_comp
    attn_out = self.attn(x_comp, pos_comp, mask=mask, transformer_options=transformer_options)
    attn_flat = self.expand_from_attn(attn_out.reshape(bl_local, self.attn_dim))
    attn_exp = attn_flat.view(bl_local, p2, self.pixel_dim)
    x = torch.addcmul(x, gate_msa, attn_exp)
    del msa_params, shift_msa, scale_msa, gate_msa

    mlp_params = self.adaLN_modulation_mlp(s_cond).view(bl_local, p2, 3 * self.pixel_dim)
    shift_mlp, scale_mlp, gate_mlp = mlp_params.chunk(3, dim=-1)
    gate_mlp = gate_mlp.contiguous()
    mlp_input = apply_adaln_(self.norm2(x), shift_mlp, scale_mlp)
    del mlp_params, shift_mlp, scale_mlp

    chunk_size = (bl_local + self.mlp_chunks - 1) // self.mlp_chunks
    for start in range(0, bl_local, chunk_size):
        end = min(start + chunk_size, bl_local)
        x[start:end].addcmul_(gate_mlp[start:end], self.mlp(mlp_input[start:end]))
    return x


class PixelDiTAdapter(Adapter):
    family = "pixeldit_comfy"
    model_base_classes = ("PixelDiTT2I",)  # PiD subclasses it
    exact_model_base_classes = _PIXELDIT_SUPPORTED_EXACT
    cfg_cond_padding = "none"  # fixed 300-token context with mask=None
    # These exclusions follow measured fidelity failures (docs/VALIDATION.md);
    # the reasons below carry the values. MXFP8 failed with two samplers, while
    # BF16 matched the reference on Ulysses2 and Ring2.
    cfg_parallel_supported = False
    cfg_parallel_refusal_reason = (
        "pixeldit cfg-parallel measured 1-step NRMS 0.206 against the 0.10 "
        "fidelity floor on 2026-08-21, so the split computes measurably wrong "
        "math and is refused. Run CFG above 1.0 on `auto` (Ulysses) or an "
        "explicit uly/ring preset."
    )
    sp_validated_quants = frozenset({"bf16"})
    sp_quant_refusal_reason = (
        "pixeldit quantized artifacts diverge under sequence sharding: the "
        "mxfp8 checkpoint measured 1-step NRMS 0.629 on both uly2 and ring2 "
        "against the 0.10 floor on 2026-08-21, while bf16 stayed exact on the "
        "same topologies. Use the bf16 artifact for parallel renders."
    )

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in self.exact_model_base_classes:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"pixeldit variant {type(base_model).__name__} is not in the dgx-monarch launch set "
            "(comfy PixelDiT T2I / PiD only, docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        """Bind the world-2 hardware-validated exact-gather SP forward."""
        def usp_forward(self, x, timesteps, context=None, attention_mask=None,
                        transformer_options={}, lq_latent=None, degrade_sigma=None, **kwargs):
            # Check PiD's required source latent before importing runtime modules.
            is_pid = hasattr(self, "lq_proj")
            if is_pid and lq_latent is None:
                raise UnsupportedModelError(
                    "PiD requires lq_latent; attach a low-res source latent via "
                    "PiDConditioning (pid.py:201)."
                )

            import comfy.ldm.common_dit
            import torch.nn.functional as F

            h_orig, w_orig = x.shape[2], x.shape[3]
            x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_size, self.patch_size))
            batch, _, height, width = x.shape
            hs = height // self.patch_size
            ws = width // self.patch_size
            length = hs * ws

            # PiD LQ features on the patch grid.
            block_kwargs: dict = {}
            if is_pid:
                expected_c = self.lq_proj.latent_channels
                if lq_latent.shape[1] != expected_c:
                    raise UnsupportedModelError(
                        f"PiD lq_latent has {lq_latent.shape[1]} channels; this variant expects "
                        f"{expected_c} (Flux1/SD3 = 16, Flux2 = 128)."
                    )
                ds = degrade_sigma.to(device=x.device, dtype=torch.float32).reshape(-1)
                if ds.numel() == 1 and batch > 1:
                    ds = ds.expand(batch).contiguous()
                # Replicated CNN outputs take the patch stream's L shard.
                lq_features = self.lq_proj(lq_latent=lq_latent.to(x), target_pH=hs, target_pW=ws)
                pit_lq_feature = lq_features.pop() if self.pit_lq_inject else None
                block_kwargs["pid_lq_features"] = _shard_lq_features(lq_features)
                block_kwargs["pid_pit_lq_feature"] = (
                    shard_seq(pit_lq_feature, dim=1)[0]
                    if pit_lq_feature is not None else None
                )
                block_kwargs["pid_degrade_sigma"] = ds

            # Patch stage: dual-stream joint attention.
            pos_img = self._fetch_patch_pos(hs, ws, x.device, x.dtype,
                                            **(transformer_options.get("rope_options") or {}))
            x_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
            t_emb = self.t_embedder(timesteps.view(-1), x.dtype).view(batch, -1, self.hidden_size)

            if context is None or context.dim() != 3:
                raise UnsupportedModelError("PixelDiT requires context (text embeddings) of shape [B, L, D].")
            ltxt = min(context.shape[1], self.txt_max_length)
            y_emb = self.y_embedder(context[:, :ltxt, :]).view(batch, ltxt, self.hidden_size)
            y_emb = y_emb + self.y_pos_embedding[:, :ltxt, :].to(y_emb)

            condition = F.silu(t_emb)
            pos_txt = self._fetch_text_pos(ltxt, x.device, x.dtype) if self.use_text_rope else None

            s = self.s_embedder(x_patches)

            # Shard each stream and every aligned position tensor identically.
            s, l_orig = shard_seq(s, dim=1)
            pos_img, _ = shard_seq(pos_img, dim=0)
            y_emb, y_orig = shard_seq(y_emb, dim=1)
            if pos_txt is not None:
                pos_txt, _ = shard_seq(pos_txt, dim=0)

            patch_segments = (
                (y_orig, y_emb.shape[1]),
                (l_orig, s.shape[1]),
            )
            usp_opts = _stock_gather_options(transformer_options, patch_segments)
            for i, blk in enumerate(self.patch_blocks):
                s = self._pre_patch_block(s, i, **block_kwargs)  # aligned PiD gate
                s, y_emb = blk(s, y_emb, condition, pos_img, pos_txt, None, transformer_options=usp_opts)
            s = F.silu(t_emb + s)
            s = self._pre_pixel_blocks(s, **block_kwargs)

            # Pixel stage: per-pixel tokens with global patch attention.
            s_cond = s.reshape(-1, self.hidden_size)  # (B*L_local, hidden)
            x_pixels = self.pixel_embedder(x, patch_size=self.patch_size)  # (B*L, P2, pixel_hidden)
            x_pixels, _ = _shard_bl(x_pixels, batch)  # (B*L_local, ...), same L shard as s_cond

            pixel_segments = ((l_orig, s.shape[1]),)
            usp_opts = _stock_gather_options(transformer_options, pixel_segments)
            usp_opts["pixeldit_batch"] = batch  # consumed by the local pixel block
            for blk in self.pixel_blocks:
                x_pixels = blk(x_pixels, s_cond, height, width, self.patch_size,
                               mask=None, transformer_options=usp_opts)

            x_pixels = self.final_layer(x_pixels)

            # Gather L once before reassembling the image.
            x_pixels = _gather_bl(x_pixels, batch, l_orig)  # (B*L, P2, C_out)
            c_out = self.out_channels
            p2 = self.patch_size * self.patch_size
            x_pixels = x_pixels.view(batch, length, p2, c_out).permute(0, 3, 2, 1).reshape(batch, c_out * p2, length)
            out = F.fold(x_pixels, (height, width), kernel_size=self.patch_size, stride=self.patch_size)
            return out[:, :, :h_orig, :w_orig]

        for blk in diffusion_model.pixel_blocks:
            self.bind(blk, "forward", _usp_pixel_block_forward)
        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("pixeldit_comfy exact-gather SP injected: %d patch + %d pixel blocks, sp=%d",
                 len(diffusion_model.patch_blocks), len(diffusion_model.pixel_blocks), ctx.topology_sp)
