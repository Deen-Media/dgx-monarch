"""LTX adapter: LTXV (video-only DiT) and LTXAV (audio-video DiT).

Covers LTX 2.3 and LTX 2.5: 2.5 binds the same two comfy classes and the same
token layout, RoPE payload, and timestep regime. See docs/MODELS.md for the
required ComfyUI revision.

Injection points:
  * `_process_transformer_blocks` is wrapped: tokens, positional embeddings
    and per-token timesteps shard before the block loop and gather after it.
    Input patchify and the output heads run on the full sequence, so keyframe
    (i2v) token filtering and `_process_output` stay stock.
  * Self-attention modules route through USP. Text cross-attention (attn2 /
    audio_attn2) stays stock: local q against full text k/v is already
    correct, and the text attention_mask keeps being honored.
  * LTXAV also shards the audio sequence and routes the bidirectional
    audio<->video cross attentions through USP. Video-to-audio needs the
    full video keys on every rank, which the ulysses all-to-all provides.
    Video-side per-frame CompressedTimestep tensors are expanded to per-token
    before sharding (contiguous token splits do not align to frame borders),
    while a timestep stream of one broadcast row is left whole on every rank.

Image conditioning needs no shard-side work. A per-frame noise mask (both i2v
routes) keeps comfy on its compressed timestep path, so the video timestep
still arrives one row per frame and expands to per-token before sharding; only
a spatial mask turns it per-token, and that carries a real token axis, so the
one-row broadcast guard never misfires. LTXV Add Guide appends whole latent
frames plus their `keyframe_idxs`, which comfy consumes in `_process_input`
ahead of this wrapper, so guide tokens ride the block loop as ordinary tokens
at the tail of the video stream.

Per-guide attention attenuation (guide strength != 1.0 / spatial guide masks)
biases the video self-attention. comfy applies that by splitting the query axis
and weighting each group against the whole sequence, which re-expresses exactly
under ulysses because the head scatter leaves every rank holding the whole
token axis and the weights are head-independent. Ring keeps the typed refusal:
its ranks see the keys one block at a time and that kernel takes no bias.

Neither stream need divide the shard degree, so `shard_seq` zero-pads the
shortfall and each patched attention is told which gathered rows are synthetic
and drops them (adapters/usp_pad_exclusion.py). Two of the four are cross
attentions reading one stream's queries against the other's keys, so they
carry both row sets.
"""
from __future__ import annotations

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .attention_patches import assert_no_foreign_attention_override
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    assert_ulysses_only_padding,
    padded_row_indices,
    shard_seq,
    sp_gather,
)

log = get_logger(__name__)

# The wrapper's private channel to the patched attentions: comfy hands one
# transformer_options to every attention in a block (BasicAVTransformerBlock.forward).
_PAD_ROWS_KEY = "_dgxm_ltx_pad_rows"
VIDEO, AUDIO = "video", "audio"

# The two comfy.model_base classes this adapter binds, matched exactly: an
# unvetted LTXV/LTXAV subclass raises typed instead of binding.
_LTX_SUPPORTED_EXACT = ("LTXV", "LTXAV")

# Guide attenuation is a request-decided property of the graph, so both ranks
# reach this on the same request and refuse together. Tagged so the driver
# retires the sample lease consumed and the next queue reuses the same fleet.
_GUIDE_ATTENUATION_RING_REFUSAL = refusal(
    RefusalClass.PHYSICS,
    "ltx guide attenuation is active: a guide carries a strength other than "
    "1.0 or a spatial attention mask, so the model biases attention between "
    "the noisy tokens and the guide tokens. Ulysses can apply that bias, "
    "because its ranks hold the whole key axis; ring and hybrid ranks see the "
    "keys one block at a time, and their kernel takes no bias. Use a "
    "pure-ulysses topology (uly2 on a two-rank fleet), or strength 1.0 on "
    "every LTXV Add Guide with its attention_mask input unconnected, or "
    "topology 'single'.",
    troubleshooting=80,
)

# Backstop one level down: any other mask reaching a patched attention. The
# guide bias arrives as query groups instead, which the sharded kernel applies.
_ATTENTION_BIAS_REFUSAL = refusal(
    RefusalClass.PHYSICS,
    "ltx: an unrecognized attention mask reached a USP-routed attention; USP "
    "kernels support no arbitrary attention bias. Use topology 'single'.",
    troubleshooting=80,
)


def _shard_pe(pe):
    """Shard each supported RoPE payload along its token axis.

    The rotation-matrix form introduced by ComfyUI 7c59a078 (#15056) is
    ``(rotation_matrix, split_mode)`` with shape
    ``(B, T, heads, head_dim/2, 2, 2)``; shard dimension 1. The legacy form is
    ``(cos, sin, split_mode)``: shard dimension 2 for split layout
    ``(B, H, T, d)`` and dimension 1 for interleaved ``(B, T, d)``.

    Reject any other arity with class P before the stock block loop so an
    unknown upstream format cannot silently assign the wrong positions.
    """
    if len(pe) == 2:
        rotation_matrix, split_mode = pe
        rotation_matrix, _ = shard_seq(rotation_matrix, dim=1)
        return (rotation_matrix, split_mode)
    if len(pe) == 3:
        cos, sin, split_mode = pe
        dim = 2 if cos.ndim == 4 else 1
        cos, _ = shard_seq(cos, dim=dim)
        sin, _ = shard_seq(sin, dim=dim)
        return (cos, sin, split_mode)
    raise UnsupportedModelError(refusal(
        RefusalClass.PHYSICS,
        f"ltx rope payload has {len(pe)} elements; this build shards only the "
        "2-element rotation-matrix form (comfy #15056, 2026-07-24) and the "
        "older 3-element cos/sin/split form. Upstream comfy changed the payload "
        "again, and this build cannot shard the new form correctly. Use topology "
        "'single' until dgx-monarch supports the new form.",
    ))


class _GuideBias:
    """comfy's guide bias as query groups, riding the stock mask channel."""

    __slots__ = ("groups",)

    def __init__(self, groups) -> None:
        self.groups = groups


def _guide_bias(mask, queries: int) -> _GuideBias:
    """Re-express a ``GuideAttentionMask`` as ordered query-bias groups.

    The split is comfy's own (``_attention_with_guide_mask``), rebuilt here so
    the sharded kernel runs it. ``queries`` is the stream's own row count, the
    one comfy built the mask over: divisibility pad rows sit past the tail of
    it and leave the kernel before any group is applied, so no group has to
    cover them and no bias grows a column for them.
    """
    guide_start = int(mask.guide_start)
    tracked_end = guide_start + int(mask.tracked_count)
    groups = []
    if guide_start > 0:
        groups.append((0, guide_start, mask.noisy_mask))
    groups.append((guide_start, tracked_end, mask.tracked_mask))
    if tracked_end < queries:
        groups.append((tracked_end, queries, None))
    return _GuideBias(tuple(groups))


def _reject_ring_guide_bias() -> None:
    """Refuse an attenuated guide on a topology that shards the key axis."""
    from xfuser.core.distributed import get_ring_parallel_world_size

    if get_ring_parallel_world_size() > 1:
        raise UnsupportedModelError(_GUIDE_ATTENUATION_RING_REFUSAL)


def _reject_ring_padding(pad_rows: dict[str, list[int]]) -> None:
    """Refuse pads ring cannot drop, before any collective; both ranks agree."""
    total = sum(len(rows) for rows in pad_rows.values())
    if not total:
        return
    from xfuser.core.distributed import get_ring_parallel_world_size

    assert_ulysses_only_padding(get_ring_parallel_world_size(), total)


def _stream_pad_rows(orig: int, local: int) -> list[int]:
    """Gathered pad-row indices; a stream shards alone, so they are its tail."""
    return padded_row_indices([(orig, local)])


def _shard_guide_bias(self_attention_mask, video_rows: int):
    """Build the bias over the video stream's own rows, pads excluded.

    Any other mask object refuses rather than being read by duck typing: it
    would carry different semantics under the same attribute names.
    """
    if self_attention_mask is None:
        return None
    from comfy.ldm.lightricks.model import GuideAttentionMask

    if not isinstance(self_attention_mask, GuideAttentionMask):
        raise UnsupportedModelError(_ATTENTION_BIAS_REFUSAL)
    return _guide_bias(self_attention_mask, video_rows)


def _make_usp_cross_attention(attn, queries: str = VIDEO, keys: str | None = None):
    """USP forward for LTX CrossAttention instances.

    It runs self-attention when context is None and audio<->video cross
    attention otherwise; text cross-attention is not patched. ``queries`` and
    ``keys`` name the streams it reads, hence whose pads it drops on each side."""
    keys = queries if keys is None else keys

    def usp_cross_attention(self, x, context=None, mask=None, pe=None, k_pe=None,
                            transformer_options={}):
        if context is None and transformer_options.get("stg_skip_self_attn", False):
            # Spatio-temporal guidance (LTX 2.5) degrades a flagged self-attention
            # to its value projection, ahead of the norms and rope. This
            # replacement bypasses that stock short circuit, so it reads the flag
            # here; otherwise the guidance perturbs nothing and scores against an
            # identical prediction. The projection is per token, so the local
            # shard is exact; a mask is ignored here as stock ignores it.
            out = self.to_v(x)
        else:
            from comfy.ldm.lightricks.model import apply_rotary_emb

            if mask is not None and not isinstance(mask, _GuideBias):
                raise UnsupportedModelError(_ATTENTION_BIAS_REFUSAL)
            q = self.q_norm(self.to_q(x))
            kv_src = x if context is None else context
            k = self.k_norm(self.to_k(kv_src))
            v = self.to_v(kv_src)
            if pe is not None:
                q = apply_rotary_emb(q, pe)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe)

            groups = mask.groups if isinstance(mask, _GuideBias) else None
            pads = transformer_options.get(_PAD_ROWS_KEY) or {}
            # `None` means "keys drop where queries do", the self-attention
            # contract; a cross attention names its stream even when empty.
            out = attn(q, k, v, self.heads, query_bias_groups=groups,
                       drop_rows=pads.get(queries) or None,
                       kv_drop_rows=(None if keys == queries
                                     else pads.get(keys) or []))

        if self.to_gate_logits is not None:
            b, t_len, _ = out.shape
            gates = 2.0 * torch.sigmoid(self.to_gate_logits(x))
            out = (out.view(b, t_len, self.heads, self.dim_head) * gates.unsqueeze(-1)).view(b, t_len, -1)
        return self.to_out(out)

    return usp_cross_attention


class LTXAdapter(Adapter):
    family = "ltx"
    model_base_classes = ("LTXV", "LTXAV")
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in _LTX_SUPPORTED_EXACT:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"ltx variant {type(base_model).__name__} is not in the dgx-monarch launch set "
            "(LTXV / LTXAV only, docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        if hasattr(diffusion_model, "audio_inner_dim"):
            self._inject_ltxav(diffusion_model, ctx)
        else:
            self._inject_ltxv(diffusion_model, ctx)

    def _inject_ltxv(self, diffusion_model, ctx: InjectionContext) -> None:
        usp_self = _make_usp_cross_attention(ctx.usp_attention)
        stock_process_blocks = type(diffusion_model)._process_transformer_blocks
        family = self.family

        def usp_process_blocks(self, x, context, attention_mask, timestep, pe,
                               transformer_options={}, self_attention_mask=None, **kwargs):
            # Before the first shard, so every rank refuses on the same request.
            assert_no_foreign_attention_override(transformer_options, family)
            if self_attention_mask is not None:
                _reject_ring_guide_bias()
            x, orig_len = shard_seq(x, dim=1)
            pad_rows = {VIDEO: _stream_pad_rows(orig_len, x.shape[1])}
            _reject_ring_padding(pad_rows)
            guide_bias = _shard_guide_bias(self_attention_mask, orig_len)
            pe = _shard_pe(pe)
            if timestep is not None and torch.is_tensor(timestep) and timestep.ndim == 3 and timestep.shape[1] > 1:
                timestep, _ = shard_seq(timestep, dim=1)

            x = stock_process_blocks(
                self, x, context, attention_mask, timestep, pe,
                transformer_options={**transformer_options,
                                     _PAD_ROWS_KEY: pad_rows},
                self_attention_mask=guide_bias, **kwargs,
            )
            return sp_gather(x, orig_len, dim=1)

        for block in diffusion_model.transformer_blocks:
            self.bind(block.attn1, "forward", usp_self)
        self.bind(diffusion_model, "_process_transformer_blocks", usp_process_blocks)
        log.info("ltxv USP injected: %d blocks, sp=%d",
                 len(diffusion_model.transformer_blocks), ctx.topology_sp)

    def _inject_ltxav(self, diffusion_model, ctx: InjectionContext) -> None:
        # One forward per role: each stream pads on its own, so a shared
        # forward could not know which rows are synthetic on which side.
        video_self = _make_usp_cross_attention(ctx.usp_attention, VIDEO)
        audio_self = _make_usp_cross_attention(ctx.usp_attention, AUDIO)
        audio_to_video = _make_usp_cross_attention(ctx.usp_attention, VIDEO, AUDIO)
        video_to_audio = _make_usp_cross_attention(ctx.usp_attention, AUDIO, VIDEO)
        stock_process_blocks = type(diffusion_model)._process_transformer_blocks
        family = self.family

        def shard_maybe_compressed(ts):
            """Per-token shard for plain tensors and CompressedTimestep.

            Per-frame compression cannot survive a contiguous token split, so
            compressed video timesteps are expanded to per-token first (small:
            tokens x feature dim at bf16) and re-wrapped uncompressed.

            A stream that carries one row carries a broadcast value, not a token
            axis, and every rank must keep it whole, in both arms. comfy builds
            that form for the video streams whenever the sampler passes a scalar
            sigma (no denoise mask, so `_prepare_timestep` takes neither the
            per-token nor the per-frame path and `CompressedTimestep` falls back
            to patches_per_frame 1 over a single row). Sharding it pads length 1
            up to the SP degree and hands every rank after the first an all-zero
            AdaLN embedding.
            """
            from comfy.ldm.lightricks.av_model import CompressedTimestep

            if ts is None:
                return None
            if isinstance(ts, CompressedTimestep):
                if ts.num_frames * ts.patches_per_frame <= 1:
                    return ts
                expanded = ts.expand()
                local, _ = shard_seq(expanded, dim=1)
                return CompressedTimestep(local, None)
            if torch.is_tensor(ts) and ts.ndim == 3 and ts.shape[1] > 1:
                local, _ = shard_seq(ts, dim=1)
                return local
            return ts

        def usp_process_blocks(self, x, context, attention_mask, timestep, pe,
                               transformer_options={}, self_attention_mask=None, **kwargs):
            # Before the first shard, so every rank refuses on the same request.
            assert_no_foreign_attention_override(transformer_options, family)
            if self_attention_mask is not None:
                _reject_ring_guide_bias()
            vx, ax = x[0], x[1]
            (v_pe, av_cross_v), (a_pe, av_cross_a) = pe[0], pe[1]

            vx, v_orig = shard_seq(vx, dim=1)
            # The bias is built over the video stream alone: comfy sizes it from
            # x[0] and hands it to attn1 only (BasicAVTransformerBlock.forward),
            # never to the audio self-attention or either cross attention.
            guide_bias = _shard_guide_bias(self_attention_mask, v_orig)
            v_pe = _shard_pe(v_pe)
            av_cross_v = _shard_pe(av_cross_v)

            pad_rows = {VIDEO: _stream_pad_rows(v_orig, vx.shape[1])}
            has_audio = ax is not None and ax.numel() > 0
            a_orig = 0
            if has_audio:
                ax, a_orig = shard_seq(ax, dim=1)
                a_pe = _shard_pe(a_pe)
                av_cross_a = _shard_pe(av_cross_a)
                pad_rows[AUDIO] = _stream_pad_rows(a_orig, ax.shape[1])
            _reject_ring_padding(pad_rows)

            v_ts = shard_maybe_compressed(timestep[0])
            a_ts = shard_maybe_compressed(timestep[1]) if has_audio else timestep[1]
            cross_ts = timestep[2]
            if cross_ts:
                a_ss, v_ss, a2v_gate, v2a_gate = cross_ts
                cross_ts = [
                    shard_maybe_compressed(a_ss) if has_audio else a_ss,
                    shard_maybe_compressed(v_ss),
                    shard_maybe_compressed(a2v_gate),
                    shard_maybe_compressed(v2a_gate) if has_audio else v2a_gate,
                ]
            timestep = [v_ts, a_ts, cross_ts, timestep[3], timestep[4]]

            out = stock_process_blocks(
                self, [vx, ax], context, attention_mask, timestep,
                [(v_pe, av_cross_v), (a_pe, av_cross_a)],
                transformer_options={**transformer_options,
                                     _PAD_ROWS_KEY: pad_rows},
                self_attention_mask=guide_bias, **kwargs,
            )
            vx, ax = out[0], out[1]
            vx = sp_gather(vx, v_orig, dim=1)
            if has_audio:
                ax = sp_gather(ax, a_orig, dim=1)
            return [vx, ax]

        for block in diffusion_model.transformer_blocks:
            # Video self, audio self and both av cross attentions run over
            # sharded sequences, so they route through USP. comfy names a cross
            # attention for the stream it carries into the other, so
            # audio_to_video takes video queries (BasicAVTransformerBlock.forward).
            self.bind(block.attn1, "forward", video_self)
            self.bind(block.audio_attn1, "forward", audio_self)
            self.bind(block.audio_to_video_attn, "forward", audio_to_video)
            self.bind(block.video_to_audio_attn, "forward", video_to_audio)
        self.bind(diffusion_model, "_process_transformer_blocks", usp_process_blocks)
        log.info("ltxav USP injected: %d blocks, sp=%d",
                 len(diffusion_model.transformer_blocks), ctx.topology_sp)
