"""MiniMax H3 packed audio-video DiT adapter.

H3 packs ``[text | condition | references | audio | video]`` into one dim-0
token stream. Packing, condition splicing, projections, timestep embedding,
and modulation indices run replicated. The stream, position IDs, and
per-position modulation indices then take identical sequence shards. Positions
are embedded locally, and the gathered stream is trimmed before the fp32 output
head slices target spans by absolute coordinate.

The packed length depends on prompt and media rows. Because attention is
maskless and bidirectional, synthetic divisibility rows must be excluded from
Ulysses attention; padded Ring or hybrid execution refuses. H3 accepts batch 1,
while CFG parallel has no combined batch to split and data parallel cannot slice
its nested audio-video latent, so both topologies refuse before dispatch. A
latent noise mask refuses as well, because masked rows carry their own timestep
and therefore per-token modulation rows.

Video uses the sampler schedule while audio modulation uses its own shifted
schedule. The bound outer forward restores and converts audio, so this inner
forward returns raw negated audio velocity and must not apply that conversion
again. FSDP admits the trained fp32 parameter islands as replicated parameters
(adapters/fsdp_islands.py), and replicates `condition_proj` and
`token_refiner` beside them because comfy runs that pair from `extra_conds`,
outside this forward. A world-2 bf16 Ulysses+FSDP render completed, but that
completion does not establish FSDP fidelity; see docs/VALIDATION.md.
"""
from __future__ import annotations

import inspect
import math
from numbers import Real
from typing import Any

import torch

from .. import accuracy_waiver
from ..log import get_logger
from ..refusal import RefusalClass, refusal
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    padded_row_indices,
    shard_seq,
    sp_gather,
    sp_world,
    usp_options,
)
from .minimax_h3_packing import (
    h3_modulation_plan,
    h3_timestep_embedding,
    pad_row_index,
    row_runs,
    splice_condition_rows,
)
from .sol_attention import (
    SOL_DENSE_BLOCKS,
    sol_dense_scope,
    sol_dense_step,
    sol_sink_scope,
)

log = get_logger(__name__)

# Exact ComfyUI model class allowed for this family. detect.py owns checkpoint
# discrimination.
_MINIMAX_H3_SUPPORTED_EXACT = ("MiniMaxH3",)

MINIMAX_H3_BATCH_CAPPED_MESSAGE = refusal(
    RefusalClass.PHYSICS,
    "minimax_h3: cfg-parallel and data-parallel cannot run this family. Every H3 "
    "conditioning publishes its own packed-layout object, so ComfyUI never merges cond "
    "and uncond into one batched call, and cfg-parallel has nothing to split. The DiT "
    "accepts batch 1 only, and the audio-video latent is a packed nested tensor that the "
    "data-parallel slicer refuses. The model is CFG distilled, so a single positive "
    "conditioning is the intended input. Run this workflow on a uly* topology, or take a "
    "true single-GPU reference (mode=local, gpus_per_host=1). The 'single' preset is not "
    "that reference: above world 1 it derives dp and refuses here too "
    "(docs/MODELS.md).",
)


def minimax_h3_topology_would_reject(cfg: int, dp: int) -> bool:
    """Return the shared structural cfg/dp refusal decision for H3."""
    return cfg > 1 or dp > 1


def _has_current_audio_carry_outer(forward: Any) -> bool:
    """Fingerprint the outer audio-carry contract required by this inner return.

    The payload and model modules can be version-skewed while method signatures
    remain identical, so code constants distinguish the required conversion.
    """
    function = getattr(forward, "__func__", forward)
    code = getattr(function, "__code__", None)
    if code is None:
        return False
    constants = frozenset(value for value in code.co_consts if isinstance(value, str))
    return {
        "audio_scale",
        "minimax_h3_sigma_shift_video",
        "minimax_h3_sigma_shift_audio",
    } <= constants and "time_shift_sigma" in code.co_names


def _build_packed_layout(packed_layout_cls, payload, text_len, latent_t,
                         lat_h, lat_w, audio_t):
    """Build the stock PackedLayout across the frame_count seam.

    Comfy e01fb4c56 removed PackedLayout's frame_count parameter (the new
    guide anchoring replaced it). Older comfy revisions still take it and use
    it for last-frame keyframes, so it is passed only where the installed
    class accepts it.
    """
    extra = {}
    init = inspect.signature(packed_layout_cls.__init__)
    if "frame_count" in init.parameters:
        extra["frame_count"] = payload.get("frame_count")
    return packed_layout_cls(text_len, latent_t, lat_h, lat_w, audio_t,
                             keyframes=payload.get("keyframes"),
                             refs=payload.get("refs"),
                             **extra)


def _reject_latent_noise_masks(denoise_mask: Any, audio_denoise_mask: Any) -> None:
    """Refuse a latent noise mask instead of dropping it.

    Comfy ff6c8a8 added the two mask parameters. The message below says what
    H3 does with a mask, in the block stream and the output head alike, and
    why this forward, which plans from the unmasked stream times, cannot
    honor it. The values are request-derived and identical on every rank
    before the first collective, so class P retires the sample lease consumed
    and the operator re-queues on the same fleet. The exception is a first-use
    gate ceremony, which proves the operator's own frozen latent: a refusal
    there aborts the ceremony and costs one Recycle, the contract
    docs/TROUBLESHOOTING.md #49 records for a mid-ceremony refusal.
    """
    for name, mask in (("denoise_mask", denoise_mask),
                       ("audio_denoise_mask", audio_denoise_mask)):
        if mask is None:
            continue
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"minimax_h3: this render carries a '{name}'. H3 runs masked rows at their "
            "own timestep: it shifts the stream's modulation row when the kept rows "
            "share one value, and turns that row into a per-token index tensor when "
            "they do not. This distributed forward plans its modulation rows from the "
            "unmasked stream times alone, so it would renoise the rows the mask "
            "preserves. Clear the latent mask, or take a true single-GPU reference "
            "(mode=local, gpus_per_host=1) (docs/MODELS.md).",
        ))


def _reject_unsupported_h3_features(
    x: Any,
    transformer_options: dict[str, Any],
    minimax_payload: Any,
    kwargs: dict[str, Any],
    outer_forward: Any,
) -> None:
    """Refuse unsupported inputs before ComfyUI imports or collectives."""
    # Request-derived payload failures are identical on every rank before the
    # first collective, so class-P handling retires the sample lease consumed.
    payload_error: str | None = None
    if not minimax_payload:
        payload_error = refusal(
            RefusalClass.PHYSICS,
            "minimax_h3: this render carries no minimax_payload content. Stock comfy always "
            "publishes at least one key (the payload holds the packed layout, the condition "
            "latents, the presentation token tags and the sampler seed), so an absent or "
            "empty payload means the conditioning never reached this worker. Rendering "
            "anyway would turn a keyframe or reference-clip render into plain "
            "text-to-video with no error. Re-run the workflow, or take a true single-GPU "
            "reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
        )
    elif not isinstance(minimax_payload, dict) or "audio_scale" not in minimax_payload:
        payload_error = refusal(
            RefusalClass.PHYSICS,
            "minimax_h3: minimax_payload is missing the required 'audio_scale' marker for "
            "ComfyUI's carried-audio outer-forward contract. This payload came from an "
            "older or partial ComfyUI path; this distributed inner forward returns raw "
            "audio velocity, so running without the matching outer conversion would "
            "silently integrate the wrong audio ODE. Update ComfyUI consistently on every "
            "host, or take a true single-GPU reference (mode=local, gpus_per_host=1) "
            "(docs/MODELS.md).",
            troubleshooting=68,
        )
    else:
        audio_scale = minimax_payload["audio_scale"]
        audio_scale_value: float | None = None
        if not isinstance(audio_scale, bool) and isinstance(audio_scale, Real):
            try:
                audio_scale_value = float(audio_scale)
            except Exception:
                # A custom Real may fail conversion; keep it inside the typed gate.
                pass
        if (
            audio_scale_value is None
            or not math.isfinite(audio_scale_value)
            or audio_scale_value <= 0.0
        ):
            payload_error = refusal(
                RefusalClass.PHYSICS,
                "minimax_h3: minimax_payload['audio_scale'] must be a finite positive "
                "numeric value from ComfyUI's carried-audio contract; got an invalid "
                f"{type(audio_scale).__name__} value. "
                "Update ComfyUI consistently on every host, or take a true single-GPU "
                "reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
                troubleshooting=68,
            )
        elif not _has_current_audio_carry_outer(outer_forward):
            # Must stay untagged: host code can differ across ranks, so one rank
            # may refuse while another enters a collective. The lease retires
            # abandoned and needs one Recycle before redispatch
            # (docs/TROUBLESHOOTING.md #68).
            payload_error = (
                "minimax_h3: minimax_payload carries 'audio_scale', but the bound stock "
                "MiniMaxH3Model.forward does not carry ComfyUI's matching audio unscale "
                "and velocity-conversion contract. This is a partial or mixed ComfyUI "
                "update; distributed execution would silently integrate the wrong audio "
                "ODE. Update ComfyUI consistently on every host, then reset the Attached "
                "mesh; or take a true single-GPU reference (mode=local, gpus_per_host=1) "
                "(docs/MODELS.md, docs/TROUBLESHOOTING.md #68)."
            )
    if payload_error is not None:
        raise UnsupportedModelError(payload_error)
    video = x[0] if isinstance(x, (list, tuple)) and len(x) else None
    shape = getattr(video, "shape", None)
    if shape is not None and len(shape) and int(shape[0]) != 1:
        raise UnsupportedModelError(
            f"minimax_h3: the packed forward received batch {int(shape[0])}, and this "
            "family runs batch 1 only. " + MINIMAX_H3_BATCH_CAPPED_MESSAGE
        )
    for name in ("attention_mask", "attn_mask"):
        if kwargs.get(name) is not None:
            raise UnsupportedModelError(
                f"minimax_h3: this render carries a '{name}', but H3's stock attention is "
                "maskless and fully bidirectional, so the sharded kernel cannot apply "
                "one. Run masked workflows on a true single-GPU reference (mode=local, "
                "gpus_per_host=1) (docs/MODELS.md)."
            )
    if transformer_options.get("patches_replace", {}).get("dit", {}):
        raise UnsupportedModelError(
            "minimax_h3: a 'dit' block replacement receives the token stream, the rotation "
            "table and the modulation segment table, all three of which are rank-local under "
            "sequence parallelism, so the patch would see a partial sequence and rank-local "
            "coordinates. Run this workflow on a true single-GPU reference (mode=local, "
            "gpus_per_host=1) (docs/MODELS.md)."
        )


# Replicated row source per packed segment kind. Text is assembled from the
# encoder output rather than a row stream, so it is absent. Every other kind
# the modulation plan vets must appear here, which tests/test_minimax_h3.py
# pins in both directions.
H3_STREAM_OF: dict[str, str] = {
    "cond": "video", "ref_img": "video", "video": "video",
    "cond_audio": "audio", "ref_audio": "audio", "audio": "audio",
}

H3_RING_PAD_GUARD = "ring_pad:minimax_h3"


def _sampler_step_position(transformer_options) -> tuple[int, int] | None:
    """(step index, total steps) read from the sampler's own sigma table; None
    when the table is unreadable, which sends the caller to the timestep fallback."""
    options = transformer_options or {}
    current = options.get("sigmas")
    table = options.get("sample_sigmas")
    if current is None or table is None or len(table) < 2:
        return None
    where = torch.where(table == current.flatten()[0].to(table.device))[0]
    if len(where) == 0:
        return None
    return int(where[0]), len(table) - 1


def _reject_padded_ring(n_pad_rows: int) -> None:
    """Refuse padded Ring or hybrid execution with an H3-valid remedy.

    The generic remedy names ``single``, which derives data parallelism above
    world 1 and is invalid for H3. This seam therefore points to Ulysses before
    the first block. It shares the generic ``ring_pad`` guard's waiver card,
    since both cover the same measured wrongness
    (docs/TROUBLESHOOTING.md #49 and #55).
    """
    from xfuser.core.distributed import get_ring_parallel_world_size

    if get_ring_parallel_world_size() > 1:
        if accuracy_waiver.waived(H3_RING_PAD_GUARD):
            return
        raise UnsupportedModelError(refusal(
            RefusalClass.KNOWN_WRONG,
            f"minimax_h3: this packed sequence needs {n_pad_rows} divisibility pad row(s), and "
            "the exact pad exclusion is Ulysses-only. H3's attention is maskless and fully "
            "bidirectional, so ring or hybrid attention would attend the synthetic rows (the "
            "left-edge corruption measured on 2026-07-10). Run this workflow on a uly* "
            "topology, or pick a canvas and prompt whose packed token total divides the "
            "sequence-parallel degree, or take a true single-GPU reference (mode=local, "
            "gpus_per_host=1).",
            guard=H3_RING_PAD_GUARD,
            waivable=True,
            panel_action=accuracy_waiver.panel_action(H3_RING_PAD_GUARD),
            troubleshooting=49,
        ) + accuracy_waiver.card_tail(H3_RING_PAD_GUARD, family_hint="minimax_h3"))


class MiniMaxH3Adapter(Adapter):
    family = "minimax_h3"
    model_base_classes = _MINIMAX_H3_SUPPORTED_EXACT
    # Exact-type matching prevents an unvetted subclass from inheriting support.
    exact_model_base_classes = _MINIMAX_H3_SUPPORTED_EXACT
    # CFG parallel is structurally unsupported, so no prompt-padding rule applies.
    # No dp_cond_exempt_keys either: data parallel is refused, so no shared
    # prompt array reaches its slicer.
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        for name in self.exact_model_base_classes:
            cls = getattr(model_base, name, None)
            if cls is not None and type(base_model) is cls:
                return True
        raise UnsupportedModelError(
            f"minimax_h3 variant {type(base_model).__name__} is a MiniMaxH3 subclass "
            "this adapter has not vetted. Take a true single-GPU reference "
            "(mode=local, gpus_per_host=1) (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention

        def usp_forward(self, x, timestep, context, transformer_options={},
                        minimax_payload=None, denoise_mask=None,
                        audio_denoise_mask=None, **kwargs):
            # transformer_options keeps the fourth positional slot required by
            # the wrapper. H3 passes it to optimized_attention, so usp_options
            # reaches every DiT attention without rebinding modules. The two
            # mask parameters are named rather than absorbed so a tree that
            # sends one is refused on both sides of comfy ff6c8a8.
            _reject_unsupported_h3_features(
                x, transformer_options, minimax_payload, kwargs, self.forward
            )
            _reject_latent_noise_masks(denoise_mask, audio_denoise_mask)

            import comfy.ldm.common_dit
            import comfy.model_prefetch
            from comfy.ldm.minimax.model import (
                AUDIO_COND_TIMESTEP,
                VISUAL_COND_TIMESTEP,
                PackedLayout,
                pack_audio,
                patchify_video,
                rope_rotation_table,
                time_shift_sigma,
                unpack_audio,
                unpatchify_video,
            )

            # ComfyUI contract: PackedLayout fields and shift keys define the
            # checkpoint interface; the timestep arrives as sigma * 1000, and
            # model time is 1 - sigma.
            payload = minimax_payload
            video_x, audio_x = x[0], x[1]
            orig_t, orig_h, orig_w = video_x.shape[2:]
            video_x = comfy.ldm.common_dit.pad_to_patch_size(video_x, self.patch_size)
            device = video_x.device
            # The packed stream uses the compute dtype; comfy casts context to it.
            dtype = context.dtype

            latent_t, lat_h, lat_w = video_x.shape[2:]
            audio_t = audio_x.shape[-1]
            text_len = context.shape[1]
            layout = payload.get("layout")
            if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
                layout = _build_packed_layout(
                    PackedLayout, payload, text_len, latent_t, lat_h, lat_w, audio_t)

            shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", self.sigma_shift_video))
            shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", self.sigma_shift_audio))
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            t_v = float(1.0 - sigma_v)
            t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
            unique_t, row_index, video_row, audio_row = h3_modulation_plan(
                layout, payload, t_v, t_a, VISUAL_COND_TIMESTEP, AUDIO_COND_TIMESTEP)

            # Condition rows are fresh and unnoised each step. Per-condition CPU
            # generator reseeding keeps this replicated span rank-invariant.
            all_video_rows = splice_condition_rows(
                patchify_video(video_x.to(torch.float32), self.patch_size),
                self._cond_video_rows(payload, device),
                layout.img_update.to(device), device)
            all_audio_rows = splice_condition_rows(
                pack_audio(audio_x.to(torch.float32)),
                self._cond_audio_rows(payload, device),
                layout.audio_update.to(device), device)

            # Both trained fp32 projections consume the helper-defined packed
            # rows; cast their outputs to the stream dtype.
            video_embed = self.video_patch_proj(all_video_rows).to(dtype)
            audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
            text_states = context[0]
            if text_states.shape[-1] != self.hidden_size:
                # Use stock options: this full replicated text sequence must not
                # enter the sharded attention override.
                text_states = self.token_refiner(self.condition_proj(text_states),
                                                 transformer_options=transformer_options)

            # Consume each replicated row source in layout order. The modulation
            # plan has already refused segment kinds outside this map.
            stream_of = H3_STREAM_OF
            embed_of = {"video": video_embed, "audio": audio_embed}
            taken = {"video": 0, "audio": 0}
            h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
            for start, stop, kind in layout.segments:
                if kind == "text":
                    h[start:stop] = text_states
                    continue
                stream = stream_of[kind]
                cursor = taken[stream]
                h[start:stop] = embed_of[stream][cursor:cursor + stop - start]
                taken[stream] = cursor + stop - start
            del all_video_rows, all_audio_rows, video_embed, audio_embed, text_states, embed_of  # freed before the blocks, as comfy 2d6b7328 does
            t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
            t_emb = h3_timestep_embedding(self, t_vals, dtype)

            # Sequence-parallel seam: identically split and pad the stream,
            # position IDs, and modulation row indices. Position IDs are embedded
            # locally. At sp=1 the worker retains the stock ComfyUI forward.
            world = sp_world()
            padded_len = -(-layout.seq_len // world) * world
            row_index = pad_row_index(row_index, padded_len)

            h, seq_orig = shard_seq(h, dim=0)
            pos_local, _ = shard_seq(layout.position_ids, dim=0)
            rows_local, _ = shard_seq(row_index, dim=0)
            local_mod = row_runs(rows_local)
            rope_freqs = rope_rotation_table(self.rope_freqs(pos_local, device), dtype)
            # Exclude synthetic keys exactly; padded Ring or hybrid refuses here
            # before any block runs.
            drop_rows = padded_row_indices([(seq_orig, h.shape[0])])
            if drop_rows:
                _reject_padded_ring(len(drop_rows))
            usp_opts = usp_options(transformer_options, attn, drop_rows=drop_rows)

            # A sparse kernel that keeps one contiguous key range exact reads
            # the prefix here. The packed order puts every non-video row before
            # the video segment and divisibility pads are a tail, so this offset
            # is a full-sequence coordinate that survives pad exclusion.
            # Kernels without a sink ignore both scopes; a sequence with no video
            # segment declares none, and the sink-taking kernel refuses rather
            # than run another configuration. The per-step dense schedule, with
            # its timestep fallback, is sol_dense_step; SOL_DENSE_BLOCKS is the
            # per-block half.
            video_start = next(
                (a for a, _, kind in layout.segments if kind == "video"), None)
            dense_step = sol_dense_step(
                _sampler_step_position(transformer_options), t_v)
            with sol_sink_scope(video_start):
                prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
                    list(self.blocks), device, transformer_options)
                for index, block in enumerate(self.blocks):
                    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, block)
                    with sol_dense_scope(dense_step or index < SOL_DENSE_BLOCKS):
                        h = block(h, t_emb, local_mod, rope_freqs,
                                  transformer_options=usp_opts)
                if prefetch_queue is not None:
                    comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, device, None)

            # Gather and trim before absolute-coordinate head slicing. t_emb is
            # indexed by modulation row rather than token and remains replicated.
            h = sp_gather(h, seq_orig, dim=0)
            video_seg = next((a, b, video_row) for a, b, kind in layout.segments if kind == "video")
            audio_seg = next((a, b, audio_row) for a, b, kind in layout.segments if kind == "audio")
            # ComfyUI final-layer contract: full stream, one
            # (start, stop, modulation-row) span per modality, then the step the
            # PDD head bank blends over (comfy 2504e68d). A stacked head reads
            # the sampler's own table, so it comes from the stock
            # transformer_options rather than the sequence-parallel copy.
            video_head, audio_head = self.final_layer(
                h, t_emb, video_seg, audio_seg, sigma_v,
                transformer_options.get("sample_sigmas"), (shift_v, shift_a))

            video_out = unpatchify_video(video_head, latent_t, lat_h // 2, lat_w // 2,
                                         self.latents_dim, self.patch_size)
            video_out = video_out[:, :, :orig_t, :orig_h, :orig_w]
            audio_out = unpack_audio(audio_head)

            # Inner return contract: negated raw flow velocities. The outer
            # forward converts audio once; never convert it here.
            return [-video_out.to(video_x.dtype), -audio_out.to(audio_x.dtype)]

        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("minimax_h3 USP injected: %d blocks, sp=%d",
                 len(diffusion_model.blocks), ctx.topology_sp)
