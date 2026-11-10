"""MiniMax-H3-Image adapter: the H3 DiT with no audio stream, denoising one latent frame.

ComfyUI 0.45.0 builds MiniMax-H3-Image as ``MiniMaxH3Image`` around
``MiniMaxH3ImageModel``, which packs ``[text | reference pictures | target]``
into one dim-0 token stream and samples a stacked PDD output head on the
checkpoint's own ``pdd_sigmas`` grid. Under sequence parallelism this adapter
rebinds ``_forward`` and keeps stock's ``forward`` (its malloc graph and
wrapper executor). Every rank runs stock's own ``_embed_and_pack`` and
``_time_embedding``, so the replicated stream and the timestep rows are stock's
by construction. The stream, the position ids and the per-token modulation rows
then take identical sequence shards; Ulysses excludes the divisibility pad rows
exactly, and padded Ring or hybrid execution refuses. The blocks run under
stock's block-scope prefetch pops with stock's ``block_index``. The gathered
stream reaches the head at stock's rows, with the checkpoint's grid.

At one rank nothing is injected: stock's forward runs, header attention pins
included. Under sequence parallelism no attention call runs a pin
(attention_pins.py). The image model ignores a latent noise mask (the sampler
applies it), so this forward ignores it as well. cfg-parallel and data parallel
refuse before dispatch (nodes/render_preflight.py), as for H3.
"""
from __future__ import annotations

from typing import Any

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
    assert_image_grid,
    h3_image_modulation_plan,
    head_form,
    head_schedule,
    pad_row_index,
    row_runs,
)
from .stock_attention import bind_stock_attention

log = get_logger(__name__)

FAMILY = "minimax_h3_image"
# Exact ComfyUI model class for this family. MiniMaxH3Image subclasses
# MiniMaxH3, whose adapter raises on any subclass, so this one registers first.
_MINIMAX_H3_IMAGE_EXACT = ("MiniMaxH3Image",)

MINIMAX_H3_IMAGE_BATCH_CAPPED_MESSAGE = refusal(
    RefusalClass.PHYSICS,
    "minimax_h3_image: cfg-parallel and data-parallel cannot run this family. Every "
    "conditioning publishes its own packed-layout object, so ComfyUI never merges cond and "
    "uncond into one batched call, and cfg-parallel has nothing to split; the DiT accepts "
    "batch 1 only, so data parallel has no batch to divide. The model is CFG distilled. Run "
    "this workflow on a uly* topology, or take a true single-GPU reference (mode=local, "
    "gpus_per_host=1). The 'single' preset is not that reference: above world 1 it derives "
    "dp and refuses here too (docs/MODELS.md).",
)


def _reject_unsupported_image_features(x: Any, transformer_options: dict[str, Any],
                                       kwargs: dict[str, Any]) -> None:
    """Refuse, class P, inputs this forward cannot shard, before any collective.

    Each reads the request, which every rank holds alike.
    """
    shape = getattr(x, "shape", None)
    if shape is not None and len(shape) and int(shape[0]) != 1:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"minimax_h3_image: this render has a batch of {int(shape[0])}. MiniMax-H3-Image "
            "renders one image per sample, as stock ComfyUI does. Use a batch size of 1.",
        ))
    for name in ("attention_mask", "attn_mask"):
        if kwargs.get(name) is not None:
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                f"minimax_h3_image: this render carries a '{name}', but the model's attention "
                "is maskless and fully bidirectional, so the sharded kernel cannot apply one. "
                "Clear the mask, or take a true single-GPU reference (mode=local, "
                "gpus_per_host=1) (docs/MODELS.md).",
            ))
    if transformer_options.get("patches_replace", {}).get("dit", {}):
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "minimax_h3_image: a 'dit' block replacement receives the token stream, the "
            "rotation table and the modulation segment table, all three of which are "
            "rank-local under sequence parallelism, so the patch would see a partial sequence "
            "and rank-local coordinates. Remove the patch, or take a true single-GPU "
            "reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
        ))


def _reject_padded_ring(n_pad_rows: int) -> None:
    """Refuse, class P, a padded Ring or hybrid render: the exact pad exclusion is Ulysses-only."""
    from xfuser.core.distributed import get_ring_parallel_world_size

    if get_ring_parallel_world_size() > 1:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            f"minimax_h3_image: this packed sequence needs {n_pad_rows} divisibility pad "
            "row(s), and the exact pad exclusion is Ulysses-only. The model's attention is "
            "maskless and fully bidirectional, so ring or hybrid attention would attend the "
            "synthetic rows. Run this workflow on a uly* topology, or take a true single-GPU "
            "reference (mode=local, gpus_per_host=1) (docs/MODELS.md).",
        ))


class MiniMaxH3ImageAdapter(Adapter):
    family = "minimax_h3_image"  # a literal: the Studio inventory reads it statically
    model_base_classes = _MINIMAX_H3_IMAGE_EXACT
    exact_model_base_classes = _MINIMAX_H3_IMAGE_EXACT
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        cls = getattr(model_base, _MINIMAX_H3_IMAGE_EXACT[0], None)
        if cls is None or not isinstance(base_model, cls):
            return False
        if type(base_model) is cls:
            return True
        # Exact-type matching keeps an unvetted subclass from inheriting support.
        raise UnsupportedModelError(
            f"minimax_h3_image variant {type(base_model).__name__} is a MiniMaxH3Image "
            "subclass this adapter has not vetted. Take a true single-GPU reference "
            "(mode=local, gpus_per_host=1) (docs/MODELS.md)."
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        attn = ctx.usp_attention
        # Both read once, before any bind: the head's form and its grid.
        form = head_form(diffusion_model.final_layer)
        assert_image_grid(diffusion_model.final_layer, diffusion_model.pdd_sigmas)

        def usp_forward(self, x, timestep, context, transformer_options={},
                        minimax_payload=None, **kwargs):
            # transformer_options keeps the fourth positional slot the wrapper
            # executor passes; a denoise mask lands in kwargs and is ignored,
            # as in stock's image forward.
            _reject_unsupported_image_features(x, transformer_options, kwargs)

            import comfy.ldm.common_dit
            import comfy.model_prefetch
            from comfy.ldm.minimax.model import (
                VISUAL_COND_TIMESTEP,
                PackedLayout,
                rope_rotation_table,
                unpatchify_video,
            )

            orig_h, orig_w = x.shape[3], x.shape[4]
            x = comfy.ldm.common_dit.pad_to_patch_size(x, self.patch_size)
            payload = minimax_payload or {}
            device = x.device
            dtype = context.dtype

            latent_t, lat_h, lat_w = x.shape[2], x.shape[3], x.shape[4]
            text_len = context.shape[1]
            layout = payload.get("layout")
            if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, 0):
                layout = PackedLayout(text_len, latent_t, lat_h, lat_w, 0,
                                      refs=payload.get("refs"))
            transformer_options["minimax_h3_layout"] = layout

            # ComfyUI passes sigma * 1000; the image model reads it unclamped.
            sigma = float(timestep.flatten()[0]) / 1000.0
            t = 1.0 - sigma
            unique_t, row_index, target_row = h3_image_modulation_plan(
                layout, payload, t, VISUAL_COND_TIMESTEP)

            # Stock's own packing and timestep rows, on every rank: the stream
            # is stock's before the shard. The token refiner inside sees the
            # stock options; the sharded override is built below.
            h = self._embed_and_pack(x, None, context, layout, payload, transformer_options)
            t_emb = self._time_embedding(unique_t, device, dtype)

            world = sp_world()
            padded_len = -(-layout.seq_len // world) * world
            row_index = pad_row_index(row_index, padded_len)
            h, seq_orig = shard_seq(h, dim=0)
            pos_local, _ = shard_seq(layout.position_ids, dim=0)
            rows_local, _ = shard_seq(row_index, dim=0)
            local_mod = row_runs(rows_local)
            rope_freqs = rope_rotation_table(self.rope_freqs(pos_local, device), h.dtype)
            drop_rows = padded_row_indices([(seq_orig, h.shape[0])])
            if drop_rows:
                _reject_padded_ring(len(drop_rows))
            usp_opts = usp_options(transformer_options, attn, drop_rows=drop_rows)

            prefetch_queue = comfy.model_prefetch.make_prefetch_queue(
                list(self.blocks), device, transformer_options)
            for index, block in enumerate(self.blocks):
                comfy.model_prefetch.prefetch_queue_pop(
                    prefetch_queue, device, block, malloc_scope="block")
                # Stock sets the index on the options its blocks receive.
                transformer_options["block_index"] = index
                usp_opts["block_index"] = index
                h = block(h, t_emb, local_mod, rope_freqs, transformer_options=usp_opts)
            comfy.model_prefetch.prefetch_queue_pop(
                prefetch_queue, device, None, malloc_scope="block")

            # Gather and trim before the head slices the target by absolute row.
            h = sp_gather(h, seq_orig, dim=0)
            start, stop, _kind = layout.segments[-1]  # the target is the last segment
            video_head, = self.final_layer(
                h, t_emb, (start, stop, target_row), None, sigma,
                transformer_options.get("sample_sigmas"),
                head_schedule(form, self.final_layer, (), None, grid=self.pdd_sigmas))
            out = unpatchify_video(video_head, latent_t, lat_h // 2, lat_w // 2,
                                   self.latents_dim, self.patch_size)
            return -out[:, :, :, :orig_h, :orig_w].to(x.dtype)

        # ComfyUI runs the token refiner from extra_conds as well, with no
        # transformer_options; its stock attention must ignore a pinned kernel.
        for block in getattr(getattr(diffusion_model, "token_refiner", None), "blocks", ()):
            bind_stock_attention(block.attn, "minimax_h3_image token refiner attention")
        self.bind(diffusion_model, "_forward", usp_forward)
        log.info("minimax_h3_image USP injected: %d blocks, sp=%d",
                 len(diffusion_model.blocks), ctx.topology_sp)
