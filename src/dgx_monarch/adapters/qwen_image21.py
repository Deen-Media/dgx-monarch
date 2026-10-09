"""Qwen-Image-2.1 adapter.

2.1 is separate from the older dual-stream Qwen Image adapter. Its one-stream
block-causal transformer splices text and zero-to-ten reference images into a
prefix before the target image. That prefix is not an ordinary maskless
sequence: text tokens are causal while each image segment attends to itself and
all preceding tokens. The generic Ulysses hook cannot express that per-query
causal boundary, so it must never be installed as if it were the older Qwen
Image joint attention path.

The stock model's prefix cache is exact only for the complete, unchanged
prefix. Each distributed run starts with the cache off (actor/qwen_image21_cache.py)
until the Qwen Image 2.1 Cache (DGX Monarch) node opts in. The distributed cache
keeps prefix K/V replicated on every rank, keys it on comfy's native prefix key
plus the sequence-parallel cohort, and has every rank take the same branch.
"""
from __future__ import annotations

from typing import Any

import torch

from ..log import get_logger
from ..refusal import RefusalClass, refusal
from . import base
from .attention_patches import assert_no_foreign_attention_override
from .base import (
    Adapter,
    InjectionContext,
    UnsupportedModelError,
    shard_seq,
    sp_gather,
)

log = get_logger(__name__)


def _take_native_attention_tensor(value: Any) -> torch.Tensor:
    """Take Comfy's documented single-owner wrapper, or retain a plain tensor."""
    if isinstance(value, torch.Tensor):
        return value
    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer
    except ImportError as exc:
        raise TypeError("qwen_image21 attention callback received a non-tensor input") from exc
    if isinstance(value, AttentionTensorContainer):
        return value.take()
    raise TypeError("qwen_image21 attention callback received an unsupported attention input")


def validate_reference_count(ref_latents) -> tuple:
    """Normalize the stock optional reference list and enforce its public bound."""
    refs = tuple(ref_latents or ())
    if len(refs) > 10:
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                f"qwen_image21 accepts at most 10 reference images, got {len(refs)}. "
                "Use at most 10 ordered reference images.",
            )
        )
    return refs


def prefix_cache_is_exact(*, enabled: bool, hooks_present: bool,
                          refs, image_slots, cache_owner: str | None) -> bool:
    """Return whether an explicitly scoped prefix cache can be reused.

    The cache entries are replicated on every sequence-parallel rank: each rank
    needs the full prefix K/V while it owns only target-query rows.
    ``cache_owner`` names that cohort, not a process-local cache; the key
    (``distributed_prefix_cache_key``) adds the sequence-parallel degree, world
    and rank to comfy's native prefix key. This guards reuse identity only:
    comfy's optional int8 and int4 cache storage is lossy by design. Raises on
    more than 10 references, or more image slots than references.
    """
    validate_reference_count(refs)
    if image_slots is not None and len(image_slots) > len(tuple(refs or ())):
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                "qwen_image21 image_slots has more entries than reference images. "
                "Use one slot per reference image.",
            )
        )
    return bool(enabled and not hooks_present and cache_owner == "distributed")


def distributed_prefix_cache_key(native_key: torch.Tensor, *, topology_sp: int,
                                 world: int, rank: int) -> torch.Tensor:
    """Bind a native full-prefix key to the exact sequence-parallel cohort.

    Comfy's native key carries every prefix-affecting value (text embeddings,
    reference latent bytes, their slots, and target geometry). It omits target
    *values*, because denoising changes them on every step. A replicated K/V
    entry cannot cross a topology or rank assignment boundary, so append that
    identity without replacing native equality semantics.
    """
    if native_key.ndim != 2:
        raise UnsupportedModelError(refusal(
            RefusalClass.PHYSICS,
            "qwen_image21 native prefix key must be rank two. "
            "Use the supported native Qwen Image 2.1 conditioning path.",
        ))
    if topology_sp < 1 or world != topology_sp or not 0 <= rank < world:
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                "qwen_image21 prefix cache requires the live sequence-parallel cohort "
                "to match the adapter topology. Use a fresh topology whose sequence-"
                "parallel degree matches the active worker cohort.",
            )
        )
    cohort = native_key.new_tensor((topology_sp, world, rank)).expand(native_key.shape[0], -1)
    return torch.cat((native_key, cohort), dim=1)


def agree_prefix_cache(cache: Any, cached: bool, device: torch.device) -> tuple[Any, bool]:
    """Choose one cache structure for the complete sequence-parallel cohort.

    Native cache selection may vary by rank because its VRAM and pinned-memory
    checks are local.  Every rank must nevertheless take the same prefix
    branch: any unavailable cache recomputes uncached everywhere; a mixed
    filled/unfilled cohort refills every selected slot.  ``PoseBranchCache.put``
    replaces an existing block entry, so refilling a selected native hit is
    supported without transferring cache ownership between ranks.
    """
    local = torch.tensor((cache is not None, cached), dtype=torch.bool, device=device)
    cohort = sp_gather(local, 2 * base.sp_world(), dim=0).reshape(base.sp_world(), 2)
    if not bool(cohort[:, 0].all()):
        return None, False
    return cache, bool(cohort[:, 1].all())


def _optimized_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int,
                         *, mask: torch.Tensor | None, transformer_options: dict,
                         preferred_attention: Any) -> torch.Tensor:
    """Run Comfy's selected attention backend and restore Qwen's BNHD layout."""
    # Import at call time because CPU-only unit tests import this adapter
    # without a Comfy checkout. This is the exact stock Qwen entry point,
    # including its selected flash/SDPA fallback and patch dispatch.
    from comfy.ldm.modules.attention import optimized_attention

    out = optimized_attention(
        q.flatten(2), k.flatten(2), v.flatten(2), heads,
        mask=mask,
        transformer_options=transformer_options,
        preferred_attention=preferred_attention,
    )
    return out.unflatten(-1, (heads, q.shape[-1]))


def cached_target_attention(prefix_k: torch.Tensor, prefix_v: torch.Tensor,
                            q_target: torch.Tensor, k_target: torch.Tensor,
                            v_target: torch.Tensor, heads: int, *,
                            transformer_options: dict, preferred_attention: Any) -> torch.Tensor:
    """Native attention for local target Q against replicated prefix and full target K/V."""
    return _optimized_attention(
        q_target, torch.cat((prefix_k, k_target), dim=1), torch.cat((prefix_v, v_target), dim=1), heads,
        mask=None, transformer_options=transformer_options, preferred_attention=preferred_attention,
    )


def block_causal_target_attention(q_prefix, k_prefix, v_prefix, q_target,
                                  k_target, v_target, prefix_segments, heads: int, *,
                                  transformer_options: dict, preferred_attention: Any):
    """Exact block-causal attention split into replicated prefix and target rows.

    ``prefix_segments`` is the native sequence's ``(start, end, mask)``
    layout. Target image queries see the complete prefix and complete target
    sequence. The prefix stays replicated, the caller all-gathers target K/V,
    and only this rank's target Q rows reach Comfy's selected attention backend.
    """
    p = q_prefix.shape[1]
    prefix_out = []
    for start, end, mask in prefix_segments:
        if not (0 <= start <= end <= p):
            raise UnsupportedModelError(refusal(
                RefusalClass.PHYSICS,
                "qwen_image21 prefix segment is outside the prefix. "
                "Use the supported native Qwen Image 2.1 conditioning path.",
            ))
        prefix_out.append(_optimized_attention(
            q_prefix[:, start:end], k_prefix[:, :end], v_prefix[:, :end], heads,
            mask=mask, transformer_options=transformer_options,
            preferred_attention=preferred_attention,
        ))
    target_out = _optimized_attention(
        q_target, torch.cat((k_prefix, k_target), dim=1), torch.cat((v_prefix, v_target), dim=1), heads,
        mask=None, transformer_options=transformer_options, preferred_attention=preferred_attention,
    )
    return (torch.cat(prefix_out, dim=1) if prefix_out else torch.empty_like(q_prefix)), target_out


class QwenImage21Adapter(Adapter):
    """Bind only ComfyUI's exact QwenImage21 model-base class."""

    family = "qwen_image21"
    model_base_classes = ("QwenImage21",)
    # Padding text changes every later causal position and reference slot; do
    # not borrow the legacy Qwen Image cfg-pad rule.
    cfg_cond_padding = "none"

    def matches(self, base_model) -> bool:
        import comfy.model_base as model_base

        if not super().matches(base_model):
            return False
        cls = getattr(model_base, "QwenImage21", None)
        if cls is not None and type(base_model) is cls:
            return True
        raise UnsupportedModelError(
            refusal(
                RefusalClass.PHYSICS,
                f"qwen_image21 variant {type(base_model).__name__} is not in the "
                "dgx-monarch launch set (plain Qwen-Image-2.1 only). "
                "Use the plain Qwen-Image-2.1 checkpoint.",
            )
        )

    def inject_usp(self, diffusion_model, ctx: InjectionContext) -> None:
        # Do not retain a slot from a prior injection. Comfy enables its
        # one-sampling-run cache when it attaches the active patcher; the
        # actor's cache policy (actor/qwen_image21_cache.py) stays device=off
        # until the cache node opts in, so this reset is not an implicit enable.
        reset = getattr(diffusion_model, "reset_prefix_cache", None)
        if callable(reset):
            reset(False)

        def usp_forward(self, x, timesteps, context, ref_latents=None, image_slots=None,
                        transformer_options={}, **kwargs):
            # This path keeps local target Q and gathers target K/V itself, then
            # calls native Comfy attention. It does not call the xFuser
            # dispatcher, so any worker kernel but TORCH_FLASH refuses here,
            # before building a sequence or reaching a collective.
            if getattr(ctx.usp_attention, "effective_kernel", "") != "TORCH_FLASH":
                raise UnsupportedModelError(
                    refusal(
                        RefusalClass.PHYSICS,
                        "qwen_image21 Ulysses calls native Comfy attention and requires "
                        "TORCH_FLASH. Use TORCH_FLASH or topology 'single'.",
                    )
                )
            assert_no_foreign_attention_override(transformer_options, "qwen_image21")
            if transformer_options.get("patches") or transformer_options.get("patches_replace"):
                raise UnsupportedModelError(
                    refusal(
                        RefusalClass.PHYSICS,
                        "qwen_image21 USP does not admit Comfy block or attention hooks: "
                        "they can change the prefix sequence between ranks. Use a hook-free "
                        "Qwen Image 2.1 workflow.",
                    )
                )
            refs = validate_reference_count(ref_latents)
            B, _C, H, W = x.shape
            image_slots = list(image_slots or [])
            states, pe, segments = self.build_sequence(x, context, list(refs), image_slots)
            prefix_len = states.shape[1] - H * W
            prefix, target = states[:, :prefix_len], states[:, prefix_len:]
            # Comfy lays RoPE out as (B, sequence, heads, rotary-pairs).
            prefix_pe, target_pe = pe[:, :prefix_len], pe[:, prefix_len:]
            target, target_len = shard_seq(target, dim=1)
            target_pe, _ = shard_seq(target_pe, dim=1)
            states = torch.cat((prefix, target), dim=1)
            pe = torch.cat((prefix_pe, target_pe), dim=1)
            dtype = x.dtype
            t = ((timesteps * 1000).to(dtype) / 1000).to(dtype)
            temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), dtype)
            scale1, gate1, scale2, gate2 = self.modulation(temb).chunk(4, dim=-1)
            from comfy.ldm.qwen_image21.model import _split_rows
            mod = (_split_rows(scale1), _split_rows(gate1.tanh()), _split_rows(scale2),
                   _split_rows(gate2.tanh()), torch.zeros_like(scale1[:1, None]))
            prefix_segments = segments[:-1]

            cache = None
            cached = False
            cache_options = transformer_options.get("qwen_image21_cache", {"device": "off"})
            if not isinstance(cache_options, dict):
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    "qwen_image21 cache options must be a mapping. "
                    "Use the Qwen Image 2.1 Cache (DGX Monarch) node.",
                ))
            cache_device = cache_options.get("device", "off")
            cache_dtype = cache_options.get("dtype", "default")
            if cache_device not in ("auto", "gpu", "cpu", "off") or cache_dtype not in ("default", "int8", "int4"):
                raise UnsupportedModelError(refusal(
                    RefusalClass.PHYSICS,
                    "qwen_image21 cache device or dtype is unsupported. "
                    "Use a device and dtype the Qwen Image 2.1 Cache (DGX Monarch) node lists.",
                ))
            hooks_present = bool(transformer_options.get("patches") or transformer_options.get("patches_replace"))
            if cache_device != "off" and prefix_cache_is_exact(
                enabled=bool(getattr(self, "prefix_cache_enabled", False)),
                hooks_present=hooks_present,
                refs=refs,
                image_slots=image_slots,
                cache_owner="distributed",
            ):
                from comfy.ldm.qwen_image21.model import prefix_cache_key

                key = distributed_prefix_cache_key(
                    prefix_cache_key(x, context, list(refs), image_slots),
                    topology_sp=ctx.topology_sp,
                    # Resolve at call time: setup installs xFuser's group after
                    # this adapter module has imported, and CPU differentials
                    # replace the same two accessors with a Gloo facade.
                    world=base.sp_world(),
                    rank=base.sp_rank(),
                )
                cache_bytes = (2 * len(self.transformer_blocks) * B * prefix_len
                               * self.inner_dim * states.element_size())
                cache, cached = self.select_prefix_cache(
                    key, cache_bytes, x.device,
                    {"device": cache_device, "dtype": cache_dtype},
                )
            # Cache eligibility and selection are local (the native enable flag,
            # comfy's memory checks), but a requested cache needs one forward
            # structure across the SP group. An ineligible rank contributes ``None``.
            if cache_device != "off":
                cache, cached = agree_prefix_cache(cache, cached, x.device)
                log.info("Qwen Image 2.1 prefix cache: selected=%s reused=%s device=%s dtype=%s",
                         cache is not None, cached, cache_device, cache_dtype)

            # In cache mode, prefix K/V remain replicated and only the target
            # rows are sharded.  A cache hit has no prefix hidden states at all;
            # a miss evaluates those states once per rank and writes each
            # block's native K/V entry before target attention consumes it.
            prefix_states = prefix_pe = None
            if cached:
                states, pe = states[:, prefix_len:], pe[:, prefix_len:]
                active_prefix_len = 0
            elif cache is not None:
                prefix_states, states = states[:, :prefix_len], states[:, prefix_len:]
                prefix_pe, pe = pe[:, :prefix_len], pe[:, prefix_len:]
                active_prefix_len = 0
            else:
                active_prefix_len = prefix_len

            for index, block in enumerate(self.transformer_blocks):
                if cache is not None:
                    if not cached:
                        from comfy.ldm.qwen_image21.model import block_causal_attention

                        prefix_attn = block_causal_attention(
                            segments[:-1], transformer_options, cache, index,
                            prefix_states.shape[1])
                        prefix_states = block(
                            prefix_states, mod, prefix_pe, prefix_attn,
                            prefix_states.shape[1], transformer_options)
                    prefix_k, prefix_v = cache.take(index, x.device, dtype, B).unbind(1)

                    def attn(q, k, v, heads, *, _prefix_k=prefix_k,
                             _prefix_v=prefix_v, **_attn_kwargs):
                        q, k, v = (_take_native_attention_tensor(q),
                                   _take_native_attention_tensor(k),
                                   _take_native_attention_tensor(v))
                        return cached_target_attention(
                            _prefix_k, _prefix_v, q,
                            sp_gather(k, target_len, dim=1),
                            sp_gather(v, target_len, dim=1),
                            heads,
                            transformer_options=transformer_options,
                            preferred_attention=_attn_kwargs.get("preferred_attention"),
                        ).flatten(2)
                else:
                    def attn(q, k, v, heads, **_attn_kwargs):
                        q, k, v = (_take_native_attention_tensor(q),
                                   _take_native_attention_tensor(k),
                                   _take_native_attention_tensor(v))
                        q_p, q_t = q[:, :active_prefix_len], q[:, active_prefix_len:]
                        k_p, k_t = k[:, :active_prefix_len], k[:, active_prefix_len:]
                        v_p, v_t = v[:, :active_prefix_len], v[:, active_prefix_len:]
                        k_full = sp_gather(k_t, target_len, dim=1)
                        v_full = sp_gather(v_t, target_len, dim=1)
                        p_out, t_out = block_causal_target_attention(
                            q_p, k_p, v_p, q_t, k_full, v_full, prefix_segments, heads,
                            transformer_options=transformer_options,
                            preferred_attention=_attn_kwargs.get("preferred_attention"),
                        )
                        return torch.cat((p_out, t_out), dim=1).flatten(2)
                states = block(states, mod, pe, attn, active_prefix_len, transformer_options)
            target = sp_gather(states[:, active_prefix_len:], target_len, dim=1)
            target = self.proj_out(self.norm_out(target, temb[:-1]))
            return target.transpose(1, 2).reshape(B, self.out_channels, H, W)

        self.bind(diffusion_model, "_forward", usp_forward)
