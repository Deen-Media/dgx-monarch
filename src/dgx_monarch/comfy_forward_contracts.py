"""ComfyUI seam contracts used by adapter-owned model-family forwards.

``comfy_surface.TOUCHPOINTS`` includes four groups: helper leaves, rebound
methods, stock callers of rebound methods, and stock callees used by adapters.
Rebound methods declare exact parameters and variadics because prefix checks
cannot detect same-arity reordering or additions absorbed by ``**kwargs``.

``BROAD_ADAPTER_SUBCLASS_CONTRACTS`` maps broad ``isinstance`` branches to
their declared rebound targets. ``ADAPTER_BIND_SITES`` inventories both bind
forms, and tests require a bijection between sites and concrete contracts.
Together these declarations make inherited and conditional branches explicit
and fail the canary on incompatible upstream signature drift.
"""

from __future__ import annotations

from .comfy_rebound_signatures import (
    BROAD_ADAPTER_SUBCLASS_CONTRACTS,
    REBOUND_METHOD_CONTRACTS,
)

# Re-exported for callers that import them from here; the leaf modules own them.
from .comfy_rebound_signatures import (
    ReboundMethodContract as ReboundMethodContract,
)
from .comfy_rebound_sites import (
    ADAPTER_BIND_SITES as ADAPTER_BIND_SITES,
)
from .comfy_rebound_sites import (
    AdapterBindSite as AdapterBindSite,
)
from .comfy_touchpoint import Touchpoint, _attr, _call

# Concrete subclasses accepted through broad roots need independent canary
# coverage because each may select a distinct diffusion-model class.
ADDITIONAL_REBOUND_MODEL_BASE_TOUCHPOINTS: tuple[str, ...] = tuple(
    branch.model_base
    for contract in BROAD_ADAPTER_SUBCLASS_CONTRACTS
    for branch in contract.branches
    if branch.model_base != contract.root_model_base
)

# Helper leaves imported by adapter-owned model-family forwards.
HELPER_LEAF_TOUCHPOINTS: tuple[Touchpoint, ...] = (
    _call("comfy.ldm.common_dit", "pad_to_patch_size", positional=("img", "patch_size")),
    _call("comfy.ldm.flux.math", "apply_rope1", positional=("x", "freqs_cis")),
    _call("comfy.ldm.flux.layers", "timestep_embedding", "time_factor", positional=("t", "dim")),
    _call("comfy.ldm.wan.model", "sinusoidal_embedding_1d", positional=("dim", "position")),
    _call("comfy.ldm.cogvideo.model", "get_timestep_embedding", positional=("timesteps", "dim", "flip_sin_to_cos", "downscale_freq_shift")),
    _call("comfy.ldm.lightricks.model", "apply_rotary_emb", positional=("input_tensor", "freqs_cis")),
    # LTX guide attenuation. The adapter reads this object's two rectangles and
    # re-expresses them as query groups, so its identity is a live dependency.
    _attr("comfy.ldm.lightricks.model", "GuideAttentionMask"),
    # Stock's own kernel for a biased attention group: comfy sends those calls
    # through comfy.ldm.modules.attention.attention_pytorch, which calls this.
    _call("comfy.ops", "scaled_dot_product_attention", positional=("q", "k", "v")),
    _call("comfy.ldm.lightricks.av_model", "CompressedTimestep", positional=("tensor", "patches_per_frame")),
    _call("comfy.ldm.lightricks.av_model", "CompressedTimestep.expand", positional=("self",)),
    _call("comfy.ldm.lumina.model", "NextDiT", "pad_tokens_multiple"),
    _call("comfy.ldm.lens.model", "_lens_position_ids", "device", positional=("frame", "height", "width", "text_seq_len")),
    _call("comfy.ldm.pixeldit.modules", "apply_adaln_", positional=("x", "shift", "scale")),
    # Qwen Image 2.1's adapter uses the stock modulation splitter and exact
    # cache-key/fill helpers while it replaces target-row attention.
    _call("comfy.ldm.qwen_image21.model", "_split_rows", positional=("p",)),
    _call("comfy.ldm.qwen_image21.model", "prefix_cache_key",
          positional=("x", "context", "refs", "slots")),
    _call("comfy.ldm.qwen_image21.model", "block_causal_attention",
          positional=("segments",)),
    _attr("comfy.ldm.ideogram4.model", "LLM_TOKEN_INDICATOR"),
    _attr("comfy.ldm.ideogram4.model", "OUTPUT_IMAGE_INDICATOR"),
    _call("comfy.ldm.ideogram4.model", "_split_half_rope_matrix", positional=("freqs_cis",)),
    # FSDP imports this class directly; derived model-base and scan channels
    # cover its module-object imports member by member.
    _attr("comfy.ldm.wan.model", "WanModel"),
    _attr("comfy.ldm.wan.model_animate2", "WanAnimate2Model"),
    # Animate2 reconstructs global frame-major K/V, then calls Comfy's exact
    # attention surface for each local spatial query group.
    _attr("comfy.ldm.modules.attention", "AttentionTensorContainer"),
    # Qwen's direct block callback owns these wrappers before its custom
    # target K/V gather; take transfers their single-use tensor exactly once.
    _call("comfy.ldm.modules.attention", "AttentionTensorContainer.take",
          positional=("self",)),
    _attr("comfy.ldm.modules.attention", "optimized_attention"),
    # Exact ops members reached through FSDP's module-object import.
    _attr("comfy.ops", "disable_weight_init.Conv3d"),
    _attr("comfy.ops", "manual_cast.Conv3d"),
    # MiniMax H3 imports these packing, rope, audio-time, and prefetch helpers
    # directly. The behavioral seam suite covers the stock outer audio path.
    _call("comfy.ldm.minimax.model", "patchify_video", positional=("latent", "patch_size")),
    _call("comfy.ldm.minimax.model", "unpatchify_video", positional=("rows", "t", "h", "w", "c", "patch_size")),
    _call("comfy.ldm.minimax.model", "pack_audio", positional=("latent",)),
    _call("comfy.ldm.minimax.model", "unpack_audio", positional=("rows",)),
    _call("comfy.ldm.minimax.model", "rope_rotation_table", positional=("angles", "dtype")),
    _call("comfy.ldm.minimax.model", "time_shift_sigma", positional=("sigma", "from_shift", "to_shift")),
    _call("comfy.ldm.minimax.model", "PackedLayout", "keyframes", "refs",
          positional=("text_len", "latent_t", "latent_h", "latent_w", "audio_t")),
    _attr("comfy.ldm.minimax.model", "VISUAL_COND_TIMESTEP"),
    _attr("comfy.ldm.minimax.model", "AUDIO_COND_TIMESTEP"),
    _call("comfy.model_prefetch", "make_prefetch_queue",
          positional=("queue", "device", "transformer_options")),
    _call("comfy.model_prefetch", "prefetch_queue_pop", positional=("queue", "device", "module")),
    # Ernie's pure-Ulysses attention builds q and k on stock's inference
    # branch: the norm weights cast for comfy-kitchen's fused kernel, then
    # released, exactly as ErnieImageAttention.forward does.
    _attr("comfy.model_management", "in_training"),
    _call("comfy.ops", "cast_bias_weight", "offloadable", positional=("s", "input")),
    _call("comfy.ops", "uncast_bias_weight", positional=("s", "weight", "bias", "offload_stream")),
    _call("comfy.quant_ops", "ck.rms_rope_split_half",
          positional=("q", "k", "freqs_cis", "q_scale", "k_scale", "epsilon")),
)

REBOUND_FORWARD_TOUCHPOINTS: tuple[Touchpoint, ...] = tuple(
    contract.touchpoint for contract in REBOUND_METHOD_CONTRACTS
)


# Stock callers are call seams, not bind targets, so prefix checks are valid.
REBOUND_CALLER_TOUCHPOINTS: tuple[Touchpoint, ...] = (
    _call(
        "comfy.ldm.krea2.model",
        "SingleStreamDiT.forward",
        positional=(
            "self", "x", "timesteps", "context", "attention_mask", "ref_latents",
            "transformer_options",
        ),
    ),
    _call(
        "comfy.ldm.minimax.model",
        "MiniMaxH3Model.forward",
        "minimax_payload",
        positional=("self", "x", "timestep", "context", "transformer_options"),
    ),
)


# Stock blocks and helpers invoked by adapter-owned forwards.
REWRITTEN_FORWARD_CALLEE_TOUCHPOINTS: tuple[Touchpoint, ...] = (
    _call(
        "comfy.ldm.krea2.model",
        "SingleStreamBlock.forward",
        "transformer_options",
        positional=("self", "x", "vec", "freqs", "mask"),
    ),
    _call(
        "comfy.ldm.minimax.model",
        "DiTBlock.forward",
        "transformer_options",
        positional=("self", "x", "t_emb", "mod_segments", "rope_freqs"),
    ),
    # comfy 2504e68d made sigma, sample_sigmas and shifts required: a PDD LoRA
    # stacks head row blocks, and the head blends the blocks the current step
    # spans. The three values are the sampler's, not the head's, so the rebound
    # forward has to hand them down.
    _call(
        "comfy.ldm.minimax.model",
        "FinalLayer.forward",
        positional=("self", "x", "t_emb", "video_seg", "audio_seg",
                    "sigma", "sample_sigmas", "shifts"),
    ),
    _call(
        "comfy.ldm.minimax.model",
        "MiniMaxH3Model.rope_freqs",
        positional=("self", "position_ids", "device"),
    ),
    _call(
        "comfy.ldm.minimax.model",
        "MiniMaxH3Model._cond_video_rows",
        positional=("self", "payload", "device"),
    ),
    _call(
        "comfy.ldm.minimax.model",
        "MiniMaxH3Model._cond_audio_rows",
        positional=("self", "payload", "device"),
    ),
    _call(
        "comfy.ldm.minimax.model",
        "TokenRefiner.forward",
        "transformer_options",
        positional=("self", "x"),
    ),
)


FAMILY_FORWARD_TOUCHPOINTS: tuple[Touchpoint, ...] = (
    HELPER_LEAF_TOUCHPOINTS
    + REBOUND_FORWARD_TOUCHPOINTS
    + REBOUND_CALLER_TOUCHPOINTS
    + REWRITTEN_FORWARD_CALLEE_TOUCHPOINTS
)
