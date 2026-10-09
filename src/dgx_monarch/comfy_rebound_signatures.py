"""Exact stock signatures for every concrete Comfy method adapters rebind."""

from __future__ import annotations

from dataclasses import dataclass

from .comfy_touchpoint import Touchpoint, _call


@dataclass(frozen=True)
class ReboundMethodContract:
    """Exact stock signature for one concrete class method an adapter rebinds."""

    module: str
    attribute: str
    families: tuple[str, ...]
    branches: tuple[str, ...]
    positional_only: tuple[str, ...] = ()
    positional_or_keyword: tuple[str, ...] = ()
    required_positional: int = 0
    keyword_only: tuple[str, ...] = ()
    required_keyword_only: tuple[str, ...] = ()
    variadic_positional: str | None = None
    variadic_keyword: str | None = None
    # Optional positional-or-keyword names a newer ComfyUI appended after the
    # declared list, so trees on both sides of that change pass.
    optional_trailing: tuple[str, ...] = ()
    # Optional positional-or-keyword names a newer ComfyUI inserted just before
    # the last declared name (transformer_options in every seam so far).
    # optional_trailing cannot say this: it only appends past the end.
    optional_infix: tuple[str, ...] = ()

    @property
    def path(self) -> str:
        return f"{self.module}.{self.attribute}"

    @property
    def touchpoint(self) -> Touchpoint:
        return _call(
            self.module,
            self.attribute,
            *self.keyword_only,
            positional=self.positional_only + self.positional_or_keyword,
        )


@dataclass(frozen=True)
class BroadAdapterBranch:
    """One concrete model-base branch admitted by a broad adapter root."""

    model_base: str
    rebound_targets: tuple[str, ...]


@dataclass(frozen=True)
class BroadAdapterSubclassContract:
    """Pinned accepted subclasses for an adapter using a broad isinstance gate."""

    family: str
    root_model_base: str
    branches: tuple[BroadAdapterBranch, ...]


def _rebound(
    module: str,
    attribute: str,
    families: tuple[str, ...],
    branches: tuple[str, ...],
    positional: str,
    required: int,
    *,
    variadic_keyword: str | None = None,
    optional_trailing: tuple[str, ...] = (),
    optional_infix: tuple[str, ...] = (),
) -> ReboundMethodContract:
    return ReboundMethodContract(
        module=module,
        attribute=attribute,
        families=families,
        branches=branches,
        positional_or_keyword=tuple(positional.split()),
        required_positional=required,
        variadic_keyword=variadic_keyword,
        optional_trailing=optional_trailing,
        optional_infix=optional_infix,
    )


# Two paths share another path's implementation today: ChromaRadiance inherits
# Chroma's forward_orig and SCAIL2 inherits SCAIL's. ChromaRadiance's `_forward`
# is its own override. Every path stays explicit so a future subclass override
# cannot escape the exact contract.
REBOUND_METHOD_CONTRACTS: tuple[ReboundMethodContract, ...] = (
    _rebound(
        "comfy.ldm.anima.model", "Anima._forward", ("anima",), ("usp",),
        "self x timesteps context fps padding_mask", 4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.boogu.model", "BooguTransformer2DModel.forward", ("boogu",),
        ("usp", "cfg-pad"),
        "self x timesteps context num_tokens ref_latents attention_mask transformer_options",
        5, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.cogvideo.model", "CogVideoXTransformer3DModel._forward",
        ("cogvideo",), ("usp",), "self x timestep context ofs transformer_options", 4,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.ernie.model", "ErnieImageModel.forward", ("ernie",), ("usp",),
        "self x timesteps context", 4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.ernie.model", "ErnieImageAttention.forward", ("ernie",),
        ("usp-self-attention",), "self x attention_mask image_rotary_emb", 2,
    ),
    _rebound(
        "comfy.ldm.chroma.model", "Chroma.forward_orig", ("chroma",), ("usp",),
        "self img img_ids txt txt_ids timesteps guidance control transformer_options attn_mask",
        6,
    ),
    _rebound(
        "comfy.ldm.chroma_radiance.model", "ChromaRadiance.forward_orig",
        ("chroma",), ("usp:radiance-inherited",),
        "self img img_ids txt txt_ids timesteps guidance control transformer_options attn_mask",
        6,
    ),
    _rebound(
        "comfy.ldm.chroma.model", "Chroma._forward", ("chroma",), ("cfg-pad",),
        "self x timestep context guidance control transformer_options", 5,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.chroma_radiance.model", "ChromaRadiance._forward", ("chroma",),
        ("cfg-pad:radiance",),
        "self x timestep context guidance control transformer_options", 5,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.flux.model", "Flux.forward_orig", ("flux", "flux2", "longcat"),
        ("usp",),
        "self img img_ids txt txt_ids timesteps y guidance control timestep_zero_index "
        "transformer_options attn_mask",
        7,
    ),
    _rebound(
        "comfy.ldm.hunyuan_video.model", "HunyuanVideo.forward_orig", ("hunyuan",),
        ("usp:image", "usp:video", "usp:refiner", "usp:sr"),
        "self img img_ids txt txt_ids txt_mask timesteps y txt_byt5 clip_fea guidance "
        "guiding_frame_index ref_latent disable_time_r control transformer_options",
        7,
    ),
    _rebound(
        "comfy.ldm.kandinsky5.model", "Kandinsky5.forward_orig", ("kandinsky5",),
        ("usp",), "self x timestep context y freqs freqs_text transformer_options", 7,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.kandinsky5.model", "Kandinsky5._forward", ("kandinsky5",),
        ("cfg-pad",), "self x timestep context y time_dim_replace transformer_options", 5,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.kandinsky5.model", "SelfAttention.forward", ("kandinsky5",),
        ("usp-self-attention",), "self x freqs transformer_options", 3,
    ),
    _rebound(
        "comfy.ldm.krea2.model", "SingleStreamDiT._forward", ("krea2",),
        ("usp", "cfg-pad"),
        "self x timesteps context attention_mask ref_latents transformer_options", 4,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.lens.model", "LensTransformer2DModel._forward", ("lens",),
        ("usp", "cfg-pad"),
        "self x timestep context attention_mask transformer_options control", 4,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.lightricks.model", "LTXVModel._process_transformer_blocks", ("ltx",),
        ("ltxv",),
        "self x context attention_mask timestep pe transformer_options self_attention_mask", 6,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.lightricks.av_model", "LTXAVModel._process_transformer_blocks",
        ("ltx",), ("ltxav",),
        "self x context attention_mask timestep pe transformer_options self_attention_mask", 6,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.lightricks.model", "CrossAttention.forward", ("ltx",),
        ("ltxv-self", "ltxav-self", "ltxav-audio-video-cross"),
        "self x context mask pe k_pe transformer_options", 2,
    ),
    # Comfy ff6c8a8 appended the two latent noise-mask parameters. Trees on
    # either side of that commit are both admitted; the adapter refuses a mask
    # that is actually set rather than dropping it into its own **kwargs.
    _rebound(
        "comfy.ldm.minimax.model", "MiniMaxH3Model._forward", ("minimax_h3",),
        ("usp",), "self x timestep context transformer_options minimax_payload", 4,
        variadic_keyword="kwargs",
        optional_trailing=("denoise_mask", "audio_denoise_mask"),
    ),
    _rebound(
        "comfy.ldm.omnigen.omnigen2", "OmniGen2Transformer2DModel.forward", ("omnigen2",),
        ("usp",),
        "self x timesteps context num_tokens ref_latents attention_mask transformer_options", 5,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.ideogram4.model", "Ideogram4Transformer2DModel._forward",
        ("ideogram4",), ("usp",),
        "self x timesteps context attention_mask transformer_options", 3,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.pixeldit.model", "PixDiT_T2I._forward", ("pixeldit_comfy",),
        ("usp:pixeldit",), "self x timesteps context attention_mask transformer_options", 3,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.pixeldit.pid", "PidNet._forward", ("pixeldit_comfy",), ("usp:pid",),
        "self x timesteps context attention_mask transformer_options lq_latent degrade_sigma", 3,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.pixeldit.modules", "PiTBlock.forward", ("pixeldit_comfy",),
        ("usp-pixel-block",),
        "self x s_cond image_height image_width patch_size mask transformer_options", 6,
    ),
    _rebound(
        "comfy.ldm.mage_flow.model", "MageFlowTransformer2DModel._forward",
        ("mage_flow",), ("usp", "cfg-pad"),
        "self x timestep context attention_mask ref_latents transformer_options control",
        4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.qwen_image.model", "QwenImageTransformer2DModel._forward",
        ("qwen_image",), ("usp", "cfg-pad"),
        "self x timesteps context attention_mask ref_latents additional_t_cond "
        "transformer_options control",
        4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.qwen_image21.model", "QwenImage21Transformer2DModel._forward",
        ("qwen_image21",), ("usp",),
        "self x timesteps context ref_latents image_slots transformer_options",
        4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.wan.model", "WanModel.forward_orig", ("wan",), ("usp",),
        "self x t context clip_fea freqs transformer_options", 4,
        variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.wan.model", "WanSelfAttention.forward",
        ("wan", "wan_scail", "wan_dancer"), ("usp-self-attention",),
        "self x freqs transformer_options", 3,
    ),
    _rebound(
        "comfy.ldm.wan.model", "SCAILWanModel.forward_orig", ("wan_scail",),
        ("usp:scail",),
        "self x t context clip_fea freqs transformer_options pose_latents reference_latent "
        "ref_mask_latents sam_latents",
        4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.wan.model", "SCAIL2WanModel.forward_orig", ("wan_scail",),
        ("usp:scail2-inherited",),
        "self x t context clip_fea freqs transformer_options pose_latents reference_latent "
        "ref_mask_latents sam_latents",
        4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.wan.model_wandancer", "WanDancerModel.forward_orig", ("wan_dancer",),
        ("usp",),
        "self x t context clip_fea clip_fea_ref freqs audio_embed fps audio_inject_scale "
        "transformer_options",
        4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.wan.model_animate2", "WanAnimate2Model.forward_orig", ("wan_animate2",),
        ("spatial-frame-gather",),
        "self x t context clip_fea freqs freqs_pose pose_latents clip_fea_pose context_pose pose_strength reference_strength transformer_options",
        4, variadic_keyword="kwargs",
    ),
    _rebound(
        "comfy.ldm.wan.model_animate2", "WanAnimate2SelfAttention.forward_pose", ("wan_animate2",),
        ("spatial-frame-gather:pose",), "self x freqs transformer_options", 3,
    ),
    _rebound(
        "comfy.ldm.wan.model_animate2", "WanAnimate2SelfAttention.forward_gen", ("wan_animate2",),
        ("spatial-frame-gather:generation",),
        "self x freqs k_pose v_pose f_gen hw buffers ref_strength transformer_options", 8,
    ),
    # Comfy 3b4c0b0e (ming-image support) inserted direct_context and ref_frames
    # between siglip_feats and transformer_options. Trees on either side of that
    # commit are both admitted; the pixel-space sibling right below did not gain
    # these two params and keeps its plain contract.
    _rebound(
        "comfy.ldm.lumina.model", "NextDiT._forward", ("zimage",),
        ("usp:latent", "cfg-pad:latent"),
        "self x timesteps context num_tokens attention_mask ref_latents ref_contexts "
        "siglip_feats transformer_options",
        5, variadic_keyword="kwargs",
        optional_infix=("direct_context", "ref_frames"),
    ),
    _rebound(
        "comfy.ldm.lumina.model", "NextDiTPixelSpace._forward", ("zimage",),
        ("usp:pixel", "cfg-pad:pixel"),
        "self x timesteps context num_tokens attention_mask ref_latents ref_contexts "
        "siglip_feats transformer_options",
        5, variadic_keyword="kwargs",
    ),
)


# These three adapters inherit Adapter.matches, whose isinstance gate accepts
# every subclass of the named root. The ComfyUI-master canary derives the real
# subclass set and fails when it differs from the branch rows here, so a new
# subclass needs its own row naming its rebound targets.
BROAD_ADAPTER_SUBCLASS_CONTRACTS: tuple[BroadAdapterSubclassContract, ...] = (
    BroadAdapterSubclassContract(
        "chroma",
        "Chroma",
        (
            BroadAdapterBranch(
                "Chroma", ("comfy.ldm.chroma.model.Chroma.forward_orig",)
            ),
            BroadAdapterBranch(
                "ChromaRadiance",
                ("comfy.ldm.chroma_radiance.model.ChromaRadiance.forward_orig",),
            ),
        ),
    ),
    BroadAdapterSubclassContract(
        "krea2",
        "Krea2",
        (
            BroadAdapterBranch(
                "Krea2", ("comfy.ldm.krea2.model.SingleStreamDiT._forward",)
            ),
        ),
    ),
    BroadAdapterSubclassContract(
        "ideogram4",
        "Ideogram4",
        (
            BroadAdapterBranch(
                "Ideogram4",
                ("comfy.ldm.ideogram4.model.Ideogram4Transformer2DModel._forward",),
            ),
        ),
    ),
)
