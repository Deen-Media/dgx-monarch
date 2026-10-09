"""Static adapter bind sites and their concrete Comfy method targets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterBindSite:
    """One static ``bind(receiver, method, replacement)`` adapter call."""

    source: str
    scope: str
    binder: str
    receiver: str
    method: str
    replacement: str
    families: tuple[str, ...]
    branch: str
    targets: tuple[str, ...]

    @property
    def ast_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.source,
            self.scope,
            self.binder,
            self.receiver,
            self.method,
            self.replacement,
        )


def _site(
    source: str,
    scope: str,
    receiver: str,
    method: str,
    replacement: str,
    families: tuple[str, ...],
    branch: str,
    targets: tuple[str, ...],
    *,
    binder: str = "self.bind",
) -> AdapterBindSite:
    return AdapterBindSite(
        source, scope, binder, receiver, method, replacement, families, branch, targets
    )


# One row per source call expression. Multiple rows may point at one target;
# conditional and inherited sites may point at multiple concrete targets.
ADAPTER_BIND_SITES: tuple[AdapterBindSite, ...] = (
    _site("anima.py", "AnimaAdapter.inject_usp", "diffusion_model", "_forward", "usp_forward",
          ("anima",), "usp", ("comfy.ldm.anima.model.Anima._forward",)),
    _site("boogu.py", "BooguAdapter.inject_usp", "diffusion_model", "forward", "usp_forward",
          ("boogu",), "usp", ("comfy.ldm.boogu.model.BooguTransformer2DModel.forward",)),
    _site("boogu.py", "BooguAdapter.inject_cfg_pad_forward", "diffusion_model", "forward",
          "cfg_pad_forward", ("boogu",), "cfg-pad",
          ("comfy.ldm.boogu.model.BooguTransformer2DModel.forward",)),
    _site("cogvideo.py", "CogVideoXAdapter.inject_usp", "diffusion_model", "_forward",
          "usp_forward", ("cogvideo",), "usp",
          ("comfy.ldm.cogvideo.model.CogVideoXTransformer3DModel._forward",)),
    _site("ernie.py", "ErnieAdapter.inject_usp", "diffusion_model", "forward", "usp_forward",
          ("ernie",), "usp:model", ("comfy.ldm.ernie.model.ErnieImageModel.forward",)),
    _site("ernie.py", "ErnieAdapter.inject_usp", "layer.self_attention", "forward",
          "usp_attention_forward", ("ernie",), "usp:self-attention",
          ("comfy.ldm.ernie.model.ErnieImageAttention.forward",)),
    _site("flux_family.py", "ChromaAdapter.inject_usp", "diffusion_model", "forward_orig",
          "usp_forward_orig", ("chroma",), "usp",
          ("comfy.ldm.chroma.model.Chroma.forward_orig",
           "comfy.ldm.chroma_radiance.model.ChromaRadiance.forward_orig")),
    _site("flux_family.py", "ChromaAdapter.inject_cfg_pad_forward", "diffusion_model",
          "_forward", "cfg_pad_forward", ("chroma",), "cfg-pad",
          ("comfy.ldm.chroma.model.Chroma._forward",
           "comfy.ldm.chroma_radiance.model.ChromaRadiance._forward")),
    _site("flux_family.py", "FluxAdapter.inject_usp", "diffusion_model", "forward_orig",
          "usp_forward_orig", ("flux", "flux2", "longcat"), "usp",
          ("comfy.ldm.flux.model.Flux.forward_orig",)),
    _site("hunyuan.py", "HunyuanAdapter.inject_usp", "diffusion_model", "forward_orig",
          "usp_forward_orig", ("hunyuan",), "usp",
          ("comfy.ldm.hunyuan_video.model.HunyuanVideo.forward_orig",)),
    _site("kandinsky5.py", "Kandinsky5Adapter.inject_usp", "diffusion_model", "forward_orig",
          "usp_forward_orig", ("kandinsky5",), "usp:model",
          ("comfy.ldm.kandinsky5.model.Kandinsky5.forward_orig",)),
    _site("kandinsky5.py", "Kandinsky5Adapter.inject_cfg_pad_forward", "diffusion_model",
          "_forward", "cfg_pad_forward", ("kandinsky5",), "cfg-pad",
          ("comfy.ldm.kandinsky5.model.Kandinsky5._forward",)),
    _site("kandinsky5.py", "Kandinsky5Adapter.inject_usp", "block.self_attention", "forward",
          "usp_self_attention", ("kandinsky5",), "usp:self-attention",
          ("comfy.ldm.kandinsky5.model.SelfAttention.forward",)),
    _site("krea2.py", "Krea2Adapter.inject_usp", "diffusion_model", "_forward", "usp_forward",
          ("krea2",), "usp", ("comfy.ldm.krea2.model.SingleStreamDiT._forward",)),
    _site("krea2.py", "Krea2Adapter.inject_cfg_pad_forward", "diffusion_model", "_forward",
          "cfg_pad_forward", ("krea2",), "cfg-pad",
          ("comfy.ldm.krea2.model.SingleStreamDiT._forward",)),
    _site("lens.py", "LensAdapter.inject_usp", "diffusion_model", "_forward", "usp_forward",
          ("lens",), "usp", ("comfy.ldm.lens.model.LensTransformer2DModel._forward",)),
    _site("lens.py", "LensAdapter.inject_cfg_pad_forward", "diffusion_model", "_forward",
          "cfg_pad_forward", ("lens",), "cfg-pad",
          ("comfy.ldm.lens.model.LensTransformer2DModel._forward",)),
    _site("ltx.py", "LTXAdapter._inject_ltxv", "diffusion_model",
          "_process_transformer_blocks", "usp_process_blocks", ("ltx",), "ltxv:model",
          ("comfy.ldm.lightricks.model.LTXVModel._process_transformer_blocks",)),
    _site("ltx.py", "LTXAdapter._inject_ltxav", "diffusion_model",
          "_process_transformer_blocks", "usp_process_blocks", ("ltx",), "ltxav:model",
          ("comfy.ldm.lightricks.av_model.LTXAVModel._process_transformer_blocks",)),
    _site("ltx.py", "LTXAdapter._inject_ltxv", "block.attn1", "forward", "usp_self",
          ("ltx",), "ltxv:self-attention",
          ("comfy.ldm.lightricks.model.CrossAttention.forward",)),
    _site("ltx.py", "LTXAdapter._inject_ltxav", "block.attn1", "forward", "video_self",
          ("ltx",), "ltxav:video-self",
          ("comfy.ldm.lightricks.model.CrossAttention.forward",)),
    _site("ltx.py", "LTXAdapter._inject_ltxav", "block.audio_attn1", "forward",
          "audio_self", ("ltx",), "ltxav:audio-self",
          ("comfy.ldm.lightricks.model.CrossAttention.forward",)),
    _site("ltx.py", "LTXAdapter._inject_ltxav", "block.audio_to_video_attn", "forward",
          "audio_to_video", ("ltx",), "ltxav:audio-to-video",
          ("comfy.ldm.lightricks.model.CrossAttention.forward",)),
    _site("ltx.py", "LTXAdapter._inject_ltxav", "block.video_to_audio_attn", "forward",
          "video_to_audio", ("ltx",), "ltxav:video-to-audio",
          ("comfy.ldm.lightricks.model.CrossAttention.forward",)),
    _site("minimax_h3.py", "MiniMaxH3Adapter.inject_usp", "diffusion_model", "_forward",
          "usp_forward", ("minimax_h3",), "usp",
          ("comfy.ldm.minimax.model.MiniMaxH3Model._forward",)),
    _site("omnigen2.py", "Omnigen2Adapter.inject_usp", "diffusion_model", "forward",
          "usp_forward", ("omnigen2",), "usp",
          ("comfy.ldm.omnigen.omnigen2.OmniGen2Transformer2DModel.forward",)),
    _site("pixeldit.py", "Ideogram4Adapter.inject_usp", "diffusion_model", "_forward",
          "usp_forward", ("ideogram4",), "usp",
          ("comfy.ldm.ideogram4.model.Ideogram4Transformer2DModel._forward",)),
    _site("pixeldit_comfy.py", "PixelDiTAdapter.inject_usp", "diffusion_model",
          "_forward", "usp_forward", ("pixeldit_comfy",), "usp:pixeldit-or-pid",
          ("comfy.ldm.pixeldit.model.PixDiT_T2I._forward",
           "comfy.ldm.pixeldit.pid.PidNet._forward")),
    _site("pixeldit_comfy.py", "PixelDiTAdapter.inject_usp", "blk", "forward",
          "_usp_pixel_block_forward", ("pixeldit_comfy",), "usp:pixel-block",
          ("comfy.ldm.pixeldit.modules.PiTBlock.forward",)),
    _site("mage_flow.py", "MageFlowAdapter.inject_usp", "diffusion_model", "_forward",
          "usp_forward", ("mage_flow",), "usp",
          ("comfy.ldm.mage_flow.model.MageFlowTransformer2DModel._forward",)),
    _site("mage_flow.py", "MageFlowAdapter.inject_cfg_pad_forward", "diffusion_model", "_forward",
          "cfg_pad_forward", ("mage_flow",), "cfg-pad",
          ("comfy.ldm.mage_flow.model.MageFlowTransformer2DModel._forward",)),
    _site("qwen_image.py", "QwenImageAdapter.inject_usp", "diffusion_model", "_forward",
          "usp_forward", ("qwen_image",), "usp",
          ("comfy.ldm.qwen_image.model.QwenImageTransformer2DModel._forward",)),
    _site("qwen_image.py", "QwenImageAdapter.inject_cfg_pad_forward", "diffusion_model",
          "_forward", "cfg_pad_forward", ("qwen_image",), "cfg-pad",
          ("comfy.ldm.qwen_image.model.QwenImageTransformer2DModel._forward",)),
    _site("qwen_image21.py", "QwenImage21Adapter.inject_usp", "diffusion_model", "_forward",
          "usp_forward", ("qwen_image21",), "usp",
          ("comfy.ldm.qwen_image21.model.QwenImage21Transformer2DModel._forward",)),
    _site("wan_family.py", "WanAdapter.inject_usp", "diffusion_model", "forward_orig",
          "usp_forward_orig", ("wan",), "usp:model",
          ("comfy.ldm.wan.model.WanModel.forward_orig",)),
    _site("wan_family.py", "WanAdapter.inject_usp", "block.self_attn", "forward",
          "usp_self_attention", ("wan",), "usp:self-attention",
          ("comfy.ldm.wan.model.WanSelfAttention.forward",)),
    _site("wan_variants.py", "_bind_usp_self_attention", "block.self_attn", "forward",
          "usp_self_attention", ("wan_scail", "wan_dancer"), "usp:self-attention",
          ("comfy.ldm.wan.model.WanSelfAttention.forward",), binder="Adapter.bind"),
    _site("wan_variants.py", "SCAILAdapter.inject_usp", "diffusion_model", "forward_orig",
          "usp_forward_orig", ("wan_scail",), "usp",
          ("comfy.ldm.wan.model.SCAILWanModel.forward_orig",
           "comfy.ldm.wan.model.SCAIL2WanModel.forward_orig")),
    _site("wan_variants.py", "WanDancerAdapter.inject_usp", "diffusion_model", "forward_orig",
          "usp_forward_orig", ("wan_dancer",), "usp",
          ("comfy.ldm.wan.model_wandancer.WanDancerModel.forward_orig",)),
    _site("wan_animate2.py", "WanAnimate2Adapter.inject_usp", "diffusion_model", "forward_orig",
          "forward_orig", ("wan_animate2",), "spatial-frame-gather",
          ("comfy.ldm.wan.model_animate2.WanAnimate2Model.forward_orig",), binder="Adapter.bind"),
    _site("wan_animate2.py", "WanAnimate2Adapter.inject_usp", "block.self_attn", "forward_pose",
          "pose_attn", ("wan_animate2",), "spatial-frame-gather:pose",
          ("comfy.ldm.wan.model_animate2.WanAnimate2SelfAttention.forward_pose",), binder="Adapter.bind"),
    _site("wan_animate2.py", "WanAnimate2Adapter.inject_usp", "block.self_attn", "forward_gen",
          "gen_attn", ("wan_animate2",), "spatial-frame-gather:generation",
          ("comfy.ldm.wan.model_animate2.WanAnimate2SelfAttention.forward_gen",), binder="Adapter.bind"),
    _site("zimage.py", "ZImageAdapter.inject_usp", "diffusion_model", "_forward",
          "pixel_forward if pixel_space else latent_forward", ("zimage",),
          "usp:latent-or-pixel", ("comfy.ldm.lumina.model.NextDiT._forward",
                                  "comfy.ldm.lumina.model.NextDiTPixelSpace._forward")),
    _site("zimage.py", "ZImageAdapter.inject_cfg_pad_forward", "diffusion_model", "_forward",
          "cfg_pad_forward", ("zimage",), "cfg-pad:latent-or-pixel",
          ("comfy.ldm.lumina.model.NextDiT._forward",
           "comfy.ldm.lumina.model.NextDiTPixelSpace._forward")),
)
